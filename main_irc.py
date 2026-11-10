from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class InteractionGraph:
    sequences: list[list[int]]
    adjacent_edges: Tensor
    cooccurring_items: list[set[int]]


class TrainableEmbeddings(nn.Module):
    """The embedding tables are the only learnable parameters."""

    def __init__(self, text: Tensor, image: Tensor) -> None:
        super().__init__()
        self.text = nn.Parameter(text.clone())
        self.image = nn.Parameter(image.clone())

    def normalized_views(
        self, item_ids: Tensor, noise_std: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        text = F.normalize(self.text[item_ids], dim=-1)
        image = F.normalize(self.image[item_ids], dim=-1)
        noisy_text = F.normalize(text + noise_std * torch.randn_like(text), dim=-1)
        noisy_image = F.normalize(image + noise_std * torch.randn_like(image), dim=-1)
        return text, image, noisy_text, noisy_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject sequence collaborative signals into two embedding tables."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="Arts_Crafts_and_Sewing")
    parser.add_argument(
        "--text-embedding", default="qwen4B_embeddings.npy"
    )
    parser.add_argument(
        "--image-embedding", default="qwen2Bvl_embeddings.npy"
    )
    parser.add_argument(
        "--output-text", default="qwen4B_calib_embeddings.npy"
    )
    parser.add_argument(
        "--output-image", default="qwen2Bvl_calib_embeddings.npy"
    )
    parser.add_argument("--dim", type=int, default=2560)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight of the original embedding in alpha*E + (1-alpha)*E'.",
    )
    parser.add_argument("--neighbor-weight", type=float, default=1.0)
    parser.add_argument("--cross-modal-weight", type=float, default=0.5)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:1'.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 matrix multiplication on CUDA.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_interaction_graph(dataset_dir: Path, num_items: int) -> InteractionGraph:
    train = load_json(dataset_dir / "train.json")
    val = load_json(dataset_dir / "val.json")
    user_ids = list(dict.fromkeys([*train.keys(), *val.keys()]))

    sequences: list[list[int]] = []
    adjacent_edges: list[tuple[int, int]] = []
    cooccurring_items = [set() for _ in range(num_items)]

    for user_id in user_ids:
        sequence = [int(item) for item in train.get(user_id, [])]
        sequence.extend(int(item) for item in val.get(user_id, []))
        if any(item < 0 or item >= num_items for item in sequence):
            raise ValueError(f"User {user_id} has an item ID outside [0, {num_items}).")
        sequences.append(sequence)

        unique_items = set(sequence)
        for item in unique_items:
            cooccurring_items[item].update(unique_items)

        adjacent_edges.extend(
            (left, right)
            for left, right in zip(sequence, sequence[1:])
            if left != right
        )

    if not adjacent_edges:
        raise ValueError("No non-self adjacent item pairs were found in train+val.")

    return InteractionGraph(
        sequences=sequences,
        adjacent_edges=torch.tensor(adjacent_edges, dtype=torch.long),
        cooccurring_items=cooccurring_items,
    )


def load_embeddings(
    text_path: Path, image_path: Path, dim: int
) -> tuple[Tensor, Tensor]:
    text_array = np.load(text_path, mmap_mode="r")
    image_array = np.load(image_path, mmap_mode="r")
    if text_array.ndim != 2 or image_array.ndim != 2:
        raise ValueError("Both embedding files must be rank-2 arrays.")
    if text_array.shape[0] != image_array.shape[0]:
        raise ValueError(
            f"Item counts differ: text={text_array.shape[0]}, "
            f"image={image_array.shape[0]}."
        )
    if text_array.shape[1] < dim or image_array.shape[1] < dim:
        raise ValueError(
            f"Requested dim={dim}, but input shapes are "
            f"{text_array.shape} and {image_array.shape}."
        )

    # copy=True makes the memory-mapped, possibly read-only arrays safe tensors.
    text = torch.from_numpy(np.array(text_array[:, :dim], dtype=np.float32, copy=True))
    image = torch.from_numpy(
        np.array(image_array[:, :dim], dtype=np.float32, copy=True)
    )
    for modality, embedding in (("text", text), ("image", image)):
        if not torch.isfinite(embedding).all():
            raise ValueError(f"{modality} embeddings contain NaN or infinity.")
        if (embedding.norm(dim=1) == 0).any():
            raise ValueError(f"{modality} embeddings contain an all-zero row.")
    return F.normalize(text, dim=-1), F.normalize(image, dim=-1)


def make_training_records(adjacent_edges: Tensor, num_items: int) -> Tensor:
    """Add one non-neighbor record per item so every item gets view positives."""
    item_records = torch.stack(
        [
            torch.arange(num_items, dtype=torch.long),
            torch.full((num_items,), -1, dtype=torch.long),
        ],
        dim=1,
    )
    return torch.cat([adjacent_edges, item_records], dim=0)


def collate_records(records: Iterable[Tensor]) -> Tensor:
    return torch.stack(list(records))


def batch_items_and_neighbors(records: Tensor) -> tuple[Tensor, Tensor]:
    """Return unique global item IDs and local-index adjacent edges."""
    valid_neighbor = records[:, 1] >= 0
    second_items = records[valid_neighbor, 1]
    all_items = torch.cat([records[:, 0], second_items])
    item_ids, inverse = torch.unique(all_items, sorted=True, return_inverse=True)

    first_local = inverse[: records.shape[0]][valid_neighbor]
    second_local = inverse[records.shape[0] :]
    local_edges = torch.stack([first_local, second_local], dim=1)
    return item_ids, local_edges


def negative_item_mask(
    item_ids: Tensor, cooccurring_items: list[set[int]]
) -> Tensor:
    ids = item_ids.tolist()
    rows = []
    for anchor in ids:
        cooccurred = cooccurring_items[anchor]
        rows.append([candidate != anchor and candidate not in cooccurred for candidate in ids])
    return torch.tensor(rows, dtype=torch.bool)


def build_positive_masks(
    num_items: int, local_edges: Tensor, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    num_views = 4
    total_views = num_items * num_views
    local_item = torch.arange(num_items, device=device).repeat_interleave(num_views)
    view_type = torch.arange(num_views, device=device).repeat(num_items)
    same_item = local_item[:, None] == local_item[None, :]

    noise_pairs = (
        ((view_type[:, None] == 0) & (view_type[None, :] == 2))
        | ((view_type[:, None] == 2) & (view_type[None, :] == 0))
        | ((view_type[:, None] == 1) & (view_type[None, :] == 3))
        | ((view_type[:, None] == 3) & (view_type[None, :] == 1))
    )
    noise_positive = same_item & noise_pairs

    cross_pairs = (
        ((view_type[:, None] == 0) & (view_type[None, :] == 1))
        | ((view_type[:, None] == 1) & (view_type[None, :] == 0))
    )
    cross_positive = same_item & cross_pairs

    neighbor_items = torch.zeros((num_items, num_items), dtype=torch.bool, device=device)
    if local_edges.numel():
        left = local_edges[:, 0].to(device)
        right = local_edges[:, 1].to(device)
        neighbor_items[left, right] = True
        neighbor_items[right, left] = True
    clean_views = (view_type[:, None] < 2) & (view_type[None, :] < 2)
    neighbor_positive = (
        neighbor_items[local_item[:, None], local_item[None, :]] & clean_views
    )

    assert noise_positive.shape == (total_views, total_views)
    return noise_positive, cross_positive, neighbor_positive


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return values.sum() * 0.0
    return values.masked_select(mask).mean()


def calib_loss(
    views: tuple[Tensor, Tensor, Tensor, Tensor],
    local_edges: Tensor,
    negative_items: Tensor,
    temperature: float,
    noise_weight: float,
    cross_modal_weight: float,
    neighbor_weight: float,
    use_amp: bool,
) -> tuple[Tensor, dict[str, float]]:
    device = views[0].device
    num_items = views[0].shape[0]
    representations = torch.stack(views, dim=1).flatten(0, 1)
    noise_pos, cross_pos, neighbor_pos = build_positive_masks(
        num_items, local_edges, device
    )
    all_positive = noise_pos | cross_pos | neighbor_pos

    local_item = torch.arange(num_items, device=device).repeat_interleave(4)
    negative_items = negative_items.to(device, non_blocking=True)
    negative_views = negative_items[
        local_item[:, None], local_item[None, :]
    ]
    denominator_mask = all_positive | negative_views

    amp_enabled = (
        use_amp
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        similarities = representations @ representations.T
    logits = similarities.float() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator_mask, -torch.inf), dim=1, keepdim=True
    )
    negative_log_probability = -(logits - log_denominator)

    noise_loss = masked_mean(negative_log_probability, noise_pos)
    cross_loss = masked_mean(negative_log_probability, cross_pos)
    neighbor_loss = masked_mean(negative_log_probability, neighbor_pos)
    loss = (
        noise_weight * noise_loss
        + cross_modal_weight * cross_loss
        + neighbor_weight * neighbor_loss
    )
    metrics = {
        "loss": float(loss.detach()),
        "noise": float(noise_loss.detach()),
        "cross": float(cross_loss.detach()),
        "neighbor": float(neighbor_loss.detach()),
    }
    return loss, metrics


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def validate_args(args: argparse.Namespace) -> None:
    if args.dim <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("dim, epochs, and batch-size must be positive.")
    if args.num_workers < 0 or args.log_every <= 0 or args.grad_clip <= 0:
        raise ValueError("num-workers must be non-negative; log-every/grad-clip positive.")
    if args.temperature <= 0 or args.noise_std < 0 or args.lr <= 0:
        raise ValueError("temperature/lr must be positive and noise-std non-negative.")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    weights = [args.noise_weight, args.cross_modal_weight, args.neighbor_weight]
    if any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("Loss weights must be non-negative and not all zero.")


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_dir = args.data_dir / args.dataset
    text_path = dataset_dir / args.text_embedding
    image_path = dataset_dir / args.image_embedding

    original_text, original_image = load_embeddings(text_path, image_path, args.dim)
    num_items = original_text.shape[0]
    graph = load_interaction_graph(dataset_dir, num_items)
    records = make_training_records(graph.adjacent_edges, num_items)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        records,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_records,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = TrainableEmbeddings(original_text, original_image).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(
        f"device={device} items={num_items} users={len(graph.sequences)} "
        f"adjacent_occurrences={len(graph.adjacent_edges)} records={len(records)}"
    )
    for epoch in range(args.epochs):
        totals = {"loss": 0.0, "noise": 0.0, "cross": 0.0, "neighbor": 0.0}
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch_records in enumerate(progress, start=1):
            item_ids, local_edges = batch_items_and_neighbors(batch_records)
            negatives = negative_item_mask(item_ids, graph.cooccurring_items)
            item_ids = item_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            views = model.normalized_views(item_ids, args.noise_std)
            loss, metrics = calib_loss(
                views=views,
                local_edges=local_edges,
                negative_items=negatives,
                temperature=args.temperature,
                noise_weight=args.noise_weight,
                cross_modal_weight=args.cross_modal_weight,
                neighbor_weight=args.neighbor_weight,
                use_amp=args.amp,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for name, value in metrics.items():
                totals[name] += value
            if step % args.log_every == 0 or step == len(loader):
                progress.set_postfix(
                    loss=f"{totals['loss'] / step:.4f}",
                    neighbor=f"{totals['neighbor'] / step:.4f}",
                )

    with torch.no_grad():
        learned_text = F.normalize(model.text, dim=-1).cpu()
        learned_image = F.normalize(model.image, dim=-1).cpu()
        fused_text = F.normalize(
            args.alpha * original_text + (1.0 - args.alpha) * learned_text, dim=-1
        )
        fused_image = F.normalize(
            args.alpha * original_image + (1.0 - args.alpha) * learned_image, dim=-1
        )

    output_text = dataset_dir / args.output_text
    output_image = dataset_dir / args.output_image
    np.save(output_text, fused_text.numpy())
    np.save(output_image, fused_image.numpy())
    metadata = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "device": str(device),
        "num_items": num_items,
        "num_users": len(graph.sequences),
        "num_adjacent_occurrences": len(graph.adjacent_edges),
        "input_text": str(text_path),
        "input_image": str(image_path),
        "output_text": str(output_text),
        "output_image": str(output_image),
    }
    metadata_path = dataset_dir / "calib_embedding_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"saved text embeddings:  {output_text} {tuple(fused_text.shape)}")
    print(f"saved image embeddings: {output_image} {tuple(fused_image.shape)}")
    print(f"saved run metadata:      {metadata_path}")


if __name__ == "__main__":
    train(parse_args())
