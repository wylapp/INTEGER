import json
from collections import defaultdict
import torch
from os import path
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
import lightning as L
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.generation.logits_process import LogitsProcessorList
import torch.optim as optim
from src.utils import get_steady_then_decay_scheduler, all_possible_ids, get_cosine_annealing_with_warmup_scheduler
from src.metrics import NDCG, Recall
from src.alter_trie import BatchedTrieLogitsProcessor
from src.multimodal_encoder import ViLBertT5Encoder
from omegaconf import OmegaConf


class _EncoderApiPlaceholder(nn.Module):
    """Parameter-free placeholder for generation API introspection."""

    main_input_name = "input_ids"


class INTEGER(nn.Module):
    def __init__(self, cfg, text_possible_ids, image_possible_ids, target_sid_length):
        super(INTEGER, self).__init__()
        self.config = cfg
        t5config = T5Config(
        num_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_heads=cfg.num_heads,
        d_kv=cfg.d_kv,
        dropout_rate=cfg.dropout_rate,
        vocab_size=cfg.vocab_size,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        decoder_start_token_id=cfg.pad_token_id,
        feed_forward_proj=cfg.feed_forward_proj,
        )
        self.text_decoder_model = T5ForConditionalGeneration(t5config)
        self.image_decoder_model = T5ForConditionalGeneration(t5config)
        shared_embedding = self.text_decoder_model.shared
        self.text_decoder_model.encoder.embed_tokens = shared_embedding
        self.text_decoder_model.decoder.embed_tokens = shared_embedding
        self.image_decoder_model.shared = shared_embedding
        self.image_decoder_model.decoder.embed_tokens = shared_embedding
        self.multimodal_encoder = ViLBertT5Encoder(
            t5config,
            shared_embedding,
            independent_layers=3,
            co_attention_layers=1,
        )
        self.multimodal_encoder.apply(self.text_decoder_model._init_weights)
        for decoder_model in (
            self.text_decoder_model,
            self.image_decoder_model,
        ):
            decoder_model.config.tie_word_embeddings = False
            decoder_model.lm_head.weight = nn.Parameter(
                decoder_model.lm_head.weight.detach().clone()
            )
        self.text_decoder_model._tied_weights_keys = {
            "decoder.embed_tokens.weight": "shared.weight",
        }
        self.image_decoder_model._tied_weights_keys = {
            "decoder.embed_tokens.weight": "shared.weight",
        }
        self.text_decoder_model.all_tied_weights_keys = (
            self.text_decoder_model._tied_weights_keys.copy()
        )
        self.image_decoder_model.all_tied_weights_keys = (
            self.image_decoder_model._tied_weights_keys.copy()
        )
        # Encoding is handled jointly by the multimodal encoder. These
        # placeholders retain the interface expected by generation utilities.
        self.text_decoder_model.encoder = _EncoderApiPlaceholder()
        self.image_decoder_model.encoder = _EncoderApiPlaceholder()
        self.target_sid_length = target_sid_length
        self.text_constraint = BatchedTrieLogitsProcessor(
            text_possible_ids,
            cfg.vocab_size,
        )
        self.image_constraint = BatchedTrieLogitsProcessor(
            image_possible_ids,
            cfg.vocab_size,
        )

    def encode(
        self,
        text_input_ids,
        text_attention_mask,
        image_input_ids,
        image_attention_mask,
    ):
        return self.multimodal_encoder(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            image_input_ids=image_input_ids,
            image_attention_mask=image_attention_mask,
        )

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        image_input_ids,
        image_attention_mask,
        text_labels,
        image_labels,
    ):
        text_encoder_outputs, image_encoder_outputs = self.encode(
            text_input_ids,
            text_attention_mask,
            image_input_ids,
            image_attention_mask,
        )
        text_outputs = self.text_decoder_model(
            encoder_outputs=text_encoder_outputs,
            attention_mask=text_attention_mask,
            labels=text_labels,
        )
        image_outputs = self.image_decoder_model(
            encoder_outputs=image_encoder_outputs,
            attention_mask=image_attention_mask,
            labels=image_labels,
        )
        loss = (text_outputs.loss + image_outputs.loss) / 2
        return loss, (text_outputs.logits, image_outputs.logits)

    def _generate_modality(
        self, decoder_model, constraint, encoder_outputs, attention_mask
    ):
        # Generation expands encoder_outputs in place for beam search. Give
        # each decoder its own container while sharing the encoded tensors.
        generation_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        return decoder_model.generate(
            encoder_outputs=generation_encoder_outputs,
            attention_mask=attention_mask,
            max_length=self.target_sid_length + 1,
            # logits_processor=LogitsProcessorList([constraint]),
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_beams,
            use_cache=False,
        )

    def generate(
        self,
        text_input_ids,
        text_attention_mask,
        image_input_ids,
        image_attention_mask,
    ):
        text_encoder_outputs, image_encoder_outputs = self.encode(
            text_input_ids,
            text_attention_mask,
            image_input_ids,
            image_attention_mask,
        )
        text_outputs = self._generate_modality(
            self.text_decoder_model,
            self.text_constraint,
            text_encoder_outputs,
            text_attention_mask,
        )
        image_outputs = self._generate_modality(
            self.image_decoder_model,
            self.image_constraint,
            image_encoder_outputs,
            image_attention_mask,
        )
        return text_outputs, image_outputs


def build_sid_item_map(possible_ids):
    sid_item_map = defaultdict(list)
    for item_id, token_ids in enumerate(possible_ids):
        sid_item_map[tuple(token_ids[1:-1])].append(item_id)
    return dict(sid_item_map)


def reciprocal_rank_fusion(
    text_outputs,
    image_outputs,
    text_sid_item_map,
    image_sid_item_map,
    pad_token_id,
    eos_token_id,
    num_beams,
    batch_size,
    rrf_k=60,
    top_k=20,
    image_rrf_weight=1.0,
    return_modality_rankings=False,
):
    def sequence_key(sequence):
        return tuple(
            token
            for token in sequence
            if token not in {pad_token_id, eos_token_id}
        )

    text_outputs = text_outputs.detach().cpu().tolist()
    image_outputs = image_outputs.detach().cpu().tolist()
    rankings = []
    text_rankings = []
    image_rankings = []
    for batch_index in range(batch_size):
        fused_scores = defaultdict(float)
        batch_modality_rankings = []
        for outputs, sid_map, modality_weight in (
            (text_outputs, text_sid_item_map, 1.0),
            (image_outputs, image_sid_item_map, image_rrf_weight),
        ):
            modality_ranking = []
            seen_items = set()
            start = batch_index * num_beams
            for sequence in outputs[start:start + num_beams]:
                sid = sequence_key(sequence)
                for item_id in sid_map.get(sid, ()):
                    if item_id in seen_items:
                        continue
                    seen_items.add(item_id)
                    modality_ranking.append(item_id)
                    rank = len(modality_ranking)
                    score = modality_weight / (rrf_k + rank)
                    fused_scores[item_id] += score
            batch_modality_rankings.append(modality_ranking[:top_k])
        ranking = sorted(fused_scores, key=lambda item_id: (-fused_scores[item_id], item_id))
        rankings.append(ranking[:top_k])
        text_rankings.append(batch_modality_rankings[0])
        image_rankings.append(batch_modality_rankings[1])

    if return_modality_rankings:
        return rankings, text_rankings, image_rankings
    return rankings


def ranking_positions(rankings, target_items, top_k=20, device=None):
    if device is None:
        device = target_items.device if torch.is_tensor(target_items) else None
    if torch.is_tensor(target_items):
        target_items = target_items.detach().cpu().tolist()
    positions = torch.zeros(
        (len(rankings), top_k),
        dtype=torch.bool,
        device=device,
    )
    for row, (ranking, target_item) in enumerate(zip(rankings, target_items)):
        for column, item_id in enumerate(ranking[:top_k]):
            positions[row, column] = item_id == target_item
    return positions
    

class INTEGERTrainer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.vocabulary = json.load(open(path.join(self.config.data_dir, self.config.dataset, f"vocabulary_{cfg.sid_group}.json")))
        self.semIDs = torch.load(
            path.join(
                self.config.data_dir,
                self.config.dataset,
                f"semantic_{cfg.sid_group}.pt",
            ),
            map_location="cpu",
        )
        if self.semIDs.ndim != 2 or self.semIDs.shape[1] % 2:
            raise ValueError(
                "Multimodal semantic IDs must have shape [num_items, 2 * num_codebooks]"
            )
        self.modality_sid_length = self.semIDs.shape[1] // 2
        text_possible_ids = all_possible_ids(
            self.vocabulary,
            self.semIDs[:, :self.modality_sid_length],
            codebook_offset=0,
        )
        image_possible_ids = all_possible_ids(
            self.vocabulary,
            self.semIDs[:, self.modality_sid_length:],
            codebook_offset=self.modality_sid_length,
        )
        self.text_sid_item_map = build_sid_item_map(text_possible_ids)
        self.image_sid_item_map = build_sid_item_map(image_possible_ids)
        self.model = INTEGER(
            cfg,
            text_possible_ids,
            image_possible_ids,
            self.modality_sid_length,
        )
        self.rrf_k = int(getattr(cfg, "rrf_k", 5))
        self.image_rrf_weight = float(getattr(cfg, "image_rrf_weight", 1.0))
        self.evaluation_top_k = 20
        self.recall5 = Recall(5)
        self.recall10 = Recall(10)
        self.recall20 = Recall(20)
        self.ndcg5 = NDCG(5)
        self.ndcg10 = NDCG(10)
        self.ndcg20 = NDCG(20)


    def forward(self, inputs):
        return self.model(*inputs)

    def training_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def _evaluation_positions(self, batch):
        text_outputs, image_outputs = self.model.generate(*batch[:4])
        fused_rankings, text_rankings, image_rankings = reciprocal_rank_fusion(
            text_outputs,
            image_outputs,
            self.text_sid_item_map,
            self.image_sid_item_map,
            self.config.pad_token_id,
            self.config.eos_token_id,
            self.config.num_beams,
            len(batch[4]),
            self.rrf_k,
            self.evaluation_top_k,
            self.image_rrf_weight,
            return_modality_rankings=True,
        )
        target_items = batch[4]
        return (
            ranking_positions(
                fused_rankings,
                target_items,
                self.evaluation_top_k,
                batch[4].device,
            ),
            ranking_positions(
                text_rankings,
                target_items,
                self.evaluation_top_k,
                batch[4].device,
            ),
            ranking_positions(
                image_rankings,
                target_items,
                self.evaluation_top_k,
                batch[4].device,
            ),
        )

    def validation_step(self, batch, batch_idx):
        fused_positions, text_positions, image_positions = self._evaluation_positions(batch)
        self.recall5.update(fused_positions)
        self.recall10.update(fused_positions)
        self.recall20.update(fused_positions)
        self.ndcg5.update(fused_positions)
        self.ndcg10.update(fused_positions)
        self.ndcg20.update(fused_positions)

    
    def on_validation_epoch_end(self):
        self.log_dict({'eval/R@5': self.recall5.compute(), 'eval/R@10': self.recall10.compute(), 'eval/R@20': self.recall20.compute(),
                       'eval/NDCG@5': self.ndcg5.compute(), 'eval/NDCG@10': self.ndcg10.compute(), 'eval/NDCG@20': self.ndcg20.compute(),
                       }, on_epoch=True, sync_dist=True)
        
        self.recall5.reset()
        self.recall10.reset()
        self.recall20.reset()
        self.ndcg5.reset()
        self.ndcg10.reset()
        self.ndcg20.reset()

    def test_step(self, batch, batch_idx):
        fused_positions, text_positions, image_positions = self._evaluation_positions(batch)
        self.recall5.update(fused_positions)
        self.recall10.update(fused_positions)
        self.recall20.update(fused_positions)
        self.ndcg5.update(fused_positions)
        self.ndcg10.update(fused_positions)
        self.ndcg20.update(fused_positions)
    
    def on_test_epoch_end(self):
        self.log_dict({'test/R@5': self.recall5.compute(), 'test/R@10': self.recall10.compute(), 'test/R@20': self.recall20.compute(),
                       'test/NDCG@5': self.ndcg5.compute(), 'test/NDCG@10': self.ndcg10.compute(), 'test/NDCG@20': self.ndcg20.compute()}, on_epoch=True, sync_dist=True)
        
        self.recall5.reset()
        self.recall10.reset()
        self.recall20.reset()
        self.ndcg5.reset()
        self.ndcg10.reset()
        self.ndcg20.reset()


    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=self.config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_annealing_with_warmup_scheduler(
            optimizer,
            num_warmup_steps=int(self.config.warmup_proportion * total_steps),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
