from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        init_method: str = "kmeans++",
        init_buffer_size: int = 1000,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta
        self.init_method = init_method
        self.init_buffer_size = init_buffer_size
        self.codebook = nn.Parameter(
            torch.zeros(codebook_size, codebook_dim), requires_grad=True
        )
        self.is_initialized: bool = False
        self._init_buffer: List[torch.Tensor] = []
        self._is_initial_step: bool = False


    @staticmethod
    def _squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = (x * x).sum(dim=-1, keepdim=True)          # (N, 1)
        yy = (y * y).sum(dim=-1, keepdim=True).T         # (1, M)
        dist = xx + yy - 2.0 * x @ y.T
        return dist.clamp(min=0.0)                       # numerical safety

    def _kmeans_pp_init(self, data: torch.Tensor) -> torch.Tensor:
        N, D = data.shape
        device, dtype = data.device, data.dtype
        centroids = torch.zeros(self.codebook_size, D, device=device, dtype=dtype)
        centroids[0] = data[torch.randint(0, N, (1,), device=device)]

        for i in range(1, self.codebook_size):
            dists = self._squared_euclidean(data, centroids[:i])  # (N, i)
            min_dists = dists.min(dim=1).values                    # (N,)
            total = min_dists.sum()
            if total == 0:
                remaining = self.codebook_size - i
                centroids[i:] = data[
                    torch.randint(0, N, (remaining,), device=device)
                ]
                break
            probs = min_dists / total
            idx = torch.multinomial(probs, num_samples=1)
            centroids[i] = data[idx]

        return centroids

    def _random_init(self, data: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(data.shape[0], device=data.device)[: self.codebook_size]
        return data[idx].clone()

    def initialize(self, data: torch.Tensor) -> None:
        if data.shape[0] < self.codebook_size:
            raise ValueError(
                f"Need at least {self.codebook_size} points to initialize, "
                f"got {data.shape[0]}."
            )
        with torch.no_grad():
            if self.init_method == "kmeans++":
                c = self._kmeans_pp_init(data)
            elif self.init_method == "random":
                c = self._random_init(data)
            else:
                raise ValueError(f"Unknown init_method '{self.init_method}'")
            self.codebook.data.copy_(c)
            print("\033[34mCodebook initialized\033[0m")
        self.is_initialized = True

    def _try_init_from_buffer(self, x: torch.Tensor) -> bool:
        if self.is_initialized:
            return True
        self._init_buffer.append(x.detach())
        buffered = torch.cat(self._init_buffer, dim=0)
        if buffered.shape[0] >= self.init_buffer_size:
            self.initialize(buffered[: self.init_buffer_size])
            self._init_buffer.clear()
            self._is_initial_step = True
            return True
        return False

    def forward(
        self, x: torch.Tensor, should_init=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Auto-initialize
        if not self.is_initialized and should_init:
            if not self._try_init_from_buffer(x):
                # Not enough data yet — return harmlessly
                return (
                    torch.zeros(x.shape[0], dtype=torch.long, device=x.device),
                    torch.zeros_like(x),
                    torch.tensor(0.0, device=x.device, requires_grad=True),
                )
        if self._is_initial_step:
            self._is_initial_step = False

        dists = self._squared_euclidean(x, self.codebook)   # (B, K)
        codes = dists.argmin(dim=-1)                         # (B,)
        raw_entries = self.codebook[codes]                    # (B, D)

        quantized = x + (raw_entries - x).detach()

        loss = (
            F.mse_loss(x.detach(), raw_entries, reduction="mean")
            + self.beta * F.mse_loss(x, raw_entries.detach(), reduction="mean")
        )
        return codes, quantized, loss

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self._squared_euclidean(x, self.codebook)
        codes = dists.argmin(dim=-1)
        return codes, self.codebook[codes]


class ResidualVectorQuantization(nn.Module):
    def __init__(
        self,
        codebook_dim: int,
        n_layers: int,
        codebook_size: int = 2048,
        beta: float = 0.25,
        normalize_residuals: bool = True,
        init_method: str = "kmeans++",
        init_buffer_size: int = 5120,
    ) -> None:
        super().__init__()
        self.codebook_dim = codebook_dim
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.normalize_residuals = normalize_residuals

        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    beta=beta,
                    init_method=init_method,
                    init_buffer_size=init_buffer_size,
                )
                for _ in range(n_layers)
            ]
        )

        self.layers_2 = nn.ModuleList(
            [
                VectorQuantize(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    beta=beta,
                    init_method=init_method,
                    init_buffer_size=init_buffer_size,
                )
                for _ in range(n_layers)
            ]
        )


    def forward(
        self,
        x: torch.Tensor,
        current_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_codes_t: List[torch.Tensor] = []
        all_codes_v: List[torch.Tensor] = []
        residual_t = x[0]
        residual_v = x[1]
        quantized_sum = torch.zeros_like(x[0])
        total_loss = torch.tensor(0.0, device=x[0].device, requires_grad=True)

        for idx, (layer_t, layer_v) in enumerate(zip(self.layers, self.layers_2)):
            if idx == 1 or idx == 2:
                residual_t_new = 0.8*residual_t+0.1*residual_v
                residual_v = 0.8*residual_v+0.1*residual_t
                residual_t = residual_t_new
            if self.normalize_residuals:
                residual_t = F.normalize(residual_t, dim=-1)
                residual_v = F.normalize(residual_v, dim=-1)
            # Decide whether this layer contributes to the loss
            should_train = (current_layer is None) or (idx == current_layer)

            if should_train and layer_t.is_initialized and layer_v.is_initialized:
                codes_t, quant_t, loss_t = layer_t(residual_t)
                codes_v, quant_v, loss_v = layer_v(residual_v)
                total_loss = total_loss + loss_t + loss_v 
            elif should_train and not layer_t.is_initialized:
                # Layer not initialized yet — forward handles auto-init
                codes_t, quant_t, loss_t = layer_t(residual_t, should_init=True)
                codes_v, quant_v, loss_v = layer_v(residual_v, should_init=True)
                total_loss = total_loss + loss_t + loss_v 
            else:
                if layer_t.is_initialized:
                    codes_t, quant_t = layer_t.predict(residual_t)
                    codes_v, quant_v = layer_v.predict(residual_v)
                else:
                    codes_t, quant_t, _ = layer_t(residual_t)
                    codes_v, quant_v, _ = layer_v(residual_v)

            all_codes_t.append(codes_t)
            all_codes_v.append(codes_v)
            quantized_sum = quantized_sum + quant_t + quant_v
            residual_t = residual_t - quant_t
            residual_v = residual_v - quant_v

        codes_t = torch.stack(all_codes_t, dim=-1)  # (B, n_layers)
        codes_v = torch.stack(all_codes_v, dim=-1)  # (B, n_layers)
        return codes_t, codes_v, quantized_sum, total_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        all_codes_t: List[torch.Tensor] = []
        all_codes_v: List[torch.Tensor] = []
        residual_t = x[0]
        residual_v = x[1]

        for idx, (layer_t, layer_v) in enumerate(zip(self.layers, self.layers_2)):
            if idx == 1 or idx == 2:
                residual_t_new = 0.8*residual_t+0.1*residual_v
                residual_v = 0.8*residual_v+0.1*residual_t
                residual_t = residual_t_new
            if self.normalize_residuals:
                residual_t = F.normalize(residual_t, dim=-1)
                residual_v = F.normalize(residual_v, dim=-1)
            
            codes_t, quant_t = layer_t.predict(residual_t)
            codes_v, quant_v = layer_v.predict(residual_v)
                
            all_codes_t.append(codes_t)
            all_codes_v.append(codes_v)
            
            residual_t = residual_t - quant_t
            residual_v = residual_v - quant_v
            
        return torch.stack(all_codes_t, dim=-1), torch.stack(all_codes_v, dim=-1)