"""Stage 2 peak set encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLP
from .components.set_attention import ISAB, PMA


class PeakSetEncoder(nn.Module):
    """Permutation-aware encoder for unordered peak features."""

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_inducing: int = 32,
        n_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [ISAB(d_model, n_heads, n_inducing=n_inducing, dropout=dropout) for _ in range(n_layers)]
        )
        self.pool = PMA(d_model, n_heads, n_seeds=n_seeds, dropout=dropout)
        self.reconstruction_head = MLP(d_model, [d_model], input_dim, dropout=dropout)

    def forward(
        self,
        peak_features: torch.Tensor,
        peak_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(peak_features)
        for layer in self.layers:
            x = layer(x, peak_mask)
            if peak_mask is not None:
                x = x * peak_mask.unsqueeze(-1).float()
        pooled = self.pool(x, peak_mask).squeeze(1)
        return {
            "peak_embeddings": x,
            "molecule_embedding": pooled,
            "reconstructed_peak_features": self.reconstruction_head(x),
        }
