"""Common neural network utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small configurable multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1) -> torch.Tensor:
    """Mean over a dimension while ignoring masked elements."""

    if mask is None:
        return x.mean(dim=dim)
    weight = mask.float().unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)
