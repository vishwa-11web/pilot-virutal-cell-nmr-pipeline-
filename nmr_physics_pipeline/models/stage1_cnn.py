"""Stage 1 spectrum CNN encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


class ConvBranch(nn.Module):
    """Multi-scale convolution branch for spectra."""

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                ]
            )
            current = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectrumCNNEncoder(nn.Module):
    """Per-spectrum CNN with peak heatmap and peak-feature outputs."""

    def __init__(
        self,
        in_channels: int = 1,
        branches: list[dict] | None = None,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        score_threshold: float = 0.3,
        max_peaks: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        branches = branches or [
            {"kernel_size": 3, "channels": [32, 64, 128]},
            {"kernel_size": 7, "channels": [32, 64, 128]},
            {"kernel_size": 15, "channels": [16, 32, 64]},
        ]
        self.score_threshold = score_threshold
        self.max_peaks = max_peaks
        self.branches = nn.ModuleList(
            [
                ConvBranch(in_channels, list(branch["channels"]), branch["kernel_size"], dropout)
                for branch in branches
            ]
        )
        fused_channels = sum(branch["channels"][-1] for branch in branches)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.peak_head = nn.Conv2d(hidden_dim, 1, 1)
        self.feature_head = MLP(hidden_dim + 2, [hidden_dim], feature_dim, dropout=dropout)
        self.property_head = MLP(feature_dim, [hidden_dim], 3, dropout=dropout)

    def forward(
        self,
        spectrum: torch.Tensor,
        peak_indices: torch.Tensor | None = None,
        max_peaks: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder.

        Parameters
        ----------
        spectrum:
            Tensor of shape ``[B, C, H, W]``.
        peak_indices:
            Optional integer peak locations ``[B, N, 2]`` in row, column order.
        max_peaks:
            Number of peaks to select from the heatmap when `peak_indices` is absent.
        """

        features = torch.cat([branch(spectrum) for branch in self.branches], dim=1)
        feature_map = self.fusion(features)
        heatmap_logits = self.peak_head(feature_map)
        heatmap = torch.sigmoid(heatmap_logits)

        if peak_indices is None:
            peak_indices, peak_mask = self._topk_peak_indices(heatmap, max_peaks or self.max_peaks)
        else:
            peak_mask = torch.ones(
                peak_indices.shape[:2], dtype=spectrum.dtype, device=spectrum.device
            )

        peak_features = self._sample_peak_features(feature_map, peak_indices)
        peak_properties = self.property_head(peak_features)

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "feature_map": feature_map,
            "peak_indices": peak_indices,
            "peak_mask": peak_mask,
            "peak_features": peak_features,
            "peak_properties": peak_properties,
        }

    @staticmethod
    def _topk_peak_indices(heatmap: torch.Tensor, max_peaks: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = heatmap.shape
        pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        local_max = heatmap * (heatmap == pooled)
        scores, flat = local_max.flatten(2).topk(k=min(max_peaks, height * width), dim=-1)
        rows = flat // width
        cols = flat % width
        indices = torch.stack([rows.squeeze(1), cols.squeeze(1)], dim=-1)
        mask = (scores.squeeze(1) > 0).float()
        return indices.long(), mask

    def _sample_peak_features(self, feature_map: torch.Tensor, peak_indices: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature_map.shape
        rows = peak_indices[..., 0].float().clamp(0, height - 1)
        cols = peak_indices[..., 1].float().clamp(0, width - 1)
        grid_x = 2.0 * cols / max(width - 1, 1) - 1.0
        grid_y = 2.0 * rows / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(feature_map, grid, align_corners=True).squeeze(-1).transpose(1, 2)
        norm_coords = torch.stack([rows / max(height - 1, 1), cols / max(width - 1, 1)], dim=-1)
        return self.feature_head(torch.cat([sampled, norm_coords], dim=-1))
