"""Stage 4 NOESY assignment and structure refinement."""

from __future__ import annotations

import torch
import torch.nn as nn


class NOESYStructureRefiner(nn.Module):
    """Differentiable NOESY scorer with lightweight coordinate refinement."""

    def __init__(
        self,
        d_model: int = 256,
        shift_tolerance: float = 0.03,
        distance_weight: float = 1.0,
        n_iterations: int = 5,
        refinement_lr: float = 0.01,
    ):
        super().__init__()
        self.shift_tolerance = shift_tolerance
        self.distance_weight = distance_weight
        self.n_iterations = n_iterations
        self.refinement_lr = refinement_lr
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        shifts_h: torch.Tensor,
        coordinates: torch.Tensor,
        cross_peaks: torch.Tensor,
        cross_peak_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score cross peaks and refine coordinates.

        `cross_peaks` is expected as ``[B, P, 4]`` containing
        ``i, j, distance, intensity``. The first two columns can be soft or hard
        residue indices from an upstream assignment step.
        """

        coords = coordinates
        i_idx = cross_peaks[..., 0].long().clamp(0, coordinates.shape[1] - 1)
        j_idx = cross_peaks[..., 1].long().clamp(0, coordinates.shape[1] - 1)
        batch_index = torch.arange(coordinates.shape[0], device=coordinates.device).unsqueeze(1)

        pred_distance = torch.norm(
            coords[batch_index, i_idx] - coords[batch_index, j_idx],
            dim=-1,
        ).clamp_min(1e-6)
        target_distance = cross_peaks[..., 2].clamp_min(1e-6)
        intensity = cross_peaks[..., 3]
        shift_delta = (shifts_h[batch_index, i_idx] - shifts_h[batch_index, j_idx]).abs()
        edge_features = torch.stack(
            [pred_distance, target_distance, intensity, shift_delta], dim=-1
        )
        network_scores = self.edge_encoder(edge_features).squeeze(-1)
        distance_scores = -self.distance_weight * (pred_distance - target_distance).abs()
        shift_scores = -shift_delta / max(self.shift_tolerance, 1e-6)
        assignment_logits = network_scores + distance_scores + shift_scores
        if cross_peak_mask is not None:
            assignment_logits = assignment_logits.masked_fill(cross_peak_mask.eq(0), -1e9)

        refined = self._refine(coords, i_idx, j_idx, target_distance, cross_peak_mask)
        return {
            "noe_assignment_logits": assignment_logits,
            "predicted_distances": pred_distance,
            "refined_coordinates": refined,
        }

    def _refine(
        self,
        coordinates: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        target_distance: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        refined = coordinates.clone()
        batch_index = torch.arange(coordinates.shape[0], device=coordinates.device).unsqueeze(1)
        weight = 1.0 if mask is None else mask.float()
        for _ in range(self.n_iterations):
            vec = refined[batch_index, i_idx] - refined[batch_index, j_idx]
            dist = torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
            error = (dist.squeeze(-1) - target_distance) * weight
            direction = vec / dist
            step = self.refinement_lr * error.unsqueeze(-1) * direction
            refined = refined.clone()
            refined[batch_index, i_idx] -= step
            refined[batch_index, j_idx] += step
        return refined
