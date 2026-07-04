"""Stage 5 shared molecular state and physics heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLP, masked_mean


class MolecularStatePhysics(nn.Module):
    """Fuse NMR observables and predict molecular physics quantities."""

    def __init__(
        self,
        d_model: int = 512,
        d_residue: int = 256,
        max_seq_len: int = 150,
        variational: bool = True,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variational = variational
        self.shift_encoder = MLP(4, [128], d_residue, dropout=dropout)
        self.coupling_encoder = MLP(2, [64], d_residue, dropout=dropout)
        self.relaxation_encoder = MLP(3, [128], d_residue, dropout=dropout)
        self.structure_encoder = MLP(3, [128], d_residue, dropout=dropout)
        self.modality_proj = nn.Linear(d_residue, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)
        head_input = latent_dim if variational else d_model
        self.thermodynamics_head = MLP(head_input, [512, 256, 128], 3, dropout=dropout)
        self.kinetics_head = MLP(head_input, [512, 256, 128], 4, dropout=dropout)
        self.chemistry_head = MLP(d_model, [256, 128, 64], 3, dropout=dropout)
        self.forces_head = MLP(d_model, [256, 128], 3, dropout=dropout)
        self.max_seq_len = max_seq_len

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        residue_states = []
        residue_mask = batch.get("shift_mask")
        if residue_mask is not None:
            residue_mask = residue_mask.any(dim=-1).float()

        if "shifts" in batch:
            residue_states.append(self.shift_encoder(batch["shifts"]))
        if "j_couplings" in batch:
            residue_states.append(self.coupling_encoder(batch["j_couplings"]))
        if "relaxation" in batch:
            relaxation = batch["relaxation"]
            if relaxation.ndim == 4:
                relaxation = relaxation.mean(dim=-1)
            residue_states.append(self.relaxation_encoder(relaxation))
        if "coordinates" in batch:
            residue_states.append(self.structure_encoder(batch["coordinates"]))
            if residue_mask is None:
                residue_mask = torch.ones(
                    batch["coordinates"].shape[:2],
                    dtype=batch["coordinates"].dtype,
                    device=batch["coordinates"].device,
                )

        if not residue_states:
            raise ValueError("At least one residue-level observable is required.")

        residue_state = torch.stack(residue_states, dim=0).mean(dim=0)
        fused = self.fusion(self.modality_proj(residue_state))
        molecule_state = masked_mean(fused, residue_mask, dim=1)

        if self.variational:
            mu = self.mu(molecule_state)
            logvar = self.logvar(molecule_state).clamp(-10, 10)
            if self.training:
                std = torch.exp(0.5 * logvar)
                latent = mu + torch.randn_like(std) * std
            else:
                latent = mu
        else:
            mu = logvar = None
            latent = molecule_state

        outputs = {
            "residue_state": fused,
            "molecule_state": molecule_state,
            "thermodynamics": self.thermodynamics_head(latent),
            "kinetics": self.kinetics_head(latent),
            "forces": self.forces_head(fused),
            "chemistry": self.chemistry_head(fused),
        }
        if mu is not None and logvar is not None:
            outputs["latent_mu"] = mu
            outputs["latent_logvar"] = logvar
        return outputs
