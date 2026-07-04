"""Stage 3 sequence-conditioned assignment network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SequenceAssignmentNetwork(nn.Module):
    """Assign peak observations to positions in an amino acid sequence."""

    def __init__(
        self,
        peak_dim: int = 3,
        vocab_size: int = 24,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 10,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.peak_proj = nn.Linear(peak_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.score = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        peak_features: torch.Tensor,
        peak_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = sequence_tokens.shape[1]
        seq_padding_mask = sequence_tokens.eq(0)
        seq = self.token_embedding(sequence_tokens) + self.position_embedding[:, :seq_len]
        residue_embeddings = self.sequence_encoder(seq, src_key_padding_mask=seq_padding_mask)

        peak_embeddings = self.peak_proj(peak_features)
        attended_peaks, _ = self.cross_attention(
            peak_embeddings,
            residue_embeddings,
            residue_embeddings,
            key_padding_mask=seq_padding_mask,
        )
        logits = torch.matmul(self.score(attended_peaks), residue_embeddings.transpose(1, 2))
        logits = logits / math.sqrt(residue_embeddings.shape[-1])
        logits = logits.masked_fill(seq_padding_mask.unsqueeze(1), -1e9)
        if peak_mask is not None:
            logits = logits.masked_fill(peak_mask.unsqueeze(-1).eq(0), -1e9)

        assignment_probs = self._sinkhorn(logits / max(self.temperature, 1e-6), peak_mask)
        return {
            "assignment_logits": logits,
            "assignment_probs": assignment_probs,
            "peak_embeddings": attended_peaks,
            "residue_embeddings": residue_embeddings,
        }

    def _sinkhorn(self, logits: torch.Tensor, peak_mask: torch.Tensor | None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if peak_mask is not None:
            probs = probs * peak_mask.unsqueeze(-1).float()
        for _ in range(self.sinkhorn_iterations):
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            if peak_mask is not None:
                probs = probs * peak_mask.unsqueeze(-1).float()
            probs = probs / probs.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        return probs
