"""
Set Attention Blocks (Set Transformer)
=======================================

Implements the core attention blocks from:
  Lee et al., "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Input" (ICML 2019).

Blocks:
  - MAB: Multihead Attention Block
  - SAB: Set Attention Block (self-attention, permutation-equivariant)
  - ISAB: Induced Set Attention Block (linear complexity via inducing points)
  - PMA: Pooling by Multihead Attention (set → fixed-size output)

All blocks maintain permutation equivariance of the set elements.
"""

from __future__ import annotations

import math

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block.

    Computes: MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Multihead(X, Y, Y))

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Query tensor, shape [B, N, d_model].
        Y : torch.Tensor
            Key/Value tensor, shape [B, M, d_model].
        mask : torch.Tensor or None
            Attention mask, shape [B, N, M] or [B, 1, M].

        Returns
        -------
        torch.Tensor
            Output, shape [B, N, d_model].
        """
        B, N, _ = X.shape
        M = Y.shape[1]

        # Multi-head attention
        Q = self.W_q(X).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(Y).view(B, M, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(Y).view(B, M, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # [B, 1, N, M]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)  # [B, n_heads, N, d_k]
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        context = self.W_o(context)

        # Residual + LayerNorm
        H = self.norm1(X + context)

        # Feedforward + Residual + LayerNorm
        return self.norm2(H + self.ff(H))


class SAB(nn.Module):
    """Set Attention Block.

    Self-attention over set elements: SAB(X) = MAB(X, X).
    Permutation equivariant — reordering inputs reorders outputs.

    Complexity: O(n²) in set size n.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Set elements, shape [B, N, d_model].
        mask : torch.Tensor or None
            Element validity mask, shape [B, N]. True/1 = valid.

        Returns
        -------
        torch.Tensor
            Output, shape [B, N, d_model].
        """
        attn_mask = None
        if mask is not None:
            # Create pairwise mask [B, N, N]
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return self.mab(X, X, attn_mask)


class ISAB(nn.Module):
    """Induced Set Attention Block.

    Uses m learned inducing points to reduce complexity from O(n²) to O(nm).

    ISAB(X) = MAB(X, H)  where  H = MAB(I, X)
    I ∈ R^{m × d} are the inducing points (learned).

    Still permutation equivariant.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_inducing : int
        Number of inducing points.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, n_heads, dropout)  # I attends to X
        self.mab2 = MAB(d_model, n_heads, dropout)  # X attends to H

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Set elements, shape [B, N, d_model].
        mask : torch.Tensor or None
            Element validity mask, shape [B, N].

        Returns
        -------
        torch.Tensor
            Output, shape [B, N, d_model].
        """
        B = X.shape[0]
        I = self.inducing_points.expand(B, -1, -1)  # [B, m, d]

        # Inducing points attend to set elements
        attn_mask1 = None
        if mask is not None:
            # [B, m, N] — inducing points can attend to valid elements
            attn_mask1 = mask.unsqueeze(1).expand(-1, I.shape[1], -1)

        H = self.mab1(I, X, attn_mask1)  # [B, m, d]

        # Set elements attend to inducing summaries
        return self.mab2(X, H)  # [B, N, d]


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Aggregates a set into k fixed-size output vectors using learned seed
    vectors that attend to the set.

    PMA(Z) = MAB(S, Z)  where S ∈ R^{k × d} are the seed vectors.

    Permutation invariant (output does not depend on input order).

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_seeds : int
        Number of output seed vectors. Use 1 for a single set summary.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(
        self,
        Z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        Z : torch.Tensor
            Set elements, shape [B, N, d_model].
        mask : torch.Tensor or None
            Element validity mask, shape [B, N].

        Returns
        -------
        torch.Tensor
            Pooled output, shape [B, n_seeds, d_model].
        """
        B = Z.shape[0]
        S = self.seeds.expand(B, -1, -1)  # [B, k, d]

        attn_mask = None
        if mask is not None:
            # [B, k, N]
            attn_mask = mask.unsqueeze(1).expand(-1, S.shape[1], -1)

        return self.mab(S, Z, attn_mask)
