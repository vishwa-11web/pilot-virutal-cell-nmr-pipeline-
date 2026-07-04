"""
NMR-to-Physics AI/ML Pipeline
==============================

A unified deep learning framework that takes raw NMR spectral observables
(chemical shifts, J-couplings, NOEs, relaxation, CEST) and extrapolates
to thermodynamics, kinetics, interaction forces, and chemical reactivity.

Architecture:
    Stage 1: Per-spectrum CNN encoder (peak detection + lineshape features)
    Stage 2: Set Transformer peak encoder (permutation-equivariant)
    Stage 3: Sequence-conditioned cross-attention assignment
    Stage 4: Iterative NOESY assignment + structure feedback
    Stage 5: Shared latent molecular state → physics extrapolation heads
"""

__version__ = "0.1.0"
