import numpy as np
import torch

from nmr_physics_pipeline.data.synthetic_spectra import SyntheticSpectraGenerator
from nmr_physics_pipeline.models import (
    MolecularStatePhysics,
    NOESYStructureRefiner,
    PeakSetEncoder,
    SequenceAssignmentNetwork,
    SpectrumCNNEncoder,
)


def test_synthetic_molecule_generation():
    generator = SyntheticSpectraGenerator(random_seed=1)
    molecule = generator.generate_molecule_data("ACDEFGHIK", include_exchange=True)

    assert molecule["hsqc"]["spectrum"].ndim == 2
    assert molecule["coordinates"].shape == (9, 3)
    assert "cest" in molecule


def test_model_stage_forward_passes():
    spectrum = torch.randn(2, 1, 64, 96)
    stage1 = SpectrumCNNEncoder(max_peaks=8)
    out1 = stage1(spectrum, max_peaks=8)

    assert out1["peak_features"].shape == (2, 8, 128)

    stage2 = PeakSetEncoder(input_dim=128, d_model=64, n_heads=4, n_layers=1)
    out2 = stage2(out1["peak_features"], out1["peak_mask"])
    assert out2["molecule_embedding"].shape == (2, 64)

    seq = torch.tensor([[1, 4, 5, 6, 2, 0], [1, 7, 8, 9, 10, 2]])
    peaks = torch.randn(2, 8, 3)
    stage3 = SequenceAssignmentNetwork(d_model=64, n_layers=1, n_heads=4, max_seq_len=16)
    out3 = stage3(seq, peaks, torch.ones(2, 8))
    assert out3["assignment_logits"].shape == (2, 8, 6)

    coords = torch.randn(2, 6, 3)
    shifts_h = torch.randn(2, 6)
    cross_peaks = torch.tensor(
        [
            [[0, 1, 3.0, 0.5], [2, 3, 4.0, 0.2]],
            [[1, 2, 3.5, 0.4], [3, 4, 4.5, 0.1]],
        ],
        dtype=torch.float32,
    )
    stage4 = NOESYStructureRefiner(d_model=32, n_iterations=1)
    out4 = stage4(shifts_h, coords, cross_peaks)
    assert out4["refined_coordinates"].shape == coords.shape

    batch = {
        "shifts": torch.randn(2, 6, 4),
        "shift_mask": torch.ones(2, 6, 4),
        "j_couplings": torch.randn(2, 6, 2),
        "relaxation": torch.randn(2, 6, 3, 1),
        "coordinates": coords,
    }
    stage5 = MolecularStatePhysics(d_model=64, d_residue=32, latent_dim=16)
    out5 = stage5(batch)
    assert out5["thermodynamics"].shape == (2, 3)
    assert out5["forces"].shape == (2, 6, 3)
