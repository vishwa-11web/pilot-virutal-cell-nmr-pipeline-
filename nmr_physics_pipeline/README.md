# NMR Physics Pipeline

NMR Physics Pipeline is a research-oriented Python package for prototyping an
end-to-end path from NMR observables to molecular physics targets.

The project currently provides:

- BMRB data access helpers and local caching.
- Signal preprocessing for 1D and 2D spectra.
- Synthetic HSQC, NOESY, CEST, relaxation, and J-coupling generation.
- PyTorch datasets for spectra, chemical shifts, and multi-experiment samples.
- Five neural model stages matching the configuration files in `configs/`.

## Install

```bash
pip install -e ".[dev]"
```

Some scientific dependencies, especially `torch-scatter`, can require
platform-specific wheels. If installation fails, install PyTorch and
torch-scatter with the wheel index recommended for your machine, then rerun the
editable install.

## Quick Start

```python
from nmr_physics_pipeline.data.synthetic_spectra import SyntheticSpectraGenerator
from nmr_physics_pipeline.models import SpectrumCNNEncoder, PeakSetEncoder, SequenceAssignmentNetwork

generator = SyntheticSpectraGenerator(random_seed=7)
sample = generator.generate_molecule_data("ACDEFGHIKLMNPQRSTVWY")

stage1 = SpectrumCNNEncoder()
stage2 = PeakSetEncoder()
stage3 = SequenceAssignmentNetwork()
```

The package is intentionally modular: stages can be trained independently, then
connected through tensors described in the dataset classes.

## Model Stages

1. `SpectrumCNNEncoder`: converts spectra into heatmaps and per-peak features.
2. `PeakSetEncoder`: encodes unordered peak sets with Set Transformer blocks.
3. `SequenceAssignmentNetwork`: assigns peaks to sequence positions with
   cross-attention and Sinkhorn-style normalization.
4. `NOESYStructureRefiner`: scores NOESY assignments and performs differentiable
   coordinate refinement.
5. `MolecularStatePhysics`: fuses observables into a latent molecular state and
   predicts thermodynamics, kinetics, forces, and chemistry targets.

## Tests

```bash
pytest
```
