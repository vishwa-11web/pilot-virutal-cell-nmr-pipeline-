"""
PyTorch Dataset Classes for NMR Data
=====================================

Provides Dataset and collation utilities for training the NMR-to-physics
pipeline. Handles variable-length peak lists, multi-experiment bundling,
and BMRB shift+sequence pairs.

Datasets:
  - HSQCDataset: 2D spectra as image tensors + peak list labels
  - MultiExperimentDataset: Bundles HSQC + NOESY + CEST + relaxation
  - BMRBShiftDataset: Chemical shift + sequence pairs from BMRB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Amino acid vocabulary for sequence encoding
AA_VOCAB = {
    "<pad>": 0, "<cls>": 1, "<eos>": 2, "<unk>": 3,
    "A": 4, "R": 5, "N": 6, "D": 7, "C": 8, "Q": 9, "E": 10,
    "G": 11, "H": 12, "I": 13, "L": 14, "K": 15, "M": 16,
    "F": 17, "P": 18, "S": 19, "T": 20, "W": 21, "Y": 22, "V": 23,
}
AA_VOCAB_SIZE = len(AA_VOCAB)


def encode_sequence(sequence: str, max_len: int | None = None) -> torch.Tensor:
    """Encode amino acid sequence to integer tensor.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    max_len : int or None
        Maximum length (pads/truncates if set).

    Returns
    -------
    torch.Tensor
        Integer-encoded sequence, shape [L] or [max_len].
    """
    tokens = [AA_VOCAB.get("<cls>", 1)]
    for aa in sequence:
        tokens.append(AA_VOCAB.get(aa, AA_VOCAB["<unk>"]))
    tokens.append(AA_VOCAB.get("<eos>", 2))

    if max_len is not None:
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([AA_VOCAB["<pad>"]] * (max_len - len(tokens)))

    return torch.tensor(tokens, dtype=torch.long)


class HSQCDataset(Dataset):
    """Dataset of 2D HSQC spectra with peak annotations.

    Each sample contains a 2D spectrum image tensor and ground-truth
    peak positions, intensities, and linewidths for training the
    Stage 1 CNN encoder.

    Parameters
    ----------
    spectra : list[dict]
        List of spectrum dictionaries from SyntheticSpectraGenerator.generate_hsqc().
        Each must have keys: 'spectrum', 'peak_positions', 'peak_indices',
        'linewidths', 'intensities'.
    sequences : list[str] or None
        Corresponding amino acid sequences.
    transform : callable or None
        Optional augmentation transform applied to spectra.
    max_peaks : int
        Maximum number of peaks per spectrum (for padding).
    """

    def __init__(
        self,
        spectra: list[dict],
        sequences: list[str] | None = None,
        transform: Any | None = None,
        max_peaks: int = 200,
    ):
        self.spectra = spectra
        self.sequences = sequences
        self.transform = transform
        self.max_peaks = max_peaks

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec_data = self.spectra[idx]

        # Spectrum as image tensor [1, H, W]
        spectrum = torch.from_numpy(spec_data["spectrum"]).unsqueeze(0).float()

        # Apply augmentation if provided
        if self.transform is not None:
            spectrum = self.transform(spectrum)

        # Peak heatmap (ground truth for detection head)
        heatmap = self._create_peak_heatmap(
            spec_data["peak_indices"],
            spec_data["spectrum"].shape,
            sigma=2.0,
        )

        # Peak properties (ground truth for regression head)
        n_peaks = len(spec_data["intensities"])
        peak_positions = np.zeros((self.max_peaks, 2), dtype=np.float32)
        peak_intensities = np.zeros(self.max_peaks, dtype=np.float32)
        peak_linewidths = np.zeros((self.max_peaks, 2), dtype=np.float32)
        peak_mask = np.zeros(self.max_peaks, dtype=np.float32)

        n_fill = min(n_peaks, self.max_peaks)
        if n_fill > 0:
            peak_positions[:n_fill] = spec_data["peak_positions"][:n_fill]
            peak_intensities[:n_fill] = spec_data["intensities"][:n_fill]
            peak_linewidths[:n_fill] = spec_data["linewidths"][:n_fill]
            peak_mask[:n_fill] = 1.0

        result = {
            "spectrum": spectrum,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0).float(),
            "peak_positions": torch.from_numpy(peak_positions),
            "peak_intensities": torch.from_numpy(peak_intensities),
            "peak_linewidths": torch.from_numpy(peak_linewidths),
            "peak_mask": torch.from_numpy(peak_mask),
            "n_peaks": torch.tensor(n_fill, dtype=torch.long),
        }

        if self.sequences is not None:
            result["sequence_tokens"] = encode_sequence(self.sequences[idx])

        return result

    @staticmethod
    def _create_peak_heatmap(
        peak_indices: np.ndarray,
        shape: tuple[int, int],
        sigma: float = 2.0,
    ) -> np.ndarray:
        """Create a Gaussian heatmap from peak index positions.

        Parameters
        ----------
        peak_indices : np.ndarray
            Peak grid positions, shape [N, 2].
        shape : tuple
            Spectrum grid shape (H, W).
        sigma : float
            Gaussian kernel width in pixels.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W), values in [0, 1].
        """
        heatmap = np.zeros(shape, dtype=np.float32)

        for k in range(len(peak_indices)):
            ci, cj = int(peak_indices[k, 0]), int(peak_indices[k, 1])

            # Only render within ±3σ for efficiency
            r = int(3 * sigma)
            i_min, i_max = max(0, ci - r), min(shape[0], ci + r + 1)
            j_min, j_max = max(0, cj - r), min(shape[1], cj + r + 1)

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    dist_sq = (i - ci) ** 2 + (j - cj) ** 2
                    heatmap[i, j] = max(
                        heatmap[i, j],
                        np.exp(-dist_sq / (2 * sigma ** 2)),
                    )

        return heatmap


class BMRBShiftDataset(Dataset):
    """Dataset of BMRB chemical shift assignments paired with sequences.

    Used for training the Stage 3 assignment model. Each sample is a
    (sequence, chemical_shifts) pair where shifts are organized per residue.

    Parameters
    ----------
    entries : list[dict]
        BMRB entries from BMRBClient.build_training_dataset().
        Each must have keys: 'sequence', 'shifts' (DataFrame).
    max_seq_len : int
        Maximum sequence length for padding.
    max_peaks : int
        Maximum number of shift observations per entry.
    nuclei : list[str]
        Which nuclei to include ('H', 'N', 'CA', 'CB', etc.).
    """

    def __init__(
        self,
        entries: list[dict],
        max_seq_len: int = 200,
        max_peaks: int = 300,
        nuclei: list[str] | None = None,
    ):
        self.entries = entries
        self.max_seq_len = max_seq_len
        self.max_peaks = max_peaks
        self.nuclei = nuclei or ["H", "N", "CA", "CB"]
        self._preprocess()

    def _preprocess(self) -> None:
        """Pre-process entries into tensors."""
        self.samples = []
        for entry in self.entries:
            seq = entry.get("sequence", "")
            shifts_df = entry.get("shifts")
            if not seq or shifts_df is None or len(shifts_df) == 0:
                continue

            # Build per-residue shift matrix [L, n_nuclei]
            L = len(seq)
            n_nuc = len(self.nuclei)
            shift_matrix = np.full((L, n_nuc), float("nan"), dtype=np.float32)
            shift_mask = np.zeros((L, n_nuc), dtype=np.float32)

            for _, row in shifts_df.iterrows():
                seq_id = row.get("seq_id")
                atom_id = str(row.get("atom_id", "")).upper()
                shift_val = row.get("shift_value")

                if seq_id is None or shift_val is None:
                    continue

                try:
                    res_idx = int(seq_id) - 1  # 1-indexed → 0-indexed
                except (ValueError, TypeError):
                    continue

                if res_idx < 0 or res_idx >= L:
                    continue

                # Map atom_id to nucleus index
                for nuc_idx, nuc in enumerate(self.nuclei):
                    if atom_id == nuc or (nuc == "H" and atom_id == "HN"):
                        shift_matrix[res_idx, nuc_idx] = float(shift_val)
                        shift_mask[res_idx, nuc_idx] = 1.0
                        break

            # Build peak list (non-NaN entries as flat list)
            peak_features = []  # Each: [shift_value, nucleus_idx, residue_idx]
            for i in range(L):
                for j in range(n_nuc):
                    if shift_mask[i, j] > 0:
                        peak_features.append([shift_matrix[i, j], float(j), float(i)])

            if len(peak_features) == 0:
                continue

            self.samples.append({
                "sequence": seq,
                "shift_matrix": shift_matrix,
                "shift_mask": shift_mask,
                "peak_features": np.array(peak_features, dtype=np.float32),
                "entry_id": entry.get("entry_id", "unknown"),
            })

        logger.info("BMRBShiftDataset: %d valid samples from %d entries",
                     len(self.samples), len(self.entries))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Encode sequence
        seq_tokens = encode_sequence(sample["sequence"], max_len=self.max_seq_len)

        # Shift matrix [L, n_nuclei] — padded to max_seq_len
        L = len(sample["sequence"])
        shift_matrix = np.zeros((self.max_seq_len, len(self.nuclei)), dtype=np.float32)
        shift_mask = np.zeros((self.max_seq_len, len(self.nuclei)), dtype=np.float32)
        fill_len = min(L, self.max_seq_len)
        shift_matrix[:fill_len] = sample["shift_matrix"][:fill_len]
        shift_mask[:fill_len] = sample["shift_mask"][:fill_len]

        # Replace NaN with 0 for tensor compatibility
        shift_matrix = np.nan_to_num(shift_matrix, nan=0.0)

        # Peak features [max_peaks, 3] — padded
        pf = sample["peak_features"]
        n_peaks = min(len(pf), self.max_peaks)
        peak_features = np.zeros((self.max_peaks, 3), dtype=np.float32)
        peak_mask = np.zeros(self.max_peaks, dtype=np.float32)
        peak_features[:n_peaks] = pf[:n_peaks]
        peak_mask[:n_peaks] = 1.0

        # Assignment ground truth: for each peak, which residue it belongs to
        # peak_features[:, 2] already contains residue index
        assignment_target = np.zeros(self.max_peaks, dtype=np.int64)
        assignment_target[:n_peaks] = pf[:n_peaks, 2].astype(np.int64)

        return {
            "sequence_tokens": seq_tokens,
            "seq_len": torch.tensor(min(L, self.max_seq_len), dtype=torch.long),
            "shift_matrix": torch.from_numpy(shift_matrix),
            "shift_mask": torch.from_numpy(shift_mask),
            "peak_features": torch.from_numpy(peak_features),
            "peak_mask": torch.from_numpy(peak_mask),
            "assignment_target": torch.from_numpy(assignment_target),
            "n_peaks": torch.tensor(n_peaks, dtype=torch.long),
        }


class MultiExperimentDataset(Dataset):
    """Dataset bundling multiple NMR experiments for the same molecule.

    Combines HSQC, NOESY, CEST, relaxation, and J-coupling data into
    a single training sample for Stages 4–5 (structure refinement and
    physics extrapolation).

    Parameters
    ----------
    molecule_data : list[dict]
        List of complete molecule datasets from
        SyntheticSpectraGenerator.generate_molecule_data().
    max_seq_len : int
        Maximum sequence length.
    max_peaks : int
        Maximum peaks per experiment.
    hsqc_grid_size : tuple
        Expected HSQC grid dimensions.
    """

    def __init__(
        self,
        molecule_data: list[dict],
        max_seq_len: int = 150,
        max_peaks: int = 200,
        hsqc_grid_size: tuple[int, int] = (256, 512),
    ):
        self.molecule_data = molecule_data
        self.max_seq_len = max_seq_len
        self.max_peaks = max_peaks
        self.hsqc_grid_size = hsqc_grid_size

    def __len__(self) -> int:
        return len(self.molecule_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mol = self.molecule_data[idx]
        seq = mol["sequence"]
        L = min(len(seq), self.max_seq_len)

        result: dict[str, torch.Tensor] = {}

        # Sequence
        result["sequence_tokens"] = encode_sequence(seq, max_len=self.max_seq_len)
        result["seq_len"] = torch.tensor(L, dtype=torch.long)

        # --- Chemical Shifts ---
        shifts = mol.get("shifts", {})
        n_nuc = 4  # H, N, CA, CB
        shift_tensor = torch.zeros(self.max_seq_len, n_nuc)
        shift_mask = torch.zeros(self.max_seq_len, n_nuc)
        for j, nuc in enumerate(["H", "N", "CA", "CB"]):
            if nuc in shifts:
                vals = shifts[nuc]
                fill = min(len(vals), self.max_seq_len)
                shift_tensor[:fill, j] = torch.from_numpy(vals[:fill]).float()
                shift_mask[:fill, j] = 1.0
        result["shifts"] = shift_tensor
        result["shift_mask"] = shift_mask

        # --- HSQC Spectrum ---
        if "hsqc" in mol:
            hsqc = mol["hsqc"]
            spectrum = torch.from_numpy(hsqc["spectrum"]).unsqueeze(0).float()
            result["hsqc_spectrum"] = spectrum
        else:
            h, w = self.hsqc_grid_size
            result["hsqc_spectrum"] = torch.zeros(1, h, w)

        # --- J-Couplings ---
        if "j_couplings" in mol:
            jc = mol["j_couplings"]
            j_data = torch.zeros(self.max_seq_len, 2)  # J_HNHa, J_HaC
            j_mask = torch.zeros(self.max_seq_len)
            fill = min(len(jc["J_HNHa"]), self.max_seq_len)
            j_data[:fill, 0] = torch.from_numpy(jc["J_HNHa"][:fill]).float()
            j_data[:fill, 1] = torch.from_numpy(jc["J_HaC"][:fill]).float()
            j_mask[:fill] = 1.0
            result["j_couplings"] = j_data
            result["j_coupling_mask"] = j_mask
            result["phi_angles"] = torch.from_numpy(
                jc["phi"][:self.max_seq_len] if len(jc["phi"]) > self.max_seq_len
                else np.pad(jc["phi"], (0, self.max_seq_len - len(jc["phi"])))
            ).float()
            result["psi_angles"] = torch.from_numpy(
                jc["psi"][:self.max_seq_len] if len(jc["psi"]) > self.max_seq_len
                else np.pad(jc["psi"], (0, self.max_seq_len - len(jc["psi"])))
            ).float()

        # --- Relaxation Rates ---
        if "relaxation" in mol:
            relax = mol["relaxation"]
            n_fields = relax["R1"].shape[1] if relax["R1"].ndim > 1 else 1
            relax_data = torch.zeros(self.max_seq_len, 3, n_fields)
            relax_mask = torch.zeros(self.max_seq_len)
            fill = min(relax["R1"].shape[0], self.max_seq_len)

            for k, key in enumerate(["R1", "R2", "hetNOE"]):
                arr = relax[key]
                if arr.ndim == 1:
                    arr = arr[:, None]
                relax_data[:fill, k, :] = torch.from_numpy(arr[:fill]).float()
            relax_mask[:fill] = 1.0
            result["relaxation"] = relax_data
            result["relaxation_mask"] = relax_mask

        # --- NOESY distances ---
        if "noesy" in mol:
            noesy = mol["noesy"]
            # Distance matrix [L, L]
            dist = noesy.get("distances")
            if dist is not None:
                d = min(dist.shape[0], self.max_seq_len)
                dist_tensor = torch.zeros(self.max_seq_len, self.max_seq_len)
                dist_tensor[:d, :d] = torch.from_numpy(dist[:d, :d]).float()
                result["noe_distances"] = dist_tensor

            # Cross-peak list
            cross_peaks = noesy.get("cross_peaks", [])
            n_cp = min(len(cross_peaks), self.max_peaks)
            cp_tensor = torch.zeros(self.max_peaks, 4)  # i, j, distance, intensity
            cp_mask = torch.zeros(self.max_peaks)
            for k in range(n_cp):
                cp_tensor[k, 0] = cross_peaks[k][0]
                cp_tensor[k, 1] = cross_peaks[k][1]
                cp_tensor[k, 2] = cross_peaks[k][2]
                cp_tensor[k, 3] = cross_peaks[k][3]
            cp_mask[:n_cp] = 1.0
            result["cross_peaks"] = cp_tensor
            result["cross_peak_mask"] = cp_mask

        # --- 3D Coordinates ---
        if "coordinates" in mol:
            coords = mol["coordinates"]
            fill = min(coords.shape[0], self.max_seq_len)
            coord_tensor = torch.zeros(self.max_seq_len, 3)
            coord_tensor[:fill] = torch.from_numpy(coords[:fill]).float()
            result["coordinates"] = coord_tensor

        # --- Model-free dynamics params (targets) ---
        if "model_free_params" in mol:
            mfp = mol["model_free_params"]
            fill = min(len(mfp), self.max_seq_len)
            dynamics_targets = torch.zeros(self.max_seq_len, 4)
            for k in range(fill):
                dynamics_targets[k, 0] = mfp[k]["tau_c"]
                dynamics_targets[k, 1] = mfp[k]["S2"]
                dynamics_targets[k, 2] = mfp[k]["tau_e"]
                dynamics_targets[k, 3] = mfp[k]["rex"]
            result["dynamics_targets"] = dynamics_targets

        return result


# ---------------------------------------------------------------------------
# Collation Functions
# ---------------------------------------------------------------------------

def collate_hsqc(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for HSQCDataset that handles variable peak counts."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)
    return collated


def collate_shifts(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for BMRBShiftDataset."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)
    return collated


def collate_multi_experiment(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for MultiExperimentDataset.

    Handles missing modalities by zero-filling absent keys.
    """
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())

    collated = {}
    for key in all_keys:
        tensors = []
        reference_shape = None
        for sample in batch:
            if key in sample:
                tensors.append(sample[key])
                if reference_shape is None:
                    reference_shape = sample[key].shape
            else:
                # Use reference shape from a sample that has this key
                if reference_shape is not None:
                    tensors.append(torch.zeros(reference_shape, dtype=torch.float32))

        if tensors:
            try:
                collated[key] = torch.stack(tensors, dim=0)
            except RuntimeError:
                # Shape mismatch — skip this key
                logger.warning("Could not collate key '%s' due to shape mismatch", key)

    return collated
