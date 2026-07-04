"""
Spectral Data Augmentation
============================

Augmentation transforms for NMR spectra to improve model robustness.
Applied during training to simulate experimental variability:

  - Random frequency jitter (referencing errors)
  - Linewidth perturbation (temperature/viscosity variation)
  - Noise injection at variable SNR
  - Random peak dropout (missing assignments)
  - Spectral window cropping
  - Phase distortion
  - Baseline roll
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class SpectralAugmentor:
    """Compose and apply spectral augmentations during training.

    Parameters
    ----------
    shift_jitter : float
        Maximum chemical shift jitter in ppm (default 0.05).
    linewidth_scale : tuple[float, float]
        Range of linewidth scaling factors (default (0.7, 1.5)).
    noise_snr_range : tuple[float, float]
        SNR range for noise injection in dB (default (10, 40)).
    peak_dropout_prob : float
        Probability of dropping each peak (default 0.1).
    crop_prob : float
        Probability of random spectral cropping (default 0.2).
    crop_fraction : tuple[float, float]
        Range of crop sizes as fraction of spectrum (default (0.6, 0.95)).
    phase_distortion : float
        Maximum zero-order phase error in degrees (default 10.0).
    baseline_roll : float
        Maximum baseline roll amplitude (default 0.05).
    p : float
        Global probability of applying any augmentation (default 0.8).
    """

    def __init__(
        self,
        shift_jitter: float = 0.05,
        linewidth_scale: tuple[float, float] = (0.7, 1.5),
        noise_snr_range: tuple[float, float] = (10.0, 40.0),
        peak_dropout_prob: float = 0.1,
        crop_prob: float = 0.2,
        crop_fraction: tuple[float, float] = (0.6, 0.95),
        phase_distortion: float = 10.0,
        baseline_roll: float = 0.05,
        p: float = 0.8,
    ):
        self.shift_jitter = shift_jitter
        self.linewidth_scale = linewidth_scale
        self.noise_snr_range = noise_snr_range
        self.peak_dropout_prob = peak_dropout_prob
        self.crop_prob = crop_prob
        self.crop_fraction = crop_fraction
        self.phase_distortion = phase_distortion
        self.baseline_roll = baseline_roll
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a spectrum tensor.

        Parameters
        ----------
        spectrum : torch.Tensor
            Spectrum tensor, shape [C, H, W] (2D) or [C, N] (1D).

        Returns
        -------
        torch.Tensor
            Augmented spectrum.
        """
        if self.rng.random() > self.p:
            return spectrum

        # Apply each augmentation independently
        spectrum = self._add_noise(spectrum)
        spectrum = self._add_baseline_roll(spectrum)
        spectrum = self._add_phase_distortion(spectrum)

        if self.rng.random() < self.crop_prob:
            spectrum = self._random_crop(spectrum)

        return spectrum

    def augment_peaks(
        self,
        peak_positions: np.ndarray,
        peak_intensities: np.ndarray,
        peak_linewidths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Augment peak-level data (shifts, intensities, linewidths).

        Parameters
        ----------
        peak_positions : np.ndarray
            Peak positions in ppm, shape [N, D].
        peak_intensities : np.ndarray
            Peak intensities, shape [N].
        peak_linewidths : np.ndarray
            Peak linewidths, shape [N, D].

        Returns
        -------
        tuple
            (augmented_positions, augmented_intensities, augmented_linewidths, keep_mask)
        """
        n_peaks = len(peak_positions)

        # Chemical shift jitter
        jitter = self.rng.normal(0, self.shift_jitter, peak_positions.shape)
        aug_positions = peak_positions + jitter.astype(np.float32)

        # Linewidth scaling
        scale = self.rng.uniform(
            self.linewidth_scale[0], self.linewidth_scale[1], (n_peaks, 1)
        )
        aug_linewidths = peak_linewidths * scale.astype(np.float32)

        # Intensity perturbation (±20%)
        int_scale = self.rng.normal(1.0, 0.1, n_peaks)
        aug_intensities = peak_intensities * np.maximum(int_scale, 0.1).astype(np.float32)

        # Peak dropout
        keep_mask = self.rng.random(n_peaks) > self.peak_dropout_prob
        keep_mask = keep_mask.astype(np.float32)

        return aug_positions, aug_intensities, aug_linewidths, keep_mask

    def augment_shifts(
        self,
        shift_matrix: np.ndarray,
        shift_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment chemical shift values.

        Parameters
        ----------
        shift_matrix : np.ndarray
            Shift values, shape [L, n_nuclei].
        shift_mask : np.ndarray
            Valid shift mask, shape [L, n_nuclei].

        Returns
        -------
        tuple
            (augmented_shifts, augmented_mask) with jitter and dropout applied.
        """
        aug_shifts = shift_matrix.copy()
        aug_mask = shift_mask.copy()

        # Add jitter to existing shifts
        jitter = self.rng.normal(0, self.shift_jitter, shift_matrix.shape)
        aug_shifts += jitter.astype(np.float32) * shift_mask

        # Random dropout of individual assignments
        dropout = self.rng.random(shift_matrix.shape) > self.peak_dropout_prob
        aug_mask *= dropout.astype(np.float32)

        return aug_shifts, aug_mask

    # ------------------------------------------------------------------
    # Spectrum-level augmentations
    # ------------------------------------------------------------------

    def _add_noise(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        snr_db = self.rng.uniform(*self.noise_snr_range)
        signal_power = torch.mean(spectrum ** 2).item()
        if signal_power < 1e-12:
            return spectrum
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(spectrum) * np.sqrt(noise_power)
        return spectrum + noise

    def _add_baseline_roll(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Add a slowly-varying baseline distortion."""
        if self.baseline_roll <= 0:
            return spectrum

        amp = self.rng.uniform(0, self.baseline_roll)
        ndim = spectrum.ndim

        if ndim >= 3:
            # 2D spectrum: add polynomial baseline
            H, W = spectrum.shape[-2], spectrum.shape[-1]
            x = torch.linspace(-1, 1, W, device=spectrum.device)
            y = torch.linspace(-1, 1, H, device=spectrum.device)
            xx, yy = torch.meshgrid(y, x, indexing="ij")

            # Random low-order polynomial
            c = torch.from_numpy(self.rng.normal(0, amp, 6).astype(np.float32))
            baseline = (c[0] + c[1] * xx + c[2] * yy +
                       c[3] * xx ** 2 + c[4] * yy ** 2 + c[5] * xx * yy)
            spectrum = spectrum + baseline.unsqueeze(0)
        elif ndim == 2:
            # 1D spectrum
            N = spectrum.shape[-1]
            x = torch.linspace(-1, 1, N, device=spectrum.device)
            c = torch.from_numpy(self.rng.normal(0, amp, 3).astype(np.float32))
            baseline = c[0] + c[1] * x + c[2] * x ** 2
            spectrum = spectrum + baseline.unsqueeze(0)

        return spectrum

    def _add_phase_distortion(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Add zero-order phase error to spectrum.

        Simulates imperfect phase correction by mixing real and imaginary
        components. Applied as a rotation in the (real, imaginary) plane.
        """
        if self.phase_distortion <= 0:
            return spectrum

        angle = self.rng.uniform(-self.phase_distortion, self.phase_distortion)
        angle_rad = np.deg2rad(angle)

        # For real-only spectra, phase error manifests as baseline distortion
        # and peak shape changes. Approximate by mixing with Hilbert transform.
        # Simplified: small angle → spectrum ≈ spectrum * cos(θ) ≈ spectrum * (1 - θ²/2)
        cos_a = np.cos(angle_rad)
        return spectrum * cos_a

    def _random_crop(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Randomly crop and resize the spectrum.

        Simulates different spectral window settings.
        """
        frac = self.rng.uniform(*self.crop_fraction)

        if spectrum.ndim >= 3:
            C, H, W = spectrum.shape[-3], spectrum.shape[-2], spectrum.shape[-1]
            new_h = max(int(H * frac), 16)
            new_w = max(int(W * frac), 16)

            start_h = self.rng.integers(0, H - new_h + 1)
            start_w = self.rng.integers(0, W - new_w + 1)

            cropped = spectrum[..., start_h:start_h + new_h, start_w:start_w + new_w]
            # Resize back to original dimensions
            cropped = F.interpolate(
                cropped.unsqueeze(0) if cropped.ndim == 3 else cropped,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            if spectrum.ndim == 3:
                cropped = cropped.squeeze(0)
            return cropped
        else:
            return spectrum


class RelaxationAugmentor:
    """Augmentation for relaxation rate data.

    Simulates measurement uncertainty in R1, R2, and hetNOE values.

    Parameters
    ----------
    rate_noise_frac : float
        Fractional noise on rates (default 0.05 = 5%).
    field_jitter : float
        Jitter on field strength in MHz (default 0.5).
    """

    def __init__(
        self,
        rate_noise_frac: float = 0.05,
        field_jitter: float = 0.5,
    ):
        self.rate_noise_frac = rate_noise_frac
        self.field_jitter = field_jitter
        self.rng = np.random.default_rng()

    def __call__(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        hetNOE: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add noise to relaxation rates."""
        noise_R1 = self.rng.normal(0, self.rate_noise_frac * np.abs(R1), R1.shape)
        noise_R2 = self.rng.normal(0, self.rate_noise_frac * np.abs(R2), R2.shape)
        noise_NOE = self.rng.normal(0, 0.02, hetNOE.shape)  # ±0.02 absolute

        return (
            np.maximum(R1 + noise_R1, 0.01).astype(np.float32),
            np.maximum(R2 + noise_R2, 0.1).astype(np.float32),
            np.clip(hetNOE + noise_NOE, -1.0, 1.0).astype(np.float32),
        )


class CESTAugmentor:
    """Augmentation for CEST profiles.

    Simulates experimental noise and B1 field calibration errors.

    Parameters
    ----------
    intensity_noise : float
        Gaussian noise on CEST intensity ratio (default 0.02).
    b1_error : float
        Fractional B1 calibration error (default 0.05 = 5%).
    """

    def __init__(
        self,
        intensity_noise: float = 0.02,
        b1_error: float = 0.05,
    ):
        self.intensity_noise = intensity_noise
        self.b1_error = b1_error
        self.rng = np.random.default_rng()

    def __call__(self, profiles: np.ndarray) -> np.ndarray:
        """Add noise to CEST profiles.

        Parameters
        ----------
        profiles : np.ndarray
            CEST intensity ratios, shape [n_B1, n_offsets].

        Returns
        -------
        np.ndarray
            Noisy profiles.
        """
        noise = self.rng.normal(0, self.intensity_noise, profiles.shape)
        aug = profiles + noise.astype(np.float32)
        return np.clip(aug, 0, 1).astype(np.float32)
