"""
NMR Signal Preprocessing
=========================

Utilities for processing raw NMR data: FID → spectrum conversion,
apodization, Fourier transform, phase correction, baseline correction,
and peak picking.

Designed to work with nmrglue for format I/O and to produce tensors
suitable for the CNN encoder (Stage 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy import ndimage, signal

logger = logging.getLogger(__name__)


@dataclass
class Peak:
    """Detected NMR peak with properties."""

    position: tuple[float, ...]  # Chemical shift coordinates (ppm)
    index: tuple[int, ...]  # Grid indices in spectrum array
    intensity: float  # Peak height
    volume: float = 0.0  # Integrated volume
    linewidth: tuple[float, ...] = ()  # Linewidth at half-height per dimension (Hz)
    snr: float = 0.0  # Signal-to-noise ratio
    phase: float = 0.0  # Phase of the peak (for phasing quality)


@dataclass
class NMRSpectrum:
    """Container for a processed NMR spectrum."""

    data: np.ndarray  # Spectral data array (real part)
    ndim: int = 1  # Number of dimensions
    sw: tuple[float, ...] = ()  # Spectral widths in Hz
    obs: tuple[float, ...] = ()  # Observe frequencies in MHz
    car: tuple[float, ...] = ()  # Carrier frequencies in Hz
    label: tuple[str, ...] = ()  # Nucleus labels (e.g., '1H', '15N')
    peaks: list[Peak] = field(default_factory=list)

    @property
    def ppm_scales(self) -> list[np.ndarray]:
        """Compute ppm scale for each dimension."""
        scales = []
        for dim in range(self.ndim):
            n = self.data.shape[dim]
            if dim < len(self.sw) and dim < len(self.obs) and self.obs[dim] > 0:
                sw_ppm = self.sw[dim] / self.obs[dim]
                car_ppm = self.car[dim] / self.obs[dim] if dim < len(self.car) else 0.0
                center = car_ppm + sw_ppm / 2.0
                scale = np.linspace(center, center - sw_ppm, n)
            else:
                scale = np.arange(n, dtype=float)
            scales.append(scale)
        return scales


class NMRPreprocessor:
    """NMR data preprocessing pipeline.

    Converts raw FIDs to frequency-domain spectra with apodization,
    Fourier transform, phase correction, and baseline correction.

    Parameters
    ----------
    apodization : str
        Apodization function: 'exponential', 'gaussian', 'sine', 'none'.
    lb : float
        Line broadening factor for exponential apodization (Hz).
    gb : float
        Gaussian broadening factor.
    phase_method : str
        Phase correction method: 'manual', 'auto_acme', 'auto_entropy'.
    baseline_method : str
        Baseline correction method: 'polynomial', 'snip', 'none'.
    baseline_order : int
        Polynomial order for baseline correction.
    zero_fill : int
        Zero-filling factor (1 = no fill, 2 = double, etc.).
    """

    def __init__(
        self,
        apodization: Literal["exponential", "gaussian", "sine", "none"] = "exponential",
        lb: float = 1.0,
        gb: float = 0.0,
        phase_method: Literal["manual", "auto_acme", "auto_entropy"] = "auto_acme",
        baseline_method: Literal["polynomial", "snip", "none"] = "snip",
        baseline_order: int = 3,
        zero_fill: int = 2,
    ):
        self.apodization = apodization
        self.lb = lb
        self.gb = gb
        self.phase_method = phase_method
        self.baseline_method = baseline_method
        self.baseline_order = baseline_order
        self.zero_fill = zero_fill

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_fid(
        self,
        fid: np.ndarray,
        sw: float = 10000.0,
        obs: float = 600.0,
        car: float = 0.0,
    ) -> NMRSpectrum:
        """Process a 1D FID to a frequency-domain spectrum.

        Parameters
        ----------
        fid : np.ndarray
            Complex free induction decay data.
        sw : float
            Spectral width in Hz.
        obs : float
            Observe frequency in MHz.
        car : float
            Carrier frequency in Hz.

        Returns
        -------
        NMRSpectrum
            Processed spectrum.
        """
        # Apodization
        fid_apod = self._apodize_1d(fid, sw)

        # Zero-fill
        if self.zero_fill > 1:
            n_new = len(fid_apod) * self.zero_fill
            fid_zf = np.zeros(n_new, dtype=complex)
            fid_zf[: len(fid_apod)] = fid_apod
        else:
            fid_zf = fid_apod

        # Fourier transform
        spec = np.fft.fftshift(np.fft.fft(fid_zf))

        # Phase correction
        spec = self._auto_phase_1d(spec)

        # Take real part
        spec_real = spec.real.copy()

        # Baseline correction
        spec_real = self._baseline_correct_1d(spec_real)

        return NMRSpectrum(
            data=spec_real,
            ndim=1,
            sw=(sw,),
            obs=(obs,),
            car=(car,),
            label=("1H",),
        )

    def process_fid_2d(
        self,
        fid: np.ndarray,
        sw: tuple[float, float] = (10000.0, 2000.0),
        obs: tuple[float, float] = (600.0, 60.8),
        car: tuple[float, float] = (0.0, 0.0),
        label: tuple[str, str] = ("1H", "15N"),
    ) -> NMRSpectrum:
        """Process a 2D FID (e.g., HSQC) to a 2D frequency-domain spectrum.

        Parameters
        ----------
        fid : np.ndarray
            2D complex FID data, shape [n_t1, n_t2].
        sw : tuple
            Spectral widths in Hz for each dimension.
        obs : tuple
            Observe frequencies in MHz.
        car : tuple
            Carrier frequencies in Hz.
        label : tuple
            Nucleus labels.

        Returns
        -------
        NMRSpectrum
            Processed 2D spectrum.
        """
        n_t1, n_t2 = fid.shape

        # Process direct dimension (t2 → F2)
        for i in range(n_t1):
            fid[i, :] = self._apodize_1d(fid[i, :], sw[1])

        # Zero-fill direct dimension
        if self.zero_fill > 1:
            n_t2_zf = n_t2 * self.zero_fill
            fid_zf = np.zeros((n_t1, n_t2_zf), dtype=complex)
            fid_zf[:, :n_t2] = fid
        else:
            fid_zf = fid
            n_t2_zf = n_t2

        # FFT direct dimension
        for i in range(n_t1):
            fid_zf[i, :] = np.fft.fftshift(np.fft.fft(fid_zf[i, :]))

        # Process indirect dimension (t1 → F1)
        for j in range(n_t2_zf):
            fid_zf[:, j] = self._apodize_1d(fid_zf[:, j], sw[0])

        # Zero-fill indirect dimension
        if self.zero_fill > 1:
            n_t1_zf = n_t1 * self.zero_fill
            fid_zf2 = np.zeros((n_t1_zf, n_t2_zf), dtype=complex)
            fid_zf2[:n_t1, :] = fid_zf
        else:
            fid_zf2 = fid_zf

        # FFT indirect dimension
        for j in range(n_t2_zf):
            fid_zf2[:, j] = np.fft.fftshift(np.fft.fft(fid_zf2[:, j]))

        # Phase correction (simple auto on each dim)
        spec = fid_zf2
        # Auto-phase rows
        ref_row = spec[spec.shape[0] // 2, :]
        p0 = self._estimate_phase_0(ref_row)
        spec = spec * np.exp(1j * p0)
        # Auto-phase columns
        ref_col = spec[:, spec.shape[1] // 2]
        p0_col = self._estimate_phase_0(ref_col)
        spec = spec * np.exp(1j * p0_col)

        spec_real = spec.real.copy()

        # Baseline correction per row
        for i in range(spec_real.shape[0]):
            spec_real[i, :] = self._baseline_correct_1d(spec_real[i, :])

        return NMRSpectrum(
            data=spec_real,
            ndim=2,
            sw=sw,
            obs=obs,
            car=car,
            label=label,
        )

    # ------------------------------------------------------------------
    # Apodization functions
    # ------------------------------------------------------------------

    def _apodize_1d(self, fid: np.ndarray, sw: float) -> np.ndarray:
        """Apply apodization window to a 1D FID.

        Parameters
        ----------
        fid : np.ndarray
            Complex FID.
        sw : float
            Spectral width in Hz.

        Returns
        -------
        np.ndarray
            Apodized FID.
        """
        n = len(fid)
        t = np.arange(n) / sw  # Time points in seconds

        if self.apodization == "exponential":
            # Exponential decay: exp(-π·lb·t)
            window = np.exp(-np.pi * self.lb * t)
        elif self.apodization == "gaussian":
            # Gaussian: exp(-((π·lb·t)² - π·lb·t) / (2·ln2·gb²))
            # Simplified: Gaussian with parameter
            tmax = n / sw
            window = np.exp(-self.lb * np.pi * t) * np.exp(
                -(self.gb * np.pi * t) ** 2 / (4 * np.log(2))
            )
        elif self.apodization == "sine":
            # Sine bell: sin(π·t/tmax)
            window = np.sin(np.pi * np.arange(n) / n)
        else:
            window = np.ones(n)

        return fid * window

    # ------------------------------------------------------------------
    # Phase correction
    # ------------------------------------------------------------------

    def _auto_phase_1d(self, spec: np.ndarray) -> np.ndarray:
        """Automatic phase correction for a 1D spectrum.

        Uses the ACME algorithm (Automated phase Correction based on
        Minimization of Entropy) or simple zero-order correction.

        Parameters
        ----------
        spec : np.ndarray
            Complex spectrum.

        Returns
        -------
        np.ndarray
            Phase-corrected complex spectrum.
        """
        if self.phase_method == "auto_acme":
            return self._acme_phase(spec)
        elif self.phase_method == "auto_entropy":
            return self._entropy_phase(spec)
        else:
            return spec

    def _acme_phase(self, spec: np.ndarray) -> np.ndarray:
        """ACME phase correction — minimize entropy of real spectrum.

        Searches for zero-order (p0) and first-order (p1) phase
        corrections that minimize the entropy of the absolute value
        of the derivative of the real spectrum.
        """
        best_p0 = 0.0
        best_entropy = np.inf

        # Coarse search for p0
        for p0_deg in range(0, 360, 5):
            p0 = np.deg2rad(p0_deg)
            corrected = spec * np.exp(1j * p0)
            entropy = self._spectral_entropy(corrected.real)
            if entropy < best_entropy:
                best_entropy = entropy
                best_p0 = p0

        # Fine search around best
        for p0_deg_fine in np.linspace(
            np.rad2deg(best_p0) - 5, np.rad2deg(best_p0) + 5, 50
        ):
            p0 = np.deg2rad(p0_deg_fine)
            corrected = spec * np.exp(1j * p0)
            entropy = self._spectral_entropy(corrected.real)
            if entropy < best_entropy:
                best_entropy = entropy
                best_p0 = p0

        return spec * np.exp(1j * best_p0)

    def _entropy_phase(self, spec: np.ndarray) -> np.ndarray:
        """Simple entropy-based phase correction (p0 only)."""
        p0 = self._estimate_phase_0(spec)
        return spec * np.exp(1j * p0)

    @staticmethod
    def _estimate_phase_0(spec: np.ndarray) -> float:
        """Estimate zero-order phase by maximizing real integral."""
        best_p0 = 0.0
        best_sum = -np.inf
        for p0_deg in range(0, 360, 2):
            p0 = np.deg2rad(p0_deg)
            real_sum = np.sum((spec * np.exp(1j * p0)).real)
            if real_sum > best_sum:
                best_sum = real_sum
                best_p0 = p0
        return best_p0

    @staticmethod
    def _spectral_entropy(real_spec: np.ndarray) -> float:
        """Compute entropy of the derivative of a real spectrum.

        Used as the objective for ACME phase correction.
        Lower entropy = better phased.
        """
        deriv = np.abs(np.diff(real_spec))
        deriv = deriv / (np.sum(deriv) + 1e-12)
        deriv = deriv[deriv > 0]
        return -np.sum(deriv * np.log(deriv + 1e-12))

    # ------------------------------------------------------------------
    # Baseline correction
    # ------------------------------------------------------------------

    def _baseline_correct_1d(self, spec: np.ndarray) -> np.ndarray:
        """Baseline correction for a 1D real spectrum.

        Parameters
        ----------
        spec : np.ndarray
            Real spectrum data.

        Returns
        -------
        np.ndarray
            Baseline-corrected spectrum.
        """
        if self.baseline_method == "polynomial":
            return self._polynomial_baseline(spec)
        elif self.baseline_method == "snip":
            return self._snip_baseline(spec)
        else:
            return spec

    def _polynomial_baseline(self, spec: np.ndarray) -> np.ndarray:
        """Polynomial baseline correction.

        Fits a polynomial to baseline regions (lowest 30% of points)
        and subtracts it.
        """
        n = len(spec)
        x = np.arange(n, dtype=float)

        # Identify baseline regions (points below 30th percentile)
        threshold = np.percentile(spec, 30)
        mask = spec < threshold

        if np.sum(mask) < self.baseline_order + 1:
            return spec

        # Fit polynomial to baseline points
        coeffs = np.polyfit(x[mask], spec[mask], self.baseline_order)
        baseline = np.polyval(coeffs, x)

        return spec - baseline

    @staticmethod
    def _snip_baseline(spec: np.ndarray, iterations: int = 40) -> np.ndarray:
        """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline.

        Robust baseline estimation that iteratively clips peaks.
        Good for complex baselines with broad features.

        Parameters
        ----------
        spec : np.ndarray
            Input spectrum.
        iterations : int
            Number of SNIP iterations.

        Returns
        -------
        np.ndarray
            Baseline-corrected spectrum.
        """
        n = len(spec)
        # Work in log-sqrt space for better behavior
        working = np.log(np.sqrt(np.maximum(spec, 1e-10)) + 1)

        for i in range(iterations, 0, -1):
            for j in range(i, n - i):
                avg = (working[j - i] + working[j + i]) / 2.0
                working[j] = min(working[j], avg)

        # Convert back
        baseline_est = (np.exp(working) - 1) ** 2
        return spec - baseline_est

    # ------------------------------------------------------------------
    # Peak picking
    # ------------------------------------------------------------------

    def pick_peaks(
        self,
        spectrum: NMRSpectrum,
        threshold: float | None = None,
        min_distance: int = 3,
        max_peaks: int = 500,
    ) -> list[Peak]:
        """Pick peaks from a processed spectrum.

        Uses local maxima detection with noise-based thresholding.

        Parameters
        ----------
        spectrum : NMRSpectrum
            Processed NMR spectrum.
        threshold : float or None
            Peak detection threshold. If None, estimated as 5× noise level.
        min_distance : int
            Minimum distance between peaks (in points).
        max_peaks : int
            Maximum number of peaks to return.

        Returns
        -------
        list[Peak]
            Detected peaks with positions, intensities, and properties.
        """
        data = spectrum.data

        # Estimate noise level from the corners of the spectrum
        if threshold is None:
            noise = self._estimate_noise(data)
            threshold = 5.0 * noise

        if spectrum.ndim == 1:
            peaks = self._pick_peaks_1d(data, threshold, min_distance, spectrum)
        elif spectrum.ndim == 2:
            peaks = self._pick_peaks_2d(data, threshold, min_distance, spectrum)
        else:
            logger.warning("Peak picking not implemented for %dD", spectrum.ndim)
            peaks = []

        # Sort by intensity and limit
        peaks.sort(key=lambda p: -p.intensity)
        peaks = peaks[:max_peaks]

        spectrum.peaks = peaks
        return peaks

    def _pick_peaks_1d(
        self,
        data: np.ndarray,
        threshold: float,
        min_distance: int,
        spectrum: NMRSpectrum,
    ) -> list[Peak]:
        """Pick peaks in a 1D spectrum."""
        # Find local maxima
        indices, properties = signal.find_peaks(
            data,
            height=threshold,
            distance=min_distance,
            prominence=threshold * 0.5,
            width=1,
        )

        noise = self._estimate_noise(data)
        ppm_scale = spectrum.ppm_scales[0] if spectrum.ppm_scales else np.arange(len(data))

        peaks = []
        for i, idx in enumerate(indices):
            # Convert index to ppm
            if idx < len(ppm_scale):
                ppm = float(ppm_scale[idx])
            else:
                ppm = float(idx)

            intensity = float(data[idx])

            # Estimate linewidth at half height
            width_pts = properties.get("widths", np.zeros(len(indices)))
            if i < len(width_pts):
                lw_pts = float(width_pts[i])
                # Convert from points to Hz
                if len(spectrum.sw) > 0:
                    lw_hz = lw_pts * spectrum.sw[0] / len(data)
                else:
                    lw_hz = lw_pts
            else:
                lw_hz = 0.0

            # Estimate volume (integral over peak region)
            left = max(0, idx - int(lw_pts / 2) if i < len(width_pts) else idx - 2)
            right = min(len(data), idx + int(lw_pts / 2) if i < len(width_pts) else idx + 2)
            volume = float(np.sum(data[left:right + 1]))

            peaks.append(Peak(
                position=(ppm,),
                index=(int(idx),),
                intensity=intensity,
                volume=volume,
                linewidth=(lw_hz,),
                snr=intensity / noise if noise > 0 else 0.0,
            ))

        return peaks

    def _pick_peaks_2d(
        self,
        data: np.ndarray,
        threshold: float,
        min_distance: int,
        spectrum: NMRSpectrum,
    ) -> list[Peak]:
        """Pick peaks in a 2D spectrum using local maximum detection."""
        # Find local maxima using maximum filter
        neighborhood = np.ones((min_distance * 2 + 1, min_distance * 2 + 1))
        local_max = ndimage.maximum_filter(data, footprint=neighborhood)
        detected = (data == local_max) & (data > threshold)

        # Get coordinates
        coords = np.argwhere(detected)
        noise = self._estimate_noise(data)

        ppm_scales = spectrum.ppm_scales

        peaks = []
        for coord in coords:
            i, j = int(coord[0]), int(coord[1])
            intensity = float(data[i, j])

            # Convert to ppm
            if len(ppm_scales) >= 2:
                ppm_f1 = float(ppm_scales[0][i]) if i < len(ppm_scales[0]) else float(i)
                ppm_f2 = float(ppm_scales[1][j]) if j < len(ppm_scales[1]) else float(j)
            else:
                ppm_f1, ppm_f2 = float(i), float(j)

            # Estimate linewidths by fitting Lorentzian in each dimension
            lw_f1 = self._estimate_linewidth_1d(data[:, j], i, spectrum.sw[0] if len(spectrum.sw) > 0 else 1.0, data.shape[0])
            lw_f2 = self._estimate_linewidth_1d(data[i, :], j, spectrum.sw[1] if len(spectrum.sw) > 1 else 1.0, data.shape[1])

            # Volume: sum over local region
            r = min_distance
            patch = data[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            volume = float(np.sum(patch))

            peaks.append(Peak(
                position=(ppm_f1, ppm_f2),
                index=(i, j),
                intensity=intensity,
                volume=volume,
                linewidth=(lw_f1, lw_f2),
                snr=intensity / noise if noise > 0 else 0.0,
            ))

        return peaks

    @staticmethod
    def _estimate_linewidth_1d(
        trace: np.ndarray, peak_idx: int, sw: float, n: int
    ) -> float:
        """Estimate linewidth at half height for a 1D trace at a peak position."""
        half_height = trace[peak_idx] / 2.0

        # Search left
        left = peak_idx
        while left > 0 and trace[left] > half_height:
            left -= 1

        # Search right
        right = peak_idx
        while right < len(trace) - 1 and trace[right] > half_height:
            right += 1

        width_pts = right - left
        return width_pts * sw / n

    @staticmethod
    def _estimate_noise(data: np.ndarray) -> float:
        """Estimate noise level from spectral data.

        Uses the MAD (Median Absolute Deviation) estimator on the
        corners/edges of the spectrum, which are assumed to be noise.

        Parameters
        ----------
        data : np.ndarray
            Spectral data (1D or 2D).

        Returns
        -------
        float
            Estimated noise standard deviation.
        """
        if data.ndim == 1:
            # Use first and last 10% of spectrum as noise regions
            n = len(data)
            edge = max(int(n * 0.1), 10)
            noise_data = np.concatenate([data[:edge], data[-edge:]])
        elif data.ndim == 2:
            # Use corners of the 2D spectrum
            n1, n2 = data.shape
            e1 = max(int(n1 * 0.1), 5)
            e2 = max(int(n2 * 0.1), 5)
            noise_data = np.concatenate([
                data[:e1, :e2].ravel(),
                data[:e1, -e2:].ravel(),
                data[-e1:, :e2].ravel(),
                data[-e1:, -e2:].ravel(),
            ])
        else:
            noise_data = data.ravel()[:100]

        # MAD estimator (robust to outliers)
        mad = np.median(np.abs(noise_data - np.median(noise_data)))
        return float(mad * 1.4826)  # Scale factor for Gaussian
