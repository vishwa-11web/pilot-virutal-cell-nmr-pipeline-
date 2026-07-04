"""
Synthetic NMR Spectra Generator
================================

Generates synthetic NMR spectra from known molecular structures and
chemical shift assignments for training data bootstrapping.

Supports:
  - 2D HSQC spectra as Lorentzian peak fields
  - NOESY cross-peaks from inter-proton distances
  - CEST profiles from two-state exchange models
  - Relaxation rate profiles (R1, R2) from model-free spectral density
  - Realistic noise, artifacts, and t1 ridges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KB = 1.380649e-23  # Boltzmann constant (J/K)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)
R_GAS = 8.314462618  # Gas constant (J/(mol·K))
MU0 = 1.2566370621e-6  # Vacuum permeability (T·m/A)

# Gyromagnetic ratios (rad/(s·T))
GAMMA = {
    "1H": 267.522e6,
    "13C": 67.2828e6,
    "15N": -27.116e6,
    "31P": 108.394e6,
}


@dataclass
class ExchangeParameters:
    """Parameters for a two-state conformational exchange model."""

    kex: float = 500.0  # Exchange rate constant (s⁻¹)
    pb: float = 0.05  # Minor state population
    dw: float = 3.0  # Chemical shift difference (ppm)
    R1_A: float = 1.5  # R1 of major state (s⁻¹)
    R2_A: float = 10.0  # R2 of major state (s⁻¹)
    R1_B: float = 1.5  # R1 of minor state (s⁻¹)
    R2_B: float = 100.0  # R2 of minor state (s⁻¹)

    @property
    def pa(self) -> float:
        return 1.0 - self.pb

    @property
    def k_AB(self) -> float:
        """Forward rate constant A → B."""
        return self.kex * self.pb

    @property
    def k_BA(self) -> float:
        """Reverse rate constant B → A."""
        return self.kex * self.pa

    @property
    def delta_G(self) -> float:
        """Free energy difference between states (J/mol) at 298K."""
        if self.pb > 0 and self.pa > 0:
            return -R_GAS * 298.15 * np.log(self.pb / self.pa)
        return float("inf")


@dataclass
class ModelFreeParams:
    """Lipari-Szabo model-free parameters for dynamics."""

    tau_c: float = 5e-9  # Overall correlation time (s)
    S2: float = 0.85  # Order parameter (0 = fully flexible, 1 = rigid)
    tau_e: float = 50e-12  # Internal correlation time (s)
    rex: float = 0.0  # Exchange contribution to R2 (s⁻¹)


class SyntheticSpectraGenerator:
    """Generate synthetic NMR spectra from molecular parameters.

    Produces training data with known ground truth for supervised
    learning of the NMR-to-physics pipeline.

    Parameters
    ----------
    field_strength : float
        Spectrometer field strength in MHz (proton frequency).
    random_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        field_strength: float = 600.0,
        random_seed: int | None = 42,
    ):
        self.field_strength = field_strength  # MHz
        self.B0 = field_strength * 2 * np.pi / (GAMMA["1H"] * 1e-6)  # Tesla
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # 2D HSQC Generation
    # ------------------------------------------------------------------

    def generate_hsqc(
        self,
        shifts_H: np.ndarray,
        shifts_N: np.ndarray,
        grid_size: tuple[int, int] = (256, 512),
        sw_H: float = 12.0,
        sw_N: float = 35.0,
        center_H: float = 8.3,
        center_N: float = 120.0,
        linewidth_H: float = 20.0,
        linewidth_N: float = 15.0,
        noise_level: float = 0.02,
        intensities: np.ndarray | None = None,
    ) -> dict:
        """Generate a synthetic 2D ¹H-¹⁵N HSQC spectrum.

        Parameters
        ----------
        shifts_H : np.ndarray
            ¹H chemical shifts in ppm, shape [N_peaks].
        shifts_N : np.ndarray
            ¹⁵N chemical shifts in ppm, shape [N_peaks].
        grid_size : tuple
            Spectrum grid dimensions (F1=¹⁵N, F2=¹H).
        sw_H, sw_N : float
            Spectral widths in ppm.
        center_H, center_N : float
            Carrier positions in ppm.
        linewidth_H, linewidth_N : float
            Linewidths in Hz.
        noise_level : float
            Noise standard deviation relative to max peak intensity.
        intensities : np.ndarray or None
            Peak intensities. If None, all peaks have intensity 1.0.

        Returns
        -------
        dict with keys:
            spectrum: np.ndarray [n_F1, n_F2] — 2D spectrum
            peak_positions: np.ndarray [N, 2] — (¹⁵N_ppm, ¹H_ppm)
            peak_indices: np.ndarray [N, 2] — grid indices
            ppm_H: np.ndarray — ¹H ppm scale
            ppm_N: np.ndarray — ¹⁵N ppm scale
            linewidths: np.ndarray [N, 2] — linewidths (Hz)
            intensities: np.ndarray [N] — peak intensities
        """
        n_f1, n_f2 = grid_size
        n_peaks = len(shifts_H)

        if intensities is None:
            intensities = np.ones(n_peaks)

        # Create ppm scales (decreasing, as is convention)
        ppm_H = np.linspace(center_H + sw_H / 2, center_H - sw_H / 2, n_f2)
        ppm_N = np.linspace(center_N + sw_N / 2, center_N - sw_N / 2, n_f1)

        # Convert linewidths from Hz to ppm
        lw_H_ppm = linewidth_H / self.field_strength
        lw_N_ppm = linewidth_N / (self.field_strength * GAMMA["15N"] / GAMMA["1H"])

        # Generate spectrum as sum of 2D Lorentzians
        spectrum = np.zeros((n_f1, n_f2), dtype=np.float64)

        # Add per-peak linewidth variation
        lw_H_array = np.abs(self.rng.normal(linewidth_H, linewidth_H * 0.2, n_peaks))
        lw_N_array = np.abs(self.rng.normal(linewidth_N, linewidth_N * 0.2, n_peaks))

        peak_indices = np.zeros((n_peaks, 2), dtype=int)

        for k in range(n_peaks):
            # Lorentzian in each dimension
            delta_H = ppm_H - shifts_H[k]
            gamma_H = lw_H_array[k] / (2 * self.field_strength)  # Half-width in ppm
            lorentz_H = gamma_H / (delta_H ** 2 + gamma_H ** 2)

            delta_N = ppm_N - shifts_N[k]
            gamma_N = lw_N_array[k] / (2 * self.field_strength * abs(GAMMA["15N"]) / GAMMA["1H"])
            lorentz_N = gamma_N / (delta_N ** 2 + gamma_N ** 2)

            # 2D peak = outer product
            peak_2d = intensities[k] * np.outer(lorentz_N, lorentz_H)
            spectrum += peak_2d

            # Store peak index (closest grid point)
            peak_indices[k, 0] = np.argmin(np.abs(ppm_N - shifts_N[k]))
            peak_indices[k, 1] = np.argmin(np.abs(ppm_H - shifts_H[k]))

        # Normalize
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        # Add noise
        noise = self.rng.normal(0, noise_level, spectrum.shape)
        spectrum += noise

        # Add t1 noise ridges (artifacts along F1 at strongest peak F2 positions)
        if n_peaks > 0:
            strongest = np.argmax(intensities)
            ridge_idx = peak_indices[strongest, 1]
            ridge = self.rng.normal(0, noise_level * 0.3, n_f1)
            if 0 <= ridge_idx < n_f2:
                spectrum[:, ridge_idx] += ridge

        return {
            "spectrum": spectrum.astype(np.float32),
            "peak_positions": np.column_stack([shifts_N, shifts_H]),
            "peak_indices": peak_indices,
            "ppm_H": ppm_H.astype(np.float32),
            "ppm_N": ppm_N.astype(np.float32),
            "linewidths": np.column_stack([lw_N_array, lw_H_array]).astype(np.float32),
            "intensities": intensities.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # NOESY Spectrum Generation
    # ------------------------------------------------------------------

    def generate_noesy(
        self,
        shifts_H: np.ndarray,
        coordinates: np.ndarray,
        grid_size: tuple[int, int] = (512, 512),
        sw: float = 12.0,
        center: float = 4.7,
        mixing_time: float = 0.100,
        distance_cutoff: float = 5.5,
        noise_level: float = 0.03,
    ) -> dict:
        """Generate a synthetic 2D NOESY spectrum.

        Cross-peak intensity ∝ r⁻⁶ for inter-proton distance r.

        Parameters
        ----------
        shifts_H : np.ndarray
            ¹H chemical shifts in ppm, shape [N_protons].
        coordinates : np.ndarray
            3D coordinates of protons, shape [N_protons, 3] in Å.
        grid_size : tuple
            Spectrum dimensions.
        sw : float
            Spectral width in ppm.
        center : float
            Carrier position in ppm.
        mixing_time : float
            NOE mixing time in seconds.
        distance_cutoff : float
            Maximum inter-proton distance for NOE (Å).
        noise_level : float
            Noise level relative to max.

        Returns
        -------
        dict with keys:
            spectrum: 2D NOESY spectrum
            cross_peaks: list of (i, j, distance, intensity) tuples
            ppm: ppm scale
        """
        n_f1, n_f2 = grid_size
        n_protons = len(shifts_H)

        ppm = np.linspace(center + sw / 2, center - sw / 2, n_f1)
        spectrum = np.zeros((n_f1, n_f2), dtype=np.float64)

        # Compute distance matrix
        diff = coordinates[:, None, :] - coordinates[None, :, :]  # [N, N, 3]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, N]

        # Generate cross-peaks
        cross_peaks = []
        lw_ppm = 30.0 / self.field_strength  # Typical linewidth

        for i in range(n_protons):
            for j in range(i + 1, n_protons):
                r = distances[i, j]
                if r < distance_cutoff and r > 0.5:
                    # NOE intensity ~ r^-6 * mixing_time (initial rate approximation)
                    intensity = mixing_time * r ** (-6)
                    cross_peaks.append((i, j, float(r), float(intensity)))

                    # Add symmetric cross-peaks
                    for (si, sj) in [(i, j), (j, i)]:
                        delta_f1 = ppm - shifts_H[si]
                        delta_f2 = ppm - shifts_H[sj]
                        lorentz_f1 = lw_ppm / (delta_f1 ** 2 + lw_ppm ** 2)
                        lorentz_f2 = lw_ppm / (delta_f2 ** 2 + lw_ppm ** 2)
                        spectrum += intensity * np.outer(lorentz_f1, lorentz_f2)

        # Add diagonal peaks
        for i in range(n_protons):
            delta = ppm - shifts_H[i]
            lorentz = lw_ppm / (delta ** 2 + lw_ppm ** 2)
            spectrum += 10.0 * np.outer(lorentz, lorentz)  # Diagonal is strong

        # Normalize and add noise
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        spectrum += self.rng.normal(0, noise_level, spectrum.shape)

        return {
            "spectrum": spectrum.astype(np.float32),
            "cross_peaks": cross_peaks,
            "distances": distances.astype(np.float32),
            "ppm": ppm.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # CEST Profile Generation
    # ------------------------------------------------------------------

    def generate_cest_profile(
        self,
        exchange_params: ExchangeParameters,
        B1_fields: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        sat_time: float = 0.5,
        B0_MHz: float | None = None,
    ) -> dict:
        """Generate a CEST (Chemical Exchange Saturation Transfer) profile.

        Uses the Bloch-McConnell equations for two-state exchange to
        simulate the CEST intensity profile as a function of saturation
        offset frequency.

        Parameters
        ----------
        exchange_params : ExchangeParameters
            Two-state exchange parameters.
        B1_fields : np.ndarray or None
            Saturation B1 field strengths in Hz. Default: [25, 50, 100, 200].
        offsets : np.ndarray or None
            Saturation offsets in ppm. Default: -10 to 10 ppm.
        sat_time : float
            Saturation time in seconds.
        B0_MHz : float or None
            Field strength in MHz. Default: self.field_strength.

        Returns
        -------
        dict with keys:
            profiles: np.ndarray [n_B1, n_offsets] — CEST intensity ratio I/I0
            B1_fields: B1 field values used
            offsets: offset values used
            exchange_params: input parameters
            kex: exchange rate
            delta_G: free energy difference
        """
        if B0_MHz is None:
            B0_MHz = self.field_strength
        if B1_fields is None:
            B1_fields = np.array([25.0, 50.0, 100.0, 200.0])
        if offsets is None:
            offsets = np.linspace(-10.0, 10.0, 201)

        ep = exchange_params
        profiles = np.zeros((len(B1_fields), len(offsets)))

        for b, B1 in enumerate(B1_fields):
            for o, offset in enumerate(offsets):
                # Bloch-McConnell for two-state CEST
                # Compute steady-state Mz/M0 under saturation

                # Convert offset from ppm to rad/s
                omega_offset = offset * B0_MHz * 2 * np.pi  # Hz → rad/s approximation
                omega_A = 0.0  # Major state is on-resonance at 0
                omega_B = ep.dw * B0_MHz * 2 * np.pi  # Minor state offset

                # Effective offset from saturation for each state
                delta_A = omega_offset - omega_A
                delta_B = omega_offset - omega_B

                # B1 in rad/s
                omega1 = B1 * 2 * np.pi

                # Effective R1rho for each state (simplified)
                R1rho_A = ep.R1_A * np.cos(np.arctan2(omega1, delta_A)) ** 2 + \
                          ep.R2_A * np.sin(np.arctan2(omega1, delta_A)) ** 2
                R1rho_B = ep.R1_B * np.cos(np.arctan2(omega1, delta_B)) ** 2 + \
                          ep.R2_B * np.sin(np.arctan2(omega1, delta_B)) ** 2

                # Rex contribution (Trott-Palmer approximation)
                if abs(delta_B) > 1e-6 or omega1 > 1e-6:
                    denom = delta_B ** 2 + omega1 ** 2 + ep.k_BA ** 2
                    Rex = ep.pa * ep.pb * ep.dw_rad(B0_MHz) ** 2 * ep.kex / max(denom, 1e-10)
                else:
                    Rex = 0.0

                # Effective R2 under saturation
                R_eff = R1rho_A + Rex

                # Steady-state saturation
                # I/I0 = R1 / (R1 + R_sat)
                theta_A = np.arctan2(omega1, delta_A)
                R_sat = (ep.R2_A + Rex) * np.sin(theta_A) ** 2 + ep.R1_A * np.cos(theta_A) ** 2

                # Time-dependent approach to steady state
                I_ratio = 1.0 - (1.0 - ep.R1_A / max(R_sat + ep.R1_A, 1e-10)) * \
                          (1.0 - np.exp(-R_sat * sat_time))

                profiles[b, o] = max(0.0, min(1.0, I_ratio))

        return {
            "profiles": profiles.astype(np.float32),
            "B1_fields": B1_fields.astype(np.float32),
            "offsets": offsets.astype(np.float32),
            "exchange_params": {
                "kex": ep.kex,
                "pb": ep.pb,
                "dw": ep.dw,
                "delta_G": ep.delta_G,
            },
        }

    # ------------------------------------------------------------------
    # Relaxation Rate Generation
    # ------------------------------------------------------------------

    def generate_relaxation_rates(
        self,
        model_free_params: list[ModelFreeParams],
        field_strengths_MHz: np.ndarray | None = None,
        nucleus: str = "15N",
        bond_length: float = 1.02e-10,
        csa: float = -170e-6,
    ) -> dict:
        """Generate R1, R2, and heteronuclear NOE from model-free parameters.

        Uses the Lipari-Szabo model-free formalism with spectral density
        functions to compute relaxation rates.

        Parameters
        ----------
        model_free_params : list[ModelFreeParams]
            Per-residue dynamics parameters.
        field_strengths_MHz : np.ndarray or None
            B0 field strengths in MHz. Default: [600].
        nucleus : str
            Heteronucleus ('15N' or '13C').
        bond_length : float
            X-H bond length in meters.
        csa : float
            Chemical shift anisotropy (dimensionless).

        Returns
        -------
        dict with keys:
            R1: np.ndarray [N_residues, N_fields]
            R2: np.ndarray [N_residues, N_fields]
            hetNOE: np.ndarray [N_residues, N_fields]
            model_free_params: input parameters
        """
        if field_strengths_MHz is None:
            field_strengths_MHz = np.array([600.0])

        n_res = len(model_free_params)
        n_fields = len(field_strengths_MHz)

        R1 = np.zeros((n_res, n_fields))
        R2 = np.zeros((n_res, n_fields))
        hetNOE = np.zeros((n_res, n_fields))

        gamma_H = GAMMA["1H"]
        gamma_X = abs(GAMMA.get(nucleus, GAMMA["15N"]))

        for f, B0_MHz in enumerate(field_strengths_MHz):
            omega_H = gamma_H * B0_MHz * 1e-6 * 2 * np.pi  # Simplified
            omega_X = gamma_X * B0_MHz * 1e-6 * 2 * np.pi

            # Actually, B0 in Tesla:
            B0 = B0_MHz * 2 * np.pi / (gamma_H * 1e-6)
            omega_H = gamma_H * B0
            omega_X = gamma_X * B0

            # Dipolar coupling constant
            d = (MU0 * HBAR * gamma_H * gamma_X) / (4 * np.pi * bond_length ** 3)
            d2 = d ** 2

            # CSA contribution
            c2 = (omega_X * csa) ** 2 / 3.0

            for i, mf in enumerate(model_free_params):
                # Spectral density function (model-free)
                def J(omega):
                    """Lipari-Szabo spectral density."""
                    tau_c = mf.tau_c
                    S2 = mf.S2
                    tau_e = mf.tau_e

                    # Effective correlation time for internal motion
                    tau_eff = tau_c * tau_e / (tau_c + tau_e) if (tau_c + tau_e) > 0 else 0

                    return (2.0 / 5.0) * (
                        S2 * tau_c / (1.0 + (omega * tau_c) ** 2)
                        + (1.0 - S2) * tau_eff / (1.0 + (omega * tau_eff) ** 2)
                    )

                # R1
                R1[i, f] = (d2 / 4.0) * (
                    J(omega_H - omega_X) + 3 * J(omega_X) + 6 * J(omega_H + omega_X)
                ) + c2 * J(omega_X)

                # R2
                R2[i, f] = (d2 / 8.0) * (
                    4 * J(0) + J(omega_H - omega_X) + 3 * J(omega_X)
                    + 6 * J(omega_H) + 6 * J(omega_H + omega_X)
                ) + (c2 / 6.0) * (4 * J(0) + 3 * J(omega_X)) + mf.rex

                # Heteronuclear NOE
                sigma_NH = (d2 / 4.0) * (
                    6 * J(omega_H + omega_X) - J(omega_H - omega_X)
                )
                hetNOE[i, f] = 1.0 + (gamma_H / gamma_X) * sigma_NH / max(R1[i, f], 1e-10)

        return {
            "R1": R1.astype(np.float32),
            "R2": R2.astype(np.float32),
            "hetNOE": hetNOE.astype(np.float32),
            "field_strengths_MHz": field_strengths_MHz.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # J-Coupling Generation
    # ------------------------------------------------------------------

    def generate_j_couplings(
        self,
        phi_angles: np.ndarray,
        psi_angles: np.ndarray,
    ) -> dict:
        """Generate backbone J-coupling constants from dihedral angles.

        Uses the Karplus equation: ³J(HNHα) = A·cos²(φ-60°) + B·cos(φ-60°) + C

        Parameters
        ----------
        phi_angles : np.ndarray
            Backbone φ angles in degrees, shape [N_residues].
        psi_angles : np.ndarray
            Backbone ψ angles in degrees, shape [N_residues].

        Returns
        -------
        dict with keys:
            J_HNHa: ³J(HN-Hα) couplings in Hz
            J_HaC: ³J(Hα-C') couplings in Hz
            phi: input φ angles
            psi: input ψ angles
        """
        # Karplus parameters for ³J(HN-Hα) (Vuister & Bax, 1993)
        A, B, C = 6.51, -1.76, 1.60

        phi_rad = np.deg2rad(phi_angles)
        theta = phi_rad - np.deg2rad(60.0)  # Karplus angle

        J_HNHa = A * np.cos(theta) ** 2 + B * np.cos(theta) + C

        # Add small random variation to simulate measurement uncertainty
        J_HNHa += self.rng.normal(0, 0.3, len(phi_angles))

        # ³J(Hα-C') from ψ angle (Wirmer & Schwalbe, 2002)
        A2, B2, C2 = 3.72, -2.18, 1.28
        psi_rad = np.deg2rad(psi_angles)
        theta2 = psi_rad + np.deg2rad(120.0)
        J_HaC = A2 * np.cos(theta2) ** 2 + B2 * np.cos(theta2) + C2
        J_HaC += self.rng.normal(0, 0.2, len(psi_angles))

        return {
            "J_HNHa": J_HNHa.astype(np.float32),
            "J_HaC": J_HaC.astype(np.float32),
            "phi": phi_angles.astype(np.float32),
            "psi": psi_angles.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Complete molecule data generation
    # ------------------------------------------------------------------

    def generate_molecule_data(
        self,
        sequence: str,
        shifts_H: np.ndarray | None = None,
        shifts_N: np.ndarray | None = None,
        shifts_CA: np.ndarray | None = None,
        shifts_CB: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        phi_angles: np.ndarray | None = None,
        psi_angles: np.ndarray | None = None,
        noise_level: float = 0.02,
        include_exchange: bool = True,
    ) -> dict:
        """Generate a complete synthetic NMR dataset for a molecule.

        If shifts/coordinates are not provided, generates random
        physically plausible values based on the sequence.

        Parameters
        ----------
        sequence : str
            One-letter amino acid sequence.
        shifts_H, shifts_N, shifts_CA, shifts_CB : np.ndarray or None
            Chemical shifts. Generated randomly if None.
        coordinates : np.ndarray or None
            3D coordinates. Generated as random coil if None.
        phi_angles, psi_angles : np.ndarray or None
            Dihedral angles. Generated from Ramachandran if None.
        noise_level : float
            Noise level for spectra.
        include_exchange : bool
            Whether to generate CEST data for exchanging residues.

        Returns
        -------
        dict
            Complete dataset with spectra, assignments, and physics targets.
        """
        from .bmrb_client import AA_1TO3, AA_SHIFT_STATS

        n_res = len(sequence)

        # Generate chemical shifts from statistical distributions if not provided
        if shifts_H is None:
            shifts_H = np.zeros(n_res)
            for i, aa in enumerate(sequence):
                aa3 = AA_1TO3.get(aa, "ALA")
                stats = AA_SHIFT_STATS.get(aa3, {"H": (8.2, 0.6)})
                mu, sigma = stats.get("H", (8.2, 0.6))
                if mu > 0:
                    shifts_H[i] = self.rng.normal(mu, sigma)
                else:
                    shifts_H[i] = 0.0  # Proline

        if shifts_N is None:
            shifts_N = np.zeros(n_res)
            for i, aa in enumerate(sequence):
                aa3 = AA_1TO3.get(aa, "ALA")
                stats = AA_SHIFT_STATS.get(aa3, {"N": (120.0, 4.0)})
                mu, sigma = stats.get("N", (120.0, 4.0))
                shifts_N[i] = self.rng.normal(mu, sigma)

        if shifts_CA is None:
            shifts_CA = np.zeros(n_res)
            for i, aa in enumerate(sequence):
                aa3 = AA_1TO3.get(aa, "ALA")
                stats = AA_SHIFT_STATS.get(aa3, {"CA": (55.0, 2.5)})
                mu, sigma = stats.get("CA", (55.0, 2.5))
                shifts_CA[i] = self.rng.normal(mu, sigma)

        if shifts_CB is None:
            shifts_CB = np.zeros(n_res)
            for i, aa in enumerate(sequence):
                aa3 = AA_1TO3.get(aa, "ALA")
                stats = AA_SHIFT_STATS.get(aa3, {"CB": (35.0, 5.0)})
                mu, sigma = stats.get("CB", (35.0, 5.0))
                if mu > 0:
                    shifts_CB[i] = self.rng.normal(mu, sigma)

        # Generate backbone dihedral angles from Ramachandran distribution
        if phi_angles is None or psi_angles is None:
            phi_angles, psi_angles = self._generate_ramachandran_angles(sequence)

        # Generate random coil coordinates if not provided
        if coordinates is None:
            coordinates = self._generate_coil_coordinates(n_res, phi_angles, psi_angles)

        # Filter out proline (no amide H)
        has_amide = np.array([aa != "P" for aa in sequence])
        hsqc_H = shifts_H[has_amide]
        hsqc_N = shifts_N[has_amide]

        # Generate spectra
        result = {
            "sequence": sequence,
            "n_residues": n_res,
        }

        # HSQC
        if len(hsqc_H) > 0:
            result["hsqc"] = self.generate_hsqc(
                hsqc_H, hsqc_N,
                noise_level=noise_level,
            )

        # Chemical shifts (ground truth)
        result["shifts"] = {
            "H": shifts_H.astype(np.float32),
            "N": shifts_N.astype(np.float32),
            "CA": shifts_CA.astype(np.float32),
            "CB": shifts_CB.astype(np.float32),
        }

        # J-couplings
        result["j_couplings"] = self.generate_j_couplings(phi_angles, psi_angles)

        # NOESY (if we have coordinates)
        # Use Hα and HN proton positions (simplified)
        result["noesy"] = self.generate_noesy(
            shifts_H, coordinates[:n_res],  # Simplified: one H per residue
            noise_level=noise_level,
        )

        # Relaxation rates
        mf_params = []
        for i in range(n_res):
            # Generate physically plausible model-free parameters
            tau_c = self.rng.uniform(3e-9, 12e-9)  # Depends on MW
            S2 = self.rng.uniform(0.6, 0.95)
            tau_e = self.rng.uniform(10e-12, 200e-12)
            rex = self.rng.exponential(0.5)  # Most residues: small Rex
            mf_params.append(ModelFreeParams(tau_c=tau_c, S2=S2, tau_e=tau_e, rex=rex))

        result["relaxation"] = self.generate_relaxation_rates(mf_params)
        result["model_free_params"] = [
            {"tau_c": m.tau_c, "S2": m.S2, "tau_e": m.tau_e, "rex": m.rex}
            for m in mf_params
        ]

        # CEST profiles for exchanging residues
        if include_exchange:
            n_exchange = max(1, n_res // 10)  # ~10% of residues exchange
            exchange_residues = self.rng.choice(n_res, n_exchange, replace=False)
            cest_profiles = {}
            for idx in exchange_residues:
                ep = ExchangeParameters(
                    kex=self.rng.uniform(100, 5000),
                    pb=self.rng.uniform(0.01, 0.15),
                    dw=self.rng.uniform(1.0, 5.0),
                )
                cest_profiles[int(idx)] = self.generate_cest_profile(ep)
            result["cest"] = cest_profiles

        # Coordinates
        result["coordinates"] = coordinates.astype(np.float32)
        result["phi_angles"] = phi_angles.astype(np.float32)
        result["psi_angles"] = psi_angles.astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _generate_ramachandran_angles(
        self, sequence: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate backbone dihedral angles from Ramachandran distribution.

        Samples from simplified Ramachandran regions:
        - α-helix: φ ≈ -57°, ψ ≈ -47°
        - β-sheet: φ ≈ -120°, ψ ≈ 120°
        - Coil: broad distribution

        Parameters
        ----------
        sequence : str
            Amino acid sequence.

        Returns
        -------
        tuple of np.ndarray
            (phi, psi) arrays in degrees.
        """
        n = len(sequence)
        phi = np.zeros(n)
        psi = np.zeros(n)

        for i in range(n):
            aa = sequence[i]

            if aa == "P":
                # Proline has restricted φ
                phi[i] = self.rng.normal(-63, 8)
                psi[i] = self.rng.choice([
                    self.rng.normal(150, 15),  # PPII
                    self.rng.normal(-30, 15),  # α
                ])
            elif aa == "G":
                # Glycine has expanded Ramachandran
                region = self.rng.choice(["alpha", "beta", "alphaL", "coil"], p=[0.3, 0.2, 0.15, 0.35])
                if region == "alpha":
                    phi[i] = self.rng.normal(-60, 15)
                    psi[i] = self.rng.normal(-45, 15)
                elif region == "beta":
                    phi[i] = self.rng.normal(-120, 20)
                    psi[i] = self.rng.normal(120, 20)
                elif region == "alphaL":
                    phi[i] = self.rng.normal(60, 15)
                    psi[i] = self.rng.normal(45, 15)
                else:
                    phi[i] = self.rng.uniform(-180, 180)
                    psi[i] = self.rng.uniform(-180, 180)
            else:
                # General amino acid
                region = self.rng.choice(["alpha", "beta", "coil"], p=[0.4, 0.35, 0.25])
                if region == "alpha":
                    phi[i] = self.rng.normal(-63, 8)
                    psi[i] = self.rng.normal(-42, 8)
                elif region == "beta":
                    phi[i] = self.rng.normal(-120, 12)
                    psi[i] = self.rng.normal(130, 12)
                else:
                    phi[i] = self.rng.normal(-80, 30)
                    psi[i] = self.rng.normal(0, 60)

        return phi, psi

    def _generate_coil_coordinates(
        self,
        n_residues: int,
        phi: np.ndarray,
        psi: np.ndarray,
    ) -> np.ndarray:
        """Generate approximate backbone Cα coordinates from dihedral angles.

        Uses a simplified backbone model with fixed bond lengths and angles,
        rotating by φ/ψ at each residue.

        Parameters
        ----------
        n_residues : int
            Number of residues.
        phi, psi : np.ndarray
            Backbone dihedral angles in degrees.

        Returns
        -------
        np.ndarray
            Cα coordinates, shape [n_residues, 3].
        """
        CA_CA_DIST = 3.8  # Approximate Cα-Cα distance (Å)

        coords = np.zeros((n_residues, 3))
        direction = np.array([1.0, 0.0, 0.0])

        for i in range(1, n_residues):
            # Simple model: rotate direction by psi and phi
            phi_rad = np.deg2rad(phi[i])
            psi_rad = np.deg2rad(psi[i - 1]) if i > 0 else 0

            # Rotation around z-axis by phi
            cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)
            rot_phi = np.array([
                [cos_p, -sin_p, 0],
                [sin_p, cos_p, 0],
                [0, 0, 1],
            ])

            # Rotation around y-axis by psi
            cos_s, sin_s = np.cos(psi_rad), np.sin(psi_rad)
            rot_psi = np.array([
                [cos_s, 0, sin_s],
                [0, 1, 0],
                [-sin_s, 0, cos_s],
            ])

            direction = rot_phi @ rot_psi @ direction
            direction = direction / np.linalg.norm(direction)
            coords[i] = coords[i - 1] + CA_CA_DIST * direction

        return coords


# Add dw_rad method to ExchangeParameters
def _dw_rad(self, B0_MHz: float) -> float:
    """Chemical shift difference in rad/s."""
    return self.dw * B0_MHz * 2 * np.pi


ExchangeParameters.dw_rad = _dw_rad
