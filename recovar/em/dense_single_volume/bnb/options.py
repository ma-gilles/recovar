"""Configuration for cryoSPARC-style branch-and-bound pose refinement (K=1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BranchBoundOptions:
    """Knobs for the cryoSPARC-faithful BnB pose refinement engine.

    Defaults match the schedule/caps from Punjani et al. 2017 Suppl Note 2:
    initial 24 deg / 5 px / L=12, doubled per stage, 7 subdivisions
    (8 evaluation passes total) to ~0.18 deg / ~0.04 px, with cryoSPARC's
    12.5%/25% pathological-image caps and the 4 sigma probabilistic bound.
    """

    enabled: bool = False

    # Schedule (Suppl §"Subdivision scheme")
    n_subdivisions: int = 7
    """Number of orientation/shift subdivisions; 8 evaluation passes total
    (initial coarse grid + n_subdivisions). Final precision = initial /
    2**n_subdivisions; 24 deg / 128 = 0.1875 deg matches paper."""

    initial_angular_spacing_deg: float = 24.0
    initial_shift_spacing_px: float = 5.0
    initial_fourier_radius: int = 12
    fourier_radius_growth: float = 2.0

    # Bound mode (see ``bounds.py``)
    bound_mode: Literal["cryosparc_prob", "deterministic_cauchy", "hybrid"] = "cryosparc_prob"
    tau_sigma: float = 4.0
    """Number of standard deviations for the probabilistic noise bound.
    cryoSPARC default is 4 (probability 0.999936)."""

    ctf_bound_mode: Literal["exact", "cryosparc_rms", "hybrid"] = "exact"
    """``exact`` uses per-image CTF/sigma in the high-frequency power bound
    (paper-validation default). ``cryosparc_rms`` uses the RMS-CTF
    approximation (|C|^2 -> 1/2) so the bound is shared across images of the
    same noise group; speed optimisation, may be looser. Phase 6+ only."""

    rms_ctf_squared: float = 0.5
    """When ``ctf_bound_mode='cryosparc_rms'``, this is the constant used to
    replace |C_l|^2 in the bound. cryoSPARC default 1/2 corresponds to a
    uniform-phase oscillating CTF; lower values make the bound looser, higher
    values can be unsafe."""

    # EM-correct pruning
    posterior_tail_tol: float = 1e-6
    """Per-image upper bound on omitted posterior mass after pruning. Used to
    derive the score margin tau = -log(posterior_tail_tol)."""

    score_margin: float | None = None
    """If set, overrides the score margin derived from ``posterior_tail_tol``."""

    min_joint_candidates_per_image: int = 128
    """Floor on retained (rotation, shift) candidates per image after pruning,
    even if the tail bound says fewer would suffice."""

    max_joint_candidates_per_image: int = 100_000
    """Ceiling on retained (rotation, shift) candidates per image. Beyond this
    the engine optionally falls back to dense/local search for that image."""

    # cryoSPARC pathological-image caps (Suppl §"Approximations")
    max_orientation_fraction: float = 0.125
    """Hard cap on retained orientations as a fraction of the current stage
    grid. cryoSPARC: 12.5%."""

    max_shift_fraction: float = 0.25
    """Hard cap on retained shifts as a fraction of the current stage grid.
    cryoSPARC: 25%."""

    min_orientations_per_image: int = 16
    """Minimum retained orientations per image, regardless of caps; prevents
    over-pruning on noise-only particles."""

    min_shifts_per_image: int = 8
    """Minimum retained shifts per image, regardless of caps."""

    # Fallbacks
    fallback_strategy: Literal["dense", "local"] = "local"
    """Which existing engine to fall back to when BnB cannot meet the tail
    bound or the survivor count exceeds ``max_joint_candidates_per_image``."""

    fallback_if_too_many_survivors: bool = True
    fallback_if_bound_diagnostic_fails: bool = True

    # Diagnostics / debug
    return_diagnostics: bool = True
    verify_bound_sample_size: int = 0
    """If >0, randomly sample this many poses per image at each stage and
    cross-check that the deterministic Cauchy bound holds. Test/debug only."""
