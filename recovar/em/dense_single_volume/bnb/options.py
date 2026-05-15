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

    subdivision_mode: Literal["fixed_grid", "axis_angle_hierarchical", "paper_faithful"] = "fixed_grid"
    """``fixed_grid`` (default) prunes a caller-supplied pose grid by
    progressive L without subdividing — fast for verifying bound math but
    NOT cryoSPARC's actual algorithm. ``axis_angle_hierarchical`` uses
    paper-faithful 8-axis-angle / 4-shift Cartesian subdivision but on a
    SHARED grid + per-image survivor mask (still scales poorly at large N
    images). ``paper_faithful`` is the per-image-ragged version where each
    image carries its own evolving cell list — required for cryoSPARC-like
    speedup at scale."""

    max_shift_px: float = 10.0
    """When ``subdivision_mode='axis_angle_hierarchical'``, this is the
    radius of the initial 2D shift disc in pixels."""

    ctf_bound_mode: Literal["exact", "cryosparc_rms", "hybrid"] = "exact"
    """``exact`` uses per-image CTF/sigma in the high-frequency power bound
    (paper-validation default). ``cryosparc_rms`` uses the RMS-CTF
    approximation (|C|^2 -> 1/2) so the bound is shared across images of the
    same noise group; speed optimisation, may be looser. Phase 6+ only."""

    prior_cone_radius_deg: float | None = None
    """When set (paper_faithful mode only), each image's initial axis-angle
    cells live in a cone of this half-angle around its previous-best pose
    instead of spanning the full SO(3) cube. RELION's local-search cone
    is 22.5 deg (3 sigma at sigma_rot=7.5 deg) — set this to a similar
    value at refined-pose iterations. Leaving as None preserves the
    paper's "no pose prior" behaviour and uses the full 24 deg SO(3)
    cube; that's correct for ab-initio / iter-1 but wastes work at
    refined poses."""

    prior_shift_radius_px: float = 5.0
    """Disc radius for the per-image shift search around the prior
    translation, in pixels. Used only when prior_cone_radius_deg is set."""

    prior_cells_across_diameter: int = 4
    """Initial cell count across the cone diameter when cone-from-prior
    is enabled. Initial spacing = cone_radius / cells_across_diameter * 2.
    Default 4 → ~30-50 cells per image at stage 0."""

    score_kernel: Literal["per_image_loop", "bucketed"] = "bucketed"
    """For ``subdivision_mode='paper_faithful'``: how to score per-image
    candidate sets. 'per_image_loop' calls JAX once per image (simpler,
    high Python overhead at 100k+ images). 'bucketed' (default) groups
    images by similar candidate count, pads to bucket max, and runs one
    JAX kernel per bucket — should be 50-200x faster at scale."""

    bucketed_axis_quantum: int = 1024
    bucketed_shift_quantum: int = 64
    """Quanta for rounding up per-image (n_axis, n_shift) to bucket size in
    the bucketed scorer. Larger quanta = fewer JIT shapes (better cache
    hit rate) at the cost of more padding."""

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
