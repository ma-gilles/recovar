"""Phase-level GPU-memory model for RECOVAR.

Each pipeline phase has a memory function returning a structured
``MemoryBreakdown`` (per-term GB + total + code references). The
SPA and tilt-series pipelines have SEPARATE functions per phase
because their batching and CTF handling differ; whether they reduce
to the same shape plus a multiplier is a question for the validation
sweep, not an assumption baked in here.

CONSTANTS PROVENANCE
====================

Every numeric constant in this file should carry a doc comment naming:

  - the sweep run-id and cell that pinned it
  - the observed peak (in GB) from that cell
  - the hotspot section in ``docs/memory_hotspots.md`` it corresponds to

If a constant has ``TODO(sweep)`` next to it, that means it's a
placeholder pending the validation sweep. The placeholder is the
inherited value (see ``recovar/utils/helpers.py`` and
``recovar/heterogeneity/covariance_estimation.py``). DO NOT trust
those — they were tuned at unknown times under unknown conditions
and the very point of this file is to replace them.

WORKFLOW FOR RE-FITTING
=======================

When hardware or JAX changes meaningfully:

    pixi run python scripts/validate_memory_formulas.py \\
        --mode record \\
        --output _diagnostics/sweep_run_<id>.json

    pixi run python scripts/fit_memory_constants.py \\
        _diagnostics/sweep_run_<id>.json \\
        --output _diagnostics/fit_report_<id>.md

    # Human reads fit_report_<id>.md. Edits constants below by hand.
    # Adds/updates the provenance doc comment on each edited constant.

    pixi run python scripts/validate_memory_formulas.py \\
        --mode validate \\
        --baseline _diagnostics/sweep_run_<id>.json

The fitter NEVER edits this file directly — that's a deliberate
decision so that someone always reads the residuals and confidence
warnings before constants are committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Constants (all TODO(sweep) — placeholders inherited from legacy code)
# ---------------------------------------------------------------------------

COMPLEX64_BYTES = 8

# Per-phase × per-backend headroom: how much of the user's budget the
# planner actually targets when picking batch sizes. Captures: JAX
# intermediate buffers, allocator fragmentation, JIT compilation peaks,
# half-spectrum vs full-spectrum overhead.
#
# Reviewer note: per-phase distinctions may turn out to be noise in the
# validation sweep — collapse to per-backend if so.
HEADROOM_FRACTION: dict[str, dict[str, float]] = {
    "custom_cuda": {
        "mean": 0.70,  # TODO(sweep)
        "covariance": 0.70,  # TODO(sweep)
        "embedding": 0.70,  # TODO(sweep)
    },
    "jax_fallback": {
        # Legacy `_effective_heterogeneity_memory_budget` used /3 ≈ 0.33.
        # Reviewer suggests bumping to 0.40 pending validation; covariance
        # likely remains tightest because that's where /3 originated.
        "mean": 0.45,  # TODO(sweep)
        "covariance": 0.40,  # TODO(sweep): legacy /3 = 0.33
        "embedding": 0.45,  # TODO(sweep)
    },
    "cpu": {
        "mean": 1.00,
        "covariance": 1.00,
        "embedding": 1.00,
    },
}

# Memory-mode multiplier on top of HEADROOM_FRACTION. ``--low-memory-option``
# and ``--very-low-memory-option`` map here.
MODE_MULTIPLIER: dict[str, float] = {
    "default": 1.00,
    "low": 0.50,
    "very_low": 0.25,
}

# ---------------------------------------------------------------------------
# Per-term coefficients
# ---------------------------------------------------------------------------
# These are the values fitted from the validation sweep. Until the sweep
# runs, they're set to the legacy values for parity.

# SVD/cross-product workspace inside covariance estimation.
#
# Legacy form: ``base_memory_coefficient = 75 / (200**4)`` in
# ``covariance_estimation.py:117``, predicting peak ∝ n_pcs⁴.
#
# Discovery sweep (2026-05-10, slurm job 7982854, SPA grid=128
# custom_cuda, n_pcs ∈ {20, 40, 80, 120, 160, 200}, log at
# /scratch/gpfs/GILLES/mg6942/_agent_scratch/memory_sweep_discover_20260507/
# discover_fitter_report.md) found the observed peak is essentially
# CONSTANT (~40 GB) across all tested n_pcs. Log-log fit gave
# exponent = -0.03, R² = 0.908. The legacy n_pcs⁴ assumption is
# refuted at this grid: SVD workspace is not the dominant peak driver
# in the observed regime.
#
# We therefore model svd_workspace as a small constant (no n_pcs
# scaling). The validation sweep at grid={64, 128, 256} will tell us
# whether the *grid*-dependence is cubic (FFT-dominated) or
# quadratic (image-batch FFT only).
SVD_WORKSPACE_COEF_GB = 0.0  # TODO(validate): grid scaling
SVD_WORKSPACE_EXPONENT = 0.0  # discovery: peak is constant in n_pcs

# Image-batch transient factor. Captures forward FFT in-place + adjoint
# workspace. Legacy: implicit in the ``2**18 / grid**2`` formula in
# helpers.py:223.
IMAGE_FFT_TRANSIENT = 2.5  # TODO(sweep)

# Volume-batch overhead. Legacy: ``25 vols at 256**3 in 38 GB`` →
# 38 / 25 / (256**3 × 8 / 1e9) ≈ 1.39.
VOL_BATCH_OVERHEAD = 1.5  # TODO(sweep)

# Column-batch overhead. Legacy: ``50 cols at 256**3 in 38 GB`` →
# similar derivation, ~0.7. Columns are stored more densely than vols.
COL_BATCH_OVERHEAD = 0.75  # TODO(sweep)

# Embedding contrast-solve workspace per image. Legacy: implicit in the
# ``/20`` safety divider in helpers.get_embedding_batch_size.
EMBEDDING_CONTRAST_FACTOR = 1.0  # TODO(sweep)

# Cryo-ET multipliers. Set to 1.0 initially → SPA and ET share constants.
# After validation, if ET shows distinct peaks, override these (or split
# the model into truly separate functions, per the plan).
TILT_MEAN_MULTIPLIER = 1.0  # TODO(sweep)
TILT_COVARIANCE_MULTIPLIER = 1.0  # TODO(sweep)
TILT_EMBEDDING_MULTIPLIER = 1.0  # TODO(sweep)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

Pipeline = Literal["spa", "tilt_series"]
Phase = Literal["mean", "covariance", "embedding"]
Backend = Literal["custom_cuda", "jax_fallback", "cpu"]
MemoryMode = Literal["default", "low", "very_low"]


@dataclass
class MemoryBreakdown:
    """Structured per-phase memory prediction.

    `terms_gb` lists each named contributor; `total_gb` is their sum.
    `code_refs` maps each term to a `file:line` location in the source
    that allocated it (best-effort; some attributions stop at JIT
    function boundaries because JAX's profiler is opaque inside `jit`).

    `snapshot_live_gb` and `observed_peak_gb` are populated only when
    the validation sweep reads them back; for predictions, they're
    None.
    """

    pipeline: Pipeline
    phase: Phase
    total_gb: float
    terms_gb: dict[str, float] = field(default_factory=dict)
    code_refs: dict[str, str] = field(default_factory=dict)
    snapshot_live_gb: float | None = None
    observed_peak_gb: float | None = None

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "phase": self.phase,
            "total_gb": self.total_gb,
            "terms_gb": dict(self.terms_gb),
            "code_refs": dict(self.code_refs),
            "snapshot_live_gb": self.snapshot_live_gb,
            "observed_peak_gb": self.observed_peak_gb,
        }


# ---------------------------------------------------------------------------
# SPA pipeline
# ---------------------------------------------------------------------------


def spa_mean_memory(grid_size: int, image_batch_size: int) -> MemoryBreakdown:
    """Predicted peak during SPA mean-volume estimation."""
    image_bytes = image_batch_size * grid_size**2 * COMPLEX64_BYTES
    fft_bytes = image_bytes * IMAGE_FFT_TRANSIENT
    volume_bytes = grid_size**3 * COMPLEX64_BYTES
    terms = {
        "image_batch": image_bytes / 1e9,
        "fft_workspace": (fft_bytes - image_bytes) / 1e9,
        "mean_volume": volume_bytes / 1e9,
    }
    return MemoryBreakdown(
        pipeline="spa",
        phase="mean",
        total_gb=sum(terms.values()),
        terms_gb=terms,
        code_refs={
            "image_batch": "recovar/data_io/cryoem_dataset.py:iter_batches",
            "fft_workspace": "recovar/core/forward.py:forward_model",
            "mean_volume": "recovar/reconstruction/homogeneous.py:get_mean_conformation_relion",
        },
    )


def spa_covariance_memory(
    grid_size: int,
    n_pcs: int,
    column_batch_size: int = 4,
    sampling_n_cols: int = 300,
) -> MemoryBreakdown:
    """Predicted peak during SPA covariance + PCA + reweighting.

    Decomposition (each term TBD by sweep):

      basis           = n_pcs × grid³ × 8        (rescaled PCs in GPU mem)
      columns         = sampling_n_cols × column_batch × grid³ × 8
      gram            = sampling_n_cols × n_pcs × 8 (column-reweighting Gram)
      svd_workspace   = a × n_pcs**p             (cross-product / eigen)
                        legacy assumes p=4, coef=75/200**4. Both TBD.
    """
    # Two-term form matching legacy `covariance_estimation.py:113-127`:
    #   total_gb ≈ basis (n_pcs × volume_bytes) + svd_workspace (n_pcs**4)
    # The `column_batch_size` and `sampling_n_cols` parameters are accepted
    # but currently NOT in the prediction. The discovery sweep (Phase 2)
    # will tell us whether `stored_columns` and `gram` need separate
    # terms or fold into `svd_workspace`.
    volume_bytes = grid_size**3 * COMPLEX64_BYTES
    basis_gb = n_pcs * volume_bytes / 1e9
    svd_gb = SVD_WORKSPACE_COEF_GB * (n_pcs**SVD_WORKSPACE_EXPONENT)
    terms = {
        "basis": basis_gb,
        "svd_workspace": svd_gb,
        # Reserved slots — populated by the sweep if validation shows
        # they matter. Kept here so the JSON schema is stable.
        "stored_columns": 0.0,  # TODO(sweep)
        "gram": 0.0,  # TODO(sweep)
    }
    return MemoryBreakdown(
        pipeline="spa",
        phase="covariance",
        total_gb=sum(terms.values()),
        terms_gb=terms,
        code_refs={
            "basis": "recovar/heterogeneity/principal_components.py",
            "svd_workspace": "recovar/heterogeneity/covariance_estimation.py",
            "stored_columns": "recovar/heterogeneity/covariance_estimation.py",
            "gram": "recovar/heterogeneity/covariance_estimation.py",
        },
    )


def spa_embedding_memory(
    grid_size: int,
    n_pcs: int,
    image_batch_size: int,
    contrast_grid_size: int = 20,
) -> MemoryBreakdown:
    """Predicted peak during SPA latent-coord embedding + contrast solve."""
    volume_bytes = grid_size**3 * COMPLEX64_BYTES
    image_bytes = grid_size**2 * COMPLEX64_BYTES
    basis_gb = n_pcs * volume_bytes / 1e9
    per_image_bytes = (
        image_bytes
        + n_pcs * COMPLEX64_BYTES
        + contrast_grid_size * n_pcs**2 * COMPLEX64_BYTES * EMBEDDING_CONTRAST_FACTOR
    )
    terms = {
        "basis": basis_gb,
        "image_batch": image_batch_size * image_bytes / 1e9,
        "projections": image_batch_size * n_pcs * COMPLEX64_BYTES / 1e9,
        "contrast_solve": image_batch_size
        * contrast_grid_size
        * n_pcs**2
        * COMPLEX64_BYTES
        * EMBEDDING_CONTRAST_FACTOR
        / 1e9,
    }
    return MemoryBreakdown(
        pipeline="spa",
        phase="embedding",
        total_gb=sum(terms.values()),
        terms_gb=terms,
        code_refs={
            "basis": "recovar/heterogeneity/embedding.py",
            "image_batch": "recovar/data_io/cryoem_dataset.py",
            "projections": "recovar/heterogeneity/embedding.py",
            "contrast_solve": "recovar/heterogeneity/embedding.py",
        },
    )


# ---------------------------------------------------------------------------
# Cryo-ET pipeline (separate functions, multiplier-only initially)
# ---------------------------------------------------------------------------


def tilt_series_mean_memory(
    grid_size: int,
    particle_batch_size: int,
    n_tilts: int,
) -> MemoryBreakdown:
    """Predicted peak during cryo-ET mean estimation.

    Effective image batch is ``particle_batch_size * n_tilts``.
    Initial assumption: same shape as SPA × TILT_MEAN_MULTIPLIER.
    Sweep validates whether this holds; if not, the function should be
    rewritten with ET-specific terms (e.g., per-tilt CTF state).
    """
    image_batch = particle_batch_size * n_tilts
    base = spa_mean_memory(grid_size, image_batch)
    multiplied = {k: v * TILT_MEAN_MULTIPLIER for k, v in base.terms_gb.items()}
    return MemoryBreakdown(
        pipeline="tilt_series",
        phase="mean",
        total_gb=sum(multiplied.values()),
        terms_gb=multiplied,
        code_refs=dict(base.code_refs),
    )


def tilt_series_covariance_memory(
    grid_size: int,
    n_pcs: int,
    column_batch_size: int = 4,
    n_tilts: int = 41,
    sampling_n_cols: int = 300,
) -> MemoryBreakdown:
    """Predicted peak during cryo-ET covariance + PCA."""
    base = spa_covariance_memory(grid_size, n_pcs, column_batch_size, sampling_n_cols)
    multiplied = {k: v * TILT_COVARIANCE_MULTIPLIER for k, v in base.terms_gb.items()}
    return MemoryBreakdown(
        pipeline="tilt_series",
        phase="covariance",
        total_gb=sum(multiplied.values()),
        terms_gb=multiplied,
        code_refs=dict(base.code_refs),
    )


def tilt_series_embedding_memory(
    grid_size: int,
    n_pcs: int,
    particle_batch_size: int,
    n_tilts: int,
    contrast_grid_size: int = 20,
) -> MemoryBreakdown:
    """Predicted peak during cryo-ET latent-coord embedding."""
    image_batch = particle_batch_size * n_tilts
    base = spa_embedding_memory(grid_size, n_pcs, image_batch, contrast_grid_size)
    multiplied = {k: v * TILT_EMBEDDING_MULTIPLIER for k, v in base.terms_gb.items()}
    return MemoryBreakdown(
        pipeline="tilt_series",
        phase="embedding",
        total_gb=sum(multiplied.values()),
        terms_gb=multiplied,
        code_refs=dict(base.code_refs),
    )


# ---------------------------------------------------------------------------
# Batch-size pickers
# ---------------------------------------------------------------------------


def _usable_target_gb(
    budget_gb: float,
    backend: Backend,
    phase: Phase,
    mode: MemoryMode = "default",
) -> float:
    headroom = HEADROOM_FRACTION[backend][phase]
    mult = MODE_MULTIPLIER[mode]
    return max(0.5, budget_gb * headroom * mult)


def pick_image_batch(
    grid_size: int,
    budget_gb: float,
    backend: Backend = "custom_cuda",
    mode: MemoryMode = "default",
    pipeline: Pipeline = "spa",
) -> int:
    """Largest image batch that fits the mean-phase budget."""
    target = _usable_target_gb(budget_gb, backend, "mean", mode)
    if pipeline == "tilt_series":
        target = target / max(1.0, TILT_MEAN_MULTIPLIER)
    bytes_per_image = grid_size**2 * COMPLEX64_BYTES * IMAGE_FFT_TRANSIENT
    return max(1, int(target * 1e9 / bytes_per_image))


def pick_volume_batch(
    grid_size: int,
    budget_gb: float,
    backend: Backend = "custom_cuda",
    mode: MemoryMode = "default",
) -> int:
    target = _usable_target_gb(budget_gb, backend, "covariance", mode)
    bytes_per_volume = grid_size**3 * COMPLEX64_BYTES * VOL_BATCH_OVERHEAD
    return max(1, int(target * 1e9 / bytes_per_volume))


def pick_column_batch(
    grid_size: int,
    budget_gb: float,
    backend: Backend = "custom_cuda",
    mode: MemoryMode = "default",
) -> int:
    target = _usable_target_gb(budget_gb, backend, "covariance", mode)
    bytes_per_column = grid_size**3 * COMPLEX64_BYTES * COL_BATCH_OVERHEAD
    return max(1, int(target * 1e9 / bytes_per_column))


def pick_n_pcs(
    grid_size: int,
    budget_gb: float,
    backend: Backend = "custom_cuda",
    mode: MemoryMode = "default",
    desired_n_pcs: int = 200,
    pipeline: Pipeline = "spa",
) -> tuple[int, MemoryBreakdown]:
    """Pick the largest n_pcs whose covariance-phase prediction fits.

    Returns ``(n_pcs, breakdown_at_that_n_pcs)`` so callers can log the
    chosen value's full term breakdown.
    """
    target = _usable_target_gb(budget_gb, backend, "covariance", mode)
    column_batch = pick_column_batch(grid_size, budget_gb, backend, mode)
    cov_fn = tilt_series_covariance_memory if pipeline == "tilt_series" else spa_covariance_memory
    last_breakdown = None
    for n in range(desired_n_pcs, 0, -1):
        bd = cov_fn(grid_size, n, column_batch_size=column_batch)
        last_breakdown = bd
        if bd.total_gb <= target:
            return n, bd
    return 1, last_breakdown


# ---------------------------------------------------------------------------
# Helper for routing
# ---------------------------------------------------------------------------


def memory_breakdown_for_phase(
    pipeline: Pipeline,
    phase: Phase,
    *,
    grid_size: int,
    n_pcs: int,
    image_batch_size: int = 1,
    particle_batch_size: int = 1,
    n_tilts: int = 41,
    column_batch_size: int = 4,
) -> MemoryBreakdown:
    """Single dispatch entry point."""
    if pipeline == "spa":
        if phase == "mean":
            return spa_mean_memory(grid_size, image_batch_size)
        if phase == "covariance":
            return spa_covariance_memory(grid_size, n_pcs, column_batch_size)
        if phase == "embedding":
            return spa_embedding_memory(grid_size, n_pcs, image_batch_size)
    elif pipeline == "tilt_series":
        if phase == "mean":
            return tilt_series_mean_memory(grid_size, particle_batch_size, n_tilts)
        if phase == "covariance":
            return tilt_series_covariance_memory(grid_size, n_pcs, column_batch_size, n_tilts)
        if phase == "embedding":
            return tilt_series_embedding_memory(grid_size, n_pcs, particle_batch_size, n_tilts)
    raise ValueError(f"unknown (pipeline={pipeline!r}, phase={phase!r})")
