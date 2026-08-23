"""Centralized GPU-memory budget + batch-size + adaptive-n_pcs planning.

This module replaces the scattered ``get_*_batch_size`` calls in
``commands/pipeline.py`` with a single ``make_memory_plan`` entry point
that:

  * collects the user's requested budget (``--gpu-budget-gb``), the JAX-reported
    bytes_limit, and (when available) the physical free memory from
    NVML / nvidia-smi
  * combines them into one ``effective_budget_gb`` with a clear
    provenance string
  * looks up the empirical calibration table (when present) to predict
    peak memory for the given ``(grid_size, n_pcs, backend)``
  * picks ``n_pcs_to_compute`` adaptively from the calibration table
    when ``--adaptive-n-pcs`` is set
  * returns a ``MemoryPlan`` bundle that downstream code reads directly
    instead of recomputing batch sizes ad hoc.

The planner NEVER refuses to launch. If the predicted peak exceeds
``effective_budget``, a loud WARNING is logged and the plan is still
returned. The error-hints classifier handles the OOM-vs-actionable-hint
side at failure time.

Calibration table loading is best-effort: if the JSON file is missing,
``calibration_data`` is ``None``. ``--adaptive-n-pcs`` then falls back
to the analytic peak formula instead of becoming a no-op.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)

# Slack multiplier baked into "will it fit" predictions. Single value
# (single H100-derived calibration table); see PR description.
PEAK_PREDICTION_SLACK = 1.2


# Backend-specific divisor for empirical batch-size formulas. The user-facing
# budget remains unchanged; this only tightens image/volume/column batch sizes
# for backends with larger transient allocations.
_BATCH_BUDGET_DIVISOR_BY_BACKEND: dict[Backend, float] = {
    "custom_cuda": 1.0,
    "jax_fallback": 5.0,
    "cpu": 1.0,
}

# Upper bound for the budget fed to empirical batch-size formulas.
# The PR-138 matrix showed jax_fallback is not monotonic-safe here: at
# grid=128 the 24 GB cell passed with 4.8 GB handed to the formulas, but
# the 40 GB cell handed over 8.0 GB and OOMed in the JAX relion kernel.
_BATCH_BUDGET_CAP_GB_BY_BACKEND: dict[Backend, float] = {
    "jax_fallback": 4.8,
}


def batch_budget_divisor_for_backend(backend: Backend) -> float:
    """Return the divisor applied to batch-size formulas for ``backend``."""
    return _BATCH_BUDGET_DIVISOR_BY_BACKEND.get(backend, 1.0)


def batch_budget_cap_gb_for_backend(backend: Backend) -> float | None:
    """Return the max formula budget for ``backend``, if one is needed."""
    return _BATCH_BUDGET_CAP_GB_BY_BACKEND.get(backend)


def batch_budget_for_backend(effective_budget_gb: float, backend: Backend) -> float:
    """Budget to feed legacy empirical batch-size formulas."""
    batch_budget_gb = effective_budget_gb / batch_budget_divisor_for_backend(backend)
    cap_gb = batch_budget_cap_gb_for_backend(backend)
    if cap_gb is not None:
        batch_budget_gb = min(batch_budget_gb, cap_gb)
    return batch_budget_gb


# Reserve subtracted from physical_free before deriving effective budget.
def _physical_reserve_gb(physical_total_gb: float) -> float:
    return max(2.0, 0.05 * physical_total_gb)


# Test-only: override JAX-reported peak memory so unit tests can
# deterministically trigger the over-budget code path. Read as float GB.
_DEBUG_FORCE_PEAK_GB_ENV = "RECOVAR_DEBUG_FORCE_PEAK_GB"


def _debug_force_peak_gb() -> float | None:
    raw = os.environ.get(_DEBUG_FORCE_PEAK_GB_ENV)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (not a float)", _DEBUG_FORCE_PEAK_GB_ENV, raw)
        return None


Backend = Literal["custom_cuda", "jax_fallback", "cpu"]
Command = Literal["pipeline", "compute_state"]


@dataclass
class GpuMemoryBudget:
    """Provenance-carrying view of how the planner picked the budget."""

    requested_gb: float | None
    physical_total_gb: float | None
    physical_free_gb: float | None
    jax_limit_gb: float | None
    effective_budget_gb: float
    backend: Backend
    custom_cuda_disabled: bool
    source: str  # which input drove the effective budget
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MemoryPlan:
    """All batch-size + prediction outputs the rest of the pipeline reads."""

    command: Command
    grid_size: int
    n_images: int
    n_pcs_to_compute: int
    desired_n_pcs: int
    adaptive_n_pcs: bool
    image_batch_size: int
    volume_batch_size: int
    column_batch_size: int
    embedding_batch_size: int | None
    sampling_n_cols: int
    randomized_sketch_size: int
    predicted_peak_gb_total: float | None
    predicted_peak_gb_by_phase: dict[str, float]
    calibration_status: str  # "exact", "rounded_up", "extrapolated", "uncalibrated"
    calibration_table_version: int | None
    budget: GpuMemoryBudget

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # GpuMemoryBudget is already serializable via asdict
        return d


# ---------------------------------------------------------------------------
# Calibration table loading
# ---------------------------------------------------------------------------

CALIBRATION_FILENAME = "memory_calibration_data.json"


@dataclass
class CalibrationCell:
    grid_size: int
    backend: Backend
    n_pcs_or_volumes: int  # n_pcs for pipeline, n_volumes for compute_state
    peak_gb_total: float
    peak_gb_by_phase: dict[str, float]
    status: str  # "ok", "oom"


@dataclass
class CalibrationTable:
    schema_version: int
    calibrated_on: dict[str, Any]
    cells_by_command: dict[str, list[CalibrationCell]]


def load_calibration_table() -> CalibrationTable | None:
    """Load the committed calibration JSON; return None if missing/invalid."""
    try:
        with resources.files("recovar.utils").joinpath(CALIBRATION_FILENAME).open("rb") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Could not load %s: %s", CALIBRATION_FILENAME, exc)
        return None

    try:
        cells_by_command: dict[str, list[CalibrationCell]] = {}
        for cmd, cell_list in raw.get("tables", {}).items():
            cells_by_command[cmd] = [
                CalibrationCell(
                    grid_size=int(c["grid_size"]),
                    backend=c["backend"],
                    n_pcs_or_volumes=int(c.get("n_pcs", c.get("n_volumes", 0))),
                    peak_gb_total=float(c["peak_gb_total"]),
                    peak_gb_by_phase=dict(c.get("peak_gb_by_phase", {})),
                    status=c.get("status", "ok"),
                )
                for c in cell_list
            ]
        return CalibrationTable(
            schema_version=int(raw.get("schema_version", 0)),
            calibrated_on=raw.get("calibrated_on", {}),
            cells_by_command=cells_by_command,
        )
    except Exception as exc:
        logger.warning("Calibration JSON is malformed: %s", exc)
        return None


def _round_up_to_calibrated(
    cells: list[CalibrationCell],
    *,
    grid_size: int,
    backend: Backend,
    n_pcs: int,
) -> tuple[CalibrationCell | None, str]:
    """Return the next-larger-or-equal calibrated cell and a status string.

    Returns:
        (cell, "exact" | "rounded_up" | "extrapolated") on success;
        (None, "uncalibrated") if no cell exists for this backend.
    """
    matching = [c for c in cells if c.backend == backend and c.status == "ok"]
    if not matching:
        return None, "uncalibrated"

    # Sort by (grid_size, n_pcs) ascending
    matching.sort(key=lambda c: (c.grid_size, c.n_pcs_or_volumes))
    candidates = [c for c in matching if c.grid_size >= grid_size and c.n_pcs_or_volumes >= n_pcs]
    if candidates:
        chosen = min(candidates, key=lambda c: (c.grid_size, c.n_pcs_or_volumes))
        if chosen.grid_size == grid_size and chosen.n_pcs_or_volumes == n_pcs:
            return chosen, "exact"
        return chosen, "rounded_up"
    # User exceeds the largest calibrated cell: extrapolate from largest
    largest = matching[-1]
    return largest, "extrapolated"


# Multiplicative slack on the analytic peak prediction. Calibrated from
# per-chunk fork-per-cell measurement 2026-05-14
# (_agent_scratch/chunk_floor_measurements.json) using device_put-only
# allocations (no `+ 1` artifact). For the projected-covariance chunk,
# measured / predicted across 13 cells spanning {grid=64,128,256} x
# {n_pcs=50,100,200} x {batch=1,16,64}: min=1.00 max=1.03.
#
# Bumped 2026-05-15 from 1.30 → 1.50 after slurm stress test 8273301
# (test_pipeline_gpu_memory_stress_256_5k) OOMed at
# jax_limit=42 GB / grid=256 / n_pcs=200: picker predicted 41.9 GB
# (under budget) but actual cross-phase peak exceeded 42 GB due to
# persistent state from earlier pipeline phases (mean recon, noise,
# cov-cols, SVD) that the proj-cov-only formula doesn't track. The
# extra 20% absolute headroom forces the picker to leave room for
# that cross-phase overlap.
_PEAK_SLACK_BY_GRID = {64: 1.50, 128: 1.50, 256: 1.50}


def _peak_slack(grid_size: int) -> float:
    """Multiplicative slack on the analytic peak prediction."""
    return _PEAK_SLACK_BY_GRID.get(grid_size, 1.50)


def _predict_covariance_peak_gb(grid_size: int, n_pcs: int, batch_size: int) -> float:
    """Predict peak GPU memory (GB) for ``compute_projected_covariance``.

    Static walk of ``_reduce_covariance_inner_explicit`` +
    ``_projected_covariance_packed_lhs_batch``. Five dominant terms:

    1. basis            — (n_pcs, grid³) complex64  : n_pcs · grid³ · 8 B
    2. AUs + AUs_noise  — 2 × (n_pcs, batch, grid²/2) complex64
                        : 2 · n_pcs · batch · grid²/2 · 8 B
                        = n_pcs · batch · grid² · 8 B
    3. cross_terms      — (P, n_pcs, n_pcs) float32 : P · n_pcs² · 4 B
    4. packed_lhs       — (P, P) float32            : P² · 4 B
    5. lhs_rows+cols    — 2 × (batch, P, n_pcs) float32
                        : 2 · batch · P · n_pcs · 4 B    (per-batch term)
                        — MISSED earlier; dominates at large batch_size.
                        At batch=488, n=200: 15.7 GB. Slurm 8252749 OOM at
                        17.6 GB came from this term.

    P = n_pcs · (n_pcs + 1) / 2. Slack absorbs FFT plans, JIT cache,
    masks, and other small transients we don't enumerate.
    """
    half_img = (grid_size * grid_size) // 2
    P = n_pcs * (n_pcs + 1) // 2

    basis_gb = n_pcs * grid_size**3 * 8 / 1e9
    aus_noise_gb = 2 * n_pcs * batch_size * half_img * 8 / 1e9
    n4_packed_gb = (P * n_pcs * n_pcs + P * P) * 4 / 1e9
    # lhs_rows + lhs_cols: each (batch, P, n_pcs) float32
    lhs_rows_cols_gb = 2 * batch_size * P * n_pcs * 4 / 1e9

    raw_peak = basis_gb + aus_noise_gb + n4_packed_gb + lhs_rows_cols_gb
    return raw_peak * _peak_slack(grid_size)


def _covariance_runtime_batch_size(grid_size: int, n_pcs: int, budget_gb: float) -> int:
    """Mirror of ``_projected_covariance_batch_size`` from
    ``recovar/heterogeneity/principal_components.py``.

    Uses the legacy ``get_embedding_batch_size``-derived per-image
    formula so the planner's prediction matches the batch the runtime
    actually picks. The formula under-counts the lhs_rows/cols per-image
    cost but its float-reduction ordering is what regression baselines
    encode — changing it perturbs ``compute_projected_covariance``
    results beyond CI tolerance.

    The picker's *peak prediction* (``_predict_covariance_peak_gb``)
    DOES include the lhs_rows/cols term so n_pcs is chosen correctly;
    only the runtime batch is left at the legacy under-count.
    """
    image_size = grid_size * grid_size
    basis_gb = n_pcs * (grid_size**3) * 8 / 1e9
    P = n_pcs * (n_pcs + 1) // 2
    # Legacy reservation (8 B/elem; the actual LHS is float32 → 4 B/elem,
    # so this over-reserves by 2x). Kept for batch-size parity with the
    # production picker; baselines depend on the same batch sizes.
    persistent_lhs_gb = 2 * P * P * 8 / 1e9
    remaining_gb = budget_gb - basis_gb - persistent_lhs_gb
    if remaining_gb <= 0:
        return 1

    # Legacy per-image cost: image_size · max(n_pcs,4) + n_pcs² (complex64).
    # Misses the lhs_rows/cols term — kept legacy for baseline stability.
    per_image_gb = (image_size * max(n_pcs, 4) + n_pcs * n_pcs) * 8 / 1e9
    batch = int(remaining_gb / per_image_gb / 20)
    return max(1, batch)


def _adaptive_n_pcs_formula_fallback(*, grid_size: int, effective_budget_gb: float, desired_n_pcs: int) -> int:
    """Heuristic n_pcs choice when no calibration cells are available.

    Walks ``n_pcs`` down from ``desired_n_pcs`` until the analytic peak
    prediction fits ``effective_budget``. The batch size is computed via
    ``_covariance_runtime_batch_size`` so it matches what the actual
    covariance code path will use at runtime (which uses a `/20` safety
    factor in ``get_embedding_batch_size``).

    History:
      Pre-2026-05-14: empirical "workload ceiling" table + n_pcs⁴ × 8B
      assuming float64 LHS. Wrong by 2-16× at grid=128.

      2026-05-14: replaced with term-by-term analytic formula
      (basis + AUs + n^4 packed) plus a grid-dependent slack. Initial
      version fed ``get_image_batch_size`` (the LARGER batch used by
      mean reconstruction etc.) which over-predicted peak — the
      production covariance call uses a much smaller batch via
      ``_projected_covariance_batch_size`` (/20 safety factor). Fixed
      by introducing ``_covariance_runtime_batch_size`` here so the
      picker mirrors the runtime exactly.
    """
    if grid_size < 1 or effective_budget_gb <= 0:
        return desired_n_pcs

    for n_pcs in range(desired_n_pcs, 0, -1):
        batch_size = _covariance_runtime_batch_size(grid_size, n_pcs, effective_budget_gb)
        predicted_peak = _predict_covariance_peak_gb(grid_size, n_pcs, batch_size)
        if predicted_peak <= effective_budget_gb:
            return n_pcs
    return min(desired_n_pcs, 50)


def _select_adaptive_n_pcs(
    cells: list[CalibrationCell],
    *,
    grid_size: int,
    backend: Backend,
    desired_n_pcs: int,
    effective_budget_gb: float,
) -> tuple[int, str]:
    """Pick the largest calibrated n_pcs that fits ``effective_budget``.

    When the calibration table is empty / does not cover this backend,
    falls back to ``_adaptive_n_pcs_formula_fallback`` so the flag is
    not a no-op.

    Returns ``(n_pcs, status)`` where status is one of:
      "exact"                 — exact grid+n_pcs match in the table
      "rounded_up"            — calibrated cell with larger grid
      "floor_used"            — even the smallest calibrated n_pcs would OOM
      "uncalibrated_formula"  — no calibration; used the formula fallback
      "uncalibrated_passthrough" — no calibration; budget large enough that
                                   the formula returned ``desired_n_pcs``
    """
    matching = [
        c
        for c in cells
        if c.backend == backend and c.status == "ok" and c.grid_size >= grid_size  # round up grid axis
    ]
    if not matching:
        n_pcs = _adaptive_n_pcs_formula_fallback(
            grid_size=grid_size,
            effective_budget_gb=effective_budget_gb,
            desired_n_pcs=desired_n_pcs,
        )
        if n_pcs < desired_n_pcs:
            return n_pcs, "uncalibrated_formula"
        return desired_n_pcs, "uncalibrated_passthrough"

    # Pick the smallest grid_size that's >= user's grid_size (round up)
    grids = sorted({c.grid_size for c in matching})
    chosen_grid = next((g for g in grids if g >= grid_size), grids[-1])
    same_grid = [c for c in matching if c.grid_size == chosen_grid]
    same_grid.sort(key=lambda c: c.n_pcs_or_volumes)

    for c in reversed(same_grid):
        if c.n_pcs_or_volumes > desired_n_pcs:
            continue
        if c.peak_gb_total * PEAK_PREDICTION_SLACK <= effective_budget_gb:
            return c.n_pcs_or_volumes, ("exact" if chosen_grid == grid_size else "rounded_up")

    floor = same_grid[0].n_pcs_or_volumes
    return floor, "floor_used"


# ---------------------------------------------------------------------------
# Budget assembly
# ---------------------------------------------------------------------------


def _assemble_budget(
    *,
    requested_gpu_gb: float | None,
    backend: Backend,
    custom_cuda_disabled: bool,
    cuda_warnings: Iterable[str],
) -> GpuMemoryBudget:
    """Combine requested / jax-reported / physical sources into one budget."""

    physical_total = None
    physical_free = None

    # Probe physical memory only when we actually have a GPU backend
    if backend != "cpu":
        try:
            from recovar.utils.gpu_preflight import get_physical_gpu_memory_info

            info = get_physical_gpu_memory_info(0)
            if info is not None:
                physical_total = info.total_gb
                physical_free = info.free_gb
        except Exception as exc:
            logger.debug("Physical memory probe failed: %s", exc)

    # JAX-reported limit (may include the XLA_PYTHON_CLIENT_MEM_FRACTION
    # cap). Best effort. Imported via the package so tests that
    # monkeypatch ``recovar.utils.get_gpu_memory_total`` are still
    # honored.
    jax_limit = None
    if backend != "cpu":
        try:
            from recovar import utils as _helpers

            jax_limit = float(_helpers.get_gpu_memory_total())
        except Exception as exc:
            logger.debug("JAX memory_stats probe failed: %s", exc)

    # CPU branch: pick something sensible from the existing helper.
    if backend == "cpu":
        try:
            from recovar import utils as _helpers

            cpu_limit = float(_helpers.get_gpu_memory_total())  # CPU branch in helper
        except Exception:
            cpu_limit = 16.0
        return GpuMemoryBudget(
            requested_gb=requested_gpu_gb,
            physical_total_gb=None,
            physical_free_gb=None,
            jax_limit_gb=cpu_limit,
            effective_budget_gb=requested_gpu_gb if requested_gpu_gb else cpu_limit,
            backend=backend,
            custom_cuda_disabled=custom_cuda_disabled,
            source="cpu_helper",
            warnings=list(cuda_warnings),
        )

    candidates: list[tuple[str, float]] = []
    if requested_gpu_gb is not None:
        candidates.append(("requested_gpu_gb", float(requested_gpu_gb)))
    if jax_limit is not None and jax_limit > 0:
        candidates.append(("jax_limit_gb", jax_limit))

    # ``physical_free - reserve`` only enters the min when the user
    # supplied an explicit ``--gpu-budget-gb`` (i.e. they asked for a budget
    # we should validate against the GPU's current free space). Without
    # an explicit request, fall through to the JAX-reported limit
    # exactly the way the legacy code did — otherwise we silently
    # shrink batch sizes by ~5-10% on a quiet GPU and shift quality
    # baselines.
    if requested_gpu_gb is not None and physical_total is not None and physical_free is not None:
        reserve = _physical_reserve_gb(physical_total)
        candidates.append(("physical_free_minus_reserve", max(1.0, physical_free - reserve)))

    if not candidates:
        # Truly nothing to go on: 80 GB matches the existing fallback in
        # ``helpers.get_gpu_memory_total``.
        effective = 80.0
        source = "fallback_80gb"
    else:
        source, effective = min(candidates, key=lambda kv: kv[1])

    return GpuMemoryBudget(
        requested_gb=requested_gpu_gb,
        physical_total_gb=physical_total,
        physical_free_gb=physical_free,
        jax_limit_gb=jax_limit,
        effective_budget_gb=effective,
        backend=backend,
        custom_cuda_disabled=custom_cuda_disabled,
        source=source,
        warnings=list(cuda_warnings),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def make_memory_plan(
    *,
    command: Command,
    grid_size: int,
    n_images: int,
    requested_gpu_gb: float | None,
    low_memory: bool,
    very_low_memory: bool,
    adaptive_n_pcs: bool,
    desired_n_pcs: int = 200,
    calibration_table: CalibrationTable | None = None,
) -> MemoryPlan:
    """Top-level planner used by every heavy-GPU command."""

    # Import via the ``recovar.utils`` package (not ``recovar.utils.helpers``)
    # so that monkeypatches like ``setattr(pipeline_cmd.utils, "get_image_batch_size", ...)``
    # in existing tests still reach the lookup site.
    from recovar import utils as _helpers
    from recovar.utils import cuda_env

    backend = cuda_env.detect_backend()
    custom_cuda_disabled, cuda_warnings = cuda_env.custom_cuda_disabled_from_env()

    budget = _assemble_budget(
        requested_gpu_gb=requested_gpu_gb,
        backend=backend,
        custom_cuda_disabled=custom_cuda_disabled,
        cuda_warnings=cuda_warnings,
    )

    # Calibration lookup
    calibration_table = calibration_table or load_calibration_table()
    cells = calibration_table.cells_by_command.get(command, []) if calibration_table is not None else []

    # Pre-launch WARNING when guaranteed covariance allocations would
    # likely OOM. We do NOT silently flip on adaptive — that mutates
    # the user's explicit choice. Instead, log the prediction and let
    # the run proceed. If the run does OOM, ``error_hints._hint_gpu_oom``
    # detects "predicted_peak > effective_budget" and recommends
    # ``--adaptive-n-pcs`` as the single specific fix in the
    # post-crash hint.
    _basis_term_gb = desired_n_pcs * (grid_size**3) * 8 / 1e9
    _packed = desired_n_pcs * (desired_n_pcs + 1) // 2
    _projected_lhs_gb = _packed * _packed * 8 / 1e9
    _fixed_allocs_gb = _basis_term_gb + _projected_lhs_gb
    if not adaptive_n_pcs and _fixed_allocs_gb > budget.effective_budget_gb * 0.50:
        logger.warning(
            "PRE-LAUNCH OOM PREDICTION: covariance fixed allocations "
            "(n_pcs=%d, grid_size=%d) = basis %.2f GB + projected_lhs "
            "%.2f GB = %.2f GB exceed 50%% of budget (%.2f GB). The run "
            "will likely OOM. To prevent, add --adaptive-n-pcs (the "
            "planner will shrink n_pcs to fit), or use a smaller grid "
            "/ larger --gpu-budget-gb. Launching anyway because the "
            "user did not pass --adaptive-n-pcs.",
            desired_n_pcs,
            grid_size,
            _basis_term_gb,
            _projected_lhs_gb,
            _fixed_allocs_gb,
            budget.effective_budget_gb,
        )

    chosen_n_pcs = desired_n_pcs
    cal_status = "uncalibrated"
    predicted_peak: float | None = None
    predicted_by_phase: dict[str, float] = {}
    table_version = calibration_table.schema_version if calibration_table else None

    if cells:
        if adaptive_n_pcs:
            chosen_n_pcs, cal_status = _select_adaptive_n_pcs(
                cells,
                grid_size=grid_size,
                backend=budget.backend,
                desired_n_pcs=desired_n_pcs,
                effective_budget_gb=budget.effective_budget_gb,
            )
            cell, _ = _round_up_to_calibrated(
                cells,
                grid_size=grid_size,
                backend=budget.backend,
                n_pcs=chosen_n_pcs,
            )
        else:
            cell, cal_status = _round_up_to_calibrated(
                cells,
                grid_size=grid_size,
                backend=budget.backend,
                n_pcs=desired_n_pcs,
            )
        if cell is not None:
            predicted_peak = cell.peak_gb_total
            predicted_by_phase = dict(cell.peak_gb_by_phase)
    elif adaptive_n_pcs:
        # No calibration table at all → still honor --adaptive-n-pcs
        # via the formula fallback. Without this branch the flag is a
        # no-op and ``run_test_dataset`` on a small GPU still launches
        # 200 PCs (review-feedback bug).
        chosen_n_pcs, cal_status = _select_adaptive_n_pcs(
            [],
            grid_size=grid_size,
            backend=budget.backend,
            desired_n_pcs=desired_n_pcs,
            effective_budget_gb=budget.effective_budget_gb,
        )

    # Pre-launch warning if we predict an OOM. We never refuse.
    if predicted_peak is not None and predicted_peak * PEAK_PREDICTION_SLACK > budget.effective_budget_gb:
        logger.warning(
            "Memory plan predicts peak %.1f GB (with %.0f%% slack: %.1f GB) which "
            "exceeds the effective budget %.1f GB. The pipeline will still launch — "
            "if it OOMs, the error message will repeat suggested flags. "
            "To recover proactively, try --adaptive-n-pcs, --low-memory-option, "
            "or a higher --gpu-budget-gb.",
            predicted_peak,
            (PEAK_PREDICTION_SLACK - 1) * 100,
            predicted_peak * PEAK_PREDICTION_SLACK,
            budget.effective_budget_gb,
        )

    # Batch sizes — keep the existing empirical formulas, fed with a
    # backend-aware budget. The user-facing effective budget remains the
    # contract recorded in memory_plan.json; fallback-only deflation just
    # prevents JAX fallback from choosing custom-CUDA-sized batches.
    batch_budget_gb = batch_budget_for_backend(budget.effective_budget_gb, budget.backend)
    image_batch = _helpers.get_image_batch_size(grid_size, batch_budget_gb)
    volume_batch = _helpers.get_vol_batch_size(grid_size, batch_budget_gb)
    column_batch = _helpers.get_column_batch_size(grid_size, batch_budget_gb)

    # At grid=256 the legacy formulas are too aggressive on an 80 GB
    # class GPU. 100-column covariance batches allocate 12.5 GiB H/B
    # accumulators per halfset, and 50-volume PCA FFT batches can ask XLA
    # for ~25 GiB temporaries. Keep the production 128^3 path unchanged,
    # but cap 256^3+ batches to keep those transient allocations bounded.
    if grid_size >= 256:
        volume_batch = min(volume_batch, 10)
        column_batch = min(column_batch, 50)

    if very_low_memory:
        image_batch = max(1, image_batch // 4)
        volume_batch = max(1, volume_batch // 4)
        column_batch = max(1, column_batch // 4)
    elif low_memory:
        image_batch = max(1, image_batch // 2)
        volume_batch = max(1, volume_batch // 2)
        column_batch = max(1, column_batch // 2)

    return MemoryPlan(
        command=command,
        grid_size=grid_size,
        n_images=n_images,
        n_pcs_to_compute=chosen_n_pcs,
        desired_n_pcs=desired_n_pcs,
        adaptive_n_pcs=adaptive_n_pcs,
        image_batch_size=image_batch,
        volume_batch_size=volume_batch,
        column_batch_size=column_batch,
        embedding_batch_size=None,  # filled in later when basis is known
        sampling_n_cols=300,
        randomized_sketch_size=300,
        predicted_peak_gb_total=predicted_peak,
        predicted_peak_gb_by_phase=predicted_by_phase,
        calibration_status=cal_status,
        calibration_table_version=table_version,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Diagnostics writers
# ---------------------------------------------------------------------------


def diagnostics_dir(outdir: str | Path) -> Path:
    """Return ``<outdir>/_diagnostics`` and ensure it exists.

    Underscore prefix signals "internal/auxiliary" and tells users
    "no need to look here unless something went wrong." All
    diagnostic artifacts (memory_plan.json, optional memory_trace.jsonl,
    optional allocator_env.json / args.json, error_hint.json, profile
    dumps) live under this directory. ``run.log`` stays at outdir root
    for backward compatibility.
    """
    p = Path(outdir) / "_diagnostics"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_memory_plan_json(plan: MemoryPlan, outdir: str | Path) -> Path:
    """Always-on artifact: writes ``_diagnostics/memory_plan.json``."""
    out_path = diagnostics_dir(outdir) / "memory_plan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(plan.to_dict(), fh, indent=2, default=str)
    return out_path


class MemoryTraceWriter:
    """Append per-phase peak-memory observations to ``memory_trace.jsonl``.

    Always recoverable: if jax memory_stats are unavailable, we emit a
    row with ``jax_memory_stats_available=False`` rather than raising.
    """

    def __init__(self, outdir: str | Path, *, enabled: bool = True):
        # Trace files live in _diagnostics/ when memory profiling is enabled.
        self.path = diagnostics_dir(outdir) / "memory_trace.jsonl"
        self.enabled = enabled
        self._opened = False
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # truncate at start of each run
            self.path.write_text("")
            self._opened = True

    def record(self, phase: str, *, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        row: dict[str, Any] = {"phase": phase, "ts": time.time()}

        # JAX peak / current
        try:
            from recovar import utils as _helpers

            row["jax_in_use_gb"] = float(_helpers.get_gpu_memory_used())
            row["jax_peak_gb"] = float(_helpers.get_peak_gpu_memory_used())
            row["jax_memory_stats_available"] = True
        except Exception:
            row["jax_memory_stats_available"] = False

        # Apply test-only override
        forced = _debug_force_peak_gb()
        if forced is not None:
            row["jax_peak_gb"] = forced
            row["jax_peak_gb_forced"] = True

        # Physical free (best effort)
        try:
            from recovar.utils.gpu_preflight import get_physical_gpu_memory_info

            info = get_physical_gpu_memory_info(0)
            if info is not None:
                row["physical_free_gb"] = info.free_gb
                row["physical_total_gb"] = info.total_gb
        except Exception:
            pass

        if extra:
            row.update(extra)

        # Resilient append: if the diagnostics dir was wiped between
        # __init__ and now (observed once during a sweep cold-compile),
        # recreate it instead of crashing the pipeline. A trace writer
        # error must never abort a real run.
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
