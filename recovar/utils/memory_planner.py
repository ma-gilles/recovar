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
``calibration_data`` is ``None`` and prediction-driven warnings + the
adaptive-n_pcs algorithm degrade to "no prediction" (the run proceeds
with the user's specified n_pcs).
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


_COVARIANCE_WORKLOAD_CEILING_GB = {64: 7.0, 128: 40.0, 256: 55.0}


def _covariance_workload_ceiling_gb(grid_size: int) -> float:
    """Workload ceiling (batches + FFT + accumulators; no basis) from
    saturation sweep 8020210 at full budget on A100 80GB / SPA /
    n_images=2000. Linear interp / quadratic extrap for unknown grids."""
    if grid_size in _COVARIANCE_WORKLOAD_CEILING_GB:
        return _COVARIANCE_WORKLOAD_CEILING_GB[grid_size]
    sizes = sorted(_COVARIANCE_WORKLOAD_CEILING_GB.keys())
    if grid_size <= sizes[0]:
        return _COVARIANCE_WORKLOAD_CEILING_GB[sizes[0]] * (grid_size / sizes[0]) ** 2
    if grid_size >= sizes[-1]:
        slope = (_COVARIANCE_WORKLOAD_CEILING_GB[sizes[-1]] - _COVARIANCE_WORKLOAD_CEILING_GB[sizes[-2]]) / (
            sizes[-1] - sizes[-2]
        )
        return _COVARIANCE_WORKLOAD_CEILING_GB[sizes[-1]] + slope * (grid_size - sizes[-1])
    for lo, hi in zip(sizes[:-1], sizes[1:]):
        if lo <= grid_size <= hi:
            t = (grid_size - lo) / (hi - lo)
            return _COVARIANCE_WORKLOAD_CEILING_GB[lo] + t * (
                _COVARIANCE_WORKLOAD_CEILING_GB[hi] - _COVARIANCE_WORKLOAD_CEILING_GB[lo]
            )
    return _COVARIANCE_WORKLOAD_CEILING_GB[128]


def _adaptive_n_pcs_formula_fallback(*, grid_size: int, effective_budget_gb: float, desired_n_pcs: int) -> int:
    """Heuristic n_pcs choice when no calibration cells are available.

    Mirrors the formula in
    ``covariance_estimation.get_default_covariance_computation_options``
    so the planner's choice is consistent with what
    ``adaptive_n_pcs=True`` would produce there. Walks ``n_pcs`` down
    from ``desired_n_pcs`` until predicted memory fits 70% of the
    budget; floors at 50.

    Without this fallback ``run_test_dataset --adaptive-n-pcs`` is a
    no-op when the calibration JSON is absent — exactly the regression
    the smoke test exists to catch.

    Updated 2026-05-11: replaced the legacy ``(75 / 200**4) × n_pcs**4``
    term (which over-estimated peak by ~2× at grid=128 per saturation
    sweep 8020210) with an empirical workload-ceiling term. See
    ``_covariance_workload_ceiling_gb`` for the data source.
    """
    if grid_size < 1 or effective_budget_gb <= 0:
        return desired_n_pcs

    volume_size = grid_size**3
    dtype_size = 8  # complex64
    available = effective_budget_gb * 0.7
    basis_coef = volume_size * dtype_size / 1e9  # basis scales linearly with n_pcs
    workload_gb = min(0.7 * effective_budget_gb, _covariance_workload_ceiling_gb(grid_size))

    for n_pcs in range(desired_n_pcs, 0, -1):
        if workload_gb + basis_coef * n_pcs <= available:
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

    # Push the effective budget into the legacy ``set_gpu_memory_limit``
    # global so any code path that still calls ``get_gpu_memory_total``
    # sees the same number.
    try:
        _helpers.set_gpu_memory_limit(budget.effective_budget_gb)
    except Exception as exc:
        logger.debug("set_gpu_memory_limit failed: %s", exc)

    # Calibration lookup
    calibration_table = calibration_table or load_calibration_table()
    cells = calibration_table.cells_by_command.get(command, []) if calibration_table is not None else []

    # Effective n_pcs: adaptive mode picks from the table; otherwise we
    # honor the caller's value verbatim.
    # AUTO-TRIGGER adaptive when the covariance basis term ALONE would
    # exceed the budget. This is independent of batch sizes — basis is
    # n_pcs × grid³ × 8 bytes, allocated once and held throughout
    # covariance estimation. If that exceeds budget, the run is
    # guaranteed to OOM with NO knob the user could turn other than
    # --adaptive-n-pcs or smaller grid. Auto-enabling avoids the
    # "user runs pipeline, OOMs, has to learn about the flag" trap.
    _basis_term_gb = desired_n_pcs * (grid_size**3) * 8 / 1e9
    if not adaptive_n_pcs and _basis_term_gb > budget.effective_budget_gb * 0.50:
        logger.warning(
            "Auto-enabling --adaptive-n-pcs: covariance basis term alone "
            "(%d PCs × grid_size=%d) = %.2f GB exceeds 50%% of budget "
            "(%.2f GB). Without adaptive, the run is guaranteed to OOM. "
            "If you specifically want to test the n_pcs=%d path, use a "
            "larger budget or smaller grid.",
            desired_n_pcs,
            grid_size,
            _basis_term_gb,
            budget.effective_budget_gb,
            desired_n_pcs,
        )
        adaptive_n_pcs = True

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

    # Batch sizes — keep the existing empirical formulas, fed with the
    # effective budget. low_memory / very_low_memory tighten further.
    image_batch = _helpers.get_image_batch_size(grid_size, budget.effective_budget_gb)
    volume_batch = _helpers.get_vol_batch_size(grid_size, budget.effective_budget_gb)
    column_batch = _helpers.get_column_batch_size(grid_size, budget.effective_budget_gb)

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
    diagnostic artifacts (memory_plan.json, memory_trace.jsonl,
    allocator_env.json, args.json, error_hint.json, profile dumps)
    live under this directory. ``run.log`` stays at outdir root for
    backward compatibility.
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
        # Always lives in _diagnostics/ now (always-on; no flag).
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
