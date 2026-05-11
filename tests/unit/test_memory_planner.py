"""Unit tests for ``recovar.utils.memory_planner``.

No GPU required: NVML / nvidia-smi is monkeypatched to return synthetic
data, and the JAX memory_stats helpers are stubbed via the ``recovar.utils``
package re-export so the planner sees deterministic numbers.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_planner_globals(monkeypatch):
    """Force a fresh backend probe per test."""
    from recovar.utils import cuda_env

    monkeypatch.setattr(cuda_env, "_warned_typo", False)
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_CUDA_DISABLE", raising=False)


def _stub_helpers(monkeypatch, *, gpu_total: float = 80.0):
    from recovar import utils as _helpers

    monkeypatch.setattr(_helpers, "set_gpu_memory_limit", lambda gb: None)
    monkeypatch.setattr(_helpers, "get_gpu_memory_total", lambda: gpu_total)
    monkeypatch.setattr(
        _helpers,
        "get_image_batch_size",
        lambda grid_size, gpu_memory: max(1, int(gpu_memory * 256 / max(1, grid_size**2 / 1024.0))),
    )
    monkeypatch.setattr(
        _helpers,
        "get_vol_batch_size",
        lambda grid_size, gpu_memory: max(1, int(gpu_memory * 100 / max(1, grid_size**3 / 1e6))),
    )
    monkeypatch.setattr(
        _helpers,
        "get_column_batch_size",
        lambda grid_size, gpu_memory: max(1, int(gpu_memory * 200 / max(1, grid_size**3 / 1e6))),
    )


def _stub_preflight(monkeypatch, *, total: float, free: float):
    from recovar.utils import gpu_preflight as gp

    def fake(_idx=0):
        return gp.PhysicalGpuMemoryInfo(
            device_idx=0,
            total_gb=total,
            used_gb=total - free,
            free_gb=free,
            processes=[],
            source="stub",
        )

    monkeypatch.setattr(gp, "get_physical_gpu_memory_info", fake)


def _stub_backend_custom(monkeypatch):
    from recovar.utils import cuda_env

    monkeypatch.setattr(cuda_env, "detect_backend", lambda: "custom_cuda")


def test_make_memory_plan_returns_int_batches(monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=10000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
        desired_n_pcs=200,
    )
    assert plan.image_batch_size >= 1
    assert plan.volume_batch_size >= 1
    assert plan.column_batch_size >= 1
    assert plan.n_pcs_to_compute == 200
    assert plan.budget.effective_budget_gb > 0


def test_low_memory_tightens_batches(monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    base = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    low = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=True,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    very = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=True,
        adaptive_n_pcs=False,
    )
    assert very.image_batch_size <= low.image_batch_size <= base.image_batch_size
    assert very.volume_batch_size <= low.volume_batch_size <= base.volume_batch_size
    assert very.column_batch_size <= low.column_batch_size <= base.column_batch_size


def test_effective_budget_is_min_of_inputs_when_user_asked(monkeypatch):
    """With a user-supplied --gpu-budget-gb, physical_free-reserve is in the min."""
    _stub_helpers(monkeypatch, gpu_total=72.0)
    _stub_preflight(monkeypatch, total=80.0, free=10.0)  # only 10 GB free
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    # min(40, 72, 10 - 4_reserve=6) = 6 GB → effective should equal that
    assert plan.budget.effective_budget_gb == pytest.approx(10.0 - max(2.0, 0.05 * 80.0))
    assert plan.budget.source == "physical_free_minus_reserve"


def test_effective_budget_skips_physical_reserve_when_no_user_request(monkeypatch):
    """Without --gpu-budget-gb, fall through to jax_limit (matches legacy behavior).

    Subtracting physical_free-reserve unconditionally would shrink
    batches by ~5% on a quiet GPU and shift quality baselines (caught
    in long-test outliers regression).
    """
    _stub_helpers(monkeypatch, gpu_total=80.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)  # quiet GPU
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=None,  # no user request
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    # Legacy behavior preserved: effective == jax_limit (80), not 80-reserve.
    assert plan.budget.effective_budget_gb == pytest.approx(80.0)
    assert plan.budget.source == "jax_limit_gb"


def test_uncalibrated_with_adaptive_uses_formula_fallback(monkeypatch):
    """No calibration table + tight budget + --adaptive-n-pcs:
    planner now uses the covariance memory-estimate formula instead
    of returning a no-op. (Originally just returned desired_n_pcs;
    review caught that ``run_test_dataset --adaptive-n-pcs`` was a
    no-op when the calibration JSON wasn't committed.)"""
    _stub_helpers(monkeypatch, gpu_total=40.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    monkeypatch.setattr(mp, "load_calibration_table", lambda: None)
    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    # Status now reflects the formula path (40 GB at grid=128 won't fit
    # 200 PCs, peak ≈ 75 GB).
    assert plan.calibration_status == "uncalibrated_formula"
    assert plan.n_pcs_to_compute < 200
    assert plan.n_pcs_to_compute >= 1
    # Without a calibration table, no peak prediction.
    assert plan.predicted_peak_gb_total is None


def test_uncalibrated_without_adaptive_keeps_desired_n_pcs(monkeypatch):
    """No calibration table AND no --adaptive-n-pcs: keep desired_n_pcs.
    The formula fallback only kicks in when the user opted into adaptive."""
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    monkeypatch.setattr(mp, "load_calibration_table", lambda: None)
    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    assert plan.calibration_status == "uncalibrated"
    assert plan.n_pcs_to_compute == 200
    assert plan.predicted_peak_gb_total is None


def _make_calibration_table_for_pipeline():
    """Synthetic table with known peaks for adaptive-n_pcs tests."""
    from recovar.utils.memory_planner import CalibrationCell, CalibrationTable

    cells = [
        CalibrationCell(
            grid_size=128,
            backend="custom_cuda",
            n_pcs_or_volumes=4,
            peak_gb_total=5.0,
            peak_gb_by_phase={"after_mean": 5.0},
            status="ok",
        ),
        CalibrationCell(
            grid_size=128,
            backend="custom_cuda",
            n_pcs_or_volumes=20,
            peak_gb_total=10.0,
            peak_gb_by_phase={"after_mean": 10.0},
            status="ok",
        ),
        CalibrationCell(
            grid_size=128,
            backend="custom_cuda",
            n_pcs_or_volumes=50,
            peak_gb_total=20.0,
            peak_gb_by_phase={"after_mean": 20.0},
            status="ok",
        ),
        CalibrationCell(
            grid_size=128,
            backend="custom_cuda",
            n_pcs_or_volumes=200,
            peak_gb_total=75.0,
            peak_gb_by_phase={"after_mean": 75.0},
            status="ok",
        ),
    ]
    return CalibrationTable(
        schema_version=1,
        calibrated_on={"gpu_kind": "H100"},
        cells_by_command={"pipeline": cells},
    )


def test_adaptive_picks_largest_fitting_n_pcs(monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    table = _make_calibration_table_for_pipeline()
    monkeypatch.setattr(mp, "load_calibration_table", lambda: table)

    # Budget large enough for 50 (peak=20, with 1.2x slack=24 ≤ 30 ✓)
    # but not 200 (peak=75 with slack=90 > 30 ✗)
    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=30.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    assert plan.n_pcs_to_compute == 50

    # Tiny budget → falls back to the smallest calibrated n_pcs
    plan_small = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=2.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    assert plan_small.n_pcs_to_compute == 4

    # Huge budget AND huge physical_free AND huge jax_limit → keeps all
    # 200 PCs. effective_budget = min(requested, jax_limit, physical_free
    # - reserve) so we have to bump all three above the 200-PC peak (75
    # GB) plus the 1.2x slack (= 90 GB).
    _stub_helpers(monkeypatch, gpu_total=240.0)
    _stub_preflight(monkeypatch, total=240.0, free=240.0)
    plan_huge = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=200.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    assert plan_huge.n_pcs_to_compute == 200


def test_round_up_to_calibrated_grid(monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    table = _make_calibration_table_for_pipeline()
    monkeypatch.setattr(mp, "load_calibration_table", lambda: table)

    # Grid 100 has no exact entry; planner rounds up to 128
    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=100,
        n_images=1000,
        requested_gpu_gb=80.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
        desired_n_pcs=200,
    )
    assert plan.calibration_status == "rounded_up"
    assert plan.predicted_peak_gb_total == 75.0


def test_warning_when_predicted_exceeds_budget(monkeypatch, caplog):
    import logging

    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    table = _make_calibration_table_for_pipeline()
    monkeypatch.setattr(mp, "load_calibration_table", lambda: table)

    caplog.set_level(logging.WARNING, logger=mp.logger.name)
    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=20.0,  # tighter than 200 PC peak
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
        desired_n_pcs=200,
    )
    # We never refuse — plan still returned
    assert plan.n_pcs_to_compute == 200
    # And a clear warning landed
    assert any("exceeds the effective budget" in r.message.lower() or "exceeds" in r.message for r in caplog.records)


def test_memory_plan_json_round_trip(tmp_path, monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=40.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
    )
    out = mp.write_memory_plan_json(plan, tmp_path)
    assert out.exists()
    with out.open() as fh:
        data = json.load(fh)
    assert data["command"] == "pipeline"
    assert data["budget"]["effective_budget_gb"] == pytest.approx(plan.budget.effective_budget_gb)


def test_trace_writer_records_phase(tmp_path, monkeypatch):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    trace = mp.MemoryTraceWriter(tmp_path, enabled=True)
    trace.record("after_mean")
    trace.record("after_covariance")
    rows = [json.loads(line) for line in (tmp_path / "_diagnostics" / "memory_trace.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["after_mean", "after_covariance"]


def test_debug_force_peak_gb_overrides_observed(monkeypatch, tmp_path):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)
    monkeypatch.setenv("RECOVAR_DEBUG_FORCE_PEAK_GB", "999.0")

    from recovar.utils import memory_planner as mp

    trace = mp.MemoryTraceWriter(tmp_path, enabled=True)
    trace.record("after_mean")
    rows = [json.loads(line) for line in (tmp_path / "_diagnostics" / "memory_trace.jsonl").read_text().splitlines()]
    assert rows[0]["jax_peak_gb"] == 999.0
    assert rows[0]["jax_peak_gb_forced"] is True


# ---------------------------------------------------------------------------
# Review-feedback: uncalibrated --adaptive-n-pcs must NOT be a no-op.
# Without this fallback, ``run_test_dataset --adaptive-n-pcs`` on a
# small GPU still tries 200 PCs and OOMs — the exact scenario the
# wrapper exists to prevent.
# ---------------------------------------------------------------------------


def test_adaptive_fallback_picks_smaller_n_pcs_when_uncalibrated(monkeypatch):
    """No calibration table + tight budget → planner uses formula fallback."""
    _stub_helpers(monkeypatch, gpu_total=40.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    monkeypatch.setattr(mp, "load_calibration_table", lambda: None)

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=8.0,  # tight: 200 PCs predicts ~75 GB, won't fit
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    # Status reflects the formula path, not "uncalibrated_passthrough".
    assert plan.calibration_status == "uncalibrated_formula"
    # And it actually shrunk n_pcs.
    assert plan.n_pcs_to_compute < 200
    assert plan.n_pcs_to_compute >= 1


def test_adaptive_fallback_passthrough_when_budget_huge(monkeypatch):
    """No table + huge budget → keep desired_n_pcs."""
    _stub_helpers(monkeypatch, gpu_total=240.0)
    _stub_preflight(monkeypatch, total=240.0, free=240.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    monkeypatch.setattr(mp, "load_calibration_table", lambda: None)

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=64,
        n_images=1000,
        requested_gpu_gb=200.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=True,
    )
    assert plan.calibration_status == "uncalibrated_passthrough"
    assert plan.n_pcs_to_compute == 200


# NOTE: ``hard_cap_applied`` / ``hard_cap_skip_reason`` were removed
# along with the --gpu-budget-gb auto-cap mechanism. The flag is now a
# soft batch-size hint only; capping JAX is the user's responsibility
# via XLA_PYTHON_CLIENT_MEM_FRACTION / XLA_PYTHON_CLIENT_PREALLOCATE.
# Their replacement tests are in test_cli_memory_args.py
# (test_pipeline_parser_rejects_bogus_gpu_gb).


# ---------------------------------------------------------------------------
# Backend-aware budget deflation (sweep 8020210 finding).
# ---------------------------------------------------------------------------


def test_jax_fallback_deflates_legacy_budget(monkeypatch, tmp_path):
    """Saturation sweep (8020210) showed jax_fallback overshoots the user's
    budget by 1.4× at g=64 and OOMs at g=128 under legacy formulas.

    ``apply_memory_planning_args`` must deflate the value handed to
    ``set_gpu_memory_limit`` (which drives the legacy batch-size
    formulas) by a backend-specific divisor — 3.0 for jax_fallback,
    1.0 for custom_cuda.
    """
    import argparse

    from recovar.utils import helpers as _utils
    from recovar.utils import parser_args

    _stub_helpers(monkeypatch, gpu_total=80.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)

    captured = {}

    def fake_set_limit(value):
        captured["value"] = float(value)

    monkeypatch.setattr(_utils, "set_gpu_memory_limit", fake_set_limit)

    # Force backend to jax_fallback.
    from recovar.utils import cuda_env

    monkeypatch.setattr(cuda_env, "detect_backend", lambda: "jax_fallback")

    parser = argparse.ArgumentParser()
    parser_args.add_memory_planning_args(parser)
    args = parser.parse_args(["--gpu-budget-gb", "60"])

    parser_args.apply_memory_planning_args(
        args,
        command="pipeline",
        grid_size=128,
        n_images=1000,
        outdir=tmp_path,
    )
    # Effective budget = min(60, jax_limit=80, physical_free-reserve=74) = 60.
    # 60 GB / 3.0 = 20 GB handed to legacy formulas.
    assert captured["value"] == pytest.approx(20.0, rel=0.02)


def test_custom_cuda_does_not_deflate_legacy_budget(monkeypatch, tmp_path):
    """Under custom_cuda the planner deliberately does NOT touch the
    global ``set_gpu_memory_limit``. The saturation sweep showed the
    custom kernel comfortably fits in budget (ratios 0.09-0.72 across
    grids). Unconditionally calling set_gpu_memory_limit even with
    the same value shifts batch sizes off whatever JAX reports as
    bytes_limit, which is enough delta to flip the noise-floor
    cryo-ET outlier metrics in long-test (slurm 8025670 caught this).
    """
    import argparse

    from recovar.utils import helpers as _utils
    from recovar.utils import parser_args

    _stub_helpers(monkeypatch, gpu_total=80.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    captured = {}

    def fake_set_limit(value):
        captured["value"] = float(value)

    monkeypatch.setattr(_utils, "set_gpu_memory_limit", fake_set_limit)

    parser = argparse.ArgumentParser()
    parser_args.add_memory_planning_args(parser)
    args = parser.parse_args(["--gpu-budget-gb", "60"])

    parser_args.apply_memory_planning_args(
        args,
        command="pipeline",
        grid_size=128,
        n_images=1000,
        outdir=tmp_path,
    )
    # custom_cuda divisor = 1.0; planner must NOT touch the global.
    # (apply_gpu_memory_arg, called separately, can set it from
    # --gpu-budget-gb; but apply_memory_planning_args must stay a
    # no-op for the limit when no deflation is needed.)
    assert "value" not in captured


# ---------------------------------------------------------------------------
# Auto-trigger adaptive-n-pcs when basis term alone exceeds budget.
# ---------------------------------------------------------------------------


def test_auto_enables_adaptive_when_basis_exceeds_budget(monkeypatch):
    """g=256 + n_pcs=200 → basis term alone is ~27 GB. At budget=20 GB the
    run is guaranteed to OOM. Planner must auto-set adaptive without
    waiting for the user to pass --adaptive-n-pcs."""
    _stub_helpers(monkeypatch, gpu_total=20.0)
    _stub_preflight(monkeypatch, total=20.0, free=19.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=256,
        n_images=1000,
        requested_gpu_gb=20.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
        desired_n_pcs=200,
    )
    # Adaptive should fire and pick something well below 200.
    assert plan.n_pcs_to_compute < 200
    assert plan.n_pcs_to_compute >= 1


def test_does_not_auto_enable_adaptive_when_basis_fits(monkeypatch):
    """Basis term safely under 50% of budget: do NOT auto-enable adaptive
    (preserve user's explicit choice)."""
    _stub_helpers(monkeypatch, gpu_total=80.0)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)

    from recovar.utils import memory_planner as mp

    plan = mp.make_memory_plan(
        command="pipeline",
        grid_size=128,
        n_images=1000,
        requested_gpu_gb=80.0,
        low_memory=False,
        very_low_memory=False,
        adaptive_n_pcs=False,
        desired_n_pcs=200,
    )
    # basis = 200 × 128³ × 8 / 1e9 = 3.36 GB, way under 50% × 80 GB = 40 GB.
    # Adaptive must not fire; n_pcs stays at desired.
    assert plan.n_pcs_to_compute == 200
