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
    """With a user-supplied --gpu-gb, physical_free-reserve is in the min."""
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
    """Without --gpu-gb, fall through to jax_limit (matches legacy behavior).

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


def test_uncalibrated_falls_back_gracefully(monkeypatch):
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
        adaptive_n_pcs=True,  # adaptive is requested but no table
    )
    assert plan.calibration_status == "uncalibrated"
    assert plan.n_pcs_to_compute == 200  # falls back to desired
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
    rows = [json.loads(line) for line in (tmp_path / "memory_trace.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["after_mean", "after_covariance"]


def test_debug_force_peak_gb_overrides_observed(monkeypatch, tmp_path):
    _stub_helpers(monkeypatch)
    _stub_preflight(monkeypatch, total=80.0, free=78.0)
    _stub_backend_custom(monkeypatch)
    monkeypatch.setenv("RECOVAR_DEBUG_FORCE_PEAK_GB", "999.0")

    from recovar.utils import memory_planner as mp

    trace = mp.MemoryTraceWriter(tmp_path, enabled=True)
    trace.record("after_mean")
    rows = [json.loads(line) for line in (tmp_path / "memory_trace.jsonl").read_text().splitlines()]
    assert rows[0]["jax_peak_gb"] == 999.0
    assert rows[0]["jax_peak_gb_forced"] is True
