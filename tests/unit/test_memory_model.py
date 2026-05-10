"""Unit tests for ``recovar.utils.memory_model``.

The tests assert *structural* properties of the model (monotonicity,
backend ordering, mode ordering, term breakdown) rather than specific
numeric values. The numeric values come from the validation sweep
(``scripts/validate_memory_formulas.py``); pinning them here would
just duplicate the sweep's record-mode output and rot if anyone
re-fits constants.
"""

from __future__ import annotations

import pytest

from recovar.utils import memory_model as mm

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Structural: dataclass round-trip
# ---------------------------------------------------------------------------


def test_breakdown_serializes_with_code_refs():
    bd = mm.spa_covariance_memory(grid_size=128, n_pcs=200)
    d = bd.to_dict()
    assert d["pipeline"] == "spa"
    assert d["phase"] == "covariance"
    assert "basis" in d["terms_gb"]
    assert "svd_workspace" in d["terms_gb"]
    assert "basis" in d["code_refs"]
    assert d["total_gb"] == sum(d["terms_gb"].values())


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_image_batch_monotone_in_budget():
    sizes = [mm.pick_image_batch(grid_size=128, budget_gb=b, backend="custom_cuda") for b in (4, 8, 16, 32, 64)]
    assert sizes == sorted(sizes)


def test_image_batch_decreases_with_grid():
    sizes = [mm.pick_image_batch(grid_size=g, budget_gb=40, backend="custom_cuda") for g in (64, 128, 256)]
    # Bigger grid → smaller batch
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_volume_batch_decreases_with_grid_cubically_ish():
    sizes = [mm.pick_volume_batch(grid_size=g, budget_gb=40, backend="custom_cuda") for g in (64, 128, 256)]
    # Volume scales as grid³, so batch should drop ~8× per doubling.
    assert sizes[0] > sizes[1] > sizes[2]


def test_n_pcs_monotone_in_budget():
    chosen = []
    for b in (8, 16, 32, 64, 128):
        n, _ = mm.pick_n_pcs(grid_size=128, budget_gb=b, backend="custom_cuda")
        chosen.append(n)
    assert chosen == sorted(chosen)
    assert chosen[-1] >= chosen[0]


# ---------------------------------------------------------------------------
# Backend ordering: jax_fallback should NEVER pick larger batches than
# custom_cuda at the same (grid, budget)
# ---------------------------------------------------------------------------


def test_jax_fallback_strictly_more_conservative():
    for grid in (64, 128, 256):
        for budget in (8, 24, 80):
            cust = mm.pick_image_batch(grid, budget, "custom_cuda")
            jax_ = mm.pick_image_batch(grid, budget, "jax_fallback")
            assert jax_ <= cust, f"jax_fallback={jax_} > custom_cuda={cust} at grid={grid}, budget={budget}"


# ---------------------------------------------------------------------------
# Memory mode ordering: very_low < low < default
# ---------------------------------------------------------------------------


def test_low_mode_more_conservative_than_default():
    default = mm.pick_image_batch(128, 40, "custom_cuda", mode="default")
    low = mm.pick_image_batch(128, 40, "custom_cuda", mode="low")
    assert low < default


def test_very_low_mode_more_conservative_than_low():
    low = mm.pick_image_batch(128, 40, "custom_cuda", mode="low")
    very_low = mm.pick_image_batch(128, 40, "custom_cuda", mode="very_low")
    assert very_low < low


# ---------------------------------------------------------------------------
# SPA / ET use distinct functions
# ---------------------------------------------------------------------------


def test_spa_and_tilt_series_distinct_functions():
    """SPA and ET memory functions are separate (per the reviewer's
    point that ET differs in batching, CTF, and tilt grouping —
    do not collapse to a single multiplier without sweep evidence)."""
    assert mm.spa_mean_memory is not mm.tilt_series_mean_memory
    assert mm.spa_covariance_memory is not mm.tilt_series_covariance_memory
    assert mm.spa_embedding_memory is not mm.tilt_series_embedding_memory


def test_dispatch_function_routes_correctly():
    spa = mm.memory_breakdown_for_phase(
        "spa",
        "covariance",
        grid_size=128,
        n_pcs=100,
    )
    et = mm.memory_breakdown_for_phase(
        "tilt_series",
        "covariance",
        grid_size=128,
        n_pcs=100,
    )
    assert spa.pipeline == "spa"
    assert et.pipeline == "tilt_series"


# ---------------------------------------------------------------------------
# Issue #135 framing: don't pin "75 GB at 200 PCs" as the truth.
# Pin "the prediction is in the same order as the legacy claim,
# but the actual peak is whatever the sweep observes." Until the
# sweep runs, we just check the formula returns something
# in a reasonable range so we know the placeholder is sane.
# ---------------------------------------------------------------------------


def test_pinned_issue_135_config_in_reasonable_range():
    """Issue #135's reporter saw a 75 GB warning for 200 PCs at
    grid=128 on a 48 GB GPU, but the run completed once
    PREALLOCATE=false was set — so the actual peak was at most 48
    GB.

    Discovery sweep (slurm 7982854, 2026-05-10) confirmed the legacy
    75-GB-at-200-PCs estimate is wrong: observed peak at grid=128
    custom_cuda is ~40 GB regardless of n_pcs in [20, 200]. The
    SVD-workspace term has effectively no n_pcs dependence in this
    regime (fit exponent = -0.03, R² = 0.908). Updated constants
    set ``SVD_WORKSPACE_COEF_GB = 0.0`` so the prediction equals
    just the basis term (small).

    This test now pins:
      - basis term > 0 (always present)
      - total_gb stays in a sane range (don't crash with negatives,
        don't return absurdly high)
    Validation sweep at grid={64, 128, 256} will refine grid scaling.
    """
    bd = mm.spa_covariance_memory(grid_size=128, n_pcs=200)
    assert 0.0 < bd.total_gb < 200.0
    # basis is the only non-zero term right now (svd_workspace=0)
    assert bd.terms_gb["basis"] > 0
    # svd_workspace can be 0 post-discovery (observed: peak doesn't
    # scale with n_pcs at grid=128).
    assert bd.terms_gb["svd_workspace"] >= 0


# ---------------------------------------------------------------------------
# Diagnostics layout (Phase 5)
# ---------------------------------------------------------------------------


def test_diagnostics_dir_helper(tmp_path):
    from recovar.utils.memory_planner import diagnostics_dir

    d = diagnostics_dir(tmp_path)
    assert d.is_dir()
    assert d.name == "_diagnostics"
    assert d.parent == tmp_path


def test_diagnostics_files_written(tmp_path):
    """Smoke test that ``apply_memory_planning_args`` writes all the
    always-on diagnostic files into _diagnostics/."""
    import argparse
    import json

    from recovar.utils import parser_args

    parser = argparse.ArgumentParser()
    parser_args.add_memory_planning_args(parser)
    args = parser.parse_args(["--gpu-budget-gb", "40"])
    plan, trace = parser_args.apply_memory_planning_args(
        args,
        command="pipeline",
        grid_size=128,
        n_images=1000,
        outdir=tmp_path,
    )
    diag = tmp_path / "_diagnostics"
    assert (diag / "memory_plan.json").is_file()
    assert (diag / "args.json").is_file()
    assert (diag / "allocator_env.json").is_file()
    # memory_trace.jsonl is created (truncated) at planner construction.
    assert (diag / "memory_trace.jsonl").is_file()

    # allocator_env captures the relevant XLA env vars
    env = json.loads((diag / "allocator_env.json").read_text())
    for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        assert key in env

    # args captures the canonical flag
    args_data = json.loads((diag / "args.json").read_text())
    assert args_data["gpu_memory"] == 40.0


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_pick_n_pcs_returns_breakdown_at_chosen_value():
    n, bd = mm.pick_n_pcs(grid_size=128, budget_gb=24, backend="custom_cuda")
    assert n >= 1
    assert bd is not None
    assert bd.phase == "covariance"
    assert "svd_workspace" in bd.terms_gb
