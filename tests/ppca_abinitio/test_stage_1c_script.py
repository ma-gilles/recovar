"""Integration test for the Stage 1C script."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_stage_1c_factor_learning import (
    evaluate_stage_1c_relaxed,
    evaluate_stage_1c_strict,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_stage_1c_script_runs_and_passes_relaxed(tmp_path):
    out_path = tmp_path / "stage_1c.json"
    rc = main(
        [
            "--out",
            str(out_path),
            "--volume-size",
            "8",
            "--healpix-order",
            "0",
            "--max-shift",
            "1",
            "--q",
            "2",
            "--n-train",
            "64",
            "--n-val",
            "32",
            "--sigma-real",
            "0.2",
            "--eps-mu",
            "0.0",
            "--eps-U",
            "0.3",
            "--n-iters",
            "2",
            "--factor-lr",
            "1e-3",
            "--factor-inner-steps",
            "2",
            "--factor-k-max",
            "2.5",
            "--seeds",
            "0",
            "1",
            "2",
        ]
    )
    assert rc == 0, f"script returned {rc}, expected 0 (relaxed Stage 1C should pass)"

    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "1C"
    # 4 families × 3 seeds = 12 records
    assert len(payload["records"]) == 12
    families = {r["family"] for r in payload["records"]}
    assert families == {"A", "B", "C", "D"}

    relaxed = payload["exit_criterion"]
    assert relaxed["criterion"] == "relaxed_toy_size"
    assert relaxed["passed"] is True
    assert relaxed["family_B_no_nan"] is True
    assert relaxed["family_B_proj_improves"] is True
    assert relaxed["family_B_gauge_preserved"] is True
    assert relaxed["family_A_ok"] is True
    assert relaxed["family_C_ok"] is True
    assert relaxed["family_D_ok"] is True

    # Strict check is reported but not gating
    strict = payload["strict_check_for_reference"]
    assert strict["criterion"] == "strict"
    assert strict["heterogeneous_em_state_baseline_check"] == "unimplemented_in_v0"


def test_stage_1c_relaxed_pure_logic():
    pass_records = (
        [
            {
                "family": "B",
                "seed": s,
                "init_proj_err": 0.6,
                "final_proj_err": 0.4,
                "proj_improvement": 0.2,
                "gauge_err_at_final_iter": 1e-12,
                "any_nan_or_inf": False,
            }
            for s in (0, 1, 2)
        ]
        + [
            {
                "family": "A",
                "seed": s,
                "init_proj_err": 0.6,
                "final_proj_err": 0.6,
                "proj_improvement": 0.0,
                "gauge_err_at_final_iter": 1e-12,
                "any_nan_or_inf": False,
            }
            for s in (0, 1, 2)
        ]
        + [
            {
                "family": "C",
                "seed": s,
                "init_proj_err": 0.7,
                "final_proj_err": 0.5,
                "proj_improvement": 0.2,
                "gauge_err_at_final_iter": 1e-12,
                "any_nan_or_inf": False,
            }
            for s in (0, 1, 2)
        ]
        + [
            {
                "family": "D",
                "seed": s,
                "init_proj_err": 0.6,
                "final_proj_err": 0.5,
                "proj_improvement": 0.1,
                "gauge_err_at_final_iter": 1e-12,
                "any_nan_or_inf": False,
            }
            for s in (0, 1, 2)
        ]
    )
    ec = evaluate_stage_1c_relaxed(pass_records)
    assert ec["passed"] is True

    # Family B no improvement → fails
    bad_b = [{**r, "proj_improvement": -0.1} if r["family"] == "B" else r for r in pass_records]
    ec = evaluate_stage_1c_relaxed(bad_b)
    assert ec["passed"] is False
    assert ec["family_B_proj_improves"] is False

    # Family B NaN → fails
    nan_b = [{**r, "any_nan_or_inf": True} if r["family"] == "B" else r for r in pass_records]
    ec = evaluate_stage_1c_relaxed(nan_b)
    assert ec["passed"] is False

    # Gauge broken → fails
    bad_gauge = [{**r, "gauge_err_at_final_iter": 1e-3} if r["family"] == "B" else r for r in pass_records]
    ec = evaluate_stage_1c_relaxed(bad_gauge)
    assert ec["passed"] is False


def test_stage_1c_strict_is_currently_not_passable_without_baseline():
    """The strict check is not passable until HeterogeneousEMState
    baseline + family C/D specific clauses are wired up."""
    records = [
        {
            "family": "B",
            "seed": 0,
            "proj_improvement": 0.5,
        }
    ]
    ec = evaluate_stage_1c_strict(records)
    assert ec["passed"] is False
    assert ec["heterogeneous_em_state_baseline_check"] == "unimplemented_in_v0"
