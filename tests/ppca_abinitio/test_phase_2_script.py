"""Integration test for the Phase 2 script."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_phase_2_external_mean_bootstrap import (
    evaluate_phase_2_relaxed,
    evaluate_phase_2_strict,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_phase_2_script_runs_and_passes_relaxed(tmp_path):
    out_path = tmp_path / "phase_2.json"
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
            "128",
            "--n-val",
            "32",
            "--sigma-real",
            "0.2",
            "--n-iters",
            "2",
            "--factor-lr",
            "1e-3",
            "--seeds",
            "0",
            "1",
            "2",
        ]
    )
    assert rc == 0, f"script returned {rc}"

    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "Phase 2"
    assert len(payload["records"]) == 3

    relaxed = payload["exit_criterion"]
    assert relaxed["passed"] is True
    assert relaxed["family_B_no_nan"] is True
    assert relaxed["family_B_gauge_preserved"] is True
    assert relaxed["family_B_proj_no_catastrophic_degradation"] is True
    assert relaxed["family_B_fre_no_catastrophic_degradation"] is True

    strict = payload["strict_check_for_reference"]
    assert strict["criterion"] == "strict"
    assert "external_mean_implementation_note" in strict


def test_phase_2_relaxed_pure_logic():
    pass_records = [
        {
            "family": "B",
            "seed": s,
            "init_proj_err": 2.0,
            "final_proj_err": 1.99,
            "init_fre_mu": 0.14,
            "final_fre_mu": 0.30,
            "gauge_err_at_final_iter": 1e-12,
            "any_nan_or_inf": False,
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_phase_2_relaxed(pass_records)
    assert ec["passed"] is True

    # Catastrophic FRE degradation
    bad = [{**r, "final_fre_mu": 0.6} for r in pass_records]
    ec = evaluate_phase_2_relaxed(bad)
    assert ec["passed"] is False
    assert ec["family_B_fre_no_catastrophic_degradation"] is False

    # Gauge broken
    bad_gauge = [{**r, "gauge_err_at_final_iter": 1e-3} for r in pass_records]
    ec = evaluate_phase_2_relaxed(bad_gauge)
    assert ec["passed"] is False

    # NaN
    nan_rec = [{**r, "any_nan_or_inf": True} for r in pass_records]
    ec = evaluate_phase_2_relaxed(nan_rec)
    assert ec["passed"] is False


def test_phase_2_strict_is_documented_as_toy_limited():
    """Direct unit test of the strict checker. At v0 toy size the
    strict criterion is reported as `not passed` because the
    low-passed external mean stand-in puts the loop in a regime
    that fails the strict gates."""
    pass_records = [
        {
            "family": "B",
            "seed": s,
            "init_proj_err": 0.6,
            "final_proj_err": 0.4,
            "proj_improvement": 0.2,
            "init_fre_mu": 0.5,
            "final_fre_mu": 0.3,
            "fre_improvement": 0.2,
            "ppca_minus_baseline_proj_err": -0.5,
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_phase_2_strict(pass_records)
    assert ec["passed"] is True
    assert ec["family_B_proj_improves_all_seeds"] is True

    fail = [{**r, "fre_improvement": -0.1} for r in pass_records]
    ec = evaluate_phase_2_strict(fail)
    assert ec["passed"] is False
