"""Integration test for the Stage 1D script."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_stage_1d_full_soft_mstep import (
    evaluate_stage_1d_relaxed,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_stage_1d_script_runs_and_passes_relaxed(tmp_path):
    out_path = tmp_path / "stage_1d.json"
    rc = main(
        [
            "--out",
            str(out_path),
            "--volume-size",
            "8",
            "--healpix-order",
            "0",
            "--q",
            "2",
            "--n-train",
            "128",
            "--n-val",
            "32",
            "--sigma-real",
            "0.2",
            "--eps-U",
            "0.3",
            "--seeds",
            "0",
            "1",
            "2",
        ]
    )
    assert rc == 0, f"script returned {rc}"

    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "1D"
    assert len(payload["records"]) == 3

    relaxed = payload["exit_criterion"]
    assert relaxed["passed"] is True
    assert relaxed["family_B_no_nan"] is True
    assert relaxed["family_B_loss_monotone"] is True
    assert relaxed["family_B_not_worse_than_1c"] is True
    assert relaxed["family_B_gauge_preserved"] is True

    # Each record should have both 1C and 1D results
    for r in payload["records"]:
        assert "stage_1c_proj_err" in r
        assert "stage_1d_proj_err" in r
        assert "ecm_n_inner_steps" in r
        assert "ecm_loss_monotone" in r


def test_stage_1d_relaxed_pure_logic():
    pass_records = [
        {
            "family": "B",
            "seed": s,
            "stage_1c_proj_err": 0.45,
            "stage_1d_proj_err": 0.44,
            "stage_1d_minus_1c_proj_err": -0.01,
            "ecm_loss_monotone": True,
            "ecm_gauge_err": 1e-12,
            "ecm_any_nan_or_inf": False,
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_stage_1d_relaxed(pass_records)
    assert ec["passed"] is True

    # 1D much worse → fail
    bad = [{**r, "stage_1d_minus_1c_proj_err": 0.5} for r in pass_records]
    ec = evaluate_stage_1d_relaxed(bad)
    assert ec["passed"] is False

    # Loss not monotone → fail
    bad_mono = [{**r, "ecm_loss_monotone": False} for r in pass_records]
    ec = evaluate_stage_1d_relaxed(bad_mono)
    assert ec["passed"] is False

    # NaN
    nan_rec = [{**r, "ecm_any_nan_or_inf": True} for r in pass_records]
    ec = evaluate_stage_1d_relaxed(nan_rec)
    assert ec["passed"] is False
