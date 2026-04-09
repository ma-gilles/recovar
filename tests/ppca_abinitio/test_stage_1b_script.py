"""Integration test for the Stage 1B script.

Verifies that
`scripts/ppca_abinitio/run_stage_1b_residualized_mean.py` runs at
toy size and that the relaxed-at-toy-size exit criterion passes.
The strict criterion is not gated at toy size — it is reported in
the JSON for traceability and is expected to fail until realistic
data is wired in.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_stage_1b_residualized_mean import (
    evaluate_stage_1b_relaxed,
    evaluate_stage_1b_strict,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_stage_1b_script_runs_and_passes_relaxed(tmp_path):
    out_path = tmp_path / "stage_1b.json"
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
            "256",
            "--n-val",
            "64",
            "--sigma-real",
            "0.3",
            "--eps-mu",
            "0.5",
            "--n-iters",
            "3",
            "--seeds",
            "0",
            "1",
            "2",
        ]
    )
    assert rc == 0, f"script returned {rc}, expected 0 (relaxed check should pass)"

    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "1B"
    assert len(payload["records"]) == 6  # 2 families × 3 seeds

    # Relaxed criterion is the gating one (default) and must pass
    relaxed = payload["exit_criterion"]
    assert relaxed["criterion"] == "relaxed_toy_size"
    assert relaxed["passed"] is True
    assert relaxed["family_B_loop_sane"] is True
    assert relaxed["family_A_null_within_0.05"] is True

    # Strict criterion is reported but not gating
    assert payload["strict_check_for_reference"]["criterion"] == "strict"

    # Each record has the expected per-iter trajectory
    for r in payload["records"]:
        assert r["family"] in {"A", "B"}
        assert len(r["ppca_traj"]) == 4  # iter 0..3
        assert len(r["homog_traj"]) == 4
        for it, fre, mass in r["ppca_traj"]:
            assert isinstance(it, int)
            assert isinstance(fre, float)
            assert isinstance(mass, float)


def test_stage_1b_strict_check_pure_logic():
    """Direct unit test of the strict checker on hand-built records."""
    pass_records = [
        {
            "family": "B",
            "seed": s,
            "ppca_final_fre": 0.20,
            "homog_final_fre": 0.25,  # PPCA improves
            "ppca_init_mass": 0.05,
            "ppca_final_mass": 0.06,  # mass stable
        }
        for s in (0, 1, 2)
    ] + [
        {
            "family": "A",
            "seed": s,
            "ppca_final_fre": 0.30,
            "homog_final_fre": 0.30,  # null
            "ppca_init_mass": 0.01,
            "ppca_final_mass": 0.01,
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_stage_1b_strict(pass_records)
    assert ec["passed"] is True

    # Family B PPCA fails to improve → strict check fails
    fail_records = [
        {**r, "ppca_final_fre": 0.30, "homog_final_fre": 0.25} if r["family"] == "B" else r for r in pass_records
    ]
    ec = evaluate_stage_1b_strict(fail_records)
    assert ec["passed"] is False
    assert ec["family_B_fre_improves"] is False


def test_stage_1b_relaxed_check_pure_logic():
    """Direct unit test of the relaxed checker."""
    pass_records = [
        {
            "family": "B",
            "seed": s,
            "init_fre": 0.50,
            "ppca_best_fre": 0.30,  # 0.20 improvement > 0.05
            "ppca_traj": [(0, 0.50, 0.05), (1, 0.30, 0.05)],
            "homog_traj": [(0, 0.50, 0.05), (1, 0.31, 0.05)],
            "ppca_final_fre": 0.30,
            "homog_final_fre": 0.31,
        }
        for s in (0, 1, 2)
    ] + [
        {
            "family": "A",
            "seed": s,
            "init_fre": 0.50,
            "ppca_best_fre": 0.30,
            "ppca_traj": [(0, 0.50, 0.01), (1, 0.30, 0.01)],
            "homog_traj": [(0, 0.50, 0.01), (1, 0.30, 0.01)],
            "ppca_final_fre": 0.30,
            "homog_final_fre": 0.30,
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_stage_1b_relaxed(pass_records)
    assert ec["passed"] is True

    # PPCA loop diverges (best_fre > init_fre)
    diverge = [{**r, "ppca_best_fre": 0.60} if r["family"] == "B" else r for r in pass_records]
    ec = evaluate_stage_1b_relaxed(diverge)
    assert ec["passed"] is False
    assert ec["family_B_loop_sane"] is False
