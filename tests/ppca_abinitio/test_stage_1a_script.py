"""Integration test for the Stage 1A script.

Verifies that
`scripts/ppca_abinitio/run_stage_1a_factor_perturbation.py` runs at
toy size and that the truth-perturbed-family-B exit clause passes.
The random-lowpass case is informational only and is not gated.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_stage_1a_factor_perturbation import (
    evaluate_stage_1a_exit_criterion,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_stage_1a_script_runs_and_passes_at_toy_size(tmp_path):
    out_path = tmp_path / "stage_1a.json"
    rc = main(
        [
            "--out",
            str(out_path),
            "--volume-size",
            "8",
            "--healpix-order",
            "1",
            "--max-shift",
            "1",
            "--q",
            "2",
            "--n-train",
            "32",
            "--n-val",
            "128",
            "--sigma-real",
            "0.2",
            "--eps-mu",
            "0.1",
            "--eps-U",
            "0.1",
            "--seeds",
            "0",
            "1",
            "2",
            "--n-bootstrap",
            "200",
        ]
    )
    assert rc == 0, f"script returned {rc}, expected 0 (Stage 1A truth-perturbed should pass)"

    payload = json.loads(out_path.read_text())
    assert payload["stage"] == "1A"

    # 2 families × 2 init_kinds × 3 seeds = 12 records
    assert len(payload["records"]) == 12
    init_kinds = {r["init_kind"] for r in payload["records"]}
    assert init_kinds == {"truth_perturbed", "random_lowpass"}

    ec = payload["exit_criterion"]
    assert ec["passed"] is True
    assert ec["truth_perturbed_family_B"]["all_seeds_pass"] is True
    for entry in ec["truth_perturbed_family_B"]["per_seed"]:
        assert entry["ok"] is True
        assert entry["ci_low"] > 0

    # Random lowpass is reported but not gating — just check the
    # field exists and is marked informational
    assert ec["random_lowpass_family_B"]["informational_only"] is True
    assert len(ec["random_lowpass_family_B"]["per_seed"]) == 3


def test_evaluate_stage_1a_only_gates_truth_perturbed():
    """Direct unit test of the exit-criterion checker."""
    pass_truth = [
        {
            "family": "B",
            "seed": s,
            "init_kind": "truth_perturbed",
            "delta": {"mean": 0.05, "ci_low": 0.02, "ci_high": 0.08},
        }
        for s in (0, 1, 2)
    ]
    # Random lowpass fails — should not block
    fail_random = [
        {
            "family": "B",
            "seed": s,
            "init_kind": "random_lowpass",
            "delta": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.0},
        }
        for s in (0, 1, 2)
    ]
    ec = evaluate_stage_1a_exit_criterion(pass_truth + fail_random)
    assert ec["passed"] is True

    # Truth-perturbed fails on one seed — should block
    one_bad = pass_truth.copy()
    one_bad[1] = {
        **one_bad[1],
        "delta": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.0},
    }
    ec = evaluate_stage_1a_exit_criterion(one_bad + fail_random)
    assert ec["passed"] is False

    # No truth-perturbed records → not passed
    only_random = fail_random
    ec = evaluate_stage_1a_exit_criterion(only_random)
    assert ec["passed"] is False
