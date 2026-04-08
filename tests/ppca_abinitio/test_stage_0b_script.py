"""Integration test for `scripts/ppca_abinitio/run_stage_0b_oracle_score.py`.

Per spec Section 13.2, stage gates are scripts that emit JSON
summaries. A lightweight integration-marked test verifies that the
script *runs* and produces the expected keys. This test goes one
step further: it runs the script in-process at a small but
realistic size and verifies that the Stage 0B exit criterion
actually evaluates to `passed = True`.

If this test starts failing, either:

  - the kernel is producing wrong scores (regression in posterior.py
    or half_volume.py), or
  - the synthetic harness is no longer producing data with the
    expected heterogeneity-to-noise ratio, or
  - the metrics aggregation or bootstrap CI computation is broken.

In any case, look at the per-(family, seed) records in the script
output to localize the regression.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_stage_0b_oracle_score import (
    evaluate_stage_0b_exit_criterion,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_stage_0b_script_runs_and_passes_at_toy_size(tmp_path):
    out_path = tmp_path / "stage_0b.json"
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
            "--seeds",
            "0",
            "1",
            "2",
            "--n-bootstrap",
            "300",
        ]
    )
    assert rc == 0, f"script returned {rc}, expected 0 (Stage 0B should pass)"

    payload = json.loads(out_path.read_text())

    # Top-level structure
    assert payload["stage"] == "0B"
    assert "config" in payload
    assert "records" in payload
    assert "exit_criterion" in payload

    # 2 families × 3 seeds = 6 records
    assert len(payload["records"]) == 6
    families = {r["family"] for r in payload["records"]}
    assert families == {"A", "B"}
    seeds = sorted({r["seed"] for r in payload["records"]})
    assert seeds == [0, 1, 2]

    # Each record has the expected sub-keys
    for r in payload["records"]:
        assert set(r.keys()) >= {"family", "seed", "n_val", "homog", "ppca", "delta"}
        for key in ("homog", "ppca", "delta"):
            assert set(r[key].keys()) == {"mean", "ci_low", "ci_high", "level"}

    # Exit criterion
    ec = payload["exit_criterion"]
    assert ec["passed"] is True
    assert ec["family_B"]["all_seeds_pass"] is True
    assert ec["family_A"]["all_seeds_pass"] is True
    for entry in ec["family_B"]["per_seed"]:
        assert entry["ok"] is True
        assert entry["ci_low"] > 0
    for entry in ec["family_A"]["per_seed"]:
        assert entry["ok"] is True


def test_evaluate_stage_0b_exit_criterion_handles_failure_modes():
    """Direct unit test of the exit-criterion checker with hand-built records."""
    # Family B fails: CI low is negative
    fail_b = [
        {"family": "B", "seed": 0, "delta": {"mean": 0.001, "ci_low": -0.01, "ci_high": 0.012}},
        {"family": "A", "seed": 0, "delta": {"mean": 0.0, "ci_low": -0.001, "ci_high": 0.001}},
    ]
    ec = evaluate_stage_0b_exit_criterion(fail_b)
    assert ec["passed"] is False
    assert ec["family_B"]["all_seeds_pass"] is False
    assert ec["family_A"]["all_seeds_pass"] is True

    # Family A fails: |delta_mean| > 0.01
    fail_a = [
        {"family": "B", "seed": 0, "delta": {"mean": 0.05, "ci_low": 0.04, "ci_high": 0.06}},
        {"family": "A", "seed": 0, "delta": {"mean": 0.05, "ci_low": 0.04, "ci_high": 0.06}},
    ]
    ec = evaluate_stage_0b_exit_criterion(fail_a)
    assert ec["passed"] is False
    assert ec["family_B"]["all_seeds_pass"] is True
    assert ec["family_A"]["all_seeds_pass"] is False

    # Both pass
    pass_both = [
        {"family": "B", "seed": s, "delta": {"mean": 0.05, "ci_low": 0.02, "ci_high": 0.08}} for s in (0, 1, 2)
    ] + [{"family": "A", "seed": s, "delta": {"mean": 1e-6, "ci_low": -1e-5, "ci_high": 1e-5}} for s in (0, 1, 2)]
    ec = evaluate_stage_0b_exit_criterion(pass_both)
    assert ec["passed"] is True

    # No family B records → not passed (need at least one of each)
    only_a = [{"family": "A", "seed": 0, "delta": {"mean": 0.0, "ci_low": -1e-5, "ci_high": 1e-5}}]
    ec = evaluate_stage_0b_exit_criterion(only_a)
    assert ec["passed"] is False
