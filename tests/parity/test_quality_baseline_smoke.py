"""Fast smoke tests for the parity quality baseline tooling.

These tests run on the login node (no GPU required, no workload) and verify
that the baseline JSON parses, the checker imports, and the schema is sane.
The full workload-based regression test is in
``test_quality_baseline.py`` and requires GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO / "tests/baselines/parity/quality_baseline_5k_128.json"
CHECK_PARITY = REPO / "scripts/parity/check_parity.py"


@pytest.mark.parity
def test_baseline_json_parses() -> None:
    assert BASELINE_PATH.exists(), f"baseline JSON not found at {BASELINE_PATH}"
    data = json.loads(BASELINE_PATH.read_text())
    assert "fixture" in data
    assert "scenarios" in data
    assert isinstance(data["scenarios"], dict)
    assert len(data["scenarios"]) >= 1, "at least one scenario must be defined"


@pytest.mark.parity
def test_baseline_scenarios_have_required_fields() -> None:
    data = json.loads(BASELINE_PATH.read_text())
    for name, scen in data["scenarios"].items():
        assert "config" in scen, f"scenario {name} missing 'config'"
        assert "expected_metrics" in scen, f"scenario {name} missing 'expected_metrics'"
        cfg = scen["config"]
        for key in ("init_iter", "max_iter", "local_engine"):
            assert key in cfg, f"scenario {name} config missing '{key}'"


@pytest.mark.parity
def test_check_parity_help_works() -> None:
    """The checker script should at least parse args and print help cleanly."""
    proc = subprocess.run(
        [sys.executable, str(CHECK_PARITY), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"check_parity --help failed:\n{proc.stderr}"
    assert "baseline" in proc.stdout
    assert "scenario" in proc.stdout


@pytest.mark.parity
def test_check_parity_extract_metrics_from_synthetic_dump(tmp_path: Path) -> None:
    """Verify the metric extraction code path runs end-to-end on synthetic data.

    Builds two NPZ dumps with the schema fields the checker expects, then
    invokes the checker and confirms it exits cleanly with no crashes. The
    actual numeric comparison uses a wide-tolerance baseline so this test is
    insensitive to the exact synthetic values.
    """
    rec_path = tmp_path / "rec.npz"
    rel_path = tmp_path / "rel.npz"
    n = 100  # small per-particle counts
    rec = {
        "iteration": np.int32(0),
        "init_relion_iteration": np.int32(0),
        "relion_iteration": np.int32(1),
        "current_size": np.int32(56),
        "sigma_offset": np.float64(2.5),
        "translation_step": np.float64(1.0),
        "translation_range": np.float64(3.0),
        "tau2_fudge": np.float64(1.0),
        "voxel_size": np.float64(4.25),
        "grid_size": np.int32(128),
        "ave_pmax": np.float64(0.5),
        "fsc": np.zeros(64, dtype=np.float64),
        "sigma2_noise": np.ones(64, dtype=np.float64),
        "wall_time_s": np.float64(120.0),
        "half1_max_posterior": np.full(n, 0.5, dtype=np.float32),
        "half1_best_eulers_total": np.zeros((n, 3), dtype=np.float32),
        "half1_best_translations_total": np.zeros((n, 2), dtype=np.float32),
        "half1_wsum_sigma2_noise": np.ones(64, dtype=np.float64),
        "half1_mean_real_ds": np.random.default_rng(0).standard_normal(64**3).astype(np.float32),
        "half2_max_posterior": np.full(n, 0.5, dtype=np.float32),
        "half2_best_eulers_total": np.zeros((n, 3), dtype=np.float32),
        "half2_best_translations_total": np.zeros((n, 2), dtype=np.float32),
        "half2_wsum_sigma2_noise": np.ones(64, dtype=np.float64),
        "half2_mean_real_ds": np.random.default_rng(1).standard_normal(64**3).astype(np.float32),
    }
    rel = {
        "iteration": np.int32(1),
        "current_image_size": np.int32(56),
        "ave_pmax_model": np.float64(0.5),
        "sigma_offset": np.float64(2.5),
        "particle_eulers": np.zeros((2 * n, 3), dtype=np.float64),
        "particle_max_pmax": np.full(2 * n, 0.5, dtype=np.float32),
        "half1_sigma2_noise": np.ones(64, dtype=np.float64),
        "half2_sigma2_noise": np.ones(64, dtype=np.float64),
        "half1_mean_real_ds": rec["half1_mean_real_ds"].copy(),
        "half2_mean_real_ds": rec["half2_mean_real_ds"].copy(),
    }
    np.savez_compressed(rec_path, **rec)
    np.savez_compressed(rel_path, **rel)

    # Build a wide-tolerance baseline JSON in tmp
    baseline = {
        "fixture": "synthetic",
        "fixture_path": str(tmp_path),
        "relion_reference_dir": str(tmp_path),
        "scenarios": {
            "synthetic_smoke": {
                "config": {"init_iter": 0, "max_iter": 1, "local_engine": "exact_v1"},
                "expected_metrics": {
                    "ave_pmax": 0.5,
                    "ave_pmax_tolerance": 0.5,
                    "ave_pmax_regression_threshold": 1.0,
                    "vol_corr_half1_floor": 0.5,
                    "vol_corr_half2_floor": 0.5,
                    "sigma_offset_a": 2.5,
                    "sigma_offset_a_tolerance": 1.0,
                },
                "wall_time_s_baseline": 120.0,
                "wall_time_s_regression_threshold_multiplier": 100.0,
            }
        },
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))

    proc = subprocess.run(
        [
            sys.executable,
            str(CHECK_PARITY),
            "--baseline",
            str(baseline_path),
            "--scenario",
            "synthetic_smoke",
            "--recovar-dump",
            str(rec_path),
            "--relion-dump",
            str(rel_path),
            "--exit-code-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert "synthetic_smoke" in proc.stdout
    assert "ave_pmax" in proc.stdout
    assert "vol_corr_half1: 1.000000" in proc.stdout, "expected perfect self-correlation"
    assert proc.returncode == 0, f"check_parity should pass on synthetic dump\n{proc.stdout}\n{proc.stderr}"
