"""Unit tests for scripts/parity/check_perf.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "parity" / "check_perf.py"


def _make_baseline(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "fixture": "test",
                "per_iter_seconds_total": {"4": 100.0, "5": 100.0, "6": 100.0},
                "per_iter_seconds_stages": {
                    "4": {"e_step": 80.0, "recon": 5.0},
                    "5": {"e_step": 80.0, "recon": 5.0},
                    "6": {"e_step": 80.0, "recon": 5.0},
                },
                "tolerance_multiplier": 2.0,
                "regression_threshold_multiplier": 3.0,
            }
        )
    )


def _write_iter(dump_dir: Path, iter_num: int, wall: float, stages: dict[str, float]) -> None:
    payload = {"wall_time_s": np.float64(wall)}
    for k, v in stages.items():
        payload[f"stage_seconds_{k}"] = np.float64(v)
    np.savez_compressed(dump_dir / f"iter_{iter_num:03d}.npz", **payload)


def _run(dump_dir: Path, baseline: Path, *extra: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dump-dir", str(dump_dir), "--baseline", str(baseline), *extra],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout


def test_ok_warn_regressed_classification(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    baseline = tmp_path / "baseline.json"
    _make_baseline(baseline)

    _write_iter(dump, 4, wall=120.0, stages={"e_step": 100.0, "recon": 5.0})  # +20% OK
    _write_iter(dump, 5, wall=250.0, stages={"e_step": 220.0, "recon": 6.0})  # 2.5x WARN
    _write_iter(dump, 6, wall=400.0, stages={"e_step": 380.0, "recon": 6.0})  # 4x REGRESSED

    rc, out = _run(dump, baseline)
    assert rc == 0
    assert "iter 4" in out and "OK" in out.split("iter 4:")[1].split("\n")[0]
    assert "iter 5" in out and "WARN" in out.split("iter 5:")[1].split("\n")[0]
    assert "iter 6" in out and "REGRESSED" in out.split("iter 6:")[1].split("\n")[0]


def test_exit_code_on_regression(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    baseline = tmp_path / "baseline.json"
    _make_baseline(baseline)

    _write_iter(dump, 4, wall=120.0, stages={"e_step": 100.0, "recon": 5.0})  # OK
    _write_iter(dump, 6, wall=400.0, stages={"e_step": 380.0, "recon": 6.0})  # REGRESSED

    rc_no_flag, _ = _run(dump, baseline)
    rc_flag, _ = _run(dump, baseline, "--exit-code-on-regression")
    assert rc_no_flag == 0
    assert rc_flag == 2


def test_single_iter_not_yet_dumped(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    baseline = tmp_path / "baseline.json"
    _make_baseline(baseline)

    rc, out = _run(dump, baseline, "--single-iter", "5")
    assert rc == 0
    assert "iter 5: not yet dumped" in out


def test_single_iter_only_checks_requested_iter(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    baseline = tmp_path / "baseline.json"
    _make_baseline(baseline)

    _write_iter(dump, 4, wall=400.0, stages={"e_step": 380.0, "recon": 6.0})  # REGRESSED
    _write_iter(dump, 5, wall=120.0, stages={"e_step": 100.0, "recon": 5.0})  # OK

    rc, out = _run(dump, baseline, "--single-iter", "5", "--exit-code-on-regression")
    # iter 4 is REGRESSED but NOT requested → rc should still be 0.
    assert rc == 0
    assert "iter 5" in out
    assert "iter 4" not in out
