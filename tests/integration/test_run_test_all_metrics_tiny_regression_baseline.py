import json
import os
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from helpers.metrics_regression import metric_direction


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io]


def _make_real_volume(idx, n_vols, grid):
    x = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    t = 2.0 * np.pi * idx / max(n_vols, 1)
    vol = (
        np.exp(-((xx - 0.3 * np.cos(t)) ** 2 + (yy - 0.25 * np.sin(t)) ** 2 + (zz - 0.2 * np.cos(2 * t)) ** 2) / (2 * 0.18**2))
        + 0.7
        * np.exp(-((xx + 0.25 * np.sin(1.3 * t)) ** 2 + (yy - 0.2 * np.cos(1.1 * t)) ** 2 + (zz + 0.2 * np.sin(t)) ** 2) / (2 * 0.16**2))
    )
    vol = vol.astype(np.float32)
    vol -= vol.mean()
    denom = np.linalg.norm(vol.ravel())
    if denom > 0:
        vol /= denom
    return vol


def _write_volumes(prefix: Path, n_vols=12, grid=32, voxel_size=4.25):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n_vols):
        with mrcfile.new(f"{prefix}{i:04d}.mrc", overwrite=True) as m:
            m.set_data(_make_real_volume(i, n_vols, grid))
            m.voxel_size = voxel_size


def _run_with_baseline(
    *,
    vols_prefix: Path,
    out_dir: Path,
    baseline_json: Path,
    tol_frac: float,
    overwrite_baseline: bool,
):
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_all_metrics",
        "--volume-input",
        str(vols_prefix),
        "--output-dir",
        str(out_dir),
        "--grid-size",
        "32",
        "--n-images",
        "800",
        "--noise-level",
        "1.0",
        "--contrast-std",
        "0.1",
        "--metrics-baseline-json",
        str(baseline_json),
        "--metrics-regression-tol-frac",
        f"{tol_frac}",
    ]
    if overwrite_baseline:
        cmd.append("--overwrite-metrics-baseline")
    subprocess.run(cmd, check=True)

    scores_json = out_dir / "test_dataset" / "metrics_plot" / "all_scores.json"
    report_json = out_dir / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    assert scores_json.exists(), f"missing scores: {scores_json}"
    assert report_json.exists(), f"missing report: {report_json}"

    with open(scores_json, "r") as f:
        scores = json.load(f)
    with open(report_json, "r") as f:
        report = json.load(f)
    return scores, report


def test_run_test_all_metrics_tiny_regression_uses_saved_baseline(tmp_path):
    """
    End-to-end regression gate:
    1) full tiny sweep writes a saved baseline JSON
    2) second full tiny sweep must be at least as good (within tolerance)
    """
    if os.environ.get("RUN_TINY_METRICS_INTEGRATION", "0") != "1":
        pytest.skip("set RUN_TINY_METRICS_INTEGRATION=1 to run")

    tol_frac = float(os.environ.get("RUN_TINY_METRICS_TOL_FRAC", "0.10"))

    vols_prefix = tmp_path / "vol"
    _write_volumes(vols_prefix, n_vols=12, grid=32, voxel_size=4.25)
    baseline_json = tmp_path / "baseline" / "all_scores.json"

    first_scores, first_report = _run_with_baseline(
        vols_prefix=vols_prefix,
        out_dir=tmp_path / "run_first",
        baseline_json=baseline_json,
        tol_frac=tol_frac,
        overwrite_baseline=True,
    )
    assert baseline_json.exists()
    assert first_report.get("status") == "baseline_written"

    second_scores, second_report = _run_with_baseline(
        vols_prefix=vols_prefix,
        out_dir=tmp_path / "run_second",
        baseline_json=baseline_json,
        tol_frac=tol_frac,
        overwrite_baseline=False,
    )
    assert second_report.get("status") == "checked"
    assert int(second_report.get("checked_metrics", 0)) > 0
    assert second_report.get("failures") == []

    directional_keys = [
        k
        for k, v in second_scores.items()
        if k in first_scores
        and isinstance(v, (int, float))
        and metric_direction(k) != "ignore"
    ]
    assert directional_keys, "expected at least one directional metric to be checked"
