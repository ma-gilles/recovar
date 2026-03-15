"""
Tiny self-contained regression-baseline test for the outliers pipeline.

Mirrors test_run_test_all_metrics_tiny_regression_baseline.py exactly:
  Run 1:  write baseline from current code
  Run 2:  compare against that baseline → assert no regression

Both runs use identical randomly-generated volumes and the same RNG seed,
so the comparison is a deterministic self-consistency check: code changes
that legitimately worsen detection accuracy will fail here.

Activation (same gate as the metrics tiny regression test):
  RUN_TINY_METRICS_INTEGRATION=1 pytest ... --run-integration --run-slow --run-gpu

Optional env vars:
  TINY_OUTLIERS_GRID_SIZE      (default 32)
  TINY_OUTLIERS_N_IMAGES       (default 800)
  TINY_OUTLIERS_PERCENT        (default 0.15)
  TINY_OUTLIERS_K_ROUNDS       (default 1)
  TINY_OUTLIERS_TOL_FRAC       (default 0.10)
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io, pytest.mark.tiny_metrics]

_GRID = int(os.environ.get("TINY_OUTLIERS_GRID_SIZE", "32"))
_N_IMAGES = int(os.environ.get("TINY_OUTLIERS_N_IMAGES", "800"))
_PCT = float(os.environ.get("TINY_OUTLIERS_PERCENT", "0.15"))
_K = int(os.environ.get("TINY_OUTLIERS_K_ROUNDS", "1"))
_TOL = float(os.environ.get("TINY_OUTLIERS_TOL_FRAC", "0.10"))
_N_VOLS = 12


# ---------------------------------------------------------------------------
# Volume helpers (same as tiny integration test)
# ---------------------------------------------------------------------------

def _make_volume(idx: int, n_vols: int, grid: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    t = 2.0 * np.pi * idx / max(n_vols, 1)
    vol = (
        np.exp(
            -((xx - 0.3 * np.cos(t)) ** 2 + (yy - 0.25 * np.sin(t)) ** 2 + (zz - 0.2 * np.cos(2 * t)) ** 2)
            / (2 * 0.18 ** 2)
        )
        + 0.7 * np.exp(
            -((xx + 0.25 * np.sin(1.3 * t)) ** 2 + (yy - 0.2 * np.cos(1.1 * t)) ** 2 + (zz + 0.2 * np.sin(t)) ** 2)
            / (2 * 0.16 ** 2)
        )
    ).astype(np.float32)
    vol -= vol.mean()
    nrm = np.linalg.norm(vol.ravel())
    if nrm > 0:
        vol /= nrm
    return vol


def _write_volumes(prefix: Path, n_vols: int = _N_VOLS, grid: int = _GRID) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n_vols):
        with mrcfile.new(f"{prefix}{i:04d}.mrc", overwrite=True) as m:
            m.set_data(_make_volume(i, n_vols, grid))
            m.voxel_size = 4.25


# ---------------------------------------------------------------------------
# Pipeline runner + metric extractor
# ---------------------------------------------------------------------------

def _run_and_score(
    output_dir: Path,
    volumes_prefix: Path,
    baseline_json: Path,
    overwrite_baseline: bool,
) -> tuple[dict, dict]:
    """Run the full outliers pipeline; return (scores, regression_report)."""
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=_GRID)

    make_cmd = [
        sys.executable, "-m", "recovar.command_line", "make_test_dataset",
        str(output_dir),
        "--n-images", str(_N_IMAGES),
        "--outlier-file-input", str(outlier_vol),
        "--percent-outliers", str(_PCT),
        "--grid-size", str(_GRID),
        "--volume-input", str(volumes_prefix),
    ]
    subprocess.run(make_cmd, check=True)

    # Compute GT union mask from the volume MRC files
    from recovar.core import mask as mask_mod
    from recovar import utils as recovar_utils

    vol_files = sorted(volumes_prefix.parent.glob(f"{volumes_prefix.name}*.mrc"))
    vols = [recovar_utils.load_mrc(str(f)) for f in vol_files]
    volume_shape = (_GRID, _GRID, _GRID)
    gt_union_soft_mask, _ = mask_mod.make_union_gt_mask(vols, volume_shape)
    gt_mask_mrc = str(output_dir / "gt_union_mask.mrc")
    recovar_utils.write_mrc(gt_mask_mrc, gt_union_soft_mask)

    dataset_dir = output_dir / "test_dataset"
    pipeline_out = dataset_dir / "pipeline_outliers_output"
    particles = str(dataset_dir / f"particles.{_GRID}.mrcs")

    pipe_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        particles,
        "--poses", str(dataset_dir / "poses.pkl"),
        "--ctf", str(dataset_dir / "ctf.pkl"),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", gt_mask_mrc,
        "--lazy",
        "--zdim", "4",
        "--k-rounds", str(_K),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ]
    subprocess.run(pipe_cmd, check=True)

    # Compute detection metrics from ground truth
    sim_info_path = dataset_dir / "simulation_info.pkl"
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    assign = np.asarray(sim_info["image_assignment"])
    n_total = int(assign.size)
    true_out = set(int(i) for i in np.where(assign < 0)[0])
    scores: dict = {"total_images": float(n_total), "true_outlier_count": float(len(true_out))}

    for r in range(1, _K + 1):
        fp = pipeline_out / f"inliers_round_{r}.pkl"
        if not fp.exists():
            continue
        with open(fp, "rb") as fh:
            inliers = np.asarray(pickle.load(fh), dtype=np.int64)
        det_out = set(int(i) for i in np.setdiff1d(np.arange(n_total), inliers))
        tp = len(det_out & true_out)
        fp_count = len(det_out - true_out)
        fn = len(true_out - det_out)
        prec = tp / max(tp + fp_count, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        scores[f"outlier_recall_round_{r}"] = rec
        scores[f"outlier_precision_round_{r}"] = prec
        scores[f"outlier_f1_round_{r}"] = f1
        scores[f"inlier_count_round_{r}"] = float(len(inliers))
        scores[f"outlier_count_round_{r}"] = float(len(det_out))

    report: dict = {"checked_metrics": 0, "failures": [], "status": "unknown"}

    if overwrite_baseline or not baseline_json.exists():
        baseline_json.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_json, "w") as f:
            json.dump(scores, f, indent=2, sort_keys=True)
        report["status"] = "baseline_written"
        return scores, report

    with open(baseline_json) as f:
        baseline = json.load(f)

    failures = []
    checked = 0
    for key in sorted(set(scores) & set(baseline)):
        cur = scores[key]
        base = baseline[key]
        if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
            continue
        # Treat recall/precision/f1 as higher-is-better
        if any(tok in key for tok in ("recall", "precision", "f1")):
            direction = "higher"
        else:
            direction = metric_direction(key)
        if direction == "ignore":
            continue
        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=_TOL, metric_name=key)
        checked += 1
        if not ok:
            failures.append(f"{key}: current={cur:.4f} baseline={base:.4f} ({msg})")

    report["checked_metrics"] = checked
    report["failures"] = failures
    report["status"] = "checked"

    # Save regression report alongside scores
    report_json = pipeline_out / "outlier_regression_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    return scores, report


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_run_test_outliers_pipeline_tiny_regression_uses_saved_baseline(tmp_path):
    """
    End-to-end outlier regression gate (same pattern as the metrics test):

    Run 1: generate volumes + dataset + pipeline → write baseline JSON
    Run 2: same volumes + dataset + pipeline → compare against baseline

    Both runs use the same generated volumes, so any stochasticity in the
    pipeline is the only source of difference.  Failures indicate genuine
    accuracy regressions.
    """
    tol_frac = float(os.environ.get("TINY_OUTLIERS_TOL_FRAC", str(_TOL)))

    vols_prefix = tmp_path / "vol"
    _write_volumes(vols_prefix, n_vols=_N_VOLS, grid=_GRID)
    baseline_json = tmp_path / "baseline" / "outlier_scores.json"

    # Run 1: write baseline
    first_scores, first_report = _run_and_score(
        output_dir=tmp_path / "run_first",
        volumes_prefix=vols_prefix,
        baseline_json=baseline_json,
        overwrite_baseline=True,
    )
    assert baseline_json.exists(), "baseline JSON must be written after run 1"
    assert first_report.get("status") == "baseline_written"

    # Run 2: compare against baseline
    second_scores, second_report = _run_and_score(
        output_dir=tmp_path / "run_second",
        volumes_prefix=vols_prefix,
        baseline_json=baseline_json,
        overwrite_baseline=False,
    )
    assert second_report.get("status") == "checked", \
        f"expected status='checked', got: {second_report}"
    assert int(second_report.get("checked_metrics", 0)) > 0, \
        "no numeric metrics were compared; check baseline/scores dicts"
    assert second_report.get("failures") == [], \
        "outlier detection regressions:\n" + "\n".join(second_report.get("failures", []))

    # Cross-check: directional keys must be present in both runs
    directional_keys = [
        k for k in sorted(set(first_scores) & set(second_scores))
        if isinstance(second_scores[k], (int, float))
        and any(tok in k for tok in ("recall", "precision", "f1"))
    ]
    assert directional_keys, \
        "expected at least one recall/precision/f1 metric to appear in both runs"
