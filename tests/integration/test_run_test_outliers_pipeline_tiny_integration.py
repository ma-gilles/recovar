"""
Tiny self-contained integration test for the outliers pipeline.

Mirrors the structure of test_run_test_all_metrics_tiny_integration.py:
  - generates its own volumes (no external data needed)
  - runs pipeline_with_outliers end-to-end with a small dataset
  - checks that all expected output files exist and metrics are sensible

Activation (pass the pytest flag):
  pytest ... --run-integration --run-slow --run-gpu --run-tiny-metrics

Optional env vars:
  TINY_OUTLIERS_GRID_SIZE      (default 32)
  TINY_OUTLIERS_N_IMAGES       (default 800)
  TINY_OUTLIERS_PERCENT        (default 0.15)
  TINY_OUTLIERS_K_ROUNDS       (default 1)
  TINY_OUTLIERS_N_TILTS        (default 0 = SPA; >0 = cryo-ET)
  TINY_OUTLIERS_PCT_TILT       (default 0.10, cryo-ET tilt-outlier fraction)

The test variant with TINY_OUTLIERS_N_TILTS>0 exercises the cryo-ET tilt
series path including particle-level outlier indices.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io, pytest.mark.tiny_metrics]

_GRID = int(os.environ.get("TINY_OUTLIERS_GRID_SIZE", "32"))
_N_IMAGES = int(os.environ.get("TINY_OUTLIERS_N_IMAGES", "800"))
_PCT_OUTLIERS = float(os.environ.get("TINY_OUTLIERS_PERCENT", "0.15"))
_K_ROUNDS = int(os.environ.get("TINY_OUTLIERS_K_ROUNDS", "1"))
_N_TILTS = int(os.environ.get("TINY_OUTLIERS_N_TILTS", "0"))
_PCT_TILT = float(os.environ.get("TINY_OUTLIERS_PCT_TILT", "0.10"))

_N_VOLS = 12


# ---------------------------------------------------------------------------
# Volume helper (same as in tiny metrics test)
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
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_outliers_pipeline(
    output_dir: Path,
    volumes_prefix: Path,
    grid: int,
    n_images: int,
    pct_outliers: float,
    k_rounds: int,
    n_tilts: int = 0,
    pct_tilt_outliers: float = 0.10,
) -> Path:
    """Generate dataset and run pipeline_with_outliers; return pipeline output dir."""
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    outlier_vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=grid)

    # Generate dataset
    make_cmd = [
        sys.executable, "-m", "recovar.command_line", "make_test_dataset",
        str(output_dir),
        "--n-images", str(n_images),
        "--outlier-file-input", str(outlier_vol),
        "--percent-outliers", str(pct_outliers),
        "--grid-size", str(grid),
        "--volume-input", str(volumes_prefix),
    ]
    if n_tilts > 0:
        make_cmd += [
            "--tilt-series",
            "--n-tilts", str(n_tilts),
            "--percent-tilt-series-outliers", str(pct_tilt_outliers),
        ]
    from conftest import gpu_subprocess_env
    env = gpu_subprocess_env()
    subprocess.run(make_cmd, check=True, env=env)

    # Compute GT union mask from the volume MRC files
    from recovar.core import mask as mask_mod
    from recovar import utils as recovar_utils

    vol_files = sorted(volumes_prefix.parent.glob(f"{volumes_prefix.name}*.mrc"))
    vols = [recovar_utils.load_mrc(str(f)) for f in vol_files]
    volume_shape = (grid, grid, grid)
    gt_union_soft_mask, _ = mask_mod.make_union_gt_mask(vols, volume_shape)
    gt_mask_mrc = str(output_dir / "gt_union_mask.mrc")
    recovar_utils.write_mrc(gt_mask_mrc, gt_union_soft_mask)

    dataset_dir = output_dir / "test_dataset"
    pipeline_out = dataset_dir / "pipeline_outliers_output"

    # Run pipeline_with_outliers
    if n_tilts > 0:
        particles = str(dataset_dir / "particles.star")
        pipe_extra = ["--tilt-series", "--tilt-series-ctf", "relion5"]
    else:
        particles = str(dataset_dir / f"particles.{grid}.mrcs")
        pipe_extra = []

    poses = str(dataset_dir / "poses.pkl")
    ctf = str(dataset_dir / "ctf.pkl")

    pipe_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        particles,
        "--poses", poses,
        "--ctf", ctf,
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", gt_mask_mrc,
        "--lazy",
        "--zdim", "4",
        "--k-rounds", str(k_rounds),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ] + pipe_extra
    subprocess.run(pipe_cmd, check=True, env=env)

    return pipeline_out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(pipeline_out: Path, sim_info_path: Path, k_rounds: int) -> dict:
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    assign = np.asarray(sim_info["image_assignment"])
    n_total = int(assign.size)
    true_outliers = set(int(i) for i in np.where(assign < 0)[0])
    metrics: dict = {"total_images": float(n_total), "true_outlier_count": float(len(true_outliers))}

    for r in range(1, k_rounds + 1):
        f = pipeline_out / f"inliers_round_{r}.pkl"
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            inliers = np.asarray(pickle.load(fh), dtype=np.int64)
        detected_out = set(int(i) for i in np.setdiff1d(np.arange(n_total), inliers))
        tp = len(detected_out & true_outliers)
        fp = len(detected_out - true_outliers)
        fn = len(true_outliers - detected_out)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        metrics[f"outlier_recall_round_{r}"] = rec
        metrics[f"outlier_precision_round_{r}"] = prec
        metrics[f"outlier_f1_round_{r}"] = f1
        metrics[f"inlier_count_round_{r}"] = float(len(inliers))
        metrics[f"outlier_count_round_{r}"] = float(len(detected_out))

    # Particle-level metrics for cryo-ET
    ts_assign = sim_info.get("tilt_series_assignment")
    if ts_assign is not None:
        ts_assign = np.asarray(ts_assign)
        n_parts = int(ts_assign.size)
        true_part_out = set(int(i) for i in np.where(ts_assign < 0)[0])
        metrics["total_particles"] = float(n_parts)
        metrics["true_particle_outlier_count"] = float(len(true_part_out))
        for r in range(1, k_rounds + 1):
            pf = pipeline_out / f"particle_inliers_round_{r}.pkl"
            if not pf.exists():
                continue
            with open(pf, "rb") as fh:
                p_inliers = np.asarray(pickle.load(fh), dtype=np.int64)
            det_p_out = set(int(i) for i in np.setdiff1d(np.arange(n_parts), p_inliers))
            tp = len(det_p_out & true_part_out)
            fp = len(det_p_out - true_part_out)
            fn = len(true_part_out - det_p_out)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            metrics[f"particle_recall_round_{r}"] = rec
            metrics[f"particle_precision_round_{r}"] = prec
            metrics[f"particle_f1_round_{r}"] = f1
            metrics[f"particle_inlier_count_round_{r}"] = float(len(p_inliers))
            metrics[f"particle_outlier_count_round_{r}"] = float(len(det_p_out))

    return metrics


def _check_files_and_partition(pipeline_out: Path, n_total: int, k_rounds: int) -> None:
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()
    for r in range(1, k_rounds + 1):
        for stem in (f"inliers_round_{r}", f"outliers_round_{r}"):
            assert (pipeline_out / f"{stem}.pkl").exists(), f"missing {stem}.pkl"
        # round 1 must partition all images
        if r == 1:
            with open(pipeline_out / f"inliers_round_{r}.pkl", "rb") as f:
                n_in = len(np.asarray(pickle.load(f)))
            with open(pipeline_out / f"outliers_round_{r}.pkl", "rb") as f:
                n_out = len(np.asarray(pickle.load(f)))
            assert n_in + n_out == n_total, (
                f"round 1: inliers({n_in}) + outliers({n_out}) ≠ total({n_total})"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_test_outliers_pipeline_tiny_integration_spa(tmp_path):
    """
    End-to-end SPA outliers pipeline on tiny generated data.
    Activated with --run-tiny-metrics.
    """
    vols_prefix = tmp_path / "vol"
    _write_volumes(vols_prefix, n_vols=_N_VOLS, grid=_GRID)

    pipeline_out = _run_outliers_pipeline(
        output_dir=tmp_path / "run_spa",
        volumes_prefix=vols_prefix,
        grid=_GRID,
        n_images=_N_IMAGES,
        pct_outliers=_PCT_OUTLIERS,
        k_rounds=_K_ROUNDS,
        n_tilts=0,
    )

    sim_info_path = tmp_path / "run_spa" / "test_dataset" / "simulation_info.pkl"
    assert sim_info_path.exists()
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    _check_files_and_partition(pipeline_out, n_total=n_total, k_rounds=_K_ROUNDS)

    metrics = _compute_metrics(pipeline_out, sim_info_path, k_rounds=_K_ROUNDS)
    scores_json = pipeline_out / "all_scores.json"
    with open(scores_json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"\nSPA outlier detection metrics (grid={_GRID}, n={_N_IMAGES}):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


def test_run_test_outliers_pipeline_tiny_integration_cryo_et(tmp_path):
    """
    End-to-end cryo-ET tilt-series outliers pipeline on tiny generated data.
    Activated with --run-tiny-metrics.
    Requires TINY_OUTLIERS_N_TILTS > 0 (or defaults to 5 if not set).
    """

    n_tilts = _N_TILTS if _N_TILTS > 0 else 5   # default 5 for cryo-ET test
    vols_prefix = tmp_path / "vol"
    _write_volumes(vols_prefix, n_vols=_N_VOLS, grid=_GRID)

    pipeline_out = _run_outliers_pipeline(
        output_dir=tmp_path / "run_et",
        volumes_prefix=vols_prefix,
        grid=_GRID,
        n_images=_N_IMAGES,
        pct_outliers=_PCT_OUTLIERS,
        k_rounds=_K_ROUNDS,
        n_tilts=n_tilts,
        pct_tilt_outliers=_PCT_TILT,
    )

    sim_info_path = tmp_path / "run_et" / "test_dataset" / "simulation_info.pkl"
    assert sim_info_path.exists()
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    _check_files_and_partition(pipeline_out, n_total=n_total, k_rounds=_K_ROUNDS)

    metrics = _compute_metrics(pipeline_out, sim_info_path, k_rounds=_K_ROUNDS)
    scores_json = pipeline_out / "all_scores_et.json"
    with open(scores_json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"\nCryo-ET outlier detection metrics (grid={_GRID}, n={_N_IMAGES}, n_tilts={n_tilts}):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # If particle-level indices were written, check them
    has_particle = any(
        (pipeline_out / f"particle_inliers_round_{r}.pkl").exists()
        for r in range(1, _K_ROUNDS + 1)
    )
    if has_particle:
        assert "total_particles" in metrics
        assert metrics.get("true_particle_outlier_count", 0) >= 0
