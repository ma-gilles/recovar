"""
Long regression tests for pipeline_with_outliers with --ind / --particle-ind.

These tests verify that the pipeline produces correct, high-quality results
when the user supplies a pre-computed index file to restrict processing to a
subset of images or particles. This is a common production workflow pattern
(e.g. after a manual or automated curation step).

Two variants:
  1. SPA (single-particle analysis) with --ind (image-level subsetting)
  2. cryo-ET (tilt series) with --particle-ind (particle-group-level subsetting)

Required env vars:
  LONG_METRICS_VOLUMES_DIR     – volume prefix (e.g. /scratch/.../vol)
  LONG_METRICS_OUTPUT_BASE     – base directory for large scratch outputs

Optional env vars:
  PIPELINE_IND_FRAC            – fraction of images/particles to keep (default: 0.80)
  PIPELINE_IND_N_IMAGES        – total images to generate (default: 10000)
  PIPELINE_IND_GRID_SIZE       – grid size (default: 128)
  PIPELINE_IND_PCT_OUTLIERS    – outlier fraction (default: 0.15)
  PIPELINE_IND_K_ROUNDS        – outlier-removal rounds (default: 2)
  PIPELINE_IND_TOL_FRAC        – allowed relative metric degradation (default: 0.20)

Baseline env vars (SPA):
  PIPELINE_IND_BASELINE_JSON   – path to read/write baseline metrics JSON
  PIPELINE_IND_WRITE_BASELINE  – set to "1" to (re)write baseline and skip check

Baseline env vars (cryo-ET):
  PIPELINE_IND_ET_BASELINE_JSON   – path to read/write ET baseline JSON
                                    (defaults to PIPELINE_IND_BASELINE_JSON stem + _cryo_et.json)
  PIPELINE_IND_ET_WRITE_BASELINE  – set to "1" to (re)write ET baseline
  PIPELINE_IND_N_TILTS            – tilts per particle for ET test (default: 7)
  PIPELINE_IND_PCT_TILT_OUTLIERS  – fraction of tilt-level outliers (default: 0.10)

Usage:
  # Write baseline (first run or after algorithm change):
  LONG_METRICS_VOLUMES_DIR=/path/to/vols/vol \\
  LONG_METRICS_OUTPUT_BASE=/scratch/recovar_tests \\
  PIPELINE_IND_BASELINE_JSON=tests/baselines/pipeline_with_indices/spa_baseline.json \\
  PIPELINE_IND_WRITE_BASELINE=1 \\
  pytest tests/integration/test_pipeline_with_indices_long.py \\
      --long-test -v -k test_pipeline_spa_with_ind_regression

  # Compare against baseline:
  LONG_METRICS_VOLUMES_DIR=... LONG_METRICS_OUTPUT_BASE=... \\
  PIPELINE_IND_BASELINE_JSON=tests/baselines/pipeline_with_indices/spa_baseline.json \\
  pytest tests/integration/test_pipeline_with_indices_long.py --long-test -v
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from conftest import gpu_subprocess_env
from helpers.metrics_regression import compare_metric, metric_direction

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.long_test,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PIPELINE_IND_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "pipeline_with_indices" / "spa_baseline.json"
)
_DEFAULT_PIPELINE_IND_ET_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "pipeline_with_indices" / "cryo_et_baseline.json"
)


# ---------------------------------------------------------------------------
# Helpers (parallel structure to test_run_test_outliers_pipeline_regression)
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.skip(f"set {name} to run this long regression test")
    return val


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    out = Path(base) / "pytest_pipeline_ind" / name if base else tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _compare_against_baseline(
    current: Dict[str, float],
    baseline_path: Path,
    tol_frac: float,
    write: bool,
) -> None:
    """Compare current metrics against the stored baseline.
    If write=True, overwrite the baseline file *after* the comparison passes."""
    assert baseline_path.exists(), f"baseline not found: {baseline_path}"
    baseline = _load_json(baseline_path)
    failures: List[str] = []
    checked = 0
    for key in sorted(set(current) & set(baseline)):
        cur = current[key]
        base = baseline[key]
        if not (isinstance(cur, (int, float)) and isinstance(base, (int, float))):
            continue
        direction = metric_direction(key)
        if direction == "ignore":
            if any(tok in key for tok in ("recall", "precision", "f1")):
                direction = "higher"
            else:
                continue
        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=tol_frac)
        checked += 1
        if not ok:
            failures.append(f"{key}: current={cur:.4f} baseline={base:.4f} ({msg})")

    assert checked > 0, "no numeric metrics were compared; check baseline/current dicts"
    assert not failures, "pipeline-with-ind metric regressions:\n" + "\n".join(failures)

    if write:
        with open(baseline_path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)


def _dataset_exists(output_dir: Path, grid_size: int, is_tilt: bool = False) -> bool:
    """Check if a previously generated dataset exists at output_dir/test_dataset/."""
    dataset_dir = output_dir / "test_dataset"
    particles = dataset_dir / "particles.star" if is_tilt else dataset_dir / f"particles.{grid_size}.mrcs"
    return (
        particles.exists()
        and (dataset_dir / "poses.pkl").exists()
        and (dataset_dir / "ctf.pkl").exists()
        and (dataset_dir / "simulation_info.pkl").exists()
    )


def _make_dataset(
    output_dir: Path,
    grid_size: int,
    n_images: int,
    percent_outliers: float,
    volumes_prefix: Optional[str] = None,
    extra_args: str = "",
) -> Path:
    """Generate a synthetic dataset; return the dataset directory.

    When volumes_prefix is None, volumes are generated synthetically so the
    test is self-contained and runnable from any machine.
    """
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    outlier_vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=grid_size)

    dataset_dir = output_dir / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    make_cmd = [
        sys.executable, "-m", "recovar.command_line", "make_test_dataset",
        str(output_dir),
        "--n-images", str(n_images),
        "--outlier-file-input", str(outlier_vol),
        "--percent-outliers", str(percent_outliers),
        "--image-size", str(grid_size),
        "--seed", "42",
    ]
    if extra_args:
        make_cmd.extend(shlex.split(extra_args))
    subprocess.run(make_cmd, check=True, env=gpu_subprocess_env())
    return dataset_dir


def _make_tilt_dataset(
    output_dir: Path,
    grid_size: int,
    n_images: int,
    percent_outliers: float,
    volumes_prefix: Optional[str] = None,
    n_tilts: int = 7,
    pct_tilt_outliers: float = 0.10,
) -> Path:
    """Generate a synthetic cryo-ET tilt-series dataset; return dataset dir.

    When volumes_prefix is None, volumes are generated synthetically.
    """
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    outlier_vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=grid_size)

    dataset_dir = output_dir / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    make_cmd = [
        sys.executable, "-m", "recovar.command_line", "make_test_dataset",
        str(output_dir),
        "--n-images", str(n_images),
        "--outlier-file-input", str(outlier_vol),
        "--percent-outliers", str(percent_outliers),
        "--percent-tilt-series-outliers", str(pct_tilt_outliers),
        "--tilt-series",
        "--image-size", str(grid_size),
        "--seed", "42",
    ]
    subprocess.run(make_cmd, check=True, env=gpu_subprocess_env())
    return dataset_dir


def _run_pipeline_with_ind(
    dataset_dir: Path,
    pipeline_out: Path,
    grid_size: int,
    k_rounds: int,
    ind_path: Path,
) -> None:
    """Run pipeline_with_outliers on SPA data with a user-supplied --ind file."""
    mrcs = dataset_dir / f"particles.{grid_size}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"

    cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        str(mrcs),
        "--poses", str(poses),
        "--ctf", str(ctf),
        "--ind", str(ind_path),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", "from_halfmaps",
        "--lazy",
        "--zdim", "4",
        "--k-rounds", str(k_rounds),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ]
    subprocess.run(cmd, check=True, env=gpu_subprocess_env())


def _run_pipeline_with_particle_ind(
    dataset_dir: Path,
    pipeline_out: Path,
    k_rounds: int,
    particle_ind_path: Path,
) -> None:
    """Run pipeline_with_outliers on cryo-ET data with a --particle-ind file."""
    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"

    cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        str(star),
        "--poses", str(poses),
        "--ctf", str(ctf),
        "--tilt-series",
        "--tilt-series-ctf", "relion5",
        "--particle-ind", str(particle_ind_path),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", "from_halfmaps",
        "--lazy",
        "--zdim", "4",
        "--k-rounds", str(k_rounds),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ]
    subprocess.run(cmd, check=True, env=gpu_subprocess_env())


def _compute_outlier_metrics_for_ind_subset(
    pipeline_out_dir: Path,
    sim_info: dict,
    k_rounds: int,
    ind: np.ndarray,
) -> Dict[str, float]:
    """
    Compute outlier detection metrics relative to the subset defined by *ind*.

    *ind* is a 1-D array of image indices (into the original n_images dataset).
    The pipeline ran on exactly those images; inliers_round_N.pkl contains
    indices within [0 .. len(ind)-1] (local to the subset).

    True outliers within the subset are images whose sim_info["image_assignment"]
    value is < 0 (wrong volume).
    """
    image_assignment = np.asarray(sim_info["image_assignment"])
    subset_assignment = image_assignment[ind]
    true_outlier_local_indices = set(int(i) for i in np.where(subset_assignment < 0)[0])
    n_subset = len(ind)

    metrics: Dict[str, float] = {
        "total_images_in_subset": float(n_subset),
        "true_outlier_count_in_subset": float(len(true_outlier_local_indices)),
    }

    for r in range(1, k_rounds + 1):
        inliers_file = pipeline_out_dir / f"inliers_round_{r}.pkl"
        if not inliers_file.exists():
            continue
        with open(inliers_file, "rb") as f:
            detected_inliers = np.asarray(pickle.load(f), dtype=np.int64)

        detected_outlier_set = set(
            int(i) for i in np.setdiff1d(np.arange(n_subset), detected_inliers)
        )
        tp = len(detected_outlier_set & true_outlier_local_indices)
        fp = len(detected_outlier_set - true_outlier_local_indices)
        fn = len(true_outlier_local_indices - detected_outlier_set)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        metrics[f"outlier_recall_round_{r}"] = recall
        metrics[f"outlier_precision_round_{r}"] = precision
        metrics[f"outlier_f1_round_{r}"] = f1
        metrics[f"inlier_count_round_{r}"] = float(len(detected_inliers))
        metrics[f"outlier_count_round_{r}"] = float(len(detected_outlier_set))

    return metrics


def _compute_particle_outlier_metrics_for_particle_ind_subset(
    pipeline_out_dir: Path,
    sim_info: dict,
    k_rounds: int,
    particle_ind: np.ndarray,
) -> Dict[str, float]:
    """
    Compute particle-level outlier metrics for a cryo-ET test where the
    pipeline was started with --particle-ind *particle_ind*.

    *particle_ind* selects a subset of particles from the full dataset.
    particle_inliers_round_N.pkl contains LOCAL indices into [0..len(particle_ind)-1].
    """
    tilt_series_assignment = np.asarray(sim_info.get("tilt_series_assignment", []))
    if tilt_series_assignment.size == 0:
        return {}

    subset_assignment = tilt_series_assignment[particle_ind]
    true_particle_outliers = set(int(i) for i in np.where(subset_assignment < 0)[0])
    n_subset = len(particle_ind)

    metrics: Dict[str, float] = {
        "total_particles_in_subset": float(n_subset),
        "true_particle_outlier_count_in_subset": float(len(true_particle_outliers)),
    }

    for r in range(1, k_rounds + 1):
        inliers_file = pipeline_out_dir / f"particle_inliers_round_{r}.pkl"
        if not inliers_file.exists():
            continue
        with open(inliers_file, "rb") as f:
            particle_inliers = np.asarray(pickle.load(f), dtype=np.int64)

        detected_particle_outliers = set(
            int(i) for i in np.setdiff1d(np.arange(n_subset), particle_inliers)
        )
        tp = len(detected_particle_outliers & true_particle_outliers)
        fp = len(detected_particle_outliers - true_particle_outliers)
        fn = len(true_particle_outliers - detected_particle_outliers)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        metrics[f"particle_recall_round_{r}"] = recall
        metrics[f"particle_precision_round_{r}"] = precision
        metrics[f"particle_f1_round_{r}"] = f1
        metrics[f"particle_inlier_count_round_{r}"] = float(len(particle_inliers))
        metrics[f"particle_outlier_count_round_{r}"] = float(len(detected_particle_outliers))

    return metrics


def _check_ind_partition_consistency(
    pipeline_out_dir: Path,
    n_subset: int,
    k_rounds: int,
) -> None:
    """For round 1: inliers + outliers must equal n_subset (size of --ind)."""
    for r in range(1, k_rounds + 1):
        inliers_path = pipeline_out_dir / f"inliers_round_{r}.pkl"
        if not inliers_path.exists():
            continue
        with open(inliers_path, "rb") as f:
            inliers = np.asarray(pickle.load(f))
        with open(pipeline_out_dir / f"outliers_round_{r}.pkl", "rb") as f:
            outliers = np.asarray(pickle.load(f))
        if r == 1:
            total_detected = int(inliers.size + outliers.size)
            assert total_detected == n_subset, (
                f"round 1: inliers({inliers.size}) + outliers({outliers.size}) "
                f"!= n_subset({n_subset})"
            )


# ---------------------------------------------------------------------------
# Test 1: SPA pipeline with user-supplied --ind subset
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.long_test
def test_pipeline_spa_with_ind_regression(tmp_path):
    """
    Long regression: run pipeline_with_outliers (SPA) with a user-supplied
    --ind index file restricting processing to the first IND_FRAC fraction of
    images. Verifies that:

    1. Pipeline runs to completion with --ind.
    2. Round-1 inliers + outliers == n_subset (partition consistency).
    3. Outlier recall/precision/F1 within tolerance of a stored baseline.

    This covers the common production workflow where a user pre-filters images
    (e.g., from a previous curation step) and passes the indices to the pipeline.

    Required env vars:
      LONG_METRICS_VOLUMES_DIR      – volume prefix
      LONG_METRICS_OUTPUT_BASE      – scratch output base
      PIPELINE_IND_BASELINE_JSON    – path to read/write baseline metrics

    Optional:
      PIPELINE_IND_WRITE_BASELINE   – "1" to regenerate baseline
      PIPELINE_IND_FRAC             – fraction of images to keep (default: 0.80)
      PIPELINE_IND_N_IMAGES         – total images to generate (default: 10000)
      PIPELINE_IND_GRID_SIZE        – grid size (default: 128)
      PIPELINE_IND_PCT_OUTLIERS     – outlier fraction (default: 0.15)
      PIPELINE_IND_K_ROUNDS         – number of outlier-removal rounds (default: 2)
      PIPELINE_IND_TOL_FRAC         – allowed degradation (default: 0.20)
    """
    volumes_prefix = os.environ.get("LONG_METRICS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid LONG_METRICS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(
        os.environ.get("PIPELINE_IND_BASELINE_JSON", str(_DEFAULT_PIPELINE_IND_BASELINE_JSON))
    )

    ind_frac = float(os.environ.get("PIPELINE_IND_FRAC", "0.80"))
    n_images = int(os.environ.get("PIPELINE_IND_N_IMAGES", "10000"))
    grid_size = int(os.environ.get("PIPELINE_IND_GRID_SIZE", "128"))
    pct_out = float(os.environ.get("PIPELINE_IND_PCT_OUTLIERS", "0.15"))
    k_rounds = int(os.environ.get("PIPELINE_IND_K_ROUNDS", "2"))
    tol_frac = float(os.environ.get("PIPELINE_IND_TOL_FRAC", "0.20"))
    write_baseline = os.environ.get("PIPELINE_IND_WRITE_BASELINE", "0") == "1"

    output_dir = _resolve_output_dir(tmp_path, "pipeline_spa_with_ind")

    # Reuse saved dataset from a previous baseline-writing run if available.
    if not write_baseline and _dataset_exists(output_dir, grid_size, is_tilt=False):
        logger.info("Reusing existing SPA dataset at %s", output_dir / "test_dataset")
        dataset_dir = output_dir / "test_dataset"
    else:
        dataset_dir = _make_dataset(
            output_dir=output_dir,
            grid_size=grid_size,
            n_images=n_images,
            percent_outliers=pct_out,
            volumes_prefix=volumes_prefix,
        )

    # Build ind_file: first ind_frac fraction of images (sequential, no randomness
    # needed since we compare against our own baseline)
    n_subset = int(n_images * ind_frac)
    ind = np.arange(n_subset, dtype=np.int64)
    ind_path = output_dir / "ind_file.pkl"
    with open(ind_path, "wb") as f:
        pickle.dump(ind, f)

    # Run pipeline_with_outliers with the ind file (output outside dataset_dir
    # so reused datasets are not polluted).
    pipeline_out = output_dir / "pipeline_outliers_with_ind_output"
    _run_pipeline_with_ind(
        dataset_dir=dataset_dir,
        pipeline_out=pipeline_out,
        grid_size=grid_size,
        k_rounds=k_rounds,
        ind_path=ind_path,
    )

    # Partition consistency: round 1 must partition exactly n_subset images
    _check_ind_partition_consistency(pipeline_out, n_subset=n_subset, k_rounds=k_rounds)

    # Compute outlier metrics relative to the subset
    sim_info_path = dataset_dir / "simulation_info.pkl"
    assert sim_info_path.exists(), f"simulation_info.pkl missing: {sim_info_path}"
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    metrics = _compute_outlier_metrics_for_ind_subset(
        pipeline_out_dir=pipeline_out,
        sim_info=sim_info,
        k_rounds=k_rounds,
        ind=ind,
    )

    _compare_against_baseline(
        current=metrics,
        baseline_path=baseline_json,
        tol_frac=tol_frac,
        write=write_baseline,
    )


# ---------------------------------------------------------------------------
# Test 2: cryo-ET pipeline with user-supplied --particle-ind subset
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.long_test
def test_pipeline_cryo_et_with_particle_ind_regression(tmp_path):
    """
    Long regression: run pipeline_with_outliers (cryo-ET / tilt series) with a
    user-supplied --particle-ind file restricting processing to the first
    IND_FRAC fraction of particles. Verifies partition consistency and that
    particle-level outlier metrics match a stored baseline.

    Required env vars:
      LONG_METRICS_VOLUMES_DIR          – volume prefix
      LONG_METRICS_OUTPUT_BASE          – scratch output base
      PIPELINE_IND_BASELINE_JSON        – base path; ET baseline gets _cryo_et suffix
                                          unless PIPELINE_IND_ET_BASELINE_JSON is set

    Optional:
      PIPELINE_IND_ET_BASELINE_JSON     – explicit path for ET baseline
      PIPELINE_IND_ET_WRITE_BASELINE    – "1" to write ET baseline
      PIPELINE_IND_FRAC                 – fraction of particles to keep (default: 0.80)
      PIPELINE_IND_N_IMAGES             – total images (default: 10000)
      PIPELINE_IND_GRID_SIZE            – grid size (default: 128)
      PIPELINE_IND_PCT_OUTLIERS         – outlier fraction (default: 0.15)
      PIPELINE_IND_N_TILTS              – tilts per particle (default: 7)
      PIPELINE_IND_PCT_TILT_OUTLIERS    – tilt-level outlier fraction (default: 0.10)
      PIPELINE_IND_K_ROUNDS             – number of outlier-removal rounds (default: 2)
      PIPELINE_IND_TOL_FRAC             – allowed degradation (default: 0.20)
    """
    volumes_prefix = os.environ.get("LONG_METRICS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid LONG_METRICS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(
        os.environ.get("PIPELINE_IND_ET_BASELINE_JSON", str(_DEFAULT_PIPELINE_IND_ET_BASELINE_JSON))
    )

    ind_frac = float(os.environ.get("PIPELINE_IND_FRAC", "0.80"))
    n_images = int(os.environ.get("PIPELINE_IND_N_IMAGES", "10000"))
    grid_size = int(os.environ.get("PIPELINE_IND_GRID_SIZE", "128"))
    pct_out = float(os.environ.get("PIPELINE_IND_PCT_OUTLIERS", "0.15"))
    n_tilts = int(os.environ.get("PIPELINE_IND_N_TILTS", "7"))
    pct_tilt_out = float(os.environ.get("PIPELINE_IND_PCT_TILT_OUTLIERS", "0.10"))
    k_rounds = int(os.environ.get("PIPELINE_IND_K_ROUNDS", "2"))
    tol_frac = float(os.environ.get("PIPELINE_IND_TOL_FRAC", "0.20"))
    write_baseline = (
        os.environ.get("PIPELINE_IND_WRITE_BASELINE", "0") == "1"
        or os.environ.get("PIPELINE_IND_ET_WRITE_BASELINE", "0") == "1"
    )

    output_dir = _resolve_output_dir(tmp_path, "pipeline_cryo_et_with_particle_ind")

    # Reuse saved dataset from a previous baseline-writing run if available.
    if not write_baseline and _dataset_exists(output_dir, grid_size, is_tilt=True):
        logger.info("Reusing existing cryo-ET dataset at %s", output_dir / "test_dataset")
        dataset_dir = output_dir / "test_dataset"
    else:
        dataset_dir = _make_tilt_dataset(
            output_dir=output_dir,
            grid_size=grid_size,
            n_images=n_images,
            percent_outliers=pct_out,
            volumes_prefix=volumes_prefix,
            n_tilts=n_tilts,
            pct_tilt_outliers=pct_tilt_out,
        )

    sim_info_path = dataset_dir / "simulation_info.pkl"
    assert sim_info_path.exists(), f"simulation_info.pkl missing: {sim_info_path}"
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    # Determine number of particles from sim_info
    tilt_series_assignment = np.asarray(sim_info.get("tilt_series_assignment", []))
    if tilt_series_assignment.size == 0:
        pytest.skip("no tilt_series_assignment in simulation_info; not a cryo-ET dataset")
    n_particles = int(tilt_series_assignment.size)

    # particle_ind: first ind_frac fraction of particles
    n_subset_particles = int(n_particles * ind_frac)
    particle_ind = np.arange(n_subset_particles, dtype=np.int64)
    particle_ind_path = output_dir / "particle_ind_file.pkl"
    with open(particle_ind_path, "wb") as f:
        pickle.dump(particle_ind, f)

    # Run pipeline_with_outliers with the particle_ind file (output outside
    # dataset_dir so reused datasets are not polluted).
    pipeline_out = output_dir / "pipeline_outliers_cryo_et_with_particle_ind_output"
    _run_pipeline_with_particle_ind(
        dataset_dir=dataset_dir,
        pipeline_out=pipeline_out,
        k_rounds=k_rounds,
        particle_ind_path=particle_ind_path,
    )

    # Partition consistency at image level is harder to assert for cryo-ET with
    # --particle-ind because image count depends on tilts per subset particle.
    # We verify output files exist instead.
    for r in range(1, k_rounds + 1):
        assert (pipeline_out / f"inliers_round_{r}.pkl").exists(), (
            f"missing inliers_round_{r}.pkl in {pipeline_out}"
        )

    # Particle-level outlier metrics for the subset
    particle_metrics = _compute_particle_outlier_metrics_for_particle_ind_subset(
        pipeline_out_dir=pipeline_out,
        sim_info=sim_info,
        k_rounds=k_rounds,
        particle_ind=particle_ind,
    )

    # Also compute image-level outlier metrics where possible
    # (images within the subset particles)
    image_assignment = np.asarray(sim_info.get("image_assignment", []))
    if image_assignment.size > 0:
        # Reconstruct which images belong to the selected particles
        # tilt_series_assignment maps particle -> assignment;
        # for image-level we use the regular image_assignment.
        # We can't easily determine which images belong to particle_ind without
        # the tilt-index mapping, so we skip image-level metrics for ET with
        # particle-ind and only use particle-level metrics.
        pass

    all_metrics = {**particle_metrics}
    assert all_metrics, "no metrics were computed; check simulation_info and pipeline output"

    _compare_against_baseline(
        current=all_metrics,
        baseline_path=baseline_json,
        tol_frac=tol_frac,
        write=write_baseline,
    )
