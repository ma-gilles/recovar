"""
Long regression tests for the outliers detection pipeline.

Two complementary approaches (following the same pattern as
test_run_test_all_metrics_regression_long.py):

  1. Baseline-comparison mode
     Run pipeline_with_outliers on a real, user-provided dataset; compare
     detection metrics (recall, precision, F1, inlier/outlier counts) against
     a stored JSON baseline.

     Required env vars:
       OUTLIERS_VOLUMES_DIR      – prefix path for ground-truth volumes
                                   (expects <prefix>NNNN.mrc, e.g. vol0000.mrc)
       OUTLIERS_BASELINE_JSON    – path to stored baseline metrics JSON

     Optional env vars:
       OUTLIERS_WRITE_BASELINE   – set to "1" to (re)write baseline from
                                   current run and then skip the test
       OUTLIERS_N_IMAGES         – number of images to simulate (default 10000)
       OUTLIERS_GRID_SIZE        – grid size (default 128)
       OUTLIERS_PERCENT_OUTLIERS – outlier fraction (default 0.15)
       OUTLIERS_K_ROUNDS         – number of outlier-removal rounds (default 2)
       OUTLIERS_TOL_FRAC         – allowed relative metric degradation
                                   (default 0.15)
       LONG_METRICS_OUTPUT_BASE  – base directory for large outputs; falls back
                                   to pytest tmp_path

  2. Tiny self-contained regression
     Runs the full outliers pipeline on a tiny simulator-generated dataset
     (grid_size=32, n_images=500) without needing external files.  Checks:
       • all expected output files exist
       • inliers + outliers = total images (per round)
       • recall ≥ MIN_RECALL_TINY  (loose bound, set conservatively)
       • precision ≥ MIN_PRECISION_TINY
       • metrics compare against a committed tiny baseline if present

     Marks: integration, slow  (not gpu – can run on CPU, just slowly)

Both tests save a JSON blob that can later serve as a regression baseline.
The JSON keys follow this schema:
  outlier_recall_round_N        (higher-is-better)
  outlier_precision_round_N     (higher-is-better)
  outlier_f1_round_N            (higher-is-better)
  inlier_count_round_N          (informational)
  outlier_count_round_N         (informational)
  true_outlier_count            (informational)
  total_images                  (informational)

For cryo-ET the same keys exist at the particle level:
  particle_recall_round_N
  particle_precision_round_N
  particle_f1_round_N
  particle_inlier_count_round_N
  particle_outlier_count_round_N
  true_particle_outlier_count
  total_particles
"""

from __future__ import annotations

import json
import math
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from conftest import gpu_subprocess_env
from helpers.metrics_regression import compare_metric, metric_direction

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Tiny-test thresholds (loose – small dataset is noisy)
# ---------------------------------------------------------------------------
MIN_RECALL_TINY = 0.30       # at least 30 % of true outliers detected
MIN_PRECISION_TINY = 0.20    # at least 20 % of detected are real outliers
TINY_GRID_SIZE = 32
TINY_N_IMAGES = 500
TINY_PERCENT_OUTLIERS = 0.20
TINY_K_ROUNDS = 1

# ---------------------------------------------------------------------------
# Fast smoke test thresholds (high-SNR, easy outliers → tight bounds)
# ---------------------------------------------------------------------------
FAST_GRID_SIZE = 32
FAST_N_IMAGES = 200
FAST_NOISE_LEVEL = 0.01       # very high SNR
FAST_PERCENT_OUTLIERS = 0.25
FAST_K_ROUNDS = 1
# With high SNR the pipeline should find outliers easily:
MIN_RECALL_FAST = 0.70
# Precision is sensitive to junk-detection over-selection on some GPU/JAX runtimes;
# keep a meaningful floor while avoiding environment-driven false failures.
MIN_PRECISION_FAST = 0.35

# ---------------------------------------------------------------------------
# Baseline path for tiny self-contained test
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TINY_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_outliers_pipeline" / "tiny_baseline.json"
)
_TINY_BASELINE_META = _TINY_BASELINE_JSON.with_name("tiny_baseline_metadata.json")
# Default in-repo baseline paths for the long tests (auto-created on first run).
_DEFAULT_OUTLIERS_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_outliers_pipeline" / "long_generated" / "all_scores.json"
)
_DEFAULT_OUTLIERS_ET_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_outliers_pipeline" / "long_generated" / "all_scores_cryo_et.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.skip(f"set {name} to run this long regression test")
    return val


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    out = Path(base) / "pytest_outliers" / name if base else tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dataset_exists_spa(output_dir: Path, grid_size: int) -> bool:
    """Check if a SPA dataset already exists at output_dir/test_dataset/."""
    d = output_dir / "test_dataset"
    return (
        (d / f"particles.{grid_size}.mrcs").exists()
        and (d / "poses.pkl").exists()
        and (d / "ctf.pkl").exists()
        and (d / "simulation_info.pkl").exists()
    )


def _dataset_exists_et(output_dir: Path) -> bool:
    """Check if a cryo-ET dataset already exists at output_dir/test_dataset/."""
    d = output_dir / "test_dataset"
    return (
        (d / "particles.star").exists()
        and (d / "poses.pkl").exists()
        and (d / "ctf.pkl").exists()
        and (d / "simulation_info.pkl").exists()
    )


def _run_outliers_pipeline(
    output_dir: Path,
    volumes_prefix: Optional[str] = None,
    grid_size: int = 128,
    n_images: int = 10000,
    percent_outliers: float = 0.15,
    k_rounds: int = 2,
    extra_args: str = "",
    accept_cpu: bool = False,
    reuse_dataset: bool = False,
) -> Path:
    """
    Generate a test dataset and run pipeline_with_outliers; return the
    pipeline output directory (round-level outputs live there).

    If *reuse_dataset* is True and the dataset already exists, skip
    dataset generation and reuse the existing one.
    """
    dataset_dir = output_dir / "test_dataset"

    if reuse_dataset and _dataset_exists_spa(output_dir, grid_size):
        pass  # skip dataset generation
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # -- create outlier volume --------------------------------------------
        from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

        outlier_vol = output_dir / "outlier_volume.mrc"
        create_outlier_volume(str(outlier_vol), grid_size=grid_size)

        # -- generate synthetic dataset ---------------------------------------
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
        env = gpu_subprocess_env()
        subprocess.run(make_cmd, check=True, env=env)

    # -- run pipeline_with_outliers ------------------------------------------
    pipeline_out = output_dir / "pipeline_outliers_output"
    mrcs = dataset_dir / f"particles.{grid_size}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"

    pipe_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        str(mrcs),
        "--poses", str(poses),
        "--ctf", str(ctf),
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
    if accept_cpu:
        pipe_cmd.append("--accept-cpu")
    subprocess.run(pipe_cmd, check=True, env=gpu_subprocess_env())

    return pipeline_out


def _compute_outlier_metrics(
    pipeline_out_dir: Path,
    sim_info_path: Path,
    k_rounds: int,
) -> Dict[str, float]:
    """
    Compare pipeline detection results against ground-truth simulation info.

    Returns a dict of precision/recall/F1/count metrics.
    """
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    image_assignment = np.asarray(sim_info["image_assignment"])
    n_total = int(image_assignment.size)
    # Outliers are assigned to -1 (wrong volume)
    true_outlier_mask = image_assignment < 0
    true_outliers = set(int(i) for i in np.where(true_outlier_mask)[0])

    metrics: Dict[str, float] = {
        "total_images": float(n_total),
        "true_outlier_count": float(len(true_outliers)),
    }

    for r in range(1, k_rounds + 1):
        inliers_file = pipeline_out_dir / f"inliers_round_{r}.pkl"
        if not inliers_file.exists():
            continue
        with open(inliers_file, "rb") as f:
            detected_inliers = np.asarray(pickle.load(f), dtype=np.int64)

        detected_outliers = set(
            int(i) for i in np.setdiff1d(np.arange(n_total), detected_inliers)
        )
        tp = len(detected_outliers & true_outliers)
        fp = len(detected_outliers - true_outliers)
        fn = len(true_outliers - detected_outliers)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        metrics[f"outlier_recall_round_{r}"] = recall
        metrics[f"outlier_precision_round_{r}"] = precision
        metrics[f"outlier_f1_round_{r}"] = f1
        metrics[f"inlier_count_round_{r}"] = float(len(detected_inliers))
        metrics[f"outlier_count_round_{r}"] = float(len(detected_outliers))

    return metrics


def _check_output_files_exist(pipeline_out_dir: Path, k_rounds: int) -> None:
    for r in range(1, k_rounds + 1):
        for stem in (f"inliers_round_{r}", f"outliers_round_{r}"):
            p = pipeline_out_dir / f"{stem}.pkl"
            assert p.exists(), f"missing expected output file: {p}"
    assert (pipeline_out_dir / "all_rounds_inliers.pkl").exists()


def _check_partition_consistency(
    pipeline_out_dir: Path, n_total: int, k_rounds: int
) -> None:
    """inliers + outliers = total_images for every round (after round 1)."""
    for r in range(1, k_rounds + 1):
        inliers_path = pipeline_out_dir / f"inliers_round_{r}.pkl"
        if not inliers_path.exists():
            continue
        with open(inliers_path, "rb") as f:
            inliers = np.asarray(pickle.load(f))
        with open(pipeline_out_dir / f"outliers_round_{r}.pkl", "rb") as f:
            outliers = np.asarray(pickle.load(f))
        # Round 1 acts on all images; subsequent rounds act only on inliers.
        # After round 1, inliers ∪ outliers must cover [0, n_total).
        if r == 1:
            total_detected = int(inliers.size + outliers.size)
            assert total_detected == n_total, (
                f"round 1: inliers({inliers.size}) + outliers({outliers.size}) "
                f"≠ total({n_total})"
            )


def _compare_against_baseline(
    current: Dict[str, float],
    baseline_path: Path,
    tol_frac: float,
    write: bool,
) -> None:
    """
    If write=True, dump current metrics to baseline_path and skip the test.
    Otherwise compare current against the stored baseline.
    """
    if write or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        pytest.skip(f"outlier baseline written to {baseline_path}")

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
            # Treat outlier_recall / outlier_precision / outlier_f1 as higher-is-better
            if any(tok in key for tok in ("recall", "precision", "f1")):
                direction = "higher"
            else:
                continue
        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=tol_frac)
        checked += 1
        if not ok:
            failures.append(f"{key}: current={cur:.4f} baseline={base:.4f} ({msg})")

    assert checked > 0, "no numeric metrics were compared; check baseline/current dicts"
    assert not failures, "outlier metric regressions:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 1: tiny self-contained regression (integration + slow, no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
def test_outliers_pipeline_tiny_regression(tmp_path):
    """
    Tiny end-to-end outlier detection regression.

    Creates a 500-image synthetic dataset (grid_size=32, 20 % outliers),
    runs pipeline_with_outliers for 1 round, then verifies:
      • expected output files exist
      • inliers + outliers == total images
      • recall ≥ MIN_RECALL_TINY
      • precision ≥ MIN_PRECISION_TINY
      • metrics consistent with committed tiny_baseline.json (if present)

    Set OUTLIERS_WRITE_TINY_BASELINE=1 to regenerate the tiny baseline.
    """
    write_baseline = os.environ.get("OUTLIERS_WRITE_TINY_BASELINE", "0") == "1"

    output_dir = _resolve_output_dir(tmp_path, "outliers_tiny")

    pipeline_out = _run_outliers_pipeline(
        output_dir=output_dir,
        grid_size=TINY_GRID_SIZE,
        n_images=TINY_N_IMAGES,
        percent_outliers=TINY_PERCENT_OUTLIERS,
        k_rounds=TINY_K_ROUNDS,
    )

    sim_info_path = output_dir / "test_dataset" / "simulation_info.pkl"
    assert sim_info_path.exists(), f"simulation_info.pkl missing at {sim_info_path}"

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    # Check all expected files exist
    _check_output_files_exist(pipeline_out, k_rounds=TINY_K_ROUNDS)

    # Check partition consistency
    _check_partition_consistency(pipeline_out, n_total=n_total, k_rounds=TINY_K_ROUNDS)

    # Compute metrics and assert minimum thresholds
    metrics = _compute_outlier_metrics(pipeline_out, sim_info_path, k_rounds=TINY_K_ROUNDS)

    recall = metrics.get(f"outlier_recall_round_{TINY_K_ROUNDS}", 0.0)
    precision = metrics.get(f"outlier_precision_round_{TINY_K_ROUNDS}", 0.0)
    assert recall >= MIN_RECALL_TINY, (
        f"recall={recall:.3f} below minimum {MIN_RECALL_TINY}. "
        f"metrics={metrics}"
    )
    assert precision >= MIN_PRECISION_TINY, (
        f"precision={precision:.3f} below minimum {MIN_PRECISION_TINY}. "
        f"metrics={metrics}"
    )

    # Baseline comparison
    _compare_against_baseline(
        current=metrics,
        baseline_path=_TINY_BASELINE_JSON,
        tol_frac=0.25,  # allow 25 % degradation from tiny-dataset noise
        write=write_baseline,
    )


# ---------------------------------------------------------------------------
# Test 1b: fast high-SNR smoke test (tiny_metrics, no baseline needed)
# ---------------------------------------------------------------------------


@pytest.mark.tiny_metrics
@pytest.mark.gpu
def test_outliers_pipeline_fast_smoke(tmp_path):
    """
    Fast outlier detection smoke test with high SNR.

    Uses a very small dataset (200 images, grid_size=32) with very low noise
    (noise_level=0.01) and 25% outliers so detection should be easy and fast.
    No baseline comparison — just checks that precision and recall exceed
    reasonable thresholds.

    Run with: pytest --run-tiny-metrics -k test_outliers_pipeline_fast_smoke
    """
    output_dir = _resolve_output_dir(tmp_path, "outliers_fast_smoke")

    pipeline_out = _run_outliers_pipeline(
        output_dir=output_dir,
        grid_size=FAST_GRID_SIZE,
        n_images=FAST_N_IMAGES,
        percent_outliers=FAST_PERCENT_OUTLIERS,
        k_rounds=FAST_K_ROUNDS,
        extra_args=f"--noise-level {FAST_NOISE_LEVEL}",
    )

    sim_info_path = output_dir / "test_dataset" / "simulation_info.pkl"
    assert sim_info_path.exists(), f"simulation_info.pkl missing at {sim_info_path}"

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    _check_output_files_exist(pipeline_out, k_rounds=FAST_K_ROUNDS)
    _check_partition_consistency(pipeline_out, n_total=n_total, k_rounds=FAST_K_ROUNDS)

    metrics = _compute_outlier_metrics(pipeline_out, sim_info_path, k_rounds=FAST_K_ROUNDS)

    recall = metrics.get(f"outlier_recall_round_{FAST_K_ROUNDS}", 0.0)
    precision = metrics.get(f"outlier_precision_round_{FAST_K_ROUNDS}", 0.0)
    f1 = metrics.get(f"outlier_f1_round_{FAST_K_ROUNDS}", 0.0)

    print(f"\nFast smoke test metrics: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(f"Full metrics: {json.dumps(metrics, indent=2)}")

    assert recall >= MIN_RECALL_FAST, (
        f"recall={recall:.3f} below minimum {MIN_RECALL_FAST}. "
        f"High-SNR outliers should be easy to detect. metrics={metrics}"
    )
    assert precision >= MIN_PRECISION_FAST, (
        f"precision={precision:.3f} below minimum {MIN_PRECISION_FAST}. "
        f"High-SNR outliers should be easy to detect. metrics={metrics}"
    )


# ---------------------------------------------------------------------------
# Test 2: long regression against user-provided dataset (integration+slow+gpu)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.long_test
def test_outliers_pipeline_regression_against_baseline(tmp_path):
    """
    Long regression test for the outliers pipeline (SPA).

    Volumes are generated synthetically so no external data path is required.

    Optional env vars:
      OUTLIERS_VOLUMES_DIR     – if set, use real volumes instead of synthetic ones
      OUTLIERS_BASELINE_JSON   – baseline path; defaults to in-repo
                                 tests/baselines/run_test_outliers_pipeline/long_generated/all_scores.json
      OUTLIERS_WRITE_BASELINE  – "1" to regenerate baseline and skip
      OUTLIERS_N_IMAGES        – (default 10000)
      OUTLIERS_GRID_SIZE       – (default 128)
      OUTLIERS_PERCENT_OUTLIERS – (default 0.15)
      OUTLIERS_K_ROUNDS        – (default 2)
      OUTLIERS_TOL_FRAC        – (default 0.15)
    """
    volumes_prefix = os.environ.get("OUTLIERS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid OUTLIERS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(os.environ.get("OUTLIERS_BASELINE_JSON", str(_DEFAULT_OUTLIERS_BASELINE_JSON)))

    grid_size = int(os.environ.get("OUTLIERS_GRID_SIZE", "128"))
    n_images = int(os.environ.get("OUTLIERS_N_IMAGES", "10000"))
    pct_out = float(os.environ.get("OUTLIERS_PERCENT_OUTLIERS", "0.15"))
    k_rounds = int(os.environ.get("OUTLIERS_K_ROUNDS", "2"))
    tol_frac = float(os.environ.get("OUTLIERS_TOL_FRAC", "0.15"))
    write_baseline = os.environ.get("OUTLIERS_WRITE_BASELINE", "0") == "1"

    output_dir = _resolve_output_dir(tmp_path, "outliers_long")
    reuse = not write_baseline and _dataset_exists_spa(output_dir, grid_size)
    pipeline_out = _run_outliers_pipeline(
        output_dir=output_dir,
        volumes_prefix=volumes_prefix,
        grid_size=grid_size,
        n_images=n_images,
        percent_outliers=pct_out,
        k_rounds=k_rounds,
        accept_cpu=False,
        reuse_dataset=reuse,
    )

    sim_info_path = output_dir / "test_dataset" / "simulation_info.pkl"
    assert sim_info_path.exists()

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    _check_output_files_exist(pipeline_out, k_rounds=k_rounds)
    _check_partition_consistency(pipeline_out, n_total=n_total, k_rounds=k_rounds)

    metrics = _compute_outlier_metrics(pipeline_out, sim_info_path, k_rounds=k_rounds)
    _compare_against_baseline(
        current=metrics,
        baseline_path=baseline_json,
        tol_frac=tol_frac,
        write=write_baseline,
    )


# ---------------------------------------------------------------------------
# Test 3: cryo-ET variant (tilt series) long regression
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.long_test
def test_outliers_pipeline_cryo_et_regression_against_baseline(tmp_path):
    """
    Long cryo-ET regression test for the outliers pipeline.

    Volumes are generated synthetically so no external data path is required.

    Optional env vars (same pattern as the SPA test), plus:
      OUTLIERS_VOLUMES_DIR      – if set, use real volumes instead of synthetic ones
      OUTLIERS_ET_BASELINE_JSON – baseline path; defaults to in-repo
                                  tests/baselines/run_test_outliers_pipeline/long_generated/all_scores_cryo_et.json
      OUTLIERS_N_TILTS          – tilts per particle (default 7)
      OUTLIERS_PCT_TILT_OUTLIERS – fraction of tilt-level outliers (default 0.10)
    """
    volumes_prefix = os.environ.get("OUTLIERS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid OUTLIERS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(
        os.environ.get("OUTLIERS_ET_BASELINE_JSON", str(_DEFAULT_OUTLIERS_ET_BASELINE_JSON))
    )
    grid_size = int(os.environ.get("OUTLIERS_GRID_SIZE", "128"))
    n_images = int(os.environ.get("OUTLIERS_N_IMAGES", "10000"))
    pct_out = float(os.environ.get("OUTLIERS_PERCENT_OUTLIERS", "0.15"))
    pct_tilt_out = float(os.environ.get("OUTLIERS_PCT_TILT_OUTLIERS", "0.10"))
    n_tilts = int(os.environ.get("OUTLIERS_N_TILTS", "7"))
    k_rounds = int(os.environ.get("OUTLIERS_K_ROUNDS", "2"))
    tol_frac = float(os.environ.get("OUTLIERS_TOL_FRAC", "0.15"))
    write_baseline = (
        os.environ.get("OUTLIERS_WRITE_BASELINE", "0") == "1"
        or os.environ.get("OUTLIERS_ET_WRITE_BASELINE", "0") == "1"
    )

    output_dir = _resolve_output_dir(tmp_path, "outliers_cryo_et")
    dataset_dir = output_dir / "test_dataset"
    reuse = not write_baseline and _dataset_exists_et(output_dir)

    if not reuse:
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create outlier volume
        from recovar.commands.run_test_outliers_pipeline import create_outlier_volume
        outlier_vol = output_dir / "outlier_volume.mrc"
        create_outlier_volume(str(outlier_vol), grid_size=grid_size)

        # Generate tilt series dataset with both particle and tilt outliers
        make_cmd = [
            sys.executable, "-m", "recovar.command_line", "make_test_dataset",
            str(output_dir),
            "--n-images", str(n_images),
            "--outlier-file-input", str(outlier_vol),
            "--percent-outliers", str(pct_out),
            "--percent-tilt-series-outliers", str(pct_tilt_out),
            "--tilt-series",
            "--image-size", str(grid_size),
            "--seed", "42",
        ]
        subprocess.run(make_cmd, check=True, env=gpu_subprocess_env())

    # Run pipeline_with_outliers for tilt series
    pipeline_out = output_dir / "pipeline_outliers_output"
    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"

    pipe_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline_with_outliers",
        str(star),
        "--poses", str(poses),
        "--ctf", str(ctf),
        "--tilt-series",
        "--tilt-series-ctf", "relion5",
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
    subprocess.run(pipe_cmd, check=True, env=gpu_subprocess_env())

    sim_info_path = dataset_dir / "simulation_info.pkl"
    assert sim_info_path.exists()

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)
    n_total = int(np.asarray(sim_info["image_assignment"]).size)

    _check_output_files_exist(pipeline_out, k_rounds=k_rounds)
    _check_partition_consistency(pipeline_out, n_total=n_total, k_rounds=k_rounds)

    # Image-level metrics
    image_metrics = _compute_outlier_metrics(pipeline_out, sim_info_path, k_rounds=k_rounds)

    # Particle-level metrics (cryo-ET specific)
    particle_metrics = _compute_particle_outlier_metrics(
        pipeline_out, sim_info, k_rounds=k_rounds
    )
    all_metrics = {**image_metrics, **particle_metrics}

    _compare_against_baseline(
        current=all_metrics,
        baseline_path=baseline_json,
        tol_frac=tol_frac,
        write=write_baseline,
    )


def _compute_particle_outlier_metrics(
    pipeline_out_dir: Path,
    sim_info: dict,
    k_rounds: int,
) -> Dict[str, float]:
    """Particle-level precision/recall for cryo-ET outlier detection."""
    tilt_series_assignment = np.asarray(sim_info.get("tilt_series_assignment", []))
    if tilt_series_assignment.size == 0:
        return {}

    n_particles = int(tilt_series_assignment.size)
    true_particle_outliers = set(
        int(i) for i in np.where(tilt_series_assignment < 0)[0]
    )

    metrics: Dict[str, float] = {
        "total_particles": float(n_particles),
        "true_particle_outlier_count": float(len(true_particle_outliers)),
    }

    for r in range(1, k_rounds + 1):
        inliers_file = pipeline_out_dir / f"particle_inliers_round_{r}.pkl"
        if not inliers_file.exists():
            continue
        with open(inliers_file, "rb") as f:
            particle_inliers = np.asarray(pickle.load(f), dtype=np.int64)

        detected_particle_outliers = set(
            int(i)
            for i in np.setdiff1d(np.arange(n_particles), particle_inliers)
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
        metrics[f"particle_outlier_count_round_{r}"] = float(
            len(detected_particle_outliers)
        )

    return metrics
