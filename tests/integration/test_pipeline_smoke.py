"""
Tiny smoke tests for the full pipeline end-to-end.

These tests run the complete pipeline with an extremely small synthetic
dataset (default: ~50 images, grid_size=16, 1 round, CPU) to verify that
the pipeline completes without crashing. They are NOT quality regression
tests — no baseline comparison is performed. The goal is a fast sanity
check during development (< 2 min on CPU).

SPA and cryo-ET variants are both covered.

Env vars (optional):
  SMOKE_N_IMAGES      – number of images for SPA test (default: 50)
  SMOKE_N_IMAGES_ET   – number of images for cryo-ET test (default: 270;
                        make_test_dataset hard-codes n_tilts=27, so 270 → 10 particles)
  SMOKE_GRID_SIZE     – grid size (default: 16)
  SMOKE_ZDIM          – latent dimensionality (default: 4)
  SMOKE_K_ROUNDS      – outlier-removal rounds (default: 1)
  SMOKE_PCT_OUTLIERS  – outlier fraction (default: 0.20)

Run with:
  pytest tests/integration/test_pipeline_smoke.py --run-integration --run-slow
  # no --run-gpu needed: tests use --accept-cpu by default
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.integration]

# Force CPU for all subprocesses spawned by smoke tests (no GPU required).
# Include PYTHONNOUSERSITE=1 to prevent importing the wrong recovar from user-site.
_CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1")


def _run(cmd, **kwargs):
    """Run a subprocess and include stderr in the failure message on crash."""
    kwargs.setdefault("env", _CPU_ENV)
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        # Show the last 80 lines of stderr for quick diagnosis
        tail = "\n".join(result.stderr.splitlines()[-80:])
        pytest.fail(
            f"Command failed (rc={result.returncode}):\n  {' '.join(cmd[:6])}...\n"
            f"--- stderr (last 80 lines) ---\n{tail}"
        )


_SMOKE_N_IMAGES = int(os.environ.get("SMOKE_N_IMAGES", "50"))
# For cryo-ET, make_test_dataset hard-codes n_tilts=27, so we need more
# images to get a useful number of particles (270 // 27 = 10 particles).
_SMOKE_N_IMAGES_ET = int(os.environ.get("SMOKE_N_IMAGES_ET", "270"))
_SMOKE_GRID = int(os.environ.get("SMOKE_GRID_SIZE", "16"))
_SMOKE_ZDIM = int(os.environ.get("SMOKE_ZDIM", "4"))
_SMOKE_K_ROUNDS = int(os.environ.get("SMOKE_K_ROUNDS", "1"))
_SMOKE_PCT_OUTLIERS = float(os.environ.get("SMOKE_PCT_OUTLIERS", "0.20"))


def _smoke_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    out = Path(base) / "pytest_smoke" / name if base else tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_outlier_vol(output_dir: Path) -> Path:
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(vol), grid_size=_SMOKE_GRID)
    return vol


def _generate_dataset(
    output_dir: Path,
    outlier_vol: Path,
    tilt_series: bool = False,
    n_images: int | None = None,
) -> Path:
    """Generate a minimal synthetic SPA or cryo-ET dataset."""
    dataset_dir = output_dir / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if n_images is None:
        n_images = _SMOKE_N_IMAGES_ET if tilt_series else _SMOKE_N_IMAGES

    make_cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "make_test_dataset",
        str(output_dir),
        "--n-images",
        str(n_images),
        "--outlier-file-input",
        str(outlier_vol),
        "--percent-outliers",
        str(_SMOKE_PCT_OUTLIERS),
        "--image-size",
        str(_SMOKE_GRID),
        "--seed",
        "42",
    ]
    if tilt_series:
        make_cmd += ["--tilt-series"]
    _run(make_cmd)
    return dataset_dir


# ---------------------------------------------------------------------------
# Test 1: SPA smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_spa_smoke(tmp_path):
    """
    Tiny end-to-end smoke test for the SPA pipeline.

    Generates a synthetic dataset with ~100 images (grid_size=32) and runs
    pipeline_with_outliers on CPU. Checks that expected output files exist.
    No quality baseline comparison — this is purely a "does it crash?" check.
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_spa")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol)

    mrcs = dataset_dir / f"particles.{_SMOKE_GRID}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(mrcs),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    # Verify expected output files exist
    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()

    # Verify partition consistency: round 1 inliers + outliers = n_images
    with open(pipeline_out / "inliers_round_1.pkl", "rb") as f:
        inliers = np.asarray(pickle.load(f))
    with open(pipeline_out / "outliers_round_1.pkl", "rb") as f:
        outliers = np.asarray(pickle.load(f))
    assert inliers.size + outliers.size == _SMOKE_N_IMAGES, (
        f"inliers({inliers.size}) + outliers({outliers.size}) != n_images({_SMOKE_N_IMAGES})"
    )


# ---------------------------------------------------------------------------
# Test 2: cryo-ET smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_cryo_et_smoke(tmp_path):
    """
    Tiny end-to-end smoke test for the cryo-ET (tilt series) pipeline.

    Generates a synthetic tilt-series dataset and runs pipeline_with_outliers
    on CPU. Checks that expected output files exist (image-level and
    particle-level indices).
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_cryo_et")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol, tilt_series=True)

    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_et_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(star),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    # Verify image-level output files
    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()


# ---------------------------------------------------------------------------
# Test 3: cryo-ET with radial_per_tilt noise model
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_cryo_et_radial_per_tilt_noise_smoke(tmp_path):
    """
    Smoke test for cryo-ET with --noise-model radial_per_tilt.

    This was crashing with IndexError because VariableRadialNoiseModel
    received a 1D noise array but expected 2D (n_dose_levels, n_radial_bins).
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_et_radial_per_tilt")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol, tilt_series=True)

    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_et_rpt_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(star),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--noise-model",
        "radial_per_tilt",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()


# ---------------------------------------------------------------------------
# Test 4: cryo-ET with premultiplied CTF
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_cryo_et_premultiplied_ctf_smoke(tmp_path):
    """
    Smoke test for cryo-ET with --premultiplied-ctf.
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_et_premultiplied_ctf")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol, tilt_series=True)

    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_et_pctf_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(star),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--premultiplied-ctf",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()


# ---------------------------------------------------------------------------
# Test 5: cryo-ET with radial_per_tilt + premultiplied CTF combined
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_cryo_et_radial_per_tilt_premultiplied_ctf_smoke(tmp_path):
    """
    Smoke test for cryo-ET with both --noise-model radial_per_tilt and
    --premultiplied-ctf, the most complex configuration.
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_et_rpt_pctf")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol, tilt_series=True)

    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_et_rpt_pctf_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(star),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--noise-model",
        "radial_per_tilt",
        "--premultiplied-ctf",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()


# ---------------------------------------------------------------------------
# Test 6: SPA with radial noise (baseline comparison to cryo-ET variants)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_spa_radial_noise_smoke(tmp_path):
    """
    Smoke test for SPA with explicit --noise-model radial, to ensure
    the noise model path works end-to-end for SPA too.
    """
    output_dir = _smoke_output_dir(tmp_path, "smoke_spa_radial")
    outlier_vol = _make_outlier_vol(output_dir)
    dataset_dir = _generate_dataset(output_dir, outlier_vol)

    mrcs = dataset_dir / f"particles.{_SMOKE_GRID}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_smoke_spa_radial_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(mrcs),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--noise-model",
        "radial",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_SMOKE_ZDIM),
        "--k-rounds",
        str(_SMOKE_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        "--accept-cpu",
        "--gpu-gb",
        "8",
    ]
    _run(cmd)

    assert (pipeline_out / f"inliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_SMOKE_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()


# ---------------------------------------------------------------------------
# GPU pipeline tests — larger datasets, require --run-gpu
# ---------------------------------------------------------------------------

_GPU_N_IMAGES = 500
_GPU_N_IMAGES_ET = 270  # 27 tilts × 10 particles
_GPU_GRID = 64
_GPU_ZDIM = 4
_GPU_K_ROUNDS = 1
_GPU_PCT_OUTLIERS = 0.20


def _gpu_env():
    """Env for GPU subprocesses with the test CUDA library when available."""
    return gpu_subprocess_env()


def _generate_gpu_dataset(
    output_dir: Path,
    outlier_vol: Path,
    tilt_series: bool = False,
) -> Path:
    dataset_dir = output_dir / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    n_images = _GPU_N_IMAGES_ET if tilt_series else _GPU_N_IMAGES
    make_cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "make_test_dataset",
        str(output_dir),
        "--n-images",
        str(n_images),
        "--outlier-file-input",
        str(outlier_vol),
        "--percent-outliers",
        str(_GPU_PCT_OUTLIERS),
        "--image-size",
        str(_GPU_GRID),
        "--seed",
        "42",
    ]
    if tilt_series:
        make_cmd += ["--tilt-series"]
    _run(make_cmd, env=_gpu_env())
    return dataset_dir


def _make_gpu_outlier_vol(output_dir: Path) -> Path:
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    vol = output_dir / "outlier_volume.mrc"
    create_outlier_volume(str(vol), grid_size=_GPU_GRID)
    return vol


@pytest.mark.gpu
def test_pipeline_spa_gpu(tmp_path):
    """SPA pipeline on GPU with a reasonably sized dataset (500 images, grid=64)."""
    output_dir = _smoke_output_dir(tmp_path, "gpu_spa")
    outlier_vol = _make_gpu_outlier_vol(output_dir)
    dataset_dir = _generate_gpu_dataset(output_dir, outlier_vol)

    mrcs = dataset_dir / f"particles.{_GPU_GRID}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_gpu_spa_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(mrcs),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_GPU_ZDIM),
        "--k-rounds",
        str(_GPU_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ]
    _run(cmd, env=_gpu_env())

    assert (pipeline_out / f"inliers_round_{_GPU_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_GPU_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()

    with open(pipeline_out / "inliers_round_1.pkl", "rb") as f:
        inliers = np.asarray(pickle.load(f))
    with open(pipeline_out / "outliers_round_1.pkl", "rb") as f:
        outliers = np.asarray(pickle.load(f))
    assert inliers.size + outliers.size == _GPU_N_IMAGES


@pytest.mark.gpu
def test_pipeline_cryo_et_gpu(tmp_path):
    """Cryo-ET pipeline on GPU with a reasonably sized dataset (270 images, grid=64)."""
    output_dir = _smoke_output_dir(tmp_path, "gpu_cryo_et")
    outlier_vol = _make_gpu_outlier_vol(output_dir)
    dataset_dir = _generate_gpu_dataset(output_dir, outlier_vol, tilt_series=True)

    star = dataset_dir / "particles.star"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = dataset_dir / "pipeline_gpu_et_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(star),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--noise-model",
        "radial_per_tilt",
        "--premultiplied-ctf",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        "from_halfmaps",
        "--lazy",
        "--zdim",
        str(_GPU_ZDIM),
        "--k-rounds",
        str(_GPU_K_ROUNDS),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
        # With only 10 particles (5 per halfset), the default ~183 PCs is
        # massively underdetermined for projected covariance.  Cap at 30.
        "--very-low-memory-option",
    ]
    _run(cmd, env=_gpu_env())

    assert (pipeline_out / f"inliers_round_{_GPU_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / f"outliers_round_{_GPU_K_ROUNDS}.pkl").exists()
    assert (pipeline_out / "all_rounds_inliers.pkl").exists()
