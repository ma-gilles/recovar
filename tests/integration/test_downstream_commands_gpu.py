"""GPU integration tests for downstream commands.

Tests exercise the full command-line interface for downstream commands
(analyze, compute_state, compute_trajectory, estimate_conformational_density,
estimate_stable_states, extract_image_subset_from_kmeans) on GPU with a
synthetically generated dataset.

These are crash tests — they verify that commands complete successfully and
produce expected output files.  No quality regression is checked.

Run with:
    pytest tests/integration/test_downstream_commands_gpu.py --run-gpu --run-integration
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

_N_IMAGES = 500
_GRID = 64
_ZDIM = 4
_N_CLUSTERS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_env():
    """Env for GPU subprocesses: inherit CUDA_VISIBLE_DEVICES, disable preallocate."""
    return dict(os.environ, PYTHONNOUSERSITE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")


def _run(cmd, **kwargs):
    """Run a subprocess and include stderr in the failure message on crash."""
    kwargs.setdefault("env", _gpu_env())
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-80:])
        pytest.fail(
            f"Command failed (rc={result.returncode}):\n  {' '.join(cmd[:6])}...\n"
            f"--- stderr (last 80 lines) ---\n{tail}"
        )
    return result


# ---------------------------------------------------------------------------
# Module-scoped fixtures: generate dataset -> run pipeline -> run analyze
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_dir(tmp_path_factory):
    """Shared temporary directory for all tests in this module."""
    return tmp_path_factory.mktemp("downstream_gpu")


@pytest.fixture(scope="module")
def pipeline_output(shared_dir):
    """Generate dataset and run pipeline, returning the pipeline output dir."""
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    outlier_vol = shared_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=_GRID)

    make_cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "make_test_dataset",
        str(shared_dir),
        "--n-images",
        str(_N_IMAGES),
        "--outlier-file-input",
        str(outlier_vol),
        "--percent-outliers",
        "0.20",
        "--image-size",
        str(_GRID),
        "--seed",
        "42",
    ]
    _run(make_cmd)

    dataset_dir = shared_dir / "test_dataset"
    mrcs = dataset_dir / f"particles.{_GRID}.mrcs"
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    pipeline_out = shared_dir / "pipeline_output"

    # Compute GT union mask from synthetic volumes
    from recovar.simulation import synthetic_dataset
    from recovar.output import metrics
    from recovar import utils

    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    gt_thing = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    volume_shape = (_GRID, _GRID, _GRID)
    gt_union_soft_mask, _ = metrics.make_union_gt_mask_from_hvd(gt_thing, volume_shape)
    mask_path = str(shared_dir / "gt_union_mask.mrc")
    utils.write_mrc(mask_path, gt_union_soft_mask)

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline",
        str(mrcs),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        mask_path,
        "--lazy",
        "--zdim",
        str(_ZDIM),
    ]
    _run(cmd)

    assert (pipeline_out / "model" / "params.pkl").exists(), "Pipeline did not produce params.pkl"
    return pipeline_out


@pytest.fixture(scope="module")
def analyze_output(pipeline_output, shared_dir):
    """Run analyze on the pipeline output, returning the analyze output dir."""
    analyze_out = shared_dir / "analysis_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "analyze",
        str(pipeline_output),
        "--zdim",
        str(_ZDIM),
        "-o",
        str(analyze_out),
        "--n-clusters",
        str(_N_CLUSTERS),
        "--skip-umap",
        "--lazy",
    ]
    _run(cmd)

    assert analyze_out.exists(), "Analyze did not create output directory"
    return analyze_out


@pytest.fixture(scope="module")
def density_output(pipeline_output, shared_dir):
    """Run estimate_conformational_density, returning the output dir."""
    density_out = shared_dir / "density_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "estimate_conformational_density",
        str(pipeline_output),
        "--output_dir",
        str(density_out),
        "--pca_dim",
        "2",
    ]
    _run(cmd)

    assert density_out.exists(), "Density estimation did not create output directory"
    return density_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_produces_expected_outputs(analyze_output):
    """analyze should create data/kmeans_result.pkl and centers.txt."""
    assert (analyze_output / "data" / "kmeans_result.pkl").exists()
    assert (analyze_output / "kmeans" / "centers.txt").exists()


def test_analyze_kmeans_result_structure(analyze_output):
    """kmeans_result.pkl should contain centers and labels."""
    from recovar import utils

    result = utils.pickle_load(str(analyze_output / "data" / "kmeans_result.pkl"))
    assert "centers" in result
    assert "labels" in result
    assert result["centers"].shape[1] == _ZDIM
    assert result["centers"].shape[0] == _N_CLUSTERS


def test_compute_state(pipeline_output, analyze_output, shared_dir):
    """compute_state should produce volumes from latent points."""
    centers_txt = analyze_output / "kmeans" / "centers.txt"
    assert centers_txt.exists()

    # Take just the first 2 centers to keep it fast
    centers = np.loadtxt(str(centers_txt))
    pts_path = shared_dir / "latent_points_2.txt"
    np.savetxt(str(pts_path), centers[:2])

    state_out = shared_dir / "compute_state_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "compute_state",
        str(pipeline_output),
        "-o",
        str(state_out),
        "--latent-points",
        str(pts_path),
        "--lazy",
    ]
    _run(cmd)

    assert state_out.exists()
    # Should produce MRC volumes
    mrc_files = list(state_out.glob("*.mrc"))
    assert len(mrc_files) > 0, "compute_state produced no MRC files"


def test_compute_trajectory(pipeline_output, analyze_output, shared_dir):
    """compute_trajectory should compute trajectory between endpoints."""
    centers_txt = analyze_output / "kmeans" / "centers.txt"
    assert centers_txt.exists()

    traj_out = shared_dir / "compute_trajectory_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "compute_trajectory",
        str(pipeline_output),
        "-o",
        str(traj_out),
        "--zdim",
        str(_ZDIM),
        "--endpts",
        str(centers_txt),
        "--ind",
        "0,1",
        "--n-vols-along-path",
        "3",
        "--lazy",
    ]
    _run(cmd)

    assert traj_out.exists()
    # Should produce MRC volumes along the trajectory
    mrc_files = list(traj_out.rglob("*.mrc"))
    assert len(mrc_files) > 0, "compute_trajectory produced no MRC files"


def test_estimate_conformational_density_produces_outputs(density_output):
    """estimate_conformational_density should produce density pkl files."""
    knee_pkl = density_output / "deconv_density_knee.pkl"
    assert knee_pkl.exists(), "No deconv_density_knee.pkl found"

    from recovar import utils

    result = utils.pickle_load(str(knee_pkl))
    assert "density" in result
    assert "latent_space_bounds" in result


def test_estimate_stable_states(density_output, shared_dir):
    """estimate_stable_states should find stable states from density."""
    knee_pkl = density_output / "deconv_density_knee.pkl"
    assert knee_pkl.exists()

    stable_out = shared_dir / "stable_states_output"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "estimate_stable_states",
        str(knee_pkl),
        "-o",
        str(stable_out),
        "--n_local_maxs",
        "2",
    ]
    _run(cmd)

    assert stable_out.exists()
    assert (stable_out / "stable_state_all_coords.txt").exists()


def test_extract_image_subset_from_kmeans(analyze_output, shared_dir):
    """extract_image_subset_from_kmeans should produce subset indices."""
    kmeans_pkl = analyze_output / "data" / "kmeans_result.pkl"
    assert kmeans_pkl.exists()

    subset_out = shared_dir / "image_subset.pkl"

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "extract_image_subset_from_kmeans",
        str(kmeans_pkl),
        str(subset_out),
        "0,1",
    ]
    _run(cmd)

    assert subset_out.exists()
    from recovar import utils

    indices = utils.pickle_load(str(subset_out))
    assert len(indices) > 0, "No images selected by kmeans subset extraction"


def test_extract_image_subset_from_kmeans_inverse(analyze_output, shared_dir):
    """extract_image_subset_from_kmeans with --inverse should complement the selection."""
    kmeans_pkl = analyze_output / "data" / "kmeans_result.pkl"

    subset_normal = shared_dir / "image_subset_normal.pkl"
    subset_inverse = shared_dir / "image_subset_inverse.pkl"

    cmd_normal = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "extract_image_subset_from_kmeans",
        str(kmeans_pkl),
        str(subset_normal),
        "0",
    ]
    cmd_inverse = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "extract_image_subset_from_kmeans",
        str(kmeans_pkl),
        str(subset_inverse),
        "0",
        "--inverse",
    ]
    _run(cmd_normal)
    _run(cmd_inverse)

    from recovar import utils

    normal = utils.pickle_load(str(subset_normal))
    inverse = utils.pickle_load(str(subset_inverse))
    # Normal + inverse should cover all non-NaN labels
    combined = np.union1d(normal, inverse)
    kmeans_result = utils.pickle_load(str(kmeans_pkl))
    n_valid = np.sum(~np.isnan(kmeans_result["labels"]))
    assert combined.size == n_valid
