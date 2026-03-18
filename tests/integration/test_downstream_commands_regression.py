"""Regression tests for downstream commands.

These tests run the full pipeline + downstream commands on a small synthetic
dataset and verify that key numerical properties hold.  Unlike the GPU
integration tests (crash tests), these check *quality*: volume norms,
embedding statistics, density properties, etc.

Baselines are committed to the repo at:
    tests/baselines/downstream_commands/<test_name>.json
Tests always compare against the committed baselines and fail if a baseline
file is missing (run scripts/capture_downstream_baseline.py to regenerate).

Run with:
    pytest tests/integration/test_downstream_commands_regression.py \
        --run-gpu --run-integration --run-slow
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction, log_comparison_table

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

_N_IMAGES = 500
_GRID = 64
_ZDIM = 4
_N_CLUSTERS = 5
_TOL_FRAC = float(os.environ.get("DOWNSTREAM_REGRESSION_TOL_FRAC", "0.10"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines" / "downstream_commands"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_env():
    return dict(os.environ, PYTHONNOUSERSITE="1",
                XLA_PYTHON_CLIENT_PREALLOCATE="false")


def _run(cmd, **kwargs):
    kwargs.setdefault("env", _gpu_env())
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-80:])
        pytest.fail(
            f"Command failed (rc={result.returncode}):\n  {' '.join(cmd[:6])}...\n"
            f"--- stderr (last 80 lines) ---\n{tail}"
        )
    return result


def _load_baseline(name: str) -> dict:
    """Load a committed baseline JSON. Fails if missing."""
    path = _BASELINE_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"Baseline file missing: {path}\n"
            f"Generate it by running the test with DOWNSTREAM_OVERWRITE_BASELINE=1"
        )
    with open(path) as f:
        return json.load(f)


def _save_baseline(name: str, data: dict):
    """Save a baseline JSON (only when DOWNSTREAM_OVERWRITE_BASELINE is set)."""
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)


def _check_regression(name: str, current: dict, tol_frac: float = _TOL_FRAC):
    """Compare current metrics against committed baseline."""
    overwrite = os.environ.get("DOWNSTREAM_OVERWRITE_BASELINE")

    if overwrite:
        _save_baseline(name, current)
        if not (_BASELINE_DIR / f"{name}.json").exists():
            pytest.skip(f"Wrote new baseline for {name}")
            return

    baseline = _load_baseline(name)

    checked, failures = log_comparison_table(current, baseline, tol_frac, title=f"Downstream: {name}")
    assert checked > 0, "no metrics compared"
    assert not failures, "regressions:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("downstream_regression")


@pytest.fixture(scope="module")
def dataset_dir(shared_dir):
    """Generate a synthetic dataset and return its path."""
    from recovar.commands.run_test_outliers_pipeline import create_outlier_volume

    outlier_vol = shared_dir / "outlier_volume.mrc"
    create_outlier_volume(str(outlier_vol), grid_size=_GRID)

    make_cmd = [
        sys.executable, "-m", "recovar.command_line", "make_test_dataset",
        str(shared_dir),
        "--n-images", str(_N_IMAGES),
        "--outlier-file-input", str(outlier_vol),
        "--percent-outliers", "0.20",
        "--image-size", str(_GRID),
        "--seed", "42",
    ]
    _run(make_cmd)
    ds = shared_dir / "test_dataset"
    assert ds.exists()
    return ds


@pytest.fixture(scope="module")
def pipeline_output(dataset_dir, shared_dir):
    """Run pipeline and return output dir."""
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
        sys.executable, "-m", "recovar.command_line", "pipeline",
        str(mrcs),
        "--poses", str(poses),
        "--ctf", str(ctf),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", mask_path,
        "--lazy",
        "--zdim", str(_ZDIM),
    ]
    _run(cmd)
    assert (pipeline_out / "model" / "params.pkl").exists()
    return pipeline_out


@pytest.fixture(scope="module")
def analyze_output(pipeline_output, shared_dir):
    """Run analyze and return output dir."""
    analyze_out = shared_dir / "analysis_output"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "analyze",
        str(pipeline_output),
        "--zdim", str(_ZDIM),
        "-o", str(analyze_out),
        "--n-clusters", str(_N_CLUSTERS),
        "--skip-umap",
        "--lazy",
    ]
    _run(cmd)
    return analyze_out


@pytest.fixture(scope="module")
def density_output(pipeline_output, shared_dir):
    """Run density estimation and return output dir."""
    density_out = shared_dir / "density_output"
    cmd = [
        sys.executable, "-m", "recovar.command_line",
        "estimate_conformational_density",
        str(pipeline_output),
        "--output_dir", str(density_out),
        "--pca_dim", "2",
    ]
    _run(cmd)
    return density_out


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_analyze_kmeans_regression(analyze_output):
    """Check k-means cluster quality: centers spread, label coverage."""
    from recovar import utils
    result = utils.pickle_load(str(analyze_output / "kmeans_result.pkl"))
    centers = np.asarray(result["centers"])
    labels = np.asarray(result["labels"])

    metrics = {
        "n_clusters": int(centers.shape[0]),
        "centers_mean_norm": float(np.mean(np.linalg.norm(centers, axis=1))),
        "centers_std_norm": float(np.std(np.linalg.norm(centers, axis=1))),
        "label_coverage": float(np.sum(~np.isnan(labels)) / labels.size),
        "n_unique_labels": int(len(np.unique(labels[~np.isnan(labels)]))),
        "centers_max_pairwise_dist": float(
            np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1))
        ),
    }

    # Sanity checks that must always hold
    assert metrics["n_clusters"] == _N_CLUSTERS
    assert metrics["label_coverage"] > 0.5, "Less than 50% of images got cluster labels"
    assert metrics["n_unique_labels"] == _N_CLUSTERS
    assert metrics["centers_max_pairwise_dist"] > 0, "All k-means centers collapsed"

    _check_regression("analyze_kmeans", metrics)


def test_compute_state_volume_quality(pipeline_output, analyze_output, shared_dir):
    """Check that compute_state produces volumes with reasonable properties."""
    centers_txt = analyze_output / "kmeans" / "centers.txt"
    centers = np.loadtxt(str(centers_txt))
    pts_path = shared_dir / "regression_latent_points.txt"
    np.savetxt(str(pts_path), centers[:2])

    state_out = shared_dir / "regression_compute_state"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "compute_state",
        str(pipeline_output),
        "-o", str(state_out),
        "--latent-points", str(pts_path),
        "--lazy",
    ]
    _run(cmd)

    from recovar import utils
    # compute_state names files state000.mrc, state001.mrc, ...
    # Exclude half-map files (*_half*_unfil.mrc)
    mrc_files = sorted(
        p for p in state_out.glob("state*.mrc")
        if "_half" not in p.name
    )
    assert len(mrc_files) >= 2, f"Expected at least 2 volumes, got {len(mrc_files)}"

    metrics = {}
    vols = []
    for i, mrc_path in enumerate(mrc_files[:2]):
        vol = utils.load_mrc(str(mrc_path))
        vols.append(vol)
        metrics[f"vol_{i}_mean"] = float(np.mean(vol))
        metrics[f"vol_{i}_std"] = float(np.std(vol))
        metrics[f"vol_{i}_max"] = float(np.max(np.abs(vol)))

    # Volumes at different latent points should differ
    if len(vols) >= 2:
        diff_norm = float(np.linalg.norm(vols[0] - vols[1]))
        metrics["vol_pair_diff_norm"] = diff_norm
        assert diff_norm > 0, "Volumes at different latent points are identical"

    # All volumes should have non-zero content
    for i in range(min(2, len(vols))):
        assert metrics[f"vol_{i}_std"] > 0, f"Volume {i} is constant"

    _check_regression("compute_state_volumes", metrics)


def test_compute_trajectory_regression(pipeline_output, analyze_output, shared_dir):
    """Check that trajectory volumes vary smoothly along the path."""
    centers_txt = analyze_output / "kmeans" / "centers.txt"

    traj_out = shared_dir / "regression_trajectory"
    n_vols = 4
    cmd = [
        sys.executable, "-m", "recovar.command_line", "compute_trajectory",
        str(pipeline_output),
        "-o", str(traj_out),
        "--zdim", str(_ZDIM),
        "--endpts", str(centers_txt),
        "--ind", "0,1",
        "--n-vols-along-path", str(n_vols),
        "--lazy",
    ]
    _run(cmd)

    from recovar import utils
    # compute_trajectory names files state000.mrc, state001.mrc, ...
    # For zdim>1, files are at output root; for zdim==1, under path0/
    # Exclude half-map files (*_half*_unfil.mrc)
    mrc_files = sorted(
        p for p in traj_out.rglob("state*.mrc")
        if "_half" not in p.name
    )
    assert len(mrc_files) >= n_vols, (
        f"Expected at least {n_vols} trajectory volumes, got {len(mrc_files)}"
    )

    vols = [utils.load_mrc(str(p)) for p in mrc_files[:n_vols]]
    metrics = {}
    for i, vol in enumerate(vols):
        metrics[f"traj_vol_{i}_std"] = float(np.std(vol))

    # Check variation: consecutive volumes should differ
    consecutive_diffs = []
    for i in range(len(vols) - 1):
        d = float(np.linalg.norm(vols[i] - vols[i + 1]))
        consecutive_diffs.append(d)
        metrics[f"traj_consecutive_diff_{i}"] = d
    metrics["traj_mean_consecutive_diff"] = float(np.mean(consecutive_diffs))

    end_to_end = float(np.linalg.norm(vols[0] - vols[-1]))
    metrics["traj_end_to_end_diff"] = end_to_end

    # On small synthetic datasets, some consecutive volumes may be very close
    # but start and end should differ
    assert end_to_end > 0, "Start and end trajectory volumes are identical"
    assert any(d > 0 for d in consecutive_diffs), "All consecutive trajectory volumes are identical"

    _check_regression("compute_trajectory", metrics)


def test_density_estimation_regression(density_output):
    """Check that estimated density has reasonable properties."""
    from recovar import utils
    knee_pkl = density_output / "deconv_density_knee.pkl"
    assert knee_pkl.exists()

    result = utils.pickle_load(str(knee_pkl))
    density = np.asarray(result["density"])
    bounds = np.asarray(result["latent_space_bounds"])

    metrics = {
        "density_ndim": int(density.ndim),
        "density_shape_0": int(density.shape[0]),
        "density_sum": float(np.sum(density)),
        "density_max": float(np.max(density)),
        "density_min": float(np.min(density)),
        "density_mean": float(np.mean(density)),
        "density_std": float(np.std(density)),
        "density_positive_frac": float(np.mean(density > 0)),
        "bounds_range": float(np.max(bounds) - np.min(bounds)),
    }

    # Density should have meaningful structure
    assert density.ndim == 2, f"Expected 2D density (pca_dim=2), got {density.ndim}D"
    assert metrics["density_max"] > 0, "Density is non-positive everywhere"
    assert metrics["density_std"] > 0, "Density is constant"

    _check_regression("density_estimation", metrics)


def test_stable_states_regression(density_output, shared_dir):
    """Check that stable states are found at reasonable locations."""
    knee_pkl = density_output / "deconv_density_knee.pkl"

    stable_out = shared_dir / "regression_stable_states"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "estimate_stable_states",
        str(knee_pkl),
        "-o", str(stable_out),
        "--n_local_maxs", "3",
    ]
    _run(cmd)

    coords_file = stable_out / "stable_state_all_coords.txt"
    assert coords_file.exists()

    coords = np.loadtxt(str(coords_file))
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    metrics = {
        "n_stable_states": int(coords.shape[0]),
        "coords_mean_norm": float(np.mean(np.linalg.norm(coords, axis=1))),
        "coords_max_norm": float(np.max(np.linalg.norm(coords, axis=1))),
    }

    if coords.shape[0] >= 2:
        pairwise = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(pairwise, np.inf)
        metrics["min_pairwise_dist"] = float(np.min(pairwise))

    assert metrics["n_stable_states"] >= 1, "No stable states found"
    assert np.all(np.isfinite(coords)), "Stable state coordinates contain NaN/Inf"

    _check_regression("stable_states", metrics)


def test_extract_kmeans_subset_regression(analyze_output, shared_dir):
    """Check that k-means subset extraction produces consistent results."""
    from recovar import utils

    kmeans_pkl = analyze_output / "kmeans_result.pkl"
    subset_out = shared_dir / "regression_kmeans_subset.pkl"

    cmd = [
        sys.executable, "-m", "recovar.command_line",
        "extract_image_subset_from_kmeans",
        str(kmeans_pkl), str(subset_out), "0,1",
    ]
    _run(cmd)

    indices = utils.pickle_load(str(subset_out))
    kmeans_result = utils.pickle_load(str(kmeans_pkl))
    labels = np.asarray(kmeans_result["labels"])
    n_valid = int(np.sum(~np.isnan(labels)))

    metrics = {
        "n_selected": int(len(indices)),
        "n_valid_labels": n_valid,
        "selection_fraction": float(len(indices) / n_valid) if n_valid > 0 else 0.0,
    }

    assert metrics["n_selected"] > 0, "No images selected"
    assert metrics["selection_fraction"] < 1.0, "All images selected (should be subset)"
    assert metrics["selection_fraction"] > 0.0, "No images selected"

    _check_regression("extract_kmeans_subset", metrics)
