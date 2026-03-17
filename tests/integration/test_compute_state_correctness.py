"""Integration tests for compute_state volume reconstruction quality.

Verifies that compute_state produces volumes with good local resolution,
for both SPA and cryo-ET data.  This exercises the full call chain:
  compute_state → compute_and_save_reweighted → make_volumes_kernel_estimate_local
  → even_less_naive_heterogeneity_scheme_relion_style → DataIterator

The cryo-ET variant specifically tests that particle-grouped iteration works
correctly in even_less_naive_heterogeneity_scheme_relion_style (the AKD fix).

Quality is measured by local resolution (locres) from the half-map FSC,
not raw FSC vs GT — locres is a much better metric for reconstruction quality.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import gpu_subprocess_env
from helpers.perf_regression import (
    perf_snapshot, stage_perf, build_perf_record,
    check_perf_regression,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.long_test]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_SIZE = 128
N_IMAGES_SPA = 50000
N_IMAGES_ET = 50000
N_TILTS = 5
NOISE_LEVEL = 0.1
ZDIM = 4
N_CENTERS = 5      # number of k-means cluster centers for compute_state
SEED = 42

# Locres thresholds — median local resolution in Angstroms (lower = better)
# At 128^3 with 50k images and voxel_size=4.25A, typical median locres is ~10-15A.
MAX_LOCRES_MEDIAN_SPA = 20.0   # Angstroms
MAX_LOCRES_MEDIAN_ET = 25.0    # ET is harder (fewer effective images per tilt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out = Path(base) / "pytest_compute_state" / name
    else:
        out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_pipeline_and_compute_state(
    output_dir: Path,
    grid_size: int,
    n_images: int,
    noise_level: float,
    zdim: int,
    n_centers: int,
    *,
    n_tilts: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Generate dataset, run pipeline, run compute_state.

    Returns (test_dataset_dir, pipeline_output_dir, state_output_dir, perf_stages).
    """
    env = gpu_subprocess_env()
    perf_stages = {}

    dataset_dir = output_dir / "dataset"
    test_dataset = dataset_dir / "test_dataset"
    pipeline_output = test_dataset / "pipeline_output"
    state_output = output_dir / "state_output"

    # 1. Generate dataset
    snap = perf_snapshot()
    make_cmd = [
        sys.executable, "-m", "recovar.commands.make_test_dataset",
        str(dataset_dir),
        "--image-size", str(grid_size),
        "--n-images", str(n_images),
        "--noise-level", str(noise_level),
        "--seed", str(seed),
    ]
    if n_tilts is not None:
        make_cmd += ["--n-tilts", str(n_tilts)]
    subprocess.run(make_cmd, check=True, env=env, timeout=600)
    perf_stages["dataset_generation"] = stage_perf(snap, perf_snapshot())

    # 2. Determine particles file
    if n_tilts is not None:
        particles = test_dataset / "particles.star"
    else:
        particles = test_dataset / f"particles.{grid_size}.mrcs"
    assert particles.exists(), f"Missing particles: {particles}"

    # 3. Run pipeline
    snap = perf_snapshot()
    pipeline_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline",
        str(particles),
        "--ctf", str(test_dataset / "ctf.pkl"),
        "--poses", str(test_dataset / "poses.pkl"),
        "--mask", "from_halfmaps",
        "-o", str(pipeline_output),
        "--zdim", str(zdim),
        "--lazy",
        "--correct-contrast",
    ]
    if n_tilts is not None:
        pipeline_cmd += ["--tilt-series", "--tilt-series-ctf", "relion5"]
    subprocess.run(pipeline_cmd, check=True, env=env, timeout=3600)
    perf_stages["pipeline"] = stage_perf(snap, perf_snapshot())
    assert pipeline_output.exists()

    # 4. Generate k-means centers from embedding
    centers_file = output_dir / "centers.txt"
    kmeans_script = f"""\
import numpy as np
from recovar.output import output as out_mod
po = out_mod.PipelineOutput('{pipeline_output}')
zs = np.array(po.get('latent_coords')[{zdim}])
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters={n_centers}, random_state={seed}, batch_size=min(5000, len(zs)))
km.fit(zs)
np.savetxt('{centers_file}', km.cluster_centers_)
"""
    subprocess.run([sys.executable, "-c", kmeans_script], check=True, env=env, timeout=120)
    assert centers_file.exists()

    # 5. Run compute_state
    snap = perf_snapshot()
    state_cmd = [
        sys.executable, "-m", "recovar.command_line", "compute_state",
        str(pipeline_output),
        "-o", str(state_output),
        "--latent-points", str(centers_file),
        "--lazy",
        "--n-bins", "30",
    ]
    subprocess.run(state_cmd, check=True, env=env, timeout=3600)
    perf_stages["compute_state"] = stage_perf(snap, perf_snapshot())
    assert state_output.exists()

    return test_dataset, pipeline_output, state_output, perf_stages


def _compute_locres_metrics(state_output: Path, n_centers: int) -> dict:
    """Extract local resolution metrics from compute_state output.

    Reads the local_resolution.mrc and half-maps from each volume's
    diagnostics directory. Returns median/90th percentile/AUC of locres.
    """
    import mrcfile

    diagnostics = state_output / "diagnostics"
    locres_medians = []
    locres_90pcts = []

    for vol_idx in range(n_centers):
        diag_dir = diagnostics / f"state{vol_idx:03d}"
        locres_mrc = diag_dir / "local_resolution.mrc"

        if not locres_mrc.exists():
            print(f"  Volume {vol_idx}: local_resolution.mrc not found, skipping")
            continue

        with mrcfile.open(str(locres_mrc), mode='r') as mrc:
            locres_map = np.array(mrc.data, dtype=np.float32)

        # Only consider voxels with finite, positive resolution
        valid = locres_map[np.isfinite(locres_map) & (locres_map > 0)]
        if valid.size == 0:
            print(f"  Volume {vol_idx}: no valid locres voxels")
            continue

        median = float(np.median(valid))
        pct90 = float(np.percentile(valid, 90))
        locres_medians.append(median)
        locres_90pcts.append(pct90)
        print(f"  Volume {vol_idx}: locres_median={median:.2f}A, locres_90pct={pct90:.2f}A")

    if not locres_medians:
        return {"locres_median": float('inf'), "locres_90pct": float('inf')}

    result = {
        "locres_median": float(np.mean(locres_medians)),
        "locres_90pct": float(np.mean(locres_90pcts)),
        "per_volume_locres_median": [float(x) for x in locres_medians],
        "per_volume_locres_90pct": [float(x) for x in locres_90pcts],
    }
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compute_state_spa(tmp_path):
    """SPA compute_state: reconstructed volumes should have good local resolution."""
    output_dir = _resolve_output_dir(tmp_path, "compute_state_spa")

    test_dataset, pipeline_output, state_output, perf_stages = _run_pipeline_and_compute_state(
        output_dir,
        grid_size=GRID_SIZE,
        n_images=N_IMAGES_SPA,
        noise_level=NOISE_LEVEL,
        zdim=ZDIM,
        n_centers=N_CENTERS,
        seed=SEED,
    )

    print(f"\n=== SPA compute_state locres scores ===")
    metrics = _compute_locres_metrics(state_output, N_CENTERS)

    mean_median = metrics["locres_median"]
    mean_90pct = metrics["locres_90pct"]
    print(f"  Mean locres_median = {mean_median:.2f}A  (threshold = {MAX_LOCRES_MEDIAN_SPA}A)")
    print(f"  Mean locres_90pct  = {mean_90pct:.2f}A")

    assert mean_median <= MAX_LOCRES_MEDIAN_SPA, (
        f"SPA compute_state locres_median {mean_median:.2f}A > {MAX_LOCRES_MEDIAN_SPA}A"
    )

    # Save scores
    with open(output_dir / "compute_state_scores.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # Perf regression check (warn only)
    perf_record = build_perf_record(perf_stages)
    perf_baseline_path = str(
        _REPO_ROOT / "tests" / "baselines" / "compute_state" / "perf_baseline_spa.json"
    )
    check_perf_regression(perf_record, perf_baseline_path, "SPA compute_state")


def test_compute_state_et(tmp_path):
    """Cryo-ET compute_state: reconstructed volumes should have good local resolution.

    This exercises the AKD fix for particle-grouped DataIterator iteration
    in even_less_naive_heterogeneity_scheme_relion_style.
    """
    output_dir = _resolve_output_dir(tmp_path, "compute_state_et")

    test_dataset, pipeline_output, state_output, perf_stages = _run_pipeline_and_compute_state(
        output_dir,
        grid_size=GRID_SIZE,
        n_images=N_IMAGES_ET,
        noise_level=NOISE_LEVEL,
        zdim=ZDIM,
        n_centers=N_CENTERS,
        n_tilts=N_TILTS,
        seed=SEED,
    )

    print(f"\n=== ET compute_state locres scores ===")
    metrics = _compute_locres_metrics(state_output, N_CENTERS)

    mean_median = metrics["locres_median"]
    mean_90pct = metrics["locres_90pct"]
    print(f"  Mean locres_median = {mean_median:.2f}A  (threshold = {MAX_LOCRES_MEDIAN_ET}A)")
    print(f"  Mean locres_90pct  = {mean_90pct:.2f}A")

    assert mean_median <= MAX_LOCRES_MEDIAN_ET, (
        f"ET compute_state locres_median {mean_median:.2f}A > {MAX_LOCRES_MEDIAN_ET}A"
    )

    # Save scores
    with open(output_dir / "compute_state_scores.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # Perf regression check (warn only)
    perf_record = build_perf_record(perf_stages)
    perf_baseline_path = str(
        _REPO_ROOT / "tests" / "baselines" / "compute_state" / "perf_baseline_cryo_et.json"
    )
    check_perf_regression(perf_record, perf_baseline_path, "ET compute_state")
