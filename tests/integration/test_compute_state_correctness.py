"""Integration tests for compute_state volume reconstruction quality.

Verifies that compute_state produces volumes that match GT volumes via FSC,
for both SPA and cryo-ET data.  This exercises the full call chain:
  compute_state → compute_and_save_reweighted → make_volumes_kernel_estimate_local
  → even_less_naive_heterogeneity_scheme_relion_style → DataIterator

The cryo-ET variant specifically tests that particle-grouped iteration works
correctly in even_less_naive_heterogeneity_scheme_relion_style (the AKD fix).
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

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.long_test]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_SIZE = 64
N_IMAGES_SPA = 5000
N_IMAGES_ET = 5000
N_TILTS = 5
NOISE_LEVEL = 0.1
ZDIM = 4
N_CENTERS = 5      # number of k-means cluster centers for compute_state
SEED = 42

# FSC thresholds — reconstructed volumes must reach at least this FSC vs GT
# at the half-Nyquist shell.  These are deliberately loose since compute_state
# uses kernel estimation (not direct backprojection).
MIN_FSC_SPA = 0.3
MIN_FSC_ET = 0.2  # ET is harder (fewer total images per tilt)


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

    Returns (test_dataset_dir, pipeline_output_dir, state_output_dir).
    """
    env = gpu_subprocess_env()

    dataset_dir = output_dir / "dataset"
    test_dataset = dataset_dir / "test_dataset"
    pipeline_output = test_dataset / "pipeline_output"
    state_output = output_dir / "state_output"

    # 1. Generate dataset
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

    # 2. Determine particles file
    if n_tilts is not None:
        particles = test_dataset / "particles.star"
    else:
        particles = test_dataset / f"particles.{grid_size}.mrcs"
    assert particles.exists(), f"Missing particles: {particles}"

    # 3. Run pipeline
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
    subprocess.run(pipeline_cmd, check=True, env=env, timeout=3600)
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
    state_cmd = [
        sys.executable, "-m", "recovar.command_line", "compute_state",
        str(pipeline_output),
        "-o", str(state_output),
        "--latent-points", str(centers_file),
        "--lazy",
        "--n-bins", "30",
    ]
    subprocess.run(state_cmd, check=True, env=env, timeout=3600)
    assert state_output.exists()

    return test_dataset, pipeline_output, state_output


def _compute_fsc_vs_gt(state_output: Path, test_dataset: Path, n_centers: int) -> list[float]:
    """For each reconstructed volume, find the best FSC match against GT.

    Returns a list of best-match FSC values (one per reconstructed volume).
    """
    from recovar.simulation import synthetic_dataset
    from recovar.core import fourier_transform_utils
    import mrcfile

    sim_path = test_dataset / "simulation_info.pkl"
    gt = synthetic_dataset.load_heterogeneous_reconstruction(str(sim_path))
    # gt.volumes is (n_vols, vol_size) in Fourier space — convert to real
    vol_shape = gt.volume_shape
    gt_volumes_real = []
    for i in range(gt.volumes.shape[0]):
        gt_real = np.real(fourier_transform_utils.get_idft3(
            gt.volumes[i].reshape(vol_shape)
        )).astype(np.float32)
        gt_volumes_real.append(gt_real)

    # Load reconstructed volumes
    mrc_files = sorted(state_output.glob("state*.mrc"))
    # Filter out half-maps and diagnostics
    mrc_files = [f for f in mrc_files if "half" not in f.name and "unfil" not in f.name]
    assert len(mrc_files) > 0, f"No state MRC files in {state_output}"

    best_fscs = []
    for mrc_path in mrc_files:
        with mrcfile.open(str(mrc_path), mode='r') as mrc:
            rec_vol = np.array(mrc.data, dtype=np.float32)

        # Find best FSC match against all GT volumes
        best_fsc = -1.0
        for gt_real in gt_volumes_real:
            fsc_curve = _fsc_between_volumes(rec_vol, gt_real)
            # Take FSC at ~half-Nyquist
            idx = len(fsc_curve) // 4
            fsc_val = float(np.mean(fsc_curve[1:idx+1]))  # skip DC
            best_fsc = max(best_fsc, fsc_val)
        best_fscs.append(best_fsc)

    return best_fscs


def _fsc_between_volumes(vol1: np.ndarray, vol2: np.ndarray) -> np.ndarray:
    """Compute FSC curve between two real-space volumes."""
    ft1 = np.fft.fftn(vol1)
    ft2 = np.fft.fftn(vol2)
    grid_size = vol1.shape[0]
    n_shells = grid_size // 2

    # Compute shell indices
    coords = np.fft.fftfreq(grid_size) * grid_size
    kx, ky, kz = np.meshgrid(coords, coords, coords, indexing='ij')
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    shell_idx = np.round(r).astype(int)

    fsc = np.zeros(n_shells)
    for s in range(n_shells):
        mask = shell_idx == s
        if mask.sum() == 0:
            continue
        num = np.real(np.sum(ft1[mask] * np.conj(ft2[mask])))
        denom = np.sqrt(np.sum(np.abs(ft1[mask])**2) * np.sum(np.abs(ft2[mask])**2))
        fsc[s] = num / max(denom, 1e-30)

    return fsc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_state_spa(tmp_path):
    """SPA compute_state: reconstructed volumes should match GT via FSC."""
    output_dir = _resolve_output_dir(tmp_path, "compute_state_spa")

    test_dataset, pipeline_output, state_output = _run_pipeline_and_compute_state(
        output_dir,
        grid_size=GRID_SIZE,
        n_images=N_IMAGES_SPA,
        noise_level=NOISE_LEVEL,
        zdim=ZDIM,
        n_centers=N_CENTERS,
        seed=SEED,
    )

    best_fscs = _compute_fsc_vs_gt(state_output, test_dataset, N_CENTERS)

    print(f"\n=== SPA compute_state FSC scores ===")
    for i, fsc in enumerate(best_fscs):
        print(f"  Volume {i}: best GT match FSC = {fsc:.4f}")

    mean_fsc = np.mean(best_fscs)
    print(f"  Mean FSC = {mean_fsc:.4f}  (threshold = {MIN_FSC_SPA})")

    assert mean_fsc >= MIN_FSC_SPA, (
        f"SPA compute_state mean FSC {mean_fsc:.4f} < {MIN_FSC_SPA}. "
        f"Per-volume: {best_fscs}"
    )

    # Save scores
    scores = {"mean_fsc": mean_fsc, "per_volume_fsc": best_fscs}
    with open(output_dir / "compute_state_scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=float)


def test_compute_state_et(tmp_path):
    """Cryo-ET compute_state: reconstructed volumes should match GT via FSC.

    This exercises the AKD fix for particle-grouped DataIterator iteration
    in even_less_naive_heterogeneity_scheme_relion_style.
    """
    output_dir = _resolve_output_dir(tmp_path, "compute_state_et")

    test_dataset, pipeline_output, state_output = _run_pipeline_and_compute_state(
        output_dir,
        grid_size=GRID_SIZE,
        n_images=N_IMAGES_ET,
        noise_level=NOISE_LEVEL,
        zdim=ZDIM,
        n_centers=N_CENTERS,
        n_tilts=N_TILTS,
        seed=SEED,
    )

    best_fscs = _compute_fsc_vs_gt(state_output, test_dataset, N_CENTERS)

    print(f"\n=== ET compute_state FSC scores ===")
    for i, fsc in enumerate(best_fscs):
        print(f"  Volume {i}: best GT match FSC = {fsc:.4f}")

    mean_fsc = np.mean(best_fscs)
    print(f"  Mean FSC = {mean_fsc:.4f}  (threshold = {MIN_FSC_ET})")

    assert mean_fsc >= MIN_FSC_ET, (
        f"ET compute_state mean FSC {mean_fsc:.4f} < {MIN_FSC_ET}. "
        f"Per-volume: {best_fscs}"
    )

    # Save scores
    scores = {"mean_fsc": mean_fsc, "per_volume_fsc": best_fscs}
    with open(output_dir / "compute_state_scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=float)
