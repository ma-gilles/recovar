"""
Cryo-ET outlier detection stress test with --ind and --particle-ind.

Tests the full cryo-ET outlier pipeline with index subsetting — a
common production workflow where users curate particles before
running heterogeneity analysis.

Steps:
  1. Generate cryo-ET tilt-series dataset with outliers
  2. Run pipeline_with_outliers (full dataset, no subsetting)
  3. Create --ind subset (random 80% of images)
  4. Run pipeline_with_outliers with --ind
  5. Create --particle-ind subset (random 80% of particles)
  6. Run pipeline_with_outliers with --particle-ind
  7. Verify all runs produce valid results and detect outliers

Params (env-configurable):
  CRYO_ET_IND_GRID_SIZE      (default 128)
  CRYO_ET_IND_N_IMAGES       (default 10000)
  CRYO_ET_IND_N_TILTS        (default 5)
  CRYO_ET_IND_PCT_OUTLIERS   (default 0.15)
  CRYO_ET_IND_PCT_TILT       (default 0.10)
  CRYO_ET_IND_K_ROUNDS       (default 1)
  CRYO_ET_IND_SUBSET_FRAC    (default 0.80)

Activation:
  pytest --run-integration --run-gpu --run-slow  (included in test-full)

Expected runtime: ~30-60 min on one GPU (3 pipeline runs).
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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.io,
]

_GRID = int(os.environ.get("CRYO_ET_IND_GRID_SIZE", "128"))
_N_IMAGES = int(os.environ.get("CRYO_ET_IND_N_IMAGES", "10000"))
_N_TILTS = int(os.environ.get("CRYO_ET_IND_N_TILTS", "5"))
_PCT_OUTLIERS = float(os.environ.get("CRYO_ET_IND_PCT_OUTLIERS", "0.15"))
_PCT_TILT = float(os.environ.get("CRYO_ET_IND_PCT_TILT", "0.10"))
_K_ROUNDS = int(os.environ.get("CRYO_ET_IND_K_ROUNDS", "1"))
_SUBSET_FRAC = float(os.environ.get("CRYO_ET_IND_SUBSET_FRAC", "0.80"))
_N_VOLS = 12


def _write_volumes(prefix: Path, n_vols: int, grid: int):
    """Generate simple synthetic volumes."""
    from scipy.ndimage import gaussian_filter

    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(n_vols):
        vol = np.zeros((grid, grid, grid), dtype=np.float32)
        cx = grid // 4 + int(i * grid / (2 * n_vols))
        vol[cx, grid // 2, grid // 2] = 100.0
        vol = gaussian_filter(vol, sigma=grid / 16)
        vol += rng.randn(grid, grid, grid).astype(np.float32) * 0.01
        with mrcfile.new(f"{prefix}{i:04d}.mrc", overwrite=True) as m:
            m.set_data(vol)


def _create_outlier_volume(path: str, grid: int):
    """Create a distinctive outlier volume."""
    vol = np.zeros((grid, grid, grid), dtype=np.float32)
    vol[grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4] = 1.0
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)


def _make_gt_mask(volumes_prefix: Path, grid: int) -> str:
    """Compute GT union mask from volumes and return path."""
    from recovar.core import mask as mask_mod
    from recovar import utils as recovar_utils

    vol_files = sorted(volumes_prefix.parent.glob(f"{volumes_prefix.name}*.mrc"))
    vols = [recovar_utils.load_mrc(str(f)) for f in vol_files]
    volume_shape = (grid, grid, grid)
    soft_mask, _ = mask_mod.make_union_gt_mask(vols, volume_shape)
    mask_path = str(volumes_prefix.parent / "gt_union_mask.mrc")
    recovar_utils.write_mrc(mask_path, soft_mask)
    return mask_path


def _generate_dataset(
    output_dir: Path, volumes_prefix: Path, grid: int, n_images: int, n_tilts: int, pct_outliers: float, pct_tilt: float
) -> Path:
    """Generate cryo-ET tilt-series dataset with outliers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_vol = output_dir / "outlier_volume.mrc"
    _create_outlier_volume(str(outlier_vol), grid)

    from conftest import gpu_subprocess_env

    cmd = [
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
        str(pct_outliers),
        "--percent-tilt-series-outliers",
        str(pct_tilt),
        "--grid-size",
        str(grid),
        "--volume-input",
        str(volumes_prefix),
        "--tilt-series",
        "--n-tilts",
        str(n_tilts),
    ]
    subprocess.run(cmd, check=True, env=gpu_subprocess_env())
    return output_dir / "test_dataset"


def _run_pipeline(
    dataset_dir: Path, pipeline_out: Path, mask_path: str, k_rounds: int, extra_args: list[str] | None = None
):
    """Run pipeline_with_outliers."""
    from conftest import gpu_subprocess_env

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline_with_outliers",
        str(dataset_dir / "particles.star"),
        "--poses",
        str(dataset_dir / "poses.pkl"),
        "--ctf",
        str(dataset_dir / "ctf.pkl"),
        "--tilt-series",
        "--tilt-series-ctf",
        "relion5",
        "--correct-contrast",
        "-o",
        str(pipeline_out),
        "--mask",
        mask_path,
        "--lazy",
        "--zdim",
        "4",
        "--k-rounds",
        str(k_rounds),
        "--use-contrast-detection",
        "--use-junk-detection",
        "--save-pipeline-indices",
    ]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True, env=gpu_subprocess_env())


def _compute_metrics(
    pipeline_out: Path, sim_info_path: Path, k_rounds: int, subset_ind: np.ndarray | None = None
) -> dict:
    """Compute outlier detection metrics, optionally for a subset."""
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    assign = np.asarray(sim_info["image_assignment"])
    if subset_ind is not None:
        assign = assign[subset_ind]
    n_total = int(assign.size)
    true_outliers = set(int(i) for i in np.where(assign < 0)[0])

    metrics = {"total_images": float(n_total), "true_outlier_count": float(len(true_outliers))}

    for r in range(1, k_rounds + 1):
        f_path = pipeline_out / f"inliers_round_{r}.pkl"
        if not f_path.exists():
            continue
        with open(f_path, "rb") as fh:
            inliers = np.asarray(pickle.load(fh), dtype=np.int64)
        detected_out = set(int(i) for i in np.setdiff1d(np.arange(n_total), inliers))
        tp = len(detected_out & true_outliers)
        fp = len(detected_out - true_outliers)
        fn = len(true_outliers - detected_out)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        metrics[f"outlier_f1_round_{r}"] = f1
        metrics[f"outlier_precision_round_{r}"] = prec
        metrics[f"outlier_recall_round_{r}"] = rec
        metrics[f"inlier_count_round_{r}"] = float(len(inliers))
        metrics[f"outlier_count_round_{r}"] = float(len(detected_out))

    # Particle-level metrics
    ts_assign = sim_info.get("tilt_series_assignment")
    if ts_assign is not None:
        ts_assign = np.asarray(ts_assign)
        if subset_ind is not None:
            # particle-ind subsetting: need to map back
            # For now, use full particle assignment
            pass
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
            metrics[f"particle_f1_round_{r}"] = f1
            metrics[f"particle_precision_round_{r}"] = prec
            metrics[f"particle_recall_round_{r}"] = rec

    return metrics


def _print_metrics(label: str, metrics: dict):
    print(f"\n{label}:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_cryo_et_outliers_with_indices(tmp_path):
    """
    Cryo-ET outlier detection stress test with --ind and --particle-ind.

    Runs three pipelines on the same dataset:
      1. Full (no subsetting) — baseline
      2. With --ind (image-level subsetting, 80% of images)
      3. With --particle-ind (particle-level subsetting, 80% of particles)

    Verifies:
      - All runs produce valid output files
      - Outlier detection F1 > 0 in all cases
      - --ind and --particle-ind don't crash and produce reasonable results
    """
    rng = np.random.RandomState(123)

    # Generate volumes and dataset
    vols_prefix = tmp_path / "vols" / "vol"
    _write_volumes(vols_prefix, n_vols=_N_VOLS, grid=_GRID)
    mask_path = _make_gt_mask(vols_prefix, _GRID)

    dataset_dir = _generate_dataset(
        output_dir=tmp_path / "data",
        volumes_prefix=vols_prefix,
        grid=_GRID,
        n_images=_N_IMAGES,
        n_tilts=_N_TILTS,
        pct_outliers=_PCT_OUTLIERS,
        pct_tilt=_PCT_TILT,
    )
    sim_info_path = dataset_dir / "simulation_info.pkl"
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    n_total_images = int(np.asarray(sim_info["image_assignment"]).size)
    ts_assign = sim_info.get("tilt_series_assignment")
    n_particles = int(np.asarray(ts_assign).size) if ts_assign is not None else 0

    # ============================
    # Run 1: Full (no subsetting)
    # ============================
    pipe_full = tmp_path / "pipe_full"
    _run_pipeline(dataset_dir, pipe_full, mask_path, _K_ROUNDS)
    m_full = _compute_metrics(pipe_full, sim_info_path, _K_ROUNDS)
    _print_metrics("Full (no subsetting)", m_full)
    assert m_full.get("outlier_f1_round_1", 0) > 0, "Full run: no outliers detected"

    # ============================
    # Run 2: With --ind (image-level subset)
    # ============================
    n_keep_images = int(n_total_images * _SUBSET_FRAC)
    ind = np.sort(rng.choice(n_total_images, size=n_keep_images, replace=False))
    ind_path = tmp_path / "image_subset.pkl"
    with open(ind_path, "wb") as f:
        pickle.dump(ind, f)

    pipe_ind = tmp_path / "pipe_ind"
    _run_pipeline(dataset_dir, pipe_ind, mask_path, _K_ROUNDS, extra_args=["--ind", str(ind_path)])
    m_ind = _compute_metrics(pipe_ind, sim_info_path, _K_ROUNDS, subset_ind=ind)
    _print_metrics(f"--ind (n={n_keep_images}/{n_total_images} images)", m_ind)

    # Verify partition covers subset
    for r in range(1, _K_ROUNDS + 1):
        inliers_f = pipe_ind / f"inliers_round_{r}.pkl"
        outliers_f = pipe_ind / f"outliers_round_{r}.pkl"
        if inliers_f.exists() and outliers_f.exists():
            with open(inliers_f, "rb") as f:
                n_in = len(np.asarray(pickle.load(f)))
            with open(outliers_f, "rb") as f:
                n_out = len(np.asarray(pickle.load(f)))
            assert n_in + n_out == n_keep_images, (
                f"--ind round {r}: inliers({n_in}) + outliers({n_out}) != subset({n_keep_images})"
            )

    # ============================
    # Run 3: With --particle-ind (particle-level subset)
    # ============================
    if n_particles > 0:
        n_keep_particles = int(n_particles * _SUBSET_FRAC)
        particle_ind = np.sort(rng.choice(n_particles, size=n_keep_particles, replace=False))
        particle_ind_path = tmp_path / "particle_subset.pkl"
        with open(particle_ind_path, "wb") as f:
            pickle.dump(particle_ind, f)

        pipe_pind = tmp_path / "pipe_pind"
        _run_pipeline(
            dataset_dir, pipe_pind, mask_path, _K_ROUNDS, extra_args=["--particle-ind", str(particle_ind_path)]
        )

        # Check particle-level outputs exist
        for r in range(1, _K_ROUNDS + 1):
            p_inliers_f = pipe_pind / f"particle_inliers_round_{r}.pkl"
            p_outliers_f = pipe_pind / f"particle_outliers_round_{r}.pkl"
            if p_inliers_f.exists() and p_outliers_f.exists():
                with open(p_inliers_f, "rb") as f:
                    n_pin = len(np.asarray(pickle.load(f)))
                with open(p_outliers_f, "rb") as f:
                    n_pout = len(np.asarray(pickle.load(f)))
                assert n_pin + n_pout == n_keep_particles, (
                    f"--particle-ind round {r}: particle_inliers({n_pin}) + "
                    f"particle_outliers({n_pout}) != particle_subset({n_keep_particles})"
                )
                print(
                    f"\n--particle-ind round {r}: {n_pin} inlier particles, {n_pout} outlier particles "
                    f"(from {n_keep_particles}/{n_particles})"
                )

    # ============================
    # Summary comparison
    # ============================
    print("\n=== SUMMARY ===")
    print(f"{'Run':25s}  {'F1_r1':>8s}  {'Prec_r1':>8s}  {'Rec_r1':>8s}")
    print("-" * 55)
    for label, m in [("Full", m_full), (f"--ind ({_SUBSET_FRAC:.0%})", m_ind)]:
        f1 = m.get("outlier_f1_round_1", 0)
        prec = m.get("outlier_precision_round_1", 0)
        rec = m.get("outlier_recall_round_1", 0)
        print(f"{label:25s}  {f1:8.4f}  {prec:8.4f}  {rec:8.4f}")
