#!/usr/bin/env python
"""Benchmark: sketched normal operator vs covariance estimation.

Generates a PDB dataset (or reuses existing), runs:
  1. Sanity: X = GT with true coords → G(X) ≈ 0 (both sketch norms ~ 0)
  2. Covariance PCA baseline → relvar, plots
  3. Right sketch from random X → QR'd columns → relvar, plots, comparison

Usage:
    pixi run python examples/bench_sketched_normal.py [--output-dir DIR]
"""

import argparse, json, logging, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────────────
GRID_SIZE   = 64
N_IMAGES    = 5000
NOISE_LEVEL = 0.1
N_VOLUMES   = 10
N_PCS       = 10
SKETCH_RANK = 15
BATCH_SIZE  = 500
DISC_TYPE   = "linear_interp"
# ───────────────────────────────────────────────────────────────────────

import jax, jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.output import metrics
from recovar.simulation import simulator
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.simulation.synthetic_dataset import load_heterogeneous_reconstruction
from recovar.ppca.ppca_scale_sweep import (
    _load_simulated_dataset, _with_trailing_separator,
)
from recovar.ppca.sketched_normal import compute_normal_residual_sketches
from recovar.heterogeneity import covariance_estimation, principal_components
from recovar.reconstruction import homogeneous, noise as noise_mod


def _generate_or_load(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    voxel_size = 4.25 * 128 / GRID_SIZE
    vol_prefix = os.path.join(base_dir, "generated_volumes", "vol")
    ds_dir = os.path.join(base_dir, "test_dataset")

    if not os.path.isfile(f"{vol_prefix}0000.mrc"):
        logger.info("Generating %d trajectory volumes (grid=%d) ...", N_VOLUMES, GRID_SIZE)
        generate_trajectory_volumes(
            base_dir, grid_size=GRID_SIZE, n_volumes=N_VOLUMES,
            voxel_size=voxel_size, Bfactor=80, max_rotation_degrees=10.0,
        )

    if not os.path.isfile(os.path.join(ds_dir, f"particles.{GRID_SIZE}.mrcs")):
        logger.info("Simulating dataset: n=%d, noise=%.2g ...", N_IMAGES, NOISE_LEVEL)
        np.random.seed(42)
        simulator.generate_synthetic_dataset(
            ds_dir, voxel_size, vol_prefix, N_IMAGES,
            grid_size=GRID_SIZE,
            noise_level=NOISE_LEVEL, noise_model="radial1",
            contrast_std=0.0, noise_scale_std=0.0,
            dataset_params_option="uniform", disc_type=DISC_TYPE,
            trailing_zero_format_in_vol_name=True,
            put_extra_particles=False, percent_outliers=0.0,
        )

    cryos, sim_info, gt, noise_var = _load_simulated_dataset(
        _with_trailing_separator(ds_dir), GRID_SIZE, N_IMAGES, lazy=False,
    )
    return cryos, sim_info, gt, noise_var


# ── Plotting helpers ──

def _plot_slices(vols_real, labels, title, path):
    n = len(vols_real)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1: axes = [axes]
    for i, (v, l) in enumerate(zip(vols_real, labels)):
        mid = v.shape[2] // 2
        im = axes[i].imshow(v[:, :, mid], cmap="RdBu_r")
        axes[i].set_title(l, fontsize=9); axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved %s", path)


def _plot_relvar_comparison(relvar_dict, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, rv in relvar_dict.items():
        ax.plot(range(1, len(rv) + 1), rv, "o-", label=label, markersize=4)
    ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative relative variance")
    ax.set_title("Subspace quality vs GT"); ax.legend(); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(
        os.environ.get("TMPDIR", "/scratch/gpfs/GILLES/mg6942/tmp"),
        "bench_sketched_normal",
    ))
    args = parser.parse_args()
    out_dir = args.output_dir
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("JAX devices: %s", jax.devices())

    cryo, sim_info, gt, noise_var = _generate_or_load(out_dir)
    vs = cryo.volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vs))
    n_images = cryo.n_images
    vol_size = int(np.prod(vs))

    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    U_gt = U_gt_all[:, :N_PCS]
    s_gt = s_gt_all[:N_PCS]
    gt_mean = gt.get_mean()
    logger.info("GT: %d PCs, top eigenvalues: %s", N_PCS, s_gt[:5])

    U_gt_half = np.asarray(ftu.full_volume_to_half_volume(U_gt.T, vs).T)
    mean_half = np.asarray(ftu.full_volume_to_half_volume(gt_mean.reshape(vs), vs).reshape(-1))

    # ==================================================================
    # 1. SANITY: X = GT with true coordinates → G(X) ≈ 0
    # ==================================================================
    logger.info("=== Sanity check: X = GT with true coords ===")
    assign = np.array(sim_info["image_assignment"])
    centered_vols = gt.volumes - gt_mean[None, :]
    # V_true[i,:] = (vol[assign[i]] - mean)^H @ U_gt / s_gt
    V_true = np.array([
        (np.conj(centered_vols[assign[i]]) @ U_gt) / s_gt
        for i in range(n_images)
    ]).real.astype(np.float32)

    rng = np.random.default_rng(0)
    Q_sanity = rng.normal(size=(n_images, 3)).astype(np.float32)

    result_sanity = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE, right_sketch=Q_sanity,
        disc_type=DISC_TYPE,
    )
    right_norm = float(np.linalg.norm(np.asarray(result_sanity["right"])))
    # Normalize by expected image scale for interpretability
    logger.info("  ||G(X_gt) @ Q|| = %.6f  (should be small)", right_norm)
    logger.info("  per-image RMS   = %.6f", right_norm / np.sqrt(n_images))

    # ==================================================================
    # 2. COVARIANCE PCA BASELINE
    # ==================================================================
    logger.info("=== Covariance PCA baseline ===")
    t0 = time.time()

    # Estimate mean
    image_cov_noise = np.asarray(noise_mod.make_radial_noise(noise_var, cryo.image_shape))
    means, _, _ = homogeneous.get_mean_conformation_relion(
        cryo, BATCH_SIZE, disc_type=DISC_TYPE,
        noise_variance=image_cov_noise, return_all=True,
    )
    mean_est = means[0]  # halfset 0 mean (or combined)

    # Covariance estimation
    options = covariance_estimation.get_default_covariance_computation_options(GRID_SIZE)
    options["n_pcs_to_compute"] = N_PCS
    options["disc_type"] = DISC_TYPE

    cov_result = principal_components.get_cov_svds(
        cryo, mean_est, image_cov_noise,
        BATCH_SIZE, N_PCS, options=options,
    )
    U_cov = cov_result["u"]  # (vol_size, n_pcs)
    s_cov = cov_result["s"]  # (n_pcs,)
    dt_cov = time.time() - t0
    logger.info("  Covariance PCA done in %.1fs", dt_cov)

    rv_cov = metrics.captured_variance(U_cov, U_gt_all, s_gt_all)
    relvar_cov = np.asarray(metrics.relative_variance_from_captured_variance(rv_cov, s_gt_all))
    logger.info("  Covariance relvar: %s", relvar_cov)

    # ==================================================================
    # 3. RIGHT SKETCH FROM RANDOM X → QR → relvar
    # ==================================================================
    logger.info("=== Right sketch (random X) ===")
    V_random = rng.normal(size=(n_images, N_PCS)).astype(np.float32)
    Q_right = rng.normal(size=(n_images, SKETCH_RANK)).astype(np.float32)

    t0 = time.time()
    result_sketch = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_random, mean_half,
        batch_size=BATCH_SIZE, right_sketch=Q_right, disc_type=DISC_TYPE,
    )
    right = np.asarray(result_sketch["right"])  # (half_vol, sketch_rank)
    jax.block_until_ready(result_sketch["right"])
    dt_sketch = time.time() - t0
    logger.info("  Right sketch done in %.2fs, shape=%s", dt_sketch, right.shape)

    # QR in volume space
    Q_vol, R_vol = np.linalg.qr(right)
    Q_vol_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(Q_vol.T), vs).T)

    rv_sketch = metrics.captured_variance(Q_vol_full, U_gt_all, s_gt_all)
    relvar_sketch = np.asarray(
        metrics.relative_variance_from_captured_variance(rv_sketch, s_gt_all)
    )
    logger.info("  Sketch relvar: %s", relvar_sketch[:N_PCS])

    # ==================================================================
    # 4. PLOTS
    # ==================================================================

    # Relvar comparison
    _plot_relvar_comparison({
        f"Covariance PCA ({dt_cov:.0f}s)": relvar_cov,
        f"Sketch QR ({dt_sketch:.1f}s)": relvar_sketch[:N_PCS],
    }, os.path.join(plots_dir, "relvar_comparison.png"))

    # Central slices: GT vs covariance vs sketch
    n_show = min(4, N_PCS)
    gt_real = [np.asarray(ftu.get_idft3((U_gt_all[:, k] * s_gt_all[k]).reshape(vs)).real)
               for k in range(n_show)]
    cov_real = [np.asarray(ftu.get_idft3((U_cov[:, k] * s_cov[k]).reshape(vs)).real)
                for k in range(n_show)]
    sketch_real = [np.asarray(ftu.get_idft3(Q_vol_full[:, k].reshape(vs)).real) * np.sqrt(vol_size)
                   for k in range(n_show)]

    _plot_slices(gt_real, [f"GT PC{k}" for k in range(n_show)],
                 "Ground truth PCs", os.path.join(plots_dir, "gt_pcs.png"))
    _plot_slices(cov_real, [f"Cov PC{k}" for k in range(n_show)],
                 "Covariance PCA", os.path.join(plots_dir, "cov_pcs.png"))
    _plot_slices(sketch_real, [f"Sketch{k}" for k in range(n_show)],
                 "Sketch QR columns", os.path.join(plots_dir, "sketch_pcs.png"))

    # ==================================================================
    # 5. SUMMARY
    # ==================================================================
    summary = {
        "config": {
            "grid_size": GRID_SIZE, "n_images": N_IMAGES,
            "noise_level": NOISE_LEVEL, "n_pcs": N_PCS,
            "sketch_rank": SKETCH_RANK,
        },
        "sanity_right_norm": right_norm,
        "covariance": {
            "relvar": [float(x) for x in relvar_cov],
            "time_s": dt_cov,
        },
        "sketch": {
            "relvar": [float(x) for x in relvar_sketch[:N_PCS]],
            "time_s": dt_sketch,
        },
    }
    with open(os.path.join(out_dir, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s/bench_summary.json", out_dir)

    # Print comparison table
    logger.info("\n  %-20s %10s %10s", "Method", "relvar@%d" % N_PCS, "Time")
    logger.info("  %-20s %10.4f %10.1fs", "Covariance PCA",
                relvar_cov[-1], dt_cov)
    logger.info("  %-20s %10.4f %10.1fs", "Sketch QR",
                relvar_sketch[min(N_PCS, len(relvar_sketch)) - 1], dt_sketch)
    logger.info("Done.")


if __name__ == "__main__":
    main()
