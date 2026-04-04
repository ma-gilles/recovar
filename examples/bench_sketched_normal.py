#!/usr/bin/env python
"""Teaching script: sketched normal-operator products.

Generates a PDB-based synthetic dataset, then demonstrates:
  1. Sanity: if b = A(X) exactly, then G(X) = A*(A(X)-b) = 0
  2. Sketch primitives at full scale (both left and right)
  3. PPCA EM baseline (no contrast, no mask, linear_interp) with relvar + plots

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
GRID_SIZE   = 128
N_IMAGES    = 100000
NOISE_LEVEL = 0.1
N_VOLUMES   = 10
N_PCS       = 10
SKETCH_RANK = 200
BATCH_SIZE  = 500
PPCA_ITERS  = 10
DISC_TYPE   = "linear_interp"
# ───────────────────────────────────────────────────────────────────────

import jax, jax.numpy as jnp
import jax.random as jr
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.core import linalg
from recovar.output import metrics
from recovar.simulation import simulator
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.ppca.ppca_scale_sweep import (
    _load_simulated_dataset, _with_trailing_separator,
)
from recovar.ppca import ppca as ppca_mod
from recovar.ppca.sketched_normal import compute_normal_residual_sketches


# ── Helpers ──

def generate_or_load(base_dir):
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
            dataset_params_option="uniform",
            disc_type=DISC_TYPE,  # same disc_type everywhere
            trailing_zero_format_in_vol_name=True,
            put_extra_particles=False, percent_outliers=0.0,
        )

    cryo, sim_info, gt, noise_var = _load_simulated_dataset(
        _with_trailing_separator(ds_dir), GRID_SIZE, N_IMAGES, lazy=False,
    )
    return cryo, sim_info, gt, noise_var


def plot_slices(vols_real, labels, title, path):
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


def plot_relvar_comparison(relvar_dict, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, rv in relvar_dict.items():
        ax.plot(range(1, len(rv) + 1), rv, "o-", label=label, markersize=4)
    ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative relative variance")
    ax.set_title("Subspace quality vs GT"); ax.legend(); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved %s", path)


def to_real(vol_ft, vs):
    return np.asarray(ftu.get_idft3(np.asarray(vol_ft).reshape(vs)).real)


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

    cryo, sim_info, gt, noise_var = generate_or_load(out_dir)
    vs = cryo.volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vs))
    vol_size = int(np.prod(vs))
    n_images = cryo.n_images

    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    U_gt = U_gt_all[:, :N_PCS]
    s_gt = s_gt_all[:N_PCS]
    gt_mean = gt.get_mean()
    logger.info("GT: %d PCs, top eigenvalues: %s", N_PCS, s_gt[:5])

    U_gt_half = np.asarray(ftu.full_volume_to_half_volume(U_gt.T, vs).T)
    mean_half = np.asarray(ftu.full_volume_to_half_volume(gt_mean.reshape(vs), vs).reshape(-1))

    # ==================================================================
    # 1. SANITY: b = A(X) ⟹ G(X) = 0
    # ==================================================================
    logger.info("=" * 60)
    logger.info("SANITY CHECK: X = GT with true image coordinates")
    logger.info("=" * 60)

    assign = np.array(sim_info["image_assignment"])
    centered_vols = np.asarray(gt.volumes - gt_mean[None, :])
    state_coords = (np.conj(centered_vols) @ np.asarray(U_gt)) / np.asarray(s_gt)
    V_true = state_coords[assign].real.astype(np.float32)

    rng = np.random.default_rng(0)
    Q_test = rng.normal(size=(n_images, 3)).astype(np.float32)

    result_sanity = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE, right_sketch=Q_test, disc_type=DISC_TYPE,
    )
    sanity_norm = float(np.linalg.norm(np.asarray(result_sanity["right"])))
    logger.info("  ||G(X_gt) @ Q|| = %.4f  (residual from noise in images)", sanity_norm)

    # ==================================================================
    # 2. SKETCH AT SCALE
    # ==================================================================
    logger.info("=" * 60)
    logger.info("SKETCH AT SCALE: sketch_rank=%d", SKETCH_RANK)
    logger.info("=" * 60)

    S_left_half = (rng.normal(size=(SKETCH_RANK, half_vol_size))
                   + 1j * rng.normal(size=(SKETCH_RANK, half_vol_size))).astype(np.complex64)
    Q_right = rng.normal(size=(n_images, SKETCH_RANK)).astype(np.float32)

    t0 = time.time()
    result_scale = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half, right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result_scale["left"])
    jax.block_until_ready(result_scale["right"])
    dt_warmup = time.time() - t0

    t0 = time.time()
    result_scale = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half, right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result_scale["left"])
    jax.block_until_ready(result_scale["right"])
    dt_compiled = time.time() - t0

    left_norm = float(np.linalg.norm(np.asarray(result_scale["left"])))
    right_norm = float(np.linalg.norm(np.asarray(result_scale["right"])))
    logger.info("  Left  (%d, %d): ||.|| = %.4f", *result_scale["left"].shape, left_norm)
    logger.info("  Right (%d, %d): ||.|| = %.4f", *result_scale["right"].shape, right_norm)
    logger.info("  Warmup: %.1fs, Compiled: %.1fs", dt_warmup, dt_compiled)

    # ==================================================================
    # 3. PPCA EM BASELINE (no contrast, no mask, linear_interp)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PPCA EM BASELINE: %d iters, %d PCs", PPCA_ITERS, N_PCS)
    logger.info("=" * 60)

    t0 = time.time()

    # Random initial W
    W_init = jr.normal(jr.PRNGKey(0), (vol_size, N_PCS), dtype=jnp.float32)
    W_init = linalg.batch_dft3(W_init, vs, N_PCS)

    # Flat prior (no regularization)
    W_prior = np.ones(vol_size, dtype=np.float32) * 1e10

    U_ppca, S_ppca, W_ppca, _, _ = ppca_mod.EM(
        cryo,
        gt_mean,
        W_init,
        W_prior,
        EM_iter=PPCA_ITERS,
        U_gt=U_gt_all,
        S_gt=s_gt_all,
        disc_type_mean=DISC_TYPE,
        disc_type=DISC_TYPE,
        contrast_mode="none",
    )
    dt_ppca = time.time() - t0
    logger.info("  PPCA EM done in %.1fs", dt_ppca)

    rv_ppca = metrics.captured_variance(U_ppca, U_gt_all, s_gt_all)
    relvar_ppca = np.asarray(metrics.relative_variance_from_captured_variance(rv_ppca, s_gt_all))
    logger.info("  relvar: %s", [f"{x:.4f}" for x in relvar_ppca])

    # ==================================================================
    # 4. PLOTS
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PLOTS")
    logger.info("=" * 60)

    n_show = min(4, N_PCS)
    gt_real = [to_real(U_gt_all[:, k] * s_gt_all[k], vs) for k in range(n_show)]
    ppca_real = [to_real(U_ppca[:, k] * S_ppca[k], vs) for k in range(n_show)]

    plot_slices(gt_real, [f"GT PC{k}" for k in range(n_show)],
                "Ground truth PCs", os.path.join(plots_dir, "gt_pcs.png"))
    plot_slices(ppca_real, [f"PPCA PC{k}" for k in range(n_show)],
                "PPCA EM PCs", os.path.join(plots_dir, "ppca_pcs.png"))
    plot_relvar_comparison({
        f"PPCA EM ({dt_ppca:.0f}s)": relvar_ppca[:N_PCS],
    }, os.path.join(plots_dir, "relvar.png"))

    # ==================================================================
    # 5. SUMMARY
    # ==================================================================
    summary = {
        "config": {
            "grid_size": GRID_SIZE, "n_images": N_IMAGES,
            "noise_level": NOISE_LEVEL, "n_pcs": N_PCS,
            "sketch_rank": SKETCH_RANK, "ppca_iters": PPCA_ITERS,
        },
        "sanity_norm": sanity_norm,
        "sketch": {
            "left_norm": left_norm, "right_norm": right_norm,
            "warmup_s": dt_warmup, "compiled_s": dt_compiled,
        },
        "ppca": {
            "relvar": [float(x) for x in relvar_ppca],
            "time_s": dt_ppca,
        },
    }
    with open(os.path.join(out_dir, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n  %-25s %12s %12s", "Method", f"relvar@{N_PCS}", "Time")
    logger.info("  %-25s %12.4f %12.1fs", "PPCA EM",
                relvar_ppca[min(N_PCS, len(relvar_ppca)) - 1], dt_ppca)
    logger.info("  %-25s %12s %12.1fs", f"Sketch (rank={SKETCH_RANK})", "—", dt_compiled)
    logger.info("  Sanity ||G(X_gt)@Q||=%.4f", sanity_norm)
    logger.info("Done. Plots in %s", plots_dir)


if __name__ == "__main__":
    main()
