#!/usr/bin/env python
"""Benchmark: sketched normal operator starting from GT subspace.

Generates a PDB dataset (or reuses existing), loads GT eigenvectors,
sets X = GT (low-rank), computes S_L @ G(X), QR-factors the sketch,
and measures how well the sketch columns span the GT subspace.

Usage:
    pixi run python examples/bench_sketched_normal.py [--output-dir DIR]
"""

import argparse, logging, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench_sketch")

# ── Defaults ───────────────────────────────────────────────────────────
GRID_SIZE   = 64
N_IMAGES    = 5000
NOISE_LEVEL = 0.1
N_VOLUMES   = 10
N_PCS       = 10            # rank of GT subspace to use
SKETCH_RANK = 15            # left sketch dimension (oversample the rank)
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
    _load_simulated_dataset, _with_trailing_separator, _build_halfset_indices,
)
from recovar.ppca.sketched_normal import compute_normal_residual_sketches


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
    return cryos, sim_info, gt, noise_var, ds_dir


def _plot_central_slices(volumes_real, labels, title, save_path):
    """Plot central z-slices of real-space volumes."""
    n = len(volumes_real)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (vol, label) in enumerate(zip(volumes_real, labels)):
        mid = vol.shape[2] // 2
        im = axes[i].imshow(vol[:, :, mid], cmap="RdBu_r")
        axes[i].set_title(label, fontsize=9)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def _plot_relvar(relvar_per_pc, labels, save_path):
    """Bar chart of cumulative relative variance per PC."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, len(relvar_per_pc) + 1)
    ax.bar(x, relvar_per_pc, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative relative variance")
    ax.set_title(f"Sketch subspace quality ({labels})")
    ax.set_ylim(0, min(1.05, max(relvar_per_pc) * 1.15))
    for i, v in enumerate(relvar_per_pc):
        ax.text(i + 1, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def _plot_cosine_similarity(cosines, labels_x, title, save_path):
    """Plot cosine similarity between sketch columns and GT PCs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(cosines)
    ax.bar(range(n), cosines, color="coral", edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylabel("|cosine|")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(cosines):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


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

    # ── Load dataset and GT ──
    cryos, sim_info, gt, noise_var, ds_dir = _generate_or_load(out_dir)
    cryo = cryos
    vs = cryo.volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vs))
    n_images = cryo.n_images

    # GT eigenvectors and eigenvalues
    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    U_gt = U_gt_all[:, :N_PCS]       # (vol_size, n_pcs) full Fourier
    s_gt = s_gt_all[:N_PCS]
    gt_mean = gt.get_mean()
    logger.info("GT: %d PCs, top eigenvalues: %s", N_PCS, s_gt[:5])

    # Convert to half-volume
    U_gt_half = np.asarray(ftu.full_volume_to_half_volume(U_gt.T, vs).T)  # (half_vol, n_pcs)
    mean_half = np.asarray(ftu.full_volume_to_half_volume(gt_mean.reshape(vs), vs).reshape(-1))

    # ── Set X = GT: U_X = U_gt, sigma = s_gt, V_X = GT coordinates ──
    # V_X: the "true" coordinates of each image in the GT basis
    # For a quick test, use random V_X (the gradient still tests the operator)
    rng = np.random.default_rng(0)
    V_X = rng.normal(size=(n_images, N_PCS)).astype(np.float32)

    # ── Random Gaussian left sketch ──
    S_left_half = (rng.normal(size=(SKETCH_RANK, half_vol_size))
                   + 1j * rng.normal(size=(SKETCH_RANK, half_vol_size))).astype(np.complex64)

    # ── Compute S_L @ G(X) ──
    logger.info("Computing left sketch: sketch_rank=%d, n_pcs=%d", SKETCH_RANK, N_PCS)
    t0 = time.time()
    result = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_X, mean_half,
        batch_size=BATCH_SIZE, left_sketch_half=S_left_half,
        disc_type=DISC_TYPE,
    )
    left = result["left"]  # (sketch_rank, n_images)
    jax.block_until_ready(left)
    dt = time.time() - t0
    logger.info("Left sketch computed in %.2fs, shape=%s, norm=%.4f",
                dt, left.shape, np.linalg.norm(np.asarray(left)))

    # ── QR factor the sketch to get an orthonormal basis for range(S_L @ G) ──
    # left has shape (sketch_rank, n_images) — each row is a "measurement"
    # The column space of left.T is what we want
    left_np = np.asarray(left).T  # (n_images, sketch_rank)
    Q, R = np.linalg.qr(left_np)  # Q: (n_images, sketch_rank)
    logger.info("QR done: Q shape=%s, R diag=%s", Q.shape, np.diag(R)[:5])

    # ── Measure quality: how well does range(Q) capture GT V_X? ──
    # captured_variance of Q columns against the true V_X coordinates
    # This measures if the sketch captures the image-space signal
    #
    # But more interesting: measure in volume space.  The sketch columns
    # S_L^T @ Q give volume-space vectors.  Check their relvar against GT U_gt.

    # Volume-space sketch basis: S_L^T @ Q = (sketch_rank, half_vol).T @ (n_images, sketch_rank) → doesn't work
    # Actually: S_left_half.T @ left.T doesn't give volume-space vectors.
    # The left sketch output is in image-coordinate space.
    # For volume-space analysis, use the right sketch instead.

    # ── Also compute right sketch for volume-space analysis ──
    Q_right = rng.normal(size=(n_images, SKETCH_RANK)).astype(np.float32)
    t0 = time.time()
    result_r = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_X, mean_half,
        batch_size=BATCH_SIZE, right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    right = result_r["right"]  # (half_vol, sketch_rank)
    jax.block_until_ready(right)
    dt_r = time.time() - t0
    logger.info("Right sketch computed in %.2fs, shape=%s", dt_r, right.shape)

    # QR the right sketch columns to get volume-space orthonormal basis
    right_np = np.asarray(right)  # (half_vol, sketch_rank)
    Q_vol, R_vol = np.linalg.qr(right_np)  # Q_vol: (half_vol, sketch_rank)
    logger.info("Volume QR done: Q_vol shape=%s", Q_vol.shape)

    # Convert Q_vol to full volume for metrics
    Q_vol_full = np.asarray(ftu.half_volume_to_full_volume(
        jnp.array(Q_vol.T), vs
    ).T)  # (vol_size, sketch_rank)

    # ── Relative variance: how much of GT variance does the sketch span? ──
    rv = metrics.captured_variance(Q_vol_full, U_gt_all, s_gt_all)
    relvar = metrics.relative_variance_from_captured_variance(rv, s_gt_all)
    logger.info("Relative variance (cumulative): %s", relvar[:SKETCH_RANK])
    logger.info("Final relvar at %d sketch cols: %.4f", SKETCH_RANK, relvar[min(SKETCH_RANK, len(relvar)) - 1])

    # ── Cosine similarity: each QR'd column vs best-matching GT PC ──
    n_show = min(N_PCS, SKETCH_RANK)
    cosines = []
    for k in range(n_show):
        best = max(
            float(np.abs(np.vdot(Q_vol_full[:, k], U_gt_all[:, j])))
            / (np.linalg.norm(Q_vol_full[:, k]) * np.linalg.norm(U_gt_all[:, j]) + 1e-30)
            for j in range(N_PCS)
        )
        cosines.append(best)
    logger.info("Best-match |cosine| per sketch col: %s", [f"{c:.3f}" for c in cosines])

    # ── Plots ──

    # 1. Relvar bar chart
    _plot_relvar(
        np.asarray(relvar[:SKETCH_RANK]),
        f"sketch_rank={SKETCH_RANK}, n_pcs={N_PCS}",
        os.path.join(plots_dir, "relvar_sketch.png"),
    )

    # 2. Cosine similarity
    _plot_cosine_similarity(
        cosines,
        [f"col {k}" for k in range(n_show)],
        f"Best |cosine| vs GT PCs (sketch_rank={SKETCH_RANK})",
        os.path.join(plots_dir, "cosine_sketch_vs_gt.png"),
    )

    # 3. Central slices: first few QR'd sketch columns vs GT PCs
    vol_size = int(np.prod(vs))
    vol_norm = np.sqrt(vol_size)
    n_show_vol = min(4, n_show)

    gt_real = [np.asarray(ftu.get_idft3((U_gt_all[:, k] * s_gt_all[k]).reshape(vs)).real)
               for k in range(n_show_vol)]
    sketch_real = [np.asarray(ftu.get_idft3(Q_vol_full[:, k].reshape(vs)).real) * vol_norm
                   for k in range(n_show_vol)]

    _plot_central_slices(
        gt_real, [f"GT PC{k}" for k in range(n_show_vol)],
        "Ground truth PCs (weighted by eigenvalue)",
        os.path.join(plots_dir, "gt_pcs.png"),
    )
    _plot_central_slices(
        sketch_real, [f"Sketch col{k}" for k in range(n_show_vol)],
        "QR'd right-sketch columns (volume space)",
        os.path.join(plots_dir, "sketch_columns.png"),
    )

    # ── Summary ──
    summary = {
        "grid_size": GRID_SIZE,
        "n_images": N_IMAGES,
        "noise_level": NOISE_LEVEL,
        "n_pcs": N_PCS,
        "sketch_rank": SKETCH_RANK,
        "relvar": [float(x) for x in relvar[:SKETCH_RANK]],
        "cosines": cosines,
        "time_left_sketch": dt,
        "time_right_sketch": dt_r,
    }
    import json
    with open(os.path.join(out_dir, "sketch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s/sketch_summary.json", out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
