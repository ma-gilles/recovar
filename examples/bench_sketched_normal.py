#!/usr/bin/env python
"""Teaching script: sketched normal-operator products.

Generates a PDB-based synthetic dataset, then demonstrates:
  1. Sanity: if b = A(X) exactly, then G(X) = A*(A(X)-b) = 0
  2. Covariance PCA baseline (using GT mean, noise, mask)
  3. Sketch primitives ready for student to build a PC algorithm

Usage:
    pixi run python examples/bench_sketched_normal.py [--output-dir DIR]

Configuration: edit the constants at the top.
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
NOISE_LEVEL = 0.1          # lower = higher SNR
N_VOLUMES   = 10            # trajectory states
N_PCS       = 10            # number of PCs to compute/compare
SKETCH_RANK = 200           # left/right sketch dimension
BATCH_SIZE  = 500
DISC_TYPE   = "linear_interp"
# ───────────────────────────────────────────────────────────────────────

import jax, jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.core import mask as mask_utils
from recovar.output import metrics
from recovar.simulation import simulator
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.simulation.synthetic_dataset import load_heterogeneous_reconstruction
from recovar.reconstruction import noise as noise_mod
from recovar.heterogeneity import covariance_estimation, principal_components
from recovar.ppca.ppca_scale_sweep import (
    _load_simulated_dataset, _with_trailing_separator,
)
from recovar.ppca.sketched_normal import compute_normal_residual_sketches


# ── Dataset generation / loading ──

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
    else:
        logger.info("Reusing volumes at %s", vol_prefix)

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
    else:
        logger.info("Reusing dataset at %s", ds_dir)

    cryo, sim_info, gt, noise_var = _load_simulated_dataset(
        _with_trailing_separator(ds_dir), GRID_SIZE, N_IMAGES, lazy=False,
    )
    return cryo, sim_info, gt, noise_var


# ── Plotting ──

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
    """Fourier volume → real-space central slice."""
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

    # ── Load dataset and GT ──
    cryo, sim_info, gt, noise_var = generate_or_load(out_dir)
    vs = cryo.volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vs))
    vol_size = int(np.prod(vs))
    n_images = cryo.n_images

    # GT eigenvectors, eigenvalues, mean
    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    U_gt = U_gt_all[:, :N_PCS]
    s_gt = s_gt_all[:N_PCS]
    gt_mean = gt.get_mean()           # full Fourier volume
    gt_noise = noise_var              # radial noise spectrum
    logger.info("GT: %d PCs, top eigenvalues: %s", N_PCS, s_gt[:5])

    # Half-volume versions for the sketch operator
    U_gt_half = np.asarray(ftu.full_volume_to_half_volume(U_gt.T, vs).T)
    mean_half = np.asarray(ftu.full_volume_to_half_volume(gt_mean.reshape(vs), vs).reshape(-1))

    # ==================================================================
    # 1. SANITY: if b = A(X) exactly, then G(X) = A*(A(X) - b) = 0
    # ==================================================================
    logger.info("=" * 60)
    logger.info("SANITY CHECK: X = GT with true image coordinates")
    logger.info("  If b_i = A_i(x_i) for all i, then G(X) = 0.")
    logger.info("=" * 60)

    # True per-image coordinates: project each image's GT volume onto U_gt
    assign = np.array(sim_info["image_assignment"])
    centered_vols = gt.volumes - gt_mean[None, :]
    V_true = np.array([
        (np.conj(centered_vols[assign[i]]) @ U_gt) / s_gt
        for i in range(n_images)
    ]).real.astype(np.float32)

    # Random test sketch
    rng = np.random.default_rng(0)
    Q_test = rng.normal(size=(n_images, 3)).astype(np.float32)

    result_sanity = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE, right_sketch=Q_test, disc_type=DISC_TYPE,
    )
    sanity_norm = float(np.linalg.norm(np.asarray(result_sanity["right"])))
    logger.info("  ||G(X_gt) @ Q|| = %.6f", sanity_norm)
    logger.info("  (This should be small — only disc_type mismatch and float32 noise)")

    # ==================================================================
    # 1b. SKETCH AT SCALE: both sketches with SKETCH_RANK dimensions
    # ==================================================================
    logger.info("=" * 60)
    logger.info("SKETCH AT SCALE: sketch_rank=%d, n_pcs=%d", SKETCH_RANK, N_PCS)
    logger.info("=" * 60)

    S_left_half = (rng.normal(size=(SKETCH_RANK, half_vol_size))
                   + 1j * rng.normal(size=(SKETCH_RANK, half_vol_size))).astype(np.complex64)
    Q_right = rng.normal(size=(n_images, SKETCH_RANK)).astype(np.float32)

    t0 = time.time()
    result_scale = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half,
        right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result_scale["left"])
    jax.block_until_ready(result_scale["right"])
    dt_sketch = time.time() - t0
    left_norm = float(np.linalg.norm(np.asarray(result_scale["left"])))
    right_norm = float(np.linalg.norm(np.asarray(result_scale["right"])))
    logger.info("  Left  sketch: shape=%s, ||S_L @ G(X)|| = %.6f", result_scale["left"].shape, left_norm)
    logger.info("  Right sketch: shape=%s, ||G(X) @ Q_R|| = %.6f", result_scale["right"].shape, right_norm)
    logger.info("  Time: %.1fs (incl JIT compile)", dt_sketch)

    # Second call (compiled)
    t0 = time.time()
    result_scale2 = compute_normal_residual_sketches(
        cryo, U_gt_half, s_gt.astype(np.float32), V_true, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half,
        right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result_scale2["left"])
    jax.block_until_ready(result_scale2["right"])
    dt_compiled = time.time() - t0
    logger.info("  Compiled run: %.1fs", dt_compiled)

    # ==================================================================
    # 2. COVARIANCE PCA BASELINE (using GT mean, GT noise, GT mask)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("COVARIANCE PCA BASELINE")
    logger.info("=" * 60)

    t0 = time.time()

    # GT noise as image-shaped array
    image_noise = np.asarray(noise_mod.make_radial_noise(gt_noise, cryo.image_shape))

    # Volume mask from GT volumes
    gt_real_vols = [to_real(gt.volumes[i], vs) for i in range(gt.volumes.shape[0])]
    volume_mask = mask_utils.make_union_gt_mask(gt_real_vols, vs)[1].astype(np.float32)
    dilated_volume_mask = volume_mask  # use same for simplicity

    # Use pipeline's estimate_principal_components
    from recovar.heterogeneity.covariance_core import get_picked_frequencies

    covariance_options = covariance_estimation.get_default_covariance_computation_options(GRID_SIZE)
    picked_frequencies = np.array(get_picked_frequencies(
        vs, radius=covariance_options["column_radius"], use_half=True,
    ))

    # Compute covariance columns
    valid_idx = np.where(volume_mask.reshape(-1) > 0.5)[0]
    means_dict = type('M', (), {
        'combined': gt_mean,
        'lhs': np.ones(vol_size),  # dummy
    })()

    class DummyOptions:
        ignore_zero_frequency = False
        disc_type = DISC_TYPE

    u_cov, s_cov, _, _, _ = principal_components.estimate_principal_components(
        cryo, DummyOptions(), means_dict,
        gt_mean,       # mean_prior
        volume_mask.reshape(vs),
        dilated_volume_mask.reshape(vs),
        valid_idx,
        BATCH_SIZE,
        gpu_memory_to_use=40,
        covariance_options=covariance_options,
        variance_estimate=None,
    )
    U_cov = u_cov["rescaled"][:, :N_PCS]
    s_cov_vals = s_cov["rescaled"][:N_PCS]
    dt_cov = time.time() - t0
    logger.info("  Covariance PCA done in %.1fs", dt_cov)

    # Relative variance against GT
    rv_cov = metrics.captured_variance(U_cov, U_gt_all, s_gt_all)
    relvar_cov = np.asarray(metrics.relative_variance_from_captured_variance(rv_cov, s_gt_all))
    logger.info("  relvar: %s", [f"{x:.4f}" for x in relvar_cov])

    # ==================================================================
    # 3. PLOTS
    # ==================================================================
    logger.info("=" * 60)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 60)

    n_show = min(4, N_PCS)

    # GT PCs
    gt_pc_real = [to_real(U_gt_all[:, k] * s_gt_all[k], vs) for k in range(n_show)]
    plot_slices(gt_pc_real, [f"GT PC{k}" for k in range(n_show)],
                "Ground truth PCs (weighted)", os.path.join(plots_dir, "gt_pcs.png"))

    # Covariance PCs
    cov_pc_real = [to_real(U_cov[:, k] * s_cov_vals[k], vs) for k in range(n_show)]
    plot_slices(cov_pc_real, [f"Cov PC{k}" for k in range(n_show)],
                "Covariance PCA", os.path.join(plots_dir, "cov_pcs.png"))

    # Relvar
    plot_relvar_comparison({
        f"Covariance PCA ({dt_cov:.0f}s)": relvar_cov[:N_PCS],
    }, os.path.join(plots_dir, "relvar.png"))

    # ==================================================================
    # 4. SUMMARY
    # ==================================================================
    summary = {
        "config": {
            "grid_size": GRID_SIZE, "n_images": N_IMAGES,
            "noise_level": NOISE_LEVEL, "n_pcs": N_PCS,
            "sketch_rank": SKETCH_RANK,
        },
        "sanity_norm": sanity_norm,
        "sketch_at_scale": {
            "left_norm": left_norm,
            "right_norm": right_norm,
            "time_warmup_s": dt_sketch,
            "time_compiled_s": dt_compiled,
        },
        "covariance": {
            "relvar": [float(x) for x in relvar_cov],
            "time_s": dt_cov,
        },
    }
    with open(os.path.join(out_dir, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n  %-25s %12s %12s", "Method", f"relvar@{N_PCS}", "Time")
    logger.info("  %-25s %12.4f %12.1fs", "Covariance PCA",
                relvar_cov[min(N_PCS, len(relvar_cov)) - 1], dt_cov)
    logger.info("  %-25s %12s %12.1fs", f"Sketch (rank={SKETCH_RANK})",
                "—", dt_compiled)
    logger.info("  Sanity ||G(X_gt)@Q|| = %.6f (should be ~0)", sanity_norm)
    logger.info("  Sketch norms at X=GT: left=%.4f, right=%.4f (should be ~0)",
                left_norm, right_norm)
    logger.info("Done. Plots in %s/plots/", out_dir)


if __name__ == "__main__":
    main()
