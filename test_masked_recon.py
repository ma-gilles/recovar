#!/usr/bin/env python3
"""Numerical test: masked reconstruction with soft penalty + gridding.

Tests the solver from pcg_variants on a single homogeneous reconstruction
(q=1), independent of EM. This isolates masking/gridding behavior.

Test:
  1. Generate a known volume with compact support
  2. Simulate projections (CTF + noise)
  3. Accumulate backprojection (d, r) via the standard pipeline
  4. Reconstruct with:
     a) Standard Wiener (no mask, no gridding)
     b) Wiener + gridding post-processing
     c) Wiener + mask projection + gridding post-processing
     d) Soft-alpha CG, no gridding in objective, gridding post-processing
     e) Soft-alpha CG, gridding in objective (correct formulation)
  5. Compare: real-space error, FSC vs ground truth
"""

import logging
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}", flush=True)

try:
    from recovar.cuda_backproject import _ensure_ffi
    _ensure_ffi()
except Exception:
    pass

import recovar.core.fourier_transform_utils as ftu
from recovar.reconstruction import relion_functions
from recovar.reconstruction.pcg_variants import (
    solve, build_alpha_weight, compute_gridding_kernel_real,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test_masked_recon")

GS = 128
OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/masked_recon_test_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def make_test_volume(gs):
    """Sphere with internal structure, compact support."""
    x = np.linspace(-1, 1, gs, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Main sphere
    vol = np.where(r < 0.3, 1.0, 0.0).astype(np.float32)
    # Internal structure: sinusoidal density
    vol += 0.5 * np.cos(8 * np.pi * X) * np.cos(8 * np.pi * Y) * np.where(r < 0.25, 1.0, 0.0)
    # Smooth edges
    edge = np.clip((0.32 - r) / 0.04, 0, 1)
    vol = vol * edge
    return vol


def make_mask(vol, gs, dilate=5):
    """Binary mask from volume + dilation."""
    from scipy.ndimage import binary_dilation
    binary = (np.abs(vol) > 0.01 * np.max(np.abs(vol)))
    mask = binary_dilation(binary, iterations=dilate).astype(np.float32)
    return mask


def simulate_data(vol, gs, n_images=10000, noise_level=1.0, seed=42):
    """Simulate cryo-EM projections and accumulate backprojection.

    Returns d (Fourier weights), r_hat (backprojected data) in half-volume.
    Also returns the forward model's accumulated d without noise (for SNR).
    """
    np.random.seed(seed)
    vs = (gs, gs, gs)
    hvs = ftu.get_real_fft_packed_shape(vs)
    hv = int(np.prod(hvs))

    vol_fourier = ftu.get_dft3_real(jnp.array(vol).reshape(vs))  # half-vol
    vol_flat = vol_fourier.reshape(-1)

    # Accumulate d (CTF²) and r (CTF * y) in half-volume
    d = jnp.zeros(hv, dtype=jnp.float32)
    r_hat = jnp.zeros(hv, dtype=jnp.complex64)

    # Simple simulation: random CTF defoci, uniform orientations
    # For this test, we use a simplified model:
    # Each "image" contributes uniformly to all Fourier voxels
    # with a random CTF amplitude
    for batch_start in range(0, n_images, 500):
        batch_end = min(batch_start + 500, n_images)
        nb = batch_end - batch_start

        # Random CTF: simple defocus model
        defoci = 0.5 + 2.0 * np.random.rand(nb)  # microns
        # CTF at each Fourier voxel (simplified: radial)
        freq = np.sqrt(np.sum(
            np.mgrid[:hvs[0], :hvs[1], :hvs[2]].astype(np.float32)**2,
            axis=0
        )).ravel() / gs

        for i in range(nb):
            ctf = jnp.sin(jnp.pi * defoci[i] * 1e4 * freq**2)
            ctf2 = ctf ** 2
            noise = jnp.array(
                np.random.randn(hv).astype(np.float32)
                + 1j * np.random.randn(hv).astype(np.float32)
            ) * noise_level / np.sqrt(2)
            # Forward: CTF * vol + noise
            y_hat = ctf * vol_flat + noise
            # Accumulate
            d = d + ctf2
            r_hat = r_hat + ctf * y_hat

    logger.info("Simulated %d images, d range=[%.1f, %.1f]",
                n_images, float(d.min()), float(d.max()))
    return d, r_hat


def wiener_solve(d, r_hat, gs, reg=1.0):
    """Standard Wiener filter: w_hat = r / (d + reg)."""
    hvs = ftu.get_real_fft_packed_shape((gs, gs, gs))
    w_hat = r_hat / jnp.maximum(d + reg, 1e-6)
    return ftu.get_idft3_real(w_hat.reshape(hvs), (gs, gs, gs))


def apply_gridding_correction(vol, gs):
    """Post-processing gridding correction."""
    corrected, _ = relion_functions.griddingCorrect_square(
        vol.reshape(gs, gs, gs), gs, 1, order=1)
    return corrected


def fsc_curve(vol1, vol2, gs):
    """Fourier shell correlation between two real-space volumes."""
    vs = (gs, gs, gs)
    f1 = ftu.get_dft3(jnp.array(vol1).reshape(vs)).reshape(-1)
    f2 = ftu.get_dft3(jnp.array(vol2).reshape(vs)).reshape(-1)

    # Radial bins
    coords = np.mgrid[:gs, :gs, :gs].astype(np.float32)
    coords = coords - gs / 2
    r = np.sqrt(np.sum(coords**2, axis=0)).ravel()
    r_int = np.round(r).astype(int)
    max_r = gs // 2

    fsc = np.zeros(max_r)
    for shell in range(max_r):
        idx = (r_int == shell)
        if idx.sum() == 0:
            continue
        num = float(jnp.sum(jnp.conj(f1[idx]) * f2[idx]).real)
        den = float(jnp.sqrt(jnp.sum(jnp.abs(f1[idx])**2)
                             * jnp.sum(jnp.abs(f2[idx])**2)))
        fsc[shell] = num / max(den, 1e-10)
    return fsc


def real_space_error(recon, gt, mask=None):
    """Relative L2 error, optionally restricted to mask."""
    if mask is not None:
        diff = (recon - gt) * mask
        ref = gt * mask
    else:
        diff = recon - gt
        ref = gt
    return float(jnp.linalg.norm(diff) / max(float(jnp.linalg.norm(ref)), 1e-10))


if __name__ == "__main__":
    logger.info("=== Masked reconstruction test: %d³ ===", GS)

    # 1. Generate volume and mask
    vol_gt = make_test_volume(GS)
    mask = make_mask(vol_gt, GS, dilate=3)
    n_mask = np.sum(mask > 0.5)
    logger.info("Volume: ||v||=%.4f, mask: %d voxels (%.1f%%)",
                np.linalg.norm(vol_gt), n_mask, 100 * n_mask / GS**3)

    # 2. Simulate data
    d, r_hat = simulate_data(vol_gt, GS, n_images=10000, noise_level=0.5)

    # 3. Reconstruct with different methods
    vs = (GS, GS, GS)
    hvs = ftu.get_real_fft_packed_shape(vs)
    hv = int(np.prod(hvs))
    reg = float(jnp.median(d)) * 0.01  # small regularization

    results = {}

    # (a) Wiener, no mask, no gridding
    t0 = time.time()
    w_wiener = wiener_solve(d, r_hat, GS, reg)
    dt = time.time() - t0
    results["wiener"] = {
        "vol": np.array(w_wiener),
        "time": dt,
        "err": real_space_error(w_wiener, vol_gt),
        "err_mask": real_space_error(w_wiener, vol_gt, mask),
    }
    logger.info("Wiener: err=%.4f err_mask=%.4f time=%.2fs",
                results["wiener"]["err"], results["wiener"]["err_mask"], dt)

    # (b) Wiener + gridding post-processing
    t0 = time.time()
    w_wiener_grid = apply_gridding_correction(w_wiener, GS)
    dt = time.time() - t0
    results["wiener+grid"] = {
        "vol": np.array(w_wiener_grid),
        "time": dt,
        "err": real_space_error(w_wiener_grid, vol_gt),
        "err_mask": real_space_error(w_wiener_grid, vol_gt, mask),
    }
    logger.info("Wiener+grid: err=%.4f err_mask=%.4f",
                results["wiener+grid"]["err"], results["wiener+grid"]["err_mask"])

    # (c) Wiener + mask projection + gridding
    t0 = time.time()
    w_masked = jnp.array(mask) * w_wiener
    w_masked_grid = apply_gridding_correction(w_masked, GS)
    dt = time.time() - t0
    results["wiener+mask+grid"] = {
        "vol": np.array(w_masked_grid),
        "time": dt,
        "err": real_space_error(w_masked_grid, vol_gt),
        "err_mask": real_space_error(w_masked_grid, vol_gt, mask),
    }
    logger.info("Wiener+mask+grid: err=%.4f err_mask=%.4f",
                results["wiener+mask+grid"]["err"],
                results["wiener+mask+grid"]["err_mask"])

    # For solve(), need q=1 format: (half_vol, 1) for lhs/rhs/reg
    d_q1 = (d + reg).reshape(-1, 1, 1)  # (hv, 1, 1) — acts as 1x1 LHS
    r_q1 = r_hat.reshape(-1, 1)  # (hv, 1) — RHS
    reg_q1 = jnp.full((hv, 1), reg, dtype=jnp.float32)

    collar = max(3, round(0.04 * GS))

    # (d) Soft-alpha CG, no gridding in objective, gridding as post-processing
    t0 = time.time()
    w_sa_post, info_d = solve(
        d.reshape(-1, 1, 1), r_q1, reg_q1,
        jnp.array(mask).reshape(vs), vs,
        lam=100.0, collar_width=collar,
        maxiter=50, tol=1e-4, use_gridding=False,
    )
    w_sa_post = w_sa_post[0]  # (1, D, D, D) → (D, D, D)
    w_sa_post_grid = apply_gridding_correction(w_sa_post, GS)
    dt = time.time() - t0
    results["soft_alpha+postgrid"] = {
        "vol": np.array(w_sa_post_grid),
        "time": dt,
        "err": real_space_error(w_sa_post_grid, vol_gt),
        "err_mask": real_space_error(w_sa_post_grid, vol_gt, mask),
        "cg_iters": info_d["n_iters"],
        "residuals": info_d["residuals"],
    }
    logger.info("Soft-alpha+postgrid: err=%.4f err_mask=%.4f iters=%d",
                results["soft_alpha+postgrid"]["err"],
                results["soft_alpha+postgrid"]["err_mask"],
                info_d["n_iters"])

    # (e) Soft-alpha CG, gridding IN objective
    t0 = time.time()
    w_sa_grid, info_e = solve(
        d.reshape(-1, 1, 1), r_q1, reg_q1,
        jnp.array(mask).reshape(vs), vs,
        lam=100.0, collar_width=collar,
        maxiter=50, tol=1e-4, use_gridding=True,
    )
    w_sa_grid = w_sa_grid[0]  # (D, D, D)
    dt = time.time() - t0
    results["soft_alpha+gridinobj"] = {
        "vol": np.array(w_sa_grid),
        "time": dt,
        "err": real_space_error(w_sa_grid, vol_gt),
        "err_mask": real_space_error(w_sa_grid, vol_gt, mask),
        "cg_iters": info_e["n_iters"],
        "residuals": info_e["residuals"],
    }
    logger.info("Soft-alpha+gridinobj: err=%.4f err_mask=%.4f iters=%d",
                results["soft_alpha+gridinobj"]["err"],
                results["soft_alpha+gridinobj"]["err_mask"],
                info_e["n_iters"])

    # Summary table
    print(f"\n{'='*70}")
    print(f"Masked reconstruction test: {GS}³, 10k images, q=1")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'L2 err':>8} {'L2 mask':>8} {'Time':>6} {'CG it':>5}")
    print("-" * 60)
    for name in ["wiener", "wiener+grid", "wiener+mask+grid",
                  "soft_alpha+postgrid", "soft_alpha+gridinobj"]:
        r = results[name]
        iters = r.get("cg_iters", "-")
        print(f"{name:<30} {r['err']:8.4f} {r['err_mask']:8.4f} "
              f"{r['time']:6.1f} {iters:>5}")

    # FSC curves
    print("\nFSC at shell 10, 20, 30:")
    for name in results:
        fsc = fsc_curve(results[name]["vol"], vol_gt, GS)
        print(f"  {name:<30} {fsc[10]:.3f}  {fsc[20]:.3f}  {fsc[30]:.3f}")

    # Save
    import pickle
    with open(os.path.join(OUT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        mid = GS // 2
        for i, name in enumerate(results):
            axes[0, i].imshow(results[name]["vol"][mid], cmap="gray")
            axes[0, i].set_title(f"{name}\nerr={results[name]['err']:.4f}")
            axes[0, i].axis("off")

            diff = results[name]["vol"][mid] - vol_gt[mid]
            vmax = np.max(np.abs(vol_gt[mid])) * 0.3
            axes[1, i].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[1, i].set_title("error")
            axes[1, i].axis("off")

        plt.suptitle(f"Masked reconstruction: {GS}³, q=1", fontsize=14)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "comparison.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"\nPlot: {p}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nResults saved to: {OUT_DIR}")
