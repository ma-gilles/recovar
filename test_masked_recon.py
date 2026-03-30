#!/usr/bin/env python3
"""Numerical test: masked reconstruction with soft penalty + gridding.

Uses the existing 128³ simulated dataset (same as PPCA benchmarks).
Runs homogeneous (q=1) mean reconstruction comparing:
  a) Wiener + gridding post-processing (standard)
  b) Wiener + mask projection + gridding (current mask approach)
  c) Soft-alpha CG, gridding as post-processing
  d) Soft-alpha CG, gridding in objective (correct formulation)
  e) Soft-alpha CG, no gridding at all

Measures: real-space error vs GT mean, FSC, wall-clock time.
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
from recovar.reconstruction import relion_functions, regularization
from recovar.reconstruction.pcg_variants import (
    solve, build_alpha_weight, compute_gridding_kernel_real,
)
from recovar.core.mask import make_moving_gt_mask
from recovar.ppca.ppca_scale_sweep import _load_simulated_dataset, _with_trailing_separator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test")

GS = 128
NI = 50000
OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/masked_recon_test_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def fsc_curve(vol1, vol2, vs):
    """Fourier shell correlation."""
    f1 = ftu.get_dft3(jnp.array(vol1).reshape(vs)).reshape(-1)
    f2 = ftu.get_dft3(jnp.array(vol2).reshape(vs)).reshape(-1)
    D = vs[0]
    coords = np.mgrid[:D, :D, :D].astype(np.float32) - D / 2
    r = np.sqrt(np.sum(coords**2, axis=0)).ravel().astype(int)
    fsc = np.zeros(D // 2)
    for s in range(D // 2):
        idx = (r == s)
        if idx.sum() == 0:
            continue
        num = float(jnp.sum(jnp.conj(f1[idx]) * f2[idx]).real)
        den = float(jnp.sqrt(jnp.sum(jnp.abs(f1[idx])**2)
                             * jnp.sum(jnp.abs(f2[idx])**2)))
        fsc[s] = num / max(den, 1e-10)
    return fsc


def real_space_error(recon, gt, mask=None):
    if mask is not None:
        return float(jnp.linalg.norm((recon - gt) * mask)
                     / max(float(jnp.linalg.norm(gt * mask)), 1e-10))
    return float(jnp.linalg.norm(recon - gt)
                 / max(float(jnp.linalg.norm(gt)), 1e-10))


if __name__ == "__main__":
    logger.info("=== Loading %d³ dataset ===", GS)
    dataset_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_pcg_5nrl_{GS}/test_dataset"
    cryos, sim_info, gt, nv = _load_simulated_dataset(
        _with_trailing_separator(dataset_dir), GS, NI, lazy=False)
    vs = gt.volume_shape

    # Ground truth mean (real space)
    gt_mean_fourier = gt.get_mean()  # Fourier, flat
    gt_mean_real = np.array(ftu.get_idft3(
        jnp.array(gt_mean_fourier).reshape(vs)).real)

    # Mask
    real_vols = [np.asarray(ftu.get_idft3(gt.volumes[i].reshape(vs)).real)
                 for i in range(gt.volumes.shape[0])]
    mov_soft, mov_bin = make_moving_gt_mask(real_vols, vs)
    mask = np.array(mov_bin, dtype=np.float32)
    n_mask = np.sum(mask > 0.5)
    logger.info("mask: %d voxels (%.1f%%)", n_mask, 100 * n_mask / np.prod(vs))

    # Accumulate backprojection using recovar's pipeline
    logger.info("Accumulating backprojection...")
    from recovar.ppca.ppca import _normalize_experiment_datasets
    full_ds, ds_list = _normalize_experiment_datasets(cryos)
    ref = full_ds if full_ds is not None else ds_list[0]
    noise_var = np.array(ref.noise.noise_variance_radial)

    t0 = time.time()
    ft_ctf, ft_y = relion_functions.relion_style_triangular_kernel(
        ref, noise_var.astype(np.float32), batch_size=200)
    logger.info("Backprojection done: %.1fs", time.time() - t0)

    # ft_ctf: CTF² weights, ft_y: backprojected data (both full Fourier, flat)
    # Convert to half-volume for our solver
    ft_ctf_half = ftu.full_volume_to_half_volume(
        jnp.array(ft_ctf).reshape(vs), vs).reshape(-1).real
    ft_y_half = ftu.full_volume_to_half_volume(
        jnp.array(ft_y).reshape(vs), vs).reshape(-1)

    hvs = ftu.get_real_fft_packed_shape(vs)
    hv = int(np.prod(hvs))

    # Regularization (simple Tikhonov)
    reg_val = float(jnp.median(ft_ctf_half)) * 0.01
    logger.info("reg=%.2e, ft_ctf range=[%.1f, %.1f]",
                reg_val, float(ft_ctf_half.min()), float(ft_ctf_half.max()))

    results = {}

    # (a) Standard Wiener + gridding post-processing
    t0 = time.time()
    w_hat = ft_y_half / jnp.maximum(ft_ctf_half + reg_val, 1e-6)
    w_real = ftu.get_idft3_real(w_hat.reshape(hvs), vs)
    w_grid, _ = relion_functions.griddingCorrect_square(w_real, GS, 1, order=1)
    dt = time.time() - t0
    results["wiener+grid"] = {"vol": np.array(w_grid), "time": dt}
    logger.info("Wiener+grid: time=%.2fs", dt)

    # (b) Wiener + mask projection + gridding
    t0 = time.time()
    w_masked = jnp.array(mask) * w_real
    w_masked_grid, _ = relion_functions.griddingCorrect_square(w_masked, GS, 1, order=1)
    dt = time.time() - t0
    results["wiener+mask+grid"] = {"vol": np.array(w_masked_grid), "time": dt}

    # For solve(): q=1 format
    d_q1 = ft_ctf_half.reshape(-1, 1, 1)  # (hv, 1, 1)
    r_q1 = ft_y_half.reshape(-1, 1)       # (hv, 1)
    reg_q1 = jnp.full((hv, 1), reg_val, dtype=jnp.float32)
    collar = max(3, round(0.04 * GS))

    # (c) Soft-alpha, no gridding in objective, gridding post-processing
    t0 = time.time()
    w_sa, info_c = solve(d_q1, r_q1, reg_q1, jnp.array(mask), vs,
                         lam=100.0, collar_width=collar,
                         maxiter=50, use_gridding=False)
    w_sa = w_sa[0]
    w_sa_grid, _ = relion_functions.griddingCorrect_square(w_sa, GS, 1, order=1)
    dt = time.time() - t0
    results["sa+postgrid"] = {"vol": np.array(w_sa_grid), "time": dt,
                               "iters": info_c["n_iters"]}
    logger.info("SA+postgrid: iters=%d time=%.2fs", info_c["n_iters"], dt)

    # (d) Soft-alpha, gridding IN objective
    t0 = time.time()
    w_sa_go, info_d = solve(d_q1, r_q1, reg_q1, jnp.array(mask), vs,
                            lam=100.0, collar_width=collar,
                            maxiter=50, use_gridding=True)
    w_sa_go = w_sa_go[0]
    dt = time.time() - t0
    results["sa+gridinobj"] = {"vol": np.array(w_sa_go), "time": dt,
                                "iters": info_d["n_iters"]}
    logger.info("SA+gridinobj: iters=%d time=%.2fs", info_d["n_iters"], dt)

    # (e) Soft-alpha, no gridding at all
    t0 = time.time()
    w_sa_ng, info_e = solve(d_q1, r_q1, reg_q1, jnp.array(mask), vs,
                            lam=100.0, collar_width=collar,
                            maxiter=50, use_gridding=False)
    w_sa_ng = w_sa_ng[0]
    dt = time.time() - t0
    results["sa+nogrid"] = {"vol": np.array(w_sa_ng), "time": dt,
                             "iters": info_e["n_iters"]}

    # Compute metrics
    print(f"\n{'='*70}")
    print(f"Masked reconstruction test: {GS}³, {NI} images, q=1")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'L2 err':>8} {'L2 mask':>8} {'Time':>6} {'CG':>4}")
    print("-" * 55)
    for name, r in results.items():
        err = real_space_error(r["vol"], gt_mean_real)
        err_m = real_space_error(r["vol"], gt_mean_real, mask)
        r["err"] = err
        r["err_mask"] = err_m
        iters = r.get("iters", "-")
        print(f"{name:<25} {err:8.4f} {err_m:8.4f} {r['time']:6.1f} {iters:>4}")

    # FSC
    print(f"\n{'Method':<25} {'FSC@10':>7} {'FSC@20':>7} {'FSC@30':>7} {'FSC@40':>7}")
    print("-" * 55)
    for name, r in results.items():
        fsc = fsc_curve(r["vol"], gt_mean_real, vs)
        r["fsc"] = fsc.tolist()
        print(f"{name:<25} {fsc[10]:7.3f} {fsc[20]:7.3f} {fsc[30]:7.3f} {fsc[40]:7.3f}")

    # Save
    import json
    saveable = {k: {kk: vv for kk, vv in v.items() if kk != "vol"}
                for k, v in results.items()}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(saveable, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(results.keys())
        fig, axes = plt.subplots(3, len(names), figsize=(4*len(names), 10))
        mid = GS // 2
        for i, name in enumerate(names):
            v = results[name]["vol"]
            axes[0, i].imshow(v[mid], cmap="gray")
            axes[0, i].set_title(f"{name}\nerr={results[name]['err']:.4f}", fontsize=8)
            axes[0, i].axis("off")

            diff = v[mid] - gt_mean_real[mid]
            vmax = np.max(np.abs(gt_mean_real[mid])) * 0.2
            axes[1, i].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[1, i].set_title("error", fontsize=8)
            axes[1, i].axis("off")

            fsc = results[name].get("fsc", [])
            if fsc:
                axes[2, i].plot(fsc)
                axes[2, i].axhline(0.143, color="r", ls="--", lw=0.5)
                axes[2, i].set_ylim(-0.1, 1.05)
                axes[2, i].set_title("FSC", fontsize=8)

        plt.suptitle(f"Masked recon: {GS}³, {NI} images", fontsize=12)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "comparison.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"\nPlot: {p}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nResults: {OUT_DIR}")
