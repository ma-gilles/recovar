#!/usr/bin/env python3
"""Focused convergence study: effect of gridding in CG, per-iter diagnostics.

Three methods only:
  1. mask_proj+grid  (baseline: Wiener + mask projection + gridding post-proc)
  2. sa_postgrid     (soft-alpha CG, gridding as post-processing)
  3. sa_gridinobj    (soft-alpha CG, gridding G inside the CG operator)

Logs per CG iteration: residual norm, solution norm.
Logs per EM iteration: neg-LL (data, prior, total), RelVar.

Tests: 128³ and 256³, q=10, 50k images, 20 EM iterations, collar=4% of grid.
"""

import json
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
assert any("gpu" in str(d).lower() or "cuda" in str(d).lower()
           for d in jax.devices()), "No GPU!"

try:
    from recovar.cuda_backproject import _ensure_ffi
    _ensure_ffi()
except Exception:
    pass

from recovar.ppca import ppca
from recovar.ppca.ppca_scale_sweep import (
    _load_simulated_dataset, _with_trailing_separator,
)
import recovar.core.fourier_transform_utils as ftu
from recovar.core.mask import make_moving_gt_mask
from recovar.output import metrics
from recovar.reconstruction import regularization
from recovar.reconstruction import pcg_variants as pv
from recovar.utils import batch_make_radial_image

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench")

NPC = 10
NI = 50000
NITER = 20
SEED = 42
LAM = 100.0
CG_MAXITER = 50

OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/convergence_study_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset(gs):
    dataset_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_pcg_5nrl_{gs}/test_dataset"
    cryos, _, gt, _ = _load_simulated_dataset(
        _with_trailing_separator(dataset_dir), gs, NI, lazy=False)
    vs = gt.volume_shape
    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    gt_mean = gt.get_mean()
    real_vols = [np.asarray(ftu.get_idft3(gt.volumes[i].reshape(vs)).real)
                 for i in range(gt.volumes.shape[0])]
    mov_soft, mov_bin = make_moving_gt_mask(real_vols, vs)
    U_gt = U_gt_all[:, :NPC]
    s_gt = s_gt_all[:NPC]
    W_gt = U_gt * s_gt
    w_avg = regularization.batch_average_over_shells(
        jnp.abs(W_gt.T) ** 2, vs, 0)
    W_prior = batch_make_radial_image(w_avg, vs, True).T
    np.random.seed(SEED)
    W_init = jnp.array(
        np.random.randn(gt_mean.shape[0], NPC).astype(np.float32) * 0.01)
    logger.info("Dataset: %d³ q=%d mask=%.1f%%",
                gs, NPC, 100 * np.mean(mov_bin > 0.5))
    return cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs


def make_solver(lam, collar, use_gridding):
    """Solver that also collects per-CG-iteration residuals."""
    all_cg_residuals = []  # list of lists, one per EM iter

    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                  maxiter=20, tol=1e-4, unpack_fn=None):
        W_real, info = pv.solve(
            lhs, rhs, reg, mask, vol_shape,
            lam=lam, collar_width=collar, outer_dilate=3,
            W0_real=W0_real, maxiter=maxiter, tol=tol,
            unpack_fn=unpack_fn, use_gridding=use_gridding)
        all_cg_residuals.append(info.get("residuals", []))
        return W_real, info

    return solver_fn, all_cg_residuals


def run(cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mask_arr,
        use_pcg, solver_fn, pcg_maxiter, gridding, label):
    logger.info("=== %s ===", label)
    t0 = time.time()
    U, S, W, ez, sm, idata = ppca.EM(
        cryos, gt_mean, W_init.copy(), W_prior,
        U_gt=U_gt, S_gt=s_gt**2,
        EM_iter=NITER,
        use_whitening=False, sparse_PCA=False,
        disc_type_mean="cubic", disc_type="linear_interp",
        return_iteration_data=True,
        use_pcg_mean=use_pcg,
        volume_mask=mask_arr,
        pcg_maxiter=pcg_maxiter,
        use_gridding_correction=gridding,
        mstep_solver_fn=solver_fn,
    )
    dt = time.time() - t0
    _, rv, _ = metrics.get_all_variance_scores(U, U_gt, s_gt**2)
    logger.info("  RelVar=%.4f time=%.0fs", rv[-1], dt)

    em_data = []
    for d in idata:
        if isinstance(d, dict):
            em_data.append({
                "Neg_LL_Total": d.get("Neg_LL_Total"),
                "Neg_LL_Data": d.get("Neg_LL_Data"),
                "Neg_LL_Prior": d.get("Neg_LL_Prior"),
                "Rel_Var_Explained": d.get("Rel_Var_Explained"),
                "W_norm": d.get("W_norm"),
            })

    return {"label": label, "relvar": float(rv[-1]),
            "relvar_per_pc": [float(x) for x in rv],
            "time": dt, "em_data": em_data}


def benchmark(gs):
    logger.info("=" * 60)
    logger.info("BENCHMARK: %d³", gs)
    logger.info("=" * 60)

    cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs = \
        load_dataset(gs)
    mask_soft = np.array(mov_soft, dtype=np.float32)
    mask_bin = np.array(mov_bin, dtype=np.float32)
    collar = max(3, round(0.04 * gs))

    # Build outer support for soft-alpha methods
    _, outer = pv.build_alpha_weight(
        np.asarray(mov_bin > 0.5, dtype=bool),
        collar_width=collar, outer_dilate=3)
    outer_f32 = np.array(outer, dtype=np.float32)

    results = {}

    # 1. mask_proj+grid (baseline)
    r = run(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
            mask_soft, False, None, 20, True, "mask_proj+grid")
    r["cg_residuals"] = []
    results["mask_proj+grid"] = r

    # 2. soft-alpha, gridding as post-processing
    solver_fn, cg_res = make_solver(LAM, collar, use_gridding=False)
    r = run(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
            outer_f32, False, solver_fn, CG_MAXITER, True,
            "sa_postgrid")
    r["cg_residuals"] = [list(x) for x in cg_res]
    results["sa_postgrid"] = r

    # 3. soft-alpha, gridding IN objective
    solver_fn, cg_res = make_solver(LAM, collar, use_gridding=True)
    r = run(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
            outer_f32, False, solver_fn, CG_MAXITER, False,
            "sa_gridinobj")
    r["cg_residuals"] = [list(x) for x in cg_res]
    results["sa_gridinobj"] = r

    return results


def benchmark_mean(gs):
    """Mean estimation (q=1): fast, isolates gridding/masking behavior."""
    logger.info("=" * 60)
    logger.info("MEAN ESTIMATION: %d³", gs)
    logger.info("=" * 60)

    dataset_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_pcg_5nrl_{gs}/test_dataset"
    cryos, _, gt, _ = _load_simulated_dataset(
        _with_trailing_separator(dataset_dir), gs, NI, lazy=False)
    vs = gt.volume_shape
    gt_mean_f = gt.get_mean()
    gt_mean_real = np.array(ftu.get_idft3(jnp.array(gt_mean_f).reshape(vs)).real)

    real_vols = [np.asarray(ftu.get_idft3(gt.volumes[i].reshape(vs)).real)
                 for i in range(gt.volumes.shape[0])]
    _, mov_bin = make_moving_gt_mask(real_vols, vs)
    mask = np.array(mov_bin, dtype=np.float32)
    collar = max(3, round(0.04 * gs))

    # Accumulate backprojection
    from recovar.ppca.ppca import _normalize_experiment_datasets
    from recovar.reconstruction import relion_functions
    full_ds, ds_list = _normalize_experiment_datasets(cryos)
    ref = full_ds if full_ds is not None else ds_list[0]
    noise_var = np.array(ref.noise.noise_variance_radial)

    t0 = time.time()
    ft_ctf, ft_y = relion_functions.relion_style_triangular_kernel(
        ref, noise_var.astype(np.float32), batch_size=200)
    logger.info("Backprojection: %.1fs", time.time() - t0)

    # Convert to half-volume
    d_half = ftu.full_volume_to_half_volume(
        jnp.array(ft_ctf).reshape(vs), vs).reshape(-1).real
    r_half = ftu.full_volume_to_half_volume(
        jnp.array(ft_y).reshape(vs), vs).reshape(-1)

    hvs = ftu.get_real_fft_packed_shape(vs)
    hv = int(np.prod(hvs))
    reg = float(jnp.median(d_half)) * 0.01

    def real_err(v):
        return float(jnp.linalg.norm(v - gt_mean_real) /
                     jnp.linalg.norm(gt_mean_real))

    def masked_err(v):
        m = jnp.array(mask)
        return float(jnp.linalg.norm((v - gt_mean_real) * m) /
                     jnp.linalg.norm(gt_mean_real * m))

    results = {}

    # q=1 format
    d_q1 = d_half.reshape(-1, 1, 1)
    r_q1 = r_half.reshape(-1, 1)
    reg_q1 = jnp.full((hv, 1), reg, dtype=jnp.float32)
    mask_j = jnp.array(mask).reshape(vs)

    # 1. Wiener + gridding (no mask)
    w_hat = r_half / jnp.maximum(d_half + reg, 1e-6)
    w = ftu.get_idft3_real(w_hat.reshape(hvs), vs)
    wg, _ = relion_functions.griddingCorrect_square(w, gs, 1, order=1)
    results["wiener+grid"] = {
        "vol": np.array(wg), "err": real_err(wg), "err_mask": masked_err(wg),
        "cg_residuals": []}

    # 2. Wiener + mask + gridding
    wm = jnp.array(mask) * w
    wmg, _ = relion_functions.griddingCorrect_square(wm, gs, 1, order=1)
    results["wiener+mask+grid"] = {
        "vol": np.array(wmg), "err": real_err(wmg), "err_mask": masked_err(wmg),
        "cg_residuals": []}

    # 3. Soft-alpha, gridding post-processing
    _, outer = pv.build_alpha_weight(mov_bin > 0.5, collar, 3)
    outer_f32 = np.array(outer, dtype=np.float32)

    for maxiter in [10, 20, 50, 100]:
        w_sa, info = pv.solve(d_q1, r_q1, reg_q1, mask_j, vs,
                              lam=LAM, collar_width=collar,
                              maxiter=maxiter, use_gridding=False)
        w_sa = w_sa[0]
        wsg, _ = relion_functions.griddingCorrect_square(w_sa, gs, 1, order=1)
        results[f"sa_postgrid_{maxiter}it"] = {
            "vol": np.array(wsg), "err": real_err(wsg),
            "err_mask": masked_err(wsg),
            "cg_residuals": [info["residuals"]]}

    # 4. Soft-alpha, gridding IN objective
    for maxiter in [10, 20, 50, 100]:
        w_go, info = pv.solve(d_q1, r_q1, reg_q1, mask_j, vs,
                              lam=LAM, collar_width=collar,
                              maxiter=maxiter, use_gridding=True)
        w_go = w_go[0]
        results[f"sa_gridinobj_{maxiter}it"] = {
            "vol": np.array(w_go), "err": real_err(w_go),
            "err_mask": masked_err(w_go),
            "cg_residuals": [info["residuals"]]}

    # Summary
    print(f"\n{'='*65}")
    print(f"Mean estimation: {gs}³, {NI} images, q=1, λ={LAM}, collar={collar}")
    print(f"{'='*65}")
    print(f"{'Method':<28} {'L2 err':>8} {'L2 mask':>8} {'CG':>4}")
    print("-" * 52)
    for name, r in results.items():
        iters = len(r["cg_residuals"][0]) if r["cg_residuals"] else "-"
        print(f"{name:<28} {r['err']:8.4f} {r['err_mask']:8.4f} {iters:>4}")

    return results


if __name__ == "__main__":
    all_results = {}

    # Mean estimation first (fast)
    for gs in [128, 256]:
        all_results[f"mean_{gs}"] = benchmark_mean(gs)

    # Then PPCA (slower)
    for gs in [128, 256]:
        results = benchmark(gs)
        all_results[gs] = results

        with open(os.path.join(OUT_DIR, f"results_{gs}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*50}")
        print(f"{gs}³: λ={LAM}, collar={max(3,round(0.04*gs))}")
        print(f"{'='*50}")
        for name, r in results.items():
            print(f"  {name:<20} RelVar={r['relvar']:.4f} time={r['time']:.0f}s")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for gs, results in all_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Top-left: EM RelVar convergence
            ax = axes[0, 0]
            for name, r in results.items():
                rvs = [d["Rel_Var_Explained"] for d in r["em_data"]
                       if d.get("Rel_Var_Explained") is not None]
                if rvs:
                    ax.plot(range(1, len(rvs)+1), rvs, "o-", label=name, ms=3)
            ax.set_xlabel("EM iteration")
            ax.set_ylabel("RelVar")
            ax.set_title("EM convergence: RelVar")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Top-right: EM neg-LL convergence
            ax = axes[0, 1]
            for name, r in results.items():
                ll = [d["Neg_LL_Data"] for d in r["em_data"]
                      if d.get("Neg_LL_Data") is not None]
                if ll:
                    ax.plot(range(1, len(ll)+1), ll, "o-", label=name, ms=3)
            ax.set_xlabel("EM iteration")
            ax.set_ylabel("Neg LL (data)")
            ax.set_title("EM convergence: Data log-likelihood")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Bottom-left: CG residuals (first EM iter)
            ax = axes[1, 0]
            for name, r in results.items():
                if r["cg_residuals"] and len(r["cg_residuals"]) > 0:
                    res = r["cg_residuals"][0]
                    if res:
                        ax.semilogy(range(1, len(res)+1), res, label=f"{name} (iter 0)")
            ax.set_xlabel("CG iteration")
            ax.set_ylabel("Relative residual")
            ax.set_title("CG convergence: EM iteration 0 (cold start)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Bottom-right: CG residuals (last EM iter, warmstarted)
            ax = axes[1, 1]
            for name, r in results.items():
                if r["cg_residuals"] and len(r["cg_residuals"]) > 1:
                    res = r["cg_residuals"][-1]
                    if res:
                        ax.semilogy(range(1, len(res)+1), res, label=f"{name} (iter {NITER-1})")
            ax.set_xlabel("CG iteration")
            ax.set_ylabel("Relative residual")
            ax.set_title(f"CG convergence: EM iteration {NITER-1} (warmstart)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.suptitle(f"{gs}³, q={NPC}, λ={LAM}, collar={max(3,round(0.04*gs))}", fontsize=13)
            plt.tight_layout()
            p = os.path.join(OUT_DIR, f"convergence_{gs}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot: {p}")

    except Exception as e:
        print(f"Plot failed: {e}")

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll saved to: {OUT_DIR}")
