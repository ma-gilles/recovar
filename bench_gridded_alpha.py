#!/usr/bin/env python3
"""Soft-alpha with gridding-in-the-objective: 128³ and 256³.

Tests:
- Gridding in objective vs gridding as post-processing vs no gridding
- λ sweep: {10, 100, 500}
- Resolution-scaled collar: collar_frac ≈ 0.04 of grid size
- CG iteration budget: 20 vs 50 vs 100
- With and without warmstart
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}", flush=True)
assert any("gpu" in str(d).lower() or "cuda" in str(d).lower()
           for d in jax.devices()), "No GPU found!"

try:
    from recovar.cuda_backproject import _ensure_ffi
    _ensure_ffi()
except Exception:
    pass

from recovar.ppca import ppca
from recovar.ppca.ppca_scale_sweep import (
    _load_simulated_dataset,
    _with_trailing_separator,
)
import recovar.core.fourier_transform_utils as ftu
from recovar.core.mask import make_moving_gt_mask
from recovar.output import metrics
from recovar.reconstruction import regularization
from recovar.reconstruction import pcg_variants as pv
from recovar.utils import batch_make_radial_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_gridded")

NPC = 10
NI = 50000
NITER = 20
SEED = 42
COLLAR_FRAC = 0.04  # collar as fraction of grid size

OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/gridded_alpha_study_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset(gs):
    dataset_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_pcg_5nrl_{gs}/test_dataset"
    cryos, sim_info, gt, nv = _load_simulated_dataset(
        _with_trailing_separator(dataset_dir), gs, NI, lazy=False
    )
    vs = gt.volume_shape
    U_gt_all, s_gt_all, _ = gt.get_vol_svd()
    gt_mean = gt.get_mean()
    real_vols = [
        np.asarray(ftu.get_idft3(gt.volumes[i].reshape(vs)).real)
        for i in range(gt.volumes.shape[0])
    ]
    mov_soft, mov_bin = make_moving_gt_mask(real_vols, vs)
    U_gt = U_gt_all[:, :NPC]
    s_gt = s_gt_all[:NPC]
    W_gt = U_gt * s_gt
    w_avg = regularization.batch_average_over_shells(
        jnp.abs(W_gt.T) ** 2, vs, 0
    )
    W_prior = batch_make_radial_image(w_avg, vs, True).T
    np.random.seed(SEED)
    W_init = jnp.array(
        np.random.randn(gt_mean.shape[0], NPC).astype(np.float32) * 0.01
    )
    mask_pct = np.mean(mov_bin > 0.5) * 100
    logger.info("Dataset: %d³ q=%d NI=%d mask=%.1f%%", gs, NPC, NI, mask_pct)
    return cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs


def run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mask_arr,
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
    iter_rvs = [float(d.get("Rel_Var_Explained", 0))
                for d in idata if isinstance(d, dict)
                and d.get("Rel_Var_Explained") is not None]
    return {"label": label, "relvar": float(rv[-1]),
            "relvar_per_pc": [float(x) for x in rv],
            "time": dt, "iter_rvs": iter_rvs}


def make_gridded_solver(lam, collar, use_gridding=True):
    """Soft-alpha with gridding in the objective."""
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return pv.solve_soft_alpha_gridded(
            lhs, rhs, reg, mask, vol_shape,
            lam=lam, collar_width=collar, outer_dilate=3,
            W0_real=W0_real, maxiter=maxiter, tol=tol,
            unpack_fn=unpack_fn, use_gridding=use_gridding,
        )
    return solver_fn


def make_noprecond_solver():
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return pv.solve_no_precond(lhs, rhs, reg, mask, vol_shape,
                                   W0_real=W0_real, maxiter=maxiter,
                                   tol=tol, unpack_fn=unpack_fn)
    return solver_fn


def benchmark_grid_size(gs):
    logger.info("=" * 70)
    logger.info("BENCHMARK: %d³, q=%d, %d images", gs, NPC, NI)
    logger.info("=" * 70)

    cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs = \
        load_dataset(gs)
    mask_soft = np.array(mov_soft, dtype=np.float32)
    mask_bin = np.array(mov_bin, dtype=np.float32)

    collar = max(3, round(COLLAR_FRAC * gs))  # resolution-scaled
    logger.info("Using collar=%d (%.1f%% of grid)", collar, 100*collar/gs)

    # Build outer support for soft-alpha solver
    # The EM applies volume_mask as safety projection after each M-step.
    # For soft-alpha, pass the outer_support so the collar is not clipped.
    alpha, outer_support_arr = pv.build_alpha_weight(
        np.asarray(mov_bin > 0.5, dtype=bool),
        collar_width=collar, outer_dilate=3,
    )
    outer_support_f32 = np.array(outer_support_arr, dtype=np.float32)
    n_outer = np.sum(outer_support_f32 > 0.5)
    logger.info("Outer support: %d voxels (%.1f%% of grid)",
                n_outer, 100 * n_outer / np.prod(vs))

    results = []

    # Baselines (use appropriate mask for EM's safety projection)
    configs = [
        ("mask_proj", mask_soft, False, None, 20, False),
        ("mask_proj+grid", mask_soft, False, None, 20, True),
        ("pcg_20it+grid", mask_bin, True, None, 20, True),
        ("noprecond_50it+grid", mask_bin, False,
         make_noprecond_solver(), 50, True),
    ]

    # Soft-alpha: gridding in objective (no post-processing gridding)
    # Pass outer_support as volume_mask so EM doesn't clip collar
    for lam in [10, 100, 500]:
        configs.append((
            f"sa_grid_l{lam}_c{collar}_50it",
            outer_support_f32, False,
            make_gridded_solver(lam, collar, use_gridding=True), 50, False,
        ))

    # Soft-alpha: NO gridding at all (for comparison)
    for lam in [10, 100]:
        configs.append((
            f"sa_nogrid_l{lam}_c{collar}_50it",
            outer_support_f32, False,
            make_gridded_solver(lam, collar, use_gridding=False), 50, False,
        ))

    # Soft-alpha: gridding as post-processing (original approach)
    for lam in [10, 100]:
        configs.append((
            f"sa_postGrid_l{lam}_c{collar}_50it",
            outer_support_f32, False,
            make_gridded_solver(lam, collar, use_gridding=False), 50, True,
        ))

    # Iteration budget for best config
    for maxiter in [20, 100]:
        configs.append((
            f"sa_grid_l100_c{collar}_{maxiter}it",
            outer_support_f32, False,
            make_gridded_solver(100, collar, use_gridding=True), maxiter, False,
        ))

    for label, mask_arr, use_pcg, solver_fn, pcg_max, gridding in configs:
        try:
            r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                      mask_arr, use_pcg, solver_fn, pcg_max, gridding, label)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", label, e)
            traceback.print_exc()
            results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    return results


if __name__ == "__main__":
    all_results = {}

    for gs in [128, 256]:
        results = benchmark_grid_size(gs)
        all_results[gs] = results

        with open(os.path.join(OUT_DIR, f"results_{gs}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"RESULTS: {gs}³, q={NPC}, {NI} images, {NITER} EM iters")
        print(f"{'='*70}")
        print(f"{'Method':<45} {'RelVar':>8} {'Time(s)':>8}")
        print("-" * 65)
        for r in sorted(results, key=lambda x: -x.get("relvar", 0)):
            print(f"{r['label']:<45} {r.get('relvar', 0):8.4f} "
                  f"{r.get('time', 0):8.0f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for gs, results in all_results.items():
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            for r in results:
                rvs = r.get("iter_rvs", [])
                if rvs:
                    ax.plot(range(1, len(rvs) + 1), rvs,
                           label=f"{r['label']} ({r.get('relvar', 0):.4f})",
                           linewidth=1.5)
            ax.set_xlabel("EM iteration")
            ax.set_ylabel("RelVar")
            ax.set_title(f"Gridded soft-alpha: {gs}³ q={NPC}")
            ax.legend(fontsize=6, ncol=2, loc="lower right")
            ax.grid(True, alpha=0.3)
            p = os.path.join(OUT_DIR, f"convergence_{gs}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot: {p}")
    except Exception as e:
        print(f"Plot failed: {e}")

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll saved to: {OUT_DIR}")
