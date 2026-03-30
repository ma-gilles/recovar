#!/usr/bin/env python3
"""Soft-penalty-in-the-objective benchmark: smooth α(x) formulation.

Tests the clean soft penalty:
  min_w Φ(w) + (λ/2) Σ_x α(x) |w(x)|²

where α(x) = 0 in core, smooth ramp in collar, 1 outside, hard zero beyond
generous outer support.

Sweeps: λ ∈ {10, 100, 500, 1000}, collar_width ∈ {3, 5, 8}
Tests on: 128³ q=10, 256³ q=10, both 50k images.

Compares against:
  - mask projection (soft mask)
  - mask projection + gridding
  - PCG baseline (hard mask, 20 CG iters)
  - unpreconditioned CG 50 iters (hard mask)
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

# Verify GPU
print(f"JAX devices: {jax.devices()}", flush=True)
assert any("gpu" in str(d).lower() or "cuda" in str(d).lower()
           for d in jax.devices()), "No GPU found!"

try:
    from recovar.cuda_backproject import _ensure_ffi
    _ensure_ffi()
except Exception:
    pass

from recovar.ppca import ppca
from recovar.ppca.ppca import unpack_tri_to_full
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
logger = logging.getLogger("bench_soft_alpha")

NITER = 20
SEED = 42
NI = 50000

OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/soft_alpha_study_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset(gs, npc):
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

    U_gt = U_gt_all[:, :npc]
    s_gt = s_gt_all[:npc]
    W_gt = U_gt * s_gt
    w_avg = regularization.batch_average_over_shells(
        jnp.abs(W_gt.T) ** 2, vs, 0
    )
    W_prior = batch_make_radial_image(w_avg, vs, True).T
    np.random.seed(SEED)
    W_init = jnp.array(
        np.random.randn(gt_mean.shape[0], npc).astype(np.float32) * 0.01
    )

    mask_pct = np.mean(mov_bin > 0.5) * 100
    logger.info("Dataset: %d³ q=%d NI=%d mask=%.1f%%", gs, npc, NI, mask_pct)
    return cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs


def run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mask_arr,
           use_pcg, solver_fn, pcg_maxiter, gridding, label):
    """Run full EM and return result dict."""
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

    # Extract per-iter RelVar
    iter_rvs = []
    for d in idata:
        if isinstance(d, dict) and d.get("Rel_Var_Explained") is not None:
            iter_rvs.append(float(d["Rel_Var_Explained"]))

    return {
        "label": label,
        "relvar": float(rv[-1]),
        "relvar_per_pc": [float(x) for x in rv],
        "time": dt,
        "iter_rvs": iter_rvs,
    }


def make_soft_alpha_solver(lam, collar_width, outer_dilate=3, precond=False):
    """Create a solver_fn that matches the mstep_solver_fn interface."""
    fn = pv.solve_soft_alpha if precond else pv.solve_soft_alpha_noprecond
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return fn(lhs, rhs, reg, mask, vol_shape,
                  lam=lam, collar_width=collar_width,
                  outer_dilate=outer_dilate,
                  W0_real=W0_real, maxiter=maxiter, tol=tol,
                  unpack_fn=unpack_fn)
    return solver_fn


def make_noprecond_solver():
    """Unpreconditioned CG on hard mask."""
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return pv.solve_no_precond(lhs, rhs, reg, mask, vol_shape,
                                   W0_real=W0_real, maxiter=maxiter,
                                   tol=tol, unpack_fn=unpack_fn)
    return solver_fn


def benchmark_grid_size(gs, npc=10):
    """Run all comparisons for one grid size."""
    logger.info("=" * 70)
    logger.info("BENCHMARK: %d³, q=%d, %d images", gs, npc, NI)
    logger.info("=" * 70)

    cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs = \
        load_dataset(gs, npc)

    mask_soft = np.array(mov_soft, dtype=np.float32)
    mask_bin = np.array(mov_bin, dtype=np.float32)

    results = []

    # --- Baselines ---
    baselines = [
        ("mask_proj", mask_soft, False, None, 20, False),
        ("mask_proj+grid", mask_soft, False, None, 20, True),
        ("pcg_20it", mask_bin, True, None, 20, False),
        ("pcg_20it+grid", mask_bin, True, None, 20, True),
        ("noprecond_50it+grid", mask_bin, False,
         make_noprecond_solver(), 50, True),
    ]

    for label, mask_arr, use_pcg, solver_fn, pcg_max, gridding in baselines:
        try:
            r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                      mask_arr, use_pcg, solver_fn, pcg_max, gridding, label)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", label, e)
            traceback.print_exc()
            results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    # --- Soft-alpha sweep ---
    for lam in [10, 100, 500, 1000]:
        for collar in [3, 5, 8]:
            # Without preconditioner (plain CG, worked better in round 1)
            label = f"soft_alpha_np_l{lam}_c{collar}+grid"
            try:
                solver = make_soft_alpha_solver(lam, collar, precond=False)
                r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                          mask_bin, False, solver, 50, True, label)
                results.append(r)
            except Exception as e:
                logger.error("FAILED %s: %s", label, e)
                traceback.print_exc()
                results.append({"label": label, "error": str(e), "relvar": 0.0})
            gc.collect()

    # Best soft-alpha with preconditioner for comparison
    for lam in [100, 1000]:
        label = f"soft_alpha_pc_l{lam}_c5+grid"
        try:
            solver = make_soft_alpha_solver(lam, 5, precond=True)
            r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                      mask_bin, False, solver, 50, True, label)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", label, e)
            traceback.print_exc()
            results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    return results


# ── Main ──
if __name__ == "__main__":
    all_results = {}

    for gs in [128, 256]:
        results = benchmark_grid_size(gs, npc=10)
        all_results[gs] = results

        # Save incrementally
        with open(os.path.join(OUT_DIR, f"results_{gs}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"results_{gs}.pkl"), "wb") as f:
            pickle.dump(results, f)

        # Print table
        print(f"\n{'='*70}")
        print(f"RESULTS: {gs}³, q=10, 50k images, {NITER} EM iterations")
        print(f"{'='*70}")
        print(f"{'Method':<40} {'RelVar':>8} {'Time(s)':>8}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: -x.get("relvar", 0)):
            print(f"{r['label']:<40} {r.get('relvar', 0):8.4f} "
                  f"{r.get('time', 0):8.0f}")

    # Generate convergence plots
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
            ax.set_title(f"Soft-alpha study: {gs}³, q=10, {NITER} EM iters")
            ax.legend(fontsize=6, ncol=2, loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            p = os.path.join(OUT_DIR, f"convergence_{gs}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot saved: {p}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Save combined
    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {OUT_DIR}")
    print("Done!")
