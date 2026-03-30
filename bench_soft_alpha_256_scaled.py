#!/usr/bin/env python3
"""Soft-alpha at 256³ with resolution-scaled collar widths.

collar_width should scale ~linearly with grid size:
  collar = round(collar_frac * grid_size)
  collar_frac ≈ 5/128 ≈ 0.04

So at 256³: collar ∈ {6, 10, 16} (matching 3, 5, 8 at 128³).

Also tests iteration budget: 20 vs 50 CG iters, warm vs cold start.
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
logger = logging.getLogger("bench_256_scaled")

GS = 256
NPC = 10
NI = 50000
NITER = 20
SEED = 42

OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/soft_alpha_256_scaled_20260330"
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset():
    dataset_dir = f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_pcg_5nrl_{GS}/test_dataset"
    cryos, sim_info, gt, nv = _load_simulated_dataset(
        _with_trailing_separator(dataset_dir), GS, NI, lazy=False
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
    logger.info("Dataset: %d³ q=%d NI=%d mask=%.1f%%", GS, NPC, NI, mask_pct)
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


def make_soft_alpha_solver(lam, collar_width, precond=False):
    fn = pv.solve_soft_alpha if precond else pv.solve_soft_alpha_noprecond
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return fn(lhs, rhs, reg, mask, vol_shape,
                  lam=lam, collar_width=collar_width, outer_dilate=3,
                  W0_real=W0_real, maxiter=maxiter, tol=tol,
                  unpack_fn=unpack_fn)
    return solver_fn


def make_noprecond_solver():
    def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                 maxiter=20, tol=1e-4, unpack_fn=None):
        return pv.solve_no_precond(lhs, rhs, reg, mask, vol_shape,
                                   W0_real=W0_real, maxiter=maxiter,
                                   tol=tol, unpack_fn=unpack_fn)
    return solver_fn


if __name__ == "__main__":
    cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs = \
        load_dataset()
    mask_soft = np.array(mov_soft, dtype=np.float32)
    mask_bin = np.array(mov_bin, dtype=np.float32)

    results = []

    # Baselines
    baselines = [
        ("mask_proj+grid", mask_soft, False, None, 20, True),
        ("pcg_20it+grid", mask_bin, True, None, 20, True),
        ("noprecond_50it+grid", mask_bin, False,
         make_noprecond_solver(), 50, True),
    ]
    for label, m, up, sf, pm, g in baselines:
        try:
            r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                      m, up, sf, pm, g, label)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", label, e)
            traceback.print_exc()
            results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    # Resolution-scaled collars: 6, 10, 16 at 256³
    # (equivalent to 3, 5, 8 at 128³)
    for lam in [10, 100, 500]:
        for collar in [6, 10, 16]:
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

    # Also test iteration budget: 20 iters vs 50 for best config
    for maxiter in [20, 100]:
        label = f"soft_alpha_np_l100_c10_{maxiter}it+grid"
        solver = make_soft_alpha_solver(100, 10, precond=False)
        try:
            r = run_em(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                      mask_bin, False, solver, maxiter, True, label)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", label, e)
            traceback.print_exc()
            results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    # Save
    with open(os.path.join(OUT_DIR, "results_256_scaled.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "results_256_scaled.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"\n{'='*70}")
    print(f"RESULTS: 256³, q=10, 50k images, {NITER} EM iterations")
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
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        for r in results:
            rvs = r.get("iter_rvs", [])
            if rvs:
                ax.plot(range(1, len(rvs) + 1), rvs,
                       label=f"{r['label']} ({r.get('relvar', 0):.4f})",
                       linewidth=1.5)
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("RelVar")
        ax.set_title(f"256³ q=10: resolution-scaled collar widths")
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        ax.grid(True, alpha=0.3)
        p = os.path.join(OUT_DIR, "convergence_256_scaled.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {p}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nResults: {OUT_DIR}")
