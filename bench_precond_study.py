#!/usr/bin/env python3
"""Preconditioner/solver study for masked PPCA M-step.

Grid=128³, q=10, 50k images. Compares:
  1. mask_projection       — per-voxel solve + mask (no CG)
  2. baseline_circulant    — PCG with circulant preconditioner (current)
  3. no_precond            — plain CG, no preconditioner
  4. reduced_coord         — CG in reduced (mask-only) coordinates, no precond
  5. reduced_circulant     — reduced CG + circulant preconditioner
  6. reduced_block_jacobi  — reduced CG + block-Jacobi q×q preconditioner
  7. reduced_diag_jacobi   — reduced CG + scalar Jacobi preconditioner
  8. soft_penalty_100      — soft penalty λ=100
  9. soft_penalty_1000     — soft penalty λ=1000
 10. two_level             — reduced circulant + coarse correction

Two benchmark modes:
  A) Single M-step: run a few EM warmup iters with mask projection,
     then compare all solvers on the same accumulated LHS/RHS.
  B) Full EM: run 20 EM iterations with each solver, compare final RelVar.
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

# Use the worktree where the variants are implemented
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

# CUDA FFI
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
logger = logging.getLogger("bench_precond")

# ── Parameters ──
GS = 128
NPC = 10
NI = 50000
NITER_WARMUP = 5    # EM warmup iterations for single-M-step test
NITER_FULL = 20     # Full EM iterations
CG_MAXITER = 50     # CG iterations budget (generous for convergence study)
CG_TOL = 1e-6       # Tight tolerance to see full convergence curve
SEED = 42

OUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/precond_study_20260330"
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
    logger.info(
        "Dataset: %d³, q=%d, %d images, mask=%.1f%%",
        GS, NPC, NI, mask_pct,
    )
    return cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs


# =====================================================================
# Part A: Single M-step comparison
# =====================================================================

def run_single_mstep_comparison(cryos, gt_mean, W_init, W_prior, U_gt,
                                 s_gt, mov_bin, vs):
    """Run warmup EM, then compare all solvers on the same LHS/RHS."""
    logger.info("=" * 60)
    logger.info("PART A: Single M-step comparison")
    logger.info("=" * 60)

    # Step 1: Run warmup EM iterations with mask projection
    logger.info("Running %d warmup EM iterations (mask projection)...", NITER_WARMUP)
    mask_arr = np.array(mov_bin, dtype=np.float32)
    t0 = time.time()
    U, S, W, ez, sm, idata = ppca.EM(
        cryos, gt_mean, W_init.copy(), W_prior,
        U_gt=U_gt, S_gt=s_gt**2,
        EM_iter=NITER_WARMUP,
        use_whitening=False, sparse_PCA=False,
        disc_type_mean="cubic", disc_type="linear_interp",
        return_iteration_data=True,
        use_pcg_mean=False,  # mask projection
        volume_mask=mask_arr,
        use_gridding_correction=False,
    )
    dt_warmup = time.time() - t0
    _, rv_warmup, _ = metrics.get_all_variance_scores(U, U_gt, s_gt**2)
    logger.info("Warmup done: RelVar=%.4f in %.0fs", rv_warmup[-1], dt_warmup)

    # Step 2: Do one more E-step to get the accumulated LHS/RHS
    # We need to replicate the E-step accumulation from EM_step_half
    logger.info("Accumulating LHS/RHS for M-step comparison...")
    from recovar.ppca.ppca import (
        _normalize_experiment_datasets,
        _iter_processed_batches_half,
        E_M_step_batch_half,
        _tri_size,
    )

    full_dataset, dataset_list = _normalize_experiment_datasets(cryos)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    volume_shape = ref.volume_shape
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    tri_sz = _tri_size(NPC)
    batch_size = 200

    # Current W is in full Fourier from EM output. Convert to half-volume.
    W_full = W  # (vol_size, q) full Fourier
    W_half = ftu.full_volume_to_half_volume(W_full.T, volume_shape).T

    # Precompute cubic mean
    from recovar.core import precompute_cubic_coefficients
    mean_est_cubic = precompute_cubic_coefficients(gt_mean, volume_shape)

    lhs_summed = jnp.zeros((half_volume_size, tri_sz), dtype=ref.dtype_real)
    rhs_summed = jnp.zeros((half_volume_size, NPC), dtype=ref.dtype)

    for experiment_dataset in dataset_list:
        for (batch_half, ctf_params, rotation_matrices, translations,
             batch_image_ind) in _iter_processed_batches_half(
                 experiment_dataset, batch_size):
            noise_variance_half = experiment_dataset.noise.get_half(batch_image_ind)
            lhs_summed, rhs_summed, _, _, _, _ = E_M_step_batch_half(
                batch_half, lhs_summed, rhs_summed,
                mean_est_cubic, W_half,
                ctf_params, rotation_matrices, translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance_half,
                experiment_dataset.ctf_evaluator,
                compute_ll=False,
                disc_type_mean="cubic",
                disc_type="linear_interp",
                compute_stats=False,
            )

    # Regularization
    W_prior_half = ftu.full_volume_to_half_volume(W_prior.T, volume_shape).T
    reg_half = 1 / (W_prior_half + 1e-16)

    # Warmstart: current W in real space
    half_vs = ftu.get_real_fft_packed_shape(volume_shape)
    W_real_warmstart = np.asarray(
        ftu.get_idft3_real(W_half.T.reshape(NPC, *half_vs), volume_shape)
    )
    # Apply mask for PCG warmstart
    W_real_warmstart = np.asarray(mask_arr.reshape(volume_shape)[None] * W_real_warmstart)

    mask_jnp = jnp.array(mask_arr).reshape(volume_shape)

    logger.info("LHS/RHS accumulated. Testing solvers...")

    # Step 3: Test each solver variant
    results = {}

    solver_configs = [
        ("mask_projection", lambda: pv.solve_mask_projection(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            unpack_fn=unpack_tri_to_full)),
        ("baseline_circulant", lambda: pv.solve_baseline_circulant(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("baseline_circulant_cold", lambda: pv.solve_baseline_circulant(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=None, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("no_precond", lambda: pv.solve_no_precond(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("no_precond_cold", lambda: pv.solve_no_precond(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=None, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("reduced_coord", lambda: pv.solve_reduced_coord(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("reduced_circulant", lambda: pv.solve_reduced_circulant(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("reduced_circulant_cold", lambda: pv.solve_reduced_circulant(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=None, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("reduced_block_jacobi", lambda: pv.solve_reduced_jacobi(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full, block=True)),
        ("reduced_diag_jacobi", lambda: pv.solve_reduced_jacobi(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full, block=False)),
        ("soft_penalty_100", lambda: pv.solve_soft_penalty(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            soft_lam=100.0, W0_real=None, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("soft_penalty_1000", lambda: pv.solve_soft_penalty(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            soft_lam=1000.0, W0_real=None, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full)),
        ("two_level", lambda: pv.solve_reduced_two_level(
            lhs_summed, rhs_summed, reg_half, mask_jnp, volume_shape,
            W0_real=W_real_warmstart, maxiter=CG_MAXITER, tol=CG_TOL,
            unpack_fn=unpack_tri_to_full, n_coarse=8)),
    ]

    for name, solver_fn in solver_configs:
        logger.info("--- Solver: %s ---", name)
        try:
            t0 = time.time()
            W_real, info = solver_fn()
            dt = time.time() - t0

            # Compute quality metric (RelVar)
            W_full_test = ftu.get_dft3(
                W_real.reshape(NPC, *volume_shape)
            ).reshape(NPC, -1).T
            U_test, S_test, _ = jnp.linalg.svd(W_full_test, full_matrices=False)
            _, rv, _ = metrics.get_all_variance_scores(U_test, U_gt, s_gt**2)

            mem_bytes = 0
            try:
                mem_bytes = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
            except Exception:
                pass

            results[name] = {
                "label": name,
                "relvar": float(rv[-1]),
                "relvar_per_pc": [float(x) for x in rv],
                "wall_time": dt,
                "n_iters": info.get("n_iters", 0),
                "residuals": [float(x) for x in info.get("residuals", [])],
                "timings": [float(x) for x in info.get("timings", [])],
                "total_solver_time": info.get("total_time", dt),
                "peak_mem_bytes": mem_bytes,
            }
            logger.info(
                "  RelVar=%.4f iters=%d time=%.1fs",
                rv[-1], info.get("n_iters", 0), dt,
            )
        except Exception as e:
            logger.error("  FAILED: %s", e)
            traceback.print_exc()
            results[name] = {
                "label": name,
                "error": str(e),
                "relvar": 0.0,
            }
        gc.collect()

    return results


# =====================================================================
# Part B: Full EM comparison
# =====================================================================

def run_full_em_comparison(cryos, gt_mean, W_init, W_prior, U_gt, s_gt,
                           mov_soft, mov_bin, vs):
    """Run full EM with each solver/method and compare final quality."""
    logger.info("=" * 60)
    logger.info("PART B: Full EM comparison (%d iterations)", NITER_FULL)
    logger.info("=" * 60)

    mask_soft = np.array(mov_soft, dtype=np.float32)
    mask_bin = np.array(mov_bin, dtype=np.float32)

    # Define solver wrappers that match the mstep_solver_fn interface
    # Interface: fn(lhs, rhs, reg, mask, vol_shape, W0_real=, maxiter=, tol=, unpack_fn=)
    def make_variant_solver(variant_fn, **extra_kwargs):
        def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                     maxiter=20, tol=1e-4, unpack_fn=None):
            return variant_fn(lhs, rhs, reg, mask, vol_shape,
                            W0_real=W0_real, maxiter=maxiter, tol=tol,
                            unpack_fn=unpack_fn, **extra_kwargs)
        return solver_fn

    def make_soft_solver(soft_lam):
        def solver_fn(lhs, rhs, reg, mask, vol_shape, W0_real=None,
                     maxiter=20, tol=1e-4, unpack_fn=None):
            return pv.solve_soft_penalty(lhs, rhs, reg, mask, vol_shape,
                                         soft_lam=soft_lam, W0_real=W0_real,
                                         maxiter=maxiter, tol=tol,
                                         unpack_fn=unpack_fn)
        return solver_fn

    configs = [
        # (label, mask_array, use_pcg, solver_fn, pcg_maxiter, gridding)
        ("mask_proj", mask_soft, False, None, 20, False),
        ("mask_proj+grid", mask_soft, False, None, 20, True),
        ("pcg_baseline", mask_bin, True, None, 20, False),
        ("pcg_baseline+grid", mask_bin, True, None, 20, True),
        ("baseline_circulant_50it", mask_bin, False,
         make_variant_solver(pv.solve_baseline_circulant), 50, True),
        ("no_precond_50it", mask_bin, False,
         make_variant_solver(pv.solve_no_precond), 50, True),
        ("reduced_circulant_50it", mask_bin, False,
         make_variant_solver(pv.solve_reduced_circulant), 50, True),
        ("reduced_block_jacobi_50it", mask_bin, False,
         make_variant_solver(pv.solve_reduced_jacobi, block=True), 50, True),
        ("soft_penalty_100", mask_bin, False,
         make_soft_solver(100.0), 50, True),
        ("soft_penalty_1000", mask_bin, False,
         make_soft_solver(1000.0), 50, True),
    ]

    all_results = []
    for label, mask_arr, use_pcg, solver_fn, pcg_max, gridding in configs:
        logger.info("\n=== Full EM: %s ===", label)
        try:
            t0 = time.time()
            U, S, W, ez, sm, idata = ppca.EM(
                cryos, gt_mean, W_init.copy(), W_prior,
                U_gt=U_gt, S_gt=s_gt**2,
                EM_iter=NITER_FULL,
                use_whitening=False, sparse_PCA=False,
                disc_type_mean="cubic", disc_type="linear_interp",
                return_iteration_data=True,
                use_pcg_mean=use_pcg,
                volume_mask=mask_arr,
                pcg_maxiter=pcg_max,
                use_gridding_correction=gridding,
                mstep_solver_fn=solver_fn,
            )
            dt = time.time() - t0
            _, rv, _ = metrics.get_all_variance_scores(U, U_gt, s_gt**2)

            result = {
                "label": label,
                "relvar": float(rv[-1]),
                "relvar_per_pc": [float(x) for x in rv],
                "time": dt,
                "idata": [
                    {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else ([float(x) for x in v] if isinstance(v, np.ndarray)
                           else v))
                     for k, v in d.items()}
                    for d in idata
                ],
            }
            all_results.append(result)
            logger.info("  RelVar=%.4f time=%.0fs", rv[-1], dt)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            traceback.print_exc()
            all_results.append({"label": label, "error": str(e), "relvar": 0.0})
        gc.collect()

    return all_results


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    logger.info("Loading dataset: %d³ q=%d NI=%d", GS, NPC, NI)
    cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs = load_dataset()

    # Part A: Single M-step
    mstep_results = run_single_mstep_comparison(
        cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_bin, vs
    )

    # Save Part A
    with open(os.path.join(OUT_DIR, "mstep_results.json"), "w") as f:
        json.dump(mstep_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "mstep_results.pkl"), "wb") as f:
        pickle.dump(mstep_results, f)

    # Print Part A summary
    print("\n" + "=" * 90)
    print("PART A: Single M-step Solver Comparison")
    print("=" * 90)
    print(f"{'Solver':<30} {'RelVar':>8} {'Iters':>6} {'Time(s)':>8} {'Residual':>10}")
    print("-" * 90)
    for name, r in sorted(mstep_results.items(), key=lambda x: -x[1].get("relvar", 0)):
        res = r.get("residuals", [])
        final_res = res[-1] if res else float("nan")
        print(f"{name:<30} {r.get('relvar', 0):8.4f} {r.get('n_iters', 0):6d} "
              f"{r.get('wall_time', 0):8.1f} {final_res:10.2e}")

    # Part B: Full EM
    em_results = run_full_em_comparison(
        cryos, gt_mean, W_init, W_prior, U_gt, s_gt, mov_soft, mov_bin, vs
    )

    # Save Part B
    with open(os.path.join(OUT_DIR, "em_results.json"), "w") as f:
        json.dump(em_results, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "em_results.pkl"), "wb") as f:
        pickle.dump(em_results, f)

    # Print Part B summary
    print("\n" + "=" * 90)
    print("PART B: Full EM Comparison (%d iterations)" % NITER_FULL)
    print("=" * 90)
    print(f"{'Method':<35} {'RelVar':>8} {'Time(s)':>8}")
    print("-" * 60)
    for r in sorted(em_results, key=lambda x: -x.get("relvar", 0)):
        print(f"{r['label']:<35} {r.get('relvar', 0):8.4f} {r.get('time', 0):8.0f}")

    # Generate convergence plot (Part A)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Residual convergence
        for name, r in mstep_results.items():
            res = r.get("residuals", [])
            if len(res) > 0:
                ax1.semilogy(range(1, len(res) + 1), res, label=name, linewidth=1.5)
        ax1.set_xlabel("CG iteration")
        ax1.set_ylabel("Relative residual")
        ax1.set_title(f"M-step convergence ({GS}³, q={NPC})")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Per-EM-iter RelVar convergence
        for r in em_results:
            if "idata" in r and isinstance(r["idata"], list):
                rvs = [d.get("Rel_Var_Explained", 0) for d in r["idata"]
                       if isinstance(d, dict)]
                if rvs:
                    ax2.plot(range(1, len(rvs) + 1), rvs,
                            label=r["label"], linewidth=1.5)
        ax2.set_xlabel("EM iteration")
        ax2.set_ylabel("RelVar")
        ax2.set_title(f"EM convergence ({GS}³, q={NPC}, {NITER_FULL} iters)")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "convergence.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"\nAll results saved to: {OUT_DIR}")
    print("Done!")
