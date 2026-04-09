"""CLI entry point for PPCA refit postprocessing.

Usage::

    python -m recovar.ppca.refit_postprocess \\
        --ppca-result-dir /path/to/ppca/result \\
        --method refit_b \\
        --output-dir /path/to/output \\
        [--n-iters 20] [--zdim 10] [--batch-size 128]

Methods:
    refit_b              Algorithm 1: fixed-span B refit via EM
    temperature_scalar   Algorithm 5: scalar temperature diagnostic
    temperature_diag     Algorithm 5: per-component diagonal temperature
    stiefel_ub           Algorithm 2: full alternating U/B on Stiefel manifold
    whitening_manifold_ub Algorithm 3: whitening-manifold + explicit B
    coord_reg_grid       Algorithm 4A: grid-coordinate regularization
    coord_reg_physical   Algorithm 4B: physical-coordinate regularization
"""

import argparse
import logging
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

ALL_METHODS = [
    "refit_b",
    "temperature_scalar",
    "temperature_diag",
    "stiefel_ub",
    "whitening_manifold_ub",
    "coord_reg_grid",
    "coord_reg_physical",
    "iterative_ppca_projcov",
    "iterative_ppca_refitb",
    "iterative_ppca_refitb_reg",
    "dmetric_em_noreg",
    "dmetric_em_reg",
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-process PPCA results by refitting the latent covariance B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ppca-result-dir", required=True,
        help="Path to existing PPCA pipeline result directory.",
    )
    parser.add_argument(
        "--method", required=True,
        choices=ALL_METHODS,
        help="Postprocessing method to run.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path for the output result directory.",
    )
    parser.add_argument("--zdim", type=int, default=None, help="Number of PCs to use (default: all saved).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for image processing.")
    parser.add_argument("--n-iters", type=int, default=20, help="Number of EM iterations.")
    parser.add_argument("--n-outer", type=int, default=10, help="Outer iterations (iterative methods).")
    parser.add_argument("--n-inner-u", type=int, default=3, help="U-step retraction steps per outer iter.")
    parser.add_argument("--lambda-u", type=float, default=0.0, help="Regularization weight for U smoothness.")
    parser.add_argument("--n-grid", type=int, default=200, help="Grid points for temperature search.")
    parser.add_argument("--rho", type=float, default=0.01, help="Regularization for diagonal temperature.")
    parser.add_argument("--maxiter", type=int, default=100, help="Max optimizer iterations (temperature_diag).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args(argv)

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Import after argparse so --help is fast
    from recovar.output.output import PipelineOutput

    logger.info("Loading PipelineOutput from: %s", args.ppca_result_dir)
    po = PipelineOutput(args.ppca_result_dir)
    logger.info(
        "  volume_shape=%s, voxel_size=%.3f, n_eigenvalues=%d",
        po.params["volume_shape"],
        po.params.get("voxel_size", -1),
        len(np.asarray(po.params.get("s", []))),
    )

    t0 = time.time()

    if args.method == "refit_b":
        from recovar.ppca.ppca_refit import run_refit_b
        result = run_refit_b(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_iters=args.n_iters,
        )
        logger.info("Refit B eigenvalues: %s", np.array2string(result["eigenvalues"], precision=4))
        logger.info("NLL history: %s", [f"{x:.4e}" for x in result["nll_history"]])

    elif args.method == "temperature_scalar":
        from recovar.ppca.ppca_refit import run_temperature_scalar
        result = run_temperature_scalar(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_grid=args.n_grid,
        )
        logger.info("Optimal tau: %.4f", result["tau_opt"])
        logger.info("Scaled eigenvalues: %s", np.array2string(result["eigenvalues"], precision=4))

    elif args.method == "temperature_diag":
        from recovar.ppca.ppca_refit import run_temperature_diagonal
        result = run_temperature_diagonal(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            rho=args.rho,
            maxiter=args.maxiter,
        )
        logger.info("Optimal D: %s", np.array2string(result["D_opt"], precision=4))
        logger.info("Scaled eigenvalues: %s", np.array2string(result["eigenvalues"], precision=4))

    elif args.method == "stiefel_ub":
        from recovar.ppca.ppca_refit_iterative import run_stiefel_ub
        result = run_stiefel_ub(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_outer=args.n_outer,
            n_inner_u=args.n_inner_u,
            lambda_U=args.lambda_u,
        )
        logger.info("Stiefel U/B eigenvalues: %s", np.array2string(result["eigenvalues"], precision=4))

    elif args.method == "whitening_manifold_ub":
        from recovar.ppca.ppca_refit_iterative import run_whitening_manifold_ub
        result = run_whitening_manifold_ub(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_iters=args.n_outer,
            lambda_U=args.lambda_u,
        )
        logger.info("Whitening manifold eigenvalues: %s", np.array2string(result["eigenvalues"], precision=4))

    elif args.method in ("coord_reg_grid", "coord_reg_physical"):
        from recovar.ppca.ppca_refit_coord_reg import run_coord_reg_experiment
        reg_mode = "grid" if args.method == "coord_reg_grid" else "physical"
        result = run_coord_reg_experiment(
            args.ppca_result_dir, args.output_dir,
            reg_mode=reg_mode,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_outer=args.n_outer,
            n_inner_U=args.n_inner_u,
            lambda_U=args.lambda_u,
        )
        eigenvalues = result[1] if isinstance(result, tuple) else result.get("eigenvalues", np.array([]))
        logger.info("Coord reg (%s) done", reg_mode)

    elif args.method == "dmetric_em_noreg":
        from recovar.ppca.ppca_dmetric_em import run_dmetric_em_noreg
        result = run_dmetric_em_noreg(
            po, args.output_dir,
            zdim=args.zdim, batch_size=args.batch_size, n_iters=args.n_iters,
        )
        logger.info("D-metric EM (noreg) eigenvalues: %s",
                    np.array2string(result["eigenvalues"][:5], precision=4))

    elif args.method == "dmetric_em_reg":
        from recovar.ppca.ppca_dmetric_em import run_dmetric_em_reg
        result = run_dmetric_em_reg(
            po, args.output_dir,
            zdim=args.zdim, batch_size=args.batch_size, n_iters=args.n_iters,
            kappa=args.rho if args.rho != 0.01 else 1.0,
        )
        logger.info("D-metric EM (reg) eigenvalues: %s",
                    np.array2string(result["eigenvalues"][:5], precision=4))

    elif args.method in ("iterative_ppca_refitb", "iterative_ppca_refitb_reg"):
        from recovar.ppca.ppca_iterative_refitb import run_iterative_ppca_refitb
        kappa_value = 100.0 if args.method == "iterative_ppca_refitb_reg" else 0.0
        result = run_iterative_ppca_refitb(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_iters=30,
            refitb_every=1,
            refitb_start=5,
            refitb_inner_iters=3,
            kappa=kappa_value,
        )
        logger.info("Iterative PPCA+RefitB (kappa=%.1f) eigenvalues: %s",
                    kappa_value, np.array2string(result["eigenvalues"][:5], precision=4))

    elif args.method == "iterative_ppca_projcov":
        from recovar.ppca.ppca_iterative_projcov import run_iterative_ppca_projcov
        # 5 warmstart EM iters, then 25 iters of EM+projcov (30 total)
        result = run_iterative_ppca_projcov(
            po, args.output_dir,
            zdim=args.zdim,
            batch_size=args.batch_size,
            n_iters=30,
            projcov_every=1,
            projcov_start=5,
        )
        logger.info("Iterative PPCA+ProjCov eigenvalues: %s", np.array2string(result["eigenvalues"][:5], precision=4))

    elapsed = time.time() - t0
    logger.info("Done in %.1f s. Output at: %s", elapsed, args.output_dir)


if __name__ == "__main__":
    main()
