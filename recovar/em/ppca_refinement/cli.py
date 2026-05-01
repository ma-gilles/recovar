"""CLI for ``recovar ppca-refine``.

Currently exposes only ``--pose-mode fixed`` (M3). ``--pose-mode dense``
and ``--pose-mode local`` will be wired up at M5 / M7 respectively.

The dispatcher in ``recovar.command_line`` invokes this via
``recovar/commands/ppca_refine.py``, which simply imports and calls
:func:`main` here.
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recovar ppca_refine",
        description=(
            "Pose-marginalized PPCA refinement. Starts from an existing "
            "consensus and jointly refines μ, W, pose posterior (M7+), "
            "noise/contrast, and the loading prior. See "
            "docs/math/ppca_refine_plan_2026_05_01.md."
        ),
    )
    parser.add_argument("particles", help="Path to particles .star file (or pickle dataset).")
    parser.add_argument("--out", required=True, help="Output directory.")

    # Initial state.
    parser.add_argument("--init-mean", required=True, help="Initial mean MRC.")
    parser.add_argument("--init-poses", default=None, help="Initial poses .star (fixed mode requires this).")
    parser.add_argument("--init-W", default=None, help="Optional initial W .npz.")
    parser.add_argument("--init-state", default=None, help="Optional state.pkl restart (M9).")

    # Mode and engine.
    parser.add_argument(
        "--pose-mode",
        choices=("fixed", "dense", "local"),
        default="fixed",
        help="fixed (M3, default), dense (M5), local (M7).",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "dense", "sparse"),
        default="auto",
        help="Pose enumeration engine. Default 'auto' picks dense for "
        "--pose-mode dense and sparse for --pose-mode local.",
    )

    # PPCA hyperparameters.
    parser.add_argument("--zdim", type=int, required=True, help="PPCA dimension q.")
    parser.add_argument("--em-iters", type=int, default=20, help="Number of EM iterations.")
    parser.add_argument("--pcg-maxiter", type=int, default=20, help="Max CG iterations in M-step.")

    # Prior.
    parser.add_argument(
        "--pc-prior",
        choices=("hybrid_shell", "supplied"),
        default="hybrid_shell",
        help="Loading prior mode. 'hybrid_shell' uses "
        "estimate_hybrid_shell_prior_from_data; 'supplied' expects "
        "--init-W-prior.",
    )
    parser.add_argument(
        "--init-W-prior", default=None, help="Optional supplied W_prior .npz (for --pc-prior supplied)."
    )
    parser.add_argument(
        "--allow-prior-recompute-once", action="store_true", help="Allow one mid-run W_prior recompute (default off)."
    )

    # Contrast.
    parser.add_argument(
        "--contrast",
        choices=("none", "profile", "marginalize"),
        default="none",
        help="Contrast handling. 'none' until M8 lands.",
    )

    # Pose schedule (M7).
    parser.add_argument(
        "--reuse-kclass-pose-schedule",
        action="store_true",
        help="Import angular/shift schedule from k-class refinement (M7+).",
    )

    # Resolution.
    parser.add_argument(
        "--max-resolution", type=float, default=None, help="Cap on resolution in Å (currently advisory; wired in M5+)."
    )

    # Numerics.
    parser.add_argument(
        "--use-float64-scoring", action="store_true", help="Use float64 in the score path (mirrors high-res EM)."
    )

    # Production CLI inputs (Phase B).
    parser.add_argument(
        "--input-bundle",
        default=None,
        help=(
            "Path to a pickle bundle dict with keys: 'cryo' "
            "(CryoEMDataset), 'mu_init' (D,D,D real), 'W_init' optional "
            "(q,D,D,D real), 'mask' (D,D,D real), 'halfset_indices' "
            "tuple, 'rotation_grid' (R,3,3), 'translation_grid' (T,2). "
            "When provided with --pose-mode {dense,local}, runs the "
            "production EM loop end-to-end and writes outputs to --out."
        ),
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=32,
        help="Image batch size in the production driver.",
    )
    parser.add_argument(
        "--rotation-block-size",
        type=int,
        default=64,
        help="Rotation block size in the production driver.",
    )
    parser.add_argument(
        "--halfset-combine",
        choices=("mean", "low_resol_join"),
        default="mean",
        help="Halfset combine method (low_resol_join uses 40 Å Fourier join).",
    )
    parser.add_argument(
        "--low-resol-join-angstrom",
        type=float,
        default=40.0,
        help="Low-resolution join radius in Å (only when --halfset-combine=low_resol_join).",
    )
    parser.add_argument(
        "--prior-recompute-iter",
        type=int,
        default=None,
        help="Iteration after which to recompute mean_prior (default: never).",
    )

    # Sparse / local-pose options (--pose-mode local).
    parser.add_argument(
        "--n-local-rotations",
        type=int,
        default=8,
        help="Per-image rotation neighborhood size for --pose-mode local.",
    )
    parser.add_argument(
        "--local-sigma-rad",
        type=float,
        default=0.05,
        help="Stddev of local rotation perturbations (radians) for --pose-mode local.",
    )

    return parser


def main(argv: list[str] | None = None, *, _bundle_override=None) -> int:
    """Entry point. ``_bundle_override`` is a test hook that bypasses
    pickle loading by passing the bundle dict directly — used by
    integration tests where the synthetic ``cryo`` is unpicklable
    (it references a dynamically-loaded helper module)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ``--input-bundle`` mode: pickle containing the loaded objects
    # (cryo, mu_init, W_init, mask, halfset_indices). This lets the CLI
    # decouple from particles.star loading (recovar.commands.pipeline
    # owns the heavy I/O) while still giving a complete recovar-style
    # production EM loop.
    if (args.input_bundle is not None or _bundle_override is not None) and args.pose_mode in ("dense", "local"):
        return _run_production_em_loop(args, bundle_override=_bundle_override)

    # Python APIs available:
    #   --pose-mode fixed  → recovar.em.ppca_refinement.iterations.run_fixed_pose_ppca_refine
    #   --pose-mode dense  → run_pose_marginal_ppca_refine(..., block_provider=...)
    #   --pose-mode local  → run_pose_marginal_ppca_refine(..., sparse_block_provider=...)
    # Full CLI pipeline glue (particles.star → CryoEMDataset → block provider →
    # backprojector → output) lands at M10 alongside dataset wiring.
    py_api = {
        "fixed": "recovar.em.ppca_refinement.iterations.run_fixed_pose_ppca_refine",
        "dense": "recovar.em.ppca_refinement.iterations.run_pose_marginal_ppca_refine "
        "(block_provider= for dense engine)",
        "local": "recovar.em.ppca_refinement.iterations.run_pose_marginal_ppca_refine "
        "(sparse_block_provider= for local-pose Mode B)",
    }[args.pose_mode]
    print(
        f"Python API for --pose-mode {args.pose_mode}: {py_api}. "
        "Full CLI pipeline glue (particles → dataset → refinement → output) "
        "lands at M10.",
        file=sys.stderr,
    )
    if args.reuse_kclass_pose_schedule and args.pose_mode != "local":
        print("--reuse-kclass-pose-schedule only applies to --pose-mode local.", file=sys.stderr)
    return 0


def _run_production_em_loop(args, *, bundle_override=None) -> int:
    """Phase B production CLI entry: load bundle → run EM loop → write outputs."""
    import pickle
    from pathlib import Path

    import jax.numpy as jnp
    import mrcfile
    import numpy as np

    import recovar.core.fourier_transform_utils as ftu
    from recovar.em.ppca_refinement.halfset_combine import make_halfset_combiner
    from recovar.em.ppca_refinement.iterations import IterationOpts
    from recovar.em.ppca_refinement.postprocess import (
        finalize_ppca_state,
        save_state,
    )
    from recovar.em.ppca_refinement.production_driver import (
        run_pose_marginal_iteration_dense_production,
        run_pose_marginal_iteration_sparse_production,
    )
    from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
    from recovar.ppca import PCPriorConfig

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bundle_override is not None:
        bundle = bundle_override
    else:
        with open(args.input_bundle, "rb") as fh:
            bundle = pickle.load(fh)

    cryo = bundle["cryo"]
    mu_init = np.asarray(bundle["mu_init"], dtype=np.float32)
    if mu_init.ndim == 1:
        mu_init = mu_init.reshape(cryo.grid_size, cryo.grid_size, cryo.grid_size)
    vol_shape = mu_init.shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    q = args.zdim

    if "W_init" in bundle and bundle["W_init"] is not None:
        W_init = np.asarray(bundle["W_init"], dtype=np.float32)
    else:
        rng = np.random.default_rng(0)
        W_init = (rng.standard_normal((q,) + vol_shape) * 1e-3).astype(np.float32)

    mask = np.asarray(bundle["mask"], dtype=np.float32)
    halfset_indices = bundle["halfset_indices"]
    rotation_grid = np.asarray(bundle["rotation_grid"], dtype=np.float32)
    translation_grid = np.asarray(bundle.get("translation_grid", np.zeros((1, 2))), dtype=np.float32)

    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu_init), jnp.asarray(mu_init)),
        W_half=(jnp.asarray(W_init), jnp.asarray(W_init)),
        mu_score=jnp.asarray(mu_init),
        W_score=jnp.asarray(W_init),
        W_prior=jnp.full((half_vol, q), 1.0, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1.0, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates={},
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
    )

    halfset_combiner = (
        make_halfset_combiner(
            method=args.halfset_combine,
            voxel_size=cryo.voxel_size if args.halfset_combine == "low_resol_join" else None,
            low_resol_join_halves_angstrom=args.low_resol_join_angstrom,
        )
        if args.halfset_combine == "low_resol_join"
        else make_halfset_combiner(method="mean")
    )

    pc_prior_cfg = PCPriorConfig(
        recompute_once_after_iter=args.prior_recompute_iter,
    )
    opts = IterationOpts(
        EM_iter=args.em_iters,
        pcg_maxiter=args.pcg_maxiter,
        pc_prior_config=pc_prior_cfg,
    )

    iter_log = []
    for it in range(args.em_iters):
        if args.pose_mode == "local":
            state, diag = run_pose_marginal_iteration_sparse_production(
                state,
                cryo,
                halfset_indices=halfset_indices,
                mask=jnp.asarray(mask),
                n_local_rotations=args.n_local_rotations,
                local_sigma_rad=args.local_sigma_rad,
                translation_grid=translation_grid,
                image_batch_size=args.image_batch_size,
                halfset_combiner=halfset_combiner,
                iteration_index=it,
                opts=opts,
            )
        else:
            state, diag = run_pose_marginal_iteration_dense_production(
                state,
                cryo,
                rotation_grid=rotation_grid,
                translation_grid=translation_grid,
                halfset_indices=halfset_indices,
                mask=jnp.asarray(mask),
                image_batch_size=args.image_batch_size,
                rotation_block_size=args.rotation_block_size,
                halfset_combiner=halfset_combiner,
                iteration_index=it,
                opts=opts,
            )
        iter_log.append({"iteration": it, **diag})
        # Per-iter MRCs.
        iter_dir = out_dir / f"iter_{it:03d}"
        iter_dir.mkdir(exist_ok=True)
        with mrcfile.new(str(iter_dir / "mu_score.mrc"), overwrite=True) as mrc:
            mrc.set_data(np.asarray(state.mu_score, dtype=np.float32))
            mrc.voxel_size = cryo.voxel_size
        for k in range(q):
            with mrcfile.new(str(iter_dir / f"W_{k:02d}_score.mrc"), overwrite=True) as mrc:
                mrc.set_data(np.asarray(state.W_score[k], dtype=np.float32))
                mrc.voxel_size = cryo.voxel_size

    # Final postprocessing.
    U, S, _W_half = finalize_ppca_state(state, volume_shape=vol_shape)
    U_np = np.asarray(U, dtype=np.float32)
    S_np = np.asarray(S, dtype=np.float32)
    for k in range(q):
        with mrcfile.new(str(out_dir / f"U_{k:02d}.mrc"), overwrite=True) as mrc:
            mrc.set_data(U_np[k])
            mrc.voxel_size = cryo.voxel_size
    np.save(out_dir / "S.npy", S_np)

    save_state(state, out_dir / "state.pkl")
    with (out_dir / "iter_log.pkl").open("wb") as fh:
        pickle.dump(iter_log, fh)

    print(f"recovar ppca_refine: wrote {q}+1 maps + state.pkl to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
