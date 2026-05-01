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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
