"""Run ONE experiment configuration and write JSON results.

Usage:
  pixi run python run_experiment.py \
      --dataset-dir /scratch/.../datasets/Ribosembly_g64_n20k_nl1 \
      --grid-size 64 --n-images 20000 \
      --method hard --target-rank 10 \
      --prior-mode radial_D --prior-alpha 1.0 \
      --init cold --n-iter 80 \
      --output results.json
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore", module="finufft")
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from sketch_lib import (
    SketchedIterationCfg,
    SketchedSolver,
    build_gt_factors,
    compute_radial_D2,
    run_iterations,
    run_iterations_backtracking,
    run_iterations_backtracking_dmetric,  # noqa: F401
)

from recovar.ppca.ppca_scale_sweep import _load_simulated_dataset, _with_trailing_separator
from recovar.ppca.sketched_normal import SketchedNormalOperator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--grid-size", type=int, required=True)
    ap.add_argument("--n-images", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--method", choices=["hard", "soft"], default="hard")
    ap.add_argument("--target-rank", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--prior-mode", choices=["none", "radial_D"], default="none")
    ap.add_argument(
        "--prior-norm",
        choices=["frob", "d"],
        default="frob",
        help="frob: standard SVT on X (nuclear in Frobenius metric). "
        "d: change-of-variables Y=DX, nuclear in D-metric (requires --prior-mode=radial_D and --step-rule=backtracking).",
    )
    ap.add_argument("--init", choices=["cold", "gt"], default="cold")
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--delta", type=float, default=1.0, help="step size when --step-rule=fixed")
    ap.add_argument(
        "--step-rule",
        choices=["fixed", "lipschitz", "backtracking"],
        default="fixed",
        help="fixed: use --delta. lipschitz: δ = delta-safety / L_hat. backtracking: Armijo on smooth f.",
    )
    ap.add_argument(
        "--delta-safety", type=float, default=0.9, help="safety factor for --step-rule=lipschitz (δ = safety/L)"
    )
    ap.add_argument("--lipschitz-power-iters", type=int, default=30)
    ap.add_argument("--bt-delta-init", type=float, default=0.1, help="initial δ for --step-rule=backtracking")
    ap.add_argument(
        "--bt-armijo-c", type=float, default=0.9, help="Armijo safety factor c (quadratic term scale 1/(2δc))"
    )
    ap.add_argument("--bt-shrink", type=float, default=0.5)
    ap.add_argument("--bt-grow", type=float, default=1.5)
    ap.add_argument("--bt-max-retries", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=15)
    ap.add_argument("--max-rank", type=int, default=30)
    ap.add_argument("--n-power", type=int, default=1)
    ap.add_argument(
        "--record-per-k",
        type=str,
        default="1,2,5,10",
        help="comma-separated list of k values for rv@k logging",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cryo, sim_info, gt, _ = _load_simulated_dataset(
        _with_trailing_separator(args.dataset_dir), args.grid_size, args.n_images, lazy=False
    )
    vs = cryo.volume_shape
    V_SIZE = int(np.prod(vs))
    n = int(cryo.n_images)
    print(f"[ds] grid={args.grid_size} n={n} V={V_SIZE}", flush=True)

    op = SketchedNormalOperator(cryo, gt.get_mean(), batch_size=args.batch_size, disc_type="linear_interp")
    gt_factors = build_gt_factors(gt, sim_info, vs, V_SIZE)
    LEFT_SCALE = V_SIZE / 2.0

    D2 = None
    shells_power = None
    if args.prior_mode == "radial_D":
        D2, shells_power, D2_radial_shells = compute_radial_D2(gt, vs)
        print(
            f"[prior] radial D built: shells={shells_power.shape}, "
            f"D2 range=[{D2.min():.3e}, {D2.max():.3e}], "
            f"power shells (first 5) {shells_power[:5]}",
            flush=True,
        )

    solver = SketchedSolver(op, vs, V_SIZE, n, LEFT_SCALE, D2_fourier=D2)

    if args.step_rule == "lipschitz":
        L_hat = solver.estimate_lipschitz(
            include_prior=(args.prior_mode == "radial_D"),
            n_power=args.lipschitz_power_iters,
            seed=args.seed,
        )
        delta_used = float(args.delta_safety / max(L_hat, 1e-30))
        print(f"[step-rule] lipschitz: L_hat={L_hat:.3e}  δ = {args.delta_safety}/L = {delta_used:.3e}", flush=True)
    elif args.step_rule == "backtracking":
        L_hat = None
        delta_used = float(args.bt_delta_init)
        print(
            f"[step-rule] backtracking: δ_init={delta_used:.3e}  c={args.bt_armijo_c}  "
            f"shrink={args.bt_shrink}  grow={args.bt_grow}  max_retries={args.bt_max_retries}",
            flush=True,
        )
    else:
        L_hat = None
        delta_used = float(args.delta)
        print(f"[step-rule] fixed: δ = {delta_used:.3e}", flush=True)

    cfg = SketchedIterationCfg(
        block_size=args.block_size,
        max_rank=args.max_rank,
        n_power=args.n_power,
        target_rank=args.target_rank,
        delta=delta_used,
        method=args.method,
        lam=args.lam,
        prior_mode=args.prior_mode,
        record_per_k=tuple(int(x) for x in args.record_per_k.split(",")),
    )

    if args.init == "cold":
        U0 = np.zeros((V_SIZE, 0), np.float32)
        s0 = np.zeros((0,), np.float32)
        V0 = np.zeros((n, 0), np.float32)
    else:
        k = args.target_rank
        U0 = gt_factors["U_gt_real"][:, :k].copy()
        s0 = gt_factors["s_gt"][:k].copy()
        V0 = gt_factors["V_gt_img"][:, :k].copy()

    if args.step_rule == "backtracking" and args.prior_norm == "d":
        if args.prior_mode != "radial_D":
            raise SystemExit("--prior-norm=d requires --prior-mode=radial_D")
        if D2 is None:
            raise SystemExit("--prior-norm=d requires D2 built from radial prior")
        print("[prior-norm] D-metric: nuclear in D-space (Y = D X)", flush=True)
        U, s, V, history, final = run_iterations_backtracking_dmetric(
            op,
            D2,
            vs,
            V_SIZE,
            n,
            LEFT_SCALE,
            cfg,
            U0,
            s0,
            V0,
            args.n_iter,
            gt_factors,
            seed=args.seed,
            log_every=5,
            logfn=lambda m: print(m, flush=True),
            delta_init=args.bt_delta_init,
            armijo_c=args.bt_armijo_c,
            shrink=args.bt_shrink,
            grow=args.bt_grow,
            max_retries=args.bt_max_retries,
        )
    elif args.step_rule == "backtracking":
        U, s, V, history, final = run_iterations_backtracking(
            solver,
            cfg,
            U0,
            s0,
            V0,
            args.n_iter,
            gt_factors,
            vs,
            V_SIZE,
            seed=args.seed,
            log_every=5,
            logfn=lambda m: print(m, flush=True),
            delta_init=args.bt_delta_init,
            armijo_c=args.bt_armijo_c,
            shrink=args.bt_shrink,
            grow=args.bt_grow,
            max_retries=args.bt_max_retries,
        )
    else:
        U, s, V, history, final = run_iterations(
            solver,
            cfg,
            U0,
            s0,
            V0,
            args.n_iter,
            gt_factors,
            vs,
            V_SIZE,
            seed=args.seed,
            log_every=5,
            logfn=lambda m: print(m, flush=True),
        )

    rv_gt_theory = {
        f"cumfrac@{k}": float(
            np.sum((gt_factors["s_gt"][:k].astype(np.float64)) ** 2)
            / np.sum((gt_factors["s_gt"].astype(np.float64)) ** 2)
        )
        for k in tuple(int(x) for x in args.record_per_k.split(","))
    }

    result = {
        "config": vars(args),
        "step_rule_used": args.step_rule,
        "delta_used": delta_used,
        "L_hat": L_hat,
        "final": final,
        "history": history,
        "rv_theory_from_s_gt": rv_gt_theory,
        "s_gt_top10": [float(x) for x in gt_factors["s_gt"][:10]],
        "elapsed_total_s": time.time() - t0,
        "V_SIZE": V_SIZE,
        "n_images": n,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.output}  total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
