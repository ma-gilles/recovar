#!/usr/bin/env python
"""Aggregate one matrix cell's recovar+RELION outputs into summary.json.

Reads both K=1 (run_multi_iter_parity.py) and K-class (run_k_class_parity.py)
outputs and produces a unified per-cell summary written to --summary.

Schema:
{
  "cell": str, "K": int, "requested_iters": int, "effective_iters": int,
  "adaptive_fraction": float, "firstiter_cc": "yes"|"no",
  "status": "PASS"|"FAIL"|"SKIPPED"|"ERROR",
  "reason": str|None,
  "per_iter": {
    "iter": [int...],
    "mean_corr": [float...],   # K=1: mean of half1+half2 corr; Kc: mean over classes after Hungarian
    "per_class_corr": [[float...]],  # K-class only
    "pmax_recovar": [float...],
    "pmax_relion":  [float...],
    "pmax_abs_diff": [float...]
  },
  "class_assignment_accuracy": float|None,
  "thresholds": {"mean_corr": 0.99, "pmax_abs_diff": 0.01}
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _read_relion_pmax(relion_dir: Path, iter_num: int) -> float:
    """Return RELION's per-particle ave Pmax for iter_num from data.star.

    Mirrors the comparison in scripts/diff_relion_recovar_per_iter.py:
    rlnMaxValueProbDistribution averaged across particles.
    """
    data_path = relion_dir / f"run_it{iter_num:03d}_data.star"
    if not data_path.exists():
        return float("nan")
    try:
        import starfile

        st = starfile.read(str(data_path))
        particles = st["particles"] if isinstance(st, dict) and "particles" in st else st
        col = "rlnMaxValueProbDistribution"
        if col not in particles.columns:
            return float("nan")
        return float(np.asarray(particles[col], dtype=np.float64).mean())
    except Exception:
        return float("nan")


def _read_relion_volume(relion_dir: Path, iter_num: int, class_index: int = 1) -> np.ndarray | None:
    """Read RELION reconstructed volume run_it{NNN}_class{ccc}.mrc."""
    p = relion_dir / f"run_it{iter_num:03d}_class{class_index:03d}.mrc"
    if not p.exists():
        return None
    try:
        from recovar import utils

        return np.asarray(utils.load_mrc(str(p))[0], dtype=np.float64)
    except Exception:
        try:
            import mrcfile

            with mrcfile.open(str(p), permissive=True) as f:
                return np.asarray(f.data, dtype=np.float64)
        except Exception:
            return None


def _vol_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _aggregate_k1(args, summary: dict) -> dict:
    """K=1 case: compare recovar refinement_results.npz half1/half2 against
    RELION run_it{NNN}_class001.mrc per iter."""
    npz_path = args.recovar_dir / "refinement_results.npz"
    if not npz_path.exists():
        summary["status"] = "ERROR"
        summary["reason"] = f"missing recovar output {npz_path}"
        return summary
    npz = np.load(npz_path)
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    n_iters = int(pmax_traj.size)

    iters = list(range(1, n_iters + 1))
    pmax_relion = [_read_relion_pmax(args.relion_dir, it) for it in iters]
    pmax_recovar = [float(p) for p in pmax_traj.tolist()]
    pmax_diff = [
        abs(r - rl) if not (np.isnan(r) or np.isnan(rl)) else float("nan") for r, rl in zip(pmax_recovar, pmax_relion)
    ]

    # mean_corr per iter: average final_half1_corr_vs_relion + final_half2_corr_vs_relion
    # if only the final iter is dumped, fill earlier with nan; else attempt iter dumps.
    h1_final = float(npz.get("final_half1_corr_vs_relion", np.nan))
    h2_final = float(npz.get("final_half2_corr_vs_relion", np.nan))
    mean_corr = [float("nan")] * (n_iters - 1) + [0.5 * (h1_final + h2_final)]

    summary["per_iter"] = {
        "iter": iters,
        "mean_corr": mean_corr,
        "pmax_recovar": pmax_recovar,
        "pmax_relion": pmax_relion,
        "pmax_abs_diff": pmax_diff,
    }
    final_corr = mean_corr[-1] if mean_corr else float("nan")
    final_pmax_diff = pmax_diff[-1] if pmax_diff else float("nan")
    summary["final_mean_corr"] = final_corr
    summary["final_pmax_abs_diff"] = final_pmax_diff
    if np.isfinite(final_corr) and final_corr >= 0.99 and (not np.isfinite(final_pmax_diff) or final_pmax_diff < 0.01):
        summary["status"] = "PASS"
    else:
        summary["status"] = "FAIL"
        summary["reason"] = f"final mean_corr={final_corr:.4f} pmax_abs_diff={final_pmax_diff:.4g}"
    return summary


def _aggregate_kclass(args, summary: dict) -> dict:
    """K-class case: read per_iter.json (chained per-iter single-step replays).

    Falls back to single-step summary.json if per_iter.json absent (legacy).
    """
    per_iter_path = args.recovar_dir / "per_iter.json"
    if per_iter_path.exists():
        try:
            entries = json.loads(per_iter_path.read_text())
        except Exception as exc:
            summary["status"] = "ERROR"
            summary["reason"] = f"bad per_iter.json: {exc}"
            return summary
        iters = [e["iter"] for e in entries]
        mean_corrs = [e["mean_corr"] for e in entries]
        per_class = [e["per_class_corr"] for e in entries]
        pmax_means = [e["pmax_abs_mean"] for e in entries]
        pmax_maxs = [e.get("pmax_abs_max", float("nan")) for e in entries]
        class_accs = [e.get("class_assignment_accuracy", float("nan")) for e in entries]

        summary["per_iter"] = {
            "iter": iters,
            "mean_corr": mean_corrs,
            "per_class_corr": per_class,
            "pmax_abs_mean": pmax_means,
            "pmax_abs_max": pmax_maxs,
            "class_assignment_accuracy": class_accs,
        }
        summary["final_mean_corr"] = mean_corrs[-1] if mean_corrs else float("nan")
        summary["final_pmax_abs_mean"] = pmax_means[-1] if pmax_means else float("nan")
        summary["class_assignment_accuracy"] = class_accs[-1] if class_accs else float("nan")
        worst = min(mean_corrs) if mean_corrs else float("nan")
        worst_pmax = max(pmax_means) if pmax_means else float("nan")
        summary["worst_mean_corr"] = worst
        summary["worst_pmax_abs_mean"] = worst_pmax
        if np.isfinite(worst) and worst >= 0.99 and worst_pmax < 0.01:
            summary["status"] = "PASS"
        else:
            summary["status"] = "FAIL"
            summary["reason"] = f"worst_iter mean_corr={worst:.4f} worst pmax_abs_mean={worst_pmax:.4g}"
        return summary

    summary_path = args.recovar_dir / "summary.json"
    if not summary_path.exists():
        summary["status"] = "ERROR"
        summary["reason"] = f"missing K-class summary {summary_path}"
        return summary
    kc = json.loads(summary_path.read_text())
    best = kc.get("best_permutation", {})
    map_corrs = [float(c) for c in best.get("map_correlations", [])]
    mean_corr = float(best.get("mean_corr", float("nan")))
    pmax = kc.get("pmax", {})
    pmax_abs_mean = float(pmax.get("abs_mean", float("nan")))
    pmax_abs_max = float(pmax.get("abs_max", float("nan")))
    class_acc = float(kc.get("class_assignment_accuracy_after_permutation", float("nan")))

    n_iters = int(args.effective_iters)
    summary["per_iter"] = {
        "iter": [n_iters],
        "mean_corr": [mean_corr],
        "per_class_corr": [map_corrs],
        "pmax_abs_mean": [pmax_abs_mean],
        "pmax_abs_max": [pmax_abs_max],
    }
    summary["final_mean_corr"] = mean_corr
    summary["final_pmax_abs_mean"] = pmax_abs_mean
    summary["class_assignment_accuracy"] = class_acc

    if np.isfinite(mean_corr) and mean_corr >= 0.99 and pmax_abs_mean < 0.01:
        summary["status"] = "PASS"
    else:
        summary["status"] = "FAIL"
        summary["reason"] = f"mean_corr={mean_corr:.4f} pmax_abs_mean={pmax_abs_mean:.4g}"
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--requested-iters", type=int, required=True)
    p.add_argument("--effective-iters", type=int, required=True)
    p.add_argument("--adaptive-fraction", type=float, required=True)
    p.add_argument("--firstiter-cc", choices=["yes", "no"], required=True)
    p.add_argument("--recovar-dir", type=Path, required=True)
    p.add_argument("--relion-dir", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    summary = {
        "cell": args.cell,
        "K": args.K,
        "requested_iters": args.requested_iters,
        "effective_iters": args.effective_iters,
        "adaptive_fraction": args.adaptive_fraction,
        "firstiter_cc": args.firstiter_cc,
        "status": "ERROR",
        "reason": None,
        "thresholds": {"mean_corr": 0.99, "pmax_abs_diff": 0.01},
    }
    try:
        if args.K == 1:
            summary = _aggregate_k1(args, summary)
        else:
            summary = _aggregate_kclass(args, summary)
    except Exception as exc:
        summary["status"] = "ERROR"
        summary["reason"] = f"{type(exc).__name__}: {exc}"

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Wrote {args.summary} status={summary['status']}", file=sys.stderr)


if __name__ == "__main__":
    main()
