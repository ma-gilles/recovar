#!/usr/bin/env python
"""Plot per-iter convergence diagnostics from a PPCA-refine cell's
``iter_log.pkl``.

Usage:
    plot_ppca_convergence.py <cell_dir> [--out <path>]

``cell_dir`` is a directory like
``/scratch/.../ppca_refine_eval_ribosembly_v3_mature/ribosembly__dense__zdim4__iters15``
containing ``iter_log.pkl``. The script loads each per-iter info dict
and renders six panels:

  1. log_evidence_total
  2. pmax_mean (and convergence threshold if present)
  3. noise_var_mean
  4. mean_prior_mean
  5. mu_delta_rms / W_delta_rms (state-change diagnostics)
  6. healpix_order / current_size schedule progression

Output is a single PNG at ``<cell_dir>/convergence.png`` or the path
given by ``--out``. The script is read-only and produces no other side
effects, so it can be run mid-job (incremental ``iter_log.pkl`` is
written by the eval callback after each iter).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_iter_log(cell_dir: Path) -> list[dict]:
    p = cell_dir / "iter_log.pkl"
    if not p.exists():
        raise FileNotFoundError(f"no iter_log.pkl in {cell_dir}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def _safe(rec: dict, key: str, default=np.nan) -> float:
    v = rec.get(key, default)
    if v is None:
        return default
    return float(v)


def _coerce_log_format(log: list[dict]) -> list[dict]:
    """Normalize older simple-pipeline iter_log records to the mature
    schema. The simple loop emits ``iteration_log_evidence: [le0, le1]``
    (per halfset list) plus ``iteration_pmax_mean: [pm0, pm1]`` etc.;
    the mature loop emits flat scalars. This shim averages or sums the
    per-halfset values to a scalar so the plotter can read both."""
    out = []
    for rec in log:
        rec2 = dict(rec)
        if "log_evidence_total" not in rec2 and "iteration_log_evidence" in rec2:
            le = rec2["iteration_log_evidence"]
            rec2["log_evidence_total"] = float(le[0]) + float(le[1]) if isinstance(le, list) else float(le)
        if "pmax_mean" not in rec2 and "iteration_pmax_mean" in rec2:
            pm = rec2["iteration_pmax_mean"]
            rec2["pmax_mean"] = 0.5 * (float(pm[0]) + float(pm[1])) if isinstance(pm, list) else float(pm)
        if "n_significant_mean" not in rec2 and "iteration_n_significant_mean" in rec2:
            ns = rec2["iteration_n_significant_mean"]
            rec2["n_significant_mean"] = 0.5 * (float(ns[0]) + float(ns[1])) if isinstance(ns, list) else float(ns)
        out.append(rec2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("cell_dir", type=Path, help="Directory containing iter_log.pkl")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    log = _load_iter_log(args.cell_dir)
    if not log:
        print(f"empty iter_log.pkl in {args.cell_dir}")
        return
    log = _coerce_log_format(log)

    iters = np.asarray([_safe(r, "iteration", i) for i, r in enumerate(log)])
    le = np.asarray([_safe(r, "log_evidence_total") for r in log])
    pm = np.asarray([_safe(r, "pmax_mean") for r in log])
    nv = np.asarray([_safe(r, "noise_var_mean") for r in log])
    mp = np.asarray([_safe(r, "mean_prior_mean") for r in log])
    mu_d = np.asarray([_safe(r, "mu_delta_rms") for r in log])
    W_d = np.asarray([_safe(r, "W_delta_rms") for r in log])
    mu_rms = np.asarray([_safe(r, "mu_rms") for r in log])
    W_frob = np.asarray([_safe(r, "W_frob") for r in log])
    hp_order = np.asarray([_safe(r, "healpix_order") for r in log])
    cur_size = np.asarray([_safe(r, "current_size") for r in log])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"PPCA convergence — {args.cell_dir.name}", fontsize=12)

    axes[0, 0].plot(iters, le, "o-")
    axes[0, 0].set_title("log_evidence_total")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, pm, "o-", color="C1")
    axes[0, 1].axhline(0.85, color="gray", ls="--", lw=0.8, label="conv threshold")
    axes[0, 1].set_title("pmax_mean")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, nv, "o-", color="C2")
    axes[1, 0].set_title("noise_var_mean (D5)")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, mp, "o-", color="C3")
    axes[1, 1].set_title("mean_prior_mean (D6)")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.semilogy(iters, np.maximum(mu_d, 1e-12), "o-", label="μ ΔRMS", color="C4")
    ax.semilogy(iters, np.maximum(W_d, 1e-12), "s-", label="W ΔRMS", color="C5")
    ax.set_title("state-change RMS per iter")
    ax.set_xlabel("iteration")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[2, 1]
    ax2 = ax.twinx()
    ln1 = ax.plot(iters, hp_order, "o-", label="healpix_order", color="C6")
    ln2 = ax2.plot(iters, cur_size, "s--", label="current_size", color="C7")
    ax.set_title("schedule progression")
    ax.set_xlabel("iteration")
    ax.set_ylabel("healpix_order")
    ax2.set_ylabel("current_size")
    lines = ln1 + ln2
    ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out if args.out is not None else args.cell_dir / "convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}  ({len(log)} iters)")
    # Print key summary numbers.
    print(f"  log_evidence_total: {le[0]:.3e} → {le[-1]:.3e}")
    print(f"  pmax_mean:          {pm[0]:.4f} → {pm[-1]:.4f}")
    print(f"  noise_var_mean:     {nv[0]:.3e} → {nv[-1]:.3e}")
    print(f"  μ ΔRMS final iter:  {mu_d[-1]:.3e}")
    print(f"  W ΔRMS final iter:  {W_d[-1]:.3e}")


if __name__ == "__main__":
    main()
