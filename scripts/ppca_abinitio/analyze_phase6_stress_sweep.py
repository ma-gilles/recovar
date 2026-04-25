"""Aggregate Phase 6 stress-sweep results: per-axis pivot tables +
hypothesis-by-hypothesis verdict.

Usage:
    pixi run python scripts/ppca_abinitio/analyze_phase6_stress_sweep.py \\
        <sweep_root_dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_cell(path: Path) -> dict:
    j = json.load(path.open())
    cfg, met = j["config"], j["metrics"]
    return {
        "axis": path.stem.split("_", 1)[0],
        "cell_id": path.stem,
        "dataset": cfg["dataset"],
        "q": int(cfg["q"]),
        "sigma": float(cfg["sigma"]),
        "n_images": int(cfg["n_images"]),
        "mu_init": cfg["mu_init"] if "mu_init" in cfg else "perturbed",
        "u_init": cfg["u_init"],
        "anneal": cfg["anneal_schedule"],
        "seed": int(cfg["data_seed"]),
        "best_pe": met["best_pe"],
        "best_fre_truth": met["best_fre_truth"],
        "best_lm": met["best_lm"],
        "hun": met.get("clust_acc_hungarian", float("nan")),
        "ari": met.get("ari", float("nan")),
        "nmi": met.get("nmi", float("nan")),
    }


def reduce_seeds(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for k, members in groups.items():
        s = {"n_seeds": len(members)}
        for m in ("hun", "ari", "nmi", "best_pe", "best_fre_truth", "best_lm"):
            v = np.array([r[m] for r in members], dtype=np.float64)
            v = v[~np.isnan(v)]
            s[f"{m}_mean"] = float(v.mean()) if v.size else float("nan")
            s[f"{m}_std"] = float(v.std(ddof=0)) if v.size else float("nan")
        out[k] = s
    return out


def axis_a_snr(rows):
    """Print hun & PE vs σ per dataset×q."""
    print("\n=== AXIS A — SNR SWEEP (mean ± std over 2 seeds) ===")
    a = [r for r in rows if r["axis"] == "A"]
    by = reduce_seeds(a, lambda r: (r["dataset"], r["q"], r["sigma"]))
    rows_set = sorted({(k[0], k[1]) for k in by})
    sigmas = sorted({k[2] for k in by})
    for ds, q in rows_set:
        print(f"\n--- {ds} q={q} ---")
        print(f"  {'σ':>8s}  {'hun':>14s}  {'PE':>14s}  {'fre_truth':>14s}")
        for sig in sigmas:
            s = by.get((ds, q, sig))
            if s is None:
                continue
            print(
                f"  {sig:>8.3f}  "
                f"{s['hun_mean']:>6.3f}±{s['hun_std']:.3f}  "
                f"{s['best_pe_mean']:>6.3f}±{s['best_pe_std']:.3f}  "
                f"{s['best_fre_truth_mean']:>6.3f}±{s['best_fre_truth_std']:.3f}"
            )


def axis_b_cold_inits(rows):
    print("\n=== AXIS B — COLD INIT (μ=zero × U init) ===")
    b = [r for r in rows if r["axis"] == "B"]
    by = reduce_seeds(b, lambda r: (r["dataset"], r["q"], r["sigma"], r["u_init"]))
    rows_set = sorted({(k[0], k[1], k[2]) for k in by})
    for ds, q, sig in rows_set:
        print(f"\n--- {ds} q={q} σ={sig} (μ=zero) ---")
        print(f"  {'U init':<7s}  {'hun':>14s}  {'PE':>14s}")
        for u_init in ("svd", "random", "zero"):
            s = by.get((ds, q, sig, u_init))
            if s is None:
                continue
            print(
                f"  {u_init:<7s}  "
                f"{s['hun_mean']:>6.3f}±{s['hun_std']:.3f}  "
                f"{s['best_pe_mean']:>6.3f}±{s['best_pe_std']:.3f}"
            )


def axis_c_small_n(rows):
    print("\n=== AXIS C — SMALL-N (Ribo q=4, IgG-RL q=2; σ=0.01) ===")
    c = [r for r in rows if r["axis"] == "C"]
    by = reduce_seeds(c, lambda r: (r["dataset"], r["q"], r["n_images"]))
    rows_set = sorted({(k[0], k[1]) for k in by})
    for ds, q in rows_set:
        print(f"\n--- {ds} q={q} ---")
        print(f"  {'n_img':>6s}  {'hun':>14s}  {'PE':>14s}")
        for n_img in sorted({k[2] for k in by if k[0] == ds and k[1] == q}):
            s = by[(ds, q, n_img)]
            print(
                f"  {n_img:>6d}  "
                f"{s['hun_mean']:>6.3f}±{s['hun_std']:.3f}  "
                f"{s['best_pe_mean']:>6.3f}±{s['best_pe_std']:.3f}"
            )


def axis_d_tomotwin(rows):
    print("\n=== AXIS D — TOMOTWIN-100 (q × σ) ===")
    d = [r for r in rows if r["axis"] == "D"]
    by = reduce_seeds(d, lambda r: (r["q"], r["sigma"]))
    print(f"  {'q':>4s} {'σ':>6s}  {'hun':>14s}  {'PE':>14s}  {'best_lm':>14s}")
    for k in sorted(by):
        q, sig = k
        s = by[k]
        print(
            f"  {q:>4d} {sig:>6.2f}  "
            f"{s['hun_mean']:>6.3f}±{s['hun_std']:.3f}  "
            f"{s['best_pe_mean']:>6.3f}±{s['best_pe_std']:.3f}  "
            f"{s['best_lm_mean']:>10.0f}"
        )


def hypothesis_verdicts(rows):
    print("\n" + "=" * 70)
    print("HYPOTHESIS VERDICTS")
    print("=" * 70)

    def cell(axis, **filters):
        members = [r for r in rows if r["axis"] == axis]
        for k, v in filters.items():
            members = [r for r in members if r[k] == v]
        if not members:
            return None
        return {
            "hun": float(np.nanmean([r["hun"] for r in members])),
            "pe": float(np.nanmean([r["best_pe"] for r in members])),
            "n": len(members),
        }

    # H1 — v0 holds at σ ≤ 0.1, breaks at σ ≥ 0.3 on Ribosembly
    print("\nH1: v0 holds at σ ≤ 0.1, breaks at σ ≥ 0.3 on Ribosembly q=4")
    for sig in [0.01, 0.1, 0.3, 1.0]:
        c = cell("A", dataset="Ribosembly", q=4, sigma=sig)
        if c:
            print(f"   σ={sig}: hun={c['hun']:.3f} (n={c['n']})")

    # H2 — cold μ + cold U collapses
    print("\nH2: zero-μ + zero-U collapses (vs zero-μ + svd-U)")
    for ds_q in [("Ribosembly", 4), ("IgG-RL", 2)]:
        ds, q = ds_q
        c_zero = cell("B", dataset=ds, q=q, sigma=0.01, u_init="zero")
        c_svd = cell("B", dataset=ds, q=q, sigma=0.01, u_init="svd")
        if c_zero and c_svd:
            print(f"   {ds} q={q} σ=0.01: zero-U hun={c_zero['hun']:.3f}, svd-U hun={c_svd['hun']:.3f}")

    # H3 — Tomotwin q=4 ~ chance, q=16 better
    print("\nH3: Tomotwin-100 q=4 near chance, q=16 meaningfully better")
    for q in [4, 8, 16]:
        c = cell("D", q=q, sigma=0.01)
        if c:
            print(f"   Tomotwin q={q} σ=0.01: hun={c['hun']:.3f}")

    # H4 — Small-N breaks IgG-RL but not Ribosembly
    print("\nH4: small-N (n=64) hurts IgG-RL more than Ribosembly")
    for ds, q in [("Ribosembly", 4), ("IgG-RL", 2)]:
        c64 = cell("C", dataset=ds, q=q, n_images=64)
        c512 = cell("C", dataset=ds, q=q, n_images=512)
        if c64 and c512:
            d = c64["hun"] - c512["hun"]
            print(f"   {ds} q={q}: n=64 hun={c64['hun']:.3f}, n=512 hun={c512['hun']:.3f}, Δ={d:+.3f}")

    # H5 — Random U survives at low σ, collapses at higher σ
    print("\nH5: random-U survives at low σ; collapses at higher σ")
    for ds, q in [("Ribosembly", 4), ("Ribosembly", 8)]:
        for sig in [0.01, 0.1]:
            c = cell("B", dataset=ds, q=q, sigma=sig, u_init="random")
            if c:
                print(f"   {ds} q={q} σ={sig} random-U: hun={c['hun']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root", type=str)
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    cells = sorted((sweep_root / "cells").glob("*.json"))
    if not cells:
        raise SystemExit(f"No cell JSONs under {sweep_root / 'cells'}")

    rows = [parse_cell(p) for p in cells]
    print(
        f"Loaded {len(rows)} cells across "
        f"{len({r['axis'] for r in rows})} axes "
        f"({len({r['dataset'] for r in rows})} datasets)"
    )

    axis_a_snr(rows)
    axis_b_cold_inits(rows)
    axis_c_small_n(rows)
    axis_d_tomotwin(rows)
    hypothesis_verdicts(rows)


if __name__ == "__main__":
    main()
