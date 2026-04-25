"""Aggregate Phase 1 ablation sweep results into pivot tables.

Usage:
    pixi run python scripts/ppca_abinitio/analyze_phase1_sweep.py \\
        <sweep_root_dir>

Reads every cells/*.json under the sweep root and emits:
- a long-form CSV (one row per cell)
- a pivot table per dataset/q row × (s_init, ridge, anneal) cell with
  seed-mean ± std for {hungarian, ari, nmi, pe, fre_truth, fre_fp,
  best_lm, final_lm}
- a decision summary against the predetermined Phase 1 rules
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_cell_json(path: Path) -> dict:
    """Read one cell JSON and flatten the metric/config fields we need."""
    with path.open() as f:
        data = json.load(f)
    cfg = data["config"]
    met = data["metrics"]
    ceilings = data.get("ceilings", {})
    row = {
        "cell_id": path.stem,
        "dataset": cfg["dataset"],
        "q": int(cfg["q"]),
        "s_init": cfg["s_init"],
        "ridge_mode": cfg["ridge_mode"],
        "anneal": cfg["anneal_schedule"],
        "anneal_factor_only": bool(cfg["anneal_factor_only"]),
        "seed": int(cfg["data_seed"]),
        "best_pe": met["best_pe"],
        "best_fre_truth": met["best_fre_truth"],
        "best_fre_fp": met["best_fre_fp"],
        "best_lm": met["best_lm"],
        "final_lm": met["final_lm"],
        "hun": met.get("clust_acc_hungarian", float("nan")),
        "ari": met.get("ari", float("nan")),
        "nmi": met.get("nmi", float("nan")),
        "ceiling_joint_hun": ceilings.get("joint_loop_hun", float("nan")),
        "ceiling_annealed_hun": ceilings.get("annealed_hun", float("nan")),
    }
    # row name combines dataset and q for grouping
    row["row"] = f"{row['dataset']}_q{row['q']}"
    # anneal label collapses (anneal=none, factor_only=False) and
    # (anneal=log1000, factor_only=True) into the experimental factor
    if row["anneal"] == "none":
        row["anneal_factor"] = "none"
    else:
        row["anneal_factor"] = "factor_only" if row["anneal_factor_only"] else "full"
    row["cell_factors"] = (row["s_init"], row["ridge_mode"], row["anneal_factor"])
    return row


def aggregate(rows: list[dict]) -> dict:
    """Group rows by (dataset, q) × (s_init, ridge, anneal) and reduce
    over seeds with mean ± std."""
    groups: dict = {}
    for r in rows:
        key = (r["row"], r["cell_factors"])
        groups.setdefault(key, []).append(r)

    metrics_to_reduce = ["hun", "ari", "nmi", "best_pe", "best_fre_truth", "best_lm", "final_lm"]
    summary = {}
    for key, members in groups.items():
        s = {"n_seeds": len(members)}
        for m in metrics_to_reduce:
            vals = np.array([mr[m] for mr in members], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                s[f"{m}_mean"], s[f"{m}_std"] = float("nan"), float("nan")
            else:
                s[f"{m}_mean"] = float(vals.mean())
                s[f"{m}_std"] = float(vals.std(ddof=0))
        # carry ceiling from first valid member
        for c in ("ceiling_joint_hun", "ceiling_annealed_hun"):
            cvals = [mr[c] for mr in members if not np.isnan(mr[c])]
            s[c] = float(cvals[0]) if cvals else float("nan")
        summary[key] = s
    return summary


def write_long_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "cell_id",
        "dataset",
        "q",
        "s_init",
        "ridge_mode",
        "anneal",
        "anneal_factor_only",
        "anneal_factor",
        "seed",
        "hun",
        "ari",
        "nmi",
        "best_pe",
        "best_fre_truth",
        "best_fre_fp",
        "best_lm",
        "final_lm",
        "ceiling_joint_hun",
        "ceiling_annealed_hun",
    ]
    with path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")


def fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100 * x:+.2f}%"


def print_pivot(summary: dict, row_name: str) -> None:
    """Per-row pivot: 8 cells (s_init × ridge × anneal_factor)."""
    print(f"\n=== Row: {row_name} ===")
    print(
        f"{'s_init':<8} {'ridge':<10} {'anneal':<14} "
        f"{'hun':>14} {'ari':>14} {'nmi':>14} {'pe':>14} "
        f"{'fre_truth':>14} {'best_lm':>14}"
    )
    cells = sorted(
        [(k[1], v) for k, v in summary.items() if k[0] == row_name],
        key=lambda kv: kv[0],
    )
    for cell_factors, s in cells:
        s_init, ridge, anneal = cell_factors
        n = s["n_seeds"]
        print(
            f"{s_init:<8} {ridge:<10} {anneal:<14} "
            f"{s['hun_mean']:>7.4f}±{s['hun_std']:.3f} "
            f"{s['ari_mean']:>7.4f}±{s['ari_std']:.3f} "
            f"{s['nmi_mean']:>7.4f}±{s['nmi_std']:.3f} "
            f"{s['best_pe_mean']:>7.4f}±{s['best_pe_std']:.3f} "
            f"{s['best_fre_truth_mean']:>7.4f}±{s['best_fre_truth_std']:.3f} "
            f"{s['best_lm_mean']:>7.0f}±{s['best_lm_std']:.0f} "
            f"(n={n})"
        )


def evaluate_decision_rules(summary: dict) -> list[str]:
    """Apply the three predetermined Phase 1 decision rules."""
    notes = []

    def cell(row, s_init, ridge, anneal):
        return summary.get((row, (s_init, ridge, anneal)))

    # Rule 1: flat ≈ truth on all 3 datasets ⇒ ship flat
    rule1_pass = []
    for row in ("Ribosembly_q4", "Ribosembly_q8", "IgG-RL_q2"):
        flat = cell(row, "flat", "scalar", "none")
        truth = cell(row, "truth", "scalar", "none")
        if flat is None or truth is None:
            rule1_pass.append((row, "MISSING"))
            continue
        # Within a 2-sigma combined band on hungarian
        diff = abs(flat["hun_mean"] - truth["hun_mean"])
        sigma_combined = np.sqrt(flat["hun_std"] ** 2 + truth["hun_std"] ** 2)
        within = diff <= 2 * max(sigma_combined, 0.01)
        rule1_pass.append((row, "PASS" if within else f"FAIL (Δ={diff:.3f})"))
    notes.append("Rule 1 — flat matches truth on hun within 2σ:")
    for r, status in rule1_pass:
        notes.append(f"    {r}: {status}")

    # Rule 2: w_prior beats scalar PE by >5% on ≥2 of {Ribo q4, Ribo q8} at s_init=flat, anneal=none
    rule2_pass = []
    for row in ("Ribosembly_q4", "Ribosembly_q8"):
        scal = cell(row, "flat", "scalar", "none")
        wpr = cell(row, "flat", "w_prior", "none")
        if scal is None or wpr is None:
            rule2_pass.append((row, "MISSING"))
            continue
        if scal["best_pe_mean"] <= 0:
            rule2_pass.append((row, "skip (pe~0)"))
            continue
        rel_gain = (scal["best_pe_mean"] - wpr["best_pe_mean"]) / scal["best_pe_mean"]
        rule2_pass.append((row, f"Δpe = {fmt_pct(rel_gain)}"))
    notes.append("Rule 2 — w_prior reduces PE by >5% on ≥2 of {Ribo q4, Ribo q8} at flat/no-anneal:")
    for r, status in rule2_pass:
        notes.append(f"    {r}: {status}")

    # Rule 3: factor-only-anneal matches no-anneal on IgG-RL AND beats no-anneal by >10% on Ribo q=8
    fr_iggrl_no = cell("IgG-RL_q2", "flat", "scalar", "none")
    fr_iggrl_fa = cell("IgG-RL_q2", "flat", "scalar", "factor_only")
    fr_ribo8_no = cell("Ribosembly_q8", "flat", "scalar", "none")
    fr_ribo8_fa = cell("Ribosembly_q8", "flat", "scalar", "factor_only")
    if all(c is not None for c in (fr_iggrl_no, fr_iggrl_fa, fr_ribo8_no, fr_ribo8_fa)):
        iggrl_safe = abs(fr_iggrl_no["hun_mean"] - fr_iggrl_fa["hun_mean"]) <= 2 * max(
            np.sqrt(fr_iggrl_no["hun_std"] ** 2 + fr_iggrl_fa["hun_std"] ** 2), 0.01
        )
        ribo8_gain = (fr_ribo8_fa["hun_mean"] - fr_ribo8_no["hun_mean"]) / max(fr_ribo8_no["hun_mean"], 1e-3)
        notes.append(
            f"Rule 3 — factor-only anneal: IgG-RL safe={iggrl_safe} (Δhun "
            f"{fr_iggrl_fa['hun_mean'] - fr_iggrl_no['hun_mean']:+.3f}), "
            f"Ribo q=8 lift {fmt_pct(ribo8_gain)}"
        )
    else:
        notes.append("Rule 3 — factor-only anneal: MISSING cells")

    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root", type=str)
    ap.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Prefix for the long CSV / summary outputs. Defaults to <sweep_root>/phase1_summary",
    )
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    json_dir = sweep_root / "cells"
    cells = sorted(json_dir.glob("*.json"))
    if not cells:
        raise SystemExit(f"No cell JSONs found under {json_dir}")

    rows = [parse_cell_json(p) for p in cells]
    summary = aggregate(rows)

    out_prefix = Path(args.out_prefix) if args.out_prefix else sweep_root / "phase1_summary"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    long_csv = out_prefix.with_suffix(".long.csv")
    write_long_csv(rows, long_csv)
    print(f"wrote {long_csv}  ({len(rows)} rows)")

    summary_json = out_prefix.with_suffix(".summary.json")
    with summary_json.open("w") as f:
        # Convert tuple keys to strings for JSON
        json.dump(
            {f"{k[0]} | {','.join(k[1])}": v for k, v in summary.items()},
            f,
            indent=2,
        )
    print(f"wrote {summary_json}")

    for row in ("Ribosembly_q4", "Ribosembly_q8", "IgG-RL_q2"):
        print_pivot(summary, row)

    print()
    print("=" * 70)
    print("DECISION RULES")
    print("=" * 70)
    for line in evaluate_decision_rules(summary):
        print(line)


if __name__ == "__main__":
    main()
