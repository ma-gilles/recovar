"""Populate quality_baseline_5k_128.json with measured values from a real run.

Reads a recovar dump + matching RELION dump, extracts metrics via
``check_parity.extract_metrics``, then writes them into the placeholder
scenario in the baseline JSON with conservative tolerances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_parity_quality_baseline_20260425_103409")
sys.path.insert(0, str(REPO / "scripts/parity"))
from check_parity import extract_metrics  # noqa: E402


def _load(p: Path) -> dict:
    return dict(np.load(p, allow_pickle=False))


def _compute_tolerance(observed: float, kind: str) -> tuple[float, float]:
    """Return ``(tolerance, regression_threshold)`` for a metric.

    Tolerance = ~10% of the observed gap from the natural reference value
    (0 for ratios, 0.5 for vol_corr, etc.). Regression threshold = 3x.
    """
    if kind == "ave_pmax":
        return 0.05, 0.10
    if kind == "vol_corr_abs":
        return 0.02, 0.05
    if kind == "sigma_offset":
        return 0.5, 1.5
    return 0.1, 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--recovar-dump", required=True, type=Path)
    ap.add_argument("--relion-dump", required=True, type=Path)
    ap.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON dict to overwrite the scenario's config block (e.g. set init_iter)",
    )
    ap.add_argument("--note", default="auto-populated by populate_baseline.py")
    args = ap.parse_args()

    rec = _load(args.recovar_dump)
    rel = _load(args.relion_dump)
    metrics = extract_metrics(rec, rel)

    baseline = json.loads(args.baseline.read_text())
    scen = baseline["scenarios"].setdefault(args.scenario, {"config": {}, "expected_metrics": {}})

    if args.config_json:
        scen["config"] = json.loads(args.config_json)

    # Build expected_metrics — use floors / abs-vol_corr where appropriate
    em: dict[str, float] = {}
    if "ave_pmax" in metrics:
        em["ave_pmax"] = float(round(metrics["ave_pmax"], 6))
        em["ave_pmax_tolerance"] = 0.05
        em["ave_pmax_regression_threshold"] = 0.10

    # Hard-assign rates: use measured-value floor minus 0.05 (or 0 if measured is low)
    for k, label in [("pp_hard_assign_match_lt_5deg_rate", "5deg"), ("pp_hard_assign_match_lt_1deg_rate", "1deg")]:
        if k in metrics and not np.isnan(metrics[k]):
            obs = float(metrics[k])
            em[f"{k}_floor"] = max(0.0, round(obs - 0.05, 4))

    # Absolute volume correlation against RELION (sign-invariant)
    for k in (1, 2):
        key = f"vol_corr_abs_half{k}"
        if key in metrics and not np.isnan(metrics[key]):
            obs = float(metrics[key])
            em[f"{key}_floor"] = max(0.0, round(obs - 0.05, 4))

    # Sigma2_noise ratio bands (factor-of-2 around 1.0 by default,
    # widened if observed is very different from 1.0)
    for k in (1, 2):
        key = f"sigma2_noise_ratio_half{k}_med"
        if key in metrics and not np.isnan(metrics[key]):
            obs = float(metrics[key])
            if abs(obs - 1.0) < 0.5:
                em[f"{key}_band"] = [0.5, 2.0]
            else:
                lo = max(0.0, obs * 0.5)
                hi = obs * 2.0 if obs > 0 else 100.0
                em[f"{key}_band"] = [round(lo, 4), round(hi, 4)]

    # sigma_offset (Angstroms)
    if "sigma_offset_a" in metrics:
        em["sigma_offset_a"] = float(round(metrics["sigma_offset_a"], 4))
        em["sigma_offset_a_tolerance"] = 0.5

    scen["expected_metrics"] = em
    scen.pop("_PLACEHOLDER", None)

    if "wall_time_s" in metrics:
        # Round to nearest 30s
        wt = max(60.0, float(metrics["wall_time_s"]))
        scen["wall_time_s_baseline"] = round(wt / 30.0) * 30.0
        scen["wall_time_s_regression_threshold_multiplier"] = 3.0

    # Append source provenance
    sr = baseline.setdefault("source_runs", [])
    if "PLACEHOLDER" in sr[0] if sr else "":
        sr.clear()
    sr.append(
        {
            "scenario": args.scenario,
            "recovar_dump": str(args.recovar_dump),
            "relion_dump": str(args.relion_dump),
            "note": args.note,
            "measured_metrics": {
                k: float(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else None
                for k, v in metrics.items()
            },
        }
    )

    args.baseline.write_text(json.dumps(baseline, indent=2))
    print("Updated baseline:", args.baseline)
    print(json.dumps(em, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
