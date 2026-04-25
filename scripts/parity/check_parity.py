#!/usr/bin/env python3
"""Check a recovar parity dump against a frozen quality baseline + RELION reference.

For one scenario (a single mid-trajectory iter replay), reads:
- the recovar ``iter_NNN.npz`` produced by ``recovar.em.dense_single_volume.parity_dump``
- the matching RELION ``iter_NNN.npz`` produced by ``scripts/parity/dump_relion_iter.py``
- the baseline JSON at ``--baseline``

and compares observed metrics against the baseline's ``expected_metrics``.

Tolerance vocabulary, mirrored on the perf-baseline branch:
- ``<metric>_tolerance``  : absolute ``WARN`` band around the baseline value
- ``<metric>_floor``      : observed must be ``>=`` floor (used for things like vol_corr or hard-assignment match)
- ``<metric>_band``       : ``[lo, hi]`` interval — observed must lie inside (used for ratios)
- ``<metric>_regression_threshold`` : absolute ``REGRESSED`` gap (defaults to 2x ``_tolerance`` if not set)

Status thresholds:
    OK         : within tolerance / inside band / above floor
    WARN       : outside tolerance but not yet REGRESSED (only when both _tolerance + _regression_threshold provided)
    REGRESSED  : outside the regression threshold

Usage:
    pixi run python scripts/parity/check_parity.py \
        --baseline tests/baselines/parity/quality_baseline_5k_128.json \
        --scenario iter7_to_8_grouped_union \
        --recovar-dump _agent_scratch/parity/recovar_iter7/iter_008.npz \
        --relion-dump _agent_scratch/parity/relion/iter_008.npz \
        [--exit-code-on-regression]

Exits 0 unless ``--exit-code-on-regression`` is set; in that case nonzero
on any REGRESSED metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# --- metric extractors ----------------------------------------------------


def _load_npz(p: Path) -> dict[str, np.ndarray]:
    return dict(np.load(p, allow_pickle=False))


def _vol_real_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.shape != b.shape:
        return float("nan")
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.linalg.norm(am) * np.linalg.norm(bm)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(am, bm) / denom)


def _euler_angle_match_rate_lt(eulers_a: np.ndarray, eulers_b: np.ndarray, threshold_deg: float) -> float:
    """Per-particle angular geodesic distance, returning fraction with diff < threshold_deg."""
    from scipy.spatial.transform import Rotation as R

    if eulers_a is None or eulers_b is None:
        return float("nan")
    n = min(eulers_a.shape[0], eulers_b.shape[0])
    if n == 0:
        return float("nan")
    a = R.from_euler("ZYZ", eulers_a[:n], degrees=True)
    b = R.from_euler("ZYZ", eulers_b[:n], degrees=True)
    rel = a.inv() * b
    angle = np.rad2deg(rel.magnitude())
    return float(np.mean(angle < threshold_deg))


def _noise_ratio_med(rec_noise: np.ndarray, rel_noise: np.ndarray) -> float:
    if rec_noise is None or rel_noise is None or rec_noise.size == 0 or rel_noise.size == 0:
        return float("nan")
    n = min(rec_noise.size, rel_noise.size)
    denom = np.maximum(np.abs(rel_noise[:n]), 1e-30)
    return float(np.median(rec_noise[:n] / denom))


def extract_metrics(rec: dict[str, np.ndarray], rel: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract every metric the baseline schema knows about for one iter."""

    metrics: dict[str, float] = {}

    # Average max-posterior (recovar vs RELION model.star)
    metrics["ave_pmax"] = float(rec.get("ave_pmax", np.nan))
    rel_pmax = float(rel.get("ave_pmax_model", np.nan))
    metrics["ave_pmax_relion_ref"] = rel_pmax
    metrics["ave_pmax_gap"] = metrics["ave_pmax"] - rel_pmax

    # Per-particle hard-assignment match rate vs RELION poses
    rec_h1 = rec.get("half1_best_eulers_total")
    rec_h2 = rec.get("half2_best_eulers_total")
    rel_eul = rel.get("particle_eulers")
    if rec_h1 is not None and rec_h2 is not None and rel_eul is not None:
        rec_eul = np.concatenate([rec_h1, rec_h2], axis=0)
        metrics["pp_hard_assign_match_lt_5deg_rate"] = _euler_angle_match_rate_lt(rec_eul, rel_eul, 5.0)
        metrics["pp_hard_assign_match_lt_1deg_rate"] = _euler_angle_match_rate_lt(rec_eul, rel_eul, 1.0)

    # Volume correlation against RELION half-maps. Both signed and absolute
    # metrics are reported -- recovar's volume can be negated relative to
    # RELION's depending on whether ``invert_data`` was applied (see project
    # memory ``project_relion_parity_negation_invert_data``). Use
    # ``vol_corr_abs_*`` for sign-invariant baselines that hold across both
    # conventions.
    for k in (1, 2):
        a = rec.get(f"half{k}_mean_real_ds")
        b = rel.get(f"half{k}_mean_real_ds")
        if a is None or b is None:
            metrics[f"vol_corr_half{k}"] = float("nan")
            metrics[f"vol_corr_abs_half{k}"] = float("nan")
        else:
            corr = _vol_real_corr(a, b)
            metrics[f"vol_corr_half{k}"] = corr
            metrics[f"vol_corr_abs_half{k}"] = float("nan") if corr != corr else abs(corr)

    # Sigma2_noise ratio (median across shells, recovar / RELION)
    for k in (1, 2):
        rec_n = rec.get(f"half{k}_wsum_sigma2_noise")
        rel_n = rel.get(f"half{k}_sigma2_noise")
        metrics[f"sigma2_noise_ratio_half{k}_med"] = _noise_ratio_med(rec_n, rel_n)

    # Sigma offset (translation prior std), in Angstroms
    metrics["sigma_offset_a"] = float(rec.get("sigma_offset", np.nan))
    metrics["sigma_offset_a_relion_ref"] = float(rel.get("sigma_offset", np.nan))
    metrics["sigma_offset_a_gap"] = metrics["sigma_offset_a"] - metrics["sigma_offset_a_relion_ref"]

    # Wall time (recovar only; from parity_dump.start_iteration → mark_stage)
    if "wall_time_s" in rec:
        metrics["wall_time_s"] = float(rec["wall_time_s"])

    return metrics


# --- comparison ------------------------------------------------------------


def _check_metric(
    name: str,
    observed: float,
    expected: dict[str, float],
) -> tuple[str, str]:
    """Return ``(status, message)`` for a single metric."""

    if math.isnan(observed):
        return "UNKNOWN", f"{name}: <nan> — could not compute"

    # tolerance form: ``<name>_tolerance`` (absolute, symmetric)
    tol = expected.get(f"{name}_tolerance")
    floor = expected.get(f"{name}_floor")
    band = expected.get(f"{name}_band")
    regress_threshold = expected.get(f"{name}_regression_threshold")
    base = expected.get(name)

    if floor is not None:
        # floor mode: observed must be >= floor
        if observed < float(floor):
            sev = "REGRESSED"
        else:
            sev = "OK"
        msg = f"{name}: {observed:.6f} vs baseline {base if base is not None else 'n/a'} (floor {floor}) {sev}"
        return sev, msg

    if band is not None:
        lo, hi = float(band[0]), float(band[1])
        if observed < lo or observed > hi:
            sev = "REGRESSED"
        else:
            sev = "OK"
        msg = f"{name}: {observed:.6f} vs baseline {base if base is not None else 'n/a'} (band [{lo}, {hi}]) {sev}"
        return sev, msg

    if base is not None and tol is not None:
        diff = observed - float(base)
        adiff = abs(diff)
        if regress_threshold is not None and adiff > float(regress_threshold):
            sev = "REGRESSED"
        elif adiff > float(tol):
            # If no explicit regression_threshold was given, treat any tolerance miss as REGRESSED
            sev = "REGRESSED" if regress_threshold is None else "WARN"
        else:
            sev = "OK"
        msg = (
            f"{name}: {observed:.6f} vs baseline {float(base):.6f} "
            f"(diff {diff:+.6f}, tol {float(tol):.4f}"
            f"{', regress at ' + str(regress_threshold) if regress_threshold is not None else ''}) {sev}"
        )
        return sev, msg

    # No constraints recorded for this metric
    return "UNKNOWN", f"{name}: {observed:.6f} (no baseline)"


def check_scenario(
    scenario_name: str,
    expected_metrics: dict[str, float],
    observed_metrics: dict[str, float],
    wall_time_baseline: float | None = None,
    wall_time_regression_multiplier: float = 3.0,
) -> tuple[list[str], list[str]]:
    """Returns (status_lines, regressed_names)."""

    lines: list[str] = [f"=== {scenario_name} ==="]
    regressed: list[str] = []

    # Iterate over expected metric "base" entries, pulling the matching observed value.
    base_keys = sorted(
        {
            k
            for k in expected_metrics.keys()
            if not (
                k.endswith("_tolerance")
                or k.endswith("_floor")
                or k.endswith("_band")
                or k.endswith("_regression_threshold")
            )
        }
    )
    for k in base_keys:
        if k not in observed_metrics:
            continue
        sev, msg = _check_metric(k, observed_metrics[k], expected_metrics)
        lines.append(msg)
        if sev == "REGRESSED":
            regressed.append(k)

    # Floor / band-only metrics (no plain base) need an extra pass.
    fb_keys = {
        k.replace("_floor", "").replace("_band", "")
        for k in expected_metrics.keys()
        if k.endswith("_floor") or k.endswith("_band")
    }
    for k in sorted(fb_keys - set(base_keys)):
        if k not in observed_metrics:
            continue
        sev, msg = _check_metric(k, observed_metrics[k], expected_metrics)
        lines.append(msg)
        if sev == "REGRESSED":
            regressed.append(k)

    # Wall time
    wt = observed_metrics.get("wall_time_s")
    if wall_time_baseline is not None and wt is not None:
        threshold = float(wall_time_baseline) * float(wall_time_regression_multiplier)
        sev = "REGRESSED" if wt > threshold else "OK"
        lines.append(
            f"wall_time_s: {wt:.1f} vs baseline {float(wall_time_baseline):.1f} "
            f"(regression threshold {threshold:.1f}) {sev}"
        )
        if sev == "REGRESSED":
            regressed.append("wall_time_s")

    if regressed:
        lines.append(f"=== REGRESSED: {', '.join(regressed)} ===")
    else:
        lines.append("=== ALL OK ===")
    return lines, regressed


# --- CLI -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--recovar-dump", required=True, type=Path)
    ap.add_argument("--relion-dump", required=True, type=Path)
    ap.add_argument("--exit-code-on-regression", action="store_true")
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    scenarios = baseline.get("scenarios", {})
    if args.scenario not in scenarios:
        print(f"ERROR: scenario {args.scenario!r} not in baseline {args.baseline}", file=sys.stderr)
        return 2
    scen = scenarios[args.scenario]
    expected = scen.get("expected_metrics", {})

    rec = _load_npz(args.recovar_dump)
    rel = _load_npz(args.relion_dump)
    observed = extract_metrics(rec, rel)

    lines, regressed = check_scenario(
        args.scenario,
        expected,
        observed,
        wall_time_baseline=scen.get("wall_time_s_baseline"),
        wall_time_regression_multiplier=float(scen.get("wall_time_s_regression_threshold_multiplier", 3.0)),
    )
    print("\n".join(lines))

    if args.exit_code_on_regression and regressed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
