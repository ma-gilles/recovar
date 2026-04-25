#!/usr/bin/env python
"""Compare a recovar parity_dump dir against a RELION reference dump dir.

Writes a markdown report with per-iter metrics and identifies the first
iteration where ave_Pmax / hard-assignment / volume parity diverges past
the configured tolerances.

Usage:
    pixi run python scripts/parity/compare_dumps.py \
        --recovar _agent_scratch/parity/recovar/<run_id> \
        --relion _agent_scratch/parity/relion \
        --out _agent_scratch/parity/reports/<tag>.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PMAX_GAP_TOL = 5e-5
VOL_CORR_FLOOR = 0.9995
HA_MATCH_FLOOR = 0.995


def _load(p: Path) -> dict[str, np.ndarray]:
    return dict(np.load(p, allow_pickle=False))


def _list_iters(d: Path) -> list[int]:
    out = []
    for f in d.glob("iter_*.npz"):
        try:
            out.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


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


def _euler_angular_distance_deg(eulers_a: np.ndarray, eulers_b: np.ndarray) -> np.ndarray:
    """Return per-row geodesic angle in degrees between rotations encoded as
    RELION eulers (rot, tilt, psi) in degrees."""
    from scipy.spatial.transform import Rotation as R

    if eulers_a is None or eulers_b is None:
        return None
    n = min(eulers_a.shape[0], eulers_b.shape[0])
    a = R.from_euler("ZYZ", eulers_a[:n], degrees=True)
    b = R.from_euler("ZYZ", eulers_b[:n], degrees=True)
    rel = a.inv() * b
    angle = rel.magnitude()
    return np.rad2deg(angle).astype(np.float64)


def _quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.99)) -> dict[str, float]:
    if x is None or x.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    return {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def _coalesce(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def compare_iter(rec: dict, rel: dict) -> dict:
    out: dict[str, object] = {}

    rec_pmax = float(rec.get("ave_pmax", np.nan))
    rel_pmax = float(rel.get("ave_pmax_model", np.nan))
    out["rec_ave_pmax"] = rec_pmax
    out["rel_ave_pmax"] = rel_pmax
    out["pmax_gap"] = rec_pmax - rel_pmax

    rec_cs = int(rec.get("current_size", -1))
    rel_cs = int(rel.get("current_image_size", -1))
    out["rec_current_size"] = rec_cs
    out["rel_current_size"] = rel_cs

    rec_so = float(rec.get("sigma_offset", np.nan))
    rel_so = float(rel.get("sigma_offset", np.nan))
    out["rec_sigma_offset"] = rec_so
    out["rel_sigma_offset"] = rel_so
    out["sigma_offset_gap"] = rec_so - rel_so

    rec_pert = float(rec.get("random_perturbation", np.nan))
    rel_pert = float(rel.get("perturb_instance", np.nan))
    out["rec_perturb"] = rec_pert
    out["rel_perturb"] = rel_pert

    # Per-particle Pmax distribution
    rel_pp_pmax = rel.get("particle_max_pmax")
    rec_pp_pmax_h1 = rec.get("half1_max_posterior")
    rec_pp_pmax_h2 = rec.get("half2_max_posterior")
    if rec_pp_pmax_h1 is not None and rec_pp_pmax_h2 is not None and rel_pp_pmax is not None:
        rec_pp = np.concatenate([rec_pp_pmax_h1, rec_pp_pmax_h2])
        if rec_pp.shape == rel_pp_pmax.shape:
            out["pp_pmax_abs_gap_q"] = _quantiles(np.abs(rec_pp.astype(np.float64) - rel_pp_pmax.astype(np.float64)))
        else:
            out["pp_pmax_abs_gap_q"] = {"q50": float("nan"), "q90": float("nan"), "q99": float("nan")}

    # Hard-assignment via euler distance
    rec_eul_h1 = rec.get("half1_best_eulers_total")
    rec_eul_h2 = rec.get("half2_best_eulers_total")
    rel_eulers = rel.get("particle_eulers")
    if rec_eul_h1 is not None and rec_eul_h2 is not None and rel_eulers is not None:
        rec_eul = np.concatenate([rec_eul_h1, rec_eul_h2], axis=0)
        n = min(rec_eul.shape[0], rel_eulers.shape[0])
        ang_diff = _euler_angular_distance_deg(rec_eul[:n], rel_eulers[:n])
        if ang_diff is not None:
            out["pp_angle_deg_q"] = _quantiles(ang_diff)
            out["pp_angle_match_lt_1deg"] = float(np.mean(ang_diff < 1.0))
            out["pp_angle_match_lt_5deg"] = float(np.mean(ang_diff < 5.0))

    # Translation distance (pixels)
    rec_trans_h1 = rec.get("half1_best_translations_total")
    rec_trans_h2 = rec.get("half2_best_translations_total")
    rel_trans = rel.get("particle_origin_pixel_xy")
    if rec_trans_h1 is not None and rec_trans_h2 is not None and rel_trans is not None:
        rec_trans = np.concatenate([rec_trans_h1, rec_trans_h2], axis=0)
        n = min(rec_trans.shape[0], rel_trans.shape[0])
        # RELION sign convention: stored origin is the prior origin to subtract from image; recovar
        # stores the absolute offset applied to the image. Compare magnitude of (rec - (-rel)) and (rec - rel).
        diff_pos = np.linalg.norm(rec_trans[:n].astype(np.float64) - rel_trans[:n].astype(np.float64), axis=1)
        diff_neg = np.linalg.norm(rec_trans[:n].astype(np.float64) - (-rel_trans[:n].astype(np.float64)), axis=1)
        diff = np.minimum(diff_pos, diff_neg)
        out["pp_trans_pix_q"] = _quantiles(diff)

    # Volume correlation
    for k in (1, 2):
        a = rec.get(f"half{k}_mean_real_ds")
        b = rel.get(f"half{k}_mean_real_ds")
        out[f"vol_corr_half{k}"] = _vol_real_corr(a, b) if a is not None and b is not None else float("nan")

    # Sigma2_noise gap (RELION typically stores per shell at full res)
    for k in (1, 2):
        rec_n = rec.get(f"half{k}_wsum_sigma2_noise")
        rel_n = rel.get(f"half{k}_sigma2_noise")
        if rec_n is not None and rel_n is not None and rec_n.size > 0 and rel_n.size > 0:
            n = min(rec_n.size, rel_n.size)
            denom = np.maximum(np.abs(rel_n[:n]), 1e-30)
            ratio = rec_n[:n] / denom
            out[f"half{k}_noise_ratio_med"] = float(np.median(ratio))
            out[f"half{k}_noise_ratio_q90"] = float(np.quantile(ratio, 0.9))

    # Ft per-shell norm ratio
    for k in (1, 2):
        rec_y = rec.get(f"half{k}_Ft_y_per_shell")
        rec_c = rec.get(f"half{k}_Ft_ctf_per_shell")
        if rec_y is not None and rec_y.size > 0:
            out[f"half{k}_Ft_y_total"] = float(rec_y.sum())
        if rec_c is not None and rec_c.size > 0:
            out[f"half{k}_Ft_ctf_total"] = float(rec_c.sum())

    return out


def render_report(rows: list[tuple[int, dict]]) -> str:
    lines = ["# RELION parity report", "", "## Per-iter summary", ""]
    lines.append(
        "| iter | rec Pmax | rel Pmax | Pmax gap | rec cs | rel cs | sigma_off gap | vol corr h1 | vol corr h2 | ang q50 | ang q99 | trans q90 | <1deg |"
    )
    lines.append(
        "|------|----------|----------|----------|--------|--------|---------------|-------------|-------------|---------|---------|-----------|-------|"
    )
    for it, r in rows:
        ang_q = r.get("pp_angle_deg_q") or {"q50": float("nan"), "q99": float("nan")}
        trans_q = r.get("pp_trans_pix_q") or {"q90": float("nan")}
        lines.append(
            "| {it} | {rp:.4f} | {lp:.4f} | {pg:+.4e} | {rcs:>4d} | {lcs:>4d} | {sog:+.3e} | {vc1:.5f} | {vc2:.5f} | {a50:.3f} | {a99:.3f} | {t90:.3f} | {pct1:.3f} |".format(
                it=it,
                rp=r.get("rec_ave_pmax", float("nan")),
                lp=r.get("rel_ave_pmax", float("nan")),
                pg=r.get("pmax_gap", float("nan")),
                rcs=r.get("rec_current_size", -1),
                lcs=r.get("rel_current_size", -1),
                sog=r.get("sigma_offset_gap", float("nan")),
                vc1=r.get("vol_corr_half1", float("nan")),
                vc2=r.get("vol_corr_half2", float("nan")),
                a50=ang_q["q50"],
                a99=ang_q["q99"],
                t90=trans_q["q90"],
                pct1=r.get("pp_angle_match_lt_1deg", float("nan")),
            )
        )

    lines += ["", "## First-divergence assessment", ""]
    first_pmax = next((it for it, r in rows if abs(r.get("pmax_gap", 0.0)) > PMAX_GAP_TOL), None)
    first_vol = next(
        (it for it, r in rows if min(r.get("vol_corr_half1", 1.0), r.get("vol_corr_half2", 1.0)) < VOL_CORR_FLOOR),
        None,
    )
    first_ha = next(
        (it for it, r in rows if (r.get("pp_angle_match_lt_1deg", 1.0) or 1.0) < HA_MATCH_FLOOR),
        None,
    )
    lines.append(f"- First iter with |pmax_gap| > {PMAX_GAP_TOL}: **{first_pmax}**")
    lines.append(f"- First iter with vol_corr < {VOL_CORR_FLOOR}: **{first_vol}**")
    lines.append(f"- First iter with <1deg-angle match rate < {HA_MATCH_FLOOR}: **{first_ha}**")

    lines += ["", "## Per-iter detail", ""]
    for it, r in rows:
        lines.append(f"### iter {it}")
        for key in sorted(r.keys()):
            val = r[key]
            if isinstance(val, dict):
                lines.append(f"- {key}: " + ", ".join(f"{k}={v:.6g}" for k, v in val.items()))
            elif isinstance(val, float):
                lines.append(f"- {key}: {val:.6g}")
            else:
                lines.append(f"- {key}: {val}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovar", required=True, type=Path)
    ap.add_argument("--relion", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rec_iters = set(_list_iters(args.recovar))
    rel_iters = set(_list_iters(args.relion))
    iters = sorted(rec_iters & rel_iters)
    if not iters:
        raise SystemExit(f"No matching iters between {args.recovar} and {args.relion}")

    rows = []
    for it in iters:
        rec = _load(args.recovar / f"iter_{it:03d}.npz")
        rel = _load(args.relion / f"iter_{it:03d}.npz")
        rows.append((it, compare_iter(rec, rel)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
