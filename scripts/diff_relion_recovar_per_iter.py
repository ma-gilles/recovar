#!/usr/bin/env python
"""Per-iteration parity diff between RELION and recovar refine outputs.

For each iteration index, loads:
  - RELION:  <relion_dir>/run_it{NNN}_optimiser.star      (scalars)
             <relion_dir>/run_it{NNN}_half1_model.star    (per-shell, half 1)
             <relion_dir>/run_it{NNN}_half2_model.star    (per-shell, half 2)
             <relion_dir>/run_it{NNN}_data.star           (per-particle, incl.
                                                          rlnMaxValueProbDistribution)
  - recovar: <recovar_dir>/refinement_results.npz         (all per-iter dumps)

Reports per-iter, side-by-side:
  - current_size (Fourier window radius — RELION's _rlnCurrentImageSize)
  - ave_Pmax     (mean of RELION's per-particle rlnMaxValueProbDistribution
                  from data.star — the apples-to-apples comparison vs
                  recovar's per-particle E-step Pmax mean)
  - ave_Pmax_mstep (RELION's _rlnAveragePmax from model.star — this is the
                    M-step accumulator; differs from the per-particle mean
                    by ~3% for half-set/full-set accounting reasons. NOT
                    directly comparable to recovar's ave_Pmax. Kept here
                    for completeness only.)
  - current_resolution  (RELION's _rlnCurrentResolution)
  - healpix_order
  - changes in angles / offsets / classes (RELION-only; recovar tracks differently)
  - per-shell sigma2_noise (avg of half1+half2 from model.star vs ?)
  - per-shell tau2 (RELION's _rlnReferenceTau2 vs recovar's prior)
  - per-shell FSC_gold_std (RELION's _rlnGoldStandardFsc vs recovar's fsc_iter_NNN)
  - per-shell data_vs_prior (RELION's _rlnDataVsPriorRatio vs recovar's
    data_vs_prior_iter_NNN)

Highlights any field that differs by more than --tol relative.

Usage:
  pixi run python scripts/diff_relion_recovar_per_iter.py \\
    --relion_dir /scratch/.../data_noise1_5k/relion_ref_parity \\
    --recovar_dir /scratch/.../runs/recovar_5k_parity \\
    --max_iter 5 --tol 0.01

WARNING: scalar field names are checked but recovar's npz currently does
NOT dump per-iter sigma2_noise / tau2. Those columns will say "MISSING IN
RECOVAR — instrument refine.py to dump per-iter sigma2/tau2".
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import starfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"


def fmt(val, w=12, prec=4):
    """Format a value for the side-by-side table."""
    if val is None:
        return f"{'—':>{w}s}"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):>{w}d}"
    if isinstance(val, (float, np.floating)):
        if abs(val) < 1e-3 or abs(val) > 1e6:
            return f"{val:>{w}.{prec}e}"
        return f"{val:>{w}.{prec}f}"
    return f"{str(val):>{w}s}"


def color_diff(a, b, tol=0.01):
    """Return color code based on relative difference."""
    if a is None or b is None:
        return YELLOW
    if isinstance(a, str) or isinstance(b, str):
        return GREEN if a == b else RED
    denom = max(abs(a), abs(b), 1e-30)
    rel = abs(a - b) / denom
    if rel < tol:
        return GREEN
    if rel < 10 * tol:
        return YELLOW
    return RED


def parse_relion_optimiser(path):
    """Parse RELION optimiser STAR — starfile returns each scalar as a top-level entry."""
    if not path.exists():
        return None
    data = starfile.read(str(path))
    if isinstance(data, dict):
        # Flatten: each rln* field is a key with a scalar value
        return data
    return dict(data)


def parse_relion_model(path):
    """Parse RELION model STAR. Multi-block:
    - model_general (scalars dict)
    - model_classes (per-class df)
    - model_class_N (per-shell df: tau2, sigma2, FSC, ssnr)
    - model_groups (per-particle-group df)
    - model_optics_group_N (per-shell sigma2_noise per optics group)
    """
    if not path.exists():
        return None
    data = starfile.read(str(path))
    if not isinstance(data, dict):
        return {"model_general": data}
    return data


def get_field(d, *names):
    """Get the first matching field from a dict (multiple possible names)."""
    if d is None:
        return None
    for name in names:
        if name in d:
            return d[name]
        # Try with underscore-name removed
        for k in d:
            if k.endswith(name) or k == name:
                return d[k]
    return None


def load_relion_iter(relion_dir, it):
    """Load all per-iter STARs for one RELION iteration."""
    nnn = f"{it:03d}"
    out = {}
    out["optimiser"] = parse_relion_optimiser(relion_dir / f"run_it{nnn}_optimiser.star")
    out["model_h1"] = parse_relion_model(relion_dir / f"run_it{nnn}_half1_model.star")
    out["model_h2"] = parse_relion_model(relion_dir / f"run_it{nnn}_half2_model.star")
    data_path = relion_dir / f"run_it{nnn}_data.star"
    out["data"] = starfile.read(str(data_path)) if data_path.exists() else None
    return out


def extract_relion_scalars(relion_iter):
    """Extract per-iter scalar values from RELION's star files.

    Per-iter scalars live in THREE places:
      - model.star::model_general — current_size, current_resolution,
        sigma_offsets, log_likelihood, norm_correction, _rlnAveragePmax
        (M-step accumulator — see note on ``ave_Pmax`` below).
      - data.star — per-particle ``rlnMaxValueProbDistribution`` column.
        Mean of this column is the apples-to-apples comparison to recovar's
        per-particle E-step Pmax mean (``ave_Pmax_trajectory[i]``).
      - optimiser.star — smallest_changes (convergence indicators), iter counters

    NOTE on ``ave_Pmax``:
        ``_rlnAveragePmax`` written into ``run_it{NNN}_half{1,2}_model.star``
        is RELION's M-step half-set accumulator. It differs from the mean of
        ``rlnMaxValueProbDistribution`` (per-particle column in
        ``run_it{NNN}_data.star``) by ~3% for half-set/full-set accounting
        reasons. recovar's ``ave_Pmax_trajectory`` is the per-particle E-step
        mean, so the apples-to-apples RELION column is the per-particle one.
        Setting ``out["ave_Pmax"]`` from the per-particle column closes the
        spurious 0.0412-vs-0.0436 "iter-1 5.5% gap" that was just two
        differently-aggregated RELION numbers.
    """
    out = {}
    opt = relion_iter["optimiser"]
    model = relion_iter["model_h1"]
    data = relion_iter.get("data")

    # Per-particle Pmax mean from data.star — this is the apples-to-apples
    # comparison versus recovar's ave_Pmax_trajectory entry.
    relion_data_df = None
    if data is not None:
        relion_data_df = data["particles"] if isinstance(data, dict) and "particles" in data else data
    if relion_data_df is not None and "rlnMaxValueProbDistribution" in relion_data_df:
        col = np.asarray(relion_data_df["rlnMaxValueProbDistribution"], dtype=np.float64)
        if col.size:
            out["ave_Pmax"] = float(col.mean())

    # From model_general (the per-iter "state" block)
    if model and "model_general" in model:
        mg = model["model_general"]
        out["current_size"] = int(mg.get("rlnCurrentImageSize", 0) or 0)
        # _rlnAveragePmax is RELION's M-step accumulator. Keep it under a
        # distinct key so the comparison table can show the discrepancy
        # without conflating it with the per-particle metric.
        out["ave_Pmax_mstep"] = float(mg.get("rlnAveragePmax", float("nan")))
        # Fallback: if data.star did not carry rlnMaxValueProbDistribution
        # (e.g. iter-0 bootstrap), use the M-step accumulator so downstream
        # code still has *some* value rather than KeyError. The comparison
        # row will then trivially be NaN-vs-recovar at that iter.
        if "ave_Pmax" not in out:
            out["ave_Pmax"] = out["ave_Pmax_mstep"]
        out["current_resolution"] = float(mg.get("rlnCurrentResolution", float("nan")))
        out["log_likelihood"] = float(mg.get("rlnLogLikelihood", float("nan")))
        out["norm_correction_avg"] = float(mg.get("rlnNormCorrectionAverage", float("nan")))
        out["sigma_offsets_angst"] = float(mg.get("rlnSigmaOffsetsAngst", float("nan")))
        out["tau2_fudge"] = float(mg.get("rlnTau2FudgeFactor", float("nan")))
        out["nr_groups"] = int(mg.get("rlnNrGroups", 0) or 0)

    # From optimiser.star (convergence indicators)
    if opt:
        out["current_iter"] = int(opt.get("rlnCurrentIteration", 0) or 0)
        out["best_resolution_so_far"] = float(opt.get("rlnBestResolutionThusFar", float("nan")))
        out["smallest_change_angles"] = float(opt.get("rlnSmallestChangesOrientations", float("nan")))
        out["smallest_change_offsets"] = float(opt.get("rlnSmallestChangesOffsets", float("nan")))
        out["smallest_change_classes"] = float(opt.get("rlnSmallestChangesClasses", float("nan")))
        out["n_iter_no_resolution_gain"] = int(opt.get("rlnNumberOfIterWithoutResolutionGain", 0) or 0)
        out["has_high_fsc_at_limit"] = int(opt.get("rlnHasHighFscAtResolLimit", 0) or 0)
        out["has_converged"] = int(opt.get("rlnHasConverged", 0) or 0)
        out["increment_image_size"] = int(opt.get("rlnIncrementImageSize", 0) or 0)
    return out


def extract_relion_per_shell(relion_iter, half):
    """Extract per-shell arrays from RELION's half model.star.

    Pulls the per-shell class table (`model_class_1`) which has
    rlnGoldStandardFsc, rlnReferenceTau2, rlnReferenceSigma2, rlnSsnrMap,
    and the per-optics-group table (`model_optics_group_1`) which has
    rlnSigma2Noise. We use the optics-group sigma2 since that's what RELION
    actually uses in the M-step.
    """
    model = relion_iter[f"model_h{half}"]
    if model is None:
        return None
    out = {"_n_shells": 0}

    # Per-shell class table (tau2, FSC, ssnr, sigma2 from prior)
    if "model_class_1" in model:
        df = model["model_class_1"]
        out["_n_shells"] = len(df)
        for col in df.columns:
            key = col.replace("rln", "")
            out[key] = np.asarray(df[col].values, dtype=np.float64)

    # Per-shell optics-group sigma2_noise (RELION's actual noise model)
    if "model_optics_group_1" in model:
        df = model["model_optics_group_1"]
        if "rlnSigma2Noise" in df.columns:
            out["Sigma2Noise"] = np.asarray(df["rlnSigma2Noise"].values, dtype=np.float64)

    return out


def load_recovar(npz_path):
    if not npz_path.exists():
        return None
    return np.load(npz_path, allow_pickle=False)


def extract_recovar_scalars(recovar, it):
    """Extract per-iter scalars from recovar's npz at iter index `it` (0-based)."""
    if recovar is None:
        return {}
    cs_arr = recovar.get("current_sizes")
    pr_arr = recovar.get("pixel_resolutions")
    pmax_arr = recovar.get("ave_Pmax_trajectory")
    hpx_arr = recovar.get("healpix_order_trajectory")
    sigma_offset_arr = recovar.get("sigma_offset_trajectory")
    sigma_offset_used_arr = recovar.get("sigma_offset_used_trajectory")
    if sigma_offset_arr is None:
        sigma_offset_arr = sigma_offset_used_arr
    frac_changed_arr = recovar.get("frac_changed_trajectory")
    acc_rot_arr = recovar.get("acc_rot_trajectory")
    smallest_change_angles_arr = recovar.get("smallest_change_angles_trajectory")
    smallest_change_offsets_arr = recovar.get("smallest_change_offsets_trajectory")
    out = {}
    if cs_arr is not None and it < len(cs_arr):
        out["current_size"] = int(cs_arr[it])
    if pmax_arr is not None and it < len(pmax_arr):
        out["ave_Pmax"] = float(pmax_arr[it])
    if pr_arr is not None and it < len(pr_arr):
        out["current_resolution_pix"] = int(pr_arr[it])
    if hpx_arr is not None and it < len(hpx_arr):
        out["healpix_order"] = int(hpx_arr[it])
    if sigma_offset_arr is not None and it < len(sigma_offset_arr):
        out["sigma_offsets_angst"] = float(sigma_offset_arr[it])
    if sigma_offset_used_arr is not None and it < len(sigma_offset_used_arr):
        out["sigma_offsets_used_angst"] = float(sigma_offset_used_arr[it])
    if frac_changed_arr is not None and it < len(frac_changed_arr):
        out["fraction_changed"] = float(frac_changed_arr[it])
    if acc_rot_arr is not None and it < len(acc_rot_arr):
        out["acc_rot"] = float(acc_rot_arr[it])
    if smallest_change_angles_arr is not None and it < len(smallest_change_angles_arr):
        out["smallest_change_angles"] = float(smallest_change_angles_arr[it])
    if smallest_change_offsets_arr is not None and it < len(smallest_change_offsets_arr):
        out["smallest_change_offsets"] = float(smallest_change_offsets_arr[it])
    return out


def extract_recovar_per_shell(recovar, it):
    """Extract per-shell arrays from recovar at iter index `it` (0-based)."""
    if recovar is None:
        return None
    nnn = f"{it:03d}"
    out = {}
    fsc_key = f"fsc_iter_{nnn}"
    dvp_key = f"data_vs_prior_iter_{nnn}"
    sig_key = f"sig_counts_iter_{nnn}"
    noise_key = f"noise_radial_iter_{nnn}"
    tau2_key = f"tau2_radial_iter_{nnn}"
    sigma2_key = f"tau2_sigma2_iter_{nnn}"
    tau2_fsc_key = f"tau2_fsc_used_iter_{nnn}"
    ssnr_key = f"tau2_ssnr_iter_{nnn}"
    if fsc_key in recovar.files:
        out["FSC_gold_std"] = np.asarray(recovar[fsc_key], dtype=np.float64)
    if dvp_key in recovar.files:
        out["DataVsPriorRatio"] = np.asarray(recovar[dvp_key], dtype=np.float64)
    if sig_key in recovar.files:
        try:
            out["_sig_counts"] = np.asarray(recovar[sig_key], dtype=np.float64)
        except ValueError:
            pass
    if noise_key in recovar.files:
        out["Sigma2Noise"] = np.asarray(recovar[noise_key], dtype=np.float64)
    if tau2_key in recovar.files:
        out["ReferenceTau2"] = np.asarray(recovar[tau2_key], dtype=np.float64)
    if sigma2_key in recovar.files:
        out["ReferenceSigma2"] = np.asarray(recovar[sigma2_key], dtype=np.float64)
    if tau2_fsc_key in recovar.files:
        out["Tau2FscUsed"] = np.asarray(recovar[tau2_fsc_key], dtype=np.float64)
        out["FSC_gold_std"] = out["Tau2FscUsed"]
    if ssnr_key in recovar.files:
        out["SsnrMap"] = np.asarray(recovar[ssnr_key], dtype=np.float64)
        out["DataVsPriorRatio"] = out["SsnrMap"]
    return out if out else None


def fsc_resolution_angstrom(fsc, voxel_size, grid_size, threshold=0.143):
    """Convert FSC curve to resolution in A using gold-std 0.143 threshold."""
    fsc = np.asarray(fsc)
    below = np.where(fsc < threshold)[0]
    if len(below) == 0:
        return float("nan")
    shell = int(below[0])
    if shell == 0:
        return float("inf")
    return float(grid_size) * float(voxel_size) / shell


def summarize_metric(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    q90, q95, q99 = np.percentile(arr, [90, 95, 99])
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(arr.max()),
    }


def fraction_within(arr, thresholds):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return {float(thr): float(np.mean(arr <= thr)) for thr in thresholds}


def load_saved_gt_metrics(recovar_dir, it):
    path = recovar_dir / f"gt_comparison_iter{it:03d}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def print_metric_block(prefix, pose_npz, metric_specs):
    for key, label, thresholds in metric_specs:
        if key not in pose_npz.files:
            continue
        summary = summarize_metric(pose_npz[key])
        fractions = fraction_within(pose_npz[key], thresholds)
        if summary is None or fractions is None:
            continue
        fraction_terms = ", ".join(f"<= {thr:g}: {100.0 * frac:5.1f}%" for thr, frac in fractions.items())
        print(
            f"    {prefix}{label:<14s} "
            f"mean={summary['mean']:.4f}, "
            f"median={summary['median']:.4f}, "
            f"p90={summary['p90']:.4f}, "
            f"p95={summary['p95']:.4f}, "
            f"p99={summary['p99']:.4f}, "
            f"max={summary['max']:.4f} | {fraction_terms}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--recovar_dir", required=True)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument(
        "--relion_start_iter",
        type=int,
        default=0,
        help="RELION iteration that recovar's iter 0 corresponds to (e.g. 3 if started from RELION iter 3)",
    )
    parser.add_argument("--tol", type=float, default=0.05, help="Relative tolerance for green/yellow/red coloring")
    parser.add_argument("--shells", type=int, default=12, help="How many low-frequency shells to print per-shell")
    args = parser.parse_args()

    relion_dir = Path(args.relion_dir)
    recovar_dir = Path(args.recovar_dir)

    recovar = load_recovar(recovar_dir / "refinement_results.npz")
    if recovar is None:
        logger.error("Missing %s/refinement_results.npz", recovar_dir)
        return 1

    voxel_size = float(recovar["voxel_size"])
    grid_size = int(recovar["volume_shape"][0])
    recovar_iter_count = int(len(recovar["current_sizes"])) if "current_sizes" in recovar.files else 0
    logger.info("Loaded recovar npz: %d files, voxel_size=%.3f, grid=%d", len(recovar.files), voxel_size, grid_size)
    recovar_half1_particles = int(recovar["n_half1_particles"]) if "n_half1_particles" in recovar.files else None

    print(f"\n{BOLD}{'=' * 100}{RESET}")
    print(f"{BOLD}RELION  vs  recovar  per-iter parity diff{RESET}")
    print(f"  RELION dir : {relion_dir}")
    print(f"  recovar dir: {recovar_dir}")
    print(f"  voxel_size : {voxel_size} Å/px,   grid: {grid_size}")
    print(f"  legend     : {GREEN}match{RESET} | {YELLOW}small diff{RESET} | {RED}LARGE DIFF{RESET}")
    print(f"{BOLD}{'=' * 100}{RESET}\n")

    # Find which iters RELION actually wrote
    relion_iters = sorted(
        {int(p.stem.split("_it")[1].split("_")[0]) for p in relion_dir.glob("run_it*_optimiser.star")}
    )
    logger.info("RELION wrote iters: %s", relion_iters)

    relion_offset = args.relion_start_iter
    n_iters_to_check = min(args.max_iter, max(relion_iters) + 1, recovar_iter_count + 1)
    if args.max_iter > recovar_iter_count + 1:
        print(
            f"  note       : recovar emitted {recovar_iter_count} iteration rows; "
            f"showing RELION init + matched rows only (requested {args.max_iter})."
        )

    for it in range(n_iters_to_check):
        relion_it = it + relion_offset
        relion_iter = load_relion_iter(relion_dir, relion_it)
        rsc = extract_relion_scalars(relion_iter)
        rps = extract_relion_per_shell(relion_iter, half=1)

        # When --relion_start_iter=S, recovar iter 0 maps to RELION iter S+1
        # (RELION iter S is the init state that recovar loaded).
        recovar_iter_index = it - 1  # may be negative for it=0
        rec_sc = extract_recovar_scalars(recovar, recovar_iter_index) if recovar_iter_index >= 0 else {}
        rec_ps = extract_recovar_per_shell(recovar, recovar_iter_index) if recovar_iter_index >= 0 else None

        print(
            f"{BOLD}{CYAN}── RELION iter {relion_it} (recovar idx {recovar_iter_index}) ─────────────────────────────{RESET}"
        )

        if not rsc:
            print(f"  [no RELION optimiser.star at iter {it}, skipping]")
            continue

        relion_half1_particles = None
        if relion_iter.get("data") is not None:
            relion_df = (
                relion_iter["data"]["particles"] if isinstance(relion_iter["data"], dict) else relion_iter["data"]
            )
            if "rlnRandomSubset" in relion_df.columns:
                relion_half1_particles = int(np.sum(np.asarray(relion_df["rlnRandomSubset"]) == 1))
        particle_scale = (
            float(relion_half1_particles) / float(recovar_half1_particles)
            if relion_half1_particles is not None and recovar_half1_particles not in (None, 0)
            else 1.0
        )

        # ---- Scalar comparison table ----
        print(f"  {'field':<28s} {'RELION':>16s}  {'recovar':>16s}")

        scalars_to_compare = [
            ("current_size", rsc.get("current_size"), rec_sc.get("current_size")),
            # ave_Pmax compares the per-particle mean of
            # rlnMaxValueProbDistribution from RELION's data.star against
            # recovar's per-particle E-step Pmax mean. This is the
            # apples-to-apples row.
            ("ave_Pmax", rsc.get("ave_Pmax"), rec_sc.get("ave_Pmax")),
            # _rlnAveragePmax from model.star is the M-step accumulator.
            # It differs from the per-particle mean by ~3% for half-set/
            # full-set accounting reasons; reported here for completeness
            # but NOT directly comparable to recovar's ave_Pmax.
            ("ave_Pmax_mstep (RELION-only)", rsc.get("ave_Pmax_mstep"), None),
            ("sigma_offsets_Å", rsc.get("sigma_offsets_angst"), rec_sc.get("sigma_offsets_angst")),
            ("sigma_offsets_used_Å", None, rec_sc.get("sigma_offsets_used_angst")),
            ("smallest_chg_angles_°", rsc.get("smallest_change_angles"), rec_sc.get("smallest_change_angles")),
            ("smallest_chg_offsets", rsc.get("smallest_change_offsets"), rec_sc.get("smallest_change_offsets")),
            ("current_resolution Å", rsc.get("current_resolution"), None),
            ("healpix_order", None, rec_sc.get("healpix_order")),
            ("frac_changed", None, rec_sc.get("fraction_changed")),
            ("acc_rot_°", None, rec_sc.get("acc_rot")),
        ]

        for label, rv, vv in scalars_to_compare:
            color = color_diff(rv, vv, tol=args.tol)
            print(f"  {label:<28s} {fmt(rv, 16):>16s}  {color}{fmt(vv, 16):>16s}{RESET}")
        if particle_scale != 1.0:
            print(
                f"  {'halfset particle scale':<28s} {fmt(relion_half1_particles, 16):>16s}  "
                f"{fmt(recovar_half1_particles, 16):>16s}  "
                f"(RELION tau2/sigma2 scaled by {particle_scale:.3f})"
            )

        # RELION-only state (per-iter)
        print(f"  {'RELION-only state:':<28s}")
        for f, label in [
            ("log_likelihood", "log_likelihood"),
            ("norm_correction_avg", "norm_correction_avg"),
            ("sigma_offsets_angst", "sigma_offsets_Å"),
            ("best_resolution_so_far", "best_res_so_far_(1/Å)"),
            ("smallest_change_angles", "smallest_chg_angles_°"),
            ("smallest_change_offsets", "smallest_chg_offsets_px"),
            ("n_iter_no_resolution_gain", "n_iter_no_res_gain"),
            ("has_high_fsc_at_limit", "has_high_fsc_at_limit"),
            ("has_converged", "has_converged"),
        ]:
            v = rsc.get(f)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                print(f"    {label:<26s} {fmt(v, 16):>16s}")

        # ---- Per-shell comparison ----
        if rps is not None and rps.get("_n_shells", 0) > 0:
            print(f"\n  {BOLD}per-shell (first {args.shells} shells; R=RELION half1, V=recoVar):{RESET}")
            header = (
                f"    {'shell':>4s}  {'res_Å':>6s}  "
                f"{'tau2_R':>10s} {'tau2_V':>10s}  "
                f"{'sig2_R':>10s} {'sig2_V':>10s}  "
                f"{'FSC_R':>7s} {'FSC_V':>7s}  "
                f"{'SSNR_R':>9s} {'SSNR_V':>9s}"
            )
            print(header)
            n_shells = min(args.shells, rps.get("_n_shells", 0))
            res = rps.get("AngstromResolution")
            tau2_r = rps.get("ReferenceTau2")
            sigma2_r = rps.get("ReferenceSigma2")
            fsc_r = rps.get("GoldStandardFsc")
            ssnr_r = rps.get("SsnrMap")
            tau2_v = rec_ps.get("ReferenceTau2") if rec_ps else None
            sigma2_v = rec_ps.get("ReferenceSigma2") if rec_ps else None
            fsc_v = rec_ps.get("FSC_gold_std") if rec_ps else None
            ssnr_v = rec_ps.get("SsnrMap") if rec_ps else None
            n4 = grid_size**4
            for s in range(n_shells):
                r = float(res[s]) if res is not None else None
                tr = float(tau2_r[s]) * n4 * particle_scale if tau2_r is not None else None
                tv = float(tau2_v[s]) if tau2_v is not None and s < len(tau2_v) else None
                sr = float(sigma2_r[s]) * n4 * particle_scale if sigma2_r is not None and s < len(sigma2_r) else None
                sv = float(sigma2_v[s]) if sigma2_v is not None and s < len(sigma2_v) else None
                f1 = float(fsc_r[s]) if fsc_r is not None else None
                f2 = float(fsc_v[s]) if fsc_v is not None and s < len(fsc_v) else None
                ssr = float(ssnr_r[s]) if ssnr_r is not None and s < len(ssnr_r) else None
                ssv = float(ssnr_v[s]) if ssnr_v is not None and s < len(ssnr_v) else None
                tcol = color_diff(tr, tv, tol=args.tol)
                scol = color_diff(sr, sv, tol=args.tol)
                fcol = color_diff(f1, f2, tol=args.tol)
                sscol = color_diff(ssr, ssv, tol=args.tol)
                print(
                    f"    {s:>4d}  {fmt(r, 6, prec=1):>6s}  "
                    f"{fmt(tr, 10):>10s} {tcol}{fmt(tv, 10):>10s}{RESET}  "
                    f"{fmt(sr, 10):>10s} {scol}{fmt(sv, 10):>10s}{RESET}  "
                    f"{fmt(f1, 7, prec=3):>7s} {fcol}{fmt(f2, 7, prec=3):>7s}{RESET}  "
                    f"{fmt(ssr, 9, prec=3):>9s} {sscol}{fmt(ssv, 9, prec=3):>9s}{RESET}"
                )

        if recovar_iter_index >= 0:
            pose_path = recovar_dir / f"pose_comparison_iter{recovar_iter_index:03d}.npz"
            if pose_path.exists():
                pose = np.load(pose_path, allow_pickle=False)
                print(f"\n  {BOLD}pose refinement metrics:{RESET}")
                pose_specs = [
                    ("angular_error_deg", "full_angle_°", [5, 10, 20]),
                    ("view_direction_error_deg", "view_dir_°", [2, 5, 10]),
                    ("inplane_error_deg", "in_plane_°", [2, 5, 10]),
                    ("translation_error_px", "trans_px", [0.25, 0.5, 1.0]),
                ]
                print_metric_block("", pose, pose_specs)
                gt_pose_specs = [
                    ("recovar_vs_gt_angular_error_deg", "rec_gt_full_°", [2, 5, 10]),
                    ("recovar_vs_gt_view_direction_error_deg", "rec_gt_view_°", [2, 5, 10]),
                    ("recovar_vs_gt_inplane_error_deg", "rec_gt_psi_°", [2, 5, 10]),
                    ("relion_vs_gt_angular_error_deg", "rel_gt_full_°", [2, 5, 10]),
                    ("relion_vs_gt_view_direction_error_deg", "rel_gt_view_°", [2, 5, 10]),
                    ("relion_vs_gt_inplane_error_deg", "rel_gt_psi_°", [2, 5, 10]),
                ]
                if any(key in pose.files for key, _, _ in gt_pose_specs):
                    print(f"\n  {BOLD}pose accuracy vs GT:{RESET}")
                    print_metric_block("", pose, gt_pose_specs)

            gt_metrics = load_saved_gt_metrics(recovar_dir, recovar_iter_index)
            if gt_metrics is not None:
                print(f"\n  {BOLD}map quality vs GT:{RESET}")
                print(f"    {'series':<18s} {'corr_vs_gt':>12s} {'FSC<0.5':>10s} {'FSC<0.143':>10s}")
                gt_rows = [
                    ("recovar_reg", "recovar_reg_merged"),
                    ("RELION", "relion_merged"),
                    ("recovar_unreg", "recovar_unreg_merged"),
                ]
                rel_corr = (
                    float(gt_metrics["relion_merged_corr_vs_gt"])
                    if "relion_merged_corr_vs_gt" in gt_metrics.files
                    else None
                )
                rel_shell_05 = (
                    int(gt_metrics["relion_merged_shell_05"]) if "relion_merged_shell_05" in gt_metrics.files else None
                )
                rel_shell_0143 = (
                    int(gt_metrics["relion_merged_shell_0143"])
                    if "relion_merged_shell_0143" in gt_metrics.files
                    else None
                )
                for label, prefix in gt_rows:
                    corr_key = f"{prefix}_corr_vs_gt"
                    shell05_key = f"{prefix}_shell_05"
                    shell0143_key = f"{prefix}_shell_0143"
                    if corr_key not in gt_metrics.files:
                        continue
                    corr_v = float(gt_metrics[corr_key])
                    shell05_v = int(gt_metrics[shell05_key])
                    shell0143_v = int(gt_metrics[shell0143_key])
                    ccol = color_diff(rel_corr, corr_v, tol=args.tol) if label != "RELION" else GREEN
                    s05col = color_diff(rel_shell_05, shell05_v, tol=args.tol) if label != "RELION" else GREEN
                    s143col = color_diff(rel_shell_0143, shell0143_v, tol=args.tol) if label != "RELION" else GREEN
                    print(
                        f"    {label:<18s} "
                        f"{ccol}{corr_v:12.6f}{RESET} "
                        f"{s05col}{shell05_v:10d}{RESET} "
                        f"{s143col}{shell0143_v:10d}{RESET}"
                    )
                if "recovar_reg_merged_aligned_corr_vs_gt" in gt_metrics.files:
                    print(
                        f"\n    {'aligned series':<18s} {'corr_vs_gt':>12s} {'FSC<0.5':>10s} "
                        f"{'FSC<0.143':>10s} {'rot':>6s} {'mirror':>7s} {'sign':>5s}"
                    )
                    for label, prefix in gt_rows:
                        corr_key = f"{prefix}_aligned_corr_vs_gt"
                        shell05_key = f"{prefix}_aligned_shell_05"
                        shell0143_key = f"{prefix}_aligned_shell_0143"
                        rot_key = f"{prefix}_gt_align_rotation_index"
                        mirror_key = f"{prefix}_gt_align_mirror_x"
                        sign_key = f"{prefix}_gt_align_sign"
                        if corr_key not in gt_metrics.files:
                            continue
                        print(
                            f"    {label:<18s} "
                            f"{float(gt_metrics[corr_key]):12.6f} "
                            f"{int(gt_metrics[shell05_key]):10d} "
                            f"{int(gt_metrics[shell0143_key]):10d} "
                            f"{int(gt_metrics[rot_key]):6d} "
                            f"{str(bool(gt_metrics[mirror_key])):>7s} "
                            f"{int(gt_metrics[sign_key]):5d}"
                        )
                if "recovar_reg_merged_corr_vs_relion" in gt_metrics.files:
                    print(
                        f"    {'recovar-vs-RELION':<18s} "
                        f"corr={float(gt_metrics['recovar_reg_merged_corr_vs_relion']):.6f}"
                    )

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
