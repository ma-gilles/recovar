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
  - ave_Pmax     (RELION's optimizer/scheduling rlnAveragePmax from model.star)
  - ave_Pmax_particles (mean of data.star::rlnMaxValueProbDistribution,
                        retained as a separate distribution diagnostic)
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


def relion_star_round_double(value):
    """Round a double exactly as RELION's MetaDataTable STAR writer does.

    This mirrors ``metadata_table.cpp``: positive values get six fractional
    digits and negative values get five so the formatted field remains 12
    characters wide.  Very small nonzero and very large values use scientific
    notation.
    """

    value = float(value)
    magnitude = abs(value)
    scientific = (0.0 < magnitude < 0.001) or magnitude > 100000.0
    precision = 5 if value < 0.0 else 6
    format_code = "e" if scientific else "f"
    return float(f"{value:12.{precision}{format_code}}")


def relion_star_round_scaled(value, scale):
    """Round a scaled RECOVAR value in RELION-native units, then rescale it."""

    if value is None:
        return None
    native_value = float(value) / float(scale)
    return relion_star_round_double(native_value) * float(scale)


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


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _model_general(model):
    if model and "model_general" in model:
        return model["model_general"]
    return None


def _sigma_offset_from_model(model):
    mg = _model_general(model)
    if mg is None:
        return float("nan")
    return _safe_float(mg.get("rlnSigmaOffsetsAngst", float("nan")))


def load_relion_iter(relion_dir, it, run_prefix="run"):
    """Load all per-iter STARs for one RELION iteration."""
    nnn = f"{it:03d}"
    out = {}
    out["optimiser"] = parse_relion_optimiser(relion_dir / f"{run_prefix}_it{nnn}_optimiser.star")
    out["model_h1"] = parse_relion_model(relion_dir / f"{run_prefix}_it{nnn}_half1_model.star")
    out["model_h2"] = parse_relion_model(relion_dir / f"{run_prefix}_it{nnn}_half2_model.star")
    if out["model_h1"] is None and out["model_h2"] is None:
        # 3D classification writes one run_itNNN_model.star instead of
        # auto-refine half-model STARs. Use it for K-class diagnostics.
        model = parse_relion_model(relion_dir / f"{run_prefix}_it{nnn}_model.star")
        out["model_h1"] = model
        out["model_h2"] = model
    data_path = relion_dir / f"{run_prefix}_it{nnn}_data.star"
    out["data"] = starfile.read(str(data_path)) if data_path.exists() else None
    return out


def extract_relion_scalars(relion_iter):
    """Extract per-iter scalar values from RELION's star files.

    Per-iter scalars live in THREE places:
      - model.star::model_general — current_size, current_resolution,
        sigma_offsets, log_likelihood, norm_correction, _rlnAveragePmax
        (the optimizer scalar used for scheduling).
      - data.star — per-particle ``rlnMaxValueProbDistribution`` column.
        Its mean is retained as a separate distribution diagnostic.
      - optimiser.star — smallest_changes (convergence indicators), iter counters

    NOTE on ``ave_Pmax``:
        RELION computes ``_rlnAveragePmax`` from the weighted-sum model and
        consumes it for current-size scheduling. It is therefore the parity
        oracle for recovar's optimizer state. The arithmetic particle-column
        mean is not interchangeable with it and is reported under
        ``ave_Pmax_particles`` only.
    """
    out = {}
    opt = relion_iter["optimiser"]
    model = relion_iter["model_h1"]
    model_h2 = relion_iter["model_h2"]
    data = relion_iter.get("data")

    # Preserve the particle-column mean as a distribution diagnostic.
    relion_data_df = None
    if data is not None:
        relion_data_df = data["particles"] if isinstance(data, dict) and "particles" in data else data
    if relion_data_df is not None and "rlnMaxValueProbDistribution" in relion_data_df:
        col = np.asarray(relion_data_df["rlnMaxValueProbDistribution"], dtype=np.float64)
        if col.size:
            out["ave_Pmax_particles"] = float(col.mean())

    # From model_general (the per-iter "state" block)
    if model and "model_general" in model:
        mg = model["model_general"]
        out["current_size"] = int(mg.get("rlnCurrentImageSize", 0) or 0)
        out["ave_Pmax"] = float(mg.get("rlnAveragePmax", float("nan")))
        # Backward-compatible alias for older consumers. Despite the historic
        # name, this is the authoritative optimizer/scheduling scalar.
        out["ave_Pmax_mstep"] = out["ave_Pmax"]
        out["current_resolution"] = float(mg.get("rlnCurrentResolution", float("nan")))
        out["log_likelihood"] = float(mg.get("rlnLogLikelihood", float("nan")))
        out["norm_correction_avg"] = float(mg.get("rlnNormCorrectionAverage", float("nan")))
        sigma_h1 = _sigma_offset_from_model(model)
        sigma_h2 = _sigma_offset_from_model(model_h2)
        if np.isfinite(sigma_h1):
            out["sigma_offsets_h1_angst"] = sigma_h1
        if np.isfinite(sigma_h2):
            out["sigma_offsets_h2_angst"] = sigma_h2
        sigma_pair = [v for v in (sigma_h1, sigma_h2) if np.isfinite(v)]
        out["sigma_offsets_angst"] = float(np.mean(sigma_pair)) if sigma_pair else float("nan")
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


def extract_relion_direction_prior(relion_iter, half):
    """Extract one half's class-1 RELION HEALPix direction prior."""

    model = relion_iter[f"model_h{half}"]
    if model is None or "model_pdf_orient_class_1" not in model:
        return None
    table = model["model_pdf_orient_class_1"]
    if "rlnOrientationDistribution" not in table.columns:
        return None
    return np.asarray(table["rlnOrientationDistribution"], dtype=np.float64)


def load_recovar(npz_path):
    if not npz_path.exists():
        return None
    # K-class refinement outputs may include object-scalar None placeholders
    # for metrics that are not defined at a given iteration.
    return np.load(npz_path, allow_pickle=True)


def _recovar_per_shell_array(recovar, key):
    """Return a numeric 1-D recovar per-shell array, or None if unavailable.

    For K-class outputs, arrays are stored as (K, n_shells). This diagnostic
    script compares against RELION's model_class_1 table, so use class 1.
    """
    if key not in recovar.files:
        return None
    arr = np.asarray(recovar[key])
    if arr.dtype == object:
        if arr.shape == () and arr.item() is None:
            return None
        try:
            arr = np.asarray(arr.item())
        except ValueError:
            return None
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        arr = arr[0]
    return arr


def _recovar_optional_pair(recovar, key, it):
    if recovar is None or key not in recovar.files:
        return None
    arr = np.asarray(recovar[key], dtype=object)
    if it >= len(arr):
        return None
    value = arr[it]
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == () and value.item() is None:
        return None
    try:
        pair = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pair.size != 2 or not np.all(np.isfinite(pair)):
        return None
    return [float(pair[0]), float(pair[1])]


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
    sigma_offset_per_half = _recovar_optional_pair(recovar, "sigma_offset_per_half_trajectory", it)
    sigma_offset_used_per_half = _recovar_optional_pair(recovar, "sigma_offset_used_per_half_trajectory", it)
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
    pmax_particles_key = f"pmax_per_image_iter_{it:03d}"
    if pmax_particles_key in recovar.files:
        pmax_particles = np.asarray(recovar[pmax_particles_key], dtype=np.float64)
        if pmax_particles.size:
            out["ave_Pmax_particles"] = float(np.mean(pmax_particles))
    if pr_arr is not None and it < len(pr_arr):
        out["current_resolution_pix"] = int(pr_arr[it])
    if hpx_arr is not None and it < len(hpx_arr):
        out["healpix_order"] = int(hpx_arr[it])
    if sigma_offset_arr is not None and it < len(sigma_offset_arr):
        out["sigma_offsets_angst"] = float(sigma_offset_arr[it])
    if sigma_offset_per_half is not None:
        out["sigma_offsets_h1_angst"] = sigma_offset_per_half[0]
        out["sigma_offsets_h2_angst"] = sigma_offset_per_half[1]
    if sigma_offset_used_arr is not None and it < len(sigma_offset_used_arr):
        out["sigma_offsets_used_angst"] = float(sigma_offset_used_arr[it])
    if sigma_offset_used_per_half is not None:
        out["sigma_offsets_used_h1_angst"] = sigma_offset_used_per_half[0]
        out["sigma_offsets_used_h2_angst"] = sigma_offset_used_per_half[1]
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
    fsc = _recovar_per_shell_array(recovar, fsc_key)
    if fsc is not None:
        out["FSC_gold_std"] = fsc
    dvp = _recovar_per_shell_array(recovar, dvp_key)
    if dvp is not None:
        out["DataVsPriorRatio"] = dvp
    sig_counts = _recovar_per_shell_array(recovar, sig_key)
    if sig_counts is not None:
        out["_sig_counts"] = sig_counts
    noise = _recovar_per_shell_array(recovar, noise_key)
    if noise is not None:
        out["Sigma2Noise"] = noise
    tau2 = _recovar_per_shell_array(recovar, tau2_key)
    if tau2 is not None:
        out["ReferenceTau2"] = tau2
    sigma2 = _recovar_per_shell_array(recovar, sigma2_key)
    if sigma2 is not None:
        out["ReferenceSigma2"] = sigma2
    tau2_fsc = _recovar_per_shell_array(recovar, tau2_fsc_key)
    if tau2_fsc is not None:
        out["Tau2FscUsed"] = tau2_fsc
        out["FSC_gold_std"] = out["Tau2FscUsed"]
    ssnr = _recovar_per_shell_array(recovar, ssnr_key)
    if ssnr is not None:
        out["SsnrMap"] = ssnr
        out["DataVsPriorRatio"] = out["SsnrMap"]
    return out if out else None


def extract_recovar_direction_prior(recovar, it):
    """Return the saved [half1, half2] direction-prior snapshot for iteration ``it``."""

    if recovar is None or "direction_prior_trajectory_per_half" not in recovar.files:
        return None, None
    trajectory = recovar["direction_prior_trajectory_per_half"]
    if it < 0 or it >= len(trajectory) or trajectory[it] is None:
        return None, None
    entry = trajectory[it]
    return tuple(
        None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)
        for value in entry
    )


def compare_direction_priors(relion_arr, recovar_arr):
    """Return direct array-error diagnostics; correlation is auxiliary only."""

    if relion_arr is None or recovar_arr is None:
        return None
    relion_arr = np.asarray(relion_arr, dtype=np.float64).reshape(-1)
    recovar_arr = np.asarray(recovar_arr, dtype=np.float64).reshape(-1)
    if relion_arr.shape != recovar_arr.shape:
        return {"mismatch": True, "n_relion": int(relion_arr.size), "n_recovar": int(recovar_arr.size)}
    difference = recovar_arr - relion_arr
    l1_diff = float(np.sum(np.abs(difference)))
    relion_l1 = float(np.sum(np.abs(relion_arr)))
    corr = float("nan")
    if relion_arr.size > 1 and np.std(relion_arr) > 0 and np.std(recovar_arr) > 0:
        corr = float(np.corrcoef(relion_arr, recovar_arr)[0, 1])
    return {
        "mismatch": False,
        "n_directions": int(relion_arr.size),
        "max_abs_diff": float(np.max(np.abs(difference), initial=0.0)),
        "l1_diff": l1_diff,
        "relative_l1_diff": float(l1_diff / max(relion_l1, np.finfo(np.float64).tiny)),
        "mass_diff": float(np.sum(recovar_arr) - np.sum(relion_arr)),
        "corr_auxiliary": corr,
    }


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


def _load_current_npz_artifact(path, refinement_results_path, *, label):
    """Ignore cached diagnostics that predate the refinement they describe."""
    if not path.exists():
        return None
    if refinement_results_path.exists() and path.stat().st_mtime_ns < refinement_results_path.stat().st_mtime_ns:
        logger.warning(
            "Ignoring stale %s %s (older than %s)",
            label,
            path,
            refinement_results_path,
        )
        return None
    return np.load(path, allow_pickle=False)


def load_saved_gt_metrics(recovar_dir, it, refinement_results_path=None):
    path = recovar_dir / f"gt_comparison_iter{it:03d}.npz"
    if refinement_results_path is None:
        refinement_results_path = recovar_dir / "refinement_results.npz"
    return _load_current_npz_artifact(
        path,
        refinement_results_path,
        label="GT-comparison artifact",
    )


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
    parser.add_argument(
        "--relion_run_prefix",
        default="run",
        help="RELION output prefix inside --relion_dir (default: run)",
    )
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

    refinement_results_path = recovar_dir / "refinement_results.npz"
    recovar = load_recovar(refinement_results_path)
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
        {
            int(p.stem.split("_it")[1].split("_")[0])
            for p in relion_dir.glob(f"{args.relion_run_prefix}_it*_optimiser.star")
        }
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
        relion_iter = load_relion_iter(relion_dir, relion_it, args.relion_run_prefix)
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
            ("ave_Pmax_optimizer", rsc.get("ave_Pmax"), rec_sc.get("ave_Pmax")),
            (
                "ave_Pmax_particles",
                rsc.get("ave_Pmax_particles"),
                rec_sc.get("ave_Pmax_particles"),
            ),
            ("sigma_offsets_mean_Å", rsc.get("sigma_offsets_angst"), rec_sc.get("sigma_offsets_angst")),
            ("sigma_offsets_h1_Å", rsc.get("sigma_offsets_h1_angst"), rec_sc.get("sigma_offsets_h1_angst")),
            ("sigma_offsets_h2_Å", rsc.get("sigma_offsets_h2_angst"), rec_sc.get("sigma_offsets_h2_angst")),
            ("sigma_offsets_used_mean_Å", None, rec_sc.get("sigma_offsets_used_angst")),
            ("sigma_offsets_used_h1_Å", None, rec_sc.get("sigma_offsets_used_h1_angst")),
            ("sigma_offsets_used_h2_Å", None, rec_sc.get("sigma_offsets_used_h2_angst")),
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
            ("sigma_offsets_angst", "sigma_offsets_mean_Å"),
            ("sigma_offsets_h1_angst", "sigma_offsets_h1_Å"),
            ("sigma_offsets_h2_angst", "sigma_offsets_h2_Å"),
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

        recovar_dir_h1, recovar_dir_h2 = (
            extract_recovar_direction_prior(recovar, recovar_iter_index)
            if recovar_iter_index >= 0
            else (None, None)
        )
        direction_prior_rows = [
            ("h1", extract_relion_direction_prior(relion_iter, half=1), recovar_dir_h1),
            ("h2", extract_relion_direction_prior(relion_iter, half=2), recovar_dir_h2),
        ]
        if any(lhs is not None or rhs is not None for _, lhs, rhs in direction_prior_rows):
            print(f"  {'pdf_orient direct array diff:':<28s}")
            for label, relion_arr, recovar_arr in direction_prior_rows:
                stats = compare_direction_priors(relion_arr, recovar_arr)
                if stats is None:
                    print(f"    {label:<26s} {'—':>16s}  (missing on one side)")
                elif stats["mismatch"]:
                    print(
                        f"    {label:<26s} HEALPix-size mismatch "
                        f"(RELION n={stats['n_relion']}, RECOVAR n={stats['n_recovar']})"
                    )
                else:
                    color = GREEN if stats["relative_l1_diff"] <= args.tol else RED
                    print(
                        f"    {label:<26s} n={stats['n_directions']:<6d} "
                        f"relative_L1={color}{stats['relative_l1_diff']:.3e}{RESET}  "
                        f"max_abs={stats['max_abs_diff']:.3e}  L1={stats['l1_diff']:.3e}  "
                        f"mass_diff={stats['mass_diff']:.3e}  "
                        f"corr={stats['corr_auxiliary']:.6f} (aux only)"
                    )

        # ---- Per-shell comparison ----
        if rps is not None and rps.get("_n_shells", 0) > 0:
            print(f"\n  {BOLD}per-shell (first {args.shells} shells; R=RELION half1, V=recoVar):{RESET}")
            header = (
                f"    {'shell':>4s}  {'res_Å':>6s}  "
                f"{'tau2_R':>10s} {'tau2_V':>10s} {'tau2_Vstar':>10s}  "
                f"{'sig2_R':>10s} {'sig2_V':>10s} {'sig2_Vstar':>10s}  "
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
                tv_star = relion_star_round_scaled(tv, n4 * particle_scale)
                sr = float(sigma2_r[s]) * n4 * particle_scale if sigma2_r is not None and s < len(sigma2_r) else None
                sv = float(sigma2_v[s]) if sigma2_v is not None and s < len(sigma2_v) else None
                sv_star = relion_star_round_scaled(sv, n4 * particle_scale)
                f1 = float(fsc_r[s]) if fsc_r is not None else None
                f2 = float(fsc_v[s]) if fsc_v is not None and s < len(fsc_v) else None
                ssr = float(ssnr_r[s]) if ssnr_r is not None and s < len(ssnr_r) else None
                ssv = float(ssnr_v[s]) if ssnr_v is not None and s < len(ssnr_v) else None
                tcol = color_diff(tr, tv, tol=args.tol)
                tstar_col = color_diff(tr, tv_star, tol=args.tol)
                scol = color_diff(sr, sv, tol=args.tol)
                sstar_col = color_diff(sr, sv_star, tol=args.tol)
                fcol = color_diff(f1, f2, tol=args.tol)
                sscol = color_diff(ssr, ssv, tol=args.tol)
                print(
                    f"    {s:>4d}  {fmt(r, 6, prec=1):>6s}  "
                    f"{fmt(tr, 10):>10s} {tcol}{fmt(tv, 10):>10s}{RESET} "
                    f"{tstar_col}{fmt(tv_star, 10):>10s}{RESET}  "
                    f"{fmt(sr, 10):>10s} {scol}{fmt(sv, 10):>10s}{RESET} "
                    f"{sstar_col}{fmt(sv_star, 10):>10s}{RESET}  "
                    f"{fmt(f1, 7, prec=3):>7s} {fcol}{fmt(f2, 7, prec=3):>7s}{RESET}  "
                    f"{fmt(ssr, 9, prec=3):>9s} {sscol}{fmt(ssv, 9, prec=3):>9s}{RESET}"
                )

        if recovar_iter_index >= 0:
            pose_path = recovar_dir / f"pose_comparison_iter{recovar_iter_index:03d}.npz"
            pose = _load_current_npz_artifact(
                pose_path,
                refinement_results_path,
                label="pose-comparison artifact",
            )
            if pose is not None:
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

            gt_metrics = load_saved_gt_metrics(
                recovar_dir,
                recovar_iter_index,
                refinement_results_path,
            )
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
