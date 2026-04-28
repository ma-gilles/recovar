#!/usr/bin/env python
"""Run N iterations of recovar in RELION mode, save results for diff comparison.

Usage:
  pixi run python scripts/run_multi_iter_parity.py \
    --relion_dir .../relion_ref_os0 \
    --data_star .../particles.star \
    --iter 3 --max_iter 15 \
    --output_dir .../recovar_15iter
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)


def stack_index_from_image_name(name: str) -> int:
    """Return the zero-based stack row encoded in a RELION image name."""
    m = re.match(r"(\d+)@", str(name))
    return int(m.group(1)) - 1 if m else -1


def replay_previous_relion_iteration(init_relion_iteration: int, recovar_iteration: int) -> int:
    """Return the RELION iteration whose particle metadata seeds this replay step."""
    return int(init_relion_iteration) + int(recovar_iteration)


def replay_control_relion_iteration(init_relion_iteration: int, recovar_iteration: int) -> int:
    """Return the RELION iteration whose control variables govern this replay step."""
    return replay_previous_relion_iteration(init_relion_iteration, recovar_iteration) + 1


def map_pose_arrays_to_particle_order(our_names, gt_rot_all, gt_trans_all=None):
    """Map pose arrays indexed by stack row onto the current particle ordering."""
    n_total = len(our_names)
    gt_rotations_orig = np.full((n_total, 3, 3), np.nan, dtype=np.float64)
    gt_translations_orig = np.full((n_total, 2), np.nan, dtype=np.float64) if gt_trans_all is not None else None
    for j, name in enumerate(our_names):
        pose_idx = stack_index_from_image_name(name)
        if 0 <= pose_idx < len(gt_rot_all):
            gt_rotations_orig[j] = gt_rot_all[pose_idx]
        if gt_translations_orig is not None and 0 <= pose_idx < len(gt_trans_all):
            gt_translations_orig[j] = gt_trans_all[pose_idx]
    return gt_rotations_orig, gt_translations_orig


def _safe_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip() or None
        )
    except Exception:
        return None


def _count_compile_lines(log_path):
    if log_path is None:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    return sum("Compiling" in line for line in text.splitlines())


def _collect_local_profile_rows(save_intermediates_dir):
    rows = []
    scalar_keys = [
        "n_chunks",
        "em_time_s",
        "accounted_em_time_s",
        "unattributed_em_time_s",
        "sum_union_rows",
        "sum_padded_rows",
        "sum_nonzero_posterior_rows",
        "sum_reconstruction_rows",
        "sum_significant_samples",
        "unique_global_rotations",
        "unique_nonzero_global_rotations",
        "unique_reconstruction_global_rotations",
        "duplicate_rotation_factor",
        "reconstruction_duplicate_rotation_factor",
        "sum_union_row_pixels",
        "adjoint_seconds_per_row_pixel",
        "union_waste_fraction",
        "padded_waste_fraction",
        "padding_only_waste_fraction",
        "materialize_projection_abs2",
        "preprocess_time_s",
        "preprocess_integer_shift_s",
        "preprocess_translation_phase_s",
        "preprocess_processed_cache_gather_s",
        "preprocess_combined_process_s",
        "preprocess_score_process_s",
        "preprocess_recon_process_s",
        "preprocess_ctf_s",
        "preprocess_tile_shift_score_s",
        "preprocess_tile_shift_recon_s",
        "preprocess_norm_s",
        "projection_time_s",
        "fused_score_mstep_s",
        "local_score_s",
        "local_normalize_s",
        "local_significance_s",
        "local_mstep_s",
        "local_pack_s",
        "local_backproject_y_s",
        "local_backproject_ctf_s",
        "local_noise_s",
        "local_postprocess_s",
        "local_host_stats_s",
        "local_final_accumulator_s",
        "local_stats_finalize_s",
        "selector_time_s",
        "metadata_build_time_s",
        "translation_prior_time_s",
        "raw_cache_build_time_s",
        "bucket_build_time_s",
        "batch_fetch_time_s",
        "processed_cache_build_time_s",
        "transfer_total_to_host_s",
        "transfer_reconstruction_mask_to_host_s",
        "transfer_mstep_posterior_sum_to_host_s",
        "transfer_postprocess_argmax_to_host_s",
        "transfer_postprocess_scores_to_host_s",
        "transfer_postprocess_posterior_to_host_s",
        "transfer_final_noise_to_host_s",
        "local_total_hypotheses",
        "local_mean_rotations_per_image",
        "local_mean_significant_samples_per_image",
        "local_mean_reconstruction_rows_per_image",
        "local_num_buckets",
        "local_pad_fraction",
        "max_hypotheses_per_microbatch",
        "n_windowed",
        "compact_zero_posterior_rows",
        "native_half_preprocess",
        "native_half_preprocess_mode",
        "combined_masked_preprocess",
        "fused_score_mstep_enabled",
        "raw_cache_enabled",
        "processed_cache_enabled",
    ]
    for npz_path in sorted(Path(save_intermediates_dir).glob("*_local_profile.npz")):
        with np.load(npz_path) as profile_npz:
            row = {"path": str(npz_path)}
            for key in scalar_keys:
                if key in profile_npz:
                    value = profile_npz[key]
                    row[key] = value.item() if np.ndim(value) == 0 else np.asarray(value).tolist()
            rows.append(row)
    return rows


def _profile_value_to_jsonable(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _collect_local_profile_history(result):
    return [
        {key: _profile_value_to_jsonable(value) for key, value in row.items()}
        for row in result.get("local_profile_history", [])
    ]


def _summarize_local_profile_rows(rows, wall_times):
    """Aggregate exact-local profile rows for timing ledgers."""
    if not rows:
        return {}
    sum_keys = [
        "em_time_s",
        "accounted_em_time_s",
        "unattributed_em_time_s",
        "preprocess_time_s",
        "projection_time_s",
        "fused_score_mstep_s",
        "local_score_s",
        "local_normalize_s",
        "local_significance_s",
        "local_mstep_s",
        "local_pack_s",
        "local_backproject_y_s",
        "local_backproject_ctf_s",
        "local_noise_s",
        "local_postprocess_s",
        "local_host_stats_s",
        "local_final_accumulator_s",
        "local_stats_finalize_s",
        "selector_time_s",
        "raw_cache_build_time_s",
        "bucket_build_time_s",
        "batch_fetch_time_s",
        "processed_cache_build_time_s",
        "transfer_total_to_host_s",
    ]
    summary = {
        "n_profile_rows": len(rows),
        "sum_wall_times_s": float(np.sum(np.asarray(wall_times, dtype=np.float64))) if wall_times else None,
    }
    for key in sum_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"sum_{key}"] = float(np.sum(values))
    if summary["sum_wall_times_s"] is not None and "sum_em_time_s" in summary:
        summary["wall_minus_exact_local_s"] = float(summary["sum_wall_times_s"] - summary["sum_em_time_s"])
    if "sum_em_time_s" in summary and "sum_accounted_em_time_s" in summary:
        summary["exact_local_unaccounted_check_s"] = float(
            summary["sum_em_time_s"] - summary["sum_accounted_em_time_s"]
        )
    for key in (
        "native_half_preprocess",
        "native_half_preprocess_mode",
        "materialize_projection_abs2",
        "fused_score_mstep_enabled",
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = values[0]
    return summary


def _read_relion_pmax_column(relion_df):
    """Return RELION per-particle Pmax values when available.

    Older benchmark directories do not always carry
    ``rlnMaxValueProbDistribution``. Keep those runs usable for timing and
    structural parity checks by treating the field as optional.
    """

    if "rlnMaxValueProbDistribution" not in relion_df:
        return None
    return np.array(relion_df["rlnMaxValueProbDistribution"], dtype=np.float64)


def parse_relion_optimiser_cli_flags(opt_text: str) -> dict[str, object]:
    """Extract selected CLI flags from RELION's optimiser STAR header."""
    cli_line = next(
        (line.lstrip("#").strip() for line in opt_text.splitlines() if line.lstrip().startswith("# --")),
        "",
    )
    ini_high_match = re.search(r"(?:^|\s)--ini_high\s+(\S+)", cli_line)
    return {
        "cli_line": cli_line,
        "do_firstiter_cc": bool(re.search(r"(?:^|\s)--firstiter_cc(?:\s|$)", cli_line)),
        "ini_high_angstrom": float(ini_high_match.group(1)) if ini_high_match else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--data_star", required=True)
    parser.add_argument("--iter", type=int, default=3, help="RELION iteration to start from")
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--save_intermediates_dir", type=str, default=None, help="Directory for manifest NPZ dumps (for replay)"
    )
    parser.add_argument("--max_healpix_order", type=int, default=8)
    parser.add_argument("--skip_final_iteration", action="store_true", help="Skip the final combined-data Nyquist iter")
    parser.add_argument(
        "--force_max_iter_after_convergence",
        action="store_true",
        help=(
            "Continue RECOVAR parity replay until --max_iter even if the RELION-mode "
            "convergence state has already converged. Use only for fixed-length diagnostics."
        ),
    )
    parser.add_argument(
        "--max_particles", type=int, default=None, help="Subsample to at most N particles (N/2 per half)"
    )
    parser.add_argument(
        "--keep_stack_indices",
        type=str,
        default=None,
        help=(
            "Comma/space-separated zero-based particle stack indices to keep. "
            "Use for focused RELION-vs-RECOVAR E-step score dumps."
        ),
    )
    parser.add_argument(
        "--gt_volume",
        type=str,
        default=None,
        help="Optional recovar-frame GT MRC for FSC/correlation checks. Defaults to sibling reference_gt.mrc if present.",
    )
    parser.add_argument(
        "--force_oversampling",
        type=int,
        default=None,
        help="Override RELION's adaptive oversampling order for debugging ablations.",
    )
    parser.add_argument(
        "--max_significants",
        type=int,
        default=None,
        help="Override RELION's maximum significant poses. Default: read _rlnMaximumSignificantPoses from optimiser.star.",
    )
    parser.add_argument(
        "--adaptive_fraction",
        type=float,
        default=None,
        help="Override RELION's adaptive oversample fraction. Default: read _rlnAdaptiveOversampleFraction from optimiser.star.",
    )
    parser.add_argument(
        "--local_search_profile",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control exact-local profile collection. 'auto' profiles only when intermediates are enabled.",
    )
    parser.add_argument(
        "--local_search_translation_prior_mode",
        choices=["perturbed", "coarse"],
        default="coarse",
        help="Evaluate local-search translation priors on the perturbed candidate grid or the unperturbed coarse RELION grid.",
    )
    parser.add_argument(
        "--first_iteration_score_mode",
        choices=["gaussian", "normalized_cc"],
        default="gaussian",
        help="Diagnostic override for the iter-0 score metric.",
    )
    parser.add_argument(
        "--first_iteration_reconstruction_mode",
        choices=["soft", "hard"],
        default="soft",
        help="Diagnostic override for the iter-0 reconstruction weights.",
    )
    parser.add_argument(
        "--relion_ini_high",
        type=float,
        default=None,
        help="Optional override for RELION --ini_high. Defaults to the optimiser flag value, or 30 A.",
    )
    parser.add_argument(
        "--disable_adjoint_y",
        action="store_true",
        help="Experimental ablation: disable weighted-image adjoint accumulation.",
    )
    parser.add_argument(
        "--disable_adjoint_ctf",
        action="store_true",
        help="Experimental ablation: disable CTF adjoint accumulation.",
    )
    parser.add_argument(
        "--benchmark_ledger_json",
        type=str,
        default=None,
        help="Optional JSON path for a machine-readable benchmark/perf ledger summary.",
    )
    parser.add_argument(
        "--timing_only",
        action="store_true",
        help=(
            "Run refinement and write only the benchmark ledger. Skips "
            "diagnostic volumes, per-particle comparisons, and diff scripts so "
            "wall time reflects refinement rather than audit I/O."
        ),
    )
    parser.add_argument(
        "--compile_log",
        type=str,
        default=None,
        help="Optional log path to scan for JAX compile lines when building the benchmark ledger.",
    )
    parser.add_argument(
        "--jax_cache_dir",
        type=str,
        default=None,
        help="Optional persistent JAX compilation cache directory for cross-process warm starts.",
    )
    args = parser.parse_args()

    if args.jax_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.jax_cache_dir)
        os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

    import jax
    import jax.numpy as jnp
    import jaxlib
    import starfile

    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset

    from recovar.em.dense_single_volume.iteration_loop import refine_single_volume
    from recovar.em.sampling import read_relion_sampling_metadata
    from recovar.output.output import save_volume
    from recovar.reconstruction import noise as recon_noise
    from recovar.reconstruction import regularization
    from recovar.utils import helpers

    def _rotation_matrices_from_eulers_deg(eulers_deg):
        return utils.R_from_relion(np.asarray(eulers_deg, dtype=np.float64))

    def _angular_distance_from_dots(dot_vals):
        return np.rad2deg(np.arccos(np.clip(np.asarray(dot_vals, dtype=np.float64), -1.0, 1.0)))

    def _angular_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_rot = np.asarray(lhs_rot, dtype=np.float64)
        rhs_rot = np.asarray(rhs_rot, dtype=np.float64)
        rdiff = np.einsum("nij,njk->nik", np.transpose(lhs_rot, (0, 2, 1)), rhs_rot)
        traces = np.trace(rdiff, axis1=1, axis2=2)
        return _angular_distance_from_dots((traces - 1.0) / 2.0)

    def _normalize_rows(vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return vectors / norms

    def _view_direction_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_view = _normalize_rows(np.asarray(lhs_rot, dtype=np.float64)[:, 2, :])
        rhs_view = _normalize_rows(np.asarray(rhs_rot, dtype=np.float64)[:, 2, :])
        return _angular_distance_from_dots(np.sum(lhs_view * rhs_view, axis=1))

    def _inplane_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_rot = np.asarray(lhs_rot, dtype=np.float64)
        rhs_rot = np.asarray(rhs_rot, dtype=np.float64)
        rhs_view = _normalize_rows(rhs_rot[:, 2, :])

        lhs_x = lhs_rot[:, 0, :]
        rhs_x = rhs_rot[:, 0, :]
        lhs_x = lhs_x - np.sum(lhs_x * rhs_view, axis=1, keepdims=True) * rhs_view
        rhs_x = rhs_x - np.sum(rhs_x * rhs_view, axis=1, keepdims=True) * rhs_view
        lhs_x = _normalize_rows(lhs_x)
        rhs_x = _normalize_rows(rhs_x)

        cross = np.cross(rhs_x, lhs_x)
        signed = np.rad2deg(
            np.arctan2(
                np.sum(rhs_view * cross, axis=1),
                np.sum(rhs_x * lhs_x, axis=1),
            )
        )
        return np.abs(signed)

    def _angular_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _angular_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _view_direction_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _view_direction_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _inplane_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _inplane_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _rotations_in_gt_frame_from_relion_eulers(eulers_deg, transpose_relion_convention):
        rot = _rotation_matrices_from_eulers_deg(eulers_deg)
        if transpose_relion_convention:
            rot = np.transpose(rot, (0, 2, 1))
        return rot

    def _format_error_summary(values, unit, thresholds):
        values = np.asarray(values, dtype=np.float64)
        percentiles = np.percentile(values, [90, 95, 99])
        frac_terms = [f"<= {thr:g}{unit}: {(100.0 * np.mean(values <= thr)):.1f}%" for thr in thresholds]
        return (
            f"mean={values.mean():.4f}{unit}, "
            f"median={np.median(values):.4f}{unit}, "
            f"p90={percentiles[0]:.4f}{unit}, "
            f"p95={percentiles[1]:.4f}{unit}, "
            f"p99={percentiles[2]:.4f}{unit}, "
            f"max={values.max():.4f}{unit}; " + ", ".join(frac_terms)
        )

    def _first_shell_below_threshold(fsc_values, threshold):
        fsc_values = np.asarray(fsc_values, dtype=np.float64)
        below = np.where(fsc_values < float(threshold))[0]
        return int(below[0]) if below.size else None

    def _shell_to_resolution_angstrom(shell_idx):
        if shell_idx is None or shell_idx <= 0:
            return np.nan
        return float(N * pixel_size) / float(shell_idx)

    def _compute_fsc_vs_gt(volume_ft_flat, gt_ft_flat):
        return np.asarray(
            regularization.get_fsc_gpu(
                jnp.asarray(volume_ft_flat),
                jnp.asarray(gt_ft_flat),
                (N, N, N),
            ),
            dtype=np.float64,
        )

    relion_dir = Path(args.relion_dir)
    iteration = args.iter
    prefix = str(relion_dir / f"run_it{iteration:03d}")

    # ---- Load RELION state ----
    model_h1 = starfile.read(f"{prefix}_half1_model.star")
    model_h2 = starfile.read(f"{prefix}_half2_model.star")
    control_model_h1 = model_h1
    control_model_path = relion_dir / f"run_it{replay_control_relion_iteration(iteration, 0):03d}_half1_model.star"
    if control_model_path.exists():
        control_model_h1 = starfile.read(control_model_path)
    N = int(model_h1["model_general"]["rlnOriginalImageSize"])
    current_size = int(control_model_h1["model_general"]["rlnCurrentImageSize"])
    pixel_size = float(model_h1["model_general"]["rlnPixelSize"])

    sigma2_h1 = np.array(model_h1["model_optics_group_1"]["rlnSigma2Noise"])
    sigma2_h2 = np.array(model_h2["model_optics_group_1"]["rlnSigma2Noise"])
    class1 = model_h1["model_class_1"]
    tau2_col = "rlnReferenceSigma2" if "rlnReferenceSigma2" in class1 else "rlnReferenceTau2"
    tau2 = np.array(class1[tau2_col])
    fsc_col = "rlnGoldStandardFsc" if "rlnGoldStandardFsc" in class1 else "rlnFourierShellCorrelationCorrected"
    fsc = np.array(class1[fsc_col])

    opt_text = (relion_dir / f"run_it{iteration:03d}_optimiser.star").read_text()
    m_pd = re.search(r"_rlnParticleDiameter\s+(\S+)", opt_text)
    particle_diameter = float(m_pd.group(1)) if m_pd else 544.0
    m_os = re.search(r"_rlnAdaptiveOversampleOrder\s+(\d+)", opt_text)
    oversampling = int(m_os.group(1)) if m_os else 0
    m_af = re.search(r"_rlnAdaptiveOversampleFraction\s+(\S+)", opt_text)
    adaptive_fraction = float(m_af.group(1)) if m_af else 0.999
    m_ms = re.search(r"_rlnMaximumSignificantPoses\s+(-?\d+)", opt_text)
    max_significants = int(m_ms.group(1)) if m_ms else 500
    optimiser_cli_flags = parse_relion_optimiser_cli_flags(opt_text)
    do_firstiter_cc = bool(optimiser_cli_flags["do_firstiter_cc"])
    relion_ini_high = (
        float(args.relion_ini_high)
        if args.relion_ini_high is not None
        else float(optimiser_cli_flags["ini_high_angstrom"])
        if optimiser_cli_flags["ini_high_angstrom"] is not None
        else 30.0
    )

    sampling_meta = read_relion_sampling_metadata(relion_dir / f"run_it{iteration:03d}_sampling.star")
    hp_order = int(sampling_meta["healpix_order"])
    offset_range = float(sampling_meta["offset_range"])
    offset_step = float(sampling_meta["offset_step"])

    # ave_Pmax from per-particle data. RELION it000 is a bootstrap state and
    # does not yet carry rlnMaxValueProbDistribution.
    relion_data = starfile.read(f"{prefix}_data.star")
    relion_df = relion_data["particles"] if isinstance(relion_data, dict) else relion_data
    relion_pmax = _read_relion_pmax_column(relion_df)
    if relion_pmax is not None:
        ave_Pmax = float(np.mean(relion_pmax))
    else:
        ave_Pmax = 0.0
        print(
            "  Initial RELION data STAR has no rlnMaxValueProbDistribution; bootstrapping init_ave_Pmax=0.0",
        )

    # has_high_fsc_at_limit (sticky flag)
    has_high_fsc_at_limit = False
    for it in range(1, iteration + 1):
        try:
            m = starfile.read(str(relion_dir / f"run_it{it:03d}_half1_model.star"))
            fc = np.array(m["model_class_1"][fsc_col])
            oc = (relion_dir / f"run_it{it:03d}_optimiser.star").read_text()
            cs_it = (
                int(re.search(r"_rlnCurrentImageSize\s+(\d+)", oc).group(1))
                if re.search(r"_rlnCurrentImageSize", oc)
                else None
            )
            if cs_it is None:
                mc = starfile.read(str(relion_dir / f"run_it{it:03d}_half1_model.star"))
                cs_it = int(mc["model_general"]["rlnCurrentImageSize"])
            shell_at_limit = cs_it // 2 - 1
            if shell_at_limit < len(fc) and fc[shell_at_limit] > 0.2:
                has_high_fsc_at_limit = True
        except Exception:
            pass

    print(f"RELION state: N={N}, hp={hp_order}, os={oversampling}, cs={current_size}")
    print(f"  pixel_size={pixel_size}, particle_diameter={particle_diameter}")
    print(
        f"  ave_Pmax={ave_Pmax:.4f}, has_high_fsc_at_limit={has_high_fsc_at_limit}, "
        f"adaptive_fraction={adaptive_fraction:.6f}, max_significants={max_significants}"
    )
    print(f"  RELION do_firstiter_cc={do_firstiter_cc}, ini_high={relion_ini_high}")
    if args.force_oversampling is not None:
        print(f"  Oversampling override: {oversampling} -> {args.force_oversampling}")
        oversampling = int(args.force_oversampling)
    if args.adaptive_fraction is not None:
        print(f"  Adaptive fraction override: {adaptive_fraction} -> {args.adaptive_fraction}")
        adaptive_fraction = float(args.adaptive_fraction)
    if args.max_significants is not None:
        print(f"  Max significants override: {max_significants} -> {args.max_significants}")
        max_significants = int(args.max_significants)

    # ---- Init volumes ----
    # RELION FFT normalization: F_relion = FFT(img)/N^d, so sigma2/tau2 from
    # model.star are in RELION's convention.  recovar uses unnormalized FFT,
    # so power spectra scale by N^4.
    n4 = N**4
    noise_variance_h1 = jnp.asarray(recon_noise.make_radial_noise(sigma2_h1 * n4, (N, N)))
    noise_variance_h2 = jnp.asarray(recon_noise.make_radial_noise(sigma2_h2 * n4, (N, N)))
    noise_variance = jnp.stack([noise_variance_h1.reshape(-1), noise_variance_h2.reshape(-1)], axis=0)
    mean_variance = jnp.asarray(utils.make_radial_image(tau2 * n4, (N, N, N), extend_last_frequency=True))

    # Volume: get_dft3(vol_real) produces the unnormalized centered DFT.
    # This matches the internal convention expected by the refinement code.
    vol_h1 = helpers.load_relion_volume(f"{prefix}_half1_class001.mrc")
    vol_h2 = helpers.load_relion_volume(f"{prefix}_half2_class001.mrc")
    vol_ft_h1 = np.array(ftu.get_dft3(jnp.array(vol_h1))).reshape(-1)
    vol_ft_h2 = np.array(ftu.get_dft3(jnp.array(vol_h2))).reshape(-1)

    # ---- Dataset + half-set split ----
    ds = load_dataset(args.data_star)
    relion_subsets = np.array(relion_df["rlnRandomSubset"])
    relion_names = list(relion_df["rlnImageName"])
    our_particles = starfile.read(args.data_star)
    our_particles = our_particles["particles"] if isinstance(our_particles, dict) else our_particles
    our_names = list(our_particles["rlnImageName"])

    def _idx(name):
        return stack_index_from_image_name(name)

    relion_idx_map = {_idx(relion_names[i]): relion_subsets[i] for i in range(len(relion_names))}
    our_subsets = np.array([relion_idx_map.get(_idx(n), 0) for n in our_names])

    # Keep exact known-bad particles when debugging per-image score parity.
    if args.keep_stack_indices:
        keep_indices = set()
        for token in args.keep_stack_indices.replace(",", " ").split():
            token = token.strip()
            if token:
                keep_indices.add(int(token))
        keep_mask = np.zeros_like(our_subsets, dtype=bool)
        observed = set()
        for i, name in enumerate(our_names):
            stack_idx = stack_index_from_image_name(name)
            if stack_idx in keep_indices:
                keep_mask[i] = True
                observed.add(stack_idx)
        our_subsets[~keep_mask] = 0
        print(
            "  Focused particle selection: "
            f"requested={sorted(keep_indices)}, kept={int(np.sum(keep_mask))}, "
            f"half1={int(np.sum(our_subsets == 1))}, half2={int(np.sum(our_subsets == 2))}, "
            f"missing={sorted(keep_indices - observed)}"
        )

    # Subsample if requested (for fast debugging)
    if args.max_particles is not None:
        rng = np.random.RandomState(42)
        h1_idx = np.where(our_subsets == 1)[0]
        h2_idx = np.where(our_subsets == 2)[0]
        n_per_half = args.max_particles // 2
        if n_per_half < len(h1_idx):
            drop_h1 = rng.choice(h1_idx, size=len(h1_idx) - n_per_half, replace=False)
            our_subsets[drop_h1] = 0
        if n_per_half < len(h2_idx):
            drop_h2 = rng.choice(h2_idx, size=len(h2_idx) - n_per_half, replace=False)
            our_subsets[drop_h2] = 0
        print(f"  Subsampled to max_particles={args.max_particles}")

    ds_half1 = ds.subset(np.where(our_subsets == 1)[0])
    ds_half2 = ds.subset(np.where(our_subsets == 2)[0])
    print(f"  Half-sets: {len(np.where(our_subsets == 1)[0])} + {len(np.where(our_subsets == 2)[0])}")

    # ---- Image corrections (RELION parity: normcorr + scale) ----
    # RELION: img *= avg_norm_correction / normcorr  (ml_optimiser.cpp:6240)
    # then   Frefctf *= scale                        (ml_optimiser.cpp:7298)
    normcorr = np.array(relion_df["rlnNormCorrection"], dtype=np.float64)
    general_h1 = model_h1["model_general"]
    general_h2 = model_h2["model_general"]
    avg_norm_h1 = float(
        general_h1["rlnNormCorrectionAverage"]
        if isinstance(general_h1, dict)
        else general_h1["rlnNormCorrectionAverage"].iloc[0]
    )
    avg_norm_h2 = float(
        general_h2["rlnNormCorrectionAverage"]
        if isinstance(general_h2, dict)
        else general_h2["rlnNormCorrectionAverage"].iloc[0]
    )
    groups_h1 = model_h1.get("model_groups", None)
    groups_h2 = model_h2.get("model_groups", None)
    scale_h1 = (
        np.array(groups_h1["rlnGroupScaleCorrection"], dtype=np.float64)
        if groups_h1 is not None and "rlnGroupScaleCorrection" in groups_h1.columns
        else np.array([1.0])
    )
    scale_h2 = (
        np.array(groups_h2["rlnGroupScaleCorrection"], dtype=np.float64)
        if groups_h2 is not None and "rlnGroupScaleCorrection" in groups_h2.columns
        else np.array([1.0])
    )
    group_numbers = (
        np.array(relion_df["rlnGroupNumber"], dtype=int)
        if "rlnGroupNumber" in relion_df.columns
        else np.ones(len(relion_df), dtype=int)
    )
    pp_scale_h1 = scale_h1[np.clip(group_numbers - 1, 0, len(scale_h1) - 1)]
    pp_scale_h2 = scale_h2[np.clip(group_numbers - 1, 0, len(scale_h2) - 1)]

    combined_h1 = (avg_norm_h1 / normcorr) * pp_scale_h1
    combined_h2 = (avg_norm_h2 / normcorr) * pp_scale_h2

    # Map to dataset ordering per half-set
    relion_idx_to_pos = {_idx(relion_names[i]): i for i in range(len(relion_names))}
    half1_mask = our_subsets == 1
    half2_mask = our_subsets == 2
    half1_our_idx = [_idx(our_names[i]) for i in np.where(half1_mask)[0]]
    half2_our_idx = [_idx(our_names[i]) for i in np.where(half2_mask)[0]]
    corr_h1 = np.array([combined_h1[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
    corr_h2 = np.array([combined_h2[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
    scale_corr_h1 = np.array([pp_scale_h1[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
    scale_corr_h2 = np.array([pp_scale_h2[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
    print(
        "  Image corrections: "
        f"avg_norm_h1={avg_norm_h1:.6f}, avg_norm_h2={avg_norm_h2:.6f}, "
        f"corr_h1 mean={corr_h1.mean():.4f}, corr_h2 mean={corr_h2.mean():.4f}"
    )

    # ---- Previous best translations (RELION parity: pre-centering) ----
    # RELION pre-centers images by old_offset before scoring
    if "rlnOriginXAngst" in relion_df.columns:
        offsets_x = np.array(relion_df["rlnOriginXAngst"], dtype=np.float64) / pixel_size
        offsets_y = np.array(relion_df["rlnOriginYAngst"], dtype=np.float64) / pixel_size
        offsets = np.stack([offsets_x, offsets_y], axis=1)
        trans_h1 = np.array([offsets[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
        trans_h2 = np.array([offsets[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
        print(
            f"  Pre-centering offsets: h1 mean_abs={np.abs(trans_h1).mean():.3f} px, h2 mean_abs={np.abs(trans_h2).mean():.3f} px"
        )
    else:
        trans_h1 = None
        trans_h2 = None

    angle_cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    euler_h1 = None
    euler_h2 = None
    if all(col in relion_df.columns for col in angle_cols):
        eulers = np.stack([np.array(relion_df[col], dtype=np.float64) for col in angle_cols], axis=1)
        euler_h1 = np.array([eulers[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
        euler_h2 = np.array([eulers[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
        print(f"  Previous best eulers: h1={euler_h1.shape[0]} particles, h2={euler_h2.shape[0]} particles")
    else:
        print("  Previous best eulers: None (angle columns not found)")

    # ---- Sigma offset from model star ----
    # RELION scores iteration N+1 with the model state written at iteration N.
    # The next model file is useful for replaying control outputs such as the
    # bootstrapped current image size, but its sigma offset is the result of
    # the E/M-step we are trying to reproduce, not an input to it.
    control_general = model_h1["model_general"]
    sigma_offset_angst = float(
        control_general["rlnSigmaOffsetsAngst"]
        if isinstance(control_general, dict)
        else control_general["rlnSigmaOffsetsAngst"].iloc[0]
    )
    print(f"  sigma_offset = {sigma_offset_angst:.4f} A")

    # ---- Direction prior from model star (RELION's pdf_orientation) ----
    pdf_orient_key = "model_pdf_orient_class_1"
    if pdf_orient_key in model_h1 and pdf_orient_key in model_h2:
        direction_prior = [
            np.array(model_h1[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
            np.array(model_h2[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
        ]
        print(
            "  direction_prior: "
            f"h1 {direction_prior[0].shape[0]} directions range=[{direction_prior[0].min():.6f}, {direction_prior[0].max():.6f}] zeros={int(np.sum(direction_prior[0] == 0))}; "
            f"h2 {direction_prior[1].shape[0]} directions range=[{direction_prior[1].min():.6f}, {direction_prior[1].max():.6f}] zeros={int(np.sum(direction_prior[1] == 0))}"
        )
    else:
        direction_prior = None
        print("  direction_prior: None (not found in model star)")

    def _load_relion_iteration_override(previous_relion_iteration, control_relion_iteration):
        iter_prefix = relion_dir / f"run_it{previous_relion_iteration:03d}"
        model_h1_iter = starfile.read(f"{iter_prefix}_half1_model.star")
        model_h2_iter = starfile.read(f"{iter_prefix}_half2_model.star")
        relion_iter_data = starfile.read(f"{iter_prefix}_data.star")
        relion_iter_df = relion_iter_data["particles"] if isinstance(relion_iter_data, dict) else relion_iter_data
        relion_iter_names = list(relion_iter_df["rlnImageName"])
        relion_iter_idx_to_pos = {_idx(relion_iter_names[i]): i for i in range(len(relion_iter_names))}
        general_h1_iter = model_h1_iter["model_general"]
        general_h2_iter = model_h2_iter["model_general"]
        avg_norm_h1_iter = float(
            general_h1_iter["rlnNormCorrectionAverage"]
            if isinstance(general_h1_iter, dict)
            else general_h1_iter["rlnNormCorrectionAverage"].iloc[0]
        )
        avg_norm_h2_iter = float(
            general_h2_iter["rlnNormCorrectionAverage"]
            if isinstance(general_h2_iter, dict)
            else general_h2_iter["rlnNormCorrectionAverage"].iloc[0]
        )
        sigma_offset_iter = float(
            general_h1_iter["rlnSigmaOffsetsAngst"]
            if isinstance(general_h1_iter, dict)
            else general_h1_iter["rlnSigmaOffsetsAngst"].iloc[0]
        )
        sigma2_h1_iter = np.array(model_h1_iter["model_optics_group_1"]["rlnSigma2Noise"])
        sigma2_h2_iter = np.array(model_h2_iter["model_optics_group_1"]["rlnSigma2Noise"])
        noise_variance_iter = jnp.stack(
            [
                jnp.asarray(recon_noise.make_radial_noise(sigma2_h1_iter * n4, (N, N))).reshape(-1),
                jnp.asarray(recon_noise.make_radial_noise(sigma2_h2_iter * n4, (N, N))).reshape(-1),
            ],
            axis=0,
        )

        normcorr_iter = np.array(relion_iter_df["rlnNormCorrection"], dtype=np.float64)
        groups_h1_iter = model_h1_iter.get("model_groups", None)
        groups_h2_iter = model_h2_iter.get("model_groups", None)
        scale_h1_iter = (
            np.array(groups_h1_iter["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h1_iter is not None and "rlnGroupScaleCorrection" in groups_h1_iter.columns
            else np.array([1.0])
        )
        scale_h2_iter = (
            np.array(groups_h2_iter["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h2_iter is not None and "rlnGroupScaleCorrection" in groups_h2_iter.columns
            else np.array([1.0])
        )
        iter_group_numbers = (
            np.array(relion_iter_df["rlnGroupNumber"], dtype=int)
            if "rlnGroupNumber" in relion_iter_df.columns
            else np.ones(len(relion_iter_df), dtype=int)
        )
        pp_scale_h1_iter = scale_h1_iter[np.clip(iter_group_numbers - 1, 0, len(scale_h1_iter) - 1)]
        pp_scale_h2_iter = scale_h2_iter[np.clip(iter_group_numbers - 1, 0, len(scale_h2_iter) - 1)]
        combined_h1_iter = (avg_norm_h1_iter / normcorr_iter) * pp_scale_h1_iter
        combined_h2_iter = (avg_norm_h2_iter / normcorr_iter) * pp_scale_h2_iter

        corr_h1_iter = np.array(
            [combined_h1_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32
        )
        corr_h2_iter = np.array(
            [combined_h2_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32
        )
        scale_corr_h1_iter = np.array(
            [pp_scale_h1_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
            dtype=np.float32,
        )
        scale_corr_h2_iter = np.array(
            [pp_scale_h2_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
            dtype=np.float32,
        )

        if "rlnOriginXAngst" in relion_iter_df.columns:
            offsets_x_iter = np.array(relion_iter_df["rlnOriginXAngst"], dtype=np.float64) / pixel_size
            offsets_y_iter = np.array(relion_iter_df["rlnOriginYAngst"], dtype=np.float64) / pixel_size
            offsets_iter = np.stack([offsets_x_iter, offsets_y_iter], axis=1)
            trans_h1_iter = np.array(
                [offsets_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            trans_h2_iter = np.array(
                [offsets_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )
        else:
            trans_h1_iter = None
            trans_h2_iter = None

        rot_h1_iter = None
        rot_h2_iter = None
        euler_h1_iter = None
        euler_h2_iter = None
        if all(col in relion_iter_df.columns for col in angle_cols):
            eulers_iter = np.stack([np.array(relion_iter_df[col], dtype=np.float64) for col in angle_cols], axis=1)
            rotations_iter = utils.R_from_relion(eulers_iter).astype(np.float32)
            rot_h1_iter = np.array(
                [rotations_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            rot_h2_iter = np.array(
                [rotations_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )
            euler_h1_iter = np.array(
                [eulers_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            euler_h2_iter = np.array(
                [eulers_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )

        pdf_iter = None
        if pdf_orient_key in model_h1_iter and pdf_orient_key in model_h2_iter:
            pdf_iter = [
                np.array(model_h1_iter[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
                np.array(model_h2_iter[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
            ]

        return {
            "translation_sigma_angstrom": np.float32(sigma_offset_iter),
            "image_corrections": [corr_h1_iter, corr_h2_iter],
            "scale_corrections": [scale_corr_h1_iter, scale_corr_h2_iter],
            "previous_best_translations": [trans_h1_iter, trans_h2_iter],
            "previous_best_rotations": [rot_h1_iter, rot_h2_iter],
            "previous_best_rotation_eulers": [euler_h1_iter, euler_h2_iter],
            "direction_prior": pdf_iter,
            "noise_variance": noise_variance_iter,
        }

    replay_iteration_overrides = [None] * args.max_iter
    for recovar_iter in range(1, args.max_iter):
        relion_prev_iter = replay_previous_relion_iteration(iteration, recovar_iter)
        relion_control_iter = replay_control_relion_iteration(iteration, recovar_iter)
        if not (relion_dir / f"run_it{relion_prev_iter:03d}_data.star").exists():
            print(
                f"  Replay state for recovar iter {recovar_iter + 1}: RELION iter {relion_prev_iter:03d} not found, leaving override unset"
            )
            continue
        replay_iteration_overrides[recovar_iter] = _load_relion_iteration_override(
            relion_prev_iter,
            relion_control_iter,
        )
        override = replay_iteration_overrides[recovar_iter]
        trans_msg = "none"
        if override["previous_best_translations"][0] is not None:
            trans_msg = (
                f"h1 mean_abs={np.abs(override['previous_best_translations'][0]).mean():.3f} px, "
                f"h2 mean_abs={np.abs(override['previous_best_translations'][1]).mean():.3f} px"
            )
        print(
            f"  Replay state for recovar iter {recovar_iter + 1}: RELION prev={relion_prev_iter:03d}, control={relion_control_iter:03d}, "
            f"sigma_offset={float(override['translation_sigma_angstrom']):.4f} A, "
            f"corr means=({override['image_corrections'][0].mean():.4f}, {override['image_corrections'][1].mean():.4f}), "
            f"pre-shifts={trans_msg}"
        )

    # ---- Output directory ----
    out_dir = args.output_dir or str(relion_dir.parent / "_agent_scratch" / f"{args.max_iter}iter_parity")
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir).joinpath("SAFE_TO_DELETE").touch()
    if args.timing_only and args.save_intermediates_dir is None:
        save_intermediates_dir = None
    else:
        save_intermediates_dir = args.save_intermediates_dir or os.path.join(out_dir, "intermediates")
        os.makedirs(save_intermediates_dir, exist_ok=True)
    print(f"  Intermediate dumps: {save_intermediates_dir if save_intermediates_dir is not None else '<disabled>'}")

    gt_path = None
    if args.gt_volume is not None:
        gt_path = Path(args.gt_volume)
    else:
        candidate_gt = Path(args.data_star).with_name("reference_gt.mrc")
        if candidate_gt.exists():
            gt_path = candidate_gt
    gt_real = None
    gt_ft = None
    if gt_path is not None and gt_path.exists():
        gt_real = helpers.load_mrc(str(gt_path))
        gt_ft = np.asarray(ftu.get_dft3(jnp.asarray(gt_real))).reshape(-1)
        print(f"  GT volume: {gt_path}")
    elif args.gt_volume is not None:
        print(f"  GT volume requested but not found: {args.gt_volume}")

    print(f"  Local-search profile: {args.local_search_profile}")
    print(f"  Local translation prior mode: {args.local_search_translation_prior_mode}")
    print(f"  First-iteration score mode: {args.first_iteration_score_mode}")
    print(f"  First-iteration reconstruction mode: {args.first_iteration_reconstruction_mode}")
    print(f"  Emulate RELION iter-1 CC: {args.iter == 0 and do_firstiter_cc}")
    print(f"  RELION ini_high: {relion_ini_high}")
    print(f"  Adjoint ablations: disable_y={args.disable_adjoint_y}, disable_ctf={args.disable_adjoint_ctf}")

    # ---- Run ----
    print(f"\nRunning {args.max_iter} iterations...")
    t0 = time.time()
    result = refine_single_volume(
        experiment_datasets=[ds_half1, ds_half2],
        init_volume=[jnp.asarray(vol_ft_h1), jnp.asarray(vol_ft_h2)],
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance.reshape(-1),
        rotations=None,
        translations=None,
        disc_type="linear_interp",
        mode="relion",
        max_iter=args.max_iter,
        image_batch_size=500,
        rotation_block_size=5000,
        init_current_size=current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=oversampling,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        init_healpix_order=hp_order,
        max_healpix_order=args.max_healpix_order,
        init_translation_range=offset_range / pixel_size,
        init_translation_step=offset_step / pixel_size,
        init_translation_sigma_angstrom=sigma_offset_angst,
        particle_diameter_ang=particle_diameter,
        tau2_fudge=1.0,
        perturb_factor=0.5,
        perturb_replay_relion_dir=str(relion_dir),
        init_relion_iteration=iteration,
        init_fsc=fsc,
        init_ave_Pmax=ave_Pmax,
        init_has_high_fsc_at_limit=has_high_fsc_at_limit,
        init_image_corrections=[corr_h1, corr_h2],
        init_scale_corrections=[scale_corr_h1, scale_corr_h2],
        init_previous_best_translations=[trans_h1, trans_h2],
        init_previous_best_rotation_eulers=[euler_h1, euler_h2],
        init_direction_prior=direction_prior,
        replay_iteration_overrides=replay_iteration_overrides,
        save_intermediates_dir=save_intermediates_dir,
        skip_final_iteration=args.skip_final_iteration,
        local_search_profile_mode=args.local_search_profile,
        local_search_translation_prior_mode=args.local_search_translation_prior_mode,
        disable_adjoint_y=args.disable_adjoint_y,
        disable_adjoint_ctf=args.disable_adjoint_ctf,
        emulate_relion_firstiter_cc=(args.iter == 0 and do_firstiter_cc),
        relion_firstiter_ini_high_angstrom=relion_ini_high if args.iter == 0 else None,
        first_iteration_score_mode=args.first_iteration_score_mode,
        first_iteration_reconstruction_mode=args.first_iteration_reconstruction_mode,
        force_max_iter_after_convergence=args.force_max_iter_after_convergence,
    )
    elapsed = time.time() - t0
    completed_iters = len(result.get("current_sizes", []))
    if completed_iters != args.max_iter:
        print(
            f"\nCompleted {completed_iters} emitted iterations in {elapsed:.1f}s "
            f"(requested --max_iter {args.max_iter}; stopped by convergence)"
        )
    else:
        print(f"\nCompleted {completed_iters} emitted iterations in {elapsed:.1f}s")

    if args.timing_only:
        ledger_path = args.benchmark_ledger_json or os.path.join(out_dir, "benchmark_ledger.json")
        local_profile_rows = _collect_local_profile_history(result)
        if not local_profile_rows and save_intermediates_dir is not None:
            local_profile_rows = _collect_local_profile_rows(save_intermediates_dir)
        wall_times = [float(x) for x in result.get("wall_times", [])]
        ledger = {
            "git_commit": _safe_git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "relion_dir": str(relion_dir),
            "data_star": str(args.data_star),
            "iter_start": int(args.iter),
            "max_iter": int(args.max_iter),
            "completed_iterations": int(completed_iters),
            "force_max_iter_after_convergence": bool(args.force_max_iter_after_convergence),
            "elapsed_s": float(elapsed),
            "timing_only": True,
            "local_search_profile_mode": args.local_search_profile,
            "disable_adjoint_y": bool(args.disable_adjoint_y),
            "disable_adjoint_ctf": bool(args.disable_adjoint_ctf),
            "compile_count_from_log": _count_compile_lines(args.compile_log),
            "wall_times_trajectory": wall_times,
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
            "ave_Pmax_trajectory": [float(x) for x in result.get("ave_Pmax_trajectory", [])],
            "local_profile_rows": local_profile_rows,
            "local_profile_summary": _summarize_local_profile_rows(local_profile_rows, wall_times),
        }
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        print(f"Saved timing-only benchmark ledger: {ledger_path}")
        return

    # ---- Save results ----
    save_dict = {
        "volume_shape": np.array([N, N, N]),
        "voxel_size": np.float64(pixel_size),
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "n_half1_particles": np.int32(len(np.where(our_subsets == 1)[0])),
        "n_half2_particles": np.int32(len(np.where(our_subsets == 2)[0])),
        "adaptive_fraction": np.float64(adaptive_fraction),
        "max_significants": np.int32(max_significants),
        "local_search_profile_mode": np.array(args.local_search_profile),
        "local_search_translation_prior_mode": np.array(args.local_search_translation_prior_mode),
        "first_iteration_score_mode": np.array(args.first_iteration_score_mode),
        "first_iteration_reconstruction_mode": np.array(args.first_iteration_reconstruction_mode),
        "relion_ini_high_angstrom": np.float64(relion_ini_high),
        "disable_adjoint_y": np.bool_(args.disable_adjoint_y),
        "disable_adjoint_ctf": np.bool_(args.disable_adjoint_ctf),
    }
    if result.get("ave_Pmax_trajectory"):
        save_dict["ave_Pmax_trajectory"] = np.array(result["ave_Pmax_trajectory"])
    if result.get("pmax_per_image_history"):
        for i, pmax_arr in enumerate(result["pmax_per_image_history"]):
            save_dict[f"pmax_per_image_iter_{i:03d}"] = np.array(pmax_arr, dtype=np.float32)
    if result.get("healpix_order_trajectory"):
        save_dict["healpix_order_trajectory"] = np.array(result["healpix_order_trajectory"])
    if result.get("wall_times"):
        save_dict["wall_times_trajectory"] = np.array(result["wall_times"], dtype=np.float64)
    if result.get("sigma_offset_trajectory"):
        save_dict["sigma_offset_trajectory"] = np.array(result["sigma_offset_trajectory"], dtype=np.float64)
    if result.get("sigma_offset_used_trajectory"):
        save_dict["sigma_offset_used_trajectory"] = np.array(result["sigma_offset_used_trajectory"], dtype=np.float64)
    for scalar_name in [
        "frac_changed_trajectory",
        "acc_rot_trajectory",
        "smallest_change_angles_trajectory",
        "smallest_change_offsets_trajectory",
    ]:
        if result.get(scalar_name):
            save_dict[scalar_name] = np.array(result[scalar_name], dtype=np.float64)
    for traj_name, prefix_name in [
        ("fsc_history", "fsc_iter"),
        ("data_vs_prior_trajectory", "data_vs_prior_iter"),
        ("noise_radial_trajectory", "noise_radial_iter"),
        ("noise_radial_per_half_trajectory", "noise_radial_per_half_iter"),
        ("tau2_radial_trajectory", "tau2_radial_iter"),
        ("tau2_sigma2_trajectory", "tau2_sigma2_iter"),
        ("tau2_avg_weight_trajectory", "tau2_avg_weight_iter"),
        ("tau2_shell_sum_trajectory", "tau2_shell_sum_iter"),
        ("tau2_shell_count_trajectory", "tau2_shell_count_iter"),
        ("tau2_fsc_used_trajectory", "tau2_fsc_used_iter"),
        ("tau2_ssnr_trajectory", "tau2_ssnr_iter"),
        ("significant_counts", "sig_counts_iter"),
    ]:
        if result.get(traj_name):
            for i, arr_i in enumerate(result[traj_name]):
                if arr_i is not None:
                    save_dict[f"{prefix_name}_{i:03d}"] = np.array(arr_i)
    for traj_name, prefix_name in [
        ("best_rotation_eulers_history", "best_rotation_eulers_iter"),
        ("best_translations_history", "best_translations_iter"),
    ]:
        if result.get(traj_name):
            for i, arr_i in enumerate(result[traj_name]):
                if arr_i is not None:
                    save_dict[f"{prefix_name}_{i:03d}"] = np.array(arr_i, dtype=np.float32)

    final_half1_ft = np.asarray(result["means"][0], dtype=np.complex64).reshape(-1)
    final_half2_ft = np.asarray(result["means"][1], dtype=np.complex64).reshape(-1)
    final_merged_ft = (final_half1_ft.astype(np.complex128) + final_half2_ft.astype(np.complex128)) / 2.0
    final_merged_ft = final_merged_ft.astype(np.complex64)

    save_dict["final_half1_ft"] = final_half1_ft
    save_dict["final_half2_ft"] = final_half2_ft
    save_dict["final_merged_ft"] = final_merged_ft

    save_volume(
        np.asarray(final_half1_ft),
        os.path.join(out_dir, "recovar_final_half1"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    save_volume(
        np.asarray(final_half2_ft),
        os.path.join(out_dir, "recovar_final_half2"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    save_volume(
        np.asarray(final_merged_ft),
        os.path.join(out_dir, "recovar_final_merged"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    print(
        f"Saved final volumes: {os.path.join(out_dir, 'recovar_final_half1.mrc')}, recovar_final_half2.mrc, recovar_final_merged.mrc"
    )

    # ---- Summary table ----
    n_iters = len(result["current_sizes"])
    print(f"\n{'iter':>4} {'cs':>4} {'pixres':>6} {'pmax':>8} {'hp':>3} {'FSC@0.5':>8} {'res(A)':>8}")
    print("-" * 50)
    for i in range(n_iters):
        cs_i = result["current_sizes"][i]
        pr_i = result["pixel_resolutions"][i]
        pmax_i = (
            result["ave_Pmax_trajectory"][i]
            if result.get("ave_Pmax_trajectory") and i < len(result["ave_Pmax_trajectory"])
            else 0
        )
        hp_i = (
            result["healpix_order_trajectory"][i]
            if result.get("healpix_order_trajectory") and i < len(result["healpix_order_trajectory"])
            else hp_order
        )
        fsc_i = (
            np.array(result["fsc_history"][i]) if result.get("fsc_history") and i < len(result["fsc_history"]) else None
        )
        fsc05 = 0
        if fsc_i is not None:
            for s in range(1, len(fsc_i)):
                if fsc_i[s] >= 0.5:
                    fsc05 = s
        res = (N * pixel_size) / max(fsc05, 1)
        print(f"{i + 1:4d} {cs_i:4d} {pr_i:6.1f} {pmax_i:8.4f} {hp_i:3d} {fsc05:8d} {res:8.1f}")

    # ---- Compare final volume with RELION ----
    last_relion_it = iteration + args.max_iter
    relion_final_real = {}
    relion_final_ft = {}
    for k_half, label in [(0, "half1"), (1, "half2")]:
        target_path = str(relion_dir / f"run_it{last_relion_it:03d}_{label}_class001.mrc")
        if not Path(target_path).exists():
            print(f"  Final {label}: RELION it{last_relion_it:03d} not found")
            continue
        recovar_vol_ft = np.asarray(result["means"][k_half])
        recovar_vol_real = np.real(np.array(ftu.get_idft3(jnp.asarray(recovar_vol_ft.reshape(N, N, N)))))
        relion_vol = helpers.load_relion_volume(target_path)
        relion_final_real[label] = relion_vol
        relion_final_ft[label] = np.asarray(ftu.get_dft3(jnp.asarray(relion_vol))).reshape(-1)
        corr = float(np.corrcoef(recovar_vol_real.ravel(), relion_vol.ravel())[0, 1])
        print(f"  Final {label} vs RELION it{last_relion_it:03d}: corr={corr:.6f}")
        save_dict[f"final_{label}_corr_vs_relion"] = np.float64(corr)

    relion_merged_ft = None
    if "half1" in relion_final_ft and "half2" in relion_final_ft:
        relion_merged_ft = (
            relion_final_ft["half1"].astype(np.complex128) + relion_final_ft["half2"].astype(np.complex128)
        ) / 2.0
        save_volume(
            np.asarray(relion_final_ft["half1"]),
            os.path.join(out_dir, "relion_final_half1"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
        save_volume(
            np.asarray(relion_final_ft["half2"]),
            os.path.join(out_dir, "relion_final_half2"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
        save_volume(
            np.asarray(relion_merged_ft.astype(np.complex64)),
            os.path.join(out_dir, "relion_final_merged"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
        print(
            "  Saved RELION-frame final volumes in recovar coordinates: "
            f"{os.path.join(out_dir, 'relion_final_half1.mrc')}, relion_final_half2.mrc, relion_final_merged.mrc"
        )

    if gt_ft is not None:
        print("\n=== Final FSC vs GT ===")
        gt_summary = {}
        recovar_final_series = {
            "recovar_half1": final_half1_ft,
            "recovar_half2": final_half2_ft,
            "recovar_merged": final_merged_ft,
        }
        if relion_merged_ft is not None:
            recovar_final_series["relion_half1"] = relion_final_ft["half1"]
            recovar_final_series["relion_half2"] = relion_final_ft["half2"]
            recovar_final_series["relion_merged"] = relion_merged_ft.astype(np.complex64)

        for label, vol_ft in recovar_final_series.items():
            fsc_vs_gt = _compute_fsc_vs_gt(vol_ft, gt_ft)
            shell_05 = _first_shell_below_threshold(fsc_vs_gt, 0.5)
            shell_0143 = _first_shell_below_threshold(fsc_vs_gt, 0.143)
            real_vol = np.real(np.array(ftu.get_idft3(jnp.asarray(np.asarray(vol_ft).reshape(N, N, N)))))
            corr_vs_gt = float(np.corrcoef(real_vol.ravel(), gt_real.ravel())[0, 1])
            print(
                f"  {label:<14s} corr={corr_vs_gt:.6f}, "
                f"FSC<0.5 shell={shell_05}, res={_shell_to_resolution_angstrom(shell_05):.2f} A, "
                f"FSC<0.143 shell={shell_0143}, res={_shell_to_resolution_angstrom(shell_0143):.2f} A"
            )
            gt_summary[f"{label}_fsc_vs_gt"] = fsc_vs_gt
            gt_summary[f"{label}_corr_vs_gt"] = np.float64(corr_vs_gt)
            gt_summary[f"{label}_shell_05"] = np.int32(-1 if shell_05 is None else shell_05)
            gt_summary[f"{label}_shell_0143"] = np.int32(-1 if shell_0143 is None else shell_0143)

        save_dict.update(gt_summary)
        gt_npz_path = os.path.join(out_dir, "gt_comparison_final.npz")
        np.savez(gt_npz_path, **gt_summary)
        print(f"  Saved GT comparison: {gt_npz_path}")

    npz_path = os.path.join(out_dir, "refinement_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved: {npz_path}")

    ledger_path = args.benchmark_ledger_json or os.path.join(out_dir, "benchmark_ledger.json")
    local_profile_rows = _collect_local_profile_rows(save_intermediates_dir)
    wall_times = [float(x) for x in result.get("wall_times", [])]
    ledger = {
        "git_commit": _safe_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": getattr(jaxlib, "__version__", None),
        "jax_devices": [str(device) for device in jax.devices()],
        "relion_dir": str(relion_dir),
        "data_star": str(args.data_star),
        "iter_start": int(args.iter),
        "max_iter": int(args.max_iter),
        "completed_iterations": int(completed_iters),
        "force_max_iter_after_convergence": bool(args.force_max_iter_after_convergence),
        "elapsed_s": float(elapsed),
        "local_search_profile_mode": args.local_search_profile,
        "disable_adjoint_y": bool(args.disable_adjoint_y),
        "disable_adjoint_ctf": bool(args.disable_adjoint_ctf),
        "compile_count_from_log": _count_compile_lines(args.compile_log),
        "wall_times_trajectory": wall_times,
        "current_sizes": [int(x) for x in result.get("current_sizes", [])],
        "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
        "local_profile_rows": local_profile_rows,
        "local_profile_summary": _summarize_local_profile_rows(local_profile_rows, wall_times),
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
    print(f"Saved benchmark ledger: {ledger_path}")

    # ---- Per-particle Pmax comparison with RELION ----
    # pmax_per_image_history entries are in (half1, half2) concatenated order.
    # Map them back to original particle ordering for matched comparison.
    half1_indices = np.where(our_subsets == 1)[0]
    half2_indices = np.where(our_subsets == 2)[0]
    n_total = len(our_names)
    gt_pose_path = Path(args.data_star).with_name("poses.pkl")
    gt_rotations_orig = None
    gt_translations_orig = None
    gt_transpose_relion_convention = None
    if gt_pose_path.exists():
        gt_pose_data = utils.pickle_load(str(gt_pose_path))
        if isinstance(gt_pose_data, tuple) and len(gt_pose_data) >= 1:
            gt_rot_all = np.asarray(gt_pose_data[0], dtype=np.float64)
            gt_trans_all = np.asarray(gt_pose_data[1], dtype=np.float64) if len(gt_pose_data) >= 2 else None
            gt_rotations_orig, gt_translations_orig = map_pose_arrays_to_particle_order(
                our_names,
                gt_rot_all,
                gt_trans_all,
            )
            print(f"  GT poses: {gt_pose_path}")
        else:
            print(f"  GT poses present but not in expected tuple format: {gt_pose_path}")

    if result.get("pmax_per_image_history"):
        for i_iter, pmax_arr in enumerate(result["pmax_per_image_history"]):
            target_it = iteration + 1 + i_iter
            target_data_star = relion_dir / f"run_it{target_it:03d}_data.star"
            if not target_data_star.exists():
                print(
                    f"\n  Iter {i_iter + 1}: RELION data star it{target_it:03d} not found, skipping per-particle comparison"
                )
                continue
            relion_data_it = starfile.read(str(target_data_star))
            relion_df_it = relion_data_it["particles"] if isinstance(relion_data_it, dict) else relion_data_it
            relion_pmax_raw = _read_relion_pmax_column(relion_df_it)
            if relion_pmax_raw is None:
                print(
                    f"\n  Iter {i_iter + 1}: RELION data star it{target_it:03d} lacks rlnMaxValueProbDistribution, skipping per-particle comparison"
                )
                continue

            # Map RELION particles to original ordering by stack index
            relion_names_it = list(relion_df_it["rlnImageName"])
            relion_idx_to_pos = {_idx(relion_names_it[j]): j for j in range(len(relion_names_it))}
            relion_pmax_map = {_idx(relion_names_it[j]): relion_pmax_raw[j] for j in range(len(relion_names_it))}

            # Reconstruct recovar Pmax in original particle ordering
            # pmax_arr = [half1_pmax (n_half1,), half2_pmax (n_half2,)] concatenated
            pmax_arr_np = np.asarray(pmax_arr, dtype=np.float64)
            n_h1 = len(half1_indices)
            recovar_pmax_orig = np.full(n_total, np.nan, dtype=np.float64)
            recovar_pmax_orig[half1_indices] = pmax_arr_np[:n_h1]
            recovar_pmax_orig[half2_indices] = pmax_arr_np[n_h1:]

            # Build matched RELION array in original ordering
            relion_pmax_orig = np.full(n_total, np.nan, dtype=np.float64)
            relion_eulers_orig = np.full((n_total, 3), np.nan, dtype=np.float64)
            relion_trans_orig = np.full((n_total, 2), np.nan, dtype=np.float64)
            has_relion_eulers = all(
                col in relion_df_it.columns for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
            )
            has_relion_trans = all(col in relion_df_it.columns for col in ["rlnOriginXAngst", "rlnOriginYAngst"])
            relion_eulers_raw = (
                np.stack(
                    [
                        np.array(relion_df_it[col], dtype=np.float64)
                        for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
                    ],
                    axis=1,
                )
                if has_relion_eulers
                else None
            )
            relion_trans_raw = (
                np.stack(
                    [
                        np.array(relion_df_it["rlnOriginXAngst"], dtype=np.float64) / pixel_size,
                        np.array(relion_df_it["rlnOriginYAngst"], dtype=np.float64) / pixel_size,
                    ],
                    axis=1,
                )
                if has_relion_trans
                else None
            )
            recovar_trans_orig = None
            for j, name in enumerate(our_names):
                idx = _idx(name)
                if idx in relion_pmax_map:
                    relion_pmax_orig[j] = relion_pmax_map[idx]
                    rel_pos = relion_idx_to_pos[idx]
                    if relion_eulers_raw is not None:
                        relion_eulers_orig[j] = relion_eulers_raw[rel_pos]
                    if relion_trans_raw is not None:
                        relion_trans_orig[j] = relion_trans_raw[rel_pos]

            # Compare only particles present in both
            valid = ~(np.isnan(recovar_pmax_orig) | np.isnan(relion_pmax_orig))
            recovar_pmax = recovar_pmax_orig[valid]
            relion_pmax_matched = relion_pmax_orig[valid]

            diff = recovar_pmax - relion_pmax_matched
            abs_diff = np.abs(diff)
            corr = float(np.corrcoef(recovar_pmax, relion_pmax_matched)[0, 1])

            print(f"\n=== Per-particle Pmax comparison: iter {i_iter + 1} (RELION it{target_it:03d}) ===")
            print(f"  N particles matched: {valid.sum()} / {n_total}")
            print(f"  recovar  ave_Pmax = {recovar_pmax.mean():.6f}")
            print(f"  RELION   ave_Pmax = {relion_pmax_matched.mean():.6f}")
            print(f"  Gap (recovar - RELION) = {diff.mean():.6f}")
            print(
                f"  Abs diff:  mean={abs_diff.mean():.6f}, median={np.median(abs_diff):.6f}, max={abs_diff.max():.6f}"
            )
            print(f"  Std diff:  {diff.std():.6f}")
            print(f"  Correlation: {corr:.6f}")
            print("  Percentiles of (recovar - RELION):")
            for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{pct:2d}: {np.percentile(diff, pct):+.6f}")

            # Save full per-particle comparison
            comp_path = os.path.join(out_dir, f"pmax_comparison_iter{i_iter:03d}.npz")
            np.savez(
                comp_path,
                recovar_pmax=recovar_pmax_orig,
                relion_pmax=relion_pmax_orig,
                diff_valid=diff,
                half1_indices=half1_indices,
                half2_indices=half2_indices,
            )
            print(f"  Saved per-particle comparison: {comp_path}")

            best_eulers_hist = result.get("best_rotation_eulers_history")
            best_trans_hist = result.get("best_translations_history")
            if best_eulers_hist and i_iter < len(best_eulers_hist) and best_eulers_hist[i_iter] is not None:
                best_eulers_arr = np.asarray(best_eulers_hist[i_iter], dtype=np.float64)
                recovar_eulers_orig = np.full((n_total, 3), np.nan, dtype=np.float64)
                recovar_eulers_orig[half1_indices] = best_eulers_arr[:n_h1]
                recovar_eulers_orig[half2_indices] = best_eulers_arr[n_h1:]
                valid_angle = ~(np.isnan(recovar_eulers_orig).any(axis=1) | np.isnan(relion_eulers_orig).any(axis=1))
                if np.any(valid_angle):
                    ang_err_deg = _angular_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    view_err_deg = _view_direction_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    inplane_err_deg = _inplane_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    print(f"  Angular error (deg): {_format_error_summary(ang_err_deg, '°', [5, 10, 20])}")
                    print(f"  View-dir error (deg): {_format_error_summary(view_err_deg, '°', [2, 5, 10])}")
                    print(f"  In-plane error (deg): {_format_error_summary(inplane_err_deg, '°', [2, 5, 10])}")
                else:
                    ang_err_deg = None
                    view_err_deg = None
                    inplane_err_deg = None

                recovar_gt_ang_err_deg = None
                recovar_gt_view_err_deg = None
                recovar_gt_inplane_err_deg = None
                relion_gt_ang_err_deg = None
                relion_gt_view_err_deg = None
                relion_gt_inplane_err_deg = None
                if gt_rotations_orig is not None:
                    valid_relion_gt = ~(
                        np.isnan(relion_eulers_orig).any(axis=1) | np.isnan(gt_rotations_orig).any(axis=(1, 2))
                    )
                    if np.any(valid_relion_gt) and gt_transpose_relion_convention is None:
                        relion_gt_direct = _rotation_matrices_from_eulers_deg(relion_eulers_orig[valid_relion_gt])
                        direct_err = _angular_error_deg_from_rotations(
                            relion_gt_direct,
                            gt_rotations_orig[valid_relion_gt],
                        )
                        transpose_err = _angular_error_deg_from_rotations(
                            np.transpose(relion_gt_direct, (0, 2, 1)),
                            gt_rotations_orig[valid_relion_gt],
                        )
                        gt_transpose_relion_convention = bool(np.nanmedian(transpose_err) < np.nanmedian(direct_err))
                        mode = "transpose" if gt_transpose_relion_convention else "direct"
                        print(
                            "  GT rotation convention: using "
                            f"{mode} RELION-like rotations "
                            f"(RELION-vs-GT median direct={np.nanmedian(direct_err):.4f}°, "
                            f"transpose={np.nanmedian(transpose_err):.4f}°)"
                        )

                    valid_recovar_gt = ~(
                        np.isnan(recovar_eulers_orig).any(axis=1) | np.isnan(gt_rotations_orig).any(axis=(1, 2))
                    )
                    if np.any(valid_recovar_gt):
                        recovar_rot_gt = _rotations_in_gt_frame_from_relion_eulers(
                            recovar_eulers_orig[valid_recovar_gt],
                            gt_transpose_relion_convention if gt_transpose_relion_convention is not None else True,
                        )
                        gt_rot_valid = gt_rotations_orig[valid_recovar_gt]
                        recovar_gt_ang_err_deg = _angular_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        recovar_gt_view_err_deg = _view_direction_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        recovar_gt_inplane_err_deg = _inplane_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        print(
                            "  RECOVAR vs GT angle error: "
                            f"{_format_error_summary(recovar_gt_ang_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RECOVAR vs GT view-dir: "
                            f"{_format_error_summary(recovar_gt_view_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RECOVAR vs GT in-plane: "
                            f"{_format_error_summary(recovar_gt_inplane_err_deg, '°', [2, 5, 10])}"
                        )
                    if np.any(valid_relion_gt):
                        relion_rot_gt = _rotations_in_gt_frame_from_relion_eulers(
                            relion_eulers_orig[valid_relion_gt],
                            gt_transpose_relion_convention if gt_transpose_relion_convention is not None else True,
                        )
                        gt_rot_valid = gt_rotations_orig[valid_relion_gt]
                        relion_gt_ang_err_deg = _angular_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        relion_gt_view_err_deg = _view_direction_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        relion_gt_inplane_err_deg = _inplane_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        print(
                            "  RELION  vs GT angle error: "
                            f"{_format_error_summary(relion_gt_ang_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RELION  vs GT view-dir: "
                            f"{_format_error_summary(relion_gt_view_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RELION  vs GT in-plane: "
                            f"{_format_error_summary(relion_gt_inplane_err_deg, '°', [2, 5, 10])}"
                        )
                if best_trans_hist and i_iter < len(best_trans_hist) and best_trans_hist[i_iter] is not None:
                    best_trans_arr = np.asarray(best_trans_hist[i_iter], dtype=np.float64)
                    recovar_trans_orig = np.full((n_total, 2), np.nan, dtype=np.float64)
                    recovar_trans_orig[half1_indices] = best_trans_arr[:n_h1]
                    recovar_trans_orig[half2_indices] = best_trans_arr[n_h1:]
                    recovar_gt_trans_err_px = None
                    relion_gt_trans_err_px = None
                    valid_trans = ~(np.isnan(recovar_trans_orig).any(axis=1) | np.isnan(relion_trans_orig).any(axis=1))
                    if np.any(valid_trans):
                        trans_err_px = np.linalg.norm(
                            recovar_trans_orig[valid_trans] - relion_trans_orig[valid_trans],
                            axis=1,
                        )
                        trans_err_ang = trans_err_px * pixel_size
                        print(
                            "  Translation error: "
                            f"{_format_error_summary(trans_err_px, ' px', [0.25, 0.5, 1.0])} "
                            f"(mean={trans_err_ang.mean():.4f} A)"
                        )
                    else:
                        trans_err_px = None
                    if gt_translations_orig is not None:
                        valid_recovar_gt_trans = ~(
                            np.isnan(recovar_trans_orig).any(axis=1) | np.isnan(gt_translations_orig).any(axis=1)
                        )
                        if np.any(valid_recovar_gt_trans):
                            recovar_gt_trans_err_px = np.linalg.norm(
                                recovar_trans_orig[valid_recovar_gt_trans]
                                - gt_translations_orig[valid_recovar_gt_trans],
                                axis=1,
                            )
                            print(
                                "  RECOVAR vs GT translation: "
                                f"{_format_error_summary(recovar_gt_trans_err_px, ' px', [0.25, 0.5, 1.0])}"
                            )
                        valid_relion_gt_trans = ~(
                            np.isnan(relion_trans_orig).any(axis=1) | np.isnan(gt_translations_orig).any(axis=1)
                        )
                        if np.any(valid_relion_gt_trans):
                            relion_gt_trans_err_px = np.linalg.norm(
                                relion_trans_orig[valid_relion_gt_trans] - gt_translations_orig[valid_relion_gt_trans],
                                axis=1,
                            )
                            print(
                                "  RELION  vs GT translation: "
                                f"{_format_error_summary(relion_gt_trans_err_px, ' px', [0.25, 0.5, 1.0])}"
                            )
                else:
                    trans_err_px = None
                    recovar_gt_trans_err_px = None
                    relion_gt_trans_err_px = None

                pose_path = os.path.join(out_dir, f"pose_comparison_iter{i_iter:03d}.npz")
                np.savez(
                    pose_path,
                    recovar_eulers=recovar_eulers_orig,
                    relion_eulers=relion_eulers_orig,
                    angular_error_deg=ang_err_deg if ang_err_deg is not None else np.array([]),
                    view_direction_error_deg=view_err_deg if view_err_deg is not None else np.array([]),
                    inplane_error_deg=inplane_err_deg if inplane_err_deg is not None else np.array([]),
                    gt_rotations=gt_rotations_orig if gt_rotations_orig is not None else np.array([]),
                    gt_translations=gt_translations_orig if gt_translations_orig is not None else np.array([]),
                    gt_transpose_relion_convention=np.array(
                        gt_transpose_relion_convention if gt_transpose_relion_convention is not None else False,
                        dtype=np.bool_,
                    ),
                    recovar_vs_gt_angular_error_deg=(
                        recovar_gt_ang_err_deg if recovar_gt_ang_err_deg is not None else np.array([])
                    ),
                    recovar_vs_gt_view_direction_error_deg=(
                        recovar_gt_view_err_deg if recovar_gt_view_err_deg is not None else np.array([])
                    ),
                    recovar_vs_gt_inplane_error_deg=(
                        recovar_gt_inplane_err_deg if recovar_gt_inplane_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_angular_error_deg=(
                        relion_gt_ang_err_deg if relion_gt_ang_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_view_direction_error_deg=(
                        relion_gt_view_err_deg if relion_gt_view_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_inplane_error_deg=(
                        relion_gt_inplane_err_deg if relion_gt_inplane_err_deg is not None else np.array([])
                    ),
                    recovar_translations=recovar_trans_orig if recovar_trans_orig is not None else np.array([]),
                    relion_translations=relion_trans_orig,
                    translation_error_px=trans_err_px if trans_err_px is not None else np.array([]),
                    recovar_vs_gt_translation_error_px=(
                        recovar_gt_trans_err_px if recovar_gt_trans_err_px is not None else np.array([])
                    ),
                    relion_vs_gt_translation_error_px=(
                        relion_gt_trans_err_px if relion_gt_trans_err_px is not None else np.array([])
                    ),
                    half1_indices=half1_indices,
                    half2_indices=half2_indices,
                )
                print(f"  Saved pose comparison: {pose_path}")

    if gt_ft is not None:
        print("\n=== Postprocessing per-iteration map quality vs GT/RELION ===")
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "scripts/postprocess_multi_iter_gt.py",
                "--recovar_dir",
                out_dir,
                "--relion_dir",
                str(relion_dir),
                "--relion_start_iter",
                str(iteration),
                "--gt_volume",
                str(gt_path),
                "--max_iter",
                str(args.max_iter),
            ],
            check=True,
        )

    # ---- Run diff script ----
    print("\n=== Running diff_relion_recovar_per_iter.py ===")
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "scripts/diff_relion_recovar_per_iter.py",
            "--relion_dir",
            str(relion_dir),
            "--recovar_dir",
            out_dir,
            "--relion_start_iter",
            str(iteration),
            "--max_iter",
            str(completed_iters + 1),
            "--tol",
            "0.05",
            "--shells",
            "10",
        ]
    )


if __name__ == "__main__":
    main()
