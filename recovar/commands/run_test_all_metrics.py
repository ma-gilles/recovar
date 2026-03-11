import os
import sys
import json
import pickle
import argparse
import math
import time
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import jax
import psutil

from recovar import utils
from recovar.simulation import simulator, synthetic_dataset
from recovar.output import output, metrics, plot_utils
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.commands import pipeline, compute_state

LOWER_IS_BETTER_TOKENS = (
    "error",
    "locres",
    "angle",
    "loss",
    "rmse",
    "mse",
    "bias",
    "constrast",
    "contrast",
)

HIGHER_IS_BETTER_TOKENS = (
    "fsc",
    "correlation",
    "variance_explained",
    "relative_variance",
)

# Canonical key renames: old_key -> new_key.
# Both old and new keys are emitted for backward compatibility; regression
# comparison deduplicates so each metric is only checked once.
CANONICAL_KEY_ALIASES = {
    "pcs_relative_variance_4": "svd_relative_variance_4",
    "pcs_relative_variance_10": "svd_relative_variance_10",
    "contrasts_4": "contrast_abs_error_4",
    "contrasts_4_noreg": "contrast_abs_error_4_noreg",
    "contrasts_10": "contrast_abs_error_10",
    "contrasts_10_noreg": "contrast_abs_error_10_noreg",
    "constrasts_4": "contrast_abs_error_4",
    "constrasts_4_noreg": "contrast_abs_error_4_noreg",
    "constrasts_10": "contrast_abs_error_10",
    "constrasts_10_noreg": "contrast_abs_error_10_noreg",
}

def _resolve_canonical_key(key):
    """Map a key to its canonical name (handles both static aliases and dynamic locres patterns)."""
    if key in CANONICAL_KEY_ALIASES:
        return CANONICAL_KEY_ALIASES[key]
    # Dynamic locres renames: state_N_ninety_pc_locres -> state_N_locres_90pct
    #                         state_N_median_locres   -> state_N_locres_median
    import re
    m = re.match(r"^(state_\d+)_ninety_pc_locres$", key)
    if m:
        return f"{m.group(1)}_locres_90pct"
    m = re.match(r"^(state_\d+)_median_locres$", key)
    if m:
        return f"{m.group(1)}_locres_median"
    return key


# Set up logging configuration
def setup_logging(output_dir):
    from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler
    log_file = os.path.join(output_dir, 'run_test.log')
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            RobustFileHandler(log_file),
            RobustStreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ── Performance tracking helpers ──────────────────────────────────────

def _gpu_name():
    """Return GPU device name or 'cpu'."""
    try:
        devs = jax.devices("gpu")
        if devs:
            return str(devs[0].device_kind)
    except Exception:
        pass
    return "cpu"


def _perf_snapshot():
    """Capture a wall-clock + memory snapshot.

    Returns a dict with:
    - wall_time: time.monotonic() value
    - cpu_rss_bytes: current process RSS
    - gpu_bytes_in_use: JAX bytes currently allocated (or 0)
    - gpu_peak_bytes: JAX peak_bytes_in_use (cumulative, or 0)
    """
    snap = {
        "wall_time": time.monotonic(),
        "cpu_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
        "gpu_bytes_in_use": 0,
        "gpu_peak_bytes": 0,
    }
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats:
            snap["gpu_bytes_in_use"] = stats.get("bytes_in_use", 0)
            snap["gpu_peak_bytes"] = stats.get("peak_bytes_in_use", 0)
    except Exception:
        pass
    return snap


def _stage_perf(before, after):
    """Compute per-stage performance from two snapshots.

    GPU peak during stage: JAX's ``peak_bytes_in_use`` is cumulative and
    cannot be reset.  We use the following heuristic:

    - If the global peak *increased* during this stage
      (``after.gpu_peak > before.gpu_peak``), the new peak happened during
      this stage: ``stage_peak = after.gpu_peak``.
    - Otherwise the peak was set by an earlier stage, so our best estimate
      is ``max(before.gpu_in_use, after.gpu_in_use)`` — the higher of the
      two endpoint measurements.

    This gives the absolute peak GPU allocation (not delta) attributable
    to the stage.  ``peak_gpu_memory_gb`` is therefore comparable across
    stages and across runs.
    """
    wall = after["wall_time"] - before["wall_time"]
    cpu_peak_bytes = max(before["cpu_rss_bytes"], after["cpu_rss_bytes"])
    if after["gpu_peak_bytes"] > before["gpu_peak_bytes"]:
        gpu_stage_peak = after["gpu_peak_bytes"]
    else:
        gpu_stage_peak = max(before["gpu_bytes_in_use"], after["gpu_bytes_in_use"])
    return {
        "wall_seconds": round(wall, 2),
        "peak_cpu_memory_gb": round(cpu_peak_bytes / 1e9, 3),
        "peak_gpu_memory_gb": round(gpu_stage_peak / 1e9, 3),
    }


# ---------------------------------------------------------------------------
# PDB-based trajectory volume generation
# ---------------------------------------------------------------------------
from recovar.simulation.trajectory_generation import (
    CHAIN_GROUPS_5NRL as _5NRL_CHAIN_GROUPS,
    split_atom_group_by_chains as _split_atom_group_by_chains,
    generate_trajectory_volumes as generate_pdb_trajectory_volumes,
)
def generate_compact_support_test_volumes(
    output_dir,
    grid_size=128,
    n_volumes=50,
    voxel_size=4.25,
    prefix_name="vol",
    output_prefix=None,
):
    """
    Generate deterministic real-space MRC volumes with compact support.

    Geometry:
    - A static chain of Gaussian/ball-like blobs on one line.
    - One additional compact ball moving horizontally on a parallel line.

    Returns
    -------
    str
        Prefix path to generated files, suitable for --volume-input
        (e.g., "<...>/generated_volumes/vol" for files vol0000.mrc, ...).
    """
    if output_prefix is None:
        vols_dir = Path(output_dir) / "generated_volumes"
        output.mkdir_safe(str(vols_dir))
        volume_prefix = str(vols_dir / prefix_name)
    else:
        volume_prefix = str(output_prefix)
        output.mkdir_safe(str(Path(volume_prefix).parent))

    # Normalized coordinate grid in [-1, 1]^3.
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2 + zz**2)

    # Soft compact support mask to keep maps object-like.
    support = np.clip((0.88 - rr) / 0.08, 0.0, 1.0)
    support = support**2

    # Static line of compact balls on y=0, z=-0.15.
    static_xs = np.array([-0.55, -0.30, -0.05, 0.20, 0.45], dtype=np.float32)
    static_y = 0.0
    static_z = -0.15
    static_radii = np.array([0.13, 0.11, 0.12, 0.11, 0.13], dtype=np.float32)
    static_edges = np.array([0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
    static_amps = np.array([1.00, 0.85, 0.95, 0.80, 0.90], dtype=np.float32)

    for idx in range(n_volumes):
        t = idx / max(n_volumes - 1, 1)
        vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        # Static compact balls with mild high-frequency texture.
        for cx, radius, edge, amp in zip(static_xs, static_radii, static_edges, static_amps):
            sr = np.sqrt((xx - cx) ** 2 + (yy - static_y) ** 2 + (zz - static_z) ** 2)
            static_ball = np.clip((radius - sr) / edge, 0.0, 1.0)
            static_tex = (
                np.cos((xx - cx) * (10.0 * np.pi))
                * np.cos((yy - static_y) * (8.0 * np.pi))
                * np.cos((zz - static_z) * (9.0 * np.pi))
            )
            vol += amp * (static_ball + 0.25 * static_ball * static_tex)

        # Moving compact ball on a different (parallel) line y=0.32, z=0.18.
        moving_x = -0.70 + 1.40 * t
        moving_y = 0.32
        moving_z = 0.18
        moving_radius = 0.16
        moving_edge = 0.03
        moving_r = np.sqrt((xx - moving_x) ** 2 + (yy - moving_y) ** 2 + (zz - moving_z) ** 2)
        moving_ball = np.clip((moving_radius - moving_r) / moving_edge, 0.0, 1.0)

        # Add high-resolution content in the moving piece:
        # a compact ripple texture whose phase drifts over states.
        # This boosts high-frequency Fourier content for resolution-metric tests
        # while keeping support local and physically bounded.
        phase = 2.0 * np.pi * t
        hf_osc = (
            np.cos((xx - moving_x) * (18.0 * np.pi) + phase)
            * np.cos((yy - moving_y) * (14.0 * np.pi) - 0.5 * phase)
            * np.cos((zz - moving_z) * (16.0 * np.pi) + 0.25 * phase)
        )
        moving_component = 1.20 * moving_ball + 0.45 * moving_ball * hf_osc
        vol += moving_component

        # Apply compact support and normalize scale.
        vol *= support
        vol -= np.mean(vol)
        norm = np.linalg.norm(vol.ravel())
        if norm > 0:
            vol /= norm

        utils.write_mrc(f"{volume_prefix}{idx:04d}.mrc", vol.astype(np.float32), voxel_size=voxel_size)

    return volume_prefix

##TODO use TMP_RECOVAR_DIR if set? (see staging.py)
def validate_storage_args_for_generated_volumes(args, argv):
    """
    Enforce explicit output location when auto-generating volumes.
    """
    if args.volume_input is not None:
        return
    has_explicit_outdir = False
    for tok in argv:
        if tok in ("--output-dir", "-o"):
            has_explicit_outdir = True
            break
        if tok.startswith("--output-dir=") or tok.startswith("-o="):
            has_explicit_outdir = True
            break
    if not has_explicit_outdir:
        raise ValueError(
            "When --volume-input is omitted (auto-generated volumes), you must pass --output-dir/-o "
            "explicitly to avoid unintended storage locations."
        )


def make_big_test_dataset(input_dir, output_dir, noise_level=0.1, grid_size=128, n_images=50000,
                          contrast_std=0.1, n_tilts=-1, premultiplied_ctf=False, noise_increase_per_tilt=None):
    output_folder = os.path.join(output_dir, 'test_dataset')
    output.mkdir_safe(output_folder)
    from scipy.stats import vonmises

    # Count available volumes from prefix input_dir + "####.mrc" to match simulator loader behavior.
    n_states = 0
    while os.path.isfile(f"{input_dir}{n_states:04d}.mrc"):
        n_states += 1
    if n_states == 0:
        raise ValueError(
            f"No volumes found for prefix {input_dir}. Expected files like {input_dir}0000.mrc, {input_dir}0001.mrc, ..."
        )

    # Define density that volumes are resampled from.
    def p(x):
        means = [np.pi/2, np.pi, 3*np.pi/2]
        kappas =  [6.0, 6.0, 6.0]
        weights = np.array([2.0, 1.0, 2.0])
        weights /= sum(weights)  
        val = 0
        for i in range(3): 
            val += weights[i]*vonmises.pdf(x, loc=means[i], kappa=kappas[i])
        return val

    x = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    volume_distribution = p(x)
    volume_distribution /= (np.sum(volume_distribution))


    
    voxel_size = 4.25 * 128 / grid_size
    _image_stack, sim_info = simulator.generate_synthetic_dataset(
        output_folder, voxel_size, input_dir, int(n_images),
        outlier_file_input=None, grid_size=grid_size,
        volume_distribution=volume_distribution, dataset_params_option="uniform",
        noise_level=noise_level, noise_model="radial1", put_extra_particles=False,
        percent_outliers=0.0, volume_radius=0.7, trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0, contrast_std=contrast_std, disc_type='cubic',
        n_tilts=n_tilts, premultiplied_ctf=premultiplied_ctf, noise_increase_per_tilt=noise_increase_per_tilt)

    logging.info("Finished generating dataset %s", output_folder)
    return sim_info


def compute_noise_variance_metrics(
    gt_noise_base,
    est_noise,
    plots_dir,
    logger,
    dose_indices=None,
    noise_increase_per_tilt=None,
):
    gt_noise_base = None if gt_noise_base is None else np.asarray(gt_noise_base).reshape(-1)
    est_noise = None if est_noise is None else np.asarray(est_noise)
    if est_noise is not None and est_noise.ndim == 0:
        est_noise = est_noise.reshape(1)

    def _safe_corrcoef(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            logger.warning("Empty vectors for noise correlation; using 0.0")
            return 0.0
        # Avoid np.corrcoef warnings/NaN when either input has (near-)zero variance.
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            logger.warning("Near-constant vector for noise correlation; using 0.0")
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(corr):
            logger.warning("Non-finite noise correlation encountered; replacing with 0.0")
            return 0.0
        return corr

    scores = {}
    if gt_noise_base is None:
        logger.warning("No ground truth noise variance found in simulation info")
        return scores
    if est_noise is None:
        logger.warning("No estimated noise variance found in pipeline output")
        return scores

    logger.info("Ground truth noise shape: %s", gt_noise_base.shape)
    logger.info("Estimated noise shape: %s", est_noise.shape if isinstance(est_noise, np.ndarray) else 'not array')

    if isinstance(est_noise, np.ndarray) and est_noise.ndim > 1:
        logger.info("Processing variable noise per tilt...")
        if dose_indices is None:
            logger.warning("No dose indices found for variable noise comparison")
            return scores
        if len(dose_indices) == 0:
            logger.warning("Empty dose indices for variable noise comparison")
            return scores

        unique_tilts, tilt_counts = np.unique(dose_indices, return_counts=True)
        n_tilts = len(unique_tilts)
        tilt_correlations = []
        tilt_mean_errors = []
        tilt_median_errors = []

        fig, axes = plt.subplots(n_tilts, 1, figsize=(10, 4 * n_tilts))
        if n_tilts == 1:
            axes = [axes]

        for i, tilt_idx in enumerate(unique_tilts):
            if noise_increase_per_tilt is not None:
                tilt_scale = 1 + noise_increase_per_tilt * tilt_idx
                tilt_gt_noise = gt_noise_base * tilt_scale
            else:
                tilt_scale = None
                tilt_gt_noise = gt_noise_base

            # Prefer direct dose-index lookup when available; otherwise fall back
            # to row position for datasets where dose labels are non-contiguous.
            if 0 <= int(tilt_idx) < est_noise.shape[0]:
                tilt_est_noise = est_noise[int(tilt_idx)]
            elif i < est_noise.shape[0]:
                logger.warning(
                    "Dose index %s is out of est_noise bounds %s; falling back to row %s.",
                    int(tilt_idx),
                    est_noise.shape[0],
                    i,
                )
                tilt_est_noise = est_noise[i]
            else:
                logger.warning(
                    "Skipping dose index %s because est_noise has only %s rows.",
                    int(tilt_idx),
                    est_noise.shape[0],
                )
                axes[i].set_title(
                    f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images) - skipped'
                )
                axes[i].text(
                    0.02,
                    0.98,
                    "Skipped: no matching estimated row",
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
                continue
            min_len = min(len(tilt_gt_noise), len(tilt_est_noise))
            if min_len == 0:
                logger.warning(
                    "Skipping tilt %s because overlapping noise profile length is zero.",
                    int(tilt_idx),
                )
                axes[i].set_title(
                    f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images) - skipped'
                )
                axes[i].text(
                    0.02,
                    0.98,
                    "Skipped: zero-length overlap",
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
                continue
            tilt_gt_noise = tilt_gt_noise[:min_len]
            tilt_est_noise = tilt_est_noise[:min_len]

            noise_relative_error = np.abs(tilt_est_noise - tilt_gt_noise) / (np.abs(tilt_gt_noise) + 1e-10)
            tilt_correlations.append(_safe_corrcoef(tilt_est_noise, tilt_gt_noise))
            tilt_mean_errors.append(float(np.mean(noise_relative_error)))
            tilt_median_errors.append(float(np.median(noise_relative_error)))

            ax = axes[i]
            ax.plot(tilt_gt_noise, label='Ground Truth', alpha=0.7)
            ax.plot(tilt_est_noise, label='Estimated', alpha=0.7)
            ax.set_xlabel('Radial Frequency Index')
            ax.set_ylabel('Noise Variance')
            if tilt_scale is None:
                ax.set_title(f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images)')
            else:
                ax.set_title(
                    f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images, scale={tilt_scale:.2f})'
                )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(
                0.02,
                0.98,
                (
                    f'Correlation: {tilt_correlations[-1]:.3f}\n'
                    f'Mean Rel. Error: {tilt_mean_errors[-1]:.3f}\n'
                    f'Median Rel. Error: {tilt_median_errors[-1]:.3f}'
                ),
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        plt.tight_layout()
        noise_plot_path = os.path.join(plots_dir, "noise_variance_comparison_per_tilt.png")
        plt.savefig(noise_plot_path)
        plt.close()
        logger.info("Noise variance comparison plot (per tilt) saved to: %s", noise_plot_path)

        if len(tilt_mean_errors) == 0:
            logger.warning("No valid per-tilt noise rows were processed; skipping aggregate metrics.")
            return scores

        scores['noise_mean_relative_error'] = float(np.mean(tilt_mean_errors))
        scores['noise_median_relative_error'] = float(np.mean(tilt_median_errors))
        scores['noise_max_relative_error'] = float(np.max(tilt_mean_errors))
        scores['noise_correlation'] = float(np.mean(tilt_correlations))
        scores['noise_correlation_per_tilt'] = list(tilt_correlations)
        scores['noise_mean_error_per_tilt'] = list(tilt_mean_errors)
        scores['noise_median_error_per_tilt'] = list(tilt_median_errors)
        return scores

    min_len = min(len(gt_noise_base), len(est_noise))
    if min_len == 0:
        logger.warning("Skipping single-noise comparison because overlapping length is zero.")
        return scores
    gt_noise_base = gt_noise_base[:min_len]
    est_noise = est_noise[:min_len]

    noise_relative_error = np.abs(est_noise - gt_noise_base) / (np.abs(gt_noise_base) + 1e-10)
    noise_correlation = _safe_corrcoef(est_noise, gt_noise_base)

    scores['noise_mean_relative_error'] = np.mean(noise_relative_error)
    scores['noise_median_relative_error'] = np.median(noise_relative_error)
    scores['noise_max_relative_error'] = np.max(noise_relative_error)
    scores['noise_correlation'] = noise_correlation

    plt.figure(figsize=(10, 6))
    plt.plot(gt_noise_base, label='Ground Truth', alpha=0.7)
    plt.plot(est_noise, label='Estimated', alpha=0.7)
    plt.xlabel('Radial Frequency Index')
    plt.ylabel('Noise Variance')
    plt.title('Noise Variance Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.98,
        (
            f'Correlation: {noise_correlation:.3f}\n'
            f'Mean Rel. Error: {np.mean(noise_relative_error):.3f}\n'
            f'Median Rel. Error: {np.median(noise_relative_error):.3f}'
        ),
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    noise_plot_path = os.path.join(plots_dir, "noise_variance_comparison.png")
    plt.savefig(noise_plot_path)
    plt.close()
    logger.info("Noise variance comparison plot saved to: %s", noise_plot_path)
    return scores


def metric_direction(metric_name):
    name = metric_name.lower()
    if any(tok in name for tok in LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(tok in name for tok in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return "ignore"


def compare_metric(current, baseline, direction, tol_frac):
    if not (math.isfinite(current) and math.isfinite(baseline)):
        return False, f"non-finite values current={current} baseline={baseline}"
    scale = max(abs(baseline), 1e-12)
    delta = (current - baseline) / scale
    if direction == "lower":
        ok = delta <= tol_frac
        msg = f"increase={delta:.4f} allowed={tol_frac:.4f}"
        return ok, msg
    if direction == "higher":
        ok = delta >= -tol_frac
        msg = f"drop={-delta:.4f} allowed={tol_frac:.4f}"
        return ok, msg
    return True, "ignored"


def _sanitize_json_value(val):
    """Replace NaN/Inf with None so json.dump produces valid JSON."""
    if isinstance(val, float):
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(val, list):
        return [_sanitize_json_value(v) for v in val]
    return val


def normalize_scores_for_json(scores_dict):
    normalized = {}
    for key, val in scores_dict.items():
        if isinstance(val, dict):
            normalized[key] = normalize_scores_for_json(val)
        elif isinstance(val, (bool, np.bool_)):
            normalized[key] = bool(val)
        elif isinstance(val, np.ndarray):
            normalized[key] = _sanitize_json_value(np.asarray(val).tolist())
        elif isinstance(val, (np.floating, np.integer)):
            normalized[key] = _sanitize_json_value(float(val))
        elif isinstance(val, (float, int)):
            normalized[key] = _sanitize_json_value(float(val))
        elif isinstance(val, str):
            normalized[key] = val
        else:
            # Handle JAX ArrayImpl and other array-like objects
            try:
                arr = np.asarray(val)
                result = float(arr) if arr.ndim == 0 else arr.tolist()
                normalized[key] = _sanitize_json_value(result)
            except Exception:
                normalized[key] = val
    return normalized


def resolve_metrics_baseline_path(args):
    if args.metrics_baseline_json is not None:
        return Path(args.metrics_baseline_json)
    if getattr(args, 'generate_pdb_volumes', False):
        return Path(args.output_dir) / "generated_volumes" / (
            f"metrics_baseline_pdb_grid{args.grid_size}_nvol{args.generated_n_volumes}.json"
        )
    if args.generate_volumes:
        return Path(args.output_dir) / "generated_volumes" / (
            f"metrics_baseline_grid{args.grid_size}_nvol{args.generated_n_volumes}.json"
        )
    return None


def compare_scores_against_baseline(current_scores, baseline_scores, tol_frac):
    checked = 0
    failures = []
    details = {}
    # Track which canonical keys have already been checked so we don't
    # double-count a metric that appears under both its old and new name.
    seen_canonical = set()
    for key in sorted(set(current_scores.keys()) & set(baseline_scores.keys())):
        cur = current_scores[key]
        base = baseline_scores[key]
        if isinstance(cur, (bool, np.bool_)) or isinstance(base, (bool, np.bool_)):
            continue
        if isinstance(cur, dict) or isinstance(base, dict):
            continue
        if not isinstance(cur, (int, float, np.floating, np.integer)):
            continue
        if not isinstance(base, (int, float, np.floating, np.integer)):
            continue
        direction = metric_direction(key)
        if direction == "ignore":
            continue
        # Deduplicate: if this key is a legacy alias for a canonical key
        # that was already checked, skip it.
        canonical = _resolve_canonical_key(key)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        checked += 1
        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=tol_frac)
        details[key] = {
            "current": float(cur),
            "baseline": float(base),
            "direction": direction,
            "ok": bool(ok),
            "message": msg,
        }
        if not ok:
            failures.append(f"{key}: current={float(cur):.6g} baseline={float(base):.6g} ({msg})")
    return checked, failures, details


def load_u_real_for_metrics(pipeline_output, n_pcs):
    """
    Load only the requested number of real-space eigen volumes when supported.
    Falls back to legacy `get('u_real')` API.
    """
    n_pcs = int(n_pcs)
    if n_pcs <= 0:
        raise ValueError(f"n_pcs must be positive, got {n_pcs}")
    if hasattr(pipeline_output, "get_u_real"):
        return np.asarray(pipeline_output.get_u_real(n_pcs))
    return np.asarray(pipeline_output.get('u_real')[:n_pcs])


def load_unsorted_embedding_component(pipeline_output, entry, key, legacy_cache=None):
    """
    Load one unsorted embedding component with minimal I/O when supported.

    Prefer `PipelineOutput.get_embedding_component(entry, key)` to avoid loading
    the entire embeddings payload. Fall back to legacy `get('unsorted_embedding')`.
    """
    if legacy_cache is None:
        legacy_cache = {}

    cache_key = ("component", entry, key)
    if cache_key in legacy_cache:
        return legacy_cache[cache_key]

    if hasattr(pipeline_output, "get_embedding_component"):
        value = np.asarray(pipeline_output.get_embedding_component(entry, key))
    else:
        if "__root__" not in legacy_cache:
            legacy_cache["__root__"] = pipeline_output.get('unsorted_embedding')
        value = np.asarray(legacy_cache["__root__"][entry][key])

    legacy_cache[cache_key] = value
    return value


def select_state_target_latent_points(unsorted_zs, particle_assignment, preferred_labels, max_points=2):
    """
    Pick representative latent points for compute_state without producing NaNs.

    Preferred labels are used when present; otherwise, non-empty labels with the
    highest particle counts are used as fallback.
    """
    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")

    zs = np.asarray(unsorted_zs)
    if zs.ndim != 2:
        raise ValueError(f"unsorted_zs must be 2D, got shape {zs.shape}")

    labels = np.asarray(particle_assignment).reshape(-1)
    if labels.size != zs.shape[0]:
        raise ValueError(
            f"Length mismatch: particle_assignment has {labels.size} items, "
            f"but unsorted_zs has {zs.shape[0]} rows."
        )
    if labels.size == 0:
        raise ValueError("particle_assignment is empty.")

    if not np.issubdtype(labels.dtype, np.integer):
        rounded = np.round(labels)
        if not np.allclose(labels, rounded):
            raise ValueError("particle_assignment must be integer-like.")
        labels = rounded.astype(np.int64)
    else:
        labels = labels.astype(np.int64, copy=False)

    finite_row_mask = np.all(np.isfinite(zs), axis=1)
    if not np.any(finite_row_mask):
        raise ValueError("All rows in unsorted_zs are non-finite.")
    if not np.all(finite_row_mask):
        zs = zs[finite_row_mask]
        labels = labels[finite_row_mask]

    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 0:
        raise ValueError("No non-empty labels available for latent point selection.")

    selected_labels = []
    present = set(unique_labels.tolist())
    for label in preferred_labels:
        label_int = int(label)
        if label_int in present and label_int not in selected_labels:
            selected_labels.append(label_int)
            if len(selected_labels) >= max_points:
                break

    # Fill remaining slots by largest class size, tie-broken by smaller label.
    if len(selected_labels) < max_points:
        order = np.lexsort((unique_labels, -counts))
        for idx in order:
            label_int = int(unique_labels[idx])
            if label_int not in selected_labels:
                selected_labels.append(label_int)
            if len(selected_labels) >= max_points:
                break

    selected_points = []
    used_labels = []
    for label_int in selected_labels:
        label_mask = labels == label_int
        if not np.any(label_mask):
            continue
        z_mean = np.mean(zs[label_mask], axis=0)
        if np.all(np.isfinite(z_mean)):
            selected_points.append(np.asarray(z_mean, dtype=np.float32))
            used_labels.append(label_int)

    if not selected_points:
        raise ValueError("Could not compute finite latent target points from assignments.")

    return np.asarray(selected_points, dtype=np.float32), used_labels


def main():
    argv = list(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run tests for recovar")
    parser.add_argument('--volume-input', '-i', required=False, default=None,
                        help='Input volume prefix containing files like <prefix>0000.mrc, <prefix>0001.mrc, ...')
    parser.add_argument('--output-dir', '-o', default='/tmp/recovar_test_all_metrics')
    parser.add_argument('--no-delete', action='store_true',
                        help='Do not delete the test dataset directory after successful tests')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (skip GPU check)')
    parser.add_argument('--n-images', type=float, default=5e4, help='Number of images in the test dataset')
    parser.add_argument('--grid-size', type=int, default=64, help='Grid size for the test dataset (default: 64, matching 5nrl notebook)')
    parser.add_argument('--tomo-tilts', type=int, default=-1,
                        help='Number of tilts in tomography (default: -1 for no tilts for cryo-EM)')
    parser.add_argument('--contrast-std', type=float, default=0.1,
                        help='Standard deviation of contrast for the test dataset')
    parser.add_argument('--premultiplied-ctf', action='store_true',
                        help='Use premultiplied CTF for the test dataset')
    parser.add_argument('--noise-increase-per-tilt', default=None, type=float,
                        help= 'Noise increase per tilt in the test dataset')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='Noise level for the test dataset')
    parser.add_argument('--noise-model', type=str, default='radial',
                        help='Noise model for the test dataset')
    parser.add_argument('--new-noise-est', action='store_true',
                        help='Use new noise estimation method')
    parser.add_argument('--generate-volumes', action='store_true',
                        help='Generate synthetic compact-support test volumes if you do not want to provide --volume-input.')
    parser.add_argument('--generate-pdb-volumes', action='store_true',
                        help='Generate test volumes from PDB 5nrl (spliceosome) with rigid-body motion '
                             'trajectory (subcomplexes rotated). More realistic than --generate-volumes.')
    parser.add_argument('--pdb-path', type=str, default=None,
                        help='Path to PDB/CIF file for --generate-pdb-volumes. Default: assets/5nrl.cif.')
    parser.add_argument('--pdb-bfactor', type=float, default=80.0,
                        help='B-factor for PDB volume generation (default: 80).')
    parser.add_argument('--pdb-max-rotation', type=float, default=10.0,
                        help='Maximum rotation angle in degrees for PDB trajectory (default: 10).')
    parser.add_argument('--generated-n-volumes', type=int, default=50,
                        help='Number of generated test volumes when --generate-volumes is used (default: 50).')
    parser.add_argument('--generated-volumes-prefix', type=str, default=None,
                        help='Optional generated volume prefix path (default: <output-dir>/generated_volumes/vol).')
    parser.add_argument('--metrics-baseline-json', type=str, default=None,
                        help='Path to baseline all_scores JSON. If omitted with generated volumes, a default baseline file under generated_volumes is used.')
    parser.add_argument('--metrics-regression-tol-frac', type=float, default=0.03,
                        help='Allowed relative degradation fraction before failing regression checks (default: 0.03).')
    parser.add_argument('--skip-metrics-regression-check', action='store_true',
                        help='Do not fail the run when baseline comparison detects regressions.')
    parser.add_argument('--overwrite-metrics-baseline', action='store_true',
                        help='Overwrite baseline JSON with current scores after this run.')
    parser.add_argument('--reuse-dataset', action='store_true',
                        help='Skip dataset generation if test_dataset/ already exists in output-dir. '
                             'Useful for regression tests that reuse a saved dataset.')

    args = parser.parse_args()
    if args.grid_size <= 0:
        raise ValueError(f"--grid-size must be positive, got {args.grid_size}")
    if args.n_images <= 0:
        raise ValueError(f"--n-images must be positive, got {args.n_images}")
    if args.noise_level < 0:
        raise ValueError(f"--noise-level must be non-negative, got {args.noise_level}")
    if args.contrast_std < 0:
        raise ValueError(f"--contrast-std must be non-negative, got {args.contrast_std}")
    if args.metrics_regression_tol_frac < 0:
        raise ValueError(
            f"--metrics-regression-tol-frac must be non-negative, got {args.metrics_regression_tol_frac}"
        )
    if args.generate_volumes or args.generate_pdb_volumes or args.volume_input is None:
        if args.generated_n_volumes <= 0:
            raise ValueError(
                f"--generated-n-volumes must be positive when generating volumes, got {args.generated_n_volumes}"
            )

    validate_storage_args_for_generated_volumes(args, argv)
    output.mkdir_safe(args.output_dir)
    logger = setup_logging(args.output_dir)

    dataset_dir = os.path.join(args.output_dir, 'test_dataset')
    grid_size = args.grid_size
    n_images = args.n_images
    tilt_series = args.tomo_tilts > 0
    sim_info_path = os.path.join(dataset_dir, 'simulation_info.pkl')

    # --reuse-dataset: skip volume + dataset generation when the dataset exists.
    _reuse = args.reuse_dataset and os.path.exists(sim_info_path)

    if not _reuse:
        # Default: use PDB-based 5nrl trajectory volumes when no input is given.
        if args.volume_input is None and not args.generate_volumes:
            args.generate_pdb_volumes = True

        if args.generate_pdb_volumes:
            gen_prefix = args.generated_volumes_prefix
            if gen_prefix is None:
                gen_prefix = str(Path(args.output_dir) / "generated_volumes" / "vol")
            logger.info(
                f"Generating PDB trajectory volumes at prefix {gen_prefix} "
                f"(n={args.generated_n_volumes}, grid_size={args.grid_size}, "
                f"Bfactor={args.pdb_bfactor}, max_rot={args.pdb_max_rotation} deg)"
            )
            args.volume_input = generate_pdb_trajectory_volumes(
                output_dir=args.output_dir,
                grid_size=args.grid_size,
                n_volumes=args.generated_n_volumes,
                voxel_size=4.25 * 128 / args.grid_size,
                Bfactor=args.pdb_bfactor,
                max_rotation_degrees=args.pdb_max_rotation,
                pdb_path=args.pdb_path,
                prefix_name=Path(gen_prefix).name,
                output_prefix=gen_prefix,
            )
            logger.info("Using PDB-generated volume input prefix: %s", args.volume_input)
        elif args.generate_volumes:
            gen_prefix = args.generated_volumes_prefix
            if gen_prefix is None:
                gen_prefix = str(Path(args.output_dir) / "generated_volumes" / "vol")
            logger.info(
                f"Generating compact-support test volumes at prefix {gen_prefix} "
                f"(n={args.generated_n_volumes}, grid_size={args.grid_size})"
            )
            args.volume_input = generate_compact_support_test_volumes(
                output_dir=args.output_dir,
                grid_size=args.grid_size,
                n_volumes=args.generated_n_volumes,
                voxel_size=4.25 * 128 / args.grid_size,
                prefix_name=Path(gen_prefix).name,
                output_prefix=gen_prefix,
            )
            logger.info("Using generated volume input prefix: %s", args.volume_input)

    # Dump parser arguments to a JSON file.
    dump_json_path = os.path.join(args.output_dir, "parser_args.json")
    with open(dump_json_path, "w") as f:
        json.dump(vars(args), f, indent=2)


    def error_message(msg="An error occurred"):
        logger.error(msg)
        sys.exit(1)

    def check_gpu():
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                logger.info("GPU devices found: %s", gpu_devices)
            else:
                error_message("No GPU devices found. Please ensure JAX is properly configured with CUDA.")
        except Exception as e:
            error_message(f"Error checking GPU devices: {e}")

    if not args.cpu:
        check_gpu()

    perf = {"gpu_name": _gpu_name()}

    _snap_before_dataset = _perf_snapshot()
    if _reuse:
        logger.info("Reusing existing dataset at %s (--reuse-dataset)", dataset_dir)
        with open(sim_info_path, 'rb') as f:
            sim_info = pickle.load(f)
    else:
        # Generate synthetic test dataset
        sim_info = make_big_test_dataset(
            args.volume_input, args.output_dir, noise_level=args.noise_level,
            grid_size=grid_size, n_images=n_images,
            contrast_std=args.contrast_std, n_tilts=args.tomo_tilts,
            premultiplied_ctf=args.premultiplied_ctf,
            noise_increase_per_tilt=args.noise_increase_per_tilt
        )
    perf["dataset_generation"] = _stage_perf(_snap_before_dataset, _perf_snapshot())

    # Compute average noise radial by counting dose indices
    if 'dose_indices' in sim_info and sim_info['dose_indices'] is not None:
        unique_doses, dose_counts = np.unique(sim_info['dose_indices'], return_counts=True)
        logger.info("\nDose index distribution:")
        for dose, count in zip(unique_doses, dose_counts):
            logger.info("Dose index %s: %s images (%.1f%)", dose, count, count/len(sim_info['dose_indices'])*100)
        
        # Save dose distribution to a file
        dose_dist_path = os.path.join(dataset_dir, 'dose_distribution.txt')
        with open(dose_dist_path, 'w') as f:
            f.write("Dose index distribution:\n")
            for dose, count in zip(unique_doses, dose_counts):
                f.write(f"Dose index {dose}: {count} images ({count/len(sim_info['dose_indices'])*100:.1f}%)\n")
        logger.info("\nDose distribution saved to %s", dose_dist_path)
    else:
        logger.info("No dose indices found in simulation info - skipping dose distribution analysis")

    # Run pipeline plugin
    cmd = [
        f"{dataset_dir}/particles.{grid_size}.mrcs" if args.tomo_tilts < 0 else f"{dataset_dir}/particles.star",
        "--poses", f"{dataset_dir}/poses.pkl",
        "--ctf", f"{dataset_dir}/ctf.pkl",
        "-o", f"{dataset_dir}/pipeline_output",
        "--mask", "from_halfmaps",
    ]
    if args.noise_model == 'radial_per_tilt':
        cmd.append("--noise-model")
        cmd.append("radial_per_tilt")
    else:
        cmd.append("--noise-model")
        cmd.append("radial")
    # Add optional arguments only if they are needed
    if args.contrast_std > 0:
        cmd.append("--correct-contrast")
    if args.tomo_tilts > 0:
        cmd.append("--tilt-series")
    if args.premultiplied_ctf:
        cmd.append("--premultiplied-ctf")

    if args.new_noise_est:
        cmd.append("--new-noise-est")
    
    pipeline_parser = pipeline.add_args(argparse.ArgumentParser())
    pipeline_args = pipeline_parser.parse_args(cmd)
    logger.info("\nRunning pipeline, as if:")
    logger.info("recovar %s", " ".join(cmd))
    _snap_before_pipeline = _perf_snapshot()
    pipeline.standard_recovar_pipeline(pipeline_args)
    perf["pipeline"] = _stage_perf(_snap_before_pipeline, _perf_snapshot())



    pipeline_output_dir = os.path.join(dataset_dir, 'pipeline_output')
    sim_info_path = os.path.join(dataset_dir, 'simulation_info.pkl')
    plots_dir = os.path.join(dataset_dir, 'metrics_plot')
    output.mkdir_safe(plots_dir)

    pipeline_output = output.PipelineOutput(pipeline_output_dir)
    legacy_embedding_cache = {}
    particle_assignment = sim_info['image_assignment'] if not tilt_series else sim_info['tilt_series_assignment']

    max_classes = int(np.max(sim_info['image_assignment'])) + 1
    requested_labels = [0, max_classes // 2]
    unsorted_zs = load_unsorted_embedding_component(
        pipeline_output, 'latent_coords', 10, legacy_cache=legacy_embedding_cache
    )
    zs_assignment, labels_to_plot = select_state_target_latent_points(
        unsorted_zs=unsorted_zs,
        particle_assignment=particle_assignment,
        preferred_labels=requested_labels,
        max_points=2,
    )
    logger.info(
        "Selected state target labels %s (requested %s)",
        labels_to_plot,
        requested_labels,
    )

    # Compute state with latent points
    output_state_dir = os.path.join(pipeline_output_dir, 'state')
    latent_points_path = os.path.join(dataset_dir, 'latent_points.txt')
    np.savetxt(latent_points_path, zs_assignment)

    cs_parser = compute_state.add_args(argparse.ArgumentParser())
    cmd = [
        f"{dataset_dir}/pipeline_output",
        "-o", output_state_dir,
        "--latent-points", latent_points_path,
        "--save-all-estimates"
    ]
    cs_args = cs_parser.parse_args(cmd)

    logger.info("\nRunning compute_state, as if:")
    logger.info("recovar compute_state %s", " ".join(cmd))
    _snap_before_cs = _perf_snapshot()
    compute_state.compute_state(cs_args)
    perf["compute_state"] = _stage_perf(_snap_before_cs, _perf_snapshot())

    # Metrics and plots
    _snap_before_metrics = _perf_snapshot()
    all_scores = {}
    cryos = pipeline_output.get('lazy_dataset')
    mean = pipeline_output.get('mean')
    gt_thing = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    gt_mean = gt_thing.get_mean()

    # FSC for mean maps
    fsc_filepath = os.path.join(plots_dir, 'fsc_mean.png')
    ax, score = plot_utils.plot_fsc_new(
        gt_mean, mean,
        np.array(cryos[0].volume_shape),
        cryos[0].voxel_size,
        threshold=0.5,
        filename=fsc_filepath,
        name="Mean FSC",
        fmat=""
    )
    all_scores['mean_fsc'] = score

    # FSC for variance maps — two metrics:
    #   variance_spatial_fsc: spatial variance from eigendecomposition vs GT, DFT both, FSC
    #   variance_fourier_fsc: Fourier-space per-voxel power vs GT Fourier variance
    volume_shape = cryos[0].volume_shape
    gt_spatial_variance = gt_thing.get_spatial_variances(contrasted=False)
    estimated_spatial_variance = pipeline_output.get('variance')

    # Spatial variance FSC: DFT both spatial variance maps, then FSC
    gt_sp_dft = fourier_transform_utils.get_dft3(
        gt_spatial_variance.reshape(volume_shape)
    ).reshape(-1)
    est_sp_dft = fourier_transform_utils.get_dft3(
        estimated_spatial_variance.reshape(volume_shape)
    ).reshape(-1)
    ax, score_spatial = plot_utils.plot_fsc_new(
        gt_sp_dft, est_sp_dft,
        np.array(volume_shape),
        cryos[0].voxel_size,
        threshold=0.5,
        filename=os.path.join(plots_dir, 'fsc_variance_spatial.png'),
        name="Variance Spatial FSC",
        fmat=""
    )
    all_scores['variance_spatial_fsc'] = score_spatial
    # Keep legacy key for backward compatibility
    all_scores['variance_fsc'] = score_spatial

    # Fourier variance FSC: GT Fourier variance vs estimated from eigendecomposition
    if hasattr(gt_thing, 'get_covariance_square_root'):
        cov_sqrt_fourier = gt_thing.get_covariance_square_root(contrasted=False)
        gt_fourier_variance = np.sum(np.abs(cov_sqrt_fourier) ** 2, axis=-1)
        # Estimated Fourier variance from eigenvectors: sum_i s_i |U_i(k)|^2
        u_fourier_all = np.asarray(pipeline_output.get('u'))
        s_all_var = np.asarray(pipeline_output.get('s'))
        n_pcs_var = min(20, u_fourier_all.shape[1])
        est_fourier_variance = utils.estimate_variance(
            u_fourier_all[:, :n_pcs_var].T, s_all_var[:n_pcs_var]
        )
        ax, score_fourier = plot_utils.plot_fsc_new(
            gt_fourier_variance, est_fourier_variance,
            np.array(volume_shape),
            cryos[0].voxel_size,
            threshold=0.5,
            filename=os.path.join(plots_dir, 'fsc_variance_fourier.png'),
            name="Variance Fourier FSC",
            fmat=""
        )
        all_scores['variance_fourier_fsc'] = score_fourier

    # SVD metrics
    synt = gt_thing
    u_gt, s_gt, vh = synt.get_vol_svd(
        contrasted=False, real_space=True, random_svd_pcs=200
    )

    take_n_pcs = 20
    u = {}
    s = {}
    u_real_subset = load_u_real_for_metrics(pipeline_output, take_n_pcs)
    take_n_pcs_eff = int(u_real_subset.shape[0])
    if take_n_pcs_eff == 0:
        raise ValueError("No principal-component volumes available for metric computation.")
    vol_norm = np.sqrt(np.prod(pipeline_output.get('volume_shape')))
    u[0] = np.array(u_real_subset.reshape(take_n_pcs_eff, -1)).T * vol_norm
    s_all = np.asarray(pipeline_output.get('s'))
    s[0] = s_all[:take_n_pcs_eff] / (vol_norm ** 2)

    rel_var = {}
    norm_var = {}
    for key in u:
        if key == 'gt':
            continue
        variance, rel_var[key], norm_var[key] = metrics.get_all_variance_scores(u[key], u_gt, s_gt)
        rv4 = rel_var[key][4] if rel_var[key].size > 4 else np.nan
        rv10 = rel_var[key][10] if rel_var[key].size > 10 else np.nan
        all_scores['svd_relative_variance_4'] = rv4
        all_scores['svd_relative_variance_10'] = rv10
        # Legacy aliases for backward compatibility with old baselines.
        all_scores['pcs_relative_variance_4'] = rv4
        all_scores['pcs_relative_variance_10'] = rv10

    angles = {}
    for key in u:
        if key == 'gt':
            continue
        max_rank = int(min(20, u_gt.shape[1], u[key].shape[1]))
        angles[key] = metrics.subspace_angles(u_gt, u[key], max_rank=max_rank)

    b = 20
    def plot_dict(data_dict, title, max_size=b, log_scale=False, filename=None):
        plt.figure()
        for key, data in data_dict.items():
            max_size_this = min(max_size, data.size)
            plt.plot(np.arange(1, max_size_this + 1), data[:max_size_this], label=str(key))
        plt.legend()
        plt.title(title)
        if log_scale:
            plt.yscale('log')
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    # Still save individual plots for reference
    plot_dict(angles, 'Angle Error', filename=os.path.join(plots_dir, 'angle_error.png'))
    plot_dict(rel_var, 'Relative Variance Explained',
              filename=os.path.join(plots_dir, 'relative_variance_explained.png'))
    plot_dict(norm_var, 'Normalized Variance Explained',
              filename=os.path.join(plots_dir, 'normalized_variance_explained.png'))
    plot_dict(s, 'Eigenvalues (log scale)', log_scale=True,
              filename=os.path.join(plots_dir, 'eigs.png'))

    # Embedding variance errors
    unsorted_zs = load_unsorted_embedding_component(
        pipeline_output, 'latent_coords', 4, legacy_cache=legacy_embedding_cache
    )
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_4'] = averaged_variance

    unsorted_zs = load_unsorted_embedding_component(
        pipeline_output, 'latent_coords', 10, legacy_cache=legacy_embedding_cache
    )
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_10'] = averaged_variance

    gt_contrasts = synt.contrasts
    for zdim_val in [4, 10]:
        for noreg in [False, True]:
            entry = 'contrasts_noreg' if noreg else 'contrasts'
            label = f'{zdim_val}_noreg' if noreg else str(zdim_val)
            unsorted_contrast = load_unsorted_embedding_component(
                pipeline_output, entry, zdim_val, legacy_cache=legacy_embedding_cache
            )
            contrast_abs_error = np.mean(np.abs(gt_contrasts - unsorted_contrast))
            all_scores[f'contrast_abs_error_{label}'] = contrast_abs_error
            # Legacy aliases for backward compatibility with old baselines.
            all_scores[f'contrasts_{label}'] = contrast_abs_error
            all_scores[f'constrasts_{label}'] = contrast_abs_error

            # Create contrast comparison plot
            plt.figure(figsize=(10, 6))
            plt.scatter(gt_contrasts, unsorted_contrast, alpha=0.5, label='Particle contrasts')
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
            plt.xlabel('Ground Truth Contrast')
            plt.ylabel('Estimated Contrast')
            plt.title(f'Contrast Comparison (zdim={label})')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f'contrast_comparison_{label}.png'))
            plt.close()
        
            # Create contrast distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(gt_contrasts, bins=50, alpha=0.5, label='Ground Truth')
            plt.hist(unsorted_contrast, bins=50, alpha=0.5, label='Estimated')
            plt.xlabel('Contrast')
            plt.ylabel('Number of particles')
            plt.title(f'Contrast Distribution (zdim={label})')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f'contrast_distribution_{label}.png'))
            plt.close()

    for l_idx, l in enumerate(labels_to_plot):
        gt_map = fourier_transform_utils.get_idft3(synt.volumes[l].reshape(cryos[0].volume_shape)).real
        estimate_map = utils.load_mrc(
            Path(output_state_dir, f'state{l_idx:03d}.mrc')
        )
        errors_metrics = metrics.compute_volume_error_metrics_from_gt(
            gt_map, estimate_map, cryos[0].voxel_size, None, partial_mask=None,
            normalize_by_map1=True
        )
        all_scores[f'state_{l_idx}_locres_90pct'] = errors_metrics.get('ninety_pc_locres')
        all_scores[f'state_{l_idx}_locres_median'] = errors_metrics.get('median_locres')
        # Legacy aliases for backward compatibility with old baselines.
        all_scores[f'state_{l_idx}_ninety_pc_locres'] = errors_metrics.get('ninety_pc_locres')
        all_scores[f'state_{l_idx}_median_locres'] = errors_metrics.get('median_locres')

        # write mask to file
        mask = errors_metrics.get('mask')
        if mask is not None:
            diag_dir = os.path.join(output_state_dir, 'diagnostics', f'state{l_idx:03d}')
            os.makedirs(diag_dir, exist_ok=True)
            mask_path = os.path.join(diag_dir, 'mask.mrc')
            utils.write_mrc(mask_path, mask.astype(np.float32), voxel_size=cryos[0].voxel_size)
            logger.info("Mask written to: %s", mask_path)

    logger.info("Computing noise variance estimation metrics...")
    all_scores.update(
        compute_noise_variance_metrics(
            sim_info.get('noise_variance'),
            pipeline_output.get('noise_var_used'),
            plots_dir,
            logger,
            dose_indices=sim_info.get('dose_indices'),
            noise_increase_per_tilt=sim_info.get('noise_increase_per_tilt'),
        )
    )


    # Create a single figure with all plots
    plt.figure(figsize=(30, 20))
    
    # Plot 1: Angle Error
    plt.subplot(3, 3, 1)
    for key, data in angles.items():
        max_size_this = min(b, data.size)
        plt.plot(np.arange(1, max_size_this + 1), data[:max_size_this], label=str(key))
    plt.legend()
    plt.title('Angle Error')
    
    # Plot 2: Relative Variance Explained
    plt.subplot(3, 3, 2)
    for key, data in rel_var.items():
        max_size_this = min(b, data.size)
        plt.plot(np.arange(1, max_size_this + 1), data[:max_size_this], label=str(key))
    plt.legend()
    plt.title('Relative Variance Explained')
    
    # Plot 3: Normalized Variance Explained
    plt.subplot(3, 3, 3)
    for key, data in norm_var.items():
        max_size_this = min(b, data.size)
        plt.plot(np.arange(1, max_size_this + 1), data[:max_size_this], label=str(key))
    plt.legend()
    plt.title('Normalized Variance Explained')
    
    # Plot 4: Eigenvalues
    plt.subplot(3, 3, 4)
    for key, data in s.items():
        max_size_this = min(b, data.size)
        plt.semilogy(np.arange(1, max_size_this + 1), data[:max_size_this], label=str(key))
    plt.legend()
    plt.title('Eigenvalues (log scale)')
    
    # Plot 5: Contrast comparison for zdim=4
    plt.subplot(3, 3, 5)
    unsorted_contrast_4 = load_unsorted_embedding_component(
        pipeline_output, 'contrasts', 4, legacy_cache=legacy_embedding_cache
    )
    plt.scatter(gt_contrasts, unsorted_contrast_4, alpha=0.5, label='Particle contrasts')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    plt.xlabel('Ground Truth Contrast')
    plt.ylabel('Estimated Contrast')
    plt.title('Contrast Comparison (zdim=4)')
    plt.legend()
    
    # Plot 6: Contrast distribution for zdim=4
    plt.subplot(3, 3, 6)
    plt.hist(gt_contrasts, bins=50, alpha=0.5, label='Ground Truth')
    plt.hist(unsorted_contrast_4, bins=50, alpha=0.5, label='Estimated')
    plt.xlabel('Contrast')
    plt.ylabel('Number of particles')
    plt.title('Contrast Distribution (zdim=4)')
    plt.legend()
    
    # Plot 7: Contrast comparison for zdim=10
    plt.subplot(3, 3, 7)
    unsorted_contrast_10 = load_unsorted_embedding_component(
        pipeline_output, 'contrasts', 10, legacy_cache=legacy_embedding_cache
    )
    plt.scatter(gt_contrasts, unsorted_contrast_10, alpha=0.5, label='Particle contrasts')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    plt.xlabel('Ground Truth Contrast')
    plt.ylabel('Estimated Contrast')
    plt.title('Contrast Comparison (zdim=10)')
    plt.legend()
    
    # Plot 8: Contrast distribution for zdim=10
    plt.subplot(3, 3, 8)
    plt.hist(gt_contrasts, bins=50, alpha=0.5, label='Ground Truth')
    plt.hist(unsorted_contrast_10, bins=50, alpha=0.5, label='Estimated')
    plt.xlabel('Contrast')
    plt.ylabel('Number of particles')
    plt.title('Contrast Distribution (zdim=10)')
    plt.legend()
    
    # Plot 9: FSC scores
    plt.subplot(3, 3, 9)
    fsc_labels = ['Mean FSC', 'Var Spatial']
    fsc_vals = [all_scores['mean_fsc'], all_scores['variance_spatial_fsc']]
    if 'variance_fourier_fsc' in all_scores:
        fsc_labels.append('Var Fourier')
        fsc_vals.append(all_scores['variance_fourier_fsc'])
    plt.bar(fsc_labels, fsc_vals)
    plt.ylim(0, 1)
    plt.title('FSC Scores')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_visualizations.png'))
    plt.close()


    scores_file = os.path.join(plots_dir, "all_scores.json")
    old_scores = None
    if os.path.exists(scores_file):
        try:
            with open(scores_file, "r") as f:
                old_scores = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Could not parse previous scores file %s: %s. Skipping comparison.", scores_file, e)
    else:
        logger.info("No previous scores file found; skipping comparison.")

    if old_scores is not None:
        all_keys = set(old_scores.keys()) | set(all_scores.keys())
        old_vals, new_vals, labels = [], [], []
        for key in sorted(all_keys):
            old_val = old_scores.get(key)
            new_val = all_scores.get(key)
            old_vals.append(old_val if isinstance(old_val, (int, float)) else 0)
            new_vals.append(new_val if isinstance(new_val, (int, float)) else 0)
            labels.append(key)

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, old_vals, width, label='Old Scores')
        ax.bar(x + width / 2, new_vals, width, label='New Scores')
        ax.set_xlabel('Score Keys')
        ax.set_ylabel('Value')
        ax.set_title('Comparison of Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        comparison_plot_path = os.path.join(plots_dir, "scores_comparison.png")
        plt.savefig(comparison_plot_path)
        plt.close()
        logger.info("Score comparison plot saved at: %s", comparison_plot_path)

    perf["metrics"] = _stage_perf(_snap_before_metrics, _perf_snapshot())
    all_scores["perf"] = perf

    all_scores = normalize_scores_for_json(all_scores)

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    logger.info("All scores saved to: %s", scores_file)

    baseline_path = resolve_metrics_baseline_path(args)
    if baseline_path is None:
        logger.info("No baseline path configured (explicit --volume-input without --metrics-baseline-json).")
        return

    output.mkdir_safe(str(baseline_path.parent))
    regression_report_path = os.path.join(plots_dir, "metrics_regression_report.json")
    write_baseline = args.overwrite_metrics_baseline or (not baseline_path.exists())
    if write_baseline:
        with open(baseline_path, "w") as f:
            json.dump(all_scores, f, indent=2)
        logger.info("Metrics baseline written to: %s", baseline_path)
        with open(regression_report_path, "w") as f:
            json.dump(
                {
                    "status": "baseline_written",
                    "baseline_path": str(baseline_path),
                    "tolerance_fraction": args.metrics_regression_tol_frac,
                },
                f,
                indent=2,
            )
        return

    with open(baseline_path, "r") as f:
        baseline_scores = json.load(f)

    checked, failures, details = compare_scores_against_baseline(
        all_scores,
        baseline_scores,
        tol_frac=args.metrics_regression_tol_frac,
    )
    with open(regression_report_path, "w") as f:
        json.dump(
            {
                "status": "checked",
                "baseline_path": str(baseline_path),
                "checked_metrics": checked,
                "failures": failures,
                "details": details,
                "tolerance_fraction": args.metrics_regression_tol_frac,
            },
            f,
            indent=2,
        )

    if checked == 0:
        logger.warning("No numeric directional metrics were checked against baseline.")
        return

    if failures:
        logger.error("Metric regressions detected against baseline:")
        for failure in failures:
            logger.error("  %s", failure)
        if not args.skip_metrics_regression_check:
            error_message(
                f"{len(failures)} metric regressions detected. See {regression_report_path} "
                "or pass --skip-metrics-regression-check to continue."
            )
    else:
        logger.info("Metrics regression check passed for %s metrics (tol=%s).", checked, args.metrics_regression_tol_frac)


if __name__ == "__main__":
    main()
