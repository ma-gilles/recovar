import os
import sys
import json
import argparse
import math
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from recovar import output, metrics, plot_utils, synthetic_dataset, utils, simulator, recovar
import recovar.fourier_transform_utils as fourier_transform_utils
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
)


# Set up logging configuration
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'run_test.log')
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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


def validate_storage_args_for_generated_volumes(args, argv):
    """
    Enforce explicit output location when auto-generating volumes.
    """
    if args.volume_input is not None:
        return
    if ("--output-dir" not in argv) and ("-o" not in argv):
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
    image_stack, sim_info = simulator.generate_synthetic_dataset(
        output_folder, voxel_size, input_dir, int(n_images),
        outlier_file_input=None, grid_size=grid_size,
        volume_distribution=volume_distribution, dataset_params_option="uniform",
        noise_level=noise_level, noise_model="radial1", put_extra_particles=False,
        percent_outliers=0.0, volume_radius=0.7, trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0, contrast_std=contrast_std, disc_type='cubic',
        n_tilts=n_tilts, premultiplied_ctf=premultiplied_ctf, noise_increase_per_tilt=noise_increase_per_tilt)

    logging.info(f"Finished generating dataset {output_folder}")
    return sim_info


def compute_noise_variance_metrics(
    gt_noise_base,
    est_noise,
    plots_dir,
    logger,
    dose_indices=None,
    noise_increase_per_tilt=None,
):
    scores = {}
    if gt_noise_base is None:
        logger.warning("No ground truth noise variance found in simulation info")
        return scores

    logger.info(f"Ground truth noise shape: {gt_noise_base.shape}")
    logger.info(
        f"Estimated noise shape: {est_noise.shape if isinstance(est_noise, np.ndarray) else 'not array'}"
    )

    if isinstance(est_noise, np.ndarray) and est_noise.ndim > 1:
        logger.info("Processing variable noise per tilt...")
        if dose_indices is None:
            logger.warning("No dose indices found for variable noise comparison")
            return scores

        unique_tilts, tilt_counts = np.unique(dose_indices, return_counts=True)
        n_tilts = len(unique_tilts)
        tilt_correlations = np.zeros(n_tilts)
        tilt_mean_errors = np.zeros(n_tilts)
        tilt_median_errors = np.zeros(n_tilts)

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

            tilt_est_noise = est_noise[tilt_idx]
            min_len = min(len(tilt_gt_noise), len(tilt_est_noise))
            tilt_gt_noise = tilt_gt_noise[:min_len]
            tilt_est_noise = tilt_est_noise[:min_len]

            noise_relative_error = np.abs(tilt_est_noise - tilt_gt_noise) / (np.abs(tilt_gt_noise) + 1e-10)
            tilt_correlations[i] = np.corrcoef(tilt_est_noise, tilt_gt_noise)[0, 1]
            tilt_mean_errors[i] = np.mean(noise_relative_error)
            tilt_median_errors[i] = np.median(noise_relative_error)

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
                    f'Correlation: {tilt_correlations[i]:.3f}\n'
                    f'Mean Rel. Error: {tilt_mean_errors[i]:.3f}\n'
                    f'Median Rel. Error: {tilt_median_errors[i]:.3f}'
                ),
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        plt.tight_layout()
        noise_plot_path = os.path.join(plots_dir, "noise_variance_comparison_per_tilt.png")
        plt.savefig(noise_plot_path)
        plt.close()
        logger.info(f"Noise variance comparison plot (per tilt) saved to: {noise_plot_path}")

        scores['noise_mean_relative_error'] = np.mean(tilt_mean_errors)
        scores['noise_median_relative_error'] = np.mean(tilt_median_errors)
        scores['noise_max_relative_error'] = np.max(tilt_mean_errors)
        scores['noise_correlation'] = np.mean(tilt_correlations)
        scores['noise_correlation_per_tilt'] = tilt_correlations.tolist()
        scores['noise_mean_error_per_tilt'] = tilt_mean_errors.tolist()
        scores['noise_median_error_per_tilt'] = tilt_median_errors.tolist()
        return scores

    min_len = min(len(gt_noise_base), len(est_noise))
    gt_noise_base = gt_noise_base[:min_len]
    est_noise = est_noise[:min_len]

    noise_relative_error = np.abs(est_noise - gt_noise_base) / (np.abs(gt_noise_base) + 1e-10)
    noise_correlation = np.corrcoef(est_noise, gt_noise_base)[0, 1]

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
    logger.info(f"Noise variance comparison plot saved to: {noise_plot_path}")
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


def normalize_scores_for_json(scores_dict):
    normalized = {}
    for key, val in scores_dict.items():
        if isinstance(val, np.ndarray):
            normalized[key] = np.asarray(val).tolist()
        elif isinstance(val, (np.floating, np.integer)):
            normalized[key] = float(val)
        elif isinstance(val, (float, int)):
            normalized[key] = float(val)
        else:
            normalized[key] = val
    return normalized


def resolve_metrics_baseline_path(args):
    if args.metrics_baseline_json is not None:
        return Path(args.metrics_baseline_json)
    if args.generate_volumes:
        return Path(args.output_dir) / "generated_volumes" / (
            f"metrics_baseline_grid{args.grid_size}_nvol{args.generated_n_volumes}.json"
        )
    return None


def compare_scores_against_baseline(current_scores, baseline_scores, tol_frac):
    checked = 0
    failures = []
    details = {}
    for key in sorted(set(current_scores.keys()) & set(baseline_scores.keys())):
        cur = current_scores[key]
        base = baseline_scores[key]
        if not isinstance(cur, (int, float, np.floating, np.integer)):
            continue
        if not isinstance(base, (int, float, np.floating, np.integer)):
            continue
        direction = metric_direction(key)
        if direction == "ignore":
            continue
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
    parser.add_argument('--grid-size', type=int, default=128, help='Grid size for the test dataset')
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

    args = parser.parse_args()
    validate_storage_args_for_generated_volumes(args, argv)
    output.mkdir_safe(args.output_dir)
    logger = setup_logging(args.output_dir)

    if args.volume_input is None:
        args.generate_volumes = True

    if args.generate_volumes:
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
        logger.info(f"Using generated volume input prefix: {args.volume_input}")

    # Dump parser arguments to a JSON file.
    dump_json_path = os.path.join(args.output_dir, "parser_args.json")
    with open(dump_json_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    dataset_dir = os.path.join(args.output_dir, 'test_dataset')
    grid_size = args.grid_size
    n_images = args.n_images
    tilt_series = args.tomo_tilts > 0

    
    def error_message(msg="An error occurred"):
        logger.error(msg)
        sys.exit(1)

    def check_gpu():
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                logger.info(f"GPU devices found: {gpu_devices}")
            else:
                error_message("No GPU devices found. Please ensure JAX is properly configured with CUDA.")
        except Exception as e:
            error_message(f"Error checking GPU devices: {e}")

    if not args.cpu:
        check_gpu()

    # Generate synthetic test dataset
    sim_info = make_big_test_dataset(
        args.volume_input, args.output_dir, noise_level=args.noise_level,
        grid_size=grid_size, n_images=n_images,
        contrast_std=args.contrast_std, n_tilts=args.tomo_tilts,
        premultiplied_ctf=args.premultiplied_ctf,
        noise_increase_per_tilt=args.noise_increase_per_tilt
    )

    # Compute average noise radial by counting dose indices
    if 'dose_indices' in sim_info and sim_info['dose_indices'] is not None:
        unique_doses, dose_counts = np.unique(sim_info['dose_indices'], return_counts=True)
        logger.info("\nDose index distribution:")
        for dose, count in zip(unique_doses, dose_counts):
            logger.info(f"Dose index {dose}: {count} images ({count/len(sim_info['dose_indices'])*100:.1f}%)")
        
        # Save dose distribution to a file
        dose_dist_path = os.path.join(dataset_dir, 'dose_distribution.txt')
        with open(dose_dist_path, 'w') as f:
            f.write("Dose index distribution:\n")
            for dose, count in zip(unique_doses, dose_counts):
                f.write(f"Dose index {dose}: {count} images ({count/len(sim_info['dose_indices'])*100:.1f}%)\n")
        logger.info(f"\nDose distribution saved to {dose_dist_path}")
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
    logger.info("recovar " + " ".join(cmd))
    pipeline.standard_recovar_pipeline(pipeline_args)


    

    pipeline_output_dir = os.path.join(dataset_dir, 'pipeline_output')
    sim_info_path = os.path.join(dataset_dir, 'simulation_info.pkl')
    plots_dir = os.path.join(dataset_dir, 'metrics_plot')
    output.mkdir_safe(plots_dir)

    pipeline_output = output.PipelineOutput(pipeline_output_dir)
    unsorted_embedding = pipeline_output.get('unsorted_embedding')
    particle_assignment = sim_info['image_assignment'] if not tilt_series else sim_info['tilt_series_assignment']

    max_classes = np.max(sim_info['image_assignment']) + 1
    labels_to_plot = [0, max_classes // 2]

    unsorted_zs = unsorted_embedding['zs'][10]
    zs_assignment = np.array([
        np.mean(unsorted_zs[particle_assignment == l], axis=0)
        for l in labels_to_plot
    ])

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
    logger.info("recovar compute_state " + " ".join(cmd))

    compute_state.compute_state(cs_args)

    # Metrics and plots
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

    # FSC for variance maps
    variance_fsc_filepath = os.path.join(plots_dir, 'fsc_variance.png')
    gt_variance = gt_thing.get_spatial_variances(contrasted=False)
    estimated_variance = pipeline_output.get('variance')
    ax, score = plot_utils.plot_fsc_new(
        gt_variance, estimated_variance,
        np.array(cryos[0].volume_shape),
        cryos[0].voxel_size,
        threshold=0.5,
        filename=variance_fsc_filepath,
        name="Variance FSC",
        fmat=""
    )
    all_scores['variance_fsc'] = score

    # SVD metrics
    synt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    u_gt, s_gt, vh = synt.get_vol_svd(
        contrasted=False, real_space=True, random_svd_pcs=200
    )

    take_n_pcs = 20
    u = {}
    s = {}
    u[0] = (np.array(pipeline_output.get('u_real')[:take_n_pcs].reshape(take_n_pcs, -1)).T *
            np.sqrt(np.prod(pipeline_output.get('volume_shape'))))
    s[0] = pipeline_output.get('s') / (np.sqrt(np.prod(pipeline_output.get('volume_shape'))) ** 2)

    rel_var = {}
    norm_var = {}
    for key in u:
        if key == 'gt':
            continue
        variance, rel_var[key], norm_var[key] = metrics.get_all_variance_scores(u[key], u_gt, s_gt)
        all_scores['pcs_relative_variance_4'] = rel_var[key][4]
        all_scores['pcs_relative_variance_10'] = rel_var[key][10]

    angles = {}
    for key in u:
        if key == 'gt':
            continue
        angles[key] = recovar.metrics.subspace_angles(u_gt, u[key], max_rank=20)

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
    unsorted_zs = unsorted_embedding['zs'][4]
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_4'] = averaged_variance

    unsorted_zs = unsorted_embedding['zs'][10]
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_10'] = averaged_variance

    gt_contrasts = synt.contrasts
    for idx in [4, 10, '4_noreg', '10_noreg']:
        unsorted_contrast = unsorted_embedding['contrasts'][idx]
        contrast_abs_error = np.mean(np.abs(gt_contrasts - unsorted_contrast))
        all_scores[f'contrasts_{idx}'] = contrast_abs_error
        # Backward-compatible key for existing comparison scripts.
        all_scores[f'constrasts_{idx}'] = contrast_abs_error
        
        # Create contrast comparison plot
        plt.figure(figsize=(10, 6))
        plt.scatter(gt_contrasts, unsorted_contrast, alpha=0.5, label='Particle contrasts')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
        plt.xlabel('Ground Truth Contrast')
        plt.ylabel('Estimated Contrast')
        plt.title(f'Contrast Comparison (zdim={idx})')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'contrast_comparison_{idx}.png'))
        plt.close()
        
        # Create contrast distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(gt_contrasts, bins=50, alpha=0.5, label='Ground Truth')
        plt.hist(unsorted_contrast, bins=50, alpha=0.5, label='Estimated')
        plt.xlabel('Contrast')
        plt.ylabel('Number of particles')
        plt.title(f'Contrast Distribution (zdim={idx})')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'contrast_distribution_{idx}.png'))
        plt.close()

    for l_idx, l in enumerate(labels_to_plot):
        gt_map = fourier_transform_utils.get_idft3(synt.volumes[l].reshape(cryos[0].volume_shape)).real
        estimate_map = utils.load_mrc(
            Path(output_state_dir, 'all_volumes', f'vol{l_idx:04d}.mrc')
        )
        errors_metrics = metrics.compute_volume_error_metrics_from_gt(
            gt_map, estimate_map, cryos[0].voxel_size, None, partial_mask=None,
            normalize_by_map1=True
        )
        all_scores[f'state_{l_idx}_ninety_pc_locres'] = errors_metrics.get('ninety_pc_locres')
        all_scores[f'state_{l_idx}_median_locres'] = errors_metrics.get('median_locres')

        # write mask to file
        mask = errors_metrics.get('mask')
        if mask is not None:
            mask_path = os.path.join(output_state_dir, 'all_volumes', f'mask_{l_idx:04d}.mrc')
            utils.write_mrc(mask_path, mask.astype(np.float32), voxel_size=cryos[0].voxel_size)
            logger.info(f"Mask written to: {mask_path}")

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
    unsorted_contrast_4 = unsorted_embedding['contrasts'][4]
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
    unsorted_contrast_10 = unsorted_embedding['contrasts'][10]
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
    plt.bar(['Mean FSC', 'Variance FSC'], [all_scores['mean_fsc'], all_scores['variance_fsc']])
    plt.ylim(0, 1)
    plt.title('FSC Scores')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_visualizations.png'))
    plt.close()


    scores_file = os.path.join(plots_dir, "all_scores.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            old_scores = json.load(f)
        all_keys = set(old_scores.keys()) | set(all_scores.keys())
        old_vals, new_vals, labels = [], [], []
        diff_scores = {}
        for key in sorted(all_keys):
            old_val = old_scores.get(key)
            new_val = all_scores.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                diff_scores[key] = new_val - old_val
            else:
                diff_scores[key] = {"old": old_val, "new": new_val}
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
        logger.info(f"Score comparison plot saved at: {comparison_plot_path}")
    else:
        logger.info("No previous scores file found; skipping comparison.")

    all_scores = normalize_scores_for_json(all_scores)

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    logger.info(f"All scores saved to: {scores_file}")

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
        logger.info(f"Metrics baseline written to: {baseline_path}")
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
            logger.error(f"  {failure}")
        if not args.skip_metrics_regression_check:
            error_message(
                f"{len(failures)} metric regressions detected. See {regression_report_path} "
                "or pass --skip-metrics-regression-check to continue."
            )
    else:
        logger.info(f"Metrics regression check passed for {checked} metrics (tol={args.metrics_regression_tol_frac}).")


if __name__ == "__main__":
    main()
