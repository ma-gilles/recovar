import os
import sys
import json
import argparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from recovar import output, metrics, plot_utils, synthetic_dataset, utils, simulator, fourier_transform_utils, recovar
from recovar.commands import pipeline, compute_state

ftu = fourier_transform_utils.fourier_transform_utils(jnp)

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

def make_big_test_dataset(input_dir, output_dir, noise_level=0.1, grid_size=128, n_images=50000,
                          contrast_std=0.1, n_tilts=-1, premultiplied_ctf=False, noise_increase_per_tilt=None):
    output_folder = os.path.join(output_dir, 'test_dataset')
    output.mkdir_safe(output_folder)

    voxel_size = 4.25 * 128 / grid_size
    image_stack, sim_info = simulator.generate_synthetic_dataset(
        output_folder, voxel_size, input_dir, int(n_images),
        outlier_file_input=None, grid_size=grid_size,
        volume_distribution=None, dataset_params_option="uniform",
        noise_level=noise_level, noise_model="radial1", put_extra_particles=False,
        percent_outliers=0.0, volume_radius=0.7, trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0, contrast_std=contrast_std, disc_type='cubic',
        n_tilts=n_tilts, premultiplied_ctf=premultiplied_ctf, noise_increase_per_tilt=noise_increase_per_tilt)

    logging.info(f"Finished generating dataset {output_folder}")
    return sim_info


def main():
    parser = argparse.ArgumentParser(description="Run tests for recovar")
    parser.add_argument('--volume-input', '-i', required=True,
                        help='Input directory containing the volume files')
    parser.add_argument('--output-dir', '-o', default='/tmp/')
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

    args = parser.parse_args()
    output.mkdir_safe(args.output_dir)
    logger = setup_logging(args.output_dir)

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
    particle_assignment = sim_info['image_assignment'] if not tilt_series else sim_info['tilt_series_assignment']

    max_classes = np.max(sim_info['image_assignment']) + 1
    labels_to_plot = [0, max_classes // 2]

    unsorted_zs = pipeline_output.get('unsorted_embedding')['zs'][10]
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
    unsorted_zs = pipeline_output.get('unsorted_embedding')['zs'][4]
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_4'] = averaged_variance

    unsorted_zs = pipeline_output.get('unsorted_embedding')['zs'][10]
    _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
    all_scores['embedding_squared_error_10'] = averaged_variance

    gt_contrasts = synt.contrasts
    for idx in [4, 10, '4_noreg', '10_noreg']:
        unsorted_contrast = pipeline_output.get('unsorted_embedding')['contrasts'][idx]
        all_scores[f'constrasts_{idx}'] = np.mean(np.abs(gt_contrasts - unsorted_contrast))
        
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
        gt_map = ftu.get_idft3(synt.volumes[l].reshape(cryos[0].volume_shape)).real
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

        # Add noise variance estimation metrics
        logger.info("Computing noise variance estimation metrics...")
        
        # Get ground truth noise from simulation
        gt_noise_base = sim_info.get('noise_variance')
        if gt_noise_base is None:
            logger.warning("No ground truth noise variance found in simulation info")
        else:
            # Get estimated noise from pipeline output
            est_noise = pipeline_output.get('noise_var_used')
            
            # Log shapes for debugging
            logger.info(f"Ground truth noise shape: {gt_noise_base.shape}")
            logger.info(f"Estimated noise shape: {est_noise.shape if isinstance(est_noise, np.ndarray) else 'not array'}")
            
            # Handle both single noise model and variable noise per tilt cases
            if isinstance(est_noise, np.ndarray) and est_noise.ndim > 1:
                # Variable noise per tilt case
                logger.info("Processing variable noise per tilt...")
                
                # Get dose indices from simulation info
                dose_indices = sim_info.get('dose_indices')
                if dose_indices is None:
                    logger.warning("No dose indices found for variable noise comparison")
                else:
                    # Get unique tilts and their counts
                    unique_tilts, tilt_counts = np.unique(dose_indices, return_counts=True)
                    n_tilts = len(unique_tilts)
                    
                    # Initialize arrays for per-tilt metrics
                    tilt_correlations = np.zeros(n_tilts)
                    tilt_mean_errors = np.zeros(n_tilts)
                    tilt_median_errors = np.zeros(n_tilts)
                    
                    # Create a figure with subplots for each tilt
                    fig, axes = plt.subplots(n_tilts, 1, figsize=(10, 4*n_tilts))
                    if n_tilts == 1:
                        axes = [axes]
                    
                    # Get noise increase per tilt if it exists
                    noise_increase_per_tilt = sim_info.get('noise_increase_per_tilt')
                    
                    # Compute metrics for each tilt
                    for i, tilt_idx in enumerate(unique_tilts):
                        # Reconstruct ground truth noise for this tilt
                        if noise_increase_per_tilt is not None:
                            # Scale noise by tilt number if noise_increase_per_tilt is set
                            tilt_scale = 1 + noise_increase_per_tilt * tilt_idx
                            tilt_gt_noise = gt_noise_base * tilt_scale
                        else:
                            tilt_gt_noise = gt_noise_base
                            
                        tilt_est_noise = est_noise[tilt_idx]
                        
                        # Ensure shapes match by truncating to shorter length
                        min_len = min(len(tilt_gt_noise), len(tilt_est_noise))
                        tilt_gt_noise = tilt_gt_noise[:min_len]
                        tilt_est_noise = tilt_est_noise[:min_len]
                        
                        # Compute metrics
                        noise_relative_error = np.abs(tilt_est_noise - tilt_gt_noise) / (np.abs(tilt_gt_noise) + 1e-10)
                        tilt_correlations[i] = np.corrcoef(tilt_est_noise, tilt_gt_noise)[0,1]
                        tilt_mean_errors[i] = np.mean(noise_relative_error)
                        tilt_median_errors[i] = np.median(noise_relative_error)
                        
                        # Plot for this tilt
                        ax = axes[i]
                        ax.plot(tilt_gt_noise, label='Ground Truth', alpha=0.7)
                        ax.plot(tilt_est_noise, label='Estimated', alpha=0.7)
                        ax.set_xlabel('Radial Frequency Index')
                        ax.set_ylabel('Noise Variance')
                        ax.set_title(f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images)')
                        if noise_increase_per_tilt is not None:
                            ax.set_title(f'Noise Variance Estimation (Tilt {tilt_idx}, {tilt_counts[i]} images, scale={tilt_scale:.2f})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Add metrics to subplot
                        ax.text(0.02, 0.98, 
                               f'Correlation: {tilt_correlations[i]:.3f}\nMean Rel. Error: {tilt_mean_errors[i]:.3f}\nMedian Rel. Error: {tilt_median_errors[i]:.3f}',
                               transform=ax.transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    noise_plot_path = os.path.join(plots_dir, "noise_variance_comparison_per_tilt.png")
                    plt.savefig(noise_plot_path)
                    plt.close()
                    logger.info(f"Noise variance comparison plot (per tilt) saved to: {noise_plot_path}")
                    
                    # Store aggregate metrics
                    all_scores['noise_mean_relative_error'] = np.mean(tilt_mean_errors)
                    all_scores['noise_median_relative_error'] = np.mean(tilt_median_errors)
                    all_scores['noise_max_relative_error'] = np.max(tilt_mean_errors)
                    all_scores['noise_correlation'] = np.mean(tilt_correlations)
                    all_scores['noise_correlation_per_tilt'] = tilt_correlations.tolist()
                    all_scores['noise_mean_error_per_tilt'] = tilt_mean_errors.tolist()
                    all_scores['noise_median_error_per_tilt'] = tilt_median_errors.tolist()
                    
            else:
                # Single noise model case
                # Ensure shapes match by truncating to shorter length
                min_len = min(len(gt_noise_base), len(est_noise))
                gt_noise_base = gt_noise_base[:min_len]
                est_noise = est_noise[:min_len]
                
                # Compute metrics
                noise_relative_error = np.abs(est_noise - gt_noise_base) / (np.abs(gt_noise_base) + 1e-10)
                noise_correlation = np.corrcoef(est_noise, gt_noise_base)[0,1]
                
                # Store metrics
                all_scores['noise_mean_relative_error'] = np.mean(noise_relative_error)
                all_scores['noise_median_relative_error'] = np.median(noise_relative_error)
                all_scores['noise_max_relative_error'] = np.max(noise_relative_error)
                all_scores['noise_correlation'] = noise_correlation
                
                # Plot noise comparison
                plt.figure(figsize=(10, 6))
                plt.plot(gt_noise_base, label='Ground Truth', alpha=0.7)
                plt.plot(est_noise, label='Estimated', alpha=0.7)
                plt.xlabel('Radial Frequency Index')
                plt.ylabel('Noise Variance')
                plt.title('Noise Variance Estimation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add correlation and error metrics to plot
                plt.text(0.02, 0.98, 
                        f'Correlation: {noise_correlation:.3f}\nMean Rel. Error: {np.mean(noise_relative_error):.3f}\nMedian Rel. Error: {np.median(noise_relative_error):.3f}',
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                noise_plot_path = os.path.join(plots_dir, "noise_variance_comparison.png")
                plt.savefig(noise_plot_path)
                plt.close()
                logger.info(f"Noise variance comparison plot saved to: {noise_plot_path}")


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
    unsorted_contrast_4 = pipeline_output.get('unsorted_embedding')['contrasts'][4]
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
    unsorted_contrast_10 = pipeline_output.get('unsorted_embedding')['contrasts'][10]
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

    # Ensure scores are of type float64 before saving.
    for key in all_scores:
        all_scores[key] = np.float64(all_scores[key])

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    logger.info(f"All scores saved to: {scores_file}")


if __name__ == "__main__":
    main()
