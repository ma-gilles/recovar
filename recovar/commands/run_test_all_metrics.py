import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from recovar import output, metrics, plot_utils, synthetic_dataset, utils, simulator, fourier_transform_utils, recovar
from recovar.commands import pipeline, compute_state

ftu = fourier_transform_utils.fourier_transform_utils(jnp)


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

    print(f"Finished generating dataset {output_folder}")
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
    # Dump parser arguments to a JSON file.
    dump_json_path = os.path.join(args.output_dir, "parser_args.json")
    with open(dump_json_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    dataset_dir = os.path.join(args.output_dir, 'test_dataset')
    grid_size = args.grid_size
    n_images = args.n_images
    tilt_series = args.tomo_tilts > 0

    
    def error_message(msg="An error occurred"):
        print(f"Error: {msg}")
        sys.exit(1)

    def check_gpu():
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                print("GPU devices found:", gpu_devices)
            else:
                error_message("No GPU devices found. Please ensure JAX is properly configured with CUDA.")
        except Exception as e:
            error_message(f"Error checking GPU devices: {e}")

    if not args.cpu:
        check_gpu()

    # Generate synthetic test dataset
    try:
        sim_info = make_big_test_dataset(
            args.volume_input, args.output_dir, noise_level=args.noise_level,
            grid_size=grid_size, n_images=n_images,
            contrast_std=args.contrast_std, n_tilts=args.tomo_tilts,
            premultiplied_ctf=args.premultiplied_ctf,
            noise_increase_per_tilt=args.noise_increase_per_tilt
        )
    except Exception as e:
        error_message(f"Failed to generate test dataset: {e}")

    # Compute average noise radial by counting dose indices
    if 'dose_indices' in sim_info:
        unique_doses, dose_counts = np.unique(sim_info['dose_indices'], return_counts=True)
        print("\nDose index distribution:")
        for dose, count in zip(unique_doses, dose_counts):
            print(f"Dose index {dose}: {count} images ({count/len(sim_info['dose_indices'])*100:.1f}%)")
        
        # Save dose distribution to a file
        dose_dist_path = os.path.join(dataset_dir, 'dose_distribution.txt')
        with open(dose_dist_path, 'w') as f:
            f.write("Dose index distribution:\n")
            for dose, count in zip(unique_doses, dose_counts):
                f.write(f"Dose index {dose}: {count} images ({count/len(sim_info['dose_indices'])*100:.1f}%)\n")
        print(f"\nDose distribution saved to {dose_dist_path}")

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
    print("\nRunning pipeline, as if:")
    print("recovar " + " ".join(cmd))
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
    cs_args = cs_parser.parse_args([
        f"{dataset_dir}/pipeline_output",
        "-o", output_state_dir,
        "--latent-points", latent_points_path
    ])
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
    for idx in [4, 10]:
        unsorted_contrast = pipeline_output.get('unsorted_embedding')['contrasts'][idx]
        all_scores[f'constrasts_{idx}'] = np.mean(np.abs(gt_contrasts - unsorted_contrast))

    for l_idx, l in enumerate(labels_to_plot):
        gt_map = ftu.get_idft3(synt.volumes[l].reshape(cryos[0].volume_shape)).real
        estimate_map = utils.load_mrc(
            Path(output_state_dir, 'all_volumes', f'vol{l_idx:04d}.mrc')
        )
        errors_metrics = metrics.compute_volume_error_metrics_from_gt(
            gt_map, estimate_map, cryos[0].voxel_size, None, partial_mask=None,
            normalize_by_map1=True
        )
        all_scores[f'state_{l_idx}_median_locres'] = errors_metrics.get('ninety_pc_locres')

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
        print(f"Score comparison plot saved at: {comparison_plot_path}")
    else:
        print("No previous scores file found; skipping comparison.")

    # Ensure scores are of type float64 before saving.
    for key in all_scores:
        all_scores[key] = np.float64(all_scores[key])

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"All scores saved to: {scores_file}")


if __name__ == "__main__":
    main()
