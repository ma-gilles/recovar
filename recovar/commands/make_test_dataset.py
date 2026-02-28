import recovar.jax_config
from recovar import simulator, output
import numpy as np
import os
import argparse


def make_test_dataset(output_dir, image_size = 64, noise_level = 0.1, n_images = None, create_nested_structure = False, nested_prefix = "Extract/job193", tilt_series = False, outlier_file_input = None, percent_outliers = 0.0, percent_tilt_series_outliers = 0.0, seed = None):
    if seed is not None:
        np.random.seed(seed)
    grid_size = image_size
    this_dir = os.path.dirname(__file__)
    volume_folder_input = os.path.join(this_dir, '..', 'data', 'vol')
    print(volume_folder_input)

    output_folder = os.path.join(output_dir, 'test_dataset')
    output.mkdir_safe(output_folder)
    n_images = 1000 if n_images is None else int(n_images)
    voxel_size = 4.25 * 128 / grid_size 

    volume_distribution = np.array([1/4, 1/4, 1/2])
    
    if tilt_series:
        # For tilt series, we need to generate multiple images per particle
        # Adjust parameters for tilt series simulation
        n_tilts = 27  # Number of tilt angles per particle
        n_particles = max(1, n_images // n_tilts)  # Adjust number of particles
        print(f"Generating tilt series with {n_particles} particles and {n_tilts} tilts per particle")
        
        image_stack, sim_info = simulator.generate_synthetic_dataset(
            output_folder, voxel_size, volume_folder_input, n_images,
            outlier_file_input=outlier_file_input, grid_size=grid_size,
            volume_distribution=volume_distribution, dataset_params_option="uniform", 
            noise_level=noise_level, noise_model="radial1", put_extra_particles=False, 
            percent_outliers=percent_outliers, volume_radius=0.7, trailing_zero_format_in_vol_name=True, 
            noise_scale_std=0.2 * 0, contrast_std=0.1, disc_type='linear_interp',
            create_nested_structure=create_nested_structure, nested_prefix=nested_prefix,
            n_tilts=n_tilts, dose_per_tilt=3, angle_per_tilt=3, percent_tilt_series_outliers=percent_tilt_series_outliers
        )
    else:
        image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, n_images,
                                         outlier_file_input = outlier_file_input, grid_size = grid_size,
                                    volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level =noise_level,
                                    noise_model = "radial1", put_extra_particles = False, percent_outliers = percent_outliers, 
                                    volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.1   , disc_type = 'linear_interp',
                                    create_nested_structure = create_nested_structure, nested_prefix = nested_prefix, percent_tilt_series_outliers = percent_tilt_series_outliers)
    
    print(f"Finished generating dataset {output_folder}")
    if create_nested_structure:
        print(f"Created nested structure with prefix: {nested_prefix}")
        print(f"Use --strip-prefix {nested_prefix} when loading the dataset")
    if tilt_series:
        print(f"Generated tilt series dataset with {n_tilts} tilts per particle")
    if percent_outliers > 0:
        print(f"Generated dataset with {percent_outliers*100:.1f}% outliers")
        if outlier_file_input:
            print(f"Used outlier volume: {outlier_file_input}")
    if tilt_series and percent_tilt_series_outliers > 0:
        print(f"Generated tilt series dataset with {percent_tilt_series_outliers*100:.1f}% tilt outliers")


def main():
    parser = argparse.ArgumentParser(description="Generate a test dataset for recovar")
    parser.add_argument("output_dir", nargs='?', default=os.getcwd(), help="Output directory for the test dataset")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (default: 128 for 128x128 images)")

    parser.add_argument("--create-nested-structure", action="store_true", help="Create a nested folder structure to test strip_prefix functionality")
    parser.add_argument("--nested-prefix", default="Extract/job193", help="Prefix path for nested structure (default: Extract/job193)")
    parser.add_argument("--tilt-series", action="store_true", help="Generate tilt series dataset instead of single particle dataset")
    parser.add_argument("--outlier-file-input", help="Path to outlier volume file")
    parser.add_argument("--percent-outliers", default=0.0, type=float, help="Percentage of outliers in the dataset")
    parser.add_argument("--percent-tilt-series-outliers", type=float, default=0.0, help="Percentage of tilt outliers in tilt series dataset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible dataset generation")

    args = parser.parse_args()

    make_test_dataset(args.output_dir, args.image_size, args.noise_level, args.n_images, args.create_nested_structure, args.nested_prefix, args.tilt_series, args.outlier_file_input, args.percent_outliers, args.percent_tilt_series_outliers, args.seed)
    print("Done")        

if __name__ == '__main__':
    main()
