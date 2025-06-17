import recovar.config
from importlib import reload
from recovar import simulator, output, utils
reload(simulator)
import numpy as np
import os
import sys
import argparse
# atom_coeff_path = 'data/atom_coeffs_extended.json'
# with open(os.path.join(os.path.dirname(__file__), atom_coeff_path), 'r') as f:
#     atom_coeffs = json.load(f)


def make_test_dataset(output_dir, noise_level = 0.1, n_images = None, create_nested_structure = False, nested_prefix = "Extract/job193"):
    grid_size =64
    this_dir = os.path.dirname(__file__)
    volume_folder_input =  this_dir+ '/../data/vol'
    print(volume_folder_input)
    output_folder = output_dir + '/test_dataset/'
    output.mkdir_safe(output_folder)
    outlier_file_input = None
    log_n = 3
    n_images = int(10**(log_n)) if n_images is None else n_images
    voxel_size = 4.25 * 128 / grid_size 

    volume_distribution = np.array([1/4, 1/4, 1/2])
    image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, n_images,
                                     outlier_file_input = outlier_file_input, grid_size = grid_size,
                                volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level =noise_level,
                                noise_model = "radial1", put_extra_particles = False, percent_outliers = 0.0, 
                                volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.1   , disc_type = 'linear_interp',
                                create_nested_structure = create_nested_structure, nested_prefix = nested_prefix)
    
    print(f"Finished generating dataset {output_folder}")
    if create_nested_structure:
        print(f"Created nested structure with prefix: {nested_prefix}")
        print(f"Use --strip-prefix {nested_prefix} when loading the dataset")


def main():
    parser = argparse.ArgumentParser(description="Generate a test dataset for recovar")
    parser.add_argument("output_dir", nargs='?', default=os.getcwd(), help="Output directory for the test dataset")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--create-nested-structure", action="store_true", help="Create a nested folder structure to test strip_prefix functionality")
    parser.add_argument("--nested-prefix", default="Extract/job193", help="Prefix path for nested structure (default: Extract/job193)")
    
    args = parser.parse_args()
    
    make_test_dataset(args.output_dir, args.noise_level, args.n_images, args.create_nested_structure, args.nested_prefix)
    print("Done")        

if __name__ == '__main__':
    main()
