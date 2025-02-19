import recovar.config
from importlib import reload
from recovar import simulator, output, utils
reload(simulator)
import numpy as np
import os
# atom_coeff_path = 'data/atom_coeffs_extended.json'
# with open(os.path.join(os.path.dirname(__file__), atom_coeff_path), 'r') as f:
#     atom_coeffs = json.load(f)


def make_test_dataset(noise_level = 0.1, n_images = None):
    grid_size =64
    this_dir = os.path.dirname(__file__)
    volume_folder_input =  this_dir+ '/recovar/data/vol'
    print(volume_folder_input)
    output_folder = this_dir + '/test_dataset/'
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
                                volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.1   , disc_type = 'linear_interp')
    
    print(f"Finished generating dataset {output_folder}")
        

if __name__ == '__main__':
    make_test_dataset()
    print("Done")