# import recovar.config
from importlib import reload
import numpy as np
from cryodrgn import utils
from cryodrgn import ctf
from recovar import plot_utils
from recovar import output, dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from recovar import simulator
reload(simulator)
import jax
import warnings
from recovar import utils
# warnings.filterwarnings("error")
grid_size =128
for log_n in [4]:
    # output_folder ='/home/mg6942/mytigress/spike256/../'
    grid_size2 = grid_size*2
    volume_folder_input =  f"/tigress/CRYOEM/singerlab/mg6942/simulated_empiar10180/volumes/vol"
    # output_folder = volume_folder_input+ f"/dataset_3e{log_n}_radial_contrast_01/"
    output_folder = volume_folder_input+ f"/dataset_outliers_lownoise/"
    # /scratch/gpfs/mg6942/cooperative/models/renamed

    outlier_file_input = "/scratch/gpfs/mg6942/cooperative/models/1.mrc"
    n_images = 3 * int(10**(log_n))
    voxel_size = 4.25 * 128 / grid_size #f"{output_folder}../spike{grid_size}_small/0000.mrc"
    output.mkdir_safe(output_folder)

    save_dir = '/tigress/CRYOEM/singerlab/mg6942/simulated_empiar10180/volumes/'
    probs = utils.pickle_load(save_dir + 'gt_probs_0605_b.pkl')
    volume_distribution = probs['all_idx'] 
    
    image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, n_images,
                                                                 outlier_file_input = outlier_file_input, grid_size = grid_size,
                                   volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = 0.1,
                                   noise_model = "radial1", put_extra_particles = False, percent_outliers = 0.10, 
                                   volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.0   , disc_type = 'nufft')
    print(f"Finished generating dataset {output_folder}")
    