# import recovar.config
from importlib import reload
import numpy as np
from cryodrgn import analysis
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
# warnings.filterwarnings("error")
grid_size =128*2
for log_n in [4]:
    # output_folder ='/home/mg6942/mytigress/spike256/../'
    volume_folder_input =  f"/tigress/CRYOEM/singerlab/mg6942/simulated_empiar10180/volumes_256/vol"
    output_folder = volume_folder_input+ f"/dataset_{log_n}/"
    outlier_file_input = "/home/mg6942/mytigress/simulated_empiar10180/volumes/vol0915.mrc"
    n_images = int(10**(log_n))
    voxel_size = 4.25 * 128 / grid_size #f"{output_folder}../spike{grid_size}_small/0000.mrc"
    output.mkdir_safe(output_folder)
    volume_distribution = np.zeros(1640)
    first_k = 1600
    volume_distribution[:first_k] = 1/first_k
    image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, 
                                                                 outlier_file_input, n_images, grid_size = grid_size,
                                   volume_distribution = volume_distribution,  dataset_params_option = "dataset2", noise_level =1e-1, 
                                   noise_model = "white", put_extra_particles = False, percent_outliers = 0.0, 
                                   volume_radius = 0.6, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.1 * 0 , disc_type = 'nufft')
    