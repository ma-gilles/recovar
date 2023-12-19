import recovar.config
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
output_folder = "/home/mg6942/mytigress/simulated_normal/"
volume_folder_input = "/scratch/gpfs/mg6942/cooperative/models/"
outlier_file_input = "/home/mg6942/mytigress/simulated_empiar10180/volumes/vol0915.mrc"
n_images = int(5e4)
voxel_size = 6
output.mkdir_safe(output_folder)
volume_distribution = np.zeros(50)
first_k = 50
volume_distribution[:first_k] = 1/first_k
# volume_distribution[0] = 1
# volume_distribution[-1] = 1/2
# volume_distribution = None
image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, 
                                                             outlier_file_input, n_images, grid_size = 128,
                               volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level =1e-0, 
                               noise_model = "white", put_extra_particles = False, percent_outliers = 0.0, 
                               volume_radius = 0.6, trailing_zero_format_in_vol_name = False, noise_scale_std = 0.2 * 0, contrast_std =0.1 * 0 , disc_type = 'nufft')

print("dumping to:" + output_folder)

