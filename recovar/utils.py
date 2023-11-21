import logging
import jax
import jax.numpy as jnp
import numpy as np
import mrcfile, os , psutil, pickle
from recovar.fourier_transform_utils import fourier_transform_utils
from recovar import core
ftu = fourier_transform_utils(jax.numpy)
    
logger = logging.getLogger(__name__)

def make_radial_image(average_image_PS, image_shape, extend_last_frequency = True):
    if extend_last_frequency:
        last_noise_band = average_image_PS[-1]
        average_image_PS = jnp.concatenate( [average_image_PS, last_noise_band * jnp.ones_like(average_image_PS) ] )
    radial_distances = ftu.get_grid_of_radial_distances(image_shape, scaled = False, frequency_shift = 0).astype(int).reshape(-1)
    prior = average_image_PS[radial_distances]
    return prior

def find_angle_between_subspaces(v1,v2, max_rank):
    ss = np.conj(v1[:,:max_rank]).T @ v2[:,:max_rank]
    s,v,d = np.linalg.svd(ss)
    if np.any(v > 1.2):
        print('v too big!')
    v = np.where(v < 1, v, 1)
    return np.sqrt( 1 - v[-1]**2)


def subspace_angles(u ,v, max_rank = None):
    max_rank = u.shape[-1] if max_rank is None else max_rank
    corr = np.zeros(max_rank)
    for k in range(1,max_rank+1):
        if k > u.shape[-1]:
            corr[k-1] = 1
        else:
            corr[k-1] = find_angle_between_subspaces(u[:,:k], v[:,:k], max_rank = k )
    return corr  


def estimate_variance(u, s):
    var = np.sum(np.abs(u)**2 * s[...,None], axis = 0)
    return var

# inner psutil function
def get_process_memory_used():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return int(mem_info.rss / 1e9)

def get_gpu_memory_total(device =0):
    if jax_has_gpu():
        return int(jax.local_devices()[device].memory_stats()['bytes_limit']/1e9)
    else:
        logger.warning("GPU not found. Using default value of 80GB")
        return int(80e9)
    


def get_gpu_memory_used(device =0):
    return int(jax.local_devices()[device].memory_stats()['bytes_in_use']/1e9)

def get_peak_gpu_memory_used(device =0):
    return int(jax.local_devices()[device].memory_stats()['peak_bytes_in_use']/1e9)

def report_memory_device(device=0, logger=None):
    output_str = f"GPU mem in use:{get_gpu_memory_used(device)}; peak:{get_peak_gpu_memory_used(device)}; total available:{get_gpu_memory_total(device)}, process mem in use:{get_process_memory_used()}"
    if logger is None:
        print(output_str)
    else:
        logger.info(output_str)

def get_size_in_gb(x):
    return x.size * x.itemsize / 1e9
    
def write_mrc(file, ar):
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(ar.real.astype(np.float32))

def load_mrc(filepath):
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
    return data
        
def symmetrize_ft_volume(vol, volume_shape):
    og_volume_shape = vol.shape
    vol = vol.reshape(volume_shape)
    vol = vol.at[1:,1:,1:].set( 0.5 * (np.conj(np.flip(vol[1:,1:,1:])) + vol[1:,1:,1:]) )
    return vol.reshape(og_volume_shape)

def get_all_dataset_indices(cryos):
    return np.concatenate([cryo.dataset_indices for cryo in cryos])

def get_inverse_dataset_indices(cryos):
    return np.argsort(np.concatenate([cryo.dataset_indices for cryo in cryos]))

def guess_grid_size_from_vol_size(vol_size):
    return np.round((vol_size)**(1/3)).astype(int)
        
def guess_vol_shape_from_vol_size(vol_size):
    return tuple(3*[guess_grid_size_from_vol_size(vol_size)])

# These should probably be set more intelligently
# Sometimes, memory can grow like O(vol_batch_size * image_batch_size)
def get_image_batch_size(grid_size, gpu_memory):
    return int(2*(2**24)/ (grid_size**2)  * gpu_memory / 38)

def get_vol_batch_size(grid_size, gpu_memory):
    return int(25 * (256 / grid_size)**3 * gpu_memory / 38) 

def get_column_batch_size(grid_size, gpu_memory):
    return int(50 * ((256/grid_size)**3) * gpu_memory / 38)

def get_latent_density_batch_size(test_pts,zdim, gpu_memory):
    return np.max([int(gpu_memory/3 * (get_size_in_gb(test_pts) * zdim**2)), 1])

def get_embedding_batch_size(basis, image_size, contrast_grid, zdim, gpu_memory):

    left_over_memory = ( gpu_memory - get_size_in_gb(basis))
    assert left_over_memory > 0, "GPU memory too small?"

    batch_size = int(left_over_memory/ ( (image_size  * np.max([zdim, 4]) + contrast_grid.size * zdim**2 ) *8/1e9 )/ 20)

    if batch_size < 1:
        logger.warning('GPU may be too small for the default parameters. Trying anyway')
        return 1

    return batch_size


def make_algorithm_options(args):
    options = {'volume_mask_option': args.mask_option,
    'zs_dim_to_test': args.zdim,
    'contrast' : "contrast_qr" if args.correct_contrast else "none",
    'ignore_zero_frequency' : args.ignore_zero_frequency 
    }
    return options


def pickle_dump(object, file):
    with open(file, "wb") as f:
        pickle.dump(object, f)

def pickle_load( file):
    with open(file, "rb") as f:
        return pickle.load(f)

    
def get_variances(covariance_cols, picked_frequencies = None):
    # picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = constants.COLUMN_RADIUS, use_half = True))

    volume_shape = guess_vol_shape_from_vol_size(covariance_cols.shape[-1])
    # freqs = core.vec_indices_to_frequencies(picked_frequencies, volume_shape)

    # Probably a better way to do this...
    variances = np.zeros(picked_frequencies.size, covariance_cols.dtype)
    for k in range(picked_frequencies.size):
        variances[k] = covariance_cols[picked_frequencies[k], k]

    return variances

def get_number_of_index_batch(n_images, batch_size):
    return int(np.ceil(n_images/batch_size))

def get_batch_of_indices(n_images, batch_size, k):
    batch_st = int(k * batch_size)
    batch_end = int(np.min( [(k+1) * batch_size, n_images] ))
    return batch_st, batch_end




def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False
