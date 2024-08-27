'''
Ewald Sphere Code: models, multipliers, etc.
This code represents generally the most updated models for forward and backwards
PART 1: Coordinate/Ewald Sphere Functions
PART 2: CTF Explicit Functions
PART 3: FORWARD/BACKWARD IMPLICIT MODELS
PART 4: ADAPTATIONS FOR CG INPUT (i.e/mask/unmask)
'''
import logging
import jax 
import jax.numpy as jnp
import numpy as np
from recovar import core, fourier_transform_utils, utils
from jax import vjp
import functools
ftu = fourier_transform_utils.fourier_transform_utils(jax.numpy)
from recovar import mask
# reload(simulator)
# from ewald_core import *

logger = logging.getLogger(__name__)

'''
PART 1: Coordinate/Ewald Sphere Functions
'''

PLANE_MODE = False # For debugging only.

if PLANE_MODE:
    logger.warning("PLANE MODE IS ON! FOR DEBUGGING ONLY")

## Get unrotated coordinates for the ewald sphere 
def get_unrotated_ewald_sphere_coords(image_shape, voxel_size, lam, scaled=True, sphere_sign = 1):
    ## Pass scaled = true
    freqs = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size=voxel_size, scaled=True)
    r = 1/lam
    z = r - jnp.sqrt(r**2 - jnp.linalg.norm(freqs, axis =-1)**2)
    z = z.reshape(-1,1) * sphere_sign
    sphere_freqs = jnp.concatenate([freqs, z], axis=-1)
    scalar = 1 if scaled else image_shape[0] * voxel_size
    return sphere_freqs * scalar

## Get coordinates for the ewald sphere that have the volume represented
# @functools.partial(jax.jit, static_argnums=[1,2,3])
def get_ewald_sphere_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam):
    unrotated_plane_indices = get_unrotated_ewald_sphere_coords(image_shape, voxel_size, lam, scaled = False)

    rotated_plane = jnp.matmul(unrotated_plane_indices, rotation_matrix, precision = jax.lax.Precision.HIGHEST)
    rotated_coords = rotated_plane + jnp.floor(1.0 * grid_size/2)
    return rotated_coords

batch_get_sphere_gridpoint_coords = jax.vmap(get_ewald_sphere_gridpoint_coords, in_axes =(0, None, None, None, None, None) ) 

def parse_disc_type(disc_type):
    if disc_type == 'nearest':
        order = 0
    elif disc_type == 'linear_interp':
        order = 1
    elif 'cubic' in disc_type:
        order = 3
    else:
        raise ValueError(f"disc_type {disc_type} not recognized")
    return order

## Get the slices for ewald sphere
def get_ewald_sphere_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam, disc_type):

    order = parse_disc_type(disc_type)
    return map_coordinates_on_ewald_sphere(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam, order)
    
# No reason not to do this for forward model, but haven't figured out how to do it for the adjoint 
# Maps coordinates onto the Ewald sphere
def map_coordinates_on_ewald_sphere(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam,  order):
    # import pdb; pdb.set_trace()
    batch_grid_pt_vec_ind_of_images = batch_get_sphere_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam )
    # batch_grid_pt_vec_ind_of_images = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T
    # slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)

    if order ==3:
        from recovar import cryojax_map_coordinates
        slices = cryojax_map_coordinates.map_coordinates_with_cubic_spline(volume, batch_grid_pt_vec_ind_of_images, mode = 'fill', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
    else:
        slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)


    return slices



def adjoint_slice_sphere_nearest(images, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam,  order, volume= None):
    grid_vec_indices = batch_get_nearest_gridpoint_indices_ewald_sphere(rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, lam )

    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype = images.dtype)

    volume = volume.at[grid_vec_indices.reshape(-1)].add(images.reshape(-1))
    return volume



# Nearest neighbor
def batch_get_nearest_gridpoint_indices_ewald_sphere(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam):
    # get_ewald_sphere_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam)
    rotated_plane = batch_get_sphere_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam)
    rotated_indices = core.round_to_int(rotated_plane)
    rotated_indices = core.vol_indices_to_vec_indices(rotated_indices, volume_shape)
    return rotated_indices


# Nearest neighbor
def get_nearest_gridpoint_indices_ewald_sphere(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam):
    # get_ewald_sphere_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam)
    rotated_plane = get_ewald_sphere_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size, voxel_size, lam)
    rotated_indices = core.round_to_int(rotated_plane)
    rotated_indices = core.vol_indices_to_vec_indices(rotated_indices, volume_shape)
    return rotated_indices

# Flips indices
# def get_flipped_indices(image_shape):
#     freqs = core.vec_indices_to_frequencies(jnp.arange(np.prod(image_shape)), image_shape)
#     minus_freqs = -freqs
#     flipped_indices = core.frequencies_to_vec_indices(minus_freqs, image_shape)
#     grid_size = image_shape[0]
#     bad_idx = jnp.nonzero(jnp.any(freqs== -grid_size//2 , axis =1),size=image_shape[0]**2)
#     flipped_indices = flipped_indices.at[bad_idx].set(0)
#     return flipped_indices

def get_flipped_indices(image_shape):
    freqs = core.vec_indices_to_frequencies(jnp.arange(np.prod(image_shape)), image_shape)
    minus_freqs = -freqs
    flipped_indices = core.frequencies_to_vec_indices(minus_freqs, image_shape)
    grid_size = image_shape[0]
    bad_idx = jnp.any(freqs== -grid_size//2 , axis =-1)
    flipped_indices = jnp.where(bad_idx, -1, flipped_indices)
    # import pdb; pdb.set_trace()
    # bad_idx = jnp.nonzero(jnp.any(freqs== -grid_size//2 , axis =1),size=image_shape[0]**2)
    # flipped_indices = flipped_indices.at[bad_idx].set(-1)
    return flipped_indices

'''
PART 2: CTF/Chi Functions
'''
def get_chi(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor):   
    volt = volt * 1000
    cs = cs * 10** 7
    dfang = dfang * jnp.pi/180
    phase_shift = phase_shift * jnp.pi / 180

    # Lambda:
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5
#     lam = 1e10

    # x = xi_1, y = xi_2, theta^2 = x^2 + y^2:
    x = freqs[...,0]
    y = freqs[...,1]
    s2 = x**2 + y**2
    # df creating:
    ang = jnp.arctan2(y,x)
    df = .5*(dfu + dfv + (dfu-dfv)*jnp.cos(2*(ang-dfang)))
    # create gamma
    chi = 2*jnp.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    return chi

@jax.jit
def get_chi_packed(freqs, CTF):
    return get_chi(freqs, CTF[0], CTF[1], CTF[2], CTF[3], CTF[4], CTF[5], CTF[6], CTF[7]) * CTF[8]

batch_get_chi_packed = jax.vmap(get_chi_packed, in_axes = (None, 0))

def compute_chi_wrapper(CTF_params, image_shape, voxel_size):
    psi = core.get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2].astype(CTF_params.dtype)
    return batch_get_chi_packed(psi, CTF_params)

'''
PART 3: FORWARD/BACKWARD MODELS
'''

# Forward model
@functools.partial(jax.jit, static_argnums=[4,5,6,7])
def ewald_sphere_forward_model(volume_real, volume_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, disc_type ):

    # This is a hack for CG... I'll take it out if cubic because it doesn't work there.
    if parse_disc_type(disc_type) < 3:
        vol_flipped = get_flipped_indices(volume_shape)

        def flip_vol(x):
            return x[...,vol_flipped]
        volume_real = 0.5 * (volume_real + flip_vol(volume_real))
        volume_imag = 0.5 * (volume_imag - flip_vol(volume_imag))

    
    chi = compute_chi_wrapper(ctf_params, image_shape, voxel_size)
    # lam = 1e-10
    lam = volt_to_wavelength(ctf_params[0,3].astype(np.float64)) if not PLANE_MODE else 1e-10
    # Slice volume on sphere
    vol_real_on_sphere = get_ewald_sphere_slices(volume_real, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  disc_type)
    vol_imag_on_sphere = get_ewald_sphere_slices(volume_imag, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  disc_type)
    
    # Get flipped versions
    flipped_idx = get_flipped_indices(image_shape)
    flipped_vol_real_on_sphere = vol_real_on_sphere[...,flipped_idx]
    flipped_vol_imag_on_sphere = vol_imag_on_sphere[...,flipped_idx]

    # The two equations in ewald.pdf section 4
    images_real = (vol_real_on_sphere + flipped_vol_real_on_sphere) * jnp.sin(chi) \
                + (vol_imag_on_sphere + flipped_vol_imag_on_sphere) * jnp.cos(chi) 


    images_imag = -(vol_real_on_sphere - flipped_vol_real_on_sphere) * jnp.cos(chi) \
                + (vol_imag_on_sphere - flipped_vol_imag_on_sphere) * jnp.sin(chi) 
    # import pdb; pdb.set_trace()
    return 0.5 * images_real, 0.5 * images_imag


# # Adjoint of Forward model - explicit version. Not used currently
# def adjoint_ewald_sphere_forward_model(images_real, images_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, disc_type ):


#     chi = compute_chi_wrapper(ctf_params, image_shape, voxel_size)
#     # lam = 1e-10
#     # Slice volume on sphere
#     # vol_real_on_sphere = get_ewald_sphere_slices(volume_real, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  disc_type)
#     # vol_imag_on_sphere = get_ewald_sphere_slices(volume_imag, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  disc_type)
    
#     # Get flipped versions
#     sin_chi = jnp.sin(chi) * 0.5 
#     cos_chi = jnp.cos(chi) * 0.5

#     # import matplotlib.pyplot as plt
#     # plt.imshow(sin_chi[0].reshape(image_shape))
#     # plt.show()
#     # # import pdb; pdb.set_trace()

#     sin_images_real = images_real * sin_chi
#     cos_images_real = images_real * cos_chi
#     cos_imag_imag = images_imag * cos_chi
#     sin_imag_imag = images_imag * sin_chi

#     flipped_idx = get_flipped_indices(image_shape)

#     def flip(x):
#         return x[...,flipped_idx]
#     # (I+ F) D1 y_r + (-I + F) D2 y_i
#     V1 = (sin_images_real + flip(sin_images_real)) + (-cos_imag_imag + flip(cos_imag_imag))
#     # ( I + F) D2 y_r + (I - F) D1 y_i
#     V2 = (cos_images_real + flip(cos_images_real)) + (sin_imag_imag - flip(sin_imag_imag))

#     V1_b = adjoint_slice_sphere_nearest(V1, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  0, volume= None)
#     V2_b = adjoint_slice_sphere_nearest(V2, rotation_matrices, image_shape, volume_shape, volume_shape[0], voxel_size, lam,  0, volume= None)
    
#     vol_flipped = get_flipped_indices(volume_shape)
#     def flip_vol(x):
#         return x[...,vol_flipped]
    
#     return 0.5 * ( V1_b + flip_vol(V1_b)), 0.5*(V2_b - flip_vol(V2_b) )





# A JAXed version of the adjoint. This is actually slightly slower but will run with disc_type = 'linear_interp'
@functools.partial(jax.jit, static_argnums=[4,5,6,7])
def adjoint_ewald_sphere_forward_model(images_real, images_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, disc_type):  
    volume_size = np.prod(volume_shape)
    f = lambda volume_real, volume_imag : ewald_sphere_forward_model(volume_real, volume_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, disc_type )
    y, u = vjp(f,jnp.zeros(volume_size, dtype = images_real.dtype ), jnp.zeros(volume_size, dtype = images_real.dtype ))
    return u((images_real, images_imag))


# Compute A^TAx (the forward, then its adjoint). For JAX reasons, this should be about 2x faster than doing each call separately.
@functools.partial(jax.jit, static_argnums=[5,6,7,8])
def compute_A_t_Av_ewald_sphere_forward_model(volume_real, volume_imag, rotation_matrices, ctf_params, noise_variance, image_shape, volume_shape, voxel_size, disc_type):    
    f = lambda volume_real, volume_imag : ewald_sphere_forward_model(volume_real, volume_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, disc_type )
    #Av
    y, u = vjp(f,volume_real, volume_imag)
    # Divide by noise for LS
    y_0 = y[0]/noise_variance
    y_1 = y[1]/noise_variance

    # y[0] = y[0]/noise_variance
    # y[1] = y[1]/noise_variance

    # A^T y 
    return u((y_0, y_1))

# ## Full model that does the following: 1) unmasks 2) applys Ax 3) applys A.T to Ax 4) masks
# def full_At_Av_model(x, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, lam, disc_type):
#     volume_real, volume_imag = unvec_masked(x, volume_shape)
#     x_real, x_imag = ewald_sphere_forward_model(volume_real, volume_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, lam, disc_type)
#     z_real, z_imag = adjoint_ewald_sphere_forward_model(x_real, x_imag, rotation_matrices, ctf_params, image_shape, volume_shape, voxel_size, lam, disc_type)
#     return vec_masked(z_real, z_imag, volume_shape)

'''
PART 4: MASK/UNMASK FUNCTIONS
'''

def get_good_idx_mask(volume_shape):
    zero_freq_idx = core.frequencies_to_vec_indices(jnp.array([0, 0, 0]), volume_shape)
    mask_real = mask.get_radial_mask(volume_shape).reshape(-1)
    # TO REMOVE IF NEED BE
    # [MARC] Should this be set to True?
    mask_real = mask_real.at[zero_freq_idx].set(True)

    # TO REMOVE IF NEED BE
    mask_imag = mask_real.copy()
    mask_imag = mask_imag.at[zero_freq_idx].set(False)
    # mask_size = jnp.sum(mask_real)

    # [MARC] What is this doing?
    mask_real_indices = jnp.where(mask_real)
    mask_imag_indices = jnp.where(mask_imag)

    return mask_real_indices, mask_imag_indices

## 
def vec_masked(vol_real, vol_imag, volume_shape):
    # Build mask
    # zero_freq_idx = core.frequencies_to_vec_indices(jnp.array([0,0,0]), volume_shape)
    # mask_real = mask.get_radial_mask(volume_shape).reshape(-1)
    # mask_imag = mask_real.copy()
    # # TO REMOVE IF NEED BE
    # mask_real = mask_real.at[zero_freq_idx].set(False)
    # # TO REMOVE IF NEED BE
    # mask_imag = mask_imag.at[zero_freq_idx].set(False)
    # mask_real_indices = jnp.where(mask_real)
    # mask_imag_indices = jnp.where(mask_imag)
    # # Place back into mask sized vectors
    mask_real_indices, mask_imag_indices = get_good_idx_mask(volume_shape)
    vol_real_masked = vol_real[mask_real_indices]
    vol_imag_masked = vol_imag[mask_imag_indices]
    return jnp.concatenate((vol_real_masked, vol_imag_masked ), dtype = vol_real.dtype)



def unvec_masked(x, volume_shape, mask_size):
    # Build the volumes
    volume_size = volume_shape[0] ** 3
    vol_real, vol_imag = jnp.zeros(volume_size, dtype = x.dtype), jnp.zeros(volume_size, dtype = x.dtype)
    # # Build the masks
    # zero_freq_idx = core.frequencies_to_vec_indices(jnp.array([0, 0, 0]), volume_shape)
    # mask_real = mask.get_radial_mask(volume_shape).reshape(-1)
    # # TO REMOVE IF NEED BE
    # mask_real = mask_real.at[zero_freq_idx].set(False)
    # # TO REMOVE IF NEED BE
    # mask_imag = mask_real.copy()
    # mask_imag = mask_imag.at[zero_freq_idx].set(False)
    # # mask_size = jnp.sum(mask_real)
    # mask_real_indices = jnp.where(mask_real)
    # mask_imag_indices = jnp.where(mask_imag)
    mask_real_indices, mask_imag_indices = get_good_idx_mask(volume_shape)

    # Now, input into vectors
    vol_real = vol_real.at[mask_real_indices].set(x[:mask_size])
    vol_imag = vol_imag.at[mask_imag_indices].set(x[mask_size:])
    return vol_real, vol_imag

'''
BATCH COMPUTATION
'''

def compute_ewald_LS_matvec_in_batches(experiment_dataset, input_volume_real, input_volume_imag, batch_size, disc_type, signal_variance, noise_variance  ):

#     logger.info(f"batch size in second order: {batch_size}")

    vol_real, vol_imag = 0, 0

    # \sum_i A_i^T (1/sigma_i^2) A_i v
    # in batches
    n_batches = utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)
    for k in range(n_batches):
        indices = utils.get_batch_of_indices_arange(experiment_dataset.n_images, batch_size, k)
        vol_real_this, vol_imag_this =compute_A_t_Av_ewald_sphere_forward_model(input_volume_real,
                                        input_volume_imag,
                                        experiment_dataset.rotation_matrices[indices],
                                        experiment_dataset.CTF_params[indices], 
                                        noise_variance,
                                        experiment_dataset.image_shape, 
                                        experiment_dataset.volume_shape, experiment_dataset.voxel_size, disc_type)
        vol_real += vol_real_this
        vol_imag += vol_imag_this

    # + I / kappa^2 * v
    vol_real += input_volume_real / signal_variance
    vol_imag += input_volume_imag / signal_variance 
        
    return vol_real, vol_imag



def compute_diag_mean(experiment_dataset, batch_size, disc_type, noise_variance  ):

#     logger.info(f"batch size in second order: {batch_size}")

    diagonal = 0 

    # \sum_i A_i^T (1/sigma_i^2) A_i v
    # in batches
    n_batches = utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)
    
    for k in range(n_batches):
        volume = jnp.ones(experiment_dataset.volume_size, dtype = jnp.float32)
        indices = utils.get_batch_of_indices_arange(experiment_dataset.n_images, batch_size, k)
        ATA_one = core.compute_A_t_Av_forward_model_from_map(volume, experiment_dataset.CTF_params[indices],
                                                   experiment_dataset.rotation_matrices[indices],
                                                   experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.voxel_size,
                                                   experiment_dataset.CTF_fun, disc_type, noise_variance=noise_variance)
        diagonal += ATA_one[0]
                                                    
    return diagonal




def sphere_sign_hard_assignment(experiment_dataset, volume, batch_size, disc_type, noise_variance):
    
    # volt = experiment_dataset.CTF_params[0,3]
    # lam = volt_to_wavelength(experiment_dataset.CTF_params[0,3])# 
    #lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5

    logger.info(f"batch size in Ewald LHS: {batch_size}")
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)

    # vol_real, vol_imag = 0, 0

    # Compute \sum_i A_i^T y_i / sigma_i^2
    for batch, _, indices in data_generator:
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        batch = core.translate_images(batch, experiment_dataset.translations[indices], experiment_dataset.image_shape)

        # batch /= noise_variance
        residuals = ewald_sphere_forward_model(batch.real, batch.imag,
                                        experiment_dataset.rotation_matrices[indices],
                                        experiment_dataset.CTF_params[indices], 
                                        experiment_dataset.image_shape, experiment_dataset.volume_shape, 
                                        experiment_dataset.voxel_size, disc_type)

        # vol_real += A_t_vol_real
        # vol_imag += A_t_vol_imag

    logger.info(f"LHS done.")
    return vol_real, vol_imag





def volt_to_wavelength(volt):
    logger.warning("IS THIS RIGHT?!")
    volt = volt * 1000
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5
    return lam #12.2639 / (volt + 0.97845e-6 * volt**2)**.5




def compute_ewald_LS_rhs_in_batches(experiment_dataset, batch_size, disc_type, noise_variance):
    
    # volt = experiment_dataset.CTF_params[0,3]
    # lam = volt_to_wavelength(experiment_dataset.CTF_params[0,3])# 
#     lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5

    logger.info(f"batch size in Ewald LHS: {batch_size}")
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)

    vol_real, vol_imag = 0, 0

    # Compute \sum_i A_i^T y_i / sigma_i^2
    for batch, _, indices in data_generator:
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        batch = core.translate_images(batch, experiment_dataset.translations[indices], experiment_dataset.image_shape)

        batch /= noise_variance
        A_t_vol_real, A_t_vol_imag = adjoint_ewald_sphere_forward_model(batch.real, batch.imag,
                                        experiment_dataset.rotation_matrices[indices],
                                        experiment_dataset.CTF_params[indices], 
                                        experiment_dataset.image_shape, experiment_dataset.volume_shape, 
                                        experiment_dataset.voxel_size, disc_type)

        vol_real += A_t_vol_real
        vol_imag += A_t_vol_imag
    logger.info(f"LHS done.")
    return vol_real, vol_imag



def solve_ewald_least_squares(experiment_dataset, batch_size, disc_type, signal_variance, noise_variance, x0 = None, max_iter = 100, tol = 1e-10):
    from recovar import noise
    from importlib import reload
    from recovar import homogeneous
    reload(homogeneous)

    if not np.isclose(experiment_dataset.CTF_params[:,core.w_ind],0).all():
        ctf_params = experiment_dataset.CTF_params
        assert np.isclose(experiment_dataset.CTF_params[:,core.phase_shift_ind],0).all(), "Either w or phase shift should be zero"

        phase_shift = np.arcsin(ctf_params[:,core.w_ind]) / np.pi * 180
        ctf_params[:,core.w_ind] = 0
        ctf_params[:,core.phase_shift_ind] = phase_shift
        # ctf_params[:,core.volt_ind] = 100
        experiment_dataset.CTF_params = ctf_params


    noise_variance = noise_variance
    noise_variance = noise.make_radial_noise(noise_variance, experiment_dataset.image_shape)
    # signal_variance = np.inf
    print(utils.report_memory_device())
    rhs_real, rhs_imag = compute_ewald_LS_rhs_in_batches(experiment_dataset, batch_size, disc_type, noise_variance)
    rhs = vec_masked(rhs_real, rhs_imag, experiment_dataset.volume_shape)
    del rhs_imag, rhs_real

    rhs = np.array(rhs)
    # del rhs

    mask_real = mask.get_radial_mask(experiment_dataset.volume_shape).reshape(-1)
    mask_size = int(jnp.sum(mask_real))

    def mat_vec_wrapped_up(x):
        volume_real, volume_imag = unvec_masked(x, experiment_dataset.volume_shape, mask_size)
        z_real, z_imag = compute_ewald_LS_matvec_in_batches(experiment_dataset, volume_real, volume_imag, batch_size, disc_type, signal_variance, noise_variance  )
        return vec_masked(z_real, z_imag, experiment_dataset.volume_shape)
    

    def mat_vec_wrapped_up_2(x):
        # volume_real, volume_imag = unvec_masked(x, experiment_dataset.volume_shape, mask_size)
        z_real, z_imag = compute_ewald_LS_matvec_in_batches(experiment_dataset, x.real, x.imag, batch_size, disc_type, signal_variance, noise_variance  )
        return z_real + 1j *  z_imag

    # experimenting with preconditioner

    diag_mean2 = compute_diag_mean(experiment_dataset, batch_size, disc_type, noise_variance  )

    # diag_mean3, _ = homogeneous.solve_least_squares_mean_iteration(experiment_dataset , None, noise_variance,  batch_size, None, disc_type = disc_type, return_lhs_rhs = True )
    diag_mean = (1/ signal_variance + diag_mean2).reshape(-1)
    diag_mean = vec_masked(diag_mean, diag_mean, experiment_dataset.volume_shape)
    diag_mean2 = np.array(diag_mean2)
    diag_mean = np.array(diag_mean)

    if x0 is not None:
        x0_masked = vec_masked(x0.reshape(-1).real, x0.reshape(-1).imag, experiment_dataset.volume_shape)
        x0_masked = np.array(x0_masked)
    else:
        x0_masked = None



    def planar_model(x):
        # get diag_mean
        # diag_mean4 = np.where(diag_mean == 0, 1, diag_mean)
        return x / diag_mean
    

    import matplotlib.pyplot as plt
    
    import scipy.sparse
    import inspect

    N = (mask_size) * 2 -1
    # print(N)
    ATA_shape = (N, N)
    ATA_op = scipy.sparse.linalg.LinearOperator(ATA_shape, mat_vec_wrapped_up)
    planar_op = scipy.sparse.linalg.LinearOperator(ATA_shape, planar_model)

    ress = []
    global iter_count
    iter_count = 0
    def report(xk):
        frame = inspect.currentframe().f_back
        ress.append(frame.f_locals['resid'] / np.linalg.norm(rhs))
        logger.info(f"CG iter {len(ress)}, residual: {ress[-1]}")
        # print()
        # iter_count += 1
    print(utils.report_memory_device())
    # import pdb; pdb.set_trace()
    x_result,_ = scipy.sparse.linalg.cg(ATA_op, rhs, x0 = x0_masked, maxiter=max_iter, tol=tol, M = planar_op, callback=report)

    # from recovar import utils
    # outdir = '/home/mg6942/mytigress/'
    # utils.pickle_dump((volume_real, volume_imag), outdir + 'volume_real_imag.pkl')

    volume_real, volume_imag = unvec_masked(x_result, experiment_dataset.volume_shape, mask_size)
    vol = ftu.get_idft3((volume_real + 1j * volume_imag).reshape(experiment_dataset.volume_shape))#.real

    return vol, ress


def matvec_experiments(y, experiment_dataset, batch_size, disc_type, signal_variance, noise_variance):
    from recovar import noise

    noise_variance = noise.make_radial_noise(noise_variance, experiment_dataset.image_shape)
    mask_real = mask.get_radial_mask(experiment_dataset.volume_shape).reshape(-1)
    mask_size = int(jnp.sum(mask_real))

    volume_real, volume_imag = unvec_masked(y, experiment_dataset.volume_shape, mask_size)
    z_real, z_imag = compute_ewald_LS_matvec_in_batches(experiment_dataset, volume_real, volume_imag, batch_size, disc_type, signal_variance, noise_variance  )
    z = vec_masked(z_real, z_imag, experiment_dataset.volume_shape)
    
#     def jp(y):
#         e = jnp.ones_like(y)
#         d = mat_vec_wrapped_up(e)
#         return d
    
#     d = jp(y)

    
    return z


'''
PRECONDITIONED VERSION
'''
