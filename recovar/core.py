import jax
import jax.numpy as jnp
import numpy as np
import functools
from jax import vjp
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jax.numpy)

## Low level functions for cryo-EM. Handling of indexing, slicing, adjoint of slicing, CTF, etc.

# Three different representation. E.g. for a 2x2x2 grid  
# vol_indices are [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]
# vec_indices are 0, 1, 2, 3, 4, 5, 6, 7
# frequencies are [-1, -1, -1], [-1, -1, 0], etc ...

# CTF inds
dfu_ind = 0# (float or Bx1 tensor): DefocusU (Angstrom)
dfv_ind = 1 #(float or Bx1 tensor): DefocusV (Angstrom)
dfang_ind = 2 #(float or Bx1 tensor): DefocusAngle (degrees)
volt_ind =3 #(float or Bx1 tensor): accelerating voltage (kV)
cs_int=4 #(float or Bx1 tensor): spherical aberration (mm)
w_ind =5 #(float or Bx1 tensor): amplitude contrast ratio
phase_shift_ind = 6 #(float or Bx1 tensor): degrees 
bfactor_ind = 7 #(float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
contrast_ind = 8
# delete this option?
tilt_number_ind = 9 # 
dose_ind = 9
tilt_angle_ind = 10

@functools.partial(jax.jit, static_argnums=[1])
def vol_indices_to_vec_indices(vol_indices, vol_shape):
    # good_idx = check_vol_indices_in_bound(vol_indices,vol_shape[0])
    og_shape = vol_indices.shape
    vol_indices = vol_indices.reshape(-1, vol_indices.shape[-1])
    vec_indices = jnp.ravel_multi_index(vol_indices.T, vol_shape, mode = 'clip').reshape(og_shape[:-1])

    return vec_indices#jnp.where(good_idx, vec_indices, -1)
    # return jnp.ravel_multi_index(vol_indices.T, vol_shape, mode = 'clip')

def vec_indices_to_vol_indices(vec_indices, vol_shape):
    vol_indices = jnp.unravel_index(vec_indices, vol_shape)
    vol_indices = jnp.stack(vol_indices, axis = -1)
    return vol_indices

def vol_indices_to_frequencies(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    # ((vol_shape - (vol_shape%2 ==1))//2 * 1)
    mid_grid =  ((vol_shape - (vol_shape%2 ==1))//2 * 1).astype(int)
    return vol_indices - mid_grid

def frequencies_to_vol_indices(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid =  ((vol_shape - (vol_shape%2 ==1))//2 * 1).astype(int)
    return vol_indices + mid_grid

def vec_indices_to_frequencies(vec_indices, vol_shape):
    return vol_indices_to_frequencies(vec_indices_to_vol_indices(vec_indices, vol_shape), vol_shape)

def frequencies_to_vec_indices(frequencies, vol_shape):
    return vol_indices_to_vec_indices(frequencies_to_vol_indices(frequencies, vol_shape), vol_shape)

def check_frequencies_in_bound(frequencies,grid_size):
    return jnp.prod((frequencies >= -grid_size/2 ) * (frequencies < grid_size/2), axis = -1).astype(bool)

def check_vol_indices_in_bound(vol_indices,grid_size):
    return jnp.prod((vol_indices >= 0 ) * (vol_indices < grid_size ), axis = -1).astype(bool)

def check_vec_indices_in_bound(vec_indices,grid_size):
    return ((vec_indices < grid_size**3 ) * vec_indices >= 0).astype(bool)


# # This function is meant to handle the heterogeneity case.
# def multi_vol_indices_to_multi_vec_indices(vol_indices, het_indices, vol_shape, het_shape):
#     og_shape = vol_indices.shape
#     vol_indices = jnp.concatenate( [het_indices, vol_indices.reshape(-1, vol_indices.shape[-1]) ], axis=-1)
#     multi_vol_shape = (het_shape, *vol_shape)
#     vec_indices = jnp.ravel_multi_index(vol_indices.T, multi_vol_shape, mode = 'clip').reshape(og_shape[:-1])
#     return


## Some similar function used for adaptive discretization: find nearby gridpoints
@functools.partial(jax.jit, static_argnums=[1])
def find_frequencies_within_grid_dist(coords, max_grid_dist: int ):
    # k 
    dim = coords.shape[-1]

    # Closest ind
    # closest_point = round_to_int(coord)
    if dim ==2:
        neighbors = jnp.mgrid[-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1]
    else:
        neighbors = jnp.mgrid[-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1]

    neighbors = neighbors.reshape(dim, -1).T

    coords_ndim = coords.ndim
    neighbors = neighbors.reshape( (coords_ndim-1) * [1]  + list(neighbors.shape))
    neighbors += coords[...,None,:]
    if coords.dtype.kind == 'f':
        neighbors = round_to_int(neighbors)
    return neighbors

## This could be improved...
# @functools.partial(jax.jit, static_argnums=[1])
# def find_nearest_k_frequencies(coord, k: int ):
#     # (1 + n)^2 (1 + 4 n + 2 n^2)
#     # 1/2 (-1 + k^(1/3))
#     #  
#     neighbors = find_frequencies_within_grid_dist(coord, max_grid_dist: int )
#     distances = jnp.linalg.norm(neighbors - coord, axis = 0)
#     # Now find k smallest distances
#     distances_at_best, indices = jax.lax.top_k(-distances, k)
    # return

batch_find_frequencies_within_grid_dist= jax.vmap(find_frequencies_within_grid_dist, in_axes = (0, None))
batch_batch_find_frequencies_within_grid_dist= jax.vmap(batch_find_frequencies_within_grid_dist, in_axes = (0, None))


# def find_frequencies_within_dist(coord, max_grid_dist):
#     dim = coord.shape[-1]
#     k = max_grid_dist * 2**dim - 1
#     return find_k_nearest_frequencies(coord, k )

def distance_to_max_grid_dist(dist):
    return np.ceil(dist).astype(int)


# There are two ways to slice volumes. One fast which just uses nearest gridpoints, and one slow which calls map_coordinates, which can use either both nearest and linear interpolation.
@jax.jit
def slice_volume_by_nearest(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid] 

# Used to project the mean
batch_slice_volume_by_nearest = jax.vmap(slice_volume_by_nearest, (None, 0))

def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    elif disc_type == "nearest":
        return 0
    elif disc_type == "cubic":
        return 3
    else:
        raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")

@functools.partial(jax.jit, static_argnums = [2,3,4])    
def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):    
    order = decide_order(disc_type)
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order)
    

@functools.partial(jax.jit, static_argnums=[2,3,4])
def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type):  
    volume_size = np.prod(volume_shape)
    f = lambda volume : slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    _, u = vjp(f,jnp.zeros(volume_size, dtype = slices.dtype ))
    return u(slices)[0]

# This is what people called the backprojection
# Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
@functools.partial(jax.jit, static_argnums=0)
def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids, volume_vec = None):
    if volume_vec is None:
        volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
    volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
    return volume_vec

batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))


# def batch_over_vol_summed_adjoint_slice_by_nearest_add():
#     return jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))
# batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))


# # Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
# @functools.partial(jax.jit, static_argnums=0)
# def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids):
#     volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
#     volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
#     return volume_vec


# batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None), out_axes = ( -1))

nosummed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, 0,0)) 

@functools.partial(jax.jit, static_argnums=0)
def sum_adj_forward_model(volume_size, images, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return summed_adjoint_slice_by_nearest(volume_size, images * jnp.conj(CTF_val_on_grid_stacked), plane_indices_on_grid_stacked)


@jax.jit
def forward_model(volume_vec, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return batch_slice_volume_by_nearest(volume_vec, plane_indices_on_grid_stacked) * CTF_val_on_grid_stacked


# @jax.jit
# def translate_single_image(image, translation, lattice):
#     phase_shift = jnp.exp( 1j * -2 * jnp.pi * (lattice @ translation[:,None] ) )[...,0]
#     return image.reshape(-1) * phase_shift

@jax.jit
def translate_single_image(image, translation, lattice):
    phase_shift = jnp.exp( 1j * -2 * jnp.pi * (lattice @ translation[:,None] ) )[...,0]
    return image * phase_shift


batch_translate = jax.vmap(translate_single_image, in_axes = (0,0, None))

@functools.partial(jax.jit, static_argnums=2)
def translate_images(image, translation, image_shape):
    twod_lattice = get_unrotated_plane_coords(image_shape, voxel_size =1, scaled = True )[:,:2]
    return batch_translate(image, translation, twod_lattice)#.astype(image.dtype)


batch_trans_translate_images = jax.vmap(translate_images, in_axes = (None,-2, None), out_axes = (-2))


def get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor = 1):
    # This is used to handle discretization of projection 
    unrotated_plane_indices =  ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = 1, scaled = False)
    unrotated_plane_indices = jnp.concatenate( [unrotated_plane_indices, jnp.zeros(unrotated_plane_indices.shape[0], dtype = unrotated_plane_indices.dtype )[...,None] ], axis = -1)
    # If we are 
    # 
    return unrotated_plane_indices * three_d_upsampling_factor


def get_unrotated_plane_coords(image_shape, voxel_size, scaled =True ):
    # These are scaled with voxel size. This is only used for CTF and translations
    plane_coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = voxel_size, scaled = scaled)
    plane_coords = jnp.concatenate( [plane_coords, jnp.zeros(plane_coords.shape[0], dtype = plane_coords.dtype)[...,None] ], axis = -1)
    return plane_coords



## JITTING THIS MAY CAUSE WEIRD ERRORS...
# @functools.partial(jax.jit, static_argnums=[1,2,3])
def get_nearest_gridpoint_indices(rotation_matrix, image_shape, volume_shape):
    rotated_plane = get_gridpoint_coords(rotation_matrix, image_shape, volume_shape)
    rotated_indices = round_to_int(rotated_plane)
    rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_indices


@functools.partial(jax.jit, static_argnums=[1,2])
def get_gridpoint_coords(rotation_matrix, image_shape, volume_shape):

    three_d_upsampling_factor = volume_shape[0] // image_shape[0]
    unrotated_plane_indices = get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor = three_d_upsampling_factor)
    rotated_plane = jnp.matmul(unrotated_plane_indices, rotation_matrix, precision = jax.lax.Precision.HIGHEST)
    # I have no idea why there is a random +1 here
    rotated_coords = rotated_plane + jnp.floor(1.0 * volume_shape[0]/2)
    
    # rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_coords

@jax.jit
def round_to_int(array):
    return jax.lax.round(array).astype(jnp.int32)


batch_get_nearest_gridpoint_indices = jax.vmap(get_nearest_gridpoint_indices, in_axes =(0, None, None) ) 
batch_get_gridpoint_coords = jax.vmap(get_gridpoint_coords, in_axes =(0, None, None) ) 


@functools.partial(jax.jit, static_argnums=[3,4,6,7,8], static_argnames=['premultiplied_ctf'])
def forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf=False ):
    slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type) 
    if not premultiplied_ctf:
        slices = slices * CTF_fun( CTF_params, image_shape, voxel_size)
    return slices

batch_forward_model_from_map = jax.vmap(forward_model_from_map, in_axes = (0, 0, 0, None, None, None, None, None, None) )



@functools.partial(jax.jit, static_argnums=[3,4,6,7,8], static_argnames=['premultiplied_ctf'])
def forward_model_from_map_and_return_adjoint(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf = False):
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf)
    slices, f_adj = vjp(f,volume)
    return slices, f_adj


# A JAXed version of the adjoint. This is actually slightly slower but will run with disc_type = 'linear_interp'. I probably should just write out an explicit adjoint of linear interpolation...
@functools.partial(jax.jit, static_argnums=[3,4,6,7,8], static_argnames=['premultiplied_ctf'])
def adjoint_forward_model_from_map(slices, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf=False ):  
    volume_size = np.prod(volume_shape)
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf)
    _, u = vjp(f,jnp.zeros(volume_size, dtype = slices.dtype ))
    return u(slices)[0]


@functools.partial(jax.jit, static_argnums=[3,4,6,7,8], static_argnames=['premultiplied_ctf'])
def adjoint_forward_model_from_trilinear(slices, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf = False):
    if not premultiplied_ctf:
        slices = slices * CTF_fun( CTF_params, image_shape, voxel_size)
    return adjoint_slice_volume_by_trilinear(slices, rotation_matrices, image_shape, volume_shape, volume = None)



# Compute A^TAx (the forward, then its adjoint). For JAX reasons, this should be about 2x faster than doing each call separately.
@functools.partial(jax.jit, static_argnums=[3,4,6,7,9], static_argnames=['premultiplied_ctf'])
def compute_A_t_Av_forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, noise_variance, premultiplied_ctf = False):    
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, premultiplied_ctf)
    y, u = vjp(f,volume)
    return u(y/noise_variance)




def rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape):
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T
    return batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape


def get_stencil(dim):
    if dim==2:
        return jnp.array([[0, 0], [0,1], [1,0], [1,1]], dtype = int)
    if dim==3:
        return jnp.array([[0, 0, 0], [0, 0,1], [0, 1,0], [0, 1,1], \
                         [1, 0, 0], [1, 0,1], [1, 1,0], [1, 1,1]] , dtype = int)


#
def get_trilinear_weights_and_vol_indices(grid_coords, volume_shape):
    # lower_grid_points = jnp.floor(grid_points).astype(int)
    lower_points_ndim = grid_coords.ndim-1
    all_grid_points = jnp.floor(grid_coords)[...,None,:] + get_stencil(grid_coords.shape[-1]).reshape( [*(lower_points_ndim * [1]), 8,3])

    all_weights = (1 - jnp.abs(grid_coords[...,None,:] - all_grid_points )).astype(np.float32)
    all_weights = jnp.where(all_weights > 0, all_weights, 0)
    all_weights = jnp.prod(all_weights, axis=-1)

    # all_weights = jnp.where(all_weights > 0, all_weights, 0)
    #**2
    # all_weights /= jnp.linalg.norm(all_weights, axis=-1, keepdims=True)

    # Zero-out out of bound for good measure.
    good_points = check_vol_indices_in_bound(all_grid_points, volume_shape[0])
    all_weights *= good_points
    # assert False

    return all_grid_points.astype(jnp.int32), all_weights

# @functools.partial(jax.jit, static_argnums = [2,3,4])    
def slice_volume_by_trilinear(volume, rotation_matrices, image_shape, volume_shape):    
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices( grid_points, volume_shape)
    sliced_volume = jnp.sum(volume[grid_vec_indices.reshape(-1)].reshape(grid_vec_indices.shape) * weights, axis=-1)
    return sliced_volume.reshape(grid_coords_og_shape[:-1]).astype(volume.dtype)


# @functools.partial(jax.jit, static_argnums = [2,3,4])    
## UNTESTED
def adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volume = None):    
    grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype = images.dtype)

    weights *= images.reshape(-1,1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))

    return volume



def adjoint_slice_volume_by_trilinear_from_weights(images, grid_vec_indices, weights, volume_shape, volume = None):    
    # grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    # grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    # grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype = images.dtype)

    weights *= images.reshape(-1,1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))

    return volume



@functools.partial(jax.jit, static_argnums = [2,3,4])    
def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order):
    # import pdb; pdb.set_trace()
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T

    if order ==3:
        from recovar import cryojax_map_coordinates
        slices = cryojax_map_coordinates.map_coordinates_with_cubic_spline(volume, batch_grid_pt_vec_ind_of_images, mode = 'fill', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
    else:
        slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)

    return slices

# # batch volumes
# batch_vol_rot_slice_volume_by_map = jax.vmap(slice_volume_by_map, in_axes = (0, 0, None, None, None) )

batch_translate_images = jax.vmap(translate_images, in_axes = (0, 0, None) )




## CTF functions.
@jax.jit
def evaluate_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor):
    '''
    Adapted for JAX from cryoDRGN: https://github.com/zhonge/cryodrgn
    
    
    Compute the 2D CTF
   
    Input: 
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees 
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * jnp.pi / 180
    phase_shift = phase_shift * jnp.pi / 180
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2

    # LAMBDA IN CRYODRGN
    # lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5
    # LAMBDA IN RELION
    lam = 12.2642598 / jnp.sqrt(volt * (1. + volt * 9.78475598e-7))
    # 	lambda=12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6)); // See http://en.wikipedia.org/wiki/Electron_diffraction

    # This causes a CTF difference of about 0.6% at 300 kV, which seems kinda big.

    x = freqs[...,0]    
    y = freqs[...,1]
    ang = jnp.arctan2(y,x)
    s2 = x**2 + y**2
    df = .5*(dfu + dfv + (dfu-dfv)*jnp.cos(2*(ang-dfang)))
    gamma = 2*jnp.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    ctf = (1-w**2)**.5*jnp.sin(gamma) - w*jnp.cos(gamma) 
    if bfactor is not None:
        ctf *= jnp.exp(-bfactor/4*s2)
    return ctf

@jax.jit
def evaluate_ctf_packed(freqs, CTF):
    return evaluate_ctf(freqs, CTF[0], CTF[1], CTF[2], CTF[3], CTF[4], CTF[5], CTF[6], CTF[7]) * CTF[8]

batch_evaluate_ctf = jax.vmap(evaluate_ctf_packed, in_axes = (None, 0))


# @functools.partial
@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series_v2(CTF_params, image_shape, voxel_size ):

    dose_filter = get_dose_filters(voxel_size, image_shape, CTF_params[:,dose_ind], CTF_params[:,tilt_angle_ind], CTF_params[0,volt_ind])

    return dose_filter * cryodrgn_CTF(CTF_params[:,:9], image_shape, voxel_size)

# def evaluate_ctf_tilt(CTF_params, )
@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series(CTF_params, image_shape, voxel_size, dose_per_tilt =None, angle_per_tilt=None ):

    dose_filter = get_dose_filters_from_tilt_number(voxel_size, image_shape, dose_per_tilt, angle_per_tilt, CTF_params[:,9], CTF_params[0,4])

    return dose_filter * cryodrgn_CTF(CTF_params[:,:9], image_shape, voxel_size)

# A wrapper to have some input as SPA CTF
def get_cryo_ET_CTF_fun( dose_per_tilt = 2.9, angle_per_tilt = 3):
    def CTF_ET_fun(*args):
        return evaluate_ctf_wrapper_tilt_series(*args, dose_per_tilt = dose_per_tilt, angle_per_tilt = angle_per_tilt )
    return CTF_ET_fun




def critical_exposure(freq, voltage):
    # Define a scale factor based on the voltage value
    scale_factor = jnp.where(jnp.isclose(voltage, 200), 0.75, 1)

    # Calculate the critical exposure
    critical_exp = freq ** (-1.665)
    critical_exp = critical_exp * scale_factor * 0.245

    return critical_exp + 2.81


def get_dose_filters_from_tilt_number(Apix, image_shape, dose_per_tilt, angle_per_tilt, tilt_numbers, voltage):
    cumulative_dose = tilt_numbers * dose_per_tilt
    tilt_angles = angle_per_tilt * jnp.ceil(tilt_numbers / 2) 
    return get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage)


def get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage):
    D = image_shape[0]

    N = len(cumulative_dose)
    # print("whats N?", N)
    # if N !=10:
    #     import pdb; pdb.set_trace()

    freqs = ftu.get_k_coordinate_of_each_pixel(image_shape, Apix, scaled=True)

    x = freqs[..., 0]
    y = freqs[..., 1]
    s2 = x**2 + y**2
    s = jnp.sqrt(s2)

    # cumulative_dose = tilt_numbers * dose_per_tilt
    # cd_tile = torch.repeat_interleave(cumulative_dose, D * D).view(N, -1)
    cd_tile = cumulative_dose[:, None] * jnp.ones([N, D * D])

    # This probably should be rewritten. For now I am trying to keep it as close to the original as possible.
    ce = critical_exposure(s, voltage)
    ce_tile = jnp.repeat(ce[None], N, axis=0)

    oe_tile = ce_tile * 2.51284  # Optimal exposure
    oe_mask = (cd_tile < oe_tile)

    freq_correction = jnp.exp(-0.5 * cd_tile / ce_tile)
    freq_correction = jnp.multiply(freq_correction, oe_mask)
    
    # tilt_angles = angle_per_tilt * jnp.ceil(tilt_numbers / 2) 
    angle_correction = jnp.cos(tilt_angles * np.pi / 180)
    ac_tile = angle_correction[:, None] * jnp.ones([N, D * D])

    # import pdb; pdb.set_trace()
    return freq_correction * ac_tile


def evaluate_ctf_wrapper(CTF_params, image_shape, voxel_size, CTF_FUNCTION_OPTION=None ):
    return cryodrgn_CTF(CTF_params, image_shape, voxel_size)

def cryodrgn_CTF(CTF_params, image_shape, voxel_size):
    psi = get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2]
    return batch_evaluate_ctf(psi, CTF_params)
