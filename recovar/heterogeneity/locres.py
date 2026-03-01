"""Local resolution estimation and local filtering of reconstructed volumes."""

import functools
import logging

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from recovar import utils
from recovar.core import mask as mask_fn
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.reconstruction import regularization
from recovar.simulation import simulator
logger = logging.getLogger(__name__)


## A copy of the relion local resolution function. See postprocessing.cpp in relion

def integral_fsc(fsc, fourier_pixel_size = 1):
    last_idx = find_first_zero_in_bool(fsc>=0)
    include_upto = jnp.where((last_idx == 0) & (fsc[0] >= 0), fsc.size, last_idx)
    good_idx = jnp.where(jnp.arange(fsc.size) < include_upto, 1, 0)
    return np.sum(fsc * good_idx) * fourier_pixel_size

integral_fscs = jax.vmap(integral_fsc, in_axes = [0, None])


def local_resolution(map1, map2, B_factor, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = True, filter_edgewidth=2, filter_map1 = False):


    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    

    step_size = np.round(locres_sampling / voxel_size).astype(int)
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)

    if maskrad_pix < 5:
        logger.warning("radius of local resolution mask is only %s pixels. Result will probably be nonsense. Should either increase locres_maskrad or do global resolution estimate", maskrad_pix)

    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)
    # logger.info("Step size: %s, maskrad_pix: %s, edgewidth_pix: %s", step_size, maskrad_pix, edgewidth_pix)
    # myrad = map1.shape[0]//2 - 1*maskrad_pix
    # myrad = 40
    # myradf = myrad / step_size
    # sampling_points = []

    # logger.info("Starting...")

    # grid = np.array(fourier_transform_utils.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    # for kk in grid:
    #     for ii in grid:
    #         for jj in grid:
    #             rad = np.sqrt(kk * kk + ii * ii + jj * jj)
    #             if rad < myrad:
    #                 sampling_points.append((kk, ii, jj))
    # sampling_points = jnp.array(sampling_points).astype(int)

    sampling_points = get_sampling_points(map1.shape[0], locres_sampling, locres_maskrad, voxel_size)


    fourier_pixel_size = 1/(map1.shape[0] * voxel_size)
    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    # logger.info("Calculating local resolution in %s sampling points ...", nr_samplings)
    if filter_map1:
        ft_sum = fourier_transform_utils.get_dft3(map1)
    else:
        ft_sum = 0.5*(fourier_transform_utils.get_dft3(map1) + fourier_transform_utils.get_dft3(map2))
    # Need to apply B-factor here I guess
    ft_sum *= simulator.get_B_factor_scaling(map1.shape, voxel_size, -B_factor).reshape(map1.shape).astype(map1.dtype)

    i_ft_sum_orig = fourier_transform_utils.get_idft3(ft_sum).real


    # for now will do batch of 1.
    i_fil = jnp.zeros_like(map1)
    i_loc_res = 0
    i_loc_auc = 0
    i_sum_w = 0 
    local_resols, fscs = [], []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)

    # /2: local resolution holds two maps simultaneously
    vol_batch_size = utils.safe_batch_size(
        utils.get_vol_batch_size(map1.shape[0], utils.get_gpu_memory_total()) / 2)
    n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)

    for k in range(n_batch):
        batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
        batch = sampling_points[batch_st:batch_end]

        if use_v2:
            if use_filter:
                ift_sum, loc_mask, fsc, local_resol, offset, radius = batch_compute_local_fsc_v2(batch, i_ft_sum_orig, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold, use_filter, filter_edgewidth )
                i_fil = add_subarrays_to_array(i_fil, ift_sum * loc_mask, offset, int(radius[0]))
            else:
                fsc, local_resol = batch_compute_local_fsc_v2(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold , use_filter, filter_edgewidth)

        else:
            if use_filter:
                ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter, filter_edgewidth)
                i_fil += jnp.sum(ift_sum * loc_mask, axis=0)
            else:
                fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter, filter_edgewidth)

        fscs.append(fsc)
        local_resols.append(local_resol)

        if jnp.isnan(i_fil).any() or jnp.isnan(i_loc_res).any():
            logger.warning("NaNs encountered in local_resolution accumulation.")

    fscs = np.concatenate(fscs)
    local_resols = np.concatenate(local_resols)
    full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
    int_fscs = integral_fscs(fscs,fourier_pixel_size)
    i_loc_res = make_local_resol_map(sampling_points, 1/local_resols, full_mask)
    i_loc_auc = make_auc_map(sampling_points, int_fscs, full_mask)

    if not use_filter:
        return fscs, local_resols, i_loc_res, i_loc_auc
    

    # i_fil3 = jnp.where( i_sum_w > 0,  i_fil / i_sum_w, 0)
    i_fil = make_i_fil_map(sampling_points, i_fil, full_mask)
    # logger.info("Done")

    return i_fil, i_loc_res, i_loc_auc, fscs, local_resols#, sampling_points

def convolve_mask_at_sampling_points(sampling_pts, local_resols, full_mask):
    # full_array = jnp.zeros(full_mask.shape)
    # full_array = full_array.at[sampling_points].set(local_resol)

    full_array = jnp.zeros(full_mask.shape, dtype = full_mask.dtype)
    sampling_points = sampling_pts + full_mask.shape[0]//2
    
    sampling_idx = jax.numpy.ravel_multi_index(sampling_points.T, full_mask.shape)

    full_array = full_array.reshape(-1)
    full_array = full_array.at[sampling_idx].set(local_resols)
    full_array = full_array.reshape(full_mask.shape)

    return jax.scipy.signal.fftconvolve(full_mask, full_array, mode = 'same')

def make_local_resol_map(sampling_points, inv_local_resol, full_mask):

    local_resol_conv = convolve_mask_at_sampling_points(sampling_points, inv_local_resol, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(inv_local_resol), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-4,  mask_conv/ local_resol_conv, 0)
    return i_loc_res


def make_auc_map(sampling_points, auc, full_mask):

    auc_conv = convolve_mask_at_sampling_points(sampling_points, auc, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(auc), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-4,  auc_conv/ mask_conv , 0)
    return i_loc_res

def make_i_fil_map(sampling_points, i_fil, full_mask):
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones(sampling_points.shape[0]), full_mask.astype(np.float64))
    i_fil_divided = jnp.where( mask_conv > 1e-4,  i_fil / mask_conv, 0)
    return i_fil_divided


def get_subsample_indices(array_shape, offset, radius):
    # This is so complicated because the simple way doesn't JIT...

    size = 2*radius
    grid = jnp.mgrid[:size,:size,:size]
    grid = grid.reshape(3, -1).T
    grid += offset - radius
    vec_indices = jnp.ravel_multi_index(grid.T, array_shape, mode = 'clip')#.reshape(og_shape[:-1])

    good_idx =  (grid >= 0).all(axis = 1) * (grid < array_shape[0]).all(axis = 1) 

    return vec_indices, good_idx

batch_get_subsample_indices = jax.vmap(get_subsample_indices, in_axes = (None, 0, None))

def subsample_array(array, offset, radius):
    # l_bounds = offset - radius
    # u_bounds = offset + radius
    size = 2*radius
    # grid = jnp.mgrid[:size,:size,:size]
    # grid = grid.reshape(3, -1).T
    # grid += offset - radius
    # vec_indices = jnp.ravel_multi_index(grid.T, array.shape, mode = 'clip')#.reshape(og_shape[:-1])
    vec_indices, good_idx = get_subsample_indices(array.shape, offset, radius)
    return (array.ravel()[vec_indices] * good_idx).reshape(size, size, size)

@functools.partial(jax.jit, static_argnums = [3])
def add_subarray_to_array_2(array, subarray, offset, radius):
    array_shape = array.shape
    vec_indices, good_idx = get_subsample_indices(array.shape, offset, radius)
    array = array.ravel()
    array = array.at[vec_indices].add(subarray.ravel() * good_idx)
    return array.reshape(array_shape)


@functools.partial(jax.jit, static_argnums = [3])
def add_subarrays_to_array(array, subarrays, offset, radius):
    array_shape = array.shape
    vec_indices, good_idx = batch_get_subsample_indices(array.shape, offset, radius)
    array = array.ravel()
    array = array.at[vec_indices.reshape(-1)].add(subarrays.reshape(-1) * good_idx.reshape(-1))
    return array.reshape(array_shape)


@functools.partial(jax.jit, static_argnums = [3])
def add_subarray_to_array(array, subarray, offset, radius):
    l_bounds = offset - radius
    return jax.lax.dynamic_update_slice(array, subarray + subsample_array(array, offset, radius), l_bounds)


@functools.partial(jax.jit, static_argnums = [4,5,9,10])    
def compute_local_fsc_v2(offset, ift_sum_orig, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter, filter_edgewidth):

    offset = offset + map1.shape[0]//2
    # Compute masked fsc

    radius = maskrad_pix + edgewidth_pix
    # l_bounds = offset - radius
    # u_bounds = offset + radius
    multiplier = 3
    # smaller_size = multiplier*radius#
    map1_sub = subsample_array(map1, offset, multiplier*radius)
    map2_sub = subsample_array(map2, offset, multiplier*radius)
    ift_sum_sub = subsample_array(ift_sum_orig, offset, multiplier*radius)
    mask = mask_fn.raised_cosine_mask(map1_sub.shape, maskrad_pix, radius, 0)
    map1_sub = map1_sub * mask
    map2_sub = map2_sub * mask

    fsc = regularization.get_fsc(fourier_transform_utils.get_dft3(map1_sub), fourier_transform_utils.get_dft3(map2_sub), volume_shape = map1_sub.shape)


    # local_resol = jnp.argmin(fsc >= fsc_treshold)
    # # If all above threhsold
    # local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)
    local_resol = find_fsc_resol(fsc, fsc_treshold)
    local_resol = jnp.where(local_resol > 0, map1_sub.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)


    if use_filter:
        ft_sum_sub = fourier_transform_utils.get_dft3(ift_sum_sub)
        ift_sum = filter_with_local_fsc(ft_sum_sub, fsc, local_resol, voxel_size, filter_edgewidth)
        # if jnp.isnan(ift_sum).any():
        return ift_sum, mask, fsc, local_resol, offset, multiplier*radius

    return fsc, local_resol


@functools.partial(jax.jit, static_argnums = [9,10])
def compute_local_fsc(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter, filter_edgewidth):

    # Compute masked fsc

    mask = mask_fn.raised_cosine_mask(ft_sum.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    map1 = map1 * mask
    map2 = map2 * mask
    fsc = regularization.get_fsc(fourier_transform_utils.get_dft3(map1), fourier_transform_utils.get_dft3(map2), volume_shape = map1.shape)

    # first fsc above threshold

    local_resol = jnp.argmin(fsc >= fsc_treshold)
    # If all above threhsold
    local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)
    local_resol = jnp.where(local_resol > 0, map1.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)

    if use_filter:
        ift_sum = filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, filter_edgewidth)
        # if jnp.isnan(ift_sum).any():
        return ift_sum, mask, fsc, local_resol

    return fsc, local_resol

batch_compute_local_fsc = jax.vmap(compute_local_fsc, in_axes = (0, None, None, None, None, None, None, None, None, None, None ) )

batch_compute_local_fsc_v2 = jax.vmap(compute_local_fsc_v2, in_axes = (0, None, None, None, None, None, None, None, None, None, None ) )


def find_first_zero_in_bool(array):
    ires_max = jnp.argmin(array)
    # There is a corner case. If iresmax is 0 because everything is 1.
    # If ires_max is 0, there are two options: either the array is all 0 or all 1.
    all_ones_flag = (ires_max ==0 ) * (array[ires_max] == True)
    ires_max = jnp.where(all_ones_flag, array.size-1, ires_max)

    return ires_max#, all_ones_flag


def find_fsc_resol(fsc_curve, threshold = 1/7):
    ires_zero = find_first_zero_in_bool(fsc_curve >= threshold)
    ires_before = jnp.where(ires_zero > 0, ires_zero - 1, 0)
    fsc_subcurve = jnp.array([fsc_curve[ires_zero], fsc_curve[ires_before]])
    ires_interp = ires_before + jnp.interp(threshold, fsc_subcurve, jnp.array([1,0]) * 1.0 )
    # zz = jnp.interp(threshold, fsc_subcurve, jnp.arange(2) * 1.0 )

    ires_interp = jnp.where(ires_zero > 0, ires_interp, 0)
    ires_interp = jnp.where((ires_zero == fsc_curve.size-1) * fsc_curve[-1] >= threshold, fsc_curve.size-1, ires_interp)
    return ires_interp


def apply_fsc_weighting(FT, fsc):
	#  Find resolution where fsc_true drops below zero for the first time
	#  Set all weights to zero beyond that resolution
    distances = fourier_transform_utils.get_grid_of_radial_distances(FT.shape,)

    # ires_max = jnp.argmin(fsc >= 0.0001)
    ires_max = find_first_zero_in_bool(fsc >= 0.0001)
    # ires_max = jnp.where(all_ones_flag, fsc.size, ires_max) # If == 0, fsc >= 0.0001 for all ires

    fsc = jnp.where( jnp.arange(fsc.size) < ires_max, fsc, 0)
    # fsc = fsc.at[ires_max:].set(0)
    fsc = jnp.sqrt((2 * fsc) / (1 + fsc))
    fsc_mask = fsc[distances]
    FT = FT * fsc_mask
    return FT

def filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, filter_edgewidth):
    # ft_sum_inp = ft_sum.copy()
    ft_sum  = apply_fsc_weighting(ft_sum, fsc) # 
    ft_sum = low_pass_filter_map(ft_sum, ft_sum.shape[0], local_resol, voxel_size, filter_edgewidth)#.astype(ft_sum.dtype)
    ift_sum = fourier_transform_utils.get_idft3(ft_sum).real
    return  ift_sum


def low_pass_filter_map(FT, ori_size, low_pass, voxel_size, filter_edgewidth, do_highpass_instead = False):
    ires_filter = jnp.round((ori_size * voxel_size) / low_pass)
    filter_edge_halfwidth = filter_edgewidth // 2

    edge_low = jnp.maximum(0., (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = jnp.minimum(FT.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low
    res = fourier_transform_utils.get_grid_of_radial_distances(FT.shape) / ori_size
    if do_highpass_instead:
        filter = jnp.where(res <  edge_low , 0, 1)
        filter = jnp.where((res >= edge_low) * (res < edge_high), 0.5 - 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    else:
        filter = jnp.where(res <=  edge_low , 1, 0)
        filter = jnp.where( (res >= edge_low) * (res < edge_high), 0.5 + 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    filter = filter.astype(utils.dtype_to_real(FT.dtype))
    
    return FT * filter


def local_error(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    

    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)

    # mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)

    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)
    mask /= np.linalg.norm(mask)
    # Compute error with convolution
    
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, voxel_size, edgewidth_pix, do_highpass_instead = False)
        map1 = fourier_transform_utils.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, voxel_size, edgewidth_pix, do_highpass_instead = False)
        map2 = fourier_transform_utils.get_idft3(map2_ft).real

    mask_ft = fourier_transform_utils.get_dft3(mask)
    map1_square_ft = fourier_transform_utils.get_dft3(map1*map1)
    map2_square_ft = fourier_transform_utils.get_dft3(map2*map2)
    map1map2_ft = fourier_transform_utils.get_dft3(map1*map2)
    # map2_ft = fourier_transform_utils.get_dft3(map2)

    local_errors = (fourier_transform_utils.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * fourier_transform_utils.get_idft3(map1map2_ft * mask_ft) \
    + (fourier_transform_utils.get_idft3(map2_square_ft * mask_ft) )
    
    # local_errors = (fourier_transform_utils.get_idft3(map1_ft * mask_ft).real )**2 \
    # - 2 * fourier_transform_utils.get_idft3(map1_ft * map2_ft * mask_ft) \
    # + (fourier_transform_utils.get_idft3(map2_ft * mask_ft) )**2
    
    local_errors = local_errors.real

    return local_errors


def local_error_with_cov(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None, noise_variance = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    

    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)


    # Raised cosine mask for local resolution computation
    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)


    # Compute error with convolution
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, voxel_size, edgewidth_pix, do_highpass_instead = False)
        map1 = fourier_transform_utils.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, voxel_size, edgewidth_pix, do_highpass_instead = False)
        map2 = fourier_transform_utils.get_idft3(map2_ft).real

    ## Whiten maps
    if noise_variance is not None:
        noise_variance = noise_variance.reshape(map1.shape)
        map1 = fourier_transform_utils.get_idft3(fourier_transform_utils.get_dft3(map1) * jnp.sqrt(noise_variance).reshape(map1.shape)).real
        map2 = fourier_transform_utils.get_idft3(fourier_transform_utils.get_dft3(map2) * jnp.sqrt(noise_variance).reshape(map1.shape)).real

    mask_ft = fourier_transform_utils.get_dft3(mask)
    map1_square_ft = fourier_transform_utils.get_dft3(map1*map1)
    map2_square_ft = fourier_transform_utils.get_dft3(map2*map2)
    map1map2_ft = fourier_transform_utils.get_dft3(map1*map2)
    # map2_ft = fourier_transform_utils.get_dft3(map2)

    local_errors = (fourier_transform_utils.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * fourier_transform_utils.get_idft3(map1map2_ft * mask_ft) \
    + (fourier_transform_utils.get_idft3(map2_square_ft * mask_ft) )
        
    local_errors = local_errors.real

    return local_errors


def get_local_error_subvolume_rad(locres_maskrad, voxel_size, multiplier=3):
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    rad = maskrad_pix * multiplier
    return rad

def get_local_error_subvolume_size(locres_maskrad, voxel_size, multiplier=3):
    # locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    # maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    # rad = maskrad_pix * multiplier
    return 2 * get_local_error_subvolume_rad(locres_maskrad, voxel_size, multiplier=multiplier)


### This is the metric which is actually used
def expensive_local_error_with_cov(map1, map2, voxel_size, noise_variance, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, use_v2 = False, debug = False, split_shell = False):
    # Keep defaults consistent with legacy behavior when optional args are omitted.
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth


    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)

    step_size = np.round(locres_sampling / voxel_size).astype(int)
    # logger.info("Step size: %s, maskrad_pix: %s, edgewidth_pix: %s", step_size, maskrad_pix, edgewidth_pix)
    # logger.info("Compute CV metric with sampling = %s and radius = %s and edgewidth = %s", locres_sampling, locres_maskrad, locres_edgwidth)


    # myrad = 40
    # myradf = myrad / step_size
    # sampling_points = []

    # # logger.info("Starting...")

    # grid = np.array(fourier_transform_utils.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    # for kk in grid:
    #     for ii in grid:
    #         for jj in grid:
    #             rad = np.sqrt(kk * kk + ii * ii + jj * jj)
    #             if rad < myrad:
    #                 sampling_points.append((kk, ii, jj))
    # sampling_points = jnp.array(sampling_points).astype(int)

    sampling_points = get_sampling_points(map1.shape[0], locres_sampling, locres_maskrad, voxel_size)

    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    # logger.info("Calculating local error in %s sampling points ...", nr_samplings)
    

    # for now will do batch of 1.
    diffs = []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)
    noise_variance = jnp.asarray(noise_variance)

    # if np.log2(map1.shape[0]) % 1 != 0:
    #     raise ValueError("Map size must be a power of 2")
    
    # sqrt_noise_variance_real = fourier_transform_utils.get_idft3(jnp.sqrt(noise_variance))
    # radius = maskrad_pix + edgewidth_pix
    # multiplier = 2
    # want_size = 2*(multiplier*radius)
    # actual_size = int(2**(np.ceil(np.log2(want_size))))
    # factor = map1.shape[0]//actual_size 
    # from skimage.transform import downscale_local_mean
    
    if use_v2:
        # multiplier = 3
        # rad = maskrad_pix * multiplier
        rad = get_local_error_subvolume_rad(locres_maskrad, voxel_size)
        # downsampled_noise_variance_ift = fourier_transform_utils.get_idft3(jnp.sqrt(noise_variance) )
        downsampled_noise_variance_ift = fourier_transform_utils.get_idft3((noise_variance) )#.real

        downsampled_noise_variance_ift_subs = subsample_array(downsampled_noise_variance_ift, map1.shape[0]//2+1, rad)
        noise_variance_small = fourier_transform_utils.get_dft3(downsampled_noise_variance_ift_subs ) * noise_variance.size / downsampled_noise_variance_ift_subs.size


    # map1_sub = subsample_array(diff, offset, multiplier*radius)
    # use_v2 = True
    diff_map = jnp.array(map1- map2)
    noise_variance = jnp.array(noise_variance)

    use_v3 = False

    single_batch = not use_v2
    if single_batch:
        # AssertionError("Not implemented")
        for k in range(nr_samplings):
            batch = sampling_points[k]
            if use_v2:
                # By noting that diagonal blocks of MCM are toeplitz, we can compute this fast
                diff = masked_noisy_error_3(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
            else:
                diff = masked_noisy_error(diff_map, noise_variance, batch, maskrad_pix, edgewidth_pix )
            diffs.append(diff[None])
            # i_loc_res += loc_mask * diff
            # i_sum_w += loc_mask
            if k % 1000 ==0:
                logger.debug("%d", k)


    else:
        # Doesn't seem to be faster.
        if use_v2:
            vol_batch_size = utils.safe_batch_size(
                utils.get_vol_batch_size(noise_variance_small.shape[0], utils.get_gpu_memory_total()))
        else:
            # /4: noise estimation holds 4 volume-sized arrays (two maps + two noise arrays)
            vol_batch_size = utils.safe_batch_size(
                utils.get_vol_batch_size(noise_variance.shape[0], utils.get_gpu_memory_total()) / 4)

        if use_v3:
            # Divide by half the volume size to fit many small subvolumes in memory
            vol_batch_size = utils.safe_batch_size(
                utils.get_vol_batch_size(noise_variance_small.shape[0], utils.get_gpu_memory_total()) / (noise_variance_small.shape[0]//2))

        n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)

        for k in range(n_batch):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
            batch = sampling_points[batch_st:batch_end]

            if split_shell:
                ### NOTE!!!
                ### THIS IS THE OPTION WHICH IS ACTUALLY USED. PROBABLY SHOULD CLEAN UP THE REST
                if use_v3:
                    # By noting that diagonal blocks of MCM are toeplitz, we can compute this fast
                    diff = batch_masked_noisy_error_split_over_shells_v2(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
                    # diff2 = batch_masked_noisy_error_split_over_shells(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
                else:
                    diff = batch_masked_noisy_error_split_over_shells(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
            else:
                if use_v2:
                    # By noting that diagonal blocks of MCM are toeplitz, we can compute this fast
                    diff = batch_masked_noisy_error_3(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
                else:
                    diff = batch_masked_noisy_error(diff_map, noise_variance, batch, maskrad_pix, edgewidth_pix )


            diffs.append(diff)
            # if k % 10 ==0:


    diffs = np.concatenate(diffs)
    if split_shell:
        return diffs

    full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    i_local_error = make_auc_map(sampling_points, diffs, full_mask)

    # i_fil = jnp.where( i_sum_w > 0,  i_loc_res / i_sum_w, 0)
    # return diffs
    if debug:
        return i_local_error, diffs
    
    return i_local_error


@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error(diff, noise_variance, offset, maskrad_pix, edgewidth_pix ):
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    diff = fourier_transform_utils.get_dft3((diff) * mask) * jnp.sqrt(noise_variance)
    return jnp.linalg.norm(diff)**2


@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error_3(diff, noise_variance_small, offset, maskrad_pix, edgewidth_pix ):

    ### THIS ALMOST WORKS BUT NOT QUITE. SOME SCALING FACTOR IN HERE
    # radius = maskrad_pix + edgewidth_pix
    # # l_bounds = offset - radius
    # # u_bounds = offset + radius
    # multiplier = 3
    # smaller_size = multiplier*radius#

    multiplier = 3
    offset += diff.shape[0]//2

    diff = subsample_array(diff, offset, multiplier*maskrad_pix)
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, edgewidth_pix, 0)
    diff_masked = diff * mask

    diff_masked = fourier_transform_utils.get_dft3(diff_masked) * jnp.sqrt(noise_variance_small)

    # mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    # diff = fourier_transform_utils.get_dft3((diff) * mask) * jnp.sqrt(noise_variance)
    return jnp.linalg.norm(diff_masked)**2


def split_by_shells(input_vec, volume_shape ):
    radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = 0).astype(int).reshape(-1) 
    
    split_by_shell = jnp.zeros((input_vec.size, volume_shape[0]//2 ), dtype = input_vec.dtype)
    split_by_shell = split_by_shell.at[(jnp.arange(input_vec.size), radial_distances)].set(input_vec)
    full_shape = split_by_shell.shape
    indices = jnp.stack([jnp.arange(input_vec.size), radial_distances], axis=0)
    
    good_indices = radial_distances < volume_shape[0]//2 
    sampling_idx = jax.numpy.ravel_multi_index(indices, split_by_shell.shape,mode='clip') #, fill_value=-1)
    
    # Some serious JAX hacking going on here to ignore bad indices. Probably should change this.
    sampling_idx = jnp.where(good_indices, sampling_idx, split_by_shell.size+1) 
    
    # This seems silly but not sure how else to do it.
    split_by_shell = split_by_shell.reshape(-1)
    split_by_shell = split_by_shell.at[sampling_idx].set(input_vec,mode='drop')
    split_by_shell = split_by_shell.reshape(full_shape)
    return split_by_shell


@functools.partial(jax.jit, static_argnums = [3,4,5])    
def masked_noisy_error_split_over_shells(diff, noise_variance_small, offset, maskrad_pix, edgewidth_pix,multiplier =3 ):
    # NOT IMPLEMENTED
    offset += diff.shape[0]//2

    diff = subsample_array(diff, offset, multiplier*maskrad_pix)
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, edgewidth_pix, 0)
    diff_masked = diff * mask
    diff_masked = fourier_transform_utils.get_dft3(diff_masked) * jnp.sqrt(noise_variance_small)

    diff_masked_norm = regularization.sum_over_shells(jnp.abs(diff_masked)**2, diff_masked.shape)

    return diff_masked_norm


@functools.partial(jax.jit, static_argnums = [3,4,5])    
def masked_noisy_error_split_over_shells_v2(diff, noise_variance_small, offset, maskrad_pix, edgewidth_pix,multiplier =3 ):
    # NOT IMPLEMENTED
    offset += diff.shape[0]//2

    # y = S M diff
    diff = subsample_array(diff, offset, multiplier*maskrad_pix)
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, edgewidth_pix, 0)
    diff_ft = fourier_transform_utils.get_dft3(diff_masked)

    # Now to compute (S M D M^* S^*)^dagger
    # = 
    split_diff = split_by_shells(diff_ft, diff.shape ) #* jnp.sqrt(noise_variance_small)
    split_diff = fourier_transform_utils.get_idft3(split_diff) * mask
    split_diff = fourier_transform_utils.get_dft3(split_diff) * jnp.sqrt(noise_variance_small)

    diff_masked_norm = jnp.sum(jnp.abs(split_diff)**2, axis=(-1,-2,-3))

    return diff_masked_norm

def split_by_shells(input_vec, volume_shape ):
    radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = 0).astype(int).reshape(-1) 
    
    split_by_shell = jnp.zeros((volume_shape[0]//2 , input_vec.size), dtype = input_vec.dtype)
    # split_by_shell = split_by_shell.at[(jnp.arange(input_vec.size), radial_distances)].set(input_vec)
    full_shape = split_by_shell.shape
    indices = jnp.stack([radial_distances, jnp.arange(input_vec.size)], axis=0)
    
    good_indices = radial_distances < volume_shape[0]//2 
    sampling_idx = jax.numpy.ravel_multi_index(indices, split_by_shell.shape,mode='clip') #, fill_value=-1)
    
    # Some serious JAX hacking going on here to ignore bad indices. Probably should change this.
    sampling_idx = jnp.where(good_indices, sampling_idx, split_by_shell.size+1) 
    
    # This seems silly but not sure how else to do it.
    split_by_shell = split_by_shell.reshape(-1)
    split_by_shell = split_by_shell.at[sampling_idx].set(input_vec.reshape(-1),mode='drop')
    split_by_shell = split_by_shell.reshape(full_shape[0], *input_vec.shape )
    return split_by_shell


batch_subsample_array_at_same_offset = jax.vmap(subsample_array, in_axes = (0, None, None) )

@functools.partial(jax.jit, static_argnums = [3,4,5])    
def recombine_with_choice(estimators, choice, offset, maskrad_pix, edgewidth_pix, multiplier =3):
    # NOT IMPLEMENTED
    # multiplier = 3
    offset += estimators.shape[1]//2
    estimators_subsampled = batch_subsample_array_at_same_offset(estimators, offset, multiplier*maskrad_pix)

    mask = mask_fn.raised_cosine_mask(estimators_subsampled.shape[1:], maskrad_pix, edgewidth_pix, 0)
    estimators_subsampled = estimators_subsampled * mask[None]
    estimators_subsampled_ft = fourier_transform_utils.get_dft3(estimators_subsampled)

    choice_radial = utils.make_radial_image(choice, estimators_subsampled.shape[1:]).reshape(estimators_subsampled.shape[1:])
    # combined_est = jnp.take(estimators_subsampled, choice_radial, axis = 0)
    combined_est = jnp.take_along_axis(estimators_subsampled_ft , choice_radial[None], axis=0)[0]
    combined_est = fourier_transform_utils.get_idft3(combined_est).real

    return combined_est


batch_masked_noisy_error = jax.vmap(masked_noisy_error, in_axes = (None,None,0, None, None) )
batch_masked_noisy_error_3 = jax.vmap(masked_noisy_error_3, in_axes = (None,None,0, None, None) )

batch_masked_noisy_error_split_over_shells = jax.vmap(masked_noisy_error_split_over_shells, in_axes = (None,None,0, None, None) )

batch_masked_noisy_error_split_over_shells_v2 = jax.vmap(masked_noisy_error_split_over_shells_v2, in_axes = (None,None,0, None, None) )

def recombine_estimates(estimators, choice, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    # Default edge width kept at 0 for recombination path.
    locres_edgwidth = 0 if locres_edgwidth is None else locres_edgwidth
    
    logger.info("Recombining estimate with sampling = %s and radius = %s and edgewidth = %s", locres_sampling, locres_maskrad, locres_edgwidth)

    step_size = np.round(locres_sampling / voxel_size).astype(int)
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)

    if maskrad_pix < 5:
        logger.warning("radius of local resolution mask is only %s pixels. Result will probably be nonsense. Should either increase locres_maskrad or do global resolution estimate", maskrad_pix)

    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)
    logger.info("Step size: %s, maskrad_pix: %s, edgewidth_pix: %s", step_size, maskrad_pix, edgewidth_pix)
    logger.info("Starting...")
    sampling_points = get_sampling_points(estimators.shape[1], locres_sampling, locres_maskrad, voxel_size)

    nr_samplings = sampling_points.shape[0]
    logger.info("Recombining estimates at %s sampling points ...", nr_samplings)

    # for now will do batch of 1.
    # Put stuff on GPU
    estimators = jnp.asarray(estimators)
    optimized_estimator = jnp.zeros_like(estimators[0])
    n_batch = sampling_points.shape[0]
    multiplier = 3
    radius =  multiplier*maskrad_pix

    # with jax.disable_jit():
    for k in range(n_batch):
        offset = sampling_points[k]
        est_patch = recombine_with_choice(estimators, choice[k], offset, maskrad_pix, edgewidth_pix, multiplier )
        offset = offset + estimators.shape[1]//2
        optimized_estimator = add_subarrays_to_array(optimized_estimator, est_patch[None], offset[None], int(radius))

    full_mask = mask_fn.raised_cosine_mask(optimized_estimator.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
    optimized_estimator = make_i_fil_map(sampling_points, optimized_estimator, full_mask)

    return optimized_estimator


def get_sampling_points(grid_size, locres_sampling, locres_maskrad, voxel_size):
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    step_size = np.round(locres_sampling / voxel_size).astype(int)
    myrad = grid_size//2 - maskrad_pix

    sampling_points = []
    grid = np.array(fourier_transform_utils.get_1d_frequency_grid(grid_size, 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)
    return sampling_points


def make_sampling_volume(grid_size, locres_sampling, voxel_size, locres_maskrad):
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    step_size = np.round(locres_sampling / voxel_size).astype(int)

    sampling_points = get_sampling_points(grid_size, locres_sampling, locres_maskrad, voxel_size)
    # Sampling points are centered at 0. We need to recenter them at half the grid
    sampling_points += grid_size//2
    # Dump 
    volume = np.ones((grid_size, grid_size, grid_size), dtype = np.float16) * -1
    for k in range(sampling_points.shape[0]):
        half_step = step_size // 2
        volume[sampling_points[k,0]-half_step:sampling_points[k,0]+half_step, sampling_points[k,1]-half_step:sampling_points[k,1]+half_step, sampling_points[k,2]-half_step:sampling_points[k,2]+half_step] = k
    return volume

def filter_with_global_fsc(ft_sum, fsc, voxel_size, filter_edgewidth, mask=None, fsc_mask=None, B_factor=None):
    """
    Apply global FSC-based filtering to a Fourier transform.
    
    Parameters:
    - ft_sum: Fourier transform of the map to filter
    - fsc: Fourier Shell Correlation curve
    - voxel_size: Voxel size in Angstroms
    - filter_edgewidth: Width of the filter edge in pixels
    - mask: Optional mask to apply to final result (if None, no mask applied)
    - fsc_mask: Optional mask used only for FSC estimation (if None, no FSC mask)
    - B_factor: Optional B-factor for sharpening (if None, no B-factor applied)
    
    Returns:
    - Filtered map in real space
    """
    # Apply B-factor sharpening if specified
    if B_factor is not None:
        B_factor_scaling = simulator.get_B_factor_scaling(ft_sum.shape, voxel_size, -B_factor).reshape(ft_sum.shape)
        ft_sum = ft_sum * B_factor_scaling.astype(ft_sum.dtype)
    
    # Apply FSC weighting
    ft_sum = apply_fsc_weighting(ft_sum, fsc)
    
    # Find global resolution from FSC curve
    global_resol = find_fsc_resol(fsc, 1/7)  # Using 1/7 threshold like in local filtering
    global_resol = ft_sum.shape[0] * voxel_size / global_resol if global_resol > 0 else 999
    
    # Apply low-pass filter
    ft_sum = low_pass_filter_map(ft_sum, ft_sum.shape[0], global_resol, voxel_size, filter_edgewidth)
    
    # Convert to real space
    ift_sum = fourier_transform_utils.get_idft3(ft_sum).real
    
    # Apply mask if provided
    if mask is not None:
        ift_sum = ift_sum * mask
    
    return ift_sum


def filter_with_global_fsc_and_mask(ft_sum, fsc, voxel_size, filter_edgewidth, mask_radius=None, mask_edgewidth=None, fsc_mask_radius=None, fsc_mask_edgewidth=None, B_factor=None, mask=None, fsc_mask=None):
    """
    Apply global FSC-based filtering with automatic spherical mask.
    
    Parameters:
    - ft_sum: Fourier transform of the map to filter
    - fsc: Fourier Shell Correlation curve
    - voxel_size: Voxel size in Angstroms
    - filter_edgewidth: Width of the filter edge in pixels
    - mask_radius: Radius of the spherical mask for final result (if None, no mask applied)
    - mask_edgewidth: Width of the mask edge (if None, uses 10% of mask_radius)
    - fsc_mask_radius: Radius of the spherical mask for FSC estimation (if None, no FSC mask)
    - fsc_mask_edgewidth: Width of the FSC mask edge (if None, uses 10% of fsc_mask_radius)
    - B_factor: Optional B-factor for sharpening (if None, no B-factor applied)
    - mask: Custom mask for final result (overrides mask_radius/mask_edgewidth if provided)
    - fsc_mask: Custom mask for FSC estimation (overrides fsc_mask_radius/fsc_mask_edgewidth if provided)
    
    Returns:
    - Filtered map in real space with spherical mask applied
    """
    # Create spherical mask for final result if parameters provided and no custom mask
    if mask is None and mask_radius is not None:
        if mask_edgewidth is None:
            mask_edgewidth = mask_radius * 0.1  # 10% of radius as default edge width
        
        # Convert to pixels
        maskrad_pix = np.round(mask_radius / voxel_size).astype(int)
        edgewidth_pix = np.round(mask_edgewidth / voxel_size).astype(int)
        
        # Create raised cosine mask
        mask = mask_fn.raised_cosine_mask(ft_sum.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    
    return filter_with_global_fsc(ft_sum, fsc, voxel_size, filter_edgewidth, mask, fsc_mask, B_factor)


def filter_maps_with_global_fsc(map1, map2, voxel_size, filter_edgewidth=2, mask_radius=None, mask_edgewidth=None, fsc_mask_radius=None, fsc_mask_edgewidth=None, B_factor=None, fsc_threshold=1/7, mask=None, fsc_mask=None):
    """
    Convenience function to filter two maps using global FSC-based filtering.
    
    Parameters:
    - map1: First map
    - map2: Second map  
    - voxel_size: Voxel size in Angstroms
    - filter_edgewidth: Width of the filter edge in pixels (default: 2)
    - mask_radius: Radius of the spherical mask for final result (if None, no mask applied)
    - mask_edgewidth: Width of the mask edge (if None, uses 10% of mask_radius)
    - fsc_mask_radius: Radius of the spherical mask for FSC estimation (if None, no FSC mask)
    - fsc_mask_edgewidth: Width of the FSC mask edge (if None, uses 10% of fsc_mask_radius)
    - B_factor: Optional B-factor for sharpening (if None, no B-factor applied)
    - fsc_threshold: FSC threshold for resolution determination (default: 1/7)
    - mask: Custom mask for final result (overrides mask_radius/mask_edgewidth if provided)
    - fsc_mask: Custom mask for FSC estimation (overrides fsc_mask_radius/fsc_mask_edgewidth if provided)
    
    Returns:
    - filtered_combined: Filtered version of the combined map (average of map1 and map2)
    - fsc: The FSC curve used for filtering
    - global_resol: The global resolution determined from FSC
    """
    # Create FSC mask if specified and no custom FSC mask provided
    if fsc_mask is None and fsc_mask_radius is not None:
        if fsc_mask_edgewidth is None:
            fsc_mask_edgewidth = fsc_mask_radius * 0.1  # 10% of radius as default edge width
        
        # Convert to pixels
        fsc_maskrad_pix = np.round(fsc_mask_radius / voxel_size).astype(int)
        fsc_edgewidth_pix = np.round(fsc_mask_edgewidth / voxel_size).astype(int)
        
        # Create raised cosine mask for FSC estimation
        fsc_mask = mask_fn.raised_cosine_mask(map1.shape, fsc_maskrad_pix, fsc_maskrad_pix + fsc_edgewidth_pix, -1)
    
    # Apply FSC mask to maps for FSC computation if specified
    map1_for_fsc = map1 * fsc_mask if fsc_mask is not None else map1
    map2_for_fsc = map2 * fsc_mask if fsc_mask is not None else map2
    
    # Compute FSC between the masked maps
    fsc = regularization.get_fsc(fourier_transform_utils.get_dft3(map1_for_fsc), fourier_transform_utils.get_dft3(map2_for_fsc), volume_shape=map1.shape)
    
    # Find global resolution
    global_resol_idx = find_fsc_resol(fsc, fsc_threshold)
    global_resol = map1.shape[0] * voxel_size / global_resol_idx if global_resol_idx > 0 else 999
    
    # Create combined Fourier transform (average of both maps)
    ft_sum = 0.5 * (fourier_transform_utils.get_dft3(map1) + fourier_transform_utils.get_dft3(map2))
    
    # Create final mask if specified and no custom mask provided
    if mask is None and mask_radius is not None:
        if mask_edgewidth is None:
            mask_edgewidth = mask_radius * 0.1  # 10% of radius as default edge width
        
        # Convert to pixels
        maskrad_pix = np.round(mask_radius / voxel_size).astype(int)
        edgewidth_pix = np.round(mask_edgewidth / voxel_size).astype(int)
        
        # Create raised cosine mask
        mask = mask_fn.raised_cosine_mask(ft_sum.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    
    # Apply global filtering to the combined map
    filtered_combined = filter_with_global_fsc(ft_sum, fsc, voxel_size, filter_edgewidth, mask, None, B_factor)
    
    return filtered_combined, fsc, global_resol


def filter_single_map_with_global_fsc(map1, map2, voxel_size, filter_edgewidth=2, mask_radius=None, mask_edgewidth=None, fsc_mask_radius=None, fsc_mask_edgewidth=None, B_factor=None, fsc_threshold=1/7, mask=None, fsc_mask=None):
    """
    Filter a single map using FSC computed from two maps.
    
    Parameters:
    - map1: Map to filter
    - map2: Reference map for FSC computation
    - voxel_size: Voxel size in Angstroms
    - filter_edgewidth: Width of the filter edge in pixels (default: 2)
    - mask_radius: Radius of the spherical mask for final result (if None, no mask applied)
    - mask_edgewidth: Width of the mask edge (if None, uses 10% of mask_radius)
    - fsc_mask_radius: Radius of the spherical mask for FSC estimation (if None, no FSC mask)
    - fsc_mask_edgewidth: Width of the FSC mask edge (if None, uses 10% of fsc_mask_radius)
    - B_factor: Optional B-factor for sharpening (if None, no B-factor applied)
    - fsc_threshold: FSC threshold for resolution determination (default: 1/7)
    - mask: Custom mask for final result (overrides mask_radius/mask_edgewidth if provided)
    - fsc_mask: Custom mask for FSC estimation (overrides fsc_mask_radius/fsc_mask_edgewidth if provided)
    
    Returns:
    - filtered_map: Filtered version of map1
    - fsc: The FSC curve used for filtering
    - global_resol: The global resolution determined from FSC
    """
    # Create FSC mask if specified and no custom FSC mask provided
    if fsc_mask is None and fsc_mask_radius is not None:
        if fsc_mask_edgewidth is None:
            fsc_mask_edgewidth = fsc_mask_radius * 0.1  # 10% of radius as default edge width
        
        # Convert to pixels
        fsc_maskrad_pix = np.round(fsc_mask_radius / voxel_size).astype(int)
        fsc_edgewidth_pix = np.round(fsc_mask_edgewidth / voxel_size).astype(int)
        
        # Create raised cosine mask for FSC estimation
        fsc_mask = mask_fn.raised_cosine_mask(map1.shape, fsc_maskrad_pix, fsc_maskrad_pix + fsc_edgewidth_pix, -1)
    
    # Apply FSC mask to maps for FSC computation if specified
    map1_for_fsc = map1 * fsc_mask if fsc_mask is not None else map1
    map2_for_fsc = map2 * fsc_mask if fsc_mask is not None else map2
    
    # Compute FSC between the masked maps
    fsc = regularization.get_fsc(fourier_transform_utils.get_dft3(map1_for_fsc), fourier_transform_utils.get_dft3(map2_for_fsc), volume_shape=map1.shape)
    
    # Find global resolution
    global_resol_idx = find_fsc_resol(fsc, fsc_threshold)
    global_resol = map1.shape[0] * voxel_size / global_resol_idx if global_resol_idx > 0 else 999
    
    # Create final mask if specified and no custom mask provided
    if mask is None and mask_radius is not None:
        if mask_edgewidth is None:
            mask_edgewidth = mask_radius * 0.1  # 10% of radius as default edge width
        
        # Convert to pixels
        maskrad_pix = np.round(mask_radius / voxel_size).astype(int)
        edgewidth_pix = np.round(mask_edgewidth / voxel_size).astype(int)
        
        # Create raised cosine mask
        mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    
    # Apply global filtering to map1
    ft1 = fourier_transform_utils.get_dft3(map1)
    filtered_map = filter_with_global_fsc(ft1, fsc, voxel_size, filter_edgewidth, mask, None, B_factor)
    
    return filtered_map, fsc, global_resol
