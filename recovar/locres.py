import numpy as np
import jax.numpy as jnp
from recovar import mask as mask_fn
from recovar.fourier_transform_utils import fourier_transform_utils
import recovar
from recovar import utils, simulator
ftu = fourier_transform_utils(jnp)
import logging
# from tqdm.notebook import tqdm
import jax
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


## A copy of the relion local resolution function. See postprocessing.cpp in relion

def integral_fsc(fsc):
    last_idx = find_first_zero_in_bool(fsc>=0)
    good_idx = jnp.where(jnp.arange(fsc.size) <= last_idx, 1, 0)
    return np.sum(fsc * good_idx) 
integral_fscs = jax.vmap(integral_fsc)


def local_resolution(map1, map2, B_factor, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = True):

    if use_filter:
        use_v2 = False

    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size


    step_size = np.round(locres_sampling / angpix).astype(int)
    maskrad_pix = np.round(locres_maskrad / angpix).astype(int)
    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)
    logger.info(f"Step size: {step_size}, maskrad_pix: {maskrad_pix}, edgewidth_pix: {edgewidth_pix}")
    myrad = map1.shape[0]//2 - maskrad_pix
    # myrad = 40
    myradf = myrad / step_size
    sampling_points = []

    logger.info(f"Starting...")

    grid = np.array(ftu.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)

    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    logger.info(f"Calculating local resolution in {nr_samplings} sampling points ...")

    ft_sum = 0.5*(ftu.get_dft3(map1) + ftu.get_dft3(map2))
    # Need to apply B-factor here I guess
    ft_sum *= simulator.get_B_factor_scaling(map1.shape, voxel_size, -B_factor).reshape(map1.shape)


    # for now will do batch of 1.
    i_fil = 0
    i_loc_res = 0
    i_loc_auc = 0
    i_sum_w = 0 
    local_resols, fscs = [], []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)
    single_batch = False
    if single_batch:

        for k in range(nr_samplings):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, 1, k)
            batch = sampling_points[batch_st:batch_end]

            if use_v2:
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol = compute_local_fsc_v2(batch[0], ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter)

                    i_fil += ift_sum * loc_mask
                    i_loc_res += loc_mask / local_resol
                    i_sum_w += loc_mask
                else:
                    fsc, local_resol = compute_local_fsc_v2(batch[0], ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter)

            else:
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol = compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter)

                    i_fil += ift_sum * loc_mask
                    i_loc_res += loc_mask / local_resol
                    i_sum_w += loc_mask
                else:
                    fsc, local_resol = compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter)



            fscs.append(fsc)
            local_resols.append(local_resol[None])
            if k % 100 == 0:
                logger.info(f"Sampling point {k} out of {nr_samplings} done")

            if jnp.isnan(i_fil).any() or jnp.isnan(i_loc_res).any():
                import pdb; pdb.set_trace()
            # print(k)
    else:
        # a =1
        # Doesn't seem to be faster.
        vol_batch_size = recovar.utils.get_vol_batch_size(map1.shape[0], recovar.utils.get_gpu_memory_total())/2
        if use_v2:
            vol_batch_size *=4
        # print(vol_batch_size)
        n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)
        # print(n_batch)

        for k in range(n_batch):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
            batch = sampling_points[batch_st:batch_end]

            if use_v2:
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc_v2(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold, use_filter )
                    i_fil += ift_sum * loc_mask
                    i_loc_res += loc_mask / local_resol
                    i_sum_w += loc_mask
                else:
                    fsc, local_resol = batch_compute_local_fsc_v2(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold , use_filter)

            else:
                    
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter)

                    i_fil += jnp.sum(ift_sum * loc_mask, axis=0)
                    i_loc_auc += jnp.sum(loc_mask * integral_fscs(fsc)[:,None,None,None], axis=0)

                    i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)

                    i_sum_w += jnp.sum(loc_mask, axis=0)
                else:
                    fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter)

                # i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)
                # i_sum_w += jnp.sum(loc_mask, axis=0)

            # ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, 1/8,  True)

            # i_fil += jnp.sum(ift_sum * loc_mask, axis=0)
            # i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)
            # i_sum_w += jnp.sum(loc_mask, axis=0)
            fscs.append(fsc)
            local_resols.append(local_resol)
            if k % 100 == 0:
                logger.info(f"Sampling point batch {k} out of {n_batch} done")

            if jnp.isnan(i_fil).any() or jnp.isnan(i_loc_res).any():
                import pdb; pdb.set_trace()

    fscs = np.concatenate(fscs)
    local_resols = np.concatenate(local_resols)

    if not use_filter:
        full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
        i_loc_res = make_local_resol_map(sampling_points, 1/local_resols, full_mask)
        int_fscs = integral_fscs(fscs)
        i_loc_auc = make_auc_map(sampling_points, int_fscs, full_mask)
        return fscs, local_resols, i_loc_res, i_loc_auc
    
    i_fil = jnp.where( i_sum_w > 0,  i_fil / i_sum_w, 0)

    # i_loc_res_og = i_loc_res.copy()
    # Does this make sense?
    i_loc_res = jnp.where( i_sum_w > 0,  1/ (i_loc_res/ i_sum_w) , 0)
    # i_loc_res = jnp.where( i_sum_w > 0,  i_sum_w/ i_loc_res, 0)

    i_loc_auc = jnp.where( i_sum_w > 0,  i_loc_auc / i_sum_w, 0)

    # int_fscs = integral_fscs(fscs)
    # full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
    # i_loc_auc2 = make_auc_map(sampling_points, int_fscs, full_mask)
    # import pdb; pdb.set_trace()

    # full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    # i_loc_res2 = make_local_resol_map(sampling_points, local_resols, full_mask)

    # import matplotlib.pyplot as plt
    # plt.figure(); plt.imshow(i_sum_w2.sum(axis=0));  plt.colorbar(); plt.show()
    # plt.figure(); plt.imshow((i_sum_w - i_sum_w2).sum(axis=0));  plt.colorbar(); plt.show()
    # np.linalg.norm(i_sum_w - i_sum_w2)
    # np.linalg.norm(i_sum_w - i_sum_w2)

    # plt.figure(); plt.imshow(i_loc_res2.sum(axis=0));  plt.colorbar(); plt.show()
    # plt.figure(); plt.imshow(i_loc_res.sum(axis=0));  plt.colorbar(); plt.show()
    # plt.figure(); plt.imshow((i_loc_res2 - i_loc_res).sum(axis=0));  plt.colorbar(); plt.show()

    # i_sum_w2 = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(local_resols), full_mask)

    # local_resol_conv = convolve_mask_at_sampling_points(sampling_points, 1/local_resols, full_mask)

    # plt.figure(); plt.imshow(i_loc_res_og.sum(axis=0)); plt.colorbar(); plt.show()
    # plt.figure(); plt.imshow(local_resol_conv.sum(axis=0));plt.colorbar(); plt.show()
    # plt.figure(); plt.imshow((i_loc_res_og - local_resol_conv).sum(axis=0)); plt.show()
    # # mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(local_resol), full_mask
    # import pdb; pdb.set_trace()

    return i_fil, i_loc_res, i_loc_auc, fscs, local_resols#, sampling_points

import jax.scipy
def convolve_mask_at_sampling_points(sampling_pts, local_resols, full_mask):
    # full_array = jnp.zeros(full_mask.shape)
    # full_array = full_array.at[sampling_points].set(local_resol)

    full_array = jnp.zeros(full_mask.shape)
    sampling_points = sampling_pts + full_mask.shape[0]//2
    
    sampling_idx = jax.numpy.ravel_multi_index(sampling_points.T, full_mask.shape)

    full_array = full_array.reshape(-1)
    full_array = full_array.at[sampling_idx].set(local_resols)
    full_array = full_array.reshape(full_mask.shape)

    return jax.scipy.signal.fftconvolve(full_mask, full_array, mode = 'same')

def make_local_resol_map(sampling_points, inv_local_resol, full_mask):

    local_resol_conv = convolve_mask_at_sampling_points(sampling_points, inv_local_resol, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(inv_local_resol), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-8,  mask_conv/ local_resol_conv, 0)

    return i_loc_res


def make_auc_map(sampling_points, auc, full_mask):

    auc_conv = convolve_mask_at_sampling_points(sampling_points, auc, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(auc), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-8,  auc_conv/ mask_conv , 0)

    return i_loc_res


def subsample_array(array, offset, radius):
    l_bounds = offset - radius
    # u_bounds = offset + radius
    size = (2 * radius, 2 * radius, 2 * radius)
    subarray  = jax.lax.dynamic_slice(array, l_bounds, size)
    return subarray


def add_subarray_to_array(array, subarray, offset, radius):
    l_bounds = offset - radius
    u_bounds = offset + radius
    # subarray jax.lax.dynamic_slice(array, l_bounds, [2 * radius, 2 * radius, 2 * radius])
    array[l_bounds[0]:u_bounds[0], l_bounds[1]:u_bounds[1], l_bounds[2]:u_bounds[2]] = subarray
    return array

# def get_

import functools
import jax
@functools.partial(jax.jit, static_argnums = [4,5,9])    
def compute_local_fsc_v2(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter):

    offset = offset + map1.shape[0]//2

    # Compute masked fsc

    radius = maskrad_pix + edgewidth_pix
    # l_bounds = offset - radius
    # u_bounds = offset + radius
    multiplier = 2
    # smaller_size = multiplier*radius#
    map1_sub = subsample_array(map1, offset, multiplier*radius)
    # print(map1_sub.shape)
    map2_sub = subsample_array(map2, offset, multiplier*radius)
    ft_sum_sub = subsample_array(ft_sum, offset, multiplier*radius)

    # import pdb; pdb.set_trace()

    # print(map1_sub.shape, map2_sub.shape)
    mask = mask_fn.raised_cosine_mask(map1_sub.shape, maskrad_pix, radius, 0)
    map1_sub = map1_sub * mask
    map2_sub = map2_sub * mask

    fsc = recovar.regularization.get_fsc(ftu.get_dft3(map1_sub), ftu.get_dft3(map2_sub), volume_shape = map1_sub.shape)


    local_resol = jnp.argmin(fsc >= fsc_treshold)
    # If all above threhsold
    local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)

    local_resol = find_fsc_resol(fsc, fsc_treshold)

    local_resol = jnp.where(local_resol > 0, map1_sub.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(map1_sub.sum(axis=0))
    # plt.show()

    # plt.figure()
    # plt.imshow(mask.sum(axis=0))
    # plt.show()

    # plt.figure()

    # plt.imshow(map1.sum(axis=0))
    # plt.show()
    # # plt.figure()
    # # plt.plot(fsc)
    # # plt.show()
    # import pdb; pdb.set_trace()

    if use_filter:
        ift_sum = filter_with_local_fsc(ft_sum_sub, fsc, local_resol, voxel_size, edgewidth_pix)
        # if jnp.isnan(ift_sum).any():
        #     import pdb; pdb.set_trace()
        return ift_sum, mask, fsc, local_resol

    return fsc, local_resol

import functools
import jax
# def compute_local_fsc(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = 1/7, use_filter = True):
@functools.partial(jax.jit, static_argnums = [9])    
def compute_local_fsc(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter):

    # Compute masked fsc

    mask = mask_fn.raised_cosine_mask(ft_sum.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    map1 = map1 * mask
    map2 = map2 * mask
    fsc = recovar.regularization.get_fsc(ftu.get_dft3(map1), ftu.get_dft3(map2), volume_shape = map1.shape)

    # first fsc above threshold

    local_resol = jnp.argmin(fsc >= fsc_treshold)
    # If all above threhsold
    local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)
    local_resol = jnp.where(local_resol > 0, map1.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)

    # import matplotlib.pyplot as plt
    # radius = maskrad_pix + edgewidth_pix

    # map1_sub = subsample_array(mask, offset[0] + map1.shape[0]//2, radius)

    # offset = offset[0]
    # l_bounds = offset - radius + map1.shape[0]//2
    # u_bounds = offset + radius + map1.shape[0]//2
    # mask2 = mask[l_bounds[0]:u_bounds[0], l_bounds[1]:u_bounds[1], l_bounds[2]:u_bounds[2]] 
    # plt.figure()
    # plt.imshow(map1_sub.sum(axis=0))
    # plt.show()
    # plt.figure()
    # plt.imshow(mask.sum(axis=0))
    # plt.show()
    # plt.figure()
    # plt.plot(fsc)
    # plt.show()
    # import pdb; pdb.set_trace()

    if use_filter:
        ift_sum = filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, edgewidth_pix)
        # if jnp.isnan(ift_sum).any():
        #     import pdb; pdb.set_trace()
        return ift_sum, mask, fsc, local_resol

    return fsc, local_resol

batch_compute_local_fsc = jax.vmap(compute_local_fsc, in_axes = (0, None, None, None, None, None, None, None, None, None ) )

batch_compute_local_fsc_v2 = jax.vmap(compute_local_fsc_v2, in_axes = (0, None, None, None, None, None, None, None, None, None ) )


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

    # import pdb; pdb.set_trace()
    ires_interp = jnp.where(ires_zero > 0, ires_interp, 0)
    ires_interp = jnp.where((ires_zero == fsc_curve.size-1) * fsc_curve[-1] >= threshold, fsc_curve.size-1, ires_interp)

    # # ires_interp = jnp.interp(threshold, fsc_subcurve, jnp.array([1,0]) * 1.0 )

    # plt.plot(fsc_subcurve, [1,0]); plt.plot(1/7, ires_interp, 'o'); plt.show()
    # plt.show()

    # plt.plot(fsc_curve); plt.plot(ires_interp, 1/7, 'o'); plt.show()

    # # import matplotlib.pyplot as plt

    # import pdb; pdb.set_trace()
    # ires_interp = 
    # jnp.where(ires_zero > 0 )
    # if ires_zero == fsc_curve.size - 1:
    #     return 999
    # from scipy import interpolate
    # fsc_curve = np.where(np.isnan(fsc_curve), 0, fsc_curve)
    # f = interpolate.interp1d( np.array([fsc_curve[idx], fsc_curve[idx+1]]), np.array([freq[idx], freq[idx+1]]) )
    return ires_interp


def apply_fsc_weighting(FT, fsc):
	#  Find resolution where fsc_true drops below zero for the first time
	#  Set all weights to zero beyond that resolution
    distances = ftu.get_grid_of_radial_distances(FT.shape,)

    # ires_max = jnp.argmin(fsc >= 0.0001)
    ires_max = find_first_zero_in_bool(fsc >= 0.0001)
    # ires_max = jnp.where(all_ones_flag, fsc.size, ires_max) # If == 0, fsc >= 0.0001 for all ires

    # import pdb; pdb.set_trace()
    fsc = jnp.where( jnp.arange(fsc.size) < ires_max, fsc, 0)
    # import pdb; pdb.set_trace()
    # fsc = fsc.at[ires_max:].set(0)
    fsc = jnp.sqrt((2 * fsc) / (1 + fsc))
    fsc_mask = fsc[distances]
    FT = FT * fsc_mask
    return FT


def filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, edgewidth_pix):

    ft_sum  = apply_fsc_weighting(ft_sum, fsc) # 
    ft_sum = low_pass_filter_map(ft_sum, ft_sum.shape[0], local_resol, voxel_size, edgewidth_pix)
    ift_sum = ftu.get_idft3(ft_sum).real
    return  ift_sum

def low_pass_filter_map(FT, ori_size, low_pass, angpix, filter_edge_width, do_highpass_instead = False):
    ires_filter = jnp.round((ori_size * angpix) / low_pass)
    filter_edge_halfwidth = filter_edge_width // 2

    edge_low = jnp.maximum(0., (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = jnp.minimum(FT.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low
    res = ftu.get_grid_of_radial_distances(FT.shape) / ori_size

    if do_highpass_instead:
        filter = jnp.where(res <  edge_low , 0, 1)
        filter = jnp.where(res >= edge_low * (res < edge_high), 0.5 - 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    else:
        filter = jnp.where(res <  edge_low , 1, 0)
        filter = jnp.where(res >= edge_low * (res < edge_high), 0.5 + 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    return FT * filter


def local_error(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size

    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)

    # mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)

    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)
    mask /= np.linalg.norm(mask)
    # Compute error with convolution
    
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map1 = ftu.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map2 = ftu.get_idft3(map2_ft).real

    mask_ft = ftu.get_dft3(mask)
    map1_square_ft = ftu.get_dft3(map1*map1)
    map2_square_ft = ftu.get_dft3(map2*map2)
    map1map2_ft = ftu.get_dft3(map1*map2)
    # map2_ft = ftu.get_dft3(map2)

    local_errors = (ftu.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * ftu.get_idft3(map1map2_ft * mask_ft) \
    + (ftu.get_idft3(map2_square_ft * mask_ft) )
    
    # local_errors = (ftu.get_idft3(map1_ft * mask_ft).real )**2 \
    # - 2 * ftu.get_idft3(map1_ft * map2_ft * mask_ft) \
    # + (ftu.get_idft3(map2_ft * mask_ft) )**2
    
    local_errors = local_errors.real

    return local_errors


def local_error_with_cov(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None, noise_variance = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size

    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)


    # TODO FIX this +/- 1 business somewhere once and for all...
    # mask = mask_fn.create_hard_edged_kernel_pxl(locres_maskrad, map1.shape)
    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)


    # Compute error with convolution
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map1 = ftu.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map2 = ftu.get_idft3(map2_ft).real

    ## Whiten maps
    if noise_variance is not None:
        noise_variance = noise_variance.reshape(map1.shape)
        map1 = ftu.get_idft3(ftu.get_dft3(map1) * jnp.sqrt(noise_variance).reshape(map1.shape)).real
        map2 = ftu.get_idft3(ftu.get_dft3(map2) * jnp.sqrt(noise_variance).reshape(map1.shape)).real

    mask_ft = ftu.get_dft3(mask)
    map1_square_ft = ftu.get_dft3(map1*map1)
    map2_square_ft = ftu.get_dft3(map2*map2)
    map1map2_ft = ftu.get_dft3(map1*map2)
    # map2_ft = ftu.get_dft3(map2)

    local_errors = (ftu.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * ftu.get_idft3(map1map2_ft * mask_ft) \
    + (ftu.get_idft3(map2_square_ft * mask_ft) )
        
    local_errors = local_errors.real

    return local_errors




def expensive_local_error_with_cov(map1, map2, voxel_size, noise_variance, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None):

    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size


    step_size = np.round(locres_sampling / angpix).astype(int)
    maskrad_pix = np.round(locres_maskrad / angpix).astype(int)
    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)
    logger.info(f"Step size: {step_size}, maskrad_pix: {maskrad_pix}, edgewidth_pix: {edgewidth_pix}")
    myrad = map1.shape[0]//2 - maskrad_pix
    # myrad = 40
    myradf = myrad / step_size
    sampling_points = []

    logger.info(f"Starting...")

    grid = np.array(ftu.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)

    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    logger.info(f"Calculating local error in {nr_samplings} sampling points ...")
    

    # for now will do batch of 1.
    i_fil = 0
    i_loc_res = 0
    i_loc_auc = 0
    i_sum_w = 0 
    diffs, fscs = [], []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)
    noise_variance = jnp.asarray(noise_variance)
    single_batch = True

    if np.log2(map1.shape[0]) % 1 != 0:
        raise ValueError("Map size must be a power of 2")
    
    sqrt_noise_variance_real = ftu.get_idft3(jnp.sqrt(noise_variance))
    radius = maskrad_pix + edgewidth_pix
    multiplier = 2
    want_size = 2*(multiplier*radius)
    actual_size = int(2**(np.ceil(np.log2(want_size))))
    factor = map1.shape[0]//actual_size 
    from skimage.transform import downscale_local_mean

    # downsampled_noise_variance = downscale_local_mean(np.array(noise_variance), (actual_size, actual_size, actual_size) )
    downsampled_noise_variance = noise_variance[::factor,::factor,::factor]
    newshape = downsampled_noise_variance.shape[0]
    # print(downsampled_noise_variance[newshape//2, newshape//2, newshape//2])
    # print(noise_variance[noise_variance.shape[0]//2, noise_variance.shape[0]//2, noise_variance.shape[0]//2])

    # import pdb; pdb.set_trace()
    # # Downsampled noise variance
    # from skimage.transform import downscale_local_mean
    # downsampled_noise_variance = downscale_local_mean(noise_variance )
    # offset = offset + diff.shape[0]//2

    # factor = np.floor(map1.shape[0]/want_size) #* want_size
    # actual_size = map1.shape[0] / factor
    # import pdb; pdb.set_trace()
    # # But want map1.shape[0] to be a mutiple of want_size
    # actual_size = 

    # map1_sub = subsample_array(diff, offset, multiplier*radius)
    # use_v2 = True
    diff_map = jnp.array(map1- map2)
    single_batch = False
    if single_batch:
        for k in range(nr_samplings):
            batch = sampling_points[k]
            if use_v2:
                diff = masked_noisy_error2(map1 -  map2, downsampled_noise_variance, batch, maskrad_pix, edgewidth_pix )
            else:
                diff = masked_noisy_error(map1, map2, noise_variance, batch, maskrad_pix, edgewidth_pix )
            diffs.append(diff[None])
            # i_loc_res += loc_mask * diff
            # i_sum_w += loc_mask
            if k % 1000 ==0:
                print(k, end = '  ')

            # print(np.linalg.norm(diff - diff2))
            # import pdb; pdb.set_trace()


    else:
        # Doesn't seem to be faster.
        vol_batch_size = recovar.utils.get_vol_batch_size(map1.shape[0], recovar.utils.get_gpu_memory_total()) * 1
        n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)

        for k in range(n_batch):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
            batch = sampling_points[batch_st:batch_end]
            diff = batch_masked_noisy_error(diff_map, noise_variance, batch, maskrad_pix, edgewidth_pix )
            diffs.append(diff)
            # import pdb; pdb.set_trace()
            # i_loc_res += jnp.sum(loc_mask * diff[:,None,None,None], axis=0)
            # i_sum_w += jnp.sum(loc_mask, axis=0)
            if k % 10 ==0:
                print(k, end = '  ')


    diffs = np.concatenate(diffs)
    full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    i_local_error = make_auc_map(sampling_points, diffs, full_mask)

    # i_fil = jnp.where( i_sum_w > 0,  i_loc_res / i_sum_w, 0)

    return i_local_error



@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error(diff, noise_variance, offset, maskrad_pix, edgewidth_pix ):
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    diff = ftu.get_dft3((diff) * mask) * jnp.sqrt(noise_variance)
    return jnp.linalg.norm(diff)**2

# @functools.partial(jax.jit, static_argnums = [3,4])    
# def masked_noisy_error2(diff, sqrt_noise_variance_real, offset, maskrad_pix, edgewidth_pix ):

#     offset = offset + diff.shape[0]//2
#     radius = maskrad_pix + edgewidth_pix
#     multiplier = 2
#     map1_sub = subsample_array(diff, offset, multiplier*radius)


#     noise_variance_real = subsample_array(sqrt_noise_variance_real, offset, multiplier*radius)
#     mask = mask_fn.raised_cosine_mask(map1_sub.shape, maskrad_pix, radius, 0)
#     result = jnp.linalg.norm(map1_sub * mask * noise_variance_real )**2
#     import pdb; pdb.set_trace()
#     return result

@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error2(diff, downsampled_noise_variance, offset, maskrad_pix, edgewidth_pix ):

    offset = offset + diff.shape[0]//2
    radius = maskrad_pix + edgewidth_pix
    # multiplier = 2
    diff_sub = subsample_array(diff, offset, downsampled_noise_variance.shape[0]//2)
    # noise_variance_real = subsample_array(sqrt_noise_variance_real, offset, multiplier*radius)
    mask = mask_fn.raised_cosine_mask(diff_sub.shape, maskrad_pix, radius, 0)
    result = jnp.linalg.norm( jnp.sqrt(downsampled_noise_variance) * ftu.get_dft3(diff_sub * mask) )**2
    # import pdb; pdb.set_trace()
    return result


batch_masked_noisy_error = jax.vmap(masked_noisy_error, in_axes = (None,None,0, None, None) )
