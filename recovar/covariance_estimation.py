import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants, noise, homogeneous
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def compute_regularized_covariance_columns(cryos, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory, noise_model = "white", disc_type = 'linear_interp', radius = 5):

    cryo = cryos[0]
    volume_shape = cryos[0].volume_shape
    radius = constants.COLUMN_RADIUS if radius is None else radius
    if cryo.grid_size == 16:
        picked_frequencies = np.arange(cryo.volume_size) 
    else:
        picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = radius, use_half = True))


    # These options should probably be left as is.
    mask_ls = dilated_volume_mask
    mask_final = volume_mask
    substract_shell_mean = False 
    shift_fsc = False
    keep_intermediate = False
    image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)

    utils.report_memory_device(logger = logger)
    disc_type = 'nearest'
    Hs, Bs = compute_both_H_B(cryos, means["combined"], mask_ls, picked_frequencies, gpu_memory, image_noise_var, disc_type, parallel_analysis = False, H_B_fn = "noisemask")
    st_time = time.time()
    volume_noise_var = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].volume_shape))

    H_comb, B_comb, prior, fscs = compute_covariance_regularization(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, prior_iterations = 3, keep_intermediate = keep_intermediate, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, substract_shell_mean = substract_shell_mean, shift_fsc = shift_fsc)
    del Hs, Bs

    H_comb = np.stack(H_comb).astype(dtype = cryo.dtype)
    B_comb = np.stack(B_comb).astype(dtype = cryo.dtype)

    st_time2 = time.time()
    covariance_cols = {}
    cols2 = []
    for col_idx in range(picked_frequencies.size):
        cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx], prior[col_idx]) * valid_idx ))
        
    logger.info(f"cov update time: {time.time() - st_time2}")
    covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
    logger.info(f"reg time: {time.time() - st_time}")
    utils.report_memory_device(logger = logger)
    return covariance_cols, picked_frequencies, np.asarray(fscs)

def compute_regularized_covariance_columns_test(cryos, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory, noise_model = "white", disc_type = 'linear_interp', radius = 5):

    cryo = cryos[0]
    volume_shape = cryos[0].volume_shape
    radius = constants.COLUMN_RADIUS if radius is None else radius
    if cryo.grid_size == 16:
        picked_frequencies = np.arange(cryo.volume_size) 
    else:
        picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = radius, use_half = True))


    # These options should probably be left as is.
    mask_ls = dilated_volume_mask
    mask_final = volume_mask
    substract_shell_mean = False 
    shift_fsc = False
    keep_intermediate = False

    # image_batch_size = utils.get_image_batch_size(cryo.grid_size, )

    # if noise_model == "white":
    #     image_noise_var = cov_noise
    # elif noise_model == "radial":
    #     image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)
    # else:
    #     assert False, "wrong noise model"
    image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)

    utils.report_memory_device(logger = logger)
    disc_type = 'nearest'
    Hs, Bs = compute_both_H_B(cryos, means["combined"], mask_ls, picked_frequencies, gpu_memory, image_noise_var, disc_type, parallel_analysis = False, H_B_fn = "noisemask")
    st_time = time.time()

    ## DEL ALL THIS
    H_comb = Hs[0].T
    B_comb = Bs[0].T

    covariance_cols = {}
    cols2 = []
    for col_idx in range(picked_frequencies.size):
        # cols2.append(np.array(regularization.covariance_update_col_with_mask(H_comb[col_idx], B_comb[col_idx], prior[col_idx], volume_mask, valid_idx, volume_shape)))
        cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx],  B_comb[col_idx]*0 + np.inf) * valid_idx ))
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    
        
    # logger.info(f"cov update time: {time.time() - st_time2}")

    covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
    utils.pickle_dump(covariance_cols, "/home/mg6942/mytigress/synthetic_clean_white/cols_one_"+ disc_type+ "_unfixed.pkl")



    utils.report_memory_device(logger = logger)
    disc_type = 'nearest'
    Hs, Bs = compute_both_H_B(cryos, means["combined"], mask_ls, picked_frequencies, gpu_memory, image_noise_var, disc_type, parallel_analysis = False, H_B_fn = "fixed")
    st_time = time.time()

    ## DEL ALL THIS
    H_comb = Hs[0].T
    B_comb = Bs[0].T

    covariance_cols = {}
    cols2 = []
    for col_idx in range(picked_frequencies.size):
        # cols2.append(np.array(regularization.covariance_update_col_with_mask(H_comb[col_idx], B_comb[col_idx], prior[col_idx], volume_mask, valid_idx, volume_shape)))
        cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx],  B_comb[col_idx]*0 + np.inf) * valid_idx ))
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
    
        
    # logger.info(f"cov update time: {time.time() - st_time2}")

    covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
    utils.pickle_dump(covariance_cols, "/home/mg6942/mytigress/synthetic_clean_white/cols_one_"+ disc_type+ "_fixed.pkl")




    assert False
    ## DEL ALL THIS -- END


    utils.report_memory_device(logger = logger)


    # if noise_model == "white":
    #     volume_noise_var = cov_noise
    # elif noise_model == "radial":
    volume_noise_var = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].volume_shape))

    H_comb, B_comb, prior, fscs = compute_covariance_regularization(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, prior_iterations = 3, keep_intermediate = keep_intermediate, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, substract_shell_mean = substract_shell_mean, shift_fsc = shift_fsc)
    del Hs, Bs

    H_comb = np.stack(H_comb).astype(dtype = cryo.dtype)
    B_comb = np.stack(B_comb).astype(dtype = cryo.dtype)

    st_time2 = time.time()
    covariance_cols = {}
    cols2 = []
    for col_idx in range(picked_frequencies.size):
        # cols2.append(np.array(regularization.covariance_update_col_with_mask(H_comb[col_idx], B_comb[col_idx], prior[col_idx], volume_mask, valid_idx, volume_shape)))
        logger.warning("TOOK OUT PRIOR STUFF. PROBABLY ")
        cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx], prior[col_idx]*0 + np.inf) * valid_idx ))

        
    logger.info(f"cov update time: {time.time() - st_time2}")

    covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
    utils.pickle_dump(covariance_cols, "/home/mg6942/mytigress/synthetic_clean_white/cols.pkl")

    del H_comb, B_comb, prior
    logger.info(f"reg time: {time.time() - st_time}")
    utils.report_memory_device(logger = logger)
    return covariance_cols, picked_frequencies, np.asarray(fscs)



def compute_both_H_B(cryos, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis, H_B_fn ):
    Hs = []
    Bs = []
    st_time = time.time()
    for _, cryo in enumerate(cryos):
        H, B = compute_H_B_in_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis, H_B_fn = H_B_fn)
        logger.info(f"Time to cov {time.time() - st_time}")
        # check_memory()
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


# Covariance_cols
def compute_H_B_in_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis = False, H_B_fn = None):

    image_batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    column_batch_size = utils.get_column_batch_size(cryo.grid_size, gpu_memory)

    # if batch_over_image_only:
    #     return compute_H_B(cryo, mean, dilated_volume_mask,
    #                                                              picked_frequencies,
    #                                                              int(image_batch_size ), (cov_noise),
    #                                                              None , disc_type = disc_type,
    #                                                              parallel_analysis = parallel_analysis,
    #                                                              jax_random_key = 0, batch_over_H_B = True)
    

    H = np.empty( [cryo.volume_size, picked_frequencies.size] , dtype = cryo.dtype)
    B = np.empty( [cryo.volume_size, picked_frequencies.size] , dtype = cryo.dtype)
    # frequency_batch = int(25 * (256 / cryo.grid_size)**3 /2) 
    frequency_batch = column_batch_size
    for k in range(0, int(np.ceil(picked_frequencies.size/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min( [(k+1) * frequency_batch ,picked_frequencies.size  ]))
        logger.info(f'outside H_B : {batch_st}, {batch_end}')
        utils.report_memory_device(logger = logger)
        H_batch, B_batch = compute_H_B(cryo, mean, dilated_volume_mask,
                                                                 picked_frequencies[batch_st:batch_end],
                                                                 int(image_batch_size / 1), (cov_noise),
                                                                 None , disc_type = disc_type,
                                                                 parallel_analysis = parallel_analysis,
                                                                 jax_random_key = 0 , H_B_fn = H_B_fn)
        H[:, batch_st:batch_end]  = np.array(H_batch)
        B[:, batch_st:batch_end]  = np.array(B_batch)
        del H_batch, B_batch
        
    return H,B


    
def compute_covariance_regularization(Hs, Bs, mean_prior, picked_frequencies, cov_noise, volume_mask, volume_shape, gpu_memory,  prior_iterations = 3, keep_intermediate = False, reg_init_multiplier = 1, substract_shell_mean = False, shift_fsc = False):

    # 
    regularization_init = (mean_prior + 1e-14) * reg_init_multiplier / cov_noise
    def init_regularization_of_column_k(k):
        return regularization_init[None] * regularization_init[picked_frequencies[np.array(k)], None] 

    # This should probably be rewritten.
    for cryo_idx in range(len(Hs)):
        Hs[cryo_idx] = Hs[cryo_idx].T
        Bs[cryo_idx] = Bs[cryo_idx].T
    
    # Column-wise regularize 
    shifts = core.vec_indices_to_frequencies(picked_frequencies, volume_shape) * (shift_fsc)
    n_freqs = picked_frequencies.size

    fsc_priors = [None] * n_freqs
    H_combined = [None] * n_freqs
    B_combined = [None] * n_freqs
    fscs = [None] * n_freqs

    batch_size = utils.get_column_batch_size(volume_shape[0], gpu_memory) // 4

    for k in range(int(np.ceil(n_freqs/batch_size))-1, -1, -1):
        batch_st = int(k * batch_size)
        batch_end = int(np.min( [(k+1) * batch_size, n_freqs]))
        indices = jnp.arange(batch_st, batch_end)
        H0_batch = Hs[0][batch_st:batch_end]
        H1_batch = Hs[1][batch_st:batch_end]
        B0_batch = Bs[0][batch_st:batch_end]
        B1_batch = Bs[1][batch_st:batch_end]   

        priors, fscs_this = regularization.prior_iteration_batch(H0_batch, H1_batch, B0_batch, B1_batch, shifts[indices], init_regularization_of_column_k(np.array(indices)), substract_shell_mean, volume_shape, prior_iterations )
        cpus = jax.devices("cpu")
        priors = jax.device_put(priors, cpus[0])
        for k,ind in enumerate(indices):
            fsc_priors[ind] = priors[k].real
            H_combined[ind] = H0_batch[k] + H1_batch[k]
            B_combined[ind] = B0_batch[k] + B1_batch[k]
            fscs[ind] = fscs_this[k]

    fsc_priors = np.stack(fsc_priors, axis =0).real
    # Symmetricize prior    
    fsc_priors[:,picked_frequencies] = 0.5 * ( fsc_priors[:,picked_frequencies] + fsc_priors[:,picked_frequencies].T ) 
    return H_combined, B_combined, fsc_priors, fscs

    
    

# @functools.partial(jax.jit, static_argnums = [6])    
def compute_H_B_inner(centered_images, ones_mapped, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked, cov_noise, picked_freq_index, volume_size):

    mask = plane_indices_on_grid_stacked == picked_freq_index
    v = centered_images * jnp.conj(CTF_val_on_grid_stacked)  

    ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 

    # C_n
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])
    C_n = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked) 

    # E_n
    delta_at_freq = jnp.zeros(volume_size, dtype = centered_images.dtype )
    delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    delta_at_freq_mapped = core.forward_model(delta_at_freq, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked) 

    # I can't remember why I decided this wasn't a good idea.
    # Apply mask
    # delta_at_freq_mapped = apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)

    delta_at_freq_mapped *= cov_noise

    # Apply mask again conjugate == apply image mask since mask is real?
    # delta_at_freq_mapped = apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)
    
    delta_at_freq_mapped = delta_at_freq_mapped  * jnp.conj(CTF_val_on_grid_stacked)
    E_n = core.summed_adjoint_projections_nearest(volume_size, delta_at_freq_mapped, plane_indices_on_grid_stacked)
    B_freq_idx = C_n - E_n

    # H
    v = ones_mapped * jnp.conj(CTF_val_on_grid_stacked)  
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])

    # Pick out only columns j for which V[idx,j] is not zero
    H_freq_idx = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked)
    
    # H_zeros = H_freq_idx == 0 
    return H_freq_idx, B_freq_idx

batch_compute_H_B_inner = jax.vmap(compute_H_B_inner, in_axes = (None, None, None, None, None, 0, None))




# Probably should delete the other version after debugging.
# @functools.partial(jax.jit, static_argnums = [6])    
def compute_H_B_inner_mask(centered_images, ones_mapped, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked, cov_noise, picked_freq_index, image_mask, image_shape, volume_size):

    mask = plane_indices_on_grid_stacked == picked_freq_index


    v = centered_images * jnp.conj(CTF_val_on_grid_stacked)

    ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 

    # C_n
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])
    C_n = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked) 

    # E_n
    delta_at_freq = jnp.zeros(volume_size, dtype = centered_images.dtype )
    delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    delta_at_freq_mapped = core.forward_model(delta_at_freq, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked) 
    
    # delta_at_freq = jnp.zeros(volume_shape, dtype = CTF_params.dtype )
    # delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    # delta_at_freq_mapped = core.forward_model_from_map(delta_at_freq, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)

    # Apply mask
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)

    delta_at_freq_mapped *= cov_noise

    # Apply mask again conjugate == apply image mask since mask is real?
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)
    
    delta_at_freq_mapped = delta_at_freq_mapped  * jnp.conj(CTF_val_on_grid_stacked)
    E_n = core.summed_adjoint_projections_nearest(volume_size, delta_at_freq_mapped, plane_indices_on_grid_stacked)
    B_freq_idx = C_n - E_n



    # H
    v = ones_mapped * jnp.conj(CTF_val_on_grid_stacked)
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])

    # Pick out only columns j for which V[idx,j] is not zero
    H_freq_idx = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked)
    # import pdb; pdb.set_trace()
    # H_zeros = H_freq_idx == 0 
    return H_freq_idx, B_freq_idx




def zero_except_in_index(size, index, dtype = jnp.float32):
    return jnp.zeros(size, dtype = dtype).at[index].set(1)

batch_zero_except_in_index = jax.vmap(zero_except_in_index, in_axes = (None, 0, None))

# Probably should delete the other version after debugging.
# @functools.partial(jax.jit, static_argnums = [6])    
def compute_H_B_inner_mask_fixed(centered_images, ones_mapped, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked, cov_noise, picked_freq_index, image_mask, image_shape, volume_size):

    image_size = np.prod(image_shape)
    # mask = plane_indices_on_grid_stacked == picked_freq_index    
    
    # good_indices = jnp.max(plane_indices_on_grid_stacked == picked_freq_index, axis = -1) > 0
    distances, indices = jax.lax.top_k((plane_indices_on_grid_stacked == picked_freq_index).astype(int), 1)#, axis = -1)
    mask = batch_zero_except_in_index(image_size, indices, int)
    mask *= distances > 0

    # plane_indices_on_grid_stacked *= mask

    ###
    centered_images = centered_images*mask
    ones_mapped = ones_mapped*mask

    # import pdb; pdb.set_trace();
    # if jnp.sum(mask, axis=0) 
    # print(np.sum())


    v = centered_images * jnp.conj(CTF_val_on_grid_stacked)

    ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 

    # C_n
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])
    C_n = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked) 

    # E_n
    delta_at_freq = jnp.zeros(volume_size, dtype = centered_images.dtype )
    delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    delta_at_freq_mapped = core.forward_model(delta_at_freq, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked) 
    
    ###
    delta_at_freq_mapped = delta_at_freq_mapped*mask


    # delta_at_freq = jnp.zeros(volume_shape, dtype = CTF_params.dtype )
    # delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    # delta_at_freq_mapped = core.forward_model_from_map(delta_at_freq, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)

    # Apply mask
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)

    delta_at_freq_mapped *= cov_noise

    # Apply mask again conjugate == apply image mask since mask is real?
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)
    
    delta_at_freq_mapped = delta_at_freq_mapped  * jnp.conj(CTF_val_on_grid_stacked)
    E_n = core.summed_adjoint_projections_nearest(volume_size, delta_at_freq_mapped, plane_indices_on_grid_stacked)
    B_freq_idx = C_n - E_n



    # H
    v = ones_mapped * jnp.conj(CTF_val_on_grid_stacked)
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])

    # Pick out only columns j for which V[idx,j] is not zero
    H_freq_idx = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked)
    # import pdb; pdb.set_trace()
    # H_zeros = H_freq_idx == 0 
    return H_freq_idx, B_freq_idx

# Probably should delete the other version after debugging.
# @functools.partial(jax.jit, static_argnums = [6])    
def compute_H_B_inner_mask_new(centered_images, image_mask, noise_variance, picked_freq_index, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type_H, disc_type_B):

    volume_size = np.prod(volume_shape)
    # This has a lot of repeated computation, e.g., the CTF is recomputed probably 6 times in here, and the grid points as well.

    delta_at_freq = jnp.zeros(volume_size, dtype = CTF_params.dtype )
    delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 

    
    # delta_at_freq_mapped = core.forward_model_from_map(delta_at_freq, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
 
    # forward_model_from_map_and_return_adjoint
    delta_at_freq_mapped, f_adjoint = core.forward_model_from_map_and_return_adjoint(delta_at_freq.astype(centered_images.dtype), CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type_B)


    inner_products = jnp.sum( jnp.conj(centered_images)* delta_at_freq_mapped, axis =-1, keepdims= True)
    centered_images = centered_images *  inner_products

    # C_n
    # summed_Pi = core.adjoint_forward_model_from_map(centered_images, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
    summed_Pi = f_adjoint(centered_images)[0]

    # E_n
    
    # Apply mask
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)

    delta_at_freq_mapped *= noise_variance

    # Apply mask again conjugate == apply image mask since mask is real?
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)
    
    # summed_En = core.adjoint_forward_model_from_map(delta_at_freq_mapped, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
    summed_En = f_adjoint(delta_at_freq_mapped)[0]


    B_k = summed_Pi - summed_En
    H_k = core.compute_covariance_column( CTF_params, rotation_matrices, picked_freq_index, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type_H)
    
    return H_k, B_k



# @functools.partial(jax.jit, static_argnums = [5])    
def compute_H_B(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, cov_noise, diag_prior, disc_type, parallel_analysis = False, jax_random_key = 0, batch_over_H_B = False, soften_mask = 3, H_B_fn = True ):
    # Memory in here scales as O (batch_size )

    utils.report_memory_device()

    volume_size = mean_estimate.size
    n_picked_indices = picked_frequency_indices.size
    H = [0] * n_picked_indices
    B = [0] * n_picked_indices
    jax_random_key = jax.random.PRNGKey(jax_random_key)
    mean_estimate = jnp.array(mean_estimate)

    print('noise in HB:', cov_noise)

    if (disc_type == 'nearest') or (disc_type== 'linear_interp'):
        disc_type_H = disc_type
        disc_type_B = disc_type
    elif disc_type == 'mixed':
        disc_type_H = 'nearest'
        disc_type_B = 'linear_interp'


    use_new_funcs = False
    apply_noise_mask = True
    # fixed = True
    if apply_noise_mask:
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
    else:
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')


    if H_B_fn =="newfuncs":
        # This is about 6x slower than the other version when using 'linear_interp'. It probably could be much faster in that case.
        # It is about 1.2x slower for 'nearest'.
        f_jit = jax.jit(compute_H_B_inner_mask_new, static_argnums = [6,7,8,9,10,11,12])
    elif H_B_fn =="fixed":
        f_jit = jax.jit(compute_H_B_inner_mask_fixed, static_argnums = [7,8])
    elif H_B_fn =="noisemask":
        f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
    else:
        f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])
        logger.warning("USING NO NOSIE MASK AND UNFIXED")

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for images, batch_image_ind in data_generator:
        
        these_disc = 'linear_interp'
        # mean_estimate*=0 
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              experiment_dataset.rotation_matrices[batch_image_ind], 
                                              experiment_dataset.image_stack.mask, 
                                              experiment_dataset.volume_mask_threshold, 
                                              experiment_dataset.image_shape, 
                                              experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                            experiment_dataset.padding, these_disc, soften = soften_mask ) #* 0 + 1
        # logger.warning('MASK IS OFF!!')
        # logger.warning('MASK IS OFF!!')
        # logger.warning('MASK IS OFF!!')
        # logger.warning('MASK IS OFF!!')
        # disc_type = 'linear_interp'
        # logger.warning('CHANGING DISC!!')

        images = experiment_dataset.image_stack.process_images(images)
        images = covariance_core.get_centered_images(images, mean_estimate,
                                     experiment_dataset.CTF_params[batch_image_ind],
                                     experiment_dataset.rotation_matrices[batch_image_ind],
                                     experiment_dataset.translations[batch_image_ind],
                                     experiment_dataset.image_shape, 
                                     experiment_dataset.volume_shape,
                                     experiment_dataset.grid_size, 
                                     experiment_dataset.voxel_size,
                                     experiment_dataset.CTF_fun,
                                     these_disc )
                
        images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  

        # images*=0

        batch_CTF = experiment_dataset.CTF_fun( experiment_dataset.CTF_params[batch_image_ind],
                                               experiment_dataset.image_shape,
                                               experiment_dataset.voxel_size)
        batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(
            experiment_dataset.rotation_matrices[batch_image_ind],
            experiment_dataset.image_shape, experiment_dataset.volume_shape, 
            experiment_dataset.grid_size )
        all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
        ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)
        

        # if use_new_funcs:
        #     f_jit = jax.jit(compute_H_B_inner_mask_new, static_argnums = [6,7,8,9,10,11])
        # elif apply_noise_mask:
        #     # logger.warning('XXXXX')
        #     # logger.warning('XXXX')
        #     f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
        # else:
        #     f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])


        for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
            
            if (k % 50 == 49) and (k > 0):
                # print( k, " cols comp.")
                f_jit._clear_cache() # Maybe this?

            if H_B_fn =="newfuncs":
                H_k, B_k = f_jit(images, image_mask, cov_noise, picked_freq_idx, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, disc_type_H, disc_type_B)
            elif (H_B_fn =="fixed") or (H_B_fn =="noisemask"):
                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size)
            else:
                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, volume_size)

            # if jnp.linalg.norm(H_k2*valid_idx)/ jnp.linalg.norm(B_k*valid_idx) < 1e5  

            # H_k3, B_k3 = f_jit(images, image_mask, cov_noise, picked_freq_idx, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, 'nearest', 'nearest')

            # H_k2, B_k2 = f_jit(images, image_mask, cov_noise, picked_freq_idx, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, 'linear_interp', 'linear_interp')
            # import pdb; pdb.set_trace()

            _cpu = jax.devices("cpu")[0]

            if batch_over_H_B:
                # Send to cpu.
                H[k] += jax.device_put(H_k, _cpu)
                B[k] += jax.device_put(B_k, _cpu)
                del H_k, B_k
            else:
                H[k] += H_k.real.astype(experiment_dataset.dtype_real)
                B[k] += B_k
        del image_mask
        del images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images
        
    H = np.stack(H, axis =1)#, dtype=H[0].dtype)
    B = np.stack(B, axis =1)#, dtype=B[0].dtype)
    return H, B



# @functools.partial(jax.jit, static_argnums = [5])    
def compute_H_B_tests(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, cov_noise, diag_prior, disc_type, parallel_analysis = False, jax_random_key = 0, batch_over_H_B = False, soften_mask = 3 ):
    # Memory in here scales as O (batch_size )

    use_new_funcs = True
    apply_noise_mask = True
    if apply_noise_mask:
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
        logger.warning('USING NOISE MASK IS ON')
    else:
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')
        logger.warning('USING NOISE MASK IS OFF')

    if (disc_type == 'nearest') or (disc_type== 'linear_interp'):
        disc_type_H = disc_type
        disc_type_B = disc_type
    elif disc_type == 'mixed':
        disc_type_H = 'nearest'
        disc_type_B = 'linear_interp'

    if use_new_funcs:
        # This is about 6x slower than the other version when using 'linear_interp'. It probably could be much faster in that case.
        # It is about 1.2x slower for 'nearest'.
        f_jit = jax.jit(compute_H_B_inner_mask_new, static_argnums = [6,7,8,9,10,11,12])
    elif apply_noise_mask:
        # logger.warning('XXXXX')
        # logger.warning('XXXX')
        f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
    else:
        f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])



    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for images, batch_image_ind in data_generator:

        # mean_estimate*=0 
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              experiment_dataset.rotation_matrices[batch_image_ind], 
                                              experiment_dataset.image_stack.mask, 
                                              experiment_dataset.volume_mask_threshold, 
                                              experiment_dataset.image_shape, 
                                              experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                            experiment_dataset.padding, disc_type, soften = soften_mask ) 
        logger.warning('MASK IS OFF!!')
        logger.warning('MASK IS OFF!!')
        logger.warning('MASK IS OFF!!')
        logger.warning('MASK IS OFF!!')
        disc_type = 'nearest'
        logger.warning('CHANGING DISC!!')

        images = experiment_dataset.image_stack.process_images(images)
        images = covariance_core.get_centered_images(images, mean_estimate,
                                     experiment_dataset.CTF_params[batch_image_ind],
                                     experiment_dataset.rotation_matrices[batch_image_ind],
                                     experiment_dataset.translations[batch_image_ind],
                                     experiment_dataset.image_shape, 
                                     experiment_dataset.volume_shape,
                                     experiment_dataset.grid_size, 
                                     experiment_dataset.voxel_size,
                                     experiment_dataset.CTF_fun,
                                     disc_type )
                
        if parallel_analysis:
            jax_random_key, subkey = jax.random.split(jax_random_key)
            images *= (np.random.randint(0, 2, images.shape)*2 - 1)
            # images *=  np.exp(1j* np.random.rand(*(images.shape)) * 2 * np.pi) 
        # images3 = covariance_core.apply_image_masks(images2, image_mask, experiment_dataset.image_shape)  

        images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  

        # images*=0

        batch_CTF = experiment_dataset.CTF_fun( experiment_dataset.CTF_params[batch_image_ind],
                                               experiment_dataset.image_shape,
                                               experiment_dataset.voxel_size)
        batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(
            experiment_dataset.rotation_matrices[batch_image_ind],
            experiment_dataset.image_shape, experiment_dataset.volume_shape, 
            experiment_dataset.grid_size )
        all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
        ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)
        
        if apply_noise_mask:
            # logger.warning('XXXXX')
            # logger.warning('XXXX')
            f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
        else:
            f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])

        for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
            
            if (k % 50 == 49) and (k > 0):
                # print( k, " cols comp.")
                f_jit._clear_cache() # Maybe this?
                
            if apply_noise_mask:
                ### CHANGE NOISE ESTIMATE HERE?
                # logger.warning('XXXXX')
                # logger.warning('XXXX')
                # H_k, B_k =  f_jit(3*ones_mapped, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise*0, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size)

                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size)
            else:
                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, volume_size)


            # H_k2 = core.compute_covariance_column( experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], picked_freq_idx, experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, disc_type)
            valid_idx = experiment_dataset.get_valid_frequency_indices(rad = experiment_dataset.grid_size//2-2)

            delta_at_freq = jnp.zeros(volume_size, dtype = images.dtype )
            delta_at_freq = delta_at_freq.at[picked_freq_idx].set(1) 
            delta_at_freq_mapped = core.forward_model(delta_at_freq, batch_CTF, batch_grid_pt_vec_ind_of_images) 

            delta_at_freq_mapped2 = core.forward_model_from_map(delta_at_freq, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, disc_type)
            dist3 = delta_at_freq_mapped - delta_at_freq_mapped2


            # mask = plane_indices_on_grid_stacked == picked_freq_index
            # v = centered_images * jnp.conj(CTF_val_on_grid_stacked)
            # ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 
            ## Two adjoints:
            # C_n
            w = images * jnp.conj(batch_CTF)
            adj_images = core.summed_adjoint_projections_nearest(volume_size, w, batch_grid_pt_vec_ind_of_images) 

            adj_images2 = core.adjoint_forward_model_from_map(images, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, disc_type)
            diff4 = adj_images- adj_images2
            dist4 = np.linalg.norm(diff4*valid_idx)
            # delta_at_freq_mapped = core.forward_model_from_map(delta_at_freq, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked) 


            H_k2, B_k2 = compute_H_B_inner_mask_new(images, image_mask, cov_noise, picked_freq_idx, experiment_dataset.CTF_params[batch_image_ind], experiment_dataset.rotation_matrices[batch_image_ind], experiment_dataset.image_shape, experiment_dataset.volume_shape, experiment_dataset.grid_size, experiment_dataset.voxel_size, experiment_dataset.CTF_fun, disc_type)


            dist = jnp.linalg.norm((H_k - H_k2)*valid_idx) /jnp.linalg.norm((H_k)*valid_idx) 
            dist2 = jnp.linalg.norm((B_k - B_k2)*valid_idx) /jnp.linalg.norm((B_k)*valid_idx) 
            dist2_unnormal = jnp.linalg.norm((B_k - B_k2)*valid_idx) 

            if dist > 1e-6:
                import pdb; pdb.set_trace()


            if dist2 > 1e-6 and dist2_unnormal> 1e-6:
                import pdb; pdb.set_trace()

            # if jnp.linalg.norm(H_k2*valid_idx)/ jnp.linalg.norm(B_k*valid_idx) < 1e5  


            _cpu = jax.devices("cpu")[0]

            if batch_over_H_B:
                # Send to cpu.
                H[k] += jax.device_put(H_k, _cpu)
                B[k] += jax.device_put(B_k, _cpu)
                del H_k, B_k
            else:
                H[k] += H_k.real.astype(experiment_dataset.dtype_real)
                B[k] += B_k
        del image_mask
        del images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images
        
    H = np.stack(H, axis =1)#, dtype=H[0].dtype)
    B = np.stack(B, axis =1)#, dtype=B[0].dtype)
    return H, B




## REDUCE COVARIANCE COMPUTATION
# @functools.partial(jax.jit, static_argnums = [5])    
def compute_projected_covariance(experiment_datasets, mean_estimate, basis, volume_mask, noise_variance, batch_size, disc_type, parallel_analysis = False ):
    
    experiment_dataset = experiment_datasets[0]

    basis = basis.T.astype(experiment_dataset.dtype)
    # Make sure variables used in every iteration are on gpu.
    basis = jnp.asarray(basis)
    volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    mean_estimate = jnp.array(mean_estimate).astype(experiment_dataset.dtype)
    # eigenvalues = jnp.array(eigenvalues).astype(experiment_dataset.dtype)
    # contrast_grid = contrast_grid.astype(experiment_dataset.dtype_real)
    no_mask = covariance_core.check_mask(volume_mask)    
    basis_size = basis.shape[0]
    jax_random_key = jax.random.PRNGKey(0)
    lhs =0
    rhs =0 

    for experiment_dataset in experiment_datasets:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
        

        for batch, batch_image_ind in data_generator:
            
            lhs_this, rhs_this = reduce_covariance_est_inner(batch, mean_estimate, volume_mask, 
                                                                            basis,
                                                                            experiment_dataset.CTF_params[batch_image_ind],
                                                                            experiment_dataset.rotation_matrices[batch_image_ind],
                                                                            experiment_dataset.translations[batch_image_ind],
                                                                            experiment_dataset.image_stack.mask,
                                                                            experiment_dataset.volume_mask_threshold,
                                                                            experiment_dataset.image_shape, 
                                                                            experiment_dataset.volume_shape, 
                                                                            experiment_dataset.grid_size, 
                                                                            experiment_dataset.voxel_size, 
                                                                            experiment_dataset.padding, 
                                                                            disc_type, np.array(noise_variance),
                                                                            experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun, parallel_analysis = parallel_analysis,
                                                                        jax_random_key =jax_random_key)

            lhs +=lhs_this
            rhs +=rhs_this
        del lhs_this, rhs_this
    del basis
    # Deallocate some memory?

    # Solve dense least squares?
    def vec(X):
        return X.T.reshape(-1)

    ## Inverse of vec function.
    def unvec(x):
        n = np.sqrt(x.size).astype(int)
        return x.reshape(n,n).T

    # import pdb; pdb.set_trace()

    rhs = vec(rhs)
    covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    covar = unvec(covar)

    return covar


@functools.partial(jax.jit, static_argnums = [8,9,10,11,12,13,14,16,17, 18])    
def reduce_covariance_est_inner(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, noise_variance, process_fn, CTF_fun, parallel_analysis = False, jax_random_key = None):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type ) * 0 + 1
    logger.warning("Not using mask in reduce_covariance_est_inner")

    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    projected_mean = core.get_projected_image(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                           
                                          )


    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
    projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    AUs = covariance_core.batch_over_vol_get_projected_image(basis,
                                         CTF_params, 
                                         rotation_matrices,
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type )    
    # Apply mask on operator
    AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)

    batch = batch - projected_mean

    # if parallel_analysis:
    #     jax_random_key, subkey = jax.random.split(jax_random_key)
    #     batch *= (np.random.randint(0, 2, batch.shape)*2 - 1)
    #     print("here")
    # import pdb; pdb.set_trace()
    AU_t_images = batch_x_T_y(AUs, batch)#.block_until_ready()

    outer_products = summed_outer_products(AU_t_images)

    AU_t_AU = batch_x_T_y(AUs,AUs).real.astype(CTF_params.dtype)
    AUs *= jnp.sqrt(noise_variance)[...,None]
    UALambdaAUs = jnp.sum(batch_x_T_y(AUs,AUs), axis=0)

    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(CTF_params.dtype)
    lhs = jnp.sum(batch_kron(AU_t_AU, AU_t_AU), axis=0)
    return lhs, rhs
    
batch_kron = jax.vmap(jnp.kron, in_axes=(0,0))
batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))

def summed_outer_products(AU_t_images):
    # Not .H because things are already transposed technically
    return AU_t_images.T @ jnp.conj(AU_t_images)

batched_summed_outer_products  = jax.vmap(summed_outer_products)



## Mean functions
def compute_covariance_discretization_weights(experiment_dataset, prior, batch_size,  picked_freq_index, order=1 ):
    order = 1   # Only implemented for order 1 but could generalize
    rr_size = 6*order + 1
    RR = jnp.zeros((experiment_dataset.volume_size, (rr_size)**2 ))
    # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()) * 3

    for i in range(utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)):
        batch_st, batch_end = utils.get_batch_of_indices(experiment_dataset.n_images, batch_size, i)
        # Make sure mean_estimate is size # volume_size ?
        RR_this = compute_covariance_discretization_weights_inner(experiment_dataset.rotation_matrices[batch_st:batch_end], experiment_dataset.CTF_params[batch_st:batch_end], experiment_dataset.voxel_size, experiment_dataset.volume_shape, experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, picked_freq_index)
        RR += RR_this

    RR = RR.reshape([experiment_dataset.volume_size, rr_size, rr_size])
    weights, good_weights = homogeneous.batch_solve_for_weights(RR, prior)
    # If bad weights, just do Weiner filtering with 0th order disc
    # good_weights = (good_weights*0).astype(bool)

    other_weights = jnp.zeros_like(weights)
    # weiner_weights = jnp.where(RR[:,0,0] > 0, 1/RR[:,0,0], jnp.zeros_like(RR[:,0,0]))
    weiner_weights = 1 / (RR[...,0,0] + prior)#jnp.where(RR[:,0,0] > 0, 1/RR[:,0,0], jnp.zeros_like(RR[:,0,0]))
    other_weights = other_weights.at[...,0].set(weiner_weights)

    # weights = jnp.where(good_weights, weights, other_weights)
    weights = weights.at[~good_weights].set(other_weights[~good_weights])
    return weights, good_weights, RR

def compute_covariance_discretization_weights_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, picked_freq_index ):

    volume_size = np.prod(np.array(volume_shape))
    C_mat, grid_point_vec_indices = make_C_mat_covariance(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun,picked_freq_index)
    # This is going to be stroed twice as much stuff as it needs to be
    C_mat_outer = homogeneous.batch_batch_outer(C_mat).reshape([C_mat.shape[0], C_mat.shape[1], C_mat.shape[2]*C_mat.shape[2]])#.transpose([2,0,1])
    RR = core.batch_over_vol_summed_adjoint_projections_nearest(volume_size, C_mat_outer, grid_point_vec_indices)
    # mean_rhs = batch_over_weights_sum_adj_forward_model(volume_size, corrected_images , CTF, grid_point_indices)
    return RR


def make_C_mat_covariance(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, picked_freq_index):

    ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 

    # C_n
    # mult = jnp.sum(v * mask, axis = -1)
    # w = v * jnp.conj(mult[:,None])
    # C_n = core.summed_adjoint_projections_nearest(volume_size, w, plane_indices_on_grid_stacked) 

    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )

    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    grid_points_coords_nearest = core.round_to_int(grid_points_coords)
    differences = grid_points_coords - grid_points_coords_nearest
    mask = grid_point_vec_indices == picked_freq_index

    # found_indices = jnp.sum(mask, axis =0) > 1
    differences_picked_freq = differences[mask].repeat(differences.shape[1], axis =1)

    # Discretized grid points
    # This could be done more efficiently

    C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_picked_freq], axis = -1)

    CTF1 = CTF_fun( CTF_params, image_shape, voxel_size)
    CTF1_CTF2 = CTF1 * CTF1[mask]

    C_mat *= CTF1_CTF2[...,None]
    return C_mat, grid_point_vec_indices
