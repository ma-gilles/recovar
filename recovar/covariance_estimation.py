import logging
import jax.numpy as jnp
import numpy as np
import jax, time

from recovar import core, covariance_core, regularization, utils, constants, noise
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

    # image_batch_size = utils.get_image_batch_size(cryo.grid_size, )

    if noise_model == "white":
        image_noise_var = cov_noise
    elif noise_model == "radial":
        image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)

    utils.report_memory_device(logger = logger)

    Hs, Bs = compute_both_H_B(cryos, means["combined"], mask_ls, picked_frequencies, gpu_memory, image_noise_var, disc_type, parallel_analysis = False)
    st_time = time.time()

    utils.report_memory_device(logger = logger)


    if noise_model == "white":
        volume_noise_var = cov_noise
    elif noise_model == "radial":
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
        cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx], prior[col_idx]) * valid_idx ))

        
    logger.info(f"cov update time: {time.time() - st_time2}")

    covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
    del H_comb, B_comb, prior
    logger.info(f"reg time: {time.time() - st_time}")
    utils.report_memory_device(logger = logger)
    return covariance_cols, picked_frequencies, np.asarray(fscs)


def compute_both_H_B(cryos, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis ):
    Hs = []
    Bs = []
    st_time = time.time()
    for _, cryo in enumerate(cryos):
        H, B = compute_H_B_in_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis)
        logger.info(f"Time to cov {time.time() - st_time}")
        # check_memory()
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


# Covariance_cols
def compute_H_B_in_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, cov_noise, disc_type, parallel_analysis = False, batch_over_image_only = False):

    image_batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    column_batch_size = utils.get_column_batch_size(cryo.grid_size, gpu_memory)

    if batch_over_image_only:
        return compute_H_B(cryo, mean, dilated_volume_mask,
                                                                 picked_frequencies,
                                                                 int(image_batch_size ), (cov_noise),
                                                                 None , disc_type = disc_type,
                                                                 parallel_analysis = parallel_analysis,
                                                                 jax_random_key = 0, batch_over_H_B = True)
    

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
                                                                 jax_random_key = 0 )
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
    fsc_priors = [] 
    H_combined = []; B_combined = []; 
    
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
    C_n = core.sum_batch_P_adjoint_mat_vec(volume_size, w, plane_indices_on_grid_stacked) 

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
    E_n = core.sum_batch_P_adjoint_mat_vec(volume_size, delta_at_freq_mapped, plane_indices_on_grid_stacked)
    B_freq_idx = C_n - E_n

    # H
    v = ones_mapped * jnp.conj(CTF_val_on_grid_stacked)  
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])

    # Pick out only columns j for which V[idx,j] is not zero
    H_freq_idx = core.sum_batch_P_adjoint_mat_vec(volume_size, w, plane_indices_on_grid_stacked)
    
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
    C_n = core.sum_batch_P_adjoint_mat_vec(volume_size, w, plane_indices_on_grid_stacked) 

    # E_n
    delta_at_freq = jnp.zeros(volume_size, dtype = centered_images.dtype )
    delta_at_freq = delta_at_freq.at[picked_freq_index].set(1) 
    delta_at_freq_mapped = core.forward_model(delta_at_freq, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked) 
    
    # Apply mask
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)

    delta_at_freq_mapped *= cov_noise

    # Apply mask again conjugate == apply image mask since mask is real?
    delta_at_freq_mapped = covariance_core.apply_image_masks(delta_at_freq_mapped, image_mask, image_shape)
    
    delta_at_freq_mapped = delta_at_freq_mapped  * jnp.conj(CTF_val_on_grid_stacked)
    E_n = core.sum_batch_P_adjoint_mat_vec(volume_size, delta_at_freq_mapped, plane_indices_on_grid_stacked)
    B_freq_idx = C_n - E_n

    # H
    v = ones_mapped * jnp.conj(CTF_val_on_grid_stacked)  
    mult = jnp.sum(v * mask, axis = -1)
    w = v * jnp.conj(mult[:,None])

    # Pick out only columns j for which V[idx,j] is not zero
    H_freq_idx = core.sum_batch_P_adjoint_mat_vec(volume_size, w, plane_indices_on_grid_stacked)
    
    # H_zeros = H_freq_idx == 0 
    return H_freq_idx, B_freq_idx


# @functools.partial(jax.jit, static_argnums = [5])    
def compute_H_B(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, cov_noise, diag_prior, disc_type, parallel_analysis = False, jax_random_key = 0, batch_over_H_B = False, soften_mask = 3 ):
    # Memory in here scales as O (batch_size )

    utils.report_memory_device()

    volume_size = mean_estimate.size
    n_picked_indices = picked_frequency_indices.size
    H = [0] * n_picked_indices
    B = [0] * n_picked_indices
    jax_random_key = jax.random.PRNGKey(jax_random_key)
    mean_estimate = jnp.array(mean_estimate)

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for images, batch_image_ind in data_generator:
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              experiment_dataset.rotation_matrices[batch_image_ind], 
                                              experiment_dataset.image_stack.mask, 
                                              experiment_dataset.volume_mask_threshold, 
                                              experiment_dataset.image_shape, 
                                              experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                            experiment_dataset.padding, disc_type, soften = soften_mask )

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
            
        images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  

        batch_CTF = experiment_dataset.CTF_fun( experiment_dataset.CTF_params[batch_image_ind],
                                               experiment_dataset.image_shape,
                                               experiment_dataset.voxel_size)
        batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(
            experiment_dataset.rotation_matrices[batch_image_ind],
            experiment_dataset.image_shape, experiment_dataset.volume_shape, 
            experiment_dataset.grid_size )
        all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
        ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)
        
        apply_noise_mask = True
        if apply_noise_mask:
            f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
        else:
            f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])

        for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
            
            if (k % 50 == 49) and (k > 0):
                # print( k, " cols comp.")
                f_jit._clear_cache() # Maybe this?
                
            if apply_noise_mask:
                ### CHANGE NOISE ESTIMATE HERE?
                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size)
            else:
                H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, volume_size)
                
        # use_noise_mask = True
        # if use_noise_mask:
        #     f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])
        # else:
        #     f_jit = jax.jit(compute_H_B_inner, static_argnums = [6])

        # for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
            
        #     if (k % 50 == 49) and (k > 0):
        #         # There seemed to be some strange JAX memory leak, so this could fix it?
        #         f_jit._clear_cache() # Maybe this? 
            
        #     if use_noise_mask:
        #         H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, volume_size)
        #     else:
        #         H_k, B_k =  f_jit(images, ones_mapped, batch_CTF, batch_grid_pt_vec_ind_of_images, cov_noise, picked_freq_idx, volume_size)

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
