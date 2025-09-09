import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants, noise, cryojax_map_coordinates
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def get_default_covariance_computation_options(grid_size=None):

    gpu_memory = utils.get_gpu_memory_total()
    
    if grid_size is not None:
        # Account for basis memory: basis has shape (volume_size, n_pcs) 
        # where volume_size = grid_size^3
        # Memory usage scales as volume_size * n_pcs * dtype_size
        volume_size = grid_size ** 3
        dtype_size = 8  # bytes for complex64
        
        # Reserve some memory for other operations (keep ~30% free)
        available_memory_gb = gpu_memory * 0.7
        
        # The original formula: n_pcs = ceil((gpu_memory / (75 / 200^4))^(1/4))
        # This implies memory scales as: base_memory = (75 / 200^4) * n_pcs^4
        # Total memory = base_memory + basis_memory
        # Total memory = (75 / 200^4) * n_pcs^4 + volume_size * n_pcs * dtype_size / 1e9
        
        base_memory_coefficient = 75 / (200**4)  # From original formula
        basis_memory_coefficient = volume_size * dtype_size / 1e9  # GB per PC
        
        # Solve: base_memory_coefficient * n_pcs^4 + basis_memory_coefficient * n_pcs <= available_memory_gb
        # This is a quartic equation, but we can approximate by trying values
        
        if gpu_memory < 70:
            # Start with original estimate and adjust down if needed
            n_pcs_original = np.ceil((gpu_memory / (75 / 200**4))**(1/4)).astype(int)
        else:
            n_pcs_original = 200
            
        # Check if original estimate fits with basis memory
        for n_pcs in range(n_pcs_original, 0, -1):
            base_memory = base_memory_coefficient * (n_pcs ** 4)
            basis_memory = basis_memory_coefficient * n_pcs
            total_memory = base_memory + basis_memory
            
            if total_memory <= available_memory_gb:
                break
        else:
            n_pcs = 50  # Fallback to minimum
            
        logger.info(f"Using {n_pcs} PCs for covariance computation (GPU memory: {gpu_memory} GB, grid_size: {grid_size}, original estimate: {n_pcs_original}, base+basis memory: {base_memory + basis_memory:.2f} GB)")
    else:
        # Fallback to original calculation if grid_size not provided
        if gpu_memory < 70:
            n_pcs = np.ceil((gpu_memory / (75 / 200**4))**(1/4)).astype(int)
            logger.info(f"Using {n_pcs} PCs for covariance computation (GPU memory: {gpu_memory} GB)")
        else:
            n_pcs = 200
            logger.info(f"Using {n_pcs} PCs for covariance computation (GPU memory: {gpu_memory} GB)")

    options = {
        "covariance_fn": "kernel",
        "reg_fn": "new",
        "left_kernel": "triangular",
        "right_kernel": "triangular",
        "left_kernel_width": 1,
        "right_kernel_width": 2, # Probably should try 1 -2 - 3 ? 
        "shift_fsc": False, # Probably should be kept like this
        "substract_shell_mean": False, # Probably should be kept like this
        "grid_correct": True, # worth trying on/off
        "use_spherical_mask": True,
        "use_mask_in_fsc": True,
        "column_sampling_scheme": 'high_snr_from_var_est',
        "column_radius": 5,
        "use_combined_mean": True, # doesn't seem to change anything? worth a try
        "sampling_avoid_in_radius": 2, # Tuned
        "sampling_n_cols": 300, # A weird number for purely historical reasons. Change?
        "n_pcs_to_compute" : n_pcs,
        "randomized_sketch_size" : 300,
        "prior_n_iterations" : 20,
        "randomize_column_sampling": False,
        "disc_type": 'cubic',
        "disc_type_u": 'linear_interp',
        "mask_images_in_proj": False,
        "mask_images_in_H_B": True,
        "downsample_from_fsc" : False,
    }
    
    print( "--------------------------------" )
    print( "--------------------------------" )
    print( "mask_images_in_proj changed" )
    print( "mask_images_in_proj changed" )
    print( "--------------------------------" )
    print( "--------------------------------" )

    return options

def set_covariance_options(args, options):
    for key in options:
        if key in args:
            options[key] = args[key]
    return options


from recovar import core
def greedy_column_choice(sampling_vec, n_samples, volume_shape, avoid_in_radius = 1, keep_only_below_freq = 32):
    if avoid_in_radius < 0 or avoid_in_radius > 20:
        raise ValueError("avoid_in_radius should be between 0 and 20")

    if n_samples < 1 or n_samples > sampling_vec.size:
        raise ValueError("n_samples should be between 1 and the size of sampling_vec")

    radial_distances = ftu.get_grid_of_radial_distances(volume_shape)
    sampling_vec *= radial_distances.reshape(-1) < keep_only_below_freq

    sorted_idx = jnp.argsort(-sampling_vec)
    sorted_idx = np.array(sorted_idx)
    picked_set = set()
    picked = []
    n_picked =0 
    sorted_frequencies = core.vec_indices_to_frequencies(sorted_idx, volume_shape)
    sorted_frequencies_norm = np.linalg.norm(sorted_frequencies, axis=-1)

    for idx in sorted_idx:
        if idx not in picked_set:
            picked_frequency = core.vec_indices_to_frequencies(idx[None], volume_shape)
            # take things only on the + x side
            if picked_frequency[0,0] < 0:
                picked_frequency = -picked_frequency # Take the complex conjugate instead
                idx = int(core.frequencies_to_vec_indices(picked_frequency, volume_shape)[0])
            picked.append(idx)
            n_picked += 1
            if n_picked >= n_samples:
                break
            picked_set.add(idx)
            # Now take out everything that is close by and their complex conjugates

            nearby_freqs = core.find_frequencies_within_grid_dist(picked_frequency, np.ceil(avoid_in_radius).astype(int) )
            nearby_freqs = np.array(nearby_freqs)
            nearby_freqs = nearby_freqs[np.linalg.norm(nearby_freqs - picked_frequency, axis=-1) <= avoid_in_radius]
            nearby_freqs_negative = -nearby_freqs
            nearby_vec_indices = core.frequencies_to_vec_indices(nearby_freqs, volume_shape)
            nearby_negative_vec_indices = core.frequencies_to_vec_indices(nearby_freqs_negative, volume_shape)
            nearby_vec_indices = np.array(nearby_vec_indices)
            nearby_negative_vec_indices = np.array(nearby_negative_vec_indices)
            for k in range(nearby_vec_indices.size):
                picked_set.add(nearby_vec_indices[k])
                picked_set.add(nearby_negative_vec_indices[k])
    
    picked_frequencies = core.vec_indices_to_frequencies(np.array(picked), volume_shape)

    return np.array(picked), np.array(picked_frequencies)

def randomized_column_choice(sampling_vec, n_samples, volume_shape, avoid_in_radius = 1):
    if avoid_in_radius < 0 or avoid_in_radius > 20:
        raise ValueError("avoid_in_radius should be between 0 and 20")

    if n_samples < 1 or n_samples > sampling_vec.size:
        raise ValueError("n_samples should be between 1 and the size of sampling_vec")

    sorted_idx = jnp.argsort(-sampling_vec)
    sorted_idx = np.array(sorted_idx)
    picked_set = set()
    picked = []
    n_picked =0 
    sorted_frequencies = core.vec_indices_to_frequencies(sorted_idx, volume_shape)
    sorted_frequencies_norm = np.linalg.norm(sorted_frequencies, axis=-1)
    running_vec = sampling_vec.copy().astype(np.float64)

    probs = running_vec/np.sum(running_vec)
    random_choices = np.random.choice(running_vec.size, size = n_samples * 100, p = probs, replace=False)
    test_idx =0 

    while n_picked < n_samples:
        if test_idx >= random_choices.size:
            random_choices = np.random.choice(running_vec.size, n_samples * 100, p = probs, replace=False)[0]
            test_idx =0 

        idx = random_choices[test_idx]
        picked_frequency = core.vec_indices_to_frequencies(idx[None], volume_shape)
        # take things only on the + x side
        if picked_frequency[0,0] < 0:
            picked_frequency = -picked_frequency # Take the complex conjugate instead
            idx = int(core.frequencies_to_vec_indices(picked_frequency, volume_shape)[0])

        test_idx +=1        
        if idx in picked_set:
            continue

        picked.append(idx)
        n_picked += 1
        if n_picked >= n_samples:
            break
        picked_set.add(idx)

        nearby_freqs = core.find_frequencies_within_grid_dist(picked_frequency, np.ceil(avoid_in_radius).astype(int) )
        nearby_freqs = np.array(nearby_freqs)
        nearby_freqs = nearby_freqs[np.linalg.norm(nearby_freqs - picked_frequency, axis=-1) <= avoid_in_radius]
        nearby_freqs_negative = -nearby_freqs
        nearby_vec_indices = core.frequencies_to_vec_indices(nearby_freqs, volume_shape)
        nearby_negative_vec_indices = core.frequencies_to_vec_indices(nearby_freqs_negative, volume_shape)
        nearby_vec_indices = np.array(nearby_vec_indices)
        nearby_negative_vec_indices = np.array(nearby_negative_vec_indices)
        for k in range(nearby_vec_indices.size):
            picked_set.add(nearby_vec_indices[k])
            picked_set.add(nearby_negative_vec_indices[k])
    
    picked_frequencies = core.vec_indices_to_frequencies(np.array(picked), volume_shape)

    return np.array(picked), np.array(picked_frequencies)


def compute_regularized_covariance_columns_in_batch(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory, options, picked_frequencies):
    
    frequency_batch = utils.get_column_batch_size(cryos[0].grid_size, gpu_memory)    

    covariance_cols = []
    fscs = []
    for k in range(0, int(np.ceil(picked_frequencies.size/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min( [(k+1) * frequency_batch ,picked_frequencies.size  ]))

        covariance_cols_b, _, fscs_b = compute_regularized_covariance_columns(cryos, means, mean_prior,  volume_mask, dilated_volume_mask, valid_idx, gpu_memory,  options, picked_frequencies[batch_st:batch_end])
        logger.info(f'batch of col done: {batch_st}, {batch_end}')

        covariance_cols.append(covariance_cols_b['est_mask'])
        fscs.append(fscs_b)

    covariance_cols = {'est_mask' : np.concatenate(covariance_cols, axis = -1)}
    fscs = np.concatenate(fscs, axis = 0)
    return covariance_cols, picked_frequencies, fscs


def compute_regularized_covariance_columns(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory,  options, picked_frequencies):

    cryo = cryos[0]
    volume_shape = cryos[0].volume_shape

    # These options should probably be left as is.
    mask_ls = dilated_volume_mask
    mask_final = volume_mask
    # substract_shell_mean = False 
    # shift_fsc = False
    keep_intermediate = False
    # image_noise_var = noise.make_radial_noise(cov_noise, cryos[0].image_shape)

    utils.report_memory_device(logger = logger)
    Hs, Bs = compute_both_H_B(cryos, means, mask_ls, picked_frequencies, gpu_memory,  parallel_analysis = False, options = options)
    st_time = time.time() 
    volume_noise_var = np.asarray(noise.make_radial_noise(cryos[0].noise.get_average_radial_noise(), cryos[0].volume_shape))

    covariance_cols = {}
    if options["reg_fn"] == "new":
        logger.info("using new covariance reg fn")
        utils.report_memory_device(logger = logger)
        covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, options = options)
        covariance_cols["est_mask"] = covariance_cols["est_mask"].T
        del Hs, Bs
        logger.info("after reg fn")
        utils.report_memory_device(logger = logger)
    elif options["reg_fn"] == "old":
        logger.info("using old covariance reg fn")
        H_comb, B_comb, prior, fscs = compute_covariance_regularization(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, prior_iterations = 3, keep_intermediate = keep_intermediate, reg_init_multiplier = constants.REG_INIT_MULTIPLIER, substract_shell_mean = options["substract_shell_mean"], shift_fsc = options["shift_fsc"])

        del Hs, Bs

        H_comb = np.stack(H_comb).astype(dtype = cryo.dtype)
        B_comb = np.stack(B_comb).astype(dtype = cryo.dtype)

        st_time2 = time.time()
        cols2 = []
        for col_idx in range(picked_frequencies.size):
            cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx], prior[col_idx]) * valid_idx ))
            

        logger.info(f"cov update time: {time.time() - st_time2}")
        covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryo.dtype)
        logger.info(f"reg time: {time.time() - st_time}")
        utils.report_memory_device(logger = logger)
    else:
        assert False, "wrong covariance reg fn"

    return covariance_cols, picked_frequencies, np.asarray(fscs)


# import functools, jax
@functools.partial(jax.jit, static_argnums=[5,6,8, 12,13,14,15, 16, 17])
def variance_relion_style_triangular_kernel_batch_trilinear(mean_estimate, images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, noise_variances, volume_mask, image_mask, volume_mask_threshold, grid_size, padding, soften = 5, disc_type= '', premultiplied_ctf = False):

    CTF = CTF_fun( CTF_params, image_shape, voxel_size)

    images = core.translate_images(images, translations, image_shape) 
    # import pdb; pdb.set_trace()

    # images = images - core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, disc_type) * CTF_squared

    if premultiplied_ctf:
        images = images - core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, disc_type) * CTF**2
        # This is going to be (y_i CTF_i - CTF_i P_i mean_i). y_i = P_i mean_i + noise, in expectation
        # So the noise variance is going to be: E[(y_i - P_i mean_i)^2 CTF_i^2] = E[noise_i CTF_i] = E[noise_i] * CTF_i
        noise_p_variance_ctf = CTF**2
    else:
        images = images - core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, disc_type) * CTF
        noise_p_variance_ctf = jnp.ones_like(images)

    # This doesn't estimate the signal variance, but signal_variance + noise variance, which can be used to upper bound the signal variance, I guess
    # Before masking?
    Ft_im = core.adjoint_slice_volume_by_trilinear(jnp.abs(images)**2, rotation_matrices, image_shape, volume_shape)
    Ft_one = core.adjoint_slice_volume_by_trilinear(noise_p_variance_ctf, rotation_matrices, image_shape, volume_shape)

    if volume_mask is not None:

        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              rotation_matrices,
                                              image_mask, 
                                              volume_mask_threshold,
                                              image_shape, 
                                              volume_shape, grid_size, 
                                              padding, 
                                              'linear_interp', soften = soften )
        
        images = covariance_core.apply_image_masks(images, image_mask, image_shape)
        # If premultiplied_ctf, the noise distribution looks like: mask @ ctf @ noise_variance @ ctf @ mask
        if premultiplied_ctf:
            noise_variances = noise_variances * CTF**2
        cov_noise = noise.get_masked_noise_variance_from_noise_variance(image_mask, noise_variances, image_shape)

    # Maybe apply mask
    images_squared = jnp.abs(images)**2  - cov_noise.reshape(images.shape) #* np.sum(mask) # May need to do something with mask
    # summed_images_squared =  jnp.abs(images)**2

    CTF_squared = CTF**2

    if not premultiplied_ctf:
        images_squared *= CTF_squared

    Ft_y = core.adjoint_slice_volume_by_trilinear(images_squared, rotation_matrices, image_shape, volume_shape)

    Ft_ctf = core.adjoint_slice_volume_by_trilinear(CTF_squared**2, rotation_matrices, image_shape, volume_shape)

    return Ft_y, Ft_ctf, Ft_im, Ft_one


# This computes the lhs and rhs of two things: the estimator for the variance of the signal, and the variance of the var(signal)*CTF**2 + var(noise)**2
def variance_relion_style_triangular_kernel(experiment_dataset, mean_estimate,  batch_size, image_subset = None, volume_mask = None, disc_type= ''):

    # if image_subset is None:
    #     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # else:
    data_generator = experiment_dataset.get_image_subset_generator(batch_size=batch_size, subset_indices = image_subset)

    Ft_y, Ft_ctf, Ft_im, Ft_one = 0, 0, 0, 0
    for batch, particles_ind, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        noise_variances = experiment_dataset.noise.get(indices)
        Ft_y_b, Ft_ctf_b, Ft_im_b, Ft_one_b = variance_relion_style_triangular_kernel_batch_trilinear(mean_estimate, 
                                                                batch,
                                                                experiment_dataset.CTF_params[indices], 
                                                                experiment_dataset.rotation_matrices[indices], 
                                                                experiment_dataset.translations[indices], 
                                                                experiment_dataset.image_shape, 
                                                                experiment_dataset.upsampled_volume_shape, 
                                                                experiment_dataset.voxel_size, 
                                                                experiment_dataset.CTF_fun, 
                                                                noise_variances,
                                                                volume_mask,
                                                                experiment_dataset.image_stack.mask,
                                                                experiment_dataset.volume_mask_threshold,
                                                                experiment_dataset.grid_size,
                                                                experiment_dataset.padding,
                                                                soften = 5,
                                                                disc_type = disc_type,
                                                                premultiplied_ctf= experiment_dataset.premultiplied_ctf
                                                                )
        Ft_y += Ft_y_b
        Ft_ctf += Ft_ctf_b
        Ft_im += Ft_im_b
        Ft_one += Ft_one_b

    return Ft_ctf, Ft_y, Ft_one, Ft_im


def compute_variance(cryos, mean_estimate, batch_size, volume_mask, image_subset = None, use_regularization = False, disc_type = '', noise_ind_subset = None):
    st_time = time.time()

    cryo = cryos[0]
    from recovar import relion_functions

    variance = dict()
    lhs_l = 2 * [None]
    rhs_l = 2 * [None]
    noise_p_variance_lhs = 2 * [None]
    noise_p_variance_rhs = 2 * [None]

    if disc_type == 'cubic':
        mean_estimate = cryojax_map_coordinates.compute_spline_coefficients(mean_estimate.reshape(cryos[0].volume_shape))


    for idx, cryo in enumerate(cryos):
        if noise_ind_subset is not None:
            image_subset = np.where(cryo.noise.dose_indices == noise_ind_subset)[0]
        else:
            image_subset = None

        lhs_l[idx], rhs_l[idx], noise_p_variance_lhs[idx] , noise_p_variance_rhs[idx] = variance_relion_style_triangular_kernel(cryo, mean_estimate, batch_size, image_subset = image_subset, volume_mask = volume_mask, disc_type = disc_type)

        lhs_l[idx] = relion_functions.adjust_regularization_relion_style(lhs_l[idx], cryos[0].volume_shape, tau = None, padding_factor = 1, max_res_shell = None)
        variance["corrected" + str(idx)] = rhs_l[idx] / lhs_l[idx]

    lhs = (lhs_l[0] + lhs_l[1])/2
    variance_prior, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, variance["corrected0"], variance["corrected1"], lhs, jnp.ones(cryos[0].volume_size, dtype = cryos[0].dtype_real) * np.inf, frequency_shift = jnp.array([0,0,0]), upsampling_factor = 1, substract_shell_mean = True)

    if use_regularization:
        for idx, cryo in enumerate(cryos):
            lhs_l[idx] = relion_functions.adjust_regularization_relion_style(lhs_l[idx], cryos[0].volume_shape, tau = variance_prior, padding_factor = 1, max_res_shell = None)
            variance["corrected" + str(idx)] = rhs_l[idx] / lhs_l[idx]

    variance_prior = np.array(variance_prior)
    variance["combined"] = (variance["corrected0"] + variance["corrected1"])/2
    variance["prior"] = variance_prior
    variance["lhs"] = lhs

    for key in variance:
        variance[key] = np.array(variance[key]).real


    noise_p_variance_est = ( noise_p_variance_rhs[0] + noise_p_variance_rhs[1]) / (noise_p_variance_lhs[0] + noise_p_variance_lhs[1])

    end_time = time.time()
    logger.info(f"time to compute variance: {end_time- st_time}")

    return variance, variance_prior.real, fsc.real, lhs.real, noise_p_variance_est.real



def compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis, options ):
    Hs = []
    Bs = []
    st_time = time.time()

    for cryo_idx, cryo in enumerate(cryos):
        mean = means["combined"] if options["use_combined_mean"] else means["corrected" + str(cryo_idx)]
        H, B = compute_H_B_in_volume_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis, options = options)
        logger.info(f"Time to cov {time.time() - st_time}")
        # check_memory()
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


# AT SOME POINT, I CONVINCED MYSELF THAT IT WAS BETTER FOR MEMORY TRANSFER REASONS TO DO THIS IN BATCHES OVER VOLS, THEN OVER IMAGES. I am not sure anymore.
# Covariance_cols
def compute_H_B_in_volume_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis = False, options = None):

    image_batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory) // (2 if options['disc_type'] =='cubic' else 1)
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
    frequency_batch = column_batch_size

    for k in range(0, int(np.ceil(picked_frequencies.size/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min( [(k+1) * frequency_batch ,picked_frequencies.size  ]))
        # logger.info(f'outside H_B : {batch_st}, {batch_end}')
        # utils.report_memory_device(logger = logger)
        H_batch, B_batch = compute_H_B(cryo, mean, dilated_volume_mask,
                                                                 picked_frequencies[batch_st:batch_end],
                                                                 int(image_batch_size / 1),
                                                                 None ,
                                                                 parallel_analysis = parallel_analysis,
                                                                 jax_random_key = 0, options = options)
        H[:, batch_st:batch_end]  = np.array(H_batch)
        B[:, batch_st:batch_end]  = np.array(B_batch)
        del H_batch, B_batch
        
    return H,B


    
def compute_covariance_regularization_relion_style(Hs, Bs, mean_prior, picked_frequencies, cov_noise, volume_mask, volume_shape, gpu_memory, reg_init_multiplier, options):

    # assert substract_shell_mean == False
    # assert shift_fsc == False
    volume_mask = volume_mask if options["use_mask_in_fsc"] else None

    # 
    regularization_init = (mean_prior + 1e-14) * reg_init_multiplier / cov_noise
    def init_regularization_of_column_k(k):
        return regularization_init[None] * regularization_init[picked_frequencies[np.array(k)], None] 

    # This should probably be rewritten.
    # for cryo_idx in range(len(Hs)):
    #     Hs[cryo_idx] = Hs[cryo_idx].T
    #     Bs[cryo_idx] = Bs[cryo_idx].T
    
    # Column-wise regularize 
    shifts = core.vec_indices_to_frequencies(picked_frequencies, volume_shape) * (options["shift_fsc"])

    n_freqs = picked_frequencies.size
    fsc_priors = [None] * n_freqs
    fscs = [None] * n_freqs
    combined_cov_cols = [None] * n_freqs

    batch_size = utils.get_column_batch_size(volume_shape[0], gpu_memory) // 4

    for k in range(int(np.ceil(n_freqs/batch_size))-1, -1, -1):
        batch_st = int(k * batch_size)
        batch_end = int(np.min( [(k+1) * batch_size, n_freqs]))
        indices = np.arange(batch_st, batch_end)
        H0_batch = Hs[0][:,batch_st:batch_end].T
        H1_batch = Hs[1][:,batch_st:batch_end].T
        B0_batch = Bs[0][:,batch_st:batch_end].T
        B1_batch = Bs[1][:,batch_st:batch_end].T 

        combined_cov_col, priors, fscs_this = regularization.prior_iteration_relion_style_batch(H0_batch, H1_batch, B0_batch, B1_batch,
        shifts[indices],
        init_regularization_of_column_k(np.array(indices)), 
        options['substract_shell_mean'], 
        volume_shape, options['left_kernel'], 
        options['use_spherical_mask'],  options['grid_correct'],  volume_mask, options["prior_n_iterations"], options["downsample_from_fsc"])

        cpus = jax.devices("cpu")
        priors = jax.device_put(priors, cpus[0])
        for k,ind in enumerate(indices):
            if options["prior_n_iterations"] >= 0:
                fsc_priors[ind] = np.asarray(priors[k].real)
            fscs[ind] = np.asarray(fscs_this[k])
            combined_cov_cols[ind] = np.asarray(combined_cov_col[k])
        del combined_cov_col, priors

    if options["prior_n_iterations"] >= 0:
        fsc_priors = np.stack(fsc_priors, axis =0).real

    combined_cov_cols = np.stack(combined_cov_cols, axis =0)

    # Symmetricize prior    
    # fsc_priors[:,picked_frequencies] = 0.5 * ( fsc_priors[:,picked_frequencies] + fsc_priors[:,picked_frequencies].T ) 
    return combined_cov_cols, fsc_priors, fscs

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

    


# @functools.partial(jax.jit, static_argnums = [5])    
def compute_H_B(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, diag_prior, parallel_analysis = False, jax_random_key = 0, batch_over_H_B = False, soften_mask = 3, options = None ):
    # Memory in here scales as O (batch_size )

    # utils.report_memory_device()

    volume_size = mean_estimate.size
    n_picked_indices = picked_frequency_indices.size
    H = [0] * n_picked_indices
    B = [0] * n_picked_indices
    jax_random_key = jax.random.PRNGKey(jax_random_key)
    mean_estimate = jnp.array(mean_estimate)


    H_B_fn = options["covariance_fn"]
    # if H_B_fn =="noisemask":
    #     f_jit = jax.jit(compute_H_B_inner_mask, static_argnums = [7,8])
    if "kernel" in H_B_fn:
        f_jit = jax.jit(compute_H_B_triangular, static_argnums = [7,8,9,10,11,12, 13])
        # f_jit = compute_H_B_triangular#jax.jit(compute_H_B_triangular, static_argnums = [7,8,9,10,11,12])
    else:
        assert False, "Not recognized covariance_fn"

    if experiment_dataset.tilt_series_flag:
        assert "kernel" in H_B_fn, "Only kernel implemented for tilt series"

    if options['disc_type'] == 'cubic':
        these_disc = 'cubic'
        from recovar import cryojax_map_coordinates
        mean_estimate = cryojax_map_coordinates.compute_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))
    else:
        these_disc = 'linear_interp'

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size, mode='images') 
    for images, particles_ind, batch_image_ind in data_generator:
        # these_disc = 'linear_interp'
        # Probably should swap this to linear interp
        noise_variances = experiment_dataset.noise.get(particles_ind)
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              experiment_dataset.rotation_matrices[batch_image_ind], 
                                              experiment_dataset.image_stack.mask, 
                                              experiment_dataset.volume_mask_threshold, 
                                              experiment_dataset.image_shape, 
                                              experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                            experiment_dataset.padding, 'linear_interp', soften = soften_mask )
        if not options["mask_images_in_H_B"]:
            image_mask = jnp.ones_like(image_mask)

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
                                     these_disc, premultiplied_ctf = experiment_dataset.premultiplied_ctf )
                
        images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  

        batch_CTF = experiment_dataset.CTF_fun( experiment_dataset.CTF_params[batch_image_ind],
                                               experiment_dataset.image_shape,
                                               experiment_dataset.voxel_size)
        
        batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(
            experiment_dataset.rotation_matrices[batch_image_ind],
            experiment_dataset.image_shape, experiment_dataset.volume_shape )
        

        # all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
        # ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)
        
        if "kernel" in H_B_fn:
            batch_grid_pt_vec_ind_of_images = core.batch_get_gridpoint_coords(
                experiment_dataset.rotation_matrices[batch_image_ind],
                experiment_dataset.image_shape, experiment_dataset.volume_shape )
        else:
            all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
            ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)

        for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
            
            # if (k % 50 == 49) and (k > 0):
            #     # print( k, " cols comp.")
            #     f_jit._clear_cache() # Maybe this?

            H_k, B_k = f_jit(images, batch_CTF, batch_grid_pt_vec_ind_of_images, experiment_dataset.rotation_matrices[batch_image_ind],  noise_variances, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size, right_kernel = options["right_kernel"], left_kernel = options["left_kernel"], kernel_width = options["right_kernel_width"], shared_label = experiment_dataset.tilt_series_flag, premultiplied_ctf = experiment_dataset.premultiplied_ctf, tilt_labels = particles_ind)

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
        del images, batch_CTF, batch_grid_pt_vec_ind_of_images
        
    H = np.stack(H, axis =1)#, dtype=H[0].dtype)
    B = np.stack(B, axis =1)#, dtype=B[0].dtype)
    return H, B


from recovar import cryojax_map_coordinates
vmap_compute_spline_coefficients = jax.vmap(cryojax_map_coordinates.compute_spline_coefficients, in_axes = 0, out_axes = 0)

def compute_spline_coeffs_in_batch(basis, volume_shape, gpu_memory= None):
    gpu_memory = utils.get_gpu_memory_total() if gpu_memory is None else gpu_memory
    vol_batch_size = utils.get_vol_batch_size(volume_shape[0], gpu_memory=gpu_memory)
    logger.info(f"memory used = {gpu_memory}, vol_batch_size in compute_spline_coeffs_in_batch {vol_batch_size}")
    utils.report_memory_device(logger=logger)
    coeffs = []
    for k in range(0, basis.shape[0], vol_batch_size):
        coeffs.append(np.array(vmap_compute_spline_coefficients(basis[k:k+vol_batch_size].reshape(-1, *volume_shape))))
    coeffs = np.concatenate(coeffs, axis = 0)
    return coeffs

## REDUCED COVARIANCE COMPUTATION

# @functools.partial(jax.jit, static_argnums = [5])    
def compute_projected_covariance(experiment_datasets, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, parallel_analysis = False, do_mask_images = True ):
    
    experiment_dataset = experiment_datasets[0]

    basis = basis.T.astype(experiment_dataset.dtype)
    # Make sure variables used in every iteration are on gpu.
    basis = jnp.asarray(basis)
    volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    mean_estimate = jnp.array(mean_estimate).astype(experiment_dataset.dtype)
    jax_random_key = jax.random.PRNGKey(0)

    lhs =0
    rhs =0 
    summed_batch_kron_cpu = jax.jit(summed_batch_kron, backend='cpu')
    logger.info(f"batch size in compute_projected_covariance {batch_size}")

    if disc_type == 'cubic':
        mean_estimate = cryojax_map_coordinates.compute_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))

    if disc_type_u == 'cubic':
        basis = compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)

    change_device= False

    for experiment_dataset in experiment_datasets:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
        
        for batch, _, batch_image_ind in data_generator:
            jax_random_key, subkey = jax.random.split(jax_random_key)
            noise_variances = experiment_dataset.noise.get(batch_image_ind)

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
                                                                            disc_type, disc_type_u, noise_variances,
                                                                            experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun, parallel_analysis = parallel_analysis,
                                                                        jax_random_key =subkey, do_mask_images = do_mask_images,
                                                                        shared_label = experiment_dataset.tilt_series_flag, 
                                                                        premultiplied_ctf= experiment_dataset.premultiplied_ctf)
            

            # lhs_this = jax.device_put(lhs_this, jax.devices("cpu")[0])
            # lhs += summed_batch_kron_cpu(lhs_this)
            lhs += lhs_this
            rhs += rhs_this
            del lhs_this, rhs_this
        # del lhs_this, rhs_this
    del basis
    # Deallocate some memory?

    # Solve dense least squares?
    def vec(X):
        return X.T.reshape(-1)

    ## Inverse of vec function.
    def unvec(x):
        n = np.sqrt(x.size).astype(int)
        return x.reshape(n,n).T
    
    logger.info("end of covariance computation - before solve")
    utils.report_memory_device(logger=logger)
    rhs = vec(rhs)

    if change_device:
        rhs = jax.device_put(rhs, jax.devices("gpu")[0])
        lhs = jax.device_put(lhs, jax.devices("gpu")[0])
    # lhs_this = jax.device_put(lhs_this, jax.devices("gpu")[0])

    covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    # covar = linalg.batch_linear_solver(lhs, rhs)
    covar = unvec(covar)
    logger.info("end of solve")

    return covar


@functools.partial(jax.jit, static_argnums = [8,9,10,11,12,13,14,15,17,18, 19,21,22,23])    
def reduce_covariance_est_inner(batch, mean_estimate, volume_mask, basis, CTF_params,
                                rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape,
                                volume_shape, grid_size, voxel_size, padding, disc_type, 
                                disc_type_u, noise_variance, process_fn, CTF_fun, parallel_analysis = False,
                                jax_random_key = None, do_mask_images = True, shared_label = False, premultiplied_ctf = False):
    


    if (disc_type != 'linear_interp') and (disc_type != 'cubic'):
        logger.warning(f"USING NEAREST NEIGHBOR DISCRETIZATION IN reduce_covariance_est_inner. disc_type={disc_type}, disc_type_u={disc_type_u}")

    if premultiplied_ctf and do_mask_images:
        logger.warning('cannot use premultiplied ctf and mask images at the same time. Using premultiplied ctf!!!! CHECK IF THIS MATTERS')
        do_mask_images = False


    # Memory to do this is ~ size(volume_mask) * batch_size
    if do_mask_images:
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                            rotation_matrices,
                                            image_mask, 
                                            volume_mask_threshold,
                                            image_shape, 
                                            volume_shape, grid_size, 
                                            padding, 
                                            'linear_interp' )
        logger.warning("USING mask in reduce_covariance_est_inner")
    else:
        image_mask = jnp.ones_like(batch).real
        logger.info("NOT using mask in reduce_covariance_est_inner")
        

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    # Always computes CTF here
    # P_i mean
    projected_mean = core.forward_model_from_map(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                           
                                          )


    ## DO MASK BUSINESS HERE.
    if do_mask_images:
        batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
        projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    

    # premultiplied_ctf = False
    # Skip CTF if premultiplied here, do it later
    AUs = covariance_core.batch_over_vol_forward_model_from_map(basis,
                                         CTF_params, 
                                         rotation_matrices,
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type_u, 
                                        premultiplied_ctf) # skip_ctf = premultiplied_ctf)
    
    # Apply mask on operator
    if do_mask_images:
        AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)
    

    # IF premultiplied_ctf, this is CTF_i * (y_i - P_i mean)
    # If not, this is just y_i - P_i mean
    if premultiplied_ctf:
        CTF = CTF_fun(CTF_params, image_shape, voxel_size) 
        batch = batch - projected_mean * CTF
        AU_t_images = batch_x_T_y(AUs, batch)
        #     # This is done here because it is not done earlier, so that we can compute AU * (CTF * image), instead of (AU * CTF) * (image)
        #     # As images are "CTF premultiplied"
        #     # Howeve,r for the rest, we do want AU * CTF

        AUs = AUs * CTF[...,None] # Then CTF multiply

    else:
        # If we are here, AUs are CTFed, so no need to multiply on images
        batch = batch - projected_mean
        AU_t_images = batch_x_T_y(AUs, batch)
    # This gets inner product of AU with images
    # AU_t_images = batch_x_T_y(AUs, batch)

    # if premultiplied_ctf:
    #     # This is done here because it is not done earlier, so that we can compute AU * (CTF * image), instead of (AU * CTF) * (image)
    #     # As images are "CTF premultiplied"
    #     # Howeve,r for the rest, we do want AU * CTF
    #     AUs = AUs * CTF[...,None]

    # This is not correct if we are using the mask and the CTF is premultiplied
    if do_mask_images:
        assert not premultiplied_ctf, "Not implemented yet"

    AU_t_AU = batch_x_T_y(AUs,AUs).real.astype(CTF_params.dtype)
    
    # To save some memory, do it in place... but still unsure if this matters for JAX
    AUs *= jnp.sqrt(noise_variance)[...,None]
    UALambdaAUs = jnp.sum(batch_x_T_y(AUs,AUs), axis=0)

    if shared_label:
        AU_t_images = jnp.sum(AU_t_images, axis=0,keepdims=True)
        AU_t_AU = jnp.sum(AU_t_AU, axis=0,keepdims=True)

    outer_products = summed_outer_products(AU_t_images)

    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(CTF_params.dtype)
    # import pdb; pdb.set_trace()
    # return AU_t_AU, rhs
    # Perhaps this should use: jax.lax.fori_loop. This is a lot of memory.
    # Or maybe jax.lax.reduce ?
    lhs = jnp.sum(batch_kron(AU_t_AU, AU_t_AU), axis=0)

    return lhs, rhs



batch_kron = jax.vmap(jnp.kron, in_axes=(0,0))

def summed_batch_kron(X):
    return jnp.sum(batch_kron(X,X), axis=0)

def summed_batch_kron_scan(X):
    def fori_loop_body(i, val):
        return val + jnp.kron(X[i], X[i])
    summed_kron = jax.lax.fori_loop(0, X.shape[0], fori_loop_body, 0.)
    return summed_kron


batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))

def summed_outer_products(AU_t_images):
    # Not .H because things are already transposed technically
    return AU_t_images.T @ jnp.conj(AU_t_images)

batched_summed_outer_products  = jax.vmap(summed_outer_products)


# ## This is work in progress. Not currently used

# def compute_covariance_discretization_weights(experiment_dataset, prior, batch_size,  picked_freq_index, order=1 ):
#     order = 1   # Only implemented for order 1 but could generalize
#     rr_size = 6*order + 1
#     RR = jnp.zeros((experiment_dataset.volume_size, (rr_size)**2 ))
#     # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()) * 3

#     for i in range(utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)):
#         batch_st, batch_end = utils.get_batch_of_indices(experiment_dataset.n_images, batch_size, i)
#         # Make sure mean_estimate is size # volume_size ?
#         RR_this = compute_covariance_discretization_weights_inner(experiment_dataset.rotation_matrices[batch_st:batch_end], experiment_dataset.CTF_params[batch_st:batch_end], experiment_dataset.voxel_size, experiment_dataset.volume_shape, experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, picked_freq_index)
#         RR += RR_this

#     RR = RR.reshape([experiment_dataset.volume_size, rr_size, rr_size])
#     weights, good_weights = homogeneous.batch_solve_for_weights(RR, prior)
#     # If bad weights, just do Weiner filtering with 0th order disc
#     # good_weights = (good_weights*0).astype(bool)

#     other_weights = jnp.zeros_like(weights)
#     # weiner_weights = jnp.where(RR[:,0,0] > 0, 1/RR[:,0,0], jnp.zeros_like(RR[:,0,0]))
#     weiner_weights = 1 / (RR[...,0,0] + prior)#jnp.where(RR[:,0,0] > 0, 1/RR[:,0,0], jnp.zeros_like(RR[:,0,0]))
#     other_weights = other_weights.at[...,0].set(weiner_weights)

#     # weights = jnp.where(good_weights, weights, other_weights)
#     weights = weights.at[~good_weights].set(other_weights[~good_weights])
#     return weights, good_weights, RR

# def compute_covariance_discretization_weights_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, picked_freq_index ):

#     volume_size = np.prod(np.array(volume_shape))
#     C_mat, grid_point_vec_indices = make_C_mat_covariance(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun,picked_freq_index)
#     # This is going to be stroed twice as much stuff as it needs to be
#     C_mat_outer = homogeneous.batch_batch_outer(C_mat).reshape([C_mat.shape[0], C_mat.shape[1], C_mat.shape[2]*C_mat.shape[2]])#.transpose([2,0,1])
#     RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer, grid_point_vec_indices)
#     # mean_rhs = batch_over_weights_sum_adj_forward_model(volume_size, corrected_images , CTF, grid_point_indices)
#     return RR

# # Not sure this was ever finished being implemented?
# def make_C_mat_covariance(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, picked_freq_index):

#     ## NOT THERE ARE SOME -1 ENTRIES. BUT THEY GET GIVEN A 0 WEIGHT. IN THEORY, JAX JUST IGNORES THEM ANYWAY BUT SHOULD FIX THIS. 

#     # C_n
#     # mult = jnp.sum(v * mask, axis = -1)
#     # w = v * jnp.conj(mult[:,None])
#     # C_n = core.summed_adjoint_slice_by_nearest(volume_size, w, plane_indices_on_grid_stacked) 

#     grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )

#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     grid_points_coords_nearest = core.round_to_int(grid_points_coords)
#     differences = grid_points_coords - grid_points_coords_nearest
#     mask = grid_point_vec_indices == picked_freq_index

#     # found_indices = jnp.sum(mask, axis =0) > 1
#     differences_picked_freq = differences[mask].repeat(differences.shape[1], axis =1)

#     # Discretized grid points
#     # This could be done more efficiently

#     C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_picked_freq], axis = -1)

#     CTF1 = CTF_fun( CTF_params, image_shape, voxel_size)
#     CTF1_CTF2 = CTF1 * CTF1[mask]

#     C_mat *= CTF1_CTF2[...,None]
#     return C_mat, grid_point_vec_indices



@functools.partial(jax.jit, static_argnums=[2])
def group_sum_by_labels(array, tilt_labels, max_groups):
    """
    General function to group and sum arrays by tilt_labels.
    This is JIT-compatible and assumes tilt_labels are consecutive indices (0, 1, 2, ...).
    
    Args:
        array: Array to sum, shape (n_images, n_features)
        tilt_labels: Group labels, shape (n_images,)
        max_groups: Maximum number of groups (should be >= max(tilt_labels) + 1)
        
    Returns:
        Array with same shape as input, where each element is replaced by the sum of its group
    """
    # Sum within each tilt label group using scatter-add
    summed_by_label = jnp.zeros((max_groups, *array.shape[1:]), dtype=array.dtype)
    summed_by_label = summed_by_label.at[tilt_labels].add(array)
    
    # Repeat the summed values back to the original image positions
    return summed_by_label[tilt_labels]


# This computes the sums
#  H_{k_1, k_2} & = \sum_{i,j_1, j_2} c_{i,j_1}^2 c_{i,j_2}^2 K\left(\xi^{k_1} , \xi_{i,j_1} \right)  K\left(\xi^{k_2} , \xi_{i,j_2} \right)   \label{eq:H} \\
#  B_{k_1, k_2} & = \sum_{i,j_1, j_2} c_{i,j_1} c_{i,j_2} \left(l_{i,j_1} \overline{l_{i,j_2}} - \Lambda^{i}_{j_1, j_2}\right)K\left(\xi^{k_1} , \xi_{i,j_1} \right)  K\left(\xi^{k_2} , \xi_{i,j_2} \right)  \label{eq:B}
# For a fixed k_2
#  Eq. 12-13 in arxiv version?
def compute_H_B_triangular(centered_images, CTF_val_on_grid_stacked, plane_coords_on_grid_stacked, rotation_matrices,  noise_variances, picked_freq_index, image_mask, image_shape, volume_size, right_kernel = "triangular", left_kernel = "triangular", kernel_width = 2, shared_label = False, premultiplied_ctf = False, tilt_labels = None):
    # print("Using kernel", right_kernel, left_kernel, kernel_width)

    volume_shape = utils.guess_vol_shape_from_vol_size(volume_size)
    picked_freq_coord = core.vec_indices_to_vol_indices(picked_freq_index, volume_shape)

    # The image term
    # this is c_i l_i...
    if premultiplied_ctf:
        ctfed_images = centered_images
    else:
        ctfed_images = centered_images * jnp.conj(CTF_val_on_grid_stacked)

    # If ctf is premultiplied. the noise distribution is mask@ctf@noise_variance@ctf@mask
    # If not premultiplied, the noise distribution is ctf@mask@noise_variance@mask@ctf

    # The assumption in this is that the mask doesn't affect the signal, that is that  mask @ ctf * P @ x = ctf * P @ x. 
    # This is not true, but hopefully okay if the mask is sufficiently dilated


    # Between SPA and tomography, the only difference here is that we want to treat all images
    # (which are assumed to be from same tilt series) as a single measurement. 
    # Why the hell did I call this images_prod??? 
    # \xi^{k_2} frequency at k_2 == picked_freq_ind == column of covariance
    # This computes K(x_i, freq) for all x_i grid frequencies, then sums up 
    # c_{i,j_2} l_{i,j_2} K( \xi_{i,j_2}, k_2) over j_2
    images_prod = covariance_core.sum_up_over_near_grid_points(ctfed_images, plane_coords_on_grid_stacked, picked_freq_coord, kernel = right_kernel, kernel_width = kernel_width)
    

    if shared_label:
        
        # Group images by tilt_labels and sum within each group
        if tilt_labels is not None:
            # Use the jittable grouping function
            max_tilt_n_groups = centered_images.shape[0]  # Worst case: each image is its own group
            tilt_labels = preprocess_tilt_labels_for_batch(tilt_labels)
            images_prod = group_sum_by_labels(images_prod, tilt_labels, max_tilt_n_groups)
        else:
            # Fallback to original behavior if no tilt_labels provided
            images_prod = jnp.repeat(jnp.sum(images_prod, axis=0, keepdims=True), images_prod.shape[0], axis=0)

    ctfed_images  *= jnp.conj(images_prod)[...,None]
    # - noise term
    ## TODO: I'm still a little iffy about this in cryo-ET
    ctfed_images -= compute_noise_term(plane_coords_on_grid_stacked, picked_freq_coord, CTF_val_on_grid_stacked, image_shape, image_mask, noise_variances, kernel = right_kernel, kernel_width = kernel_width, premultiplied_ctf= premultiplied_ctf)
    
    rhs_summed_up = adjoint_kernel_slice(ctfed_images, rotation_matrices, image_shape, volume_shape, left_kernel)

    # lhs term 
    ctf_squared = CTF_val_on_grid_stacked * jnp.conj(CTF_val_on_grid_stacked)

    ctfs_prods = covariance_core.sum_up_over_near_grid_points(ctf_squared, plane_coords_on_grid_stacked, picked_freq_coord , kernel = right_kernel, kernel_width = kernel_width)

    if shared_label:
        # Group images by tilt_labels and sum within each group
        if tilt_labels is not None:
            # Use the jittable grouping function
            max_tilt_n_groups = centered_images.shape[0]  # Worst case: each image is its own group
            ctfs_prods = group_sum_by_labels(ctfs_prods, tilt_labels, max_tilt_n_groups)
        else:
            # Fallback to original behavior if no tilt_labels provided
            ctfs_prods = jnp.repeat(jnp.sum(ctfs_prods, axis=0, keepdims=True), ctfs_prods.shape[0], axis=0)

    ctf_squared *= ctfs_prods[...,None]
    lhs_summed_up = adjoint_kernel_slice(ctf_squared, rotation_matrices, image_shape, volume_shape, left_kernel)

    return lhs_summed_up, rhs_summed_up


def adjoint_kernel_slice(images, rotation_matrices, image_shape, volume_shape, kernel = "triangular"):
    if kernel == "triangular":
        lhs_summed_up = core.adjoint_slice_volume_by_trilinear(images, rotation_matrices,image_shape, volume_shape )
    elif kernel == "square":
        lhs_summed_up = core.adjoint_slice_volume_by_map(images, rotation_matrices,image_shape, volume_shape, 'nearest' )
    else:
        raise ValueError("Kernel not implemented")
    return lhs_summed_up


# This computes the sums
#  B_{k_1, k_2} & = \sum_{i,j_1, j_2} c_{i,j_1} c_{i,j_2}  \Lambda^{i}_{j_1, j_2} K\left(\xi^{k_1} , \xi_{i,j_1} \right)  K\left(\xi^{k_2} , \xi_{i,j_2} \right)  \label{eq:B}
# For a fixed k_2

def compute_noise_term(plane_coords, target_coord, CTF_on_grid, image_shape, image_mask, noise_variances, kernel = "triangular", kernel_width = 1, premultiplied_ctf = False):
    # Evaluate kernel

    # If ctf is premultiplied. the noise distribution is mask@ctf@noise_variance@ctf@mask
    # If not premultiplied, the noise distribution is ctf@mask@noise_variance@mask@ctf



    k_xi_x1 = covariance_core.evaluate_kernel_on_grid(plane_coords, target_coord, kernel = kernel, kernel_width = kernel_width) 
    if not premultiplied_ctf:
        k_xi_x1 *= CTF_on_grid

    # Apply mask
    k_xi_x1 = covariance_core.apply_image_masks(k_xi_x1, image_mask, image_shape)

    if premultiplied_ctf:
        k_xi_x1 *= jnp.conj(CTF_on_grid)

    # Multiply by noise
    k_xi_x1 *= noise_variances

    if premultiplied_ctf:
        k_xi_x1 *= jnp.conj(CTF_on_grid)

    # Apply mask again
    k_xi_x1 = covariance_core.apply_image_masks(k_xi_x1, image_mask, image_shape)

    if not premultiplied_ctf:
        k_xi_x1 *= jnp.conj(CTF_on_grid)

    # mutiply by CTF again
    return k_xi_x1 


def preprocess_tilt_labels_for_batch(tilt_labels):
    """
    Pre-process tilt_labels to be consecutive indices starting from 0.
    This should be called outside JIT to handle arbitrary tilt label values.
    
    Args:
        tilt_labels: Array of arbitrary tilt label values
        
    Returns:
        mapped_labels: Array of consecutive indices (0, 1, 2, ...)
        max_tilt_n_groups: Number of unique tilt groups
    """
    if tilt_labels is None:
        return None, None
    
    # Get unique labels and their inverse indices
    unique_labels, inverse_indices = jnp.unique(tilt_labels, return_inverse=True, size = tilt_labels.shape[0])
    
    # inverse_indices already gives us the mapping to consecutive indices
    return inverse_indices

