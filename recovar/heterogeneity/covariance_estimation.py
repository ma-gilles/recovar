"""Regularized covariance matrix estimation from half-set cryo-EM data."""

import functools
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import nvtx

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils, jax_config
from recovar.core import cubic_interpolation
from recovar.core.configs import ForwardModelConfig, BatchData, ModelState, CovarianceOpts
from recovar.heterogeneity import covariance_core
from recovar.reconstruction import regularization, noise

logger = logging.getLogger(__name__)

# CUDA Profiler for selective profiling with nsys
# Uses ctypes to call CUDA runtime API directly (no PyTorch needed)
try:
    import ctypes
    # Load CUDA runtime library
    _cudart = ctypes.CDLL('libcudart.so')
    
    def cudaProfilerStart():
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            logger.warning("cudaProfilerStart returned error code: %s", ret)
    
    def cudaProfilerStop():
        ret = _cudart.cudaProfilerStop()
        if ret != 0:
            logger.warning("cudaProfilerStop returned error code: %s", ret)
    
    CUDA_PROFILER_AVAILABLE = True
except Exception as e:
    CUDA_PROFILER_AVAILABLE = False
    logger.warning("CUDA profiler not available - profiling disabled: %s", e)

# ============================================================================
# NVTX Domain Configuration for Selective Profiling
# ============================================================================
# 
# This file uses NVTX domains to enable selective profiling of specific code paths.
# 
# USAGE:
# ------
# 1. Profile ONLY the compute_H_B function and its call chain:
#    nsys profile --nvtx-domain-include="compute_H_B" python your_script.py
#
# 2. Profile everything EXCEPT compute_H_B:
#    nsys profile --nvtx-domain-exclude="compute_H_B" python your_script.py
#
# 3. Profile multiple specific domains (if you add more):
#    nsys profile --nvtx-domain-include="compute_H_B,other_domain" python your_script.py
#
# BENEFITS:
# ---------
# - Reduces profile size and complexity
# - Focuses on performance-critical sections
# - Makes timeline easier to analyze
# - Filters out initialization and preprocessing overhead
#
# ANNOTATED FUNCTIONS IN compute_H_B DOMAIN:
# -------------------------------------------
# - compute_H_B (main function)
# - frequency_loop (inner loop over frequencies)
# - jit_compute_H_B_triangular_freq_{k} (per-frequency computation)
# - accumulate_H_B (result accumulation)
# - compute_H_B_triangular (core computation kernel)
# - compute_noise_term (noise correction)
# - adjoint_kernel_slice (backprojection)
# - get_per_image_tight_mask, process_and_center_images, compute_CTF, etc.
#
# ============================================================================

# Domain name for compute_H_B profiling (use as string in decorators)
NVTX_DOMAIN_H_B = "compute_H_B"

@nvtx.annotate("get_default_covariance_computation_options", color="red")
def get_default_covariance_computation_options(grid_size=None):
    """Return default options dict for covariance computation.

    Automatically sizes the number of principal components and column
    sampling parameters based on available GPU memory and the
    reconstruction grid size.

    Args:
        grid_size: Side length of the 3-D reconstruction grid.  When
            provided, the number of PCs is scaled to fit in GPU memory.

    Returns:
        Dictionary with keys ``covariance_fn``, ``reg_fn``,
        ``left_kernel``, ``right_kernel``, ``column_sampling_scheme``,
        ``n_pcs_to_compute``, among others.
    """

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
            
        logger.info("Using %s PCs for covariance computation (GPU memory: %s GB, grid_size: %s, original estimate: %s, base+basis memory: %.2f GB)", n_pcs, gpu_memory, grid_size, n_pcs_original, base_memory + basis_memory)
    else:
        # Fallback to original calculation if grid_size not provided
        if gpu_memory < 70:
            n_pcs = np.ceil((gpu_memory / (75 / 200**4))**(1/4)).astype(int)
            logger.info("Using %s PCs for covariance computation (GPU memory: %s GB)", n_pcs, gpu_memory)
        else:
            n_pcs = 200
            logger.info("Using %s PCs for covariance computation (GPU memory: %s GB)", n_pcs, gpu_memory)

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
    
    return options

@nvtx.annotate("set_covariance_options", color="red")
def set_covariance_options(args, options):
    for key in options:
        if key in args:
            options[key] = args[key]
    return options


from recovar import core
@nvtx.annotate("greedy_column_choice", color="orange")
def greedy_column_choice(sampling_vec, n_samples, volume_shape, avoid_in_radius = 1, keep_only_below_freq = 32):
    if avoid_in_radius < 0 or avoid_in_radius > 20:
        raise ValueError("avoid_in_radius should be between 0 and 20")

    if n_samples < 1 or n_samples > sampling_vec.size:
        raise ValueError("n_samples should be between 1 and the size of sampling_vec")

    radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape)
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
                if idx in picked_set:
                    continue
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

@nvtx.annotate("randomized_column_choice", color="orange")
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
    draw_size = min(running_vec.size, n_samples * 100)
    random_choices = np.random.choice(running_vec.size, size=draw_size, p=probs, replace=False)
    test_idx =0 

    while n_picked < n_samples:
        if test_idx >= random_choices.size:
            random_choices = np.random.choice(running_vec.size, size=draw_size, p=probs, replace=False)
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


@nvtx.annotate("compute_regularized_covariance_columns_in_batch", color="purple")
def compute_regularized_covariance_columns_in_batch(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory, options, picked_frequencies, use_multi_gpu = False, n_gpus = None, mean_cubic=None):
    """Compute regularized covariance matrix columns in GPU-sized batches.

    Iterates over *picked_frequencies* in batches that fit in GPU memory
    and concatenates the results.

    Args:
        cryos: Half-set datasets (``CryoEMHalfsets``).
        means: Dict with keys ``'combined'``, ``'prior'``, ``'lhs'``.
        mean_prior: Prior mean volume (Fourier coefficients).
        volume_mask: Binary mask selecting valid voxels.
        dilated_volume_mask: Dilated version of *volume_mask*.
        valid_idx: Indices of valid Fourier frequencies.
        gpu_memory: Available GPU memory in GB.
        options: Pipeline options namespace.
        picked_frequencies: 1-D array of frequency indices to compute.
        use_multi_gpu: Distribute across multiple GPUs.
        n_gpus: Number of GPUs (``None`` = auto-detect).
        mean_cubic: Pre-computed cubic-interpolation coefficients for the mean.

    Returns:
        Tuple ``(covariance_cols, picked_frequencies, fscs)`` where
        *covariance_cols* is a dict with key ``'est_mask'`` and *fscs*
        contains per-column FSC curves.
    """

    frequency_batch = utils.get_column_batch_size(cryos.grid_size, gpu_memory)

    covariance_cols = []
    fscs = []
    for k in range(0, int(np.ceil(picked_frequencies.size/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min( [(k+1) * frequency_batch ,picked_frequencies.size  ]))

        covariance_cols_b, _, fscs_b = compute_regularized_covariance_columns(cryos, means, mean_prior,  volume_mask, dilated_volume_mask, valid_idx, gpu_memory,  options, picked_frequencies[batch_st:batch_end], use_multi_gpu = use_multi_gpu, n_gpus = n_gpus, mean_cubic=mean_cubic)
        logger.info('batch of col done: %s, %s', batch_st, batch_end)

        covariance_cols.append(covariance_cols_b['est_mask'])
        fscs.append(fscs_b)

    covariance_cols = {'est_mask' : np.concatenate(covariance_cols, axis = -1)}
    fscs = np.concatenate(fscs, axis = 0)
    return covariance_cols, picked_frequencies, fscs


@nvtx.annotate("compute_regularized_covariance_columns", color="purple")
def compute_regularized_covariance_columns(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory,  options, picked_frequencies, use_multi_gpu = False, n_gpus = None, mean_cubic=None):

    volume_shape = cryos.volume_shape

    # These options should probably be left as is.
    mask_ls = dilated_volume_mask
    mask_final = volume_mask
    keep_intermediate = False

    utils.report_memory_device(logger = logger)
    
    # Start CUDA profiler for covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStart()
        logger.info("CUDA Profiler: Started profiling covariance computation")
    
    Hs, Bs = compute_both_H_B(cryos, means, mask_ls, picked_frequencies, gpu_memory,  parallel_analysis = False, options = options, use_multi_gpu = use_multi_gpu, n_gpus = n_gpus, mean_cubic=mean_cubic)
    
    # Stop CUDA profiler after covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStop()
        logger.info("CUDA Profiler: Stopped profiling covariance computation")
    
    st_time = time.time() 
    volume_noise_var = np.asarray(noise.make_radial_noise(cryos[0].noise.get_average_radial_noise(), cryos.volume_shape))

    covariance_cols = {}
    if options["reg_fn"] == "new":
        logger.info("using new covariance reg fn")
        utils.report_memory_device(logger = logger)
        covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, reg_init_multiplier = jax_config.REG_INIT_MULTIPLIER, options = options)
        covariance_cols["est_mask"] = covariance_cols["est_mask"].T
        del Hs, Bs
        logger.info("after reg fn")
        utils.report_memory_device(logger = logger)
    elif options["reg_fn"] == "old":
        logger.info("using old covariance reg fn")
        H_comb, B_comb, prior, fscs = compute_covariance_regularization(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape,  gpu_memory, prior_iterations = 3, keep_intermediate = keep_intermediate, reg_init_multiplier = jax_config.REG_INIT_MULTIPLIER, substract_shell_mean = options["substract_shell_mean"], shift_fsc = options["shift_fsc"])

        del Hs, Bs

        H_comb = np.stack(H_comb).astype(dtype = cryos.dtype)
        B_comb = np.stack(B_comb).astype(dtype = cryos.dtype)

        st_time2 = time.time()
        cols2 = []
        for col_idx in range(picked_frequencies.size):
            cols2.append(np.array(regularization.covariance_update_col(H_comb[col_idx], B_comb[col_idx], prior[col_idx]) * valid_idx ))
            

        logger.info("cov update time: %s", time.time() - st_time2)
        covariance_cols["est_mask"] = np.stack(cols2, axis =-1).astype(cryos.dtype)
        logger.info("reg time: %s", time.time() - st_time)
        utils.report_memory_device(logger = logger)
    else:
        assert False, "wrong covariance reg fn"

    return covariance_cols, picked_frequencies, np.asarray(fscs)


# ============================================================================
# New Equinox-based variance estimation
# ============================================================================


@eqx.filter_jit
@nvtx.annotate("variance_relion_kernel_trilinear", color="yellow")
def variance_relion_kernel_trilinear(
    config: ForwardModelConfig,
    batch_data: BatchData,
    mean_estimate: jax.Array,
    volume_mask: jax.Array,
    image_mask: jax.Array,
    soften: int = 5,
):
    """Variance estimation via RELION-style trilinear kernel — Equinox API.

    Replaces the 18-param ``variance_relion_style_triangular_kernel_batch_trilinear``.
    """
    images = batch_data.images
    ctf_params = batch_data.ctf_params
    rotation_matrices = batch_data.rotation_matrices
    translations = batch_data.translations
    noise_variances = batch_data.noise_variance

    CTF = config.compute_ctf(ctf_params)
    images = core.translate_images(images, translations, config.image_shape)

    if config.premultiplied_ctf:
        images = images - core.slice_volume_by_map(
            mean_estimate, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
        ) * CTF ** 2
        noise_p_variance_ctf = CTF ** 2
    else:
        images = images - core.slice_volume_by_map(
            mean_estimate, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
        ) * CTF
        noise_p_variance_ctf = jnp.ones_like(images)

    Ft_im = core.adjoint_slice_volume_by_trilinear(
        jnp.abs(images) ** 2, rotation_matrices, config.image_shape, config.volume_shape
    )
    Ft_one = core.adjoint_slice_volume_by_trilinear(
        noise_p_variance_ctf, rotation_matrices, config.image_shape, config.volume_shape
    )

    if volume_mask is not None:
        image_mask = covariance_core.get_per_image_tight_mask(
            volume_mask, rotation_matrices, image_mask, config.volume_mask_threshold,
            config.image_shape, config.volume_shape, config.grid_size, config.padding,
            'linear_interp', soften=soften,
        )
        images = covariance_core.apply_image_masks(images, image_mask, config.image_shape)
        if config.premultiplied_ctf:
            noise_variances = noise_variances * CTF ** 2
        cov_noise = noise.get_masked_noise_variance_from_noise_variance(
            image_mask, noise_variances, config.image_shape
        )

    images_squared = jnp.abs(images) ** 2 - cov_noise.reshape(images.shape)
    CTF_squared = CTF ** 2

    if not config.premultiplied_ctf:
        images_squared *= CTF_squared

    Ft_y = core.adjoint_slice_volume_by_trilinear(
        images_squared, rotation_matrices, config.image_shape, config.volume_shape
    )
    Ft_ctf = core.adjoint_slice_volume_by_trilinear(
        CTF_squared ** 2, rotation_matrices, config.image_shape, config.volume_shape
    )
    return Ft_y, Ft_ctf, Ft_im, Ft_one


@nvtx.annotate("variance_relion_style_triangular_kernel", color="yellow")
def variance_relion_style_triangular_kernel(experiment_dataset, mean_estimate, batch_size, image_subset=None, volume_mask=None, disc_type=''):

    data_generator = experiment_dataset.get_image_subset_generator(batch_size=batch_size, subset_indices=image_subset)

    # Construct config once (uses upsampled_volume_shape for variance estimation)
    config = ForwardModelConfig(
        image_shape=tuple(experiment_dataset.image_shape),
        volume_shape=tuple(experiment_dataset.upsampled_volume_shape),
        grid_size=int(experiment_dataset.grid_size),
        voxel_size=float(experiment_dataset.voxel_size),
        padding=int(experiment_dataset.padding),
        disc_type=disc_type,
        CTF_fun=experiment_dataset.CTF_fun,
        premultiplied_ctf=bool(experiment_dataset.premultiplied_ctf),
        volume_mask_threshold=float(experiment_dataset.volume_mask_threshold),
    )

    Ft_y, Ft_ctf, Ft_im, Ft_one = 0, 0, 0, 0
    for batch, particles_ind, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask=False)
        noise_variances = experiment_dataset.noise.get(indices)

        batch_data = BatchData(
            images=batch,
            ctf_params=experiment_dataset.CTF_params[indices],
            rotation_matrices=experiment_dataset.rotation_matrices[indices],
            translations=experiment_dataset.translations[indices],
            noise_variance=noise_variances,
        )
        Ft_y_b, Ft_ctf_b, Ft_im_b, Ft_one_b = variance_relion_kernel_trilinear(
            config, batch_data, mean_estimate, volume_mask,
            experiment_dataset.image_stack.mask, soften=5,
        )
        Ft_y += Ft_y_b
        Ft_ctf += Ft_ctf_b
        Ft_im += Ft_im_b
        Ft_one += Ft_one_b

    return Ft_ctf, Ft_y, Ft_one, Ft_im


@nvtx.annotate("compute_variance", color="yellow")
def compute_variance(cryos, mean_estimate, batch_size, volume_mask, image_subset = None, use_regularization = False, disc_type = '', noise_ind_subset = None, mean_cubic=None):
    st_time = time.time()

    from recovar.reconstruction import relion_functions

    variance = dict()
    lhs_l = 2 * [None]
    rhs_l = 2 * [None]
    noise_p_variance_lhs = 2 * [None]
    noise_p_variance_rhs = 2 * [None]

    if disc_type == 'cubic':
        if mean_cubic is not None:
            mean_estimate = mean_cubic
        else:
            mean_estimate = cubic_interpolation.calculate_spline_coefficients(mean_estimate.reshape(cryos.volume_shape))


    for idx, cryo in enumerate(cryos):
        if noise_ind_subset is not None:
            image_subset = np.where(cryo.noise.dose_indices == noise_ind_subset)[0]
        else:
            image_subset = None

        lhs_l[idx], rhs_l[idx], noise_p_variance_lhs[idx] , noise_p_variance_rhs[idx] = variance_relion_style_triangular_kernel(cryo, mean_estimate, batch_size, image_subset = image_subset, volume_mask = volume_mask, disc_type = disc_type)

        lhs_l[idx] = relion_functions.adjust_regularization_relion_style(lhs_l[idx], cryos.volume_shape, tau = None, padding_factor = 1, max_res_shell = None)
        variance["corrected" + str(idx)] = rhs_l[idx] / lhs_l[idx]

    lhs = (lhs_l[0] + lhs_l[1])/2
    variance_prior, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryos.volume_shape, variance["corrected0"], variance["corrected1"], lhs, jnp.ones(cryos.volume_size, dtype = cryos.dtype_real) * np.inf, frequency_shift = jnp.array([0,0,0]), upsampling_factor = 1, substract_shell_mean = True)

    if use_regularization:
        for idx, cryo in enumerate(cryos):
            lhs_l[idx] = relion_functions.adjust_regularization_relion_style(lhs_l[idx], cryos.volume_shape, tau = variance_prior, padding_factor = 1, max_res_shell = None)
            variance["corrected" + str(idx)] = rhs_l[idx] / lhs_l[idx]

    variance_prior = np.array(variance_prior)
    variance["combined"] = (variance["corrected0"] + variance["corrected1"])/2
    variance["prior"] = variance_prior
    variance["lhs"] = lhs

    for key in variance:
        variance[key] = np.array(variance[key]).real


    noise_p_variance_est = ( noise_p_variance_rhs[0] + noise_p_variance_rhs[1]) / (noise_p_variance_lhs[0] + noise_p_variance_lhs[1])

    end_time = time.time()
    logger.info("time to compute variance: %s", end_time- st_time)

    return variance, variance_prior.real, fsc.real, lhs.real, noise_p_variance_est.real


@nvtx.annotate("compute_both_H_B", color="blue")
def compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis, options, use_multi_gpu = False, n_gpus = None, mean_cubic=None):
    Hs = []
    Bs = []
    st_time = time.time()

    for cryo_idx, cryo in enumerate(cryos):
        mean = means["combined"] if options["use_combined_mean"] else means["corrected" + str(cryo_idx)]
        H, B = compute_H_B_in_volume_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis, options = options, use_multi_gpu = use_multi_gpu, n_gpus = n_gpus, mean_cubic=mean_cubic)
        logger.info("Time to cov %s", time.time() - st_time)
        # check_memory()
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


# AT SOME POINT, I CONVINCED MYSELF THAT IT WAS BETTER FOR MEMORY TRANSFER REASONS TO DO THIS IN BATCHES OVER VOLS, THEN OVER IMAGES. I am not sure anymore.
# Covariance_cols
@nvtx.annotate("compute_H_B_in_volume_batch", color="blue")
def compute_H_B_in_volume_batch(cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, parallel_analysis = False, options = None, use_multi_gpu = False, n_gpus = None, mean_cubic=None):

    # //2 for cubic: spline coefficient arrays use ~2x more memory than linear
    image_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(cryo.grid_size, gpu_memory) // (2 if options['disc_type'] =='cubic' else 1))
    column_batch_size = utils.get_column_batch_size(cryo.grid_size, gpu_memory)

    # Multi-GPU path
    if use_multi_gpu:
        from recovar.utils import multi_gpu as multi_gpu_utils
        
        logger.info("=" * 60)
        logger.info("MULTI-GPU MODE ENABLED")
        logger.info("=" * 60)
        
        # Define wrapper function that matches the signature expected by multi_gpu_utils
        def compute_H_B_single_gpu_wrapper(experiment_dataset, mean_estimate, volume_mask, 
                                            picked_frequency_indices, batch_size, diag_prior,
                                            parallel_analysis, jax_random_key, options, image_subset=None):
            """Wrapper that processes all frequencies for a subset of images."""
            H = np.empty([experiment_dataset.volume_size, picked_frequency_indices.size], dtype=experiment_dataset.dtype)
            B = np.empty([experiment_dataset.volume_size, picked_frequency_indices.size], dtype=experiment_dataset.dtype)
            frequency_batch = column_batch_size
            
            for k in range(0, int(np.ceil(picked_frequency_indices.size/frequency_batch))):
                batch_st = int(k * frequency_batch)
                batch_end = int(np.min([(k+1) * frequency_batch, picked_frequency_indices.size]))
                
                H_batch, B_batch = compute_H_B(experiment_dataset, mean_estimate, volume_mask,
                                                picked_frequency_indices[batch_st:batch_end],
                                                batch_size,
                                                None,
                                                parallel_analysis=parallel_analysis,
                                                jax_random_key=jax_random_key,
                                                options=options,
                                                image_subset=image_subset,
                                                mean_cubic=mean_cubic)
                
                # Tag the array copy operations after compute_H_B returns
                with nvtx.annotate("copy_H_B_to_output", color="gold", domain=NVTX_DOMAIN_H_B):
                    # Avoid unnecessary copy if already NumPy array
                    H[:, batch_st:batch_end] = H_batch if isinstance(H_batch, np.ndarray) else np.array(H_batch)
                    B[:, batch_st:batch_end] = B_batch if isinstance(B_batch, np.ndarray) else np.array(B_batch)
                
                # Tag the cleanup operation
                with nvtx.annotate("delete_H_B_batch", color="gray", domain=NVTX_DOMAIN_H_B):
                    del H_batch, B_batch
            
            return H, B
        
        # Use multi-GPU computation
        H, B = multi_gpu_utils.compute_H_B_multi_gpu(
            compute_H_B_fn=compute_H_B_single_gpu_wrapper,
            experiment_dataset=cryo,
            n_gpus=n_gpus,
            mean_estimate=mean,
            volume_mask=dilated_volume_mask,
            picked_frequency_indices=picked_frequencies,
            batch_size=image_batch_size,
            diag_prior=None,
            parallel_analysis=parallel_analysis,
            jax_random_key=0,
            options=options
        )
        
        logger.info("=" * 60)
        logger.info("MULTI-GPU COMPUTATION COMPLETED")
        logger.info("=" * 60)
        
        return H, B
    
    # Original single-GPU path
    H = np.empty( [cryo.volume_size, picked_frequencies.size] , dtype = cryo.dtype)
    B = np.empty( [cryo.volume_size, picked_frequencies.size] , dtype = cryo.dtype)
    frequency_batch = column_batch_size

    for k in range(0, int(np.ceil(picked_frequencies.size/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min( [(k+1) * frequency_batch ,picked_frequencies.size  ]))
        H_batch, B_batch = compute_H_B(cryo, mean, dilated_volume_mask,
                                                                 picked_frequencies[batch_st:batch_end],
                                                                 image_batch_size,
                                                                 None,
                                                                 parallel_analysis = parallel_analysis,
                                                                 jax_random_key = 0, options = options,
                                                                 mean_cubic=mean_cubic)
        H[:, batch_st:batch_end]  = H_batch
        B[:, batch_st:batch_end]  = B_batch
        del H_batch, B_batch
        
    return H,B


@nvtx.annotate("compute_covariance_regularization_relion_style", color="cyan")
def compute_covariance_regularization_relion_style(Hs, Bs, mean_prior, picked_frequencies, cov_noise, volume_mask, volume_shape, gpu_memory, reg_init_multiplier, options):

    volume_mask = volume_mask if options["use_mask_in_fsc"] else None

    # 
    regularization_init = (mean_prior + 1e-14) * reg_init_multiplier / cov_noise
    def init_regularization_of_column_k(k):
        return regularization_init[None] * regularization_init[picked_frequencies[np.array(k)], None] 

    
    # Column-wise regularize 
    shifts = core.vec_indices_to_frequencies(picked_frequencies, volume_shape) * (options["shift_fsc"])

    n_freqs = picked_frequencies.size
    fsc_priors = [None] * n_freqs
    fscs = [None] * n_freqs
    combined_cov_cols = [None] * n_freqs

    # //4: regularization needs 4 volume-sized arrays simultaneously (H0, H1, B0, B1)
    batch_size = utils.safe_batch_size(utils.get_column_batch_size(volume_shape[0], gpu_memory) // 4)

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

    return combined_cov_cols, fsc_priors, fscs

@nvtx.annotate("compute_covariance_regularization", color="cyan")
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

    # //4: regularization needs 4 volume-sized arrays simultaneously (H0, H1, B0, B1)
    batch_size = utils.safe_batch_size(utils.get_column_batch_size(volume_shape[0], gpu_memory) // 4)

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

    
@nvtx.annotate("compute_H_B", color="blue", domain=NVTX_DOMAIN_H_B)
def compute_H_B(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, diag_prior, parallel_analysis = False, jax_random_key = 0, batch_over_H_B = False, soften_mask = 3, options = None, image_subset = None, mean_cubic=None):
    volume_size = mean_estimate.size
    n_picked_indices = picked_frequency_indices.size

    # Use lists to accumulate JAX arrays (keeps data on GPU/in JAX)
    H = [0] * n_picked_indices
    B = [0] * n_picked_indices

    jax_random_key = jax.random.PRNGKey(jax_random_key)
    mean_estimate = jnp.array(mean_estimate)


    H_B_fn = options["covariance_fn"]
    with nvtx.annotate("create_jit_function", color="pink", domain=NVTX_DOMAIN_H_B):
        if "kernel" in H_B_fn:
            f_jit = jax.jit(compute_H_B_triangular, static_argnums = [7,8,9,10,11,12, 13])
        else:
            assert False, "Not recognized covariance_fn"

    if experiment_dataset.tilt_series_flag:
        assert "kernel" in H_B_fn, "Only kernel implemented for tilt series"

    if options['disc_type'] == 'cubic':
        these_disc = 'cubic'
        if mean_cubic is not None:
            mean_estimate = mean_cubic
        else:
            mean_estimate = cubic_interpolation.calculate_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))
    else:
        these_disc = 'linear_interp'

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=these_disc)

    # Use subset generator if image_subset is provided, otherwise use full dataset
    if image_subset is not None:
        data_generator = experiment_dataset.get_image_subset_generator(batch_size=batch_size, subset_indices=image_subset)
    else:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size, mode='images')
    for images, particles_ind, batch_image_ind in data_generator:
        noise_variances = experiment_dataset.noise.get(particles_ind)
        
        with nvtx.annotate("get_per_image_tight_mask", color="green", domain=NVTX_DOMAIN_H_B):
            image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                                  experiment_dataset.rotation_matrices[batch_image_ind], 
                                                  experiment_dataset.image_stack.mask, 
                                                  experiment_dataset.volume_mask_threshold, 
                                                  experiment_dataset.image_shape, 
                                                  experiment_dataset.volume_shape, experiment_dataset.grid_size, 
                                                experiment_dataset.padding, 'linear_interp', soften = soften_mask )
            if not options["mask_images_in_H_B"]:
                image_mask = jnp.ones_like(image_mask)

        with nvtx.annotate("process_and_center_images", color="cyan", domain=NVTX_DOMAIN_H_B):
            images = experiment_dataset.image_stack.process_images(images)
            images = covariance_core.centered_images(config, images, mean_estimate,
                                         experiment_dataset.CTF_params[batch_image_ind],
                                         experiment_dataset.rotation_matrices[batch_image_ind],
                                         experiment_dataset.translations[batch_image_ind])
                    
            images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  

        with nvtx.annotate("compute_CTF", color="yellow", domain=NVTX_DOMAIN_H_B):
            batch_CTF = experiment_dataset.CTF_fun( experiment_dataset.CTF_params[batch_image_ind],
                                                   experiment_dataset.image_shape,
                                                   experiment_dataset.voxel_size)
        
        with nvtx.annotate("get_gridpoint_indices", color="magenta", domain=NVTX_DOMAIN_H_B):
            batch_grid_pt_vec_ind_of_images = core.batch_get_nearest_gridpoint_indices(
                experiment_dataset.rotation_matrices[batch_image_ind],
                experiment_dataset.image_shape, experiment_dataset.volume_shape )
        

        if "kernel" in H_B_fn:
            with nvtx.annotate("get_gridpoint_coords", color="magenta", domain=NVTX_DOMAIN_H_B):
                batch_grid_pt_vec_ind_of_images = core.batch_get_gridpoint_coords(
                    experiment_dataset.rotation_matrices[batch_image_ind],
                    experiment_dataset.image_shape, experiment_dataset.volume_shape )
        else:
            all_one_volume = jnp.ones(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
            ones_mapped = core.forward_model(all_one_volume, batch_CTF, batch_grid_pt_vec_ind_of_images)

        with nvtx.annotate("frequency_loop", color="red", domain=NVTX_DOMAIN_H_B):
            for (k, picked_freq_idx) in enumerate(picked_frequency_indices):
                

                with nvtx.annotate(f"jit_compute_H_B_triangular_freq_{k}", color="orange", domain=NVTX_DOMAIN_H_B):
                    H_k, B_k = f_jit(images, batch_CTF, batch_grid_pt_vec_ind_of_images, experiment_dataset.rotation_matrices[batch_image_ind],  noise_variances, picked_freq_idx, image_mask, experiment_dataset.image_shape, volume_size, right_kernel = options["right_kernel"], left_kernel = options["left_kernel"], kernel_width = options["right_kernel_width"], shared_label = experiment_dataset.tilt_series_flag, premultiplied_ctf = experiment_dataset.premultiplied_ctf, tilt_labels = particles_ind)

                _cpu = jax.devices("cpu")[0]

                with nvtx.annotate("accumulate_H_B", color="purple", domain=NVTX_DOMAIN_H_B):
                    if batch_over_H_B:
                        # Send to cpu and accumulate in list (keeps as JAX arrays)
                        H[k] += jax.device_put(H_k, _cpu)
                        B[k] += jax.device_put(B_k, _cpu)
                        del H_k, B_k
                    else:
                        # Accumulate in list (keeps as JAX arrays)
                        H[k] += H_k.real.astype(experiment_dataset.dtype_real)
                        B[k] += B_k
        with nvtx.annotate("cleanup_batch", color="gray", domain=NVTX_DOMAIN_H_B):
            del image_mask
            del images, batch_CTF, batch_grid_pt_vec_ind_of_images
    
    # Stack arrays in batches to avoid GPU memory exhaustion
    # OPTIMIZATION: Batched GPU stack + batched transfers
    # Previous approach: np.stack() triggered 300+ sequential DtoH transfers (~15s)
    # New approach: Batched stack + transfer (~7s) - 2.25x speedup, 55.6% improvement
    # Note: Full GPU stack (Strategy A) is faster but can cause OOM in multi-GPU mode
    # See stacking_bench/benchmark_results.json for benchmarking results
    
    # Pre-allocate output arrays
    H_out = np.empty([volume_size, n_picked_indices], dtype=experiment_dataset.dtype)
    B_out = np.empty([volume_size, n_picked_indices], dtype=experiment_dataset.dtype)
    
    # Transfer in batches to balance memory and performance.
    # batch_size=50 is tuned for grid_size<=256. For larger grids, scale
    # down to avoid OOM (each element is volume_size * 8 bytes, stacking
    # N needs ~2*N*element_bytes on GPU).
    batch_size = 50
    element_bytes = volume_size * 8  # complex64
    if 2 * batch_size * element_bytes > 0.10 * utils.get_gpu_memory_total() * 1e9:
        gpu_mem_bytes = utils.get_gpu_memory_total() * 1e9
        batch_size = max(1, int(0.10 * gpu_mem_bytes / (2 * element_bytes)))
    with nvtx.annotate("batched_stack_transfer", color="yellow", domain=NVTX_DOMAIN_H_B):
        for batch_start in range(0, n_picked_indices, batch_size):
            batch_end = min(batch_start + batch_size, n_picked_indices)
            
            # Stack batch on GPU
            with nvtx.annotate(f"stack_batch_{batch_start}", color="orange", domain=NVTX_DOMAIN_H_B):
                H_batch_jax = jnp.stack(H[batch_start:batch_end], axis=1)
                B_batch_jax = jnp.stack(B[batch_start:batch_end], axis=1)
            
            # Transfer batch to CPU
            with nvtx.annotate(f"transfer_batch_{batch_start}", color="cyan", domain=NVTX_DOMAIN_H_B):
                H_out[:, batch_start:batch_end] = np.asarray(H_batch_jax)
                B_out[:, batch_start:batch_end] = np.asarray(B_batch_jax)
            
            # Clean up batch
            del H_batch_jax, B_batch_jax
    
    return H_out, B_out


from recovar.core import cubic_interpolation
vmap_calculate_spline_coefficients = jax.vmap(cubic_interpolation.calculate_spline_coefficients, in_axes = 0, out_axes = 0)

@nvtx.annotate("compute_spline_coeffs_in_batch", color="magenta")
def compute_spline_coeffs_in_batch(basis, volume_shape, gpu_memory= None):
    gpu_memory = utils.get_gpu_memory_total() if gpu_memory is None else gpu_memory
    vol_batch_size = utils.get_vol_batch_size(volume_shape[0], gpu_memory=gpu_memory)
    logger.info("memory used = %s, vol_batch_size in compute_spline_coeffs_in_batch %s", gpu_memory, vol_batch_size)
    utils.report_memory_device(logger=logger)
    coeffs = []
    for k in range(0, basis.shape[0], vol_batch_size):
        coeffs.append(np.array(vmap_calculate_spline_coefficients(basis[k:k+vol_batch_size].reshape(-1, *volume_shape))))
    coeffs = np.concatenate(coeffs, axis = 0)
    return coeffs

## REDUCED COVARIANCE COMPUTATION

@nvtx.annotate("compute_projected_covariance", color="green")
def compute_projected_covariance(experiment_datasets, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, parallel_analysis = False, do_mask_images = True, mean_cubic=None):

    experiment_dataset = experiment_datasets[0]

    basis = basis.T.astype(experiment_dataset.dtype)
    basis = jnp.asarray(basis)
    volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    mean_estimate = jnp.array(mean_estimate).astype(experiment_dataset.dtype)
    jax_random_key = jax.random.PRNGKey(0)

    lhs = 0
    rhs = 0
    summed_batch_kron_cpu = jax.jit(summed_batch_kron, backend='cpu')
    logger.info("batch size in compute_projected_covariance %s", batch_size)

    if disc_type == 'cubic':
        if mean_cubic is not None:
            mean_estimate = mean_cubic
        else:
            mean_estimate = cubic_interpolation.calculate_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))

    if disc_type_u == 'cubic':
        basis = compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)

    change_device= False

    for experiment_dataset in experiment_datasets:
        config = ForwardModelConfig.from_dataset(
            experiment_dataset, disc_type=disc_type,
            process_fn=experiment_dataset.image_stack.process_images,
        )
        model = ModelState(
            mean_estimate=mean_estimate,
            volume_mask=volume_mask,
            basis=basis,
        )
        opts = CovarianceOpts(
            disc_type_u=disc_type_u,
            do_mask_images=do_mask_images,
            shared_label=experiment_dataset.tilt_series_flag,
            parallel_analysis=parallel_analysis,
        )

        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
        for batch, _, batch_image_ind in data_generator:
            jax_random_key, subkey = jax.random.split(jax_random_key)
            noise_variances = experiment_dataset.noise.get(batch_image_ind)

            batch_data = BatchData(
                images=batch,
                ctf_params=experiment_dataset.CTF_params[batch_image_ind],
                rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
                translations=experiment_dataset.translations[batch_image_ind],
                noise_variance=noise_variances,
            )
            lhs_this, rhs_this = reduce_covariance_inner(
                config, batch_data, model, opts,
                experiment_dataset.image_stack.mask,
                jax_random_key=subkey,
            )

            lhs += lhs_this
            rhs += rhs_this
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
    
    logger.info("end of covariance computation - before solve")
    utils.report_memory_device(logger=logger)
    rhs = vec(rhs)

    if change_device:
        rhs = jax.device_put(rhs, jax.devices("gpu")[0])
        lhs = jax.device_put(lhs, jax.devices("gpu")[0])

    covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    covar = unvec(covar)
    logger.info("end of solve")

    return covar


@functools.partial(jax.jit, static_argnums = [8,9,10,11,12,13,14,15,17,18, 19,21,22,23])    
@nvtx.annotate("reduce_covariance_est_inner", color="green")
def reduce_covariance_est_inner(batch, mean_estimate, volume_mask, basis, CTF_params,
                                rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape,
                                volume_shape, grid_size, voxel_size, padding, disc_type, 
                                disc_type_u, noise_variance, process_fn, CTF_fun, parallel_analysis = False,
                                jax_random_key = None, do_mask_images = True, shared_label = False, premultiplied_ctf = False):
    

    if (disc_type != 'linear_interp') and (disc_type != 'cubic'):
        logger.warning("USING NEAREST NEIGHBOR DISCRETIZATION IN reduce_covariance_est_inner. disc_type=%s, disc_type_u=%s", disc_type, disc_type_u)

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

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=volume_shape[0], voxel_size=voxel_size,
        padding=padding, disc_type=disc_type, CTF_fun=CTF_fun,
        premultiplied_ctf=premultiplied_ctf, volume_mask_threshold=volume_mask_threshold,
    )

    # P_i mean
    projected_mean = core_forward.forward_model(
        config, mean_estimate, CTF_params, rotation_matrices,
    )

    ## DO MASK BUSINESS HERE.
    if do_mask_images:
        batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
        projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    config_u = config.replace(disc_type=disc_type_u)
    AUs = covariance_core.batch_vol_forward_from_map(
        config_u, basis, CTF_params, rotation_matrices,
        skip_ctf=premultiplied_ctf,
    )
    
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

        AUs = AUs * CTF[...,None] # Then CTF multiply

    else:
        # If we are here, AUs are CTFed, so no need to multiply on images
        batch = batch - projected_mean
        AU_t_images = batch_x_T_y(AUs, batch)
    # This gets inner product of AU with images

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
    lhs = jnp.sum(batch_kron(AU_t_AU, AU_t_AU), axis=0)

    return lhs, rhs


# ============================================================================
# New Equinox-based API for covariance estimation
# ============================================================================


@eqx.filter_jit
@nvtx.annotate("reduce_covariance_inner", color="green")
def reduce_covariance_inner(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    opts: CovarianceOpts,
    image_mask: jax.Array,
    jax_random_key=None,
):
    """Covariance estimation inner loop — Equinox API.

    Replaces the 24-param ``reduce_covariance_est_inner`` with structured inputs.
    All geometry/CTF/discretization params come from ``config``.
    Per-batch arrays come from ``batch_data``.
    Reconstruction state (mean, mask, basis) comes from ``model``.
    Boolean flags come from ``opts``.
    """
    batch = batch_data.images
    ctf_params = batch_data.ctf_params
    rotation_matrices = batch_data.rotation_matrices
    translations = batch_data.translations
    noise_variance = batch_data.noise_variance

    if (config.disc_type != 'linear_interp') and (config.disc_type != 'cubic'):
        logger.warning("USING NEAREST NEIGHBOR DISCRETIZATION. disc_type=%s", config.disc_type)

    do_mask_images = opts.do_mask_images
    if config.premultiplied_ctf and do_mask_images:
        logger.warning('cannot use premultiplied ctf and mask images at the same time.')
        do_mask_images = False

    if do_mask_images:
        image_mask = covariance_core.get_per_image_tight_mask(
            model.volume_mask, rotation_matrices, image_mask,
            config.volume_mask_threshold, config.image_shape, config.volume_shape,
            config.grid_size, config.padding, 'linear_interp',
        )

    else:
        image_mask = jnp.ones_like(batch).real

    if config.process_fn is not None:
        batch = config.process_fn(batch)
    batch = core.translate_images(batch, translations, config.image_shape)

    projected_mean = core_forward.forward_model(
        config, model.mean_estimate, ctf_params, rotation_matrices,
    )

    if do_mask_images:
        batch = covariance_core.apply_image_masks(batch, image_mask, config.image_shape)
        projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, config.image_shape)

    # Forward model basis vectors — use disc_type_u for eigenvectors
    config_u = config.replace(disc_type=opts.disc_type_u)
    AUs = covariance_core.batch_vol_forward_from_map(
        config_u, model.basis, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf,
    )

    if do_mask_images:
        AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, config.image_shape)
    AUs = AUs.transpose(1, 2, 0)

    if config.premultiplied_ctf:
        CTF = config.compute_ctf(ctf_params)
        batch = batch - projected_mean * CTF
        AU_t_images = batch_x_T_y(AUs, batch)
        AUs = AUs * CTF[..., None]
    else:
        batch = batch - projected_mean
        AU_t_images = batch_x_T_y(AUs, batch)

    if do_mask_images:
        assert not config.premultiplied_ctf, "Not implemented yet"

    AU_t_AU = batch_x_T_y(AUs, AUs).real.astype(ctf_params.dtype)

    AUs *= jnp.sqrt(noise_variance)[..., None]
    UALambdaAUs = jnp.sum(batch_x_T_y(AUs, AUs), axis=0)

    if opts.shared_label:
        AU_t_images = jnp.sum(AU_t_images, axis=0, keepdims=True)
        AU_t_AU = jnp.sum(AU_t_AU, axis=0, keepdims=True)

    outer_products = summed_outer_products(AU_t_images)
    rhs = outer_products - UALambdaAUs
    rhs = rhs.real.astype(ctf_params.dtype)

    lhs = jnp.sum(batch_kron(AU_t_AU, AU_t_AU), axis=0)
    return lhs, rhs


batch_kron = jax.vmap(jnp.kron, in_axes=(0,0))

@nvtx.annotate("summed_batch_kron", color="gray")
def summed_batch_kron(X):
    return jnp.sum(batch_kron(X,X), axis=0)

@nvtx.annotate("summed_batch_kron_scan", color="gray")
def summed_batch_kron_scan(X):
    init = jnp.zeros((X.shape[1] * X.shape[1],), dtype=X.dtype)
    def fori_loop_body(i, val):
        return val + jnp.kron(X[i], X[i])
    summed_kron = jax.lax.fori_loop(0, X.shape[0], fori_loop_body, init)
    return summed_kron


batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))

@nvtx.annotate("summed_outer_products", color="gray")
def summed_outer_products(AU_t_images):
    # Not .H because things are already transposed technically
    return AU_t_images.T @ jnp.conj(AU_t_images)

batched_summed_outer_products  = jax.vmap(summed_outer_products)


@functools.partial(jax.jit, static_argnums=[2])
@nvtx.annotate("group_sum_by_labels", color="brown")
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
@nvtx.annotate("compute_H_B_triangular", color="blue", domain=NVTX_DOMAIN_H_B)
def compute_H_B_triangular(centered_images, CTF_val_on_grid_stacked, plane_coords_on_grid_stacked, rotation_matrices,  noise_variances, picked_freq_index, image_mask, image_shape, volume_size, right_kernel = "triangular", left_kernel = "triangular", kernel_width = 2, shared_label = False, premultiplied_ctf = False, tilt_labels = None):

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
    # Noise subtraction term (may need revisiting for cryo-ET)
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


@nvtx.annotate("adjoint_kernel_slice", color="blue", domain=NVTX_DOMAIN_H_B)
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

@nvtx.annotate("compute_noise_term", color="blue", domain=NVTX_DOMAIN_H_B)
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


@nvtx.annotate("preprocess_tilt_labels_for_batch", color="brown")
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
