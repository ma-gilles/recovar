"""Regularized covariance matrix estimation from half-set cryo-EM data."""

import functools
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils, jax_config
from recovar.core import cubic_interpolation
from recovar.core.configs import ForwardModelConfig, BatchData, DataIterator, ModelState, CovarianceOpts, CovColumnOpts
from recovar.heterogeneity import covariance_core
from recovar.reconstruction import regularization, relion_functions, noise

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
except (ImportError, OSError, AttributeError) as e:
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
        Dictionary with keys ``reg_fn``, ``left_kernel``,
        ``right_kernel``, ``column_sampling_scheme``,
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
        "reg_fn": "new",
        "left_kernel": "triangular",
        "right_kernel": "triangular",
        "left_kernel_width": 1,
        "right_kernel_width": 2,
        "shift_fsc": False,
        "substract_shell_mean": False,
        "grid_correct": True,
        "use_spherical_mask": True,
        "use_mask_in_fsc": True,
        "column_sampling_scheme": 'high_snr_from_var_est',
        "column_radius": 5,
        "use_combined_mean": True,
        "sampling_avoid_in_radius": 2,
        "sampling_n_cols": 300,
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
    running_vec = sampling_vec.astype(np.float64)

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
def compute_regularized_covariance_columns(cryos, means, mean_prior, volume_mask, dilated_volume_mask, valid_idx, gpu_memory, options, picked_frequencies, use_multi_gpu=False, n_gpus=None, mean_cubic=None):

    volume_shape = cryos.volume_shape
    mask_final = volume_mask

    utils.report_memory_device(logger=logger)

    # Start CUDA profiler for covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStart()
        logger.info("CUDA Profiler: Started profiling covariance computation")

    Hs, Bs = compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies,
                               gpu_memory, options=options, use_multi_gpu=use_multi_gpu,
                               n_gpus=n_gpus, mean_cubic=mean_cubic)

    # Stop CUDA profiler after covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStop()
        logger.info("CUDA Profiler: Stopped profiling covariance computation")

    volume_noise_var = np.asarray(noise.make_radial_noise(
        cryos[0].noise.get_average_radial_noise(), cryos.volume_shape))

    covariance_cols = {}
    utils.report_memory_device(logger=logger)
    covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(
        Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final,
        volume_shape, gpu_memory, reg_init_multiplier=jax_config.REG_INIT_MULTIPLIER,
        options=options)
    covariance_cols["est_mask"] = covariance_cols["est_mask"].T
    del Hs, Bs
    logger.info("after reg fn")
    utils.report_memory_device(logger=logger)

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
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    Ft_im: jax.Array = None,
    Ft_one: jax.Array = None,
):
    """Variance estimation via RELION-style trilinear kernel — Equinox API.

    Backprojects into half-volume accumulators (Ft_y, Ft_ctf, Ft_im, Ft_one).
    Pass None to initialise from zero; pass an existing half-volume to accumulate.
    """
    rotation_matrices = batch_data.rotation_matrices
    noise_variances = batch_data.noise_variance

    # batch_data.images is already in half-image (rfft-packed) format.
    half_images = core.translate_half_images(batch_data.images, batch_data.translations, config.image_shape)
    half_ctf = config.compute_ctf_half(batch_data.ctf_params)
    CTF_squared = half_ctf ** 2

    mean_slice_half = core.slice_volume_by_map_to_half_image(
        mean_estimate, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
    )

    if config.premultiplied_ctf:
        half_images = half_images - mean_slice_half * CTF_squared
        # CTF² needed in full space for noise scaling; computed from scratch for this case only.
        noise_variances = noise_variances * config.compute_ctf(batch_data.ctf_params) ** 2
    else:
        half_images = half_images - mean_slice_half * half_ctf

    img_power_half = jnp.abs(half_images) ** 2
    noise_p_variance_ctf = CTF_squared if config.premultiplied_ctf else jnp.ones_like(half_images)
    cov_noise_half = jnp.zeros_like(img_power_half)

    if volume_mask is not None:
        image_mask = covariance_core.get_per_image_tight_mask(
            volume_mask, rotation_matrices, image_mask, config.volume_mask_threshold,
            config.image_shape, config.volume_shape, config.grid_size, config.padding,
            'linear_interp', soften=soften,
        )
        # apply_image_masks with half_images=True: IRFFT2 → real-space mask → RFFT2.
        half_images = covariance_core.apply_image_masks(
            half_images, image_mask, config.image_shape, half_images=True
        )
        cov_noise_half = fourier_transform_utils.full_image_to_half_image(
            noise.get_masked_noise_variance_from_noise_variance(
                image_mask, noise_variances, config.image_shape
            ).reshape(-1, config.image_size),
            config.image_shape,
        )

    images_squared = jnp.abs(half_images) ** 2 - cov_noise_half
    if not config.premultiplied_ctf:
        images_squared = images_squared * CTF_squared

    def _backproject(half_imgs, volume):
        return core.adjoint_slice_volume_by_trilinear_from_half_images(
            half_imgs, rotation_matrices, config.image_shape, config.volume_shape,
            volume=volume, half_volume=True,
        )

    Ft_y = _backproject(images_squared, Ft_y)
    Ft_ctf = _backproject(CTF_squared ** 2, Ft_ctf)
    Ft_im = _backproject(img_power_half, Ft_im)
    Ft_one = _backproject(noise_p_variance_ctf, Ft_one)

    return Ft_y, Ft_ctf, Ft_im, Ft_one


@nvtx.annotate("variance_relion_style_triangular_kernel", color="yellow")
def variance_relion_style_triangular_kernel(experiment_dataset, mean_estimate, batch_size, image_subset=None, volume_mask=None, disc_type=''):
    # Variance uses upsampled_volume_shape (not regular volume_shape).
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

    # Full-spectrum noise needed for masked noise variance inside the kernel.
    Ft_y, Ft_ctf, Ft_im, Ft_one = None, None, None, None
    for batch_data in DataIterator(
        experiment_dataset, batch_size,
        noise_model=experiment_dataset.noise,
        noise_half=False,
        apply_process_images=True,
        half_images=True,
        index_subset=image_subset,
    ):
        Ft_y, Ft_ctf, Ft_im, Ft_one = variance_relion_kernel_trilinear(
            config, batch_data, mean_estimate, volume_mask,
            experiment_dataset.image_stack.mask, soften=5,
            Ft_y=Ft_y, Ft_ctf=Ft_ctf, Ft_im=Ft_im, Ft_one=Ft_one,
        )

    if Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(Ft_y, config.volume_shape)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, config.volume_shape)
        Ft_im = fourier_transform_utils.half_volume_to_full_volume(Ft_im, config.volume_shape)
        Ft_one = fourier_transform_utils.half_volume_to_full_volume(Ft_one, config.volume_shape)

    return Ft_ctf, Ft_y, Ft_one, Ft_im


def _safe_div(numerator, denominator, threshold=1e-20):
    """Element-wise division, returning 0 where denominator is below threshold."""
    safe_denom = jnp.where(denominator > threshold, denominator, jnp.float32(1.0))
    return jnp.where(denominator > threshold, numerator / safe_denom, jnp.float32(0.0))


@nvtx.annotate("compute_variance", color="yellow")
def compute_variance(
    cryos,
    mean_estimate,
    batch_size,
    volume_mask,
    image_subset=None,
    use_regularization=False,
    disc_type='',
    noise_ind_subset=None,
    mean_cubic=None,
):
    st = time.time()

    if disc_type == 'cubic':
        mean_estimate = (
            mean_cubic if mean_cubic is not None
            else cubic_interpolation.calculate_spline_coefficients(
                mean_estimate.reshape(cryos.volume_shape)
            )
        )

    # Run variance kernel for each half-set.
    # variance_relion_style_triangular_kernel returns (Ft_ctf, Ft_y, Ft_one, Ft_im):
    #   ctf_w   — CTF^4 accumulator (Wiener denominator)
    #   signal  — residual^2·CTF^2 accumulator (Wiener numerator)
    #   noise_w — noise-normalisation denominator
    #   noise_s — |residuals|^2 accumulator (noise-normalisation numerator)
    ctf_w, signal, noise_w, noise_s = [], [], [], []
    for cryo in cryos:
        subset = (
            np.where(cryo.noise.dose_indices == noise_ind_subset)[0]
            if noise_ind_subset is not None else image_subset
        )
        fw, sig, nw, ns = variance_relion_style_triangular_kernel(
            cryo, mean_estimate, batch_size,
            image_subset=subset, volume_mask=volume_mask, disc_type=disc_type,
        )
        ctf_w.append(relion_functions.adjust_regularization_relion_style(fw, cryos.volume_shape))
        signal.append(sig)
        noise_w.append(nw)
        noise_s.append(ns)

    variance = {f"corrected{i}": _safe_div(signal[i], ctf_w[i]) for i in range(2)}

    lhs = (ctf_w[0] + ctf_w[1]) / 2
    variance_prior, fsc, _ = regularization.compute_fsc_prior_gpu_v2(
        cryos.volume_shape,
        variance["corrected0"], variance["corrected1"],
        lhs,
        jnp.ones(cryos.volume_size, dtype=cryos.dtype_real) * np.inf,
        frequency_shift=jnp.array([0, 0, 0]),
        upsampling_factor=1,
        substract_shell_mean=True,
    )

    if use_regularization:
        for i in range(2):
            reg_lhs = relion_functions.adjust_regularization_relion_style(
                ctf_w[i], cryos.volume_shape, tau=variance_prior,
            )
            variance[f"corrected{i}"] = _safe_div(signal[i], reg_lhs)

    variance["combined"] = (variance["corrected0"] + variance["corrected1"]) / 2
    variance["prior"] = np.array(variance_prior)
    variance["lhs"] = lhs
    variance = {k: np.array(v).real for k, v in variance.items()}

    noise_p_variance_est = _safe_div(noise_s[0] + noise_s[1], noise_w[0] + noise_w[1])

    logger.info("time to compute variance: %.1fs", time.time() - st)
    return (
        variance,
        np.array(variance_prior).real,
        np.array(fsc).real,
        np.array(lhs).real,
        np.array(noise_p_variance_est).real,
    )


@nvtx.annotate("compute_both_H_B", color="blue")
def compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies,
                     gpu_memory, options, use_multi_gpu=False, n_gpus=None,
                     mean_cubic=None):
    """Compute H and B matrices for both half-sets."""
    Hs = []
    Bs = []
    st_time = time.time()

    for cryo_idx, cryo in enumerate(cryos):
        mean = means["combined"] if options["use_combined_mean"] else means["corrected" + str(cryo_idx)]
        if use_multi_gpu:
            H, B = _compute_H_B_multi_gpu(
                cryo, mean, dilated_volume_mask, picked_frequencies,
                gpu_memory, options, n_gpus=n_gpus, mean_cubic=mean_cubic)
        else:
            H, B = compute_H_B_for_halfset(
                cryo, mean, dilated_volume_mask, picked_frequencies,
                gpu_memory, options, mean_cubic=mean_cubic)
        logger.info("Time to cov %s", time.time() - st_time)
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


def _compute_H_B_multi_gpu(cryo, mean, dilated_volume_mask, picked_frequencies,
                            gpu_memory, options, n_gpus=None, mean_cubic=None):
    """Multi-GPU wrapper: splits images across GPUs, each calls compute_H_B_for_halfset."""
    from recovar.utils import multi_gpu as multi_gpu_utils

    logger.info("=" * 60)
    logger.info("MULTI-GPU MODE ENABLED")
    logger.info("=" * 60)

    def _single_gpu_wrapper(experiment_dataset, mean_estimate, volume_mask,
                            picked_frequency_indices, batch_size, options,
                            image_subset=None):
        return compute_H_B_for_halfset(
            experiment_dataset, mean_estimate, volume_mask,
            picked_frequency_indices, gpu_memory, options,
            image_subset=image_subset, mean_cubic=mean_cubic)

    H, B = multi_gpu_utils.compute_H_B_multi_gpu(
        compute_H_B_fn=_single_gpu_wrapper,
        experiment_dataset=cryo,
        n_gpus=n_gpus,
        mean_estimate=mean,
        volume_mask=dilated_volume_mask,
        picked_frequency_indices=picked_frequencies,
        batch_size=0,  # unused — compute_H_B_for_halfset sizes its own batches
        options=options,
    )

    logger.info("=" * 60)
    logger.info("MULTI-GPU COMPUTATION COMPLETED")
    logger.info("=" * 60)

    return H, B


def preprocess_covariance_batch(config, batch_data, mean_estimate, volume_mask,
                                image_stack_mask, opts):
    """Preprocess one image batch for covariance column computation.

    Applies tight mask, centers images (y_i - A_i*mu), masks, computes CTF,
    and gets grid-point coordinates.  Each sub-step calls existing JIT-compiled
    helpers; this function is a Python-level orchestrator (not itself jitted).

    Returns
    -------
    tuple of (centered_images, ctf_on_grid, plane_coords, image_mask, tilt_labels)
    """
    images = batch_data.images
    rotation_matrices = batch_data.rotation_matrices
    ctf_params = batch_data.ctf_params

    # 1. Per-image tight mask
    image_mask = covariance_core.get_per_image_tight_mask(
        volume_mask, rotation_matrices, image_stack_mask,
        config.volume_mask_threshold,
        config.image_shape, config.volume_shape,
        config.grid_size, config.padding,
        'linear_interp', soften=opts.soften_mask)
    if not opts.mask_images:
        image_mask = jnp.ones_like(image_mask)

    # 2. Center images: y_i - A_i * mu
    images = covariance_core.centered_images(
        config, images, mean_estimate,
        ctf_params, rotation_matrices,
        batch_data.translations)

    # 3. Apply image masks
    images = covariance_core.apply_image_masks(images, image_mask, config.image_shape)

    # 4. Compute CTF
    ctf_on_grid = config.compute_ctf(ctf_params)

    # 5. Grid-point coordinates
    plane_coords = core.batch_get_gridpoint_coords(
        rotation_matrices, config.image_shape, config.volume_shape)

    # 6. Tilt labels (from particle_indices)
    tilt_labels = batch_data.particle_indices

    return images, ctf_on_grid, plane_coords, image_mask, tilt_labels


def _batched_stack_transfer(H, B, H_out, B_out, freq_offset, volume_size, n_items):
    """Stack JAX arrays in GPU batches and transfer to pre-allocated CPU arrays.

    Avoids OOM from stacking all columns at once and minimizes the number
    of individual DtoH transfers.
    """
    batch_size = 50
    element_bytes = volume_size * 8  # complex64
    gpu_mem_total = utils.get_gpu_memory_total()
    if 2 * batch_size * element_bytes > 0.10 * gpu_mem_total * 1e9:
        gpu_mem_bytes = gpu_mem_total * 1e9
        batch_size = max(1, int(0.10 * gpu_mem_bytes / (2 * element_bytes)))

    for batch_start in range(0, n_items, batch_size):
        batch_end = min(batch_start + batch_size, n_items)
        H_batch_jax = jnp.stack(H[batch_start:batch_end], axis=1)
        B_batch_jax = jnp.stack(B[batch_start:batch_end], axis=1)
        col_start = freq_offset + batch_start
        col_end = freq_offset + batch_end
        H_out[:, col_start:col_end] = np.asarray(H_batch_jax)
        B_out[:, col_start:col_end] = np.asarray(B_batch_jax)
        del H_batch_jax, B_batch_jax


@nvtx.annotate("compute_H_B_for_halfset", color="blue", domain=NVTX_DOMAIN_H_B)
def compute_H_B_for_halfset(cryo, mean_estimate, volume_mask, picked_frequencies,
                            gpu_memory, options, image_subset=None, mean_cubic=None):
    """Compute H and B matrices for one half-set.

    Replaces the old ``compute_H_B`` + ``compute_H_B_in_volume_batch`` pair.
    Uses :class:`DataIterator` for data loading and :class:`CovColumnOpts`
    for static kernel options.
    """
    volume_size = cryo.volume_size

    # Resolve discretization and mean
    if options['disc_type'] == 'cubic':
        disc_type = 'cubic'
        if mean_cubic is not None:
            mean_estimate = mean_cubic
        else:
            mean_estimate = cubic_interpolation.calculate_spline_coefficients(
                jnp.array(mean_estimate).reshape(cryo.volume_shape))
    else:
        disc_type = 'linear_interp'

    mean_estimate = jnp.array(mean_estimate)

    config = ForwardModelConfig.from_dataset(cryo, disc_type=disc_type)
    opts = CovColumnOpts(
        right_kernel=options["right_kernel"],
        left_kernel=options["left_kernel"],
        right_kernel_width=options["right_kernel_width"],
        mask_images=options["mask_images_in_H_B"],
        soften_mask=3,
    )

    # Batch sizes
    image_batch_size = utils.safe_batch_size(
        utils.get_image_batch_size(cryo.grid_size, gpu_memory) // (2 if disc_type == 'cubic' else 1))
    column_batch_size = utils.get_column_batch_size(cryo.grid_size, gpu_memory)

    n_picked = picked_frequencies.size
    H_out = np.empty([volume_size, n_picked], dtype=cryo.dtype)
    B_out = np.empty([volume_size, n_picked], dtype=cryo.dtype)

    # Outer loop: frequency batches (for GPU memory)
    for freq_k in range(0, int(np.ceil(n_picked / column_batch_size))):
        freq_st = int(freq_k * column_batch_size)
        freq_end = int(np.min([(freq_k + 1) * column_batch_size, n_picked]))
        freq_batch = picked_frequencies[freq_st:freq_end]
        n_freq_batch = freq_batch.size

        H = [0] * n_freq_batch
        B = [0] * n_freq_batch

        # Inner loop: image batches via DataIterator
        for batch_data in DataIterator(
                cryo, image_batch_size,
                noise_model=cryo.noise, noise_half=False,
                noise_by_particle=True,
                index_subset=image_subset,
                use_image_generator=image_subset is not None,
                apply_process_images=True):

            images, ctf_on_grid, plane_coords, image_mask, tilt_labels = \
                preprocess_covariance_batch(
                    config, batch_data, mean_estimate, volume_mask,
                    cryo.image_stack.mask, opts)

            # Per-frequency accumulation
            for k, freq_idx in enumerate(freq_batch):
                H_k, B_k = compute_H_B_triangular(
                    images, ctf_on_grid, plane_coords,
                    batch_data.rotation_matrices,
                    batch_data.noise_variance,
                    freq_idx, image_mask,
                    config.image_shape, volume_size,
                    right_kernel=opts.right_kernel,
                    left_kernel=opts.left_kernel,
                    kernel_width=opts.right_kernel_width,
                    shared_label=cryo.tilt_series_flag,
                    premultiplied_ctf=cryo.premultiplied_ctf,
                    tilt_labels=tilt_labels)
                H[k] += H_k.real.astype(cryo.dtype_real)
                B[k] += B_k

            del images, ctf_on_grid, plane_coords, image_mask

        # Batched stack + GPU→CPU transfer
        _batched_stack_transfer(H, B, H_out, B_out, freq_st, volume_size, n_freq_batch)

    return H_out, B_out


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
    cpu_device = jax.devices("cpu")[0]

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

        priors = jax.device_put(priors, cpu_device)
        for j, ind in enumerate(indices):
            if options["prior_n_iterations"] >= 0:
                fsc_priors[ind] = np.asarray(priors[j].real)
            fscs[ind] = np.asarray(fscs_this[j])
            combined_cov_cols[ind] = np.asarray(combined_cov_col[j])
        del combined_cov_col, priors

    if options["prior_n_iterations"] >= 0:
        fsc_priors = np.stack(fsc_priors, axis =0).real

    combined_cov_cols = np.stack(combined_cov_cols, axis =0)

    return combined_cov_cols, fsc_priors, fscs

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
def compute_projected_covariance(experiment_datasets, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, do_mask_images=True, mean_cubic=None):

    experiment_dataset = experiment_datasets[0]

    basis = basis.T.astype(experiment_dataset.dtype)
    basis = jnp.asarray(basis)
    volume_mask = jnp.asarray(volume_mask, dtype=experiment_dataset.dtype_real)
    mean_estimate = jnp.asarray(mean_estimate, dtype=experiment_dataset.dtype)

    lhs = 0
    rhs = 0
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
        )

        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
        for batch, _, batch_image_ind in data_generator:
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
            )
            if not jnp.all(jnp.isfinite(lhs_this)) or not jnp.all(jnp.isfinite(rhs_this)):
                logger.error("NaN/Inf in covariance batch (images %s); zeroing batch contribution", batch_image_ind[:5])
                lhs_this = jnp.where(jnp.isfinite(lhs_this), lhs_this, 0.0)
                rhs_this = jnp.where(jnp.isfinite(rhs_this), rhs_this, 0.0)

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

    # Tikhonov regularization: prevents NaN from near-singular LHS
    # (can happen when n_images is small relative to basis_size)
    trace_val = jnp.trace(lhs)
    # Guard against zero/NaN trace (e.g., all batches produced NaN)
    trace_val = jnp.where(jnp.isfinite(trace_val) & (trace_val > 0), trace_val, jnp.float32(1.0))
    reg = jnp.float32(1e-6) * trace_val / lhs.shape[0]
    diag_idx = jnp.arange(lhs.shape[0])
    lhs = lhs.at[diag_idx, diag_idx].add(reg)

    covar = jax.scipy.linalg.solve( lhs ,rhs, assume_a='pos')
    covar = unvec(covar)
    logger.info("end of solve")

    return covar


# ============================================================================
# Equinox-based API for covariance estimation
# ============================================================================


@eqx.filter_jit
@nvtx.annotate("reduce_covariance_inner", color="green")
def reduce_covariance_inner(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    opts: CovarianceOpts,
    image_mask: jax.Array,
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
        if config.premultiplied_ctf:
            raise NotImplementedError("Masking with premultiplied CTF is not implemented yet")

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
@functools.partial(jax.jit, static_argnums=[7, 8, 9, 10, 11, 12, 13])
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
