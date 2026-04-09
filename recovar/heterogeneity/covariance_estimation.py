"""Regularized covariance matrix estimation from half-set cryo-EM data."""

import functools
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config, utils
from recovar.core import cubic_interpolation, linalg
from recovar.core.configs import CovarianceOpts, CovColumnOpts, ForwardModelConfig, ModelState
from recovar.heterogeneity import covariance_core
from recovar.jax_config import _to_cpu
from recovar.reconstruction import noise, regularization, relion_functions
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)


def _load_cuda_profiler():
    """Best-effort CUDA profiler loader used by performance debugging."""
    try:
        import ctypes

        cudart = ctypes.CDLL("libcudart.so")

        def _cuda_profiler_start():
            ret = cudart.cudaProfilerStart()
            if ret != 0:
                logger.warning("cudaProfilerStart returned error code: %s", ret)

        def _cuda_profiler_stop():
            ret = cudart.cudaProfilerStop()
            if ret != 0:
                logger.warning("cudaProfilerStop returned error code: %s", ret)

        return True, _cuda_profiler_start, _cuda_profiler_stop
    except (ImportError, OSError, AttributeError) as e:
        logger.warning("CUDA profiler not available - profiling disabled: %s", e)
        return False, None, None


# CUDA Profiler for selective profiling with nsys
# Uses ctypes to call CUDA runtime API directly (no PyTorch needed)
CUDA_PROFILER_AVAILABLE, cudaProfilerStart, cudaProfilerStop = _load_cuda_profiler()

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
# - compute_freq_batch (batched frequency computation via fori_loop)
# - adjoint_kernel_slice (backprojection)
# - get_per_image_tight_mask, process_and_center_images, compute_CTF, etc.
#
# ============================================================================

# Domain name for compute_H_B profiling (use as string in decorators)
NVTX_DOMAIN_H_B = "compute_H_B"


@nvtx.annotate("get_default_covariance_computation_options", color="red")
def get_default_covariance_computation_options(grid_size=None, adaptive_n_pcs=False):
    """Return default options dict for covariance computation.

    Uses a fixed number of principal components (200) by default for
    reproducibility across different GPU configurations. If
    ``adaptive_n_pcs=True``, the number of PCs is reduced to fit in
    available GPU memory (useful for smaller GPUs, but results will
    depend on hardware).

    Args:
        grid_size: Side length of the 3-D reconstruction grid. Used for
            memory estimation and adaptive PC count.
        adaptive_n_pcs: If True, reduce the number of PCs to fit in GPU
            memory. Default False (always use 200 PCs).

    Returns:
        Dictionary with keys ``reg_fn``, ``left_kernel``,
        ``right_kernel``, ``column_sampling_scheme``,
        ``n_pcs_to_compute``, among others.
    """

    gpu_memory = utils.get_gpu_memory_total()

    if adaptive_n_pcs and grid_size is not None:
        volume_size = grid_size**3
        dtype_size = 8  # bytes for complex64
        available_memory_gb = gpu_memory * 0.7
        base_memory_coefficient = 75 / (200**4)
        basis_memory_coefficient = volume_size * dtype_size / 1e9

        n_pcs = 200
        for n_pcs in range(200, 0, -1):
            base_memory = base_memory_coefficient * (n_pcs**4)
            basis_memory = basis_memory_coefficient * n_pcs
            if base_memory + basis_memory <= available_memory_gb:
                break
        else:
            n_pcs = 50

        logger.info(
            "Adaptive n_pcs: using %s PCs for covariance computation "
            "(GPU memory: %.1f GB, grid_size: %s, estimated memory: %.1f GB)",
            n_pcs,
            gpu_memory,
            grid_size,
            base_memory + basis_memory,
        )
    else:
        n_pcs = 200

        # Estimate memory usage so we can warn if it might OOM
        if grid_size is not None:
            volume_size = grid_size**3
            dtype_size = 8
            base_memory_gb = (75 / (200**4)) * (n_pcs**4)
            basis_memory_gb = volume_size * dtype_size * n_pcs / 1e9
            total_memory_gb = base_memory_gb + basis_memory_gb
            logger.info(
                "Using %s PCs for covariance computation "
                "(GPU memory: %.1f GB, grid_size: %s, estimated memory: %.1f GB)",
                n_pcs,
                gpu_memory,
                grid_size,
                total_memory_gb,
            )
            if total_memory_gb > gpu_memory * 0.8:
                logger.warning(
                    "Estimated memory (%.1f GB) may exceed available GPU memory (%.1f GB). "
                    "If you get OOM errors, use --low-memory-option to adaptively reduce n_pcs.",
                    total_memory_gb,
                    gpu_memory,
                )
        else:
            logger.info("Using %s PCs for covariance computation (GPU memory: %.1f GB)", n_pcs, gpu_memory)

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
        "column_sampling_scheme": "high_snr_from_var_est",
        "column_radius": 5,
        "use_combined_mean": True,
        "sampling_avoid_in_radius": 2,
        "sampling_n_cols": 300,
        "n_pcs_to_compute": n_pcs,
        "randomized_sketch_size": 300,
        "prior_n_iterations": 20,
        "randomize_column_sampling": False,
        "disc_type": "cubic",
        "disc_type_u": "linear_interp",
        "mask_images_in_proj": False,
        "mask_images_in_H_B": True,
        "downsample_from_fsc": False,
    }

    return options


@nvtx.annotate("set_covariance_options", color="red")
def set_covariance_options(args, options):
    for key in options:
        if key in args:
            options[key] = args[key]
    return options


@nvtx.annotate("greedy_column_choice", color="orange")
def greedy_column_choice(sampling_vec, n_samples, volume_shape, avoid_in_radius=1, keep_only_below_freq=32):
    if avoid_in_radius < 0 or avoid_in_radius > 20:
        raise ValueError("avoid_in_radius should be between 0 and 20")

    if n_samples < 1 or n_samples > sampling_vec.size:
        raise ValueError("n_samples should be between 1 and the size of sampling_vec")

    radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape)
    sampling_vec *= radial_distances.reshape(-1) < keep_only_below_freq

    sorted_idx = np.asarray(jnp.argsort(-sampling_vec))
    picked_set = set()
    picked = []
    n_picked = 0
    sorted_frequencies = core.vec_indices_to_frequencies(sorted_idx, volume_shape)
    sorted_frequencies_norm = np.linalg.norm(sorted_frequencies, axis=-1)

    for idx in sorted_idx:
        if idx not in picked_set:
            picked_frequency = core.vec_indices_to_frequencies(idx[None], volume_shape)
            # take things only on the + x side
            if picked_frequency[0, 0] < 0:
                picked_frequency = -picked_frequency  # Take the complex conjugate instead
                idx = int(core.frequencies_to_vec_indices(picked_frequency, volume_shape)[0])
                if idx in picked_set:
                    continue
            picked.append(idx)
            n_picked += 1
            if n_picked >= n_samples:
                break
            picked_set.add(idx)
            # Now take out everything that is close by and their complex conjugates

            nearby_freqs = core.find_frequencies_within_grid_dist(
                picked_frequency, np.ceil(avoid_in_radius).astype(int)
            )
            nearby_freqs = np.asarray(nearby_freqs)
            nearby_freqs = nearby_freqs[np.linalg.norm(nearby_freqs - picked_frequency, axis=-1) <= avoid_in_radius]
            nearby_freqs_negative = -nearby_freqs
            nearby_vec_indices = np.asarray(core.frequencies_to_vec_indices(nearby_freqs, volume_shape))
            nearby_negative_vec_indices = np.asarray(
                core.frequencies_to_vec_indices(nearby_freqs_negative, volume_shape)
            )
            for k in range(nearby_vec_indices.size):
                picked_set.add(nearby_vec_indices[k])
                picked_set.add(nearby_negative_vec_indices[k])

    picked_frequencies = core.vec_indices_to_frequencies(np.asarray(picked), volume_shape)

    return np.asarray(picked), np.asarray(picked_frequencies)


@nvtx.annotate("randomized_column_choice", color="orange")
def randomized_column_choice(
    sampling_vec,
    n_samples,
    volume_shape,
    avoid_in_radius=1,
    random_seed=None,
):
    if avoid_in_radius < 0 or avoid_in_radius > 20:
        raise ValueError("avoid_in_radius should be between 0 and 20")

    if n_samples < 1 or n_samples > sampling_vec.size:
        raise ValueError("n_samples should be between 1 and the size of sampling_vec")

    sorted_idx = np.asarray(jnp.argsort(-sampling_vec))
    picked_set = set()
    picked = []
    n_picked = 0
    sorted_frequencies = core.vec_indices_to_frequencies(sorted_idx, volume_shape)
    sorted_frequencies_norm = np.linalg.norm(sorted_frequencies, axis=-1)
    running_vec = np.asarray(sampling_vec).astype(np.float64)
    probs = running_vec / np.sum(running_vec)
    draw_size = min(running_vec.size, n_samples * 100)
    if random_seed is None:
        draw = np.random.choice
    else:
        rng = np.random.default_rng(random_seed)
        draw = rng.choice
    random_choices = draw(running_vec.size, size=draw_size, p=probs, replace=False)
    test_idx = 0

    while n_picked < n_samples:
        if test_idx >= random_choices.size:
            random_choices = draw(running_vec.size, size=draw_size, p=probs, replace=False)
            test_idx = 0

        idx = random_choices[test_idx]
        picked_frequency = core.vec_indices_to_frequencies(idx[None], volume_shape)
        # take things only on the + x side
        if picked_frequency[0, 0] < 0:
            picked_frequency = -picked_frequency  # Take the complex conjugate instead
            idx = int(core.frequencies_to_vec_indices(picked_frequency, volume_shape)[0])

        test_idx += 1
        if idx in picked_set:
            continue

        picked.append(idx)
        n_picked += 1
        if n_picked >= n_samples:
            break
        picked_set.add(idx)

        nearby_freqs = core.find_frequencies_within_grid_dist(picked_frequency, np.ceil(avoid_in_radius).astype(int))
        nearby_freqs = np.asarray(nearby_freqs)
        nearby_freqs = nearby_freqs[np.linalg.norm(nearby_freqs - picked_frequency, axis=-1) <= avoid_in_radius]
        nearby_freqs_negative = -nearby_freqs
        nearby_vec_indices = np.asarray(core.frequencies_to_vec_indices(nearby_freqs, volume_shape))
        nearby_negative_vec_indices = np.asarray(core.frequencies_to_vec_indices(nearby_freqs_negative, volume_shape))
        for k in range(nearby_vec_indices.size):
            picked_set.add(nearby_vec_indices[k])
            picked_set.add(nearby_negative_vec_indices[k])

    picked_frequencies = core.vec_indices_to_frequencies(np.asarray(picked), volume_shape)

    return np.asarray(picked), np.asarray(picked_frequencies)


@nvtx.annotate("compute_regularized_covariance_columns_in_batch", color="purple")
def compute_regularized_covariance_columns_in_batch(
    dataset,
    means,
    mean_prior,
    volume_mask,
    dilated_volume_mask,
    valid_idx,
    gpu_memory,
    options,
    picked_frequencies,
    use_multi_gpu=False,
    n_gpus=None,
):
    """Compute regularized covariance matrix columns in GPU-sized batches.

    Iterates over *picked_frequencies* in batches that fit in GPU memory
    and concatenates the results.

    Args:
        dataset: A ``CryoEMDataset`` with ``halfset_indices`` set.
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

    Returns:
        Tuple ``(covariance_cols, picked_frequencies, fscs)`` where
        *covariance_cols* is a dict with key ``'est_mask'`` and *fscs*
        contains per-column FSC curves.
    """
    frequency_batch = utils.get_column_batch_size(dataset.grid_size, gpu_memory)

    covariance_cols = []
    fscs = []
    for k in range(0, int(np.ceil(picked_frequencies.size / frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min([(k + 1) * frequency_batch, picked_frequencies.size]))

        covariance_cols_b, _, fscs_b = compute_regularized_covariance_columns(
            dataset,
            means,
            mean_prior,
            volume_mask,
            dilated_volume_mask,
            valid_idx,
            gpu_memory,
            options,
            picked_frequencies[batch_st:batch_end],
            use_multi_gpu=use_multi_gpu,
            n_gpus=n_gpus,
        )
        logger.info("batch of col done: %s, %s", batch_st, batch_end)

        covariance_cols.append(covariance_cols_b["est_mask"])
        fscs.append(fscs_b)

    covariance_cols = {"est_mask": np.concatenate(covariance_cols, axis=-1)}
    fscs = np.concatenate(fscs, axis=0)
    return covariance_cols, picked_frequencies, fscs


@nvtx.annotate("compute_regularized_covariance_columns", color="purple")
def compute_regularized_covariance_columns(
    dataset,
    means,
    mean_prior,
    volume_mask,
    dilated_volume_mask,
    valid_idx,
    gpu_memory,
    options,
    picked_frequencies,
    use_multi_gpu=False,
    n_gpus=None,
):
    volume_shape = dataset.volume_shape
    mask_final = volume_mask

    utils.report_memory_device(logger=logger)

    # Start CUDA profiler for covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStart()
        logger.info("CUDA Profiler: Started profiling covariance computation")

    Hs, Bs = compute_both_H_B(
        dataset,
        means,
        dilated_volume_mask,
        picked_frequencies,
        gpu_memory,
        options=options,
        use_multi_gpu=use_multi_gpu,
        n_gpus=n_gpus,
    )

    # Stop CUDA profiler after covariance computation
    if CUDA_PROFILER_AVAILABLE:
        cudaProfilerStop()
        logger.info("CUDA Profiler: Stopped profiling covariance computation")

    volume_noise_var = np.asarray(
        noise.make_radial_noise(dataset.noise.get_average_radial_noise(), dataset.volume_shape)
    )

    covariance_cols = {}
    utils.report_memory_device(logger=logger)
    covariance_cols["est_mask"], prior, fscs = compute_covariance_regularization_relion_style(
        Hs,
        Bs,
        mean_prior,
        picked_frequencies,
        volume_noise_var,
        mask_final,
        volume_shape,
        gpu_memory,
        reg_init_multiplier=jax_config.REG_INIT_MULTIPLIER,
        options=options,
    )
    covariance_cols["est_mask"] = np.asarray(covariance_cols["est_mask"].T)
    del Hs, Bs
    logger.info("after reg fn")
    utils.report_memory_device(logger=logger)

    return covariance_cols, picked_frequencies, np.stack(fscs, axis=0) if isinstance(fscs, list) else np.asarray(fscs)


# ============================================================================
# New Equinox-based variance estimation
# ============================================================================


@nvtx.annotate("variance_relion_kernel_trilinear", color="yellow")
def variance_relion_kernel_trilinear(
    config: ForwardModelConfig,
    images,
    mean_estimate: jax.Array,
    volume_mask: jax.Array,
    image_mask: jax.Array,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    soften: int = 5,
    Ft_y: jax.Array | None = None,
    Ft_ctf: jax.Array | None = None,
    Ft_im: jax.Array | None = None,
    Ft_one: jax.Array | None = None,
):
    """Variance estimation via RELION-style trilinear kernel — Equinox API.

    Backprojects into half-volume accumulators (Ft_y, Ft_ctf, Ft_im, Ft_one).
    Pass None to initialise from zero; pass an existing half-volume to accumulate.
    """
    return _variance_relion_kernel_trilinear_explicit(
        config,
        images,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        mean_estimate,
        volume_mask,
        image_mask,
        soften=soften,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        Ft_im=Ft_im,
        Ft_one=Ft_one,
    )


@eqx.filter_jit
def _variance_relion_kernel_trilinear_explicit(
    config: ForwardModelConfig,
    images,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    mean_estimate: jax.Array,
    volume_mask: jax.Array,
    image_mask: jax.Array,
    soften: int = 5,
    Ft_y: jax.Array | None = None,
    Ft_ctf: jax.Array | None = None,
    Ft_im: jax.Array | None = None,
    Ft_one: jax.Array | None = None,
):
    noise_variances = noise_variance

    half_images = core.translate_images(images, translations, config.image_shape, half_image=True)
    half_ctf = config.compute_ctf_half(ctf_params)
    CTF_squared = half_ctf**2

    mean_volume = (
        core.VolumeRepr(mean_estimate, disc_type=config.disc_type, prefiltered=True)
        if config.disc_type == "cubic"
        else mean_estimate
    )
    mean_slice_half = core.slice_volume(
        mean_volume,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        config.disc_type,
        half_image=True,
    )

    if config.premultiplied_ctf:
        half_images = half_images - mean_slice_half * CTF_squared
        # CTF² needed in full space for noise scaling; computed from scratch for this case only.
        noise_variances = noise_variances * config.compute_ctf(ctf_params) ** 2
    else:
        half_images = half_images - mean_slice_half * half_ctf

    img_power_half = jnp.abs(half_images) ** 2
    noise_p_variance_ctf = CTF_squared if config.premultiplied_ctf else jnp.ones_like(half_images)
    cov_noise_half = jnp.zeros_like(img_power_half)

    if volume_mask is not None:
        image_mask = covariance_core.get_per_image_tight_mask(
            volume_mask,
            rotation_matrices,
            image_mask,
            config.volume_mask_threshold,
            config.image_shape,
            config.volume_shape,
            config.grid_size,
            config.padding,
            "linear_interp",
            soften=soften,
        )
        # apply_image_masks with half_images=True: IRFFT2 → real-space mask → RFFT2.
        half_images = covariance_core.apply_image_masks(half_images, image_mask, config.image_shape, half_images=True)
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
        return core.adjoint_slice_volume(
            half_imgs,
            rotation_matrices,
            config.image_shape,
            config.volume_shape,
            "linear_interp",
            volume=volume,
            half_image=True,
            half_volume=True,
        )

    Ft_y = _backproject(images_squared, Ft_y)
    Ft_ctf = _backproject(CTF_squared**2, Ft_ctf)
    Ft_im = _backproject(img_power_half, Ft_im)
    Ft_one = _backproject(noise_p_variance_ctf, Ft_one)

    return Ft_y, Ft_ctf, Ft_im, Ft_one


@nvtx.annotate("variance_relion_style_triangular_kernel", color="yellow")
def variance_relion_style_triangular_kernel(
    experiment_dataset, mean_estimate, batch_size, image_subset=None, volume_mask=None, disc_type=""
):
    volume_shape = tuple(
        getattr(experiment_dataset, "upsampled_volume_shape", (int(experiment_dataset.grid_size),) * 3)
    )
    config = ForwardModelConfig(
        image_shape=tuple(experiment_dataset.image_shape),
        volume_shape=volume_shape,
        grid_size=int(experiment_dataset.grid_size),
        voxel_size=float(experiment_dataset.voxel_size),
        padding=int(experiment_dataset.padding),
        disc_type="linear_interp",  # trilinear kernel only supports linear_interp
        ctf=experiment_dataset.ctf_evaluator,
        premultiplied_ctf=bool(experiment_dataset.premultiplied_ctf),
        volume_mask_threshold=float(experiment_dataset.volume_mask_threshold),
    )

    # Full-spectrum noise needed for masked noise variance inside the kernel.
    Ft_y, Ft_ctf, Ft_im, Ft_one = None, None, None, None
    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        _particle_indices,
        _image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        noise_model=experiment_dataset.noise,
        noise_half=False,
        indices=image_subset,
    ):
        images = experiment_dataset.process_images_half(images)
        Ft_y, Ft_ctf, Ft_im, Ft_one = variance_relion_kernel_trilinear(
            config,
            images,
            mean_estimate,
            volume_mask,
            experiment_dataset.image_mask,
            rotation_matrices=rotation_matrices,
            translations=translations,
            ctf_params=ctf_params,
            noise_variance=noise_variance,
            soften=5,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            Ft_im=Ft_im,
            Ft_one=Ft_one,
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
    dataset,
    mean_estimate,
    batch_size,
    volume_mask,
    image_subset=None,
    use_regularization=False,
    disc_type="",
    noise_ind_subset=None,
):
    st = time.time()

    # Run variance kernel for each half-set.
    # variance_relion_style_triangular_kernel returns (Ft_ctf, Ft_y, Ft_one, Ft_im):
    #   ctf_w   — CTF^4 accumulator (Wiener denominator)
    #   signal  — residual^2·CTF^2 accumulator (Wiener numerator)
    #   noise_w — noise-normalisation denominator
    #   noise_s — |residuals|^2 accumulator (noise-normalisation numerator)
    ctf_w, signal, noise_w, noise_s = [], [], [], []
    halfset_datasets = dataset.materialize_halfset_datasets()
    subset_original = None
    if image_subset is not None:
        subset_original = dataset.original_image_indices_from_local(image_subset)

    for halfset_dataset in halfset_datasets:
        if noise_ind_subset is not None:
            subset = np.where(halfset_dataset.noise.dose_indices == noise_ind_subset)[0]
        elif subset_original is not None:
            subset = halfset_dataset.local_image_indices_from_original(
                subset_original,
                allow_missing=True,
            )
        else:
            subset = None
        fw, sig, nw, ns = variance_relion_style_triangular_kernel(
            halfset_dataset,
            mean_estimate,
            batch_size,
            image_subset=subset,
            volume_mask=volume_mask,
            disc_type=disc_type,
        )
        ctf_w.append(relion_functions.adjust_regularization_relion_style(fw, halfset_dataset.volume_shape))
        signal.append(sig)
        noise_w.append(nw)
        noise_s.append(ns)

    variance = {f"corrected{i}": _safe_div(signal[i], ctf_w[i]) for i in range(2)}

    lhs = (ctf_w[0] + ctf_w[1]) / 2
    variance_prior, fsc, _ = regularization.compute_fsc_prior_gpu_v2(
        dataset.volume_shape,
        variance["corrected0"],
        variance["corrected1"],
        lhs,
        jnp.ones(dataset.volume_size, dtype=dataset.dtype_real) * np.inf,
        frequency_shift=jnp.array([0, 0, 0]),
        upsampling_factor=1,
        substract_shell_mean=True,
    )

    if use_regularization:
        for i in range(2):
            reg_lhs = relion_functions.adjust_regularization_relion_style(
                ctf_w[i],
                dataset.volume_shape,
                tau=variance_prior,
            )
            variance[f"corrected{i}"] = _safe_div(signal[i], reg_lhs)

    variance["combined"] = (variance["corrected0"] + variance["corrected1"]) / 2
    variance["prior"] = _to_cpu(variance_prior)
    variance["lhs"] = lhs
    variance = {k: _to_cpu(v).real for k, v in variance.items()}

    noise_p_variance_est = _safe_div(noise_s[0] + noise_s[1], noise_w[0] + noise_w[1])

    logger.info("time to compute variance: %.1fs", time.time() - st)
    return (
        variance,
        _to_cpu(variance_prior).real,
        _to_cpu(fsc).real,
        _to_cpu(lhs).real,
        _to_cpu(noise_p_variance_est).real,
    )


@nvtx.annotate("compute_both_H_B", color="blue")
def compute_both_H_B(
    dataset, means, dilated_volume_mask, picked_frequencies, gpu_memory, options, use_multi_gpu=False, n_gpus=None
):
    """Compute H and B matrices for both half-sets."""
    Hs = []
    Bs = []
    st_time = time.time()
    halfset_datasets = dataset.materialize_halfset_datasets()

    for halfset_id, halfset_dataset in enumerate(halfset_datasets):
        mean = means.combined if options["use_combined_mean"] else means.corrected(halfset_id)
        if options.get("disc_type") == "cubic":
            mean = cubic_interpolation.calculate_spline_coefficients(
                jnp.array(mean).reshape(halfset_dataset.volume_shape)
            )
        if use_multi_gpu:
            H, B = _compute_H_B_multi_gpu(
                halfset_dataset, mean, dilated_volume_mask, picked_frequencies, gpu_memory, options, n_gpus=n_gpus
            )
        else:
            H, B = compute_H_B_for_halfset(
                halfset_dataset, mean, dilated_volume_mask, picked_frequencies, gpu_memory, options
            )
        logger.info("Time to cov %s", time.time() - st_time)
        Hs.append(H)
        Bs.append(B)
    return Hs, Bs


def _compute_H_B_multi_gpu(
    cryo, mean, dilated_volume_mask, picked_frequencies, gpu_memory, options, n_gpus=None, image_subset=None
):
    """Multi-GPU wrapper: splits images across GPUs, each calls compute_H_B_for_halfset."""
    from recovar.utils import multi_gpu as multi_gpu_utils

    logger.info("=" * 60)
    logger.info("MULTI-GPU MODE ENABLED")
    logger.info("=" * 60)

    def _single_gpu_wrapper(
        experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, options, image_subset=None
    ):
        return compute_H_B_for_halfset(
            experiment_dataset,
            mean_estimate,
            volume_mask,
            picked_frequency_indices,
            gpu_memory,
            options,
            image_subset=image_subset,
        )

    H, B = multi_gpu_utils.compute_H_B_multi_gpu(
        compute_H_B_fn=_single_gpu_wrapper,
        experiment_dataset=cryo,
        n_gpus=n_gpus,
        mean_estimate=mean,
        volume_mask=dilated_volume_mask,
        picked_frequency_indices=picked_frequencies,
        batch_size=0,  # unused — compute_H_B_for_halfset sizes its own batches
        options=options,
        image_subset=image_subset,
    )

    logger.info("=" * 60)
    logger.info("MULTI-GPU COMPUTATION COMPLETED")
    logger.info("=" * 60)

    return H, B


def preprocess_covariance_batch(
    config,
    images,
    rotation_matrices,
    translations,
    ctf_params,
    particle_indices,
    mean_estimate,
    volume_mask,
    image_stack_mask,
    opts,
):
    """Preprocess one image batch for covariance column computation.

    Applies tight mask, centers images (y_i - A_i*mu), masks, computes CTF,
    and gets grid-point coordinates.  Each sub-step calls existing JIT-compiled
    helpers; this function is a Python-level orchestrator (not itself jitted).

    Returns
    -------
    tuple of (centered_images, ctf_on_grid, plane_coords, image_mask, tilt_labels)
    """
    # 1. Per-image tight mask (skip expensive 3D FFT + projection when not masking)
    if opts.mask_images:
        # Tight mask projection currently uses the linear-interpolation path.
        image_mask = covariance_core.get_per_image_tight_mask(
            volume_mask,
            rotation_matrices,
            image_stack_mask,
            config.volume_mask_threshold,
            config.image_shape,
            config.volume_shape,
            config.grid_size,
            config.padding,
            "linear_interp",
            soften=opts.soften_mask,
        )
    else:
        image_mask = jnp.ones((rotation_matrices.shape[0], *config.image_shape), dtype=jnp.float32)

    # 2. Center images: y_i - A_i * mu
    mean_volume = (
        core.VolumeRepr(mean_estimate, disc_type=config.disc_type, prefiltered=True)
        if config.disc_type == "cubic"
        else mean_estimate
    )
    images = covariance_core.subtract_projected_mean(
        config, images, mean_volume, ctf_params, rotation_matrices, translations
    )

    # 3. Apply image masks (skip FFT pair when mask is all ones)
    if opts.mask_images:
        images = covariance_core.apply_image_masks(images, image_mask, config.image_shape)

    # 4. Compute CTF
    ctf_on_grid = config.compute_ctf(ctf_params)

    # 5. Grid-point coordinates
    plane_coords = core.batch_get_gridpoint_coords(rotation_matrices, config.image_shape, config.volume_shape)

    # 6. Tilt labels (from particle_indices)
    # particle_indices from the per-particle tilt-series loader (batch_size=1) is shape (1,),
    # but we need per-image labels for the scatter-add in group_sum_by_labels.
    # Broadcast the single particle index to match all images in the tilt-series batch.
    tilt_labels = particle_indices
    if tilt_labels is not None and tilt_labels.shape[0] != images.shape[0]:
        tilt_labels = jnp.broadcast_to(tilt_labels, (images.shape[0],))

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
        H_out[:, col_start:col_end] = _to_cpu(H_batch_jax)
        B_out[:, col_start:col_end] = _to_cpu(B_batch_jax)
        del H_batch_jax, B_batch_jax


def _iter_column_batch_ranges(n_cols, batch_size):
    """Yield ``(start, end)`` pairs covering ``range(n_cols)``."""
    for start in range(0, n_cols, batch_size):
        yield start, min(start + batch_size, n_cols)


def _transposed_column_batch(matrix, start, end):
    """Return the column slice ``matrix[:, start:end]`` as a row-major batch."""
    return matrix[:, start:end].T


def _vec_square_matrix(matrix):
    return matrix.T.reshape(-1)


def _unvec_square_matrix(vector):
    n = np.sqrt(vector.size).astype(int)
    return vector.reshape(n, n).T


def _symmetric_matrix_packed_size(n):
    return n * (n + 1) // 2


def _packed_symmetric_matrix_size_to_full(packed_size):
    n = int((np.sqrt(8 * packed_size + 1) - 1) // 2)
    if _symmetric_matrix_packed_size(n) != packed_size:
        raise ValueError(f"packed_size={packed_size} is not a valid symmetric-matrix packed size")
    return n


@functools.lru_cache(maxsize=None)
def _symmetric_packed_metadata(n):
    row_idx, col_idx = np.triu_indices(n)
    offdiag = row_idx != col_idx
    scales = np.where(offdiag, np.sqrt(2.0), 1.0)
    return row_idx.astype(np.int32), col_idx.astype(np.int32), offdiag, scales


def _symmetric_packed_metadata_jax(n, dtype):
    row_idx, col_idx, offdiag, scales = _symmetric_packed_metadata(n)
    return (
        jnp.asarray(row_idx),
        jnp.asarray(col_idx),
        jnp.asarray(offdiag),
        jnp.asarray(scales, dtype=dtype),
    )


def _pack_symmetric_matrix_svec(matrix):
    n = matrix.shape[-1]
    row_idx, col_idx, _, scales = _symmetric_packed_metadata_jax(n, matrix.dtype)
    return matrix[..., row_idx, col_idx] * scales


def _unpack_symmetric_matrix_svec(vector):
    n = _packed_symmetric_matrix_size_to_full(vector.shape[-1])
    row_idx, col_idx, _, scales = _symmetric_packed_metadata_jax(n, vector.dtype)
    values = vector / scales
    matrix = jnp.zeros((*vector.shape[:-1], n, n), dtype=vector.dtype)
    matrix = matrix.at[..., row_idx, col_idx].set(values)
    matrix = matrix.at[..., col_idx, row_idx].set(values)
    return matrix


def _projected_covariance_dense_lhs_batch(AU_t_AU):
    n_basis = AU_t_AU.shape[-1]
    return jnp.einsum("bik,bjl->ijkl", AU_t_AU, AU_t_AU).reshape(n_basis * n_basis, n_basis * n_basis)


def _projected_covariance_packed_lhs_batch(AU_t_AU):
    n_basis = AU_t_AU.shape[-1]
    row_idx, col_idx, offdiag, scales = _symmetric_packed_metadata_jax(n_basis, AU_t_AU.dtype)

    lhs_rows = AU_t_AU[:, row_idx, :]
    lhs_cols = AU_t_AU[:, col_idx, :]
    cross_terms = jnp.einsum("bpk,bpl->pkl", lhs_rows, lhs_cols)

    packed_lhs = cross_terms[:, row_idx, col_idx]
    packed_lhs = packed_lhs + offdiag.astype(AU_t_AU.dtype)[None, :] * cross_terms[:, col_idx, row_idx]
    packed_lhs = packed_lhs * (scales[:, None] / scales[None, :])
    return packed_lhs


@nvtx.annotate("compute_H_B_for_halfset", color="blue", domain=NVTX_DOMAIN_H_B)
def compute_H_B_for_halfset(
    cryo, mean_estimate, volume_mask, picked_frequencies, gpu_memory, options, image_subset=None, halfset_id=None
):
    """Compute H and B matrices for one half-set.

    Replaces the old ``compute_H_B`` + ``compute_H_B_in_volume_batch`` pair.
    Uses explicit dataset batch iteration and :class:`CovColumnOpts` for
    static kernel options.

    ``mean_estimate`` must already be in the correct form for ``disc_type``:
    raw Fourier coefficients for ``'linear_interp'``, or spline coefficients
    (from ``cubic_interpolation.calculate_spline_coefficients``) for ``'cubic'``.
    The caller (``compute_both_H_B``) is responsible for the conversion.
    """
    volume_size = cryo.volume_size

    disc_type = "cubic" if options["disc_type"] == "cubic" else "linear_interp"
    mean_estimate = jnp.asarray(mean_estimate)

    config = ForwardModelConfig.from_dataset(cryo, disc_type=disc_type)
    opts = CovColumnOpts(
        right_kernel=options["right_kernel"],
        left_kernel=options["left_kernel"],
        right_kernel_width=options["right_kernel_width"],
        mask_images=options["mask_images_in_H_B"],
        soften_mask=3,
    )

    # Batch sizes
    image_batch_size = utils.safe_batch_size(utils.get_image_batch_size(cryo.grid_size, gpu_memory))
    column_batch_size = utils.get_column_batch_size(cryo.grid_size, gpu_memory)

    n_picked = picked_frequencies.size
    H_out = np.empty([volume_size, n_picked], dtype=cryo.dtype)
    B_out = np.empty([volume_size, n_picked], dtype=cryo.dtype)

    for freq_st, freq_end in _iter_column_batch_ranges(n_picked, column_batch_size):
        freq_batch = picked_frequencies[freq_st:freq_end]
        n_freq_batch = freq_batch.size

        freq_batch_jax = jnp.asarray(freq_batch)
        no_mask = not opts.mask_images

        # Accumulators: backprojection adds into these via volumes= parameter
        H_accum = jnp.zeros((n_freq_batch, volume_size), dtype=jnp.complex64)
        B_accum = jnp.zeros((n_freq_batch, volume_size), dtype=jnp.complex64)

        # Inner loop: image batches
        _iter_kw = dict(
            noise_model=cryo.noise,
            noise_half=False,
            noise_by_particle=True,
            by_image=not cryo.tilt_series_flag,
            pack_groups=cryo.tilt_series_flag,
        )
        if halfset_id is not None:
            _iter_kw["halfset_id"] = halfset_id
        elif image_subset is not None:
            _iter_kw["indices"] = image_subset
        for (
            images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            particle_indices,
            _image_indices,
        ) in cryo.iter_batches(image_batch_size, **_iter_kw):
            images = cryo.process_images(images)
            images, ctf_on_grid, plane_coords, image_mask, tilt_labels = preprocess_covariance_batch(
                config,
                images,
                rotation_matrices,
                translations,
                ctf_params,
                particle_indices,
                mean_estimate,
                volume_mask,
                cryo.image_mask,
                opts,
            )
            if tilt_labels is not None and cryo.tilt_series_flag:
                tilt_labels = preprocess_tilt_labels_for_batch(tilt_labels)

            # All frequencies in single XLA program (fori_loop, accumulates via .at[k].add)
            H_accum, B_accum = compute_freq_batch(
                config,
                opts,
                freq_batch_jax,
                images,
                ctf_on_grid,
                plane_coords,
                rotation_matrices,
                noise_variance,
                image_mask,
                tilt_labels,
                cryo.premultiplied_ctf,
                cryo.tilt_series_flag,
                no_mask,
                H_accum,
                B_accum,
            )

            del images, ctf_on_grid, plane_coords, image_mask

        # GPU → CPU transfer (H is real, B is complex)
        H_out[:, freq_st:freq_end] = _to_cpu(H_accum.real).T
        B_out[:, freq_st:freq_end] = _to_cpu(B_accum).T

    return H_out, B_out


@nvtx.annotate("compute_covariance_regularization_relion_style", color="cyan")
def compute_covariance_regularization_relion_style(
    Hs,
    Bs,
    mean_prior,
    picked_frequencies,
    cov_noise,
    volume_mask,
    volume_shape,
    gpu_memory,
    reg_init_multiplier,
    options,
):
    volume_mask = volume_mask if options["use_mask_in_fsc"] else None

    regularization_init = (mean_prior + 1e-14) * reg_init_multiplier / cov_noise

    # Per-column regularization: outer product of regularization_init with itself
    # at the picked frequency.  Evaluated lazily per batch to avoid a large
    # [n_freqs × volume_size] allocation.
    def _reg_init_batch(indices):
        return regularization_init[picked_frequencies[indices], None] * regularization_init[None]

    shifts = core.vec_indices_to_frequencies(picked_frequencies, volume_shape) * options["shift_fsc"]

    n_freqs = picked_frequencies.size
    # //8: regularization vmaps iDFT+crop+mask+DFT per column; peak memory
    # scales as ~10 volume-sized arrays per column, not 4.
    batch_size = utils.safe_batch_size(utils.get_column_batch_size(volume_shape[0], gpu_memory) // 8)

    fsc_priors = [None] * n_freqs
    fscs = [None] * n_freqs
    combined_cov_cols = [None] * n_freqs

    for batch_st, batch_end in _iter_column_batch_ranges(n_freqs, batch_size):
        indices = np.arange(batch_st, batch_end)

        combined_cov_col, priors, fscs_this = regularization.prior_iteration_relion_style_batch(
            _transposed_column_batch(Hs[0], batch_st, batch_end),
            _transposed_column_batch(Hs[1], batch_st, batch_end),
            _transposed_column_batch(Bs[0], batch_st, batch_end),
            _transposed_column_batch(Bs[1], batch_st, batch_end),
            shifts[indices],
            _reg_init_batch(indices),
            options["substract_shell_mean"],
            volume_shape,
            options["left_kernel"],
            options["use_spherical_mask"],
            options["grid_correct"],
            volume_mask,
            options["prior_n_iterations"],
            options["downsample_from_fsc"],
        )

        # Transfer entire batches to CPU at once — avoids per-element DtoH copies.
        combined_cov_cols[batch_st:batch_end] = list(_to_cpu(combined_cov_col))
        fscs[batch_st:batch_end] = list(_to_cpu(fscs_this))
        del combined_cov_col, fscs_this

        if options["prior_n_iterations"] >= 0:
            fsc_priors[batch_st:batch_end] = list(_to_cpu(priors).real)
        del priors

    if options["prior_n_iterations"] >= 0:
        fsc_priors = np.stack(fsc_priors, axis=0).real

    combined_cov_cols = np.stack(combined_cov_cols, axis=0)
    return combined_cov_cols, fsc_priors, fscs


_batch_calculate_spline_coefficients = jax.vmap(
    cubic_interpolation.calculate_spline_coefficients, in_axes=0, out_axes=0
)
_batch_half_volume_to_full_volume = jax.vmap(
    fourier_transform_utils.half_volume_to_full_volume, in_axes=(0, None), out_axes=0
)


def _volume_layout_sizes(volume_shape):
    full_size = int(np.prod(volume_shape))
    half_size = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)))
    return full_size, half_size


def _mean_is_half_volume(mean_estimate, volume_shape):
    _, half_size = _volume_layout_sizes(volume_shape)
    return int(np.prod(mean_estimate.shape)) == half_size


def _basis_is_half_volume(basis, volume_shape):
    _, half_size = _volume_layout_sizes(volume_shape)
    if basis is None or basis.ndim < 1:
        return False
    per_vec_size = int(np.prod(basis.shape[1:])) if basis.shape[0] != 0 else 0
    return per_vec_size == half_size


def _convert_half_volumes_to_full_in_batch(volumes, volume_shape, gpu_memory=None):
    gpu_memory = utils.get_gpu_memory_total() if gpu_memory is None else gpu_memory
    vol_batch_size = utils.safe_batch_size(utils.get_vol_batch_size(volume_shape[0], gpu_memory=gpu_memory))
    logger.info(
        "memory used = %s, vol_batch_size in convert_half_volumes_to_full_in_batch %s",
        gpu_memory,
        vol_batch_size,
    )
    utils.report_memory_device(logger=logger)

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    volumes_4d = jnp.asarray(volumes).reshape(-1, *half_shape)
    if volumes_4d.shape[0] == 0:
        return np.empty((0, *volume_shape), dtype=np.asarray(volumes).dtype)

    full = []
    for k in range(0, volumes_4d.shape[0], vol_batch_size):
        full_block = _batch_half_volume_to_full_volume(volumes_4d[k : k + vol_batch_size], volume_shape)
        full.append(np.asarray(full_block))
    return np.concatenate(full, axis=0).reshape(volumes_4d.shape[0], -1)


def _prepare_model_half_volumes(volume_shape, mean_estimate, basis, mean_disc_type, basis_disc_type, gpu_memory=None):
    """Normalize model layouts for projected covariance.

    Preserve the caller's existing full-vs-half layout whenever the downstream
    interpolation path can consume it directly. Cubic interpolation needs full
    grids, so packed half-volumes must be expanded before use.
    """
    full_size, half_size = _volume_layout_sizes(volume_shape)

    mean_size = int(np.prod(mean_estimate.shape))
    if mean_disc_type == "cubic":
        if mean_size == half_size:
            mean_estimate = fourier_transform_utils.half_volume_to_full_volume(
                mean_estimate.reshape(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)),
                volume_shape,
            ).reshape(-1)
    elif mean_size not in (full_size, half_size):
        logger.warning(
            "Unexpected mean_estimate size %d for volume_shape %s; expected %d (full) or %d (half).",
            mean_size,
            volume_shape,
            full_size,
            half_size,
        )

    if basis is not None and basis.ndim >= 1 and basis.shape[0] > 0:
        basis_size = int(np.prod(basis.shape[1:]))
        if basis_disc_type == "cubic":
            if basis_size == half_size:
                basis = _convert_half_volumes_to_full_in_batch(basis, volume_shape, gpu_memory=gpu_memory)
        elif basis_size not in (full_size, half_size):
            logger.warning(
                "Unexpected basis vector size %d for volume_shape %s; expected %d (full) or %d (half).",
                basis_size,
                volume_shape,
                full_size,
                half_size,
            )

    return mean_estimate, basis


@nvtx.annotate("compute_spline_coeffs_in_batch", color="magenta")
def compute_spline_coeffs_in_batch(basis, volume_shape, gpu_memory=None):
    gpu_memory = utils.get_gpu_memory_total() if gpu_memory is None else gpu_memory
    vol_batch_size = utils.safe_batch_size(utils.get_vol_batch_size(volume_shape[0], gpu_memory=gpu_memory))
    logger.info(
        "memory used = %s, vol_batch_size in compute_spline_coeffs_in_batch %s",
        gpu_memory,
        vol_batch_size,
    )
    utils.report_memory_device(logger=logger)

    basis_4d = jnp.asarray(basis).reshape(-1, *volume_shape)
    if basis_4d.shape[0] == 0:
        return np.empty((0, *volume_shape), dtype=np.asarray(basis).dtype)

    coeffs = []
    for k in range(0, basis_4d.shape[0], vol_batch_size):
        coeffs_block = _batch_calculate_spline_coefficients(basis_4d[k : k + vol_batch_size])
        coeffs.append(np.asarray(coeffs_block))
    return np.concatenate(coeffs, axis=0)


## REDUCED COVARIANCE COMPUTATION


def _compute_projected_covariance_single(
    experiment_dataset, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, do_mask_images=True
):
    """Projected covariance for a single dataset (no halfset split).

    Backward-compat path used by ``em/heterogeneity.py`` and tests that pass a
    single ``CryoEMDataset`` without ``halfset_indices``.
    """
    basis = basis.T.astype(experiment_dataset.dtype)
    basis = jnp.asarray(basis)
    volume_mask = jnp.asarray(volume_mask, dtype=experiment_dataset.dtype_real)
    mean_estimate = jnp.asarray(mean_estimate, dtype=experiment_dataset.dtype)

    n_basis = basis.shape[0]
    lhs_size = _symmetric_matrix_packed_size(n_basis)
    lhs = jnp.zeros((lhs_size, lhs_size), dtype=experiment_dataset.dtype_real)
    rhs = jnp.zeros((n_basis, n_basis), dtype=experiment_dataset.dtype_real)
    logger.info("batch size in compute_projected_covariance %s", batch_size)

    if disc_type == "cubic":
        mean_estimate = cubic_interpolation.calculate_spline_coefficients(
            mean_estimate.reshape(experiment_dataset.volume_shape)
        )
    if disc_type_u == "cubic":
        basis = compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory=None)

    mean_estimate, basis = _prepare_model_half_volumes(
        experiment_dataset.volume_shape,
        mean_estimate,
        basis,
        mean_disc_type=disc_type,
        basis_disc_type=disc_type_u,
    )

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    model = ModelState(mean_estimate=mean_estimate, volume_mask=volume_mask, basis=basis)
    opts = CovarianceOpts(
        disc_type_u=disc_type_u, do_mask_images=do_mask_images, shared_label=experiment_dataset.tilt_series_flag
    )
    hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape, dtype=experiment_dataset.dtype_real)

    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        particle_indices,
        _image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        noise_model=experiment_dataset.noise,
        noise_half=False,
        by_image=not experiment_dataset.tilt_series_flag,
        pack_groups=experiment_dataset.tilt_series_flag,
    ):
        tilt_labels = None
        if experiment_dataset.tilt_series_flag:
            tilt_labels = preprocess_tilt_labels_for_batch(
                _expand_particle_indices_to_per_image(particle_indices, images.shape[0])
            )
        lhs, rhs = reduce_covariance_inner(
            config,
            images,
            model,
            opts,
            experiment_dataset.image_mask,
            rotation_matrices=rotation_matrices,
            translations=translations,
            ctf_params=ctf_params,
            noise_variance=noise_variance,
            hermitian_weights=hermitian_weights,
            lhs=lhs,
            rhs=rhs,
            tilt_labels=tilt_labels,
        )
    del basis

    logger.info("end of covariance computation - before solve")
    utils.report_memory_device(logger=logger)
    covar = _solve_projected_covariance_system(lhs, rhs)
    logger.info("end of solve")
    return covar


@nvtx.annotate("compute_projected_covariance", color="green")
def compute_projected_covariance(
    dataset, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, do_mask_images=True
):
    from recovar.data_io.cryoem_dataset import CryoEMDataset

    # Backward compat: accept list of datasets (legacy API from em/heterogeneity.py)
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0] if len(dataset) == 1 else dataset
    if isinstance(dataset, CryoEMDataset) and dataset.halfset_indices is None:
        # Single dataset without halfset split — iterate over all images (no half-set logic)
        return _compute_projected_covariance_single(
            dataset, mean_estimate, basis, volume_mask, batch_size, disc_type, disc_type_u, do_mask_images
        )

    basis = basis.T.astype(dataset.dtype)
    basis = jnp.asarray(basis)
    volume_mask = jnp.asarray(volume_mask, dtype=dataset.dtype_real)
    mean_estimate = jnp.asarray(mean_estimate, dtype=dataset.dtype)

    n_basis = basis.shape[0]  # basis is (n_pcs, vol_size) after .T
    lhs_size = _symmetric_matrix_packed_size(n_basis)
    lhs = jnp.zeros((lhs_size, lhs_size), dtype=dataset.dtype_real)
    rhs = jnp.zeros((n_basis, n_basis), dtype=dataset.dtype_real)
    logger.info("batch size in compute_projected_covariance %s", batch_size)

    if disc_type == "cubic":
        mean_estimate = cubic_interpolation.calculate_spline_coefficients(mean_estimate.reshape(dataset.volume_shape))

    if disc_type_u == "cubic":
        basis = compute_spline_coeffs_in_batch(basis, dataset.volume_shape, gpu_memory=None)

    mean_estimate, basis = _prepare_model_half_volumes(
        dataset.volume_shape,
        mean_estimate,
        basis,
        mean_disc_type=disc_type,
        basis_disc_type=disc_type_u,
    )

    for halfset_id in range(2):
        config = ForwardModelConfig.from_dataset(
            dataset,
            disc_type=disc_type,
            process_fn=dataset.process_images,
        )
        model = ModelState(
            mean_estimate=mean_estimate,
            volume_mask=volume_mask,
            basis=basis,
        )
        opts = CovarianceOpts(
            disc_type_u=disc_type_u,
            do_mask_images=do_mask_images,
            shared_label=dataset.tilt_series_flag,
        )

        hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape, dtype=dataset.dtype_real)

        for (
            images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            particle_indices,
            _image_indices,
        ) in dataset.iter_batches(
            batch_size,
            noise_model=dataset.noise,
            noise_half=False,
            halfset_id=halfset_id,
            by_image=not dataset.tilt_series_flag,
            pack_groups=dataset.tilt_series_flag,
        ):
            tilt_labels = None
            if dataset.tilt_series_flag:
                tilt_labels = preprocess_tilt_labels_for_batch(
                    _expand_particle_indices_to_per_image(particle_indices, images.shape[0])
                )
            lhs, rhs = reduce_covariance_inner(
                config,
                images,
                model,
                opts,
                dataset.image_mask,
                rotation_matrices=rotation_matrices,
                translations=translations,
                ctf_params=ctf_params,
                noise_variance=noise_variance,
                hermitian_weights=hermitian_weights,
                lhs=lhs,
                rhs=rhs,
                tilt_labels=tilt_labels,
            )
    del basis
    # Deallocate some memory?

    logger.info("end of covariance computation - before solve")
    utils.report_memory_device(logger=logger)
    covar = _solve_projected_covariance_system(lhs, rhs)
    logger.info("end of solve")

    return covar


# ============================================================================
# Equinox-based API for covariance estimation
# ============================================================================


@nvtx.annotate("reduce_covariance_inner", color="green")
def reduce_covariance_inner(
    config: ForwardModelConfig,
    images,
    model: ModelState,
    opts: CovarianceOpts,
    image_mask: jax.Array,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    hermitian_weights=None,
    lhs=None,
    rhs=None,
    tilt_labels=None,
):
    """Covariance estimation inner loop — Equinox API.

    Replaces the 24-param ``reduce_covariance_est_inner`` with structured inputs.
    All geometry/CTF/discretization params come from ``config``.
    Per-batch arrays are passed explicitly.
    Reconstruction state (mean, mask, basis) comes from ``model``.
    Boolean flags come from ``opts``.
    """
    return _reduce_covariance_inner_explicit(
        config,
        images,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        model,
        opts,
        image_mask,
        hermitian_weights=hermitian_weights,
        lhs=lhs,
        rhs=rhs,
        tilt_labels=tilt_labels,
    )


@eqx.filter_jit
def _reduce_covariance_inner_explicit(
    config: ForwardModelConfig,
    batch,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    model: ModelState,
    opts: CovarianceOpts,
    image_mask: jax.Array,
    hermitian_weights=None,
    lhs=None,
    rhs=None,
    tilt_labels=None,
):

    if (config.disc_type != "linear_interp") and (config.disc_type != "cubic"):
        logger.warning("USING NEAREST NEIGHBOR DISCRETIZATION. disc_type=%s", config.disc_type)

    do_mask_images = opts.do_mask_images
    if config.premultiplied_ctf and do_mask_images:
        logger.warning("cannot use premultiplied ctf and mask images at the same time.")
        do_mask_images = False

    if do_mask_images:
        image_mask = covariance_core.get_per_image_tight_mask(
            model.volume_mask,
            rotation_matrices,
            image_mask,
            config.volume_mask_threshold,
            config.image_shape,
            config.volume_shape,
            config.grid_size,
            config.padding,
            "linear_interp",
        )
    # (no else — image_mask is unused when do_mask_images=False)

    if config.process_fn is not None:
        batch = config.process_fn(batch)

    # Always project to half-image format when hermitian weights are available.
    _use_half_proj = hermitian_weights is not None

    # Convert batch to half-image early and translate in half format.
    if _use_half_proj:
        batch = fourier_transform_utils.full_image_to_half_image(batch, config.image_shape)
        batch = core.translate_images(batch, translations, config.image_shape, half_image=True)
    else:
        batch = core.translate_images(batch, translations, config.image_shape)

    mean_half_volume = _mean_is_half_volume(model.mean_estimate, config.volume_shape)
    basis_half_volume = _basis_is_half_volume(model.basis, config.volume_shape)
    mean_volume = (
        core.VolumeRepr(
            model.mean_estimate,
            disc_type=config.disc_type,
            half_volume=mean_half_volume,
            prefiltered=True,
        )
        if config.disc_type == "cubic"
        else model.mean_estimate
    )
    projected_mean = core_forward.forward_model(
        config,
        mean_volume,
        ctf_params,
        rotation_matrices,
        half_image=_use_half_proj,
        half_volume=mean_half_volume,
    )

    if do_mask_images:
        batch = covariance_core.apply_image_masks(batch, image_mask, config.image_shape, half_images=_use_half_proj)
        projected_mean = covariance_core.apply_image_masks(
            projected_mean, image_mask, config.image_shape, half_images=_use_half_proj
        )

    # Forward model basis vectors — use disc_type_u for eigenvectors
    basis = model.basis
    assert basis is not None
    config_u = config.replace(disc_type=opts.disc_type_u)
    basis_volume = (
        core.VolumeRepr(
            basis,
            disc_type=opts.disc_type_u,
            half_volume=basis_half_volume,
            prefiltered=True,
        )
        if opts.disc_type_u == "cubic"
        else basis
    )
    AUs = covariance_core.batch_vol_forward_from_map(
        config_u,
        basis_volume,
        ctf_params,
        rotation_matrices,
        skip_ctf=config.premultiplied_ctf,
        half_image=_use_half_proj,
        half_volume=basis_half_volume,
    )

    if do_mask_images:
        AUs = covariance_core.apply_image_masks_to_eigen(
            AUs, image_mask, config.image_shape, half_images=_use_half_proj
        )

    # --- Half-spectrum weighting ---
    # batch, projected_mean, and AUs are already in half-image format when
    # _use_half_proj is True.  Apply sqrt(w) Hermitian weights so that plain
    # inner products equal the correct full-spectrum inner products.
    if hermitian_weights is not None:
        batch = batch * hermitian_weights
        projected_mean = projected_mean * hermitian_weights
        AUs = AUs * hermitian_weights[None, None, :]
        noise_variance = fourier_transform_utils.full_image_to_half_image(noise_variance, config.image_shape)

    AUs = AUs.transpose(1, 2, 0)

    if config.premultiplied_ctf:
        CTF = config.compute_ctf(ctf_params, half_image=(hermitian_weights is not None))
        batch = batch - projected_mean * CTF
        AU_t_images = batch_x_T_y(AUs, batch)
        AUs = AUs * CTF[..., None]
    else:
        batch = batch - projected_mean
        AU_t_images = batch_x_T_y(AUs, batch)

    # When using half-spectrum weights, the inner product sum_k w*conj(a)*b
    # has the correct real part but spurious imaginary part (interior pixels
    # contribute 2*conj(a)*b instead of 2*Re(conj(a)*b)).  Taking .real here
    # eliminates the imaginary bias that would otherwise inflate the outer
    # product x*conj(x)^T by |Im(x)|^2.
    if hermitian_weights is not None:
        AU_t_images = AU_t_images.real

    if do_mask_images:
        if config.premultiplied_ctf:
            raise NotImplementedError("Masking with premultiplied CTF is not implemented yet")

    AU_t_AU = batch_x_T_y(AUs, AUs).real.astype(ctf_params.dtype)

    # Per-image noise bias: M_i = AU_i^H diag(noise_i) AU_i
    # Uses per-image vmapped matmul (matching old code's accumulation pattern).
    AUs_noise = AUs * jnp.sqrt(noise_variance)[..., None]
    per_image_noise_bias = batch_x_T_y(AUs_noise, AUs_noise)

    if opts.shared_label:
        # For tilt series: sum AU_t_images and AU_t_AU over tilts of each particle.
        # With packed multi-particle batches, use group_sum_by_labels to sum
        # per-particle instead of across the entire batch.
        if tilt_labels is not None:
            # Per-particle segment sums (not broadcast back — avoids double-counting
            # in the subsequent sum(axis=0)).
            n_groups = tilt_labels.shape[0]  # safe upper bound
            summed_noise_bias = jax.ops.segment_sum(per_image_noise_bias, tilt_labels, num_segments=n_groups)
            AU_t_images = jax.ops.segment_sum(AU_t_images, tilt_labels, num_segments=n_groups)
            AU_t_AU = jax.ops.segment_sum(AU_t_AU, tilt_labels, num_segments=n_groups)
        else:
            # Single-particle batch (legacy path)
            summed_noise_bias = jnp.sum(per_image_noise_bias, axis=0, keepdims=True)
            AU_t_images = jnp.sum(AU_t_images, axis=0, keepdims=True)
            AU_t_AU = jnp.sum(AU_t_AU, axis=0, keepdims=True)
        per_image_outer = AU_t_images[:, :, None] * jnp.conj(AU_t_images[:, None, :])
        rhs_batch = (per_image_outer - summed_noise_bias).sum(axis=0).real.astype(ctf_params.dtype)
    else:
        # For SPA: per-image subtraction avoids catastrophic cancellation.
        # When accumulated first, both sum_i(x_i x_i^H) and sum_i(M_i) are ~1e21
        # while their difference is ~1e14 — a 7-order-of-magnitude cancellation
        # that exceeds float32 precision.
        per_image_outer = AU_t_images[:, :, None] * jnp.conj(AU_t_images[:, None, :])
        rhs_batch = (per_image_outer - per_image_noise_bias).sum(axis=0).real.astype(ctf_params.dtype)

    lhs_batch = _projected_covariance_packed_lhs_batch(AU_t_AU)
    if lhs is not None:
        lhs_batch = lhs_batch + lhs
    if rhs is not None:
        rhs_batch = rhs_batch + rhs
    return lhs_batch, rhs_batch


batch_kron = jax.vmap(jnp.kron, in_axes=(0, 0))


@nvtx.annotate("summed_batch_kron", color="gray")
def summed_batch_kron(X):
    return jnp.sum(batch_kron(X, X), axis=0)


@nvtx.annotate("summed_batch_kron_scan", color="gray")
def summed_batch_kron_scan(X):
    init = jnp.zeros((X.shape[1] * X.shape[1],), dtype=X.dtype)

    def fori_loop_body(i, val):
        return val + jnp.kron(X[i], X[i])

    summed_kron = jax.lax.fori_loop(0, X.shape[0], fori_loop_body, init)
    return summed_kron


def _regularize_projected_covariance_lhs(lhs, reg_scale):
    trace_val = jnp.trace(lhs)
    trace_val = jnp.where(jnp.isfinite(trace_val) & (trace_val > 0), trace_val, jnp.float32(1.0))
    diag_idx = jnp.arange(lhs.shape[0])
    reg = jnp.asarray(reg_scale, dtype=lhs.dtype) * trace_val / lhs.shape[0]
    return lhs.at[diag_idx, diag_idx].add(reg)


def _raise_on_nonfinite_projected_covariance(covar, reg_scale):
    covar_np = np.asarray(covar)
    if not np.isfinite(covar_np).all():
        n_nan = int(np.isnan(covar_np).sum())
        n_inf = int(np.isinf(covar_np).sum())
        raise ValueError(
            "projected covariance solve returned non-finite output after "
            f"regularization scale {reg_scale}: {n_nan} NaN, {n_inf} Inf"
        )
    return covar


def _solve_projected_covariance_system_dense(lhs, rhs, reg_scale=1e-6):
    rhs = _vec_square_matrix(rhs)
    lhs_reg = _regularize_projected_covariance_lhs(lhs, reg_scale)
    covar = jax.scipy.linalg.solve(lhs_reg, rhs, assume_a="pos")
    covar = _unvec_square_matrix(covar)
    _raise_on_nonfinite_projected_covariance(covar, reg_scale)
    return covar


def _solve_projected_covariance_system_packed(lhs, rhs, reg_scale=1e-6):
    rhs = _pack_symmetric_matrix_svec(rhs)
    lhs_reg = _regularize_projected_covariance_lhs(lhs, reg_scale)
    covar = jax.scipy.linalg.solve(lhs_reg, rhs, assume_a="pos")
    covar = _unpack_symmetric_matrix_svec(covar)
    _raise_on_nonfinite_projected_covariance(covar, reg_scale)
    return covar


def _solve_projected_covariance_system(lhs, rhs, reg_scale=1e-6):
    """Solve the projected-covariance normal equations and fail on non-finite output."""
    n_basis = rhs.shape[-1]
    packed_dim = _symmetric_matrix_packed_size(n_basis)
    full_dim = n_basis * n_basis
    if lhs.shape == (packed_dim, packed_dim):
        return _solve_projected_covariance_system_packed(lhs, rhs, reg_scale=reg_scale)
    if lhs.shape == (full_dim, full_dim):
        return _solve_projected_covariance_system_dense(lhs, rhs, reg_scale=reg_scale)
    raise ValueError(
        "projected covariance lhs has incompatible shape "
        f"{lhs.shape} for rhs shape {rhs.shape}; expected {(packed_dim, packed_dim)} or {(full_dim, full_dim)}"
    )


batch_x_T_y = jax.vmap(lambda x, y: jnp.conj(x).T @ y, in_axes=(0, 0))


@nvtx.annotate("summed_outer_products", color="gray")
def summed_outer_products(AU_t_images):
    # Not .H because things are already transposed technically
    return AU_t_images.T @ jnp.conj(AU_t_images)


batched_summed_outer_products = jax.vmap(summed_outer_products)


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


@nvtx.annotate("adjoint_kernel_slice", color="blue", domain=NVTX_DOMAIN_H_B)
def adjoint_kernel_slice(images, rotation_matrices, image_shape, volume_shape, kernel="triangular", volumes=None):
    """Backproject images to volume(s).

    When *images* is 3-D ``(batch, n_images, n_pix)``, vmaps over the batch.
    When *images* is 2-D ``(n_images, n_pix)``, backprojects a single batch.
    *volumes* is an optional accumulator: the backprojection is **added** into
    *volumes* instead of starting from zeros.
    """
    disc_type = "linear_interp" if kernel == "triangular" else "nearest"
    if kernel not in ("triangular", "square"):
        raise ValueError("Kernel not implemented")
    if images.ndim == 3:
        return core.batch_adjoint_slice_volume(
            images, rotation_matrices, image_shape, volume_shape, disc_type, volumes=volumes
        )
    return core.adjoint_slice_volume(images, rotation_matrices, image_shape, volume_shape, disc_type, volume=volumes)


def _compute_noise_from_kernel_vals(
    kernel_vals, CTF_on_grid, image_shape, image_mask, noise_variances, premultiplied_ctf
):
    """Noise term using pre-computed kernel_vals.

    If ctf is premultiplied: noise = mask @ ctf @ noise_variance @ ctf @ mask
    If not premultiplied:    noise = ctf @ mask @ noise_variance @ mask @ ctf
    """
    k = kernel_vals
    if not premultiplied_ctf:
        k = k * CTF_on_grid
    k = covariance_core.apply_image_masks(k, image_mask, image_shape)
    if premultiplied_ctf:
        k = k * jnp.conj(CTF_on_grid)
    k = k * noise_variances
    if premultiplied_ctf:
        k = k * jnp.conj(CTF_on_grid)
    k = covariance_core.apply_image_masks(k, image_mask, image_shape)
    if not premultiplied_ctf:
        k = k * jnp.conj(CTF_on_grid)
    return k


@eqx.filter_jit
@nvtx.annotate("compute_freq_batch", color="red", domain=NVTX_DOMAIN_H_B)
def compute_freq_batch(
    config: ForwardModelConfig,
    opts: CovColumnOpts,
    freq_batch: jax.Array,
    images: jax.Array,
    ctf_on_grid: jax.Array,
    plane_coords: jax.Array,
    rotation_matrices: jax.Array,
    noise_variances: jax.Array,
    image_mask: jax.Array,
    tilt_labels,
    premultiplied_ctf: bool,
    shared_label: bool,
    no_mask: bool,
    H_accum: jax.Array | None = None,
    B_accum: jax.Array | None = None,
):
    """Compute H and B for a batch of frequencies in a single XLA program.

    Replaces the Python for-loop over frequencies with
    ``jax.lax.fori_loop``.  Each iteration computes one frequency's images
    and backprojects them, accumulating into *H_accum* / *B_accum* via
    ``.at[k].add()``.

    Memory: only one frequency's intermediate images live at a time.
    """
    n_freq = freq_batch.shape[0]

    # Pre-compute frequency-independent terms ONCE
    if premultiplied_ctf:
        ctfed_images = images
    else:
        ctfed_images = images * jnp.conj(ctf_on_grid)
    ctf_squared = ctf_on_grid * jnp.conj(ctf_on_grid)

    max_groups = images.shape[0]

    if H_accum is None:
        H_accum = jnp.zeros((n_freq, config.volume_size), dtype=images.dtype)
    if B_accum is None:
        B_accum = jnp.zeros((n_freq, config.volume_size), dtype=images.dtype)

    def body(k, carry):
        H_acc, B_acc = carry
        freq_idx = freq_batch[k]
        picked_freq_coord = core.vec_indices_to_vol_indices(freq_idx, config.volume_shape)

        kernel_vals = covariance_core.evaluate_kernel_on_grid(
            plane_coords, picked_freq_coord, kernel=opts.right_kernel, kernel_width=opts.right_kernel_width
        )

        # ── B term ──
        images_prod = jnp.sum(kernel_vals * ctfed_images, axis=-1)
        if shared_label:
            if tilt_labels is not None:
                images_prod = group_sum_by_labels(images_prod, tilt_labels, max_groups)
            else:
                images_prod = jnp.repeat(jnp.sum(images_prod, axis=0, keepdims=True), images_prod.shape[0], axis=0)

        rhs = ctfed_images * jnp.conj(images_prod)[..., None]
        if no_mask:
            if not premultiplied_ctf:
                noise = kernel_vals * ctf_squared * noise_variances
            else:
                noise = kernel_vals * jnp.conj(ctf_on_grid) ** 2 * noise_variances
        else:
            noise = _compute_noise_from_kernel_vals(
                kernel_vals, ctf_on_grid, config.image_shape, image_mask, noise_variances, premultiplied_ctf
            )
        rhs = rhs - noise

        B_k = adjoint_kernel_slice(rhs, rotation_matrices, config.image_shape, config.volume_shape, opts.left_kernel)

        # ── H term ──
        ctfs_prods = jnp.sum(kernel_vals * ctf_squared, axis=-1)
        if shared_label:
            if tilt_labels is not None:
                ctfs_prods = group_sum_by_labels(ctfs_prods, tilt_labels, max_groups)
            else:
                ctfs_prods = jnp.repeat(jnp.sum(ctfs_prods, axis=0, keepdims=True), ctfs_prods.shape[0], axis=0)
        lhs = ctf_squared * ctfs_prods[..., None]

        H_k = adjoint_kernel_slice(lhs, rotation_matrices, config.image_shape, config.volume_shape, opts.left_kernel)

        return H_acc.at[k].add(H_k), B_acc.at[k].add(B_k)

    return jax.lax.fori_loop(0, n_freq, body, (H_accum, B_accum))


def _expand_particle_indices_to_per_image(particle_indices, n_images):
    """Expand scalar/single-element particle_indices to per-image array.

    Redundant when the backend already yields per-image particle_indices
    (e.g. _ImageCountBatchLoader), but needed for the old batch_size=1
    tilt-series path where particle_indices is a scalar.
    """
    p = jnp.asarray(particle_indices).reshape(-1)
    if p.shape[0] != n_images and p.shape[0] == 1:
        p = jnp.broadcast_to(p, (n_images,))
    return p


@nvtx.annotate("preprocess_tilt_labels_for_batch", color="brown")
def preprocess_tilt_labels_for_batch(tilt_labels):
    """
    Pre-process tilt_labels to be consecutive indices starting from 0.
    This should be called outside JIT to handle arbitrary tilt label values.

    Args:
        tilt_labels: Array of arbitrary tilt label values

    Returns:
        Array of consecutive indices ``0, 1, 2, ...`` with the same grouping
        structure as the input labels.
    """
    if tilt_labels is None:
        return None, None

    # Get unique labels and their inverse indices
    unique_labels, inverse_indices = jnp.unique(tilt_labels, return_inverse=True, size=tilt_labels.shape[0])

    # inverse_indices already gives us the mapping to consecutive indices
    return inverse_indices
