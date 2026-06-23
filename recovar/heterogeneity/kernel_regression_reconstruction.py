"""Kernel-regression reconstruction helpers used by compute_state.

This module owns the direct RELION-style volume reconstruction path used by
``heterogeneity_volume.make_volumes_kernel_estimate_local``.  The older
adaptive/precompute polynomial discretization routines remain in
``adaptive_kernel_discretization.py``.
"""

import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
from recovar.cuda_backproject import custom_cuda_requested
from recovar.reconstruction import relion_functions

logger = logging.getLogger(__name__)


def _effective_heterogeneity_memory_budget(avail_gb):
    """Apply the fallback-path safety margin to the batch-size budget."""
    if custom_cuda_requested():
        return avail_gb

    scaled_gb = max(1.0, avail_gb / 3.0)
    logger.info(
        "RECOVAR_DISABLE_CUDA is active - scaling heterogeneity-kernel "
        "memory budget to 1/3 of available (%.1f GB) to account for the "
        "JAX-native fallback path's higher per-image memory cost.",
        scaled_gb,
    )
    return scaled_gb


def _pad_noise_variance_for_fixed_batch(noise_variance, current_batch_size, target_batch_size):
    """Pad per-image noise rows so padded batch items contribute zero weight."""
    if noise_variance is None:
        return None

    arr = np.asarray(noise_variance)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"noise_variance must be a 1D or 2D array when padding batches, got shape {arr.shape}")

    if arr.shape[0] == 1:
        arr = np.repeat(arr, current_batch_size, axis=0)
    elif arr.shape[0] != current_batch_size:
        raise ValueError(
            f"noise_variance rows must match batch size when padding batches: {arr.shape[0]} != {current_batch_size}"
        )

    if current_batch_size == target_batch_size:
        return arr

    padded = np.full((target_batch_size, arr.shape[1]), np.inf, dtype=arr.dtype)
    padded[:current_batch_size] = arr
    return padded


def _pad_heterogeneity_kernel_batch(
    images,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    *,
    target_batch_size,
):
    """Pad the final image batch to a fixed size to avoid JIT recompilation."""
    current_batch_size = int(images.shape[0])
    if current_batch_size == target_batch_size:
        return images, rotation_matrices, translations, ctf_params, noise_variance
    if current_batch_size > target_batch_size:
        raise ValueError(f"batch size {current_batch_size} exceeds target_batch_size {target_batch_size}")

    pad = target_batch_size - current_batch_size

    def _pad_zeros(arr, n):
        if isinstance(arr, jax.Array):
            return jnp.concatenate([arr, jnp.zeros((n, *arr.shape[1:]), dtype=arr.dtype)])
        arr = np.asarray(arr)
        return np.concatenate([arr, np.zeros((n, *arr.shape[1:]), dtype=arr.dtype)])

    def _pad_repeat_first(arr, n):
        if isinstance(arr, jax.Array):
            return jnp.concatenate([arr, jnp.repeat(arr[:1], n, axis=0)])
        arr = np.asarray(arr)
        return np.concatenate([arr, np.repeat(arr[:1], n, axis=0)])

    images = _pad_zeros(images, pad)
    rotation_matrices = _pad_repeat_first(rotation_matrices, pad)
    translations = _pad_repeat_first(translations, pad)
    ctf_params = _pad_repeat_first(ctf_params, pad)
    noise_variance = _pad_noise_variance_for_fixed_batch(
        noise_variance,
        current_batch_size,
        target_batch_size,
    )
    return images, rotation_matrices, translations, ctf_params, noise_variance


def _process_images_half_fast(images, dataset):
    """Convert raw spatial images to half-spectrum using JIT'd rfft2."""
    src = getattr(dataset, "image_source", None)
    if src is not None and hasattr(src, "data_multiplier") and hasattr(src, "padding"):
        from recovar.core import padding as pad

        images = jnp.asarray(images) * src.data_multiplier
        return pad.padded_rfft(images, dataset.grid_size, src.padding)
    return dataset.process_images_half(images)


def _candidate_half_volume_size(experiment_dataset, upsampling_factor):
    scale = 1 if upsampling_factor is None else int(upsampling_factor)
    volume_shape = tuple(3 * [experiment_dataset.grid_size * scale])
    return int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)))


def _reconstruction_config(experiment_dataset, disc_type, upsampling_factor):
    config_upsampling = 2 if upsampling_factor is None else int(upsampling_factor)
    return ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        upsampling_factor=config_upsampling,
    )


def _postprocess_upsampling_factor(upsampling_factor):
    return 1 if upsampling_factor is None else int(upsampling_factor)


def _prepare_half_image_batch(
    experiment_dataset,
    raw_images,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    *,
    batch_size,
    use_fast_rfft,
):
    """Return a fixed-size half-spectrum batch plus original row count."""
    current_batch_size = int(raw_images.shape[0])
    if use_fast_rfft:
        raw_images, rotation_matrices, translations, ctf_params, noise_variance = _pad_heterogeneity_kernel_batch(
            raw_images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            target_batch_size=batch_size,
        )
        images = _process_images_half_fast(raw_images, experiment_dataset)
    else:
        images = experiment_dataset.process_images_half(raw_images)
        images, rotation_matrices, translations, ctf_params, noise_variance = _pad_heterogeneity_kernel_batch(
            images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            target_batch_size=batch_size,
        )
    return current_batch_size, images, rotation_matrices, translations, ctf_params, noise_variance


def _postprocess_candidate_estimates(
    lhs_all,
    rhs_all,
    experiment_dataset,
    *,
    upsampling_factor,
    tau,
    disc_type,
    use_spherical_mask,
    grid_correct,
    return_real_space,
):
    kernel_type = "triangular" if disc_type == "linear_interp" else "square"
    vol_upsample = _postprocess_upsampling_factor(upsampling_factor)
    estimates = [
        relion_functions.post_process_from_filter_v2(
            lhs,
            rhs,
            experiment_dataset.volume_shape,
            vol_upsample,
            tau=tau,
            kernel=kernel_type,
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct="square",
            kernel_width=1,
            return_real_space=return_real_space,
            input_half_volume=True,
        ).reshape(-1)
        for lhs, rhs in zip(lhs_all, rhs_all)
    ]
    return np.asarray(jnp.stack(estimates, axis=0))


def _apply_standard_kernel_weights(lhs_all, rhs_all, bins, heterogeneity_kernel):
    if heterogeneity_kernel == "square":
        return np.cumsum(lhs_all, axis=0), np.cumsum(rhs_all, axis=0)

    if heterogeneity_kernel not in ("parabola", "triangle"):
        raise NotImplementedError

    h_grid = 2 * bins
    weight_matrix = np.zeros((bins.size, bins.size), dtype=np.float32)
    weight_matrix[0, 0] = 1
    for idx in range(1, bins.size):
        dist = np.sqrt(bins / h_grid[idx])
        if heterogeneity_kernel == "triangle":
            weights = np.where(np.abs(dist) < 1, 1 - np.abs(dist), 0)
        else:
            weights = np.where(np.abs(dist) < 1, 3 / 4 * (1 - dist**2), 0)
        weight_matrix[:, idx] = weights

    rhs_all = np.asarray(weight_matrix.T.astype(rhs_all.real.dtype) @ rhs_all)
    lhs_all = np.asarray(weight_matrix.T @ lhs_all)
    return lhs_all, rhs_all


def _image_and_ctf_terms_from_fft(
    config: ForwardModelConfig,
    images,
    ctf_params,
    translations,
    noise_variance,
    upsample_ctf: bool = True,
):
    from recovar.core.geometry import translate_images
    from recovar.reconstruction import noise as noise_mod

    half_images = translate_images(images, translations, config.image_shape, half_image=True)
    noise_half = noise_mod.to_batched_half_pixel_noise(
        noise_variance, config.image_shape, batch_size=half_images.shape[0]
    )
    half_images = half_images / noise_half

    ctf = config.compute_ctf_half(ctf_params)
    if not config.premultiplied_ctf:
        half_images = half_images * ctf

    if upsample_ctf:
        from recovar.core.ctf import compute_antialiased_ctf_squared

        ctf_half = (
            compute_antialiased_ctf_squared(
                config.ctf,
                ctf_params,
                config.image_shape,
                config.voxel_size,
                half_image=True,
            )
            / noise_half
        )
    else:
        ctf_half = ctf**2 / noise_half

    return half_images, ctf_half


@eqx.filter_jit
def _backproject_weighted_batch_from_fft(
    config: ForwardModelConfig,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    image_weights,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    upsample_ctf: bool = True,
):
    """Backproject half-spectrum images with one scalar weight per image."""
    half_images, ctf_half = _image_and_ctf_terms_from_fft(
        config,
        images,
        ctf_params,
        translations,
        noise_variance,
        upsample_ctf=upsample_ctf,
    )
    weights = jnp.asarray(image_weights, dtype=half_images.real.dtype)
    weights = jnp.reshape(weights, (weights.shape[0],) + (1,) * (half_images.ndim - 1))
    half_images = half_images * weights
    ctf_half = ctf_half * weights

    Ft_y = core.adjoint_slice_volume(
        half_images,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        config.disc_type,
        volume=Ft_y,
        half_image=True,
        half_volume=True,
        max_r=None,
    )
    Ft_ctf = core.adjoint_slice_volume(
        ctf_half,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        config.disc_type,
        volume=Ft_ctf,
        half_image=True,
        half_volume=True,
        max_r=None,
    )
    return Ft_y, Ft_ctf.real


def _broadcast_rows_and_flatten(arr, n_rows):
    """Return ``arr`` as ``(n_rows, -1)``, expanding broadcast rows when needed."""
    if arr.shape[0] == 1 and n_rows != 1:
        arr = jnp.broadcast_to(arr, (n_rows,) + arr.shape[1:])
    return arr.reshape(n_rows, -1)


def _pad_image_weight_matrix_for_fixed_batch(image_weights, current_batch_size, target_batch_size):
    """Pad a ``(n_weights, n_images)`` weight matrix with zero image rows."""
    arr = np.asarray(image_weights)
    if arr.ndim != 2:
        raise ValueError(f"image_weights must have shape (n_weights, n_images), got {arr.shape}")
    if arr.shape[1] != current_batch_size:
        raise ValueError(f"image_weights image axis must match batch size: {arr.shape[1]} != {current_batch_size}")
    if current_batch_size == target_batch_size:
        return arr
    if current_batch_size > target_batch_size:
        raise ValueError(f"batch size {current_batch_size} exceeds target_batch_size {target_batch_size}")
    padded = np.zeros((arr.shape[0], target_batch_size), dtype=arr.dtype)
    padded[:, :current_batch_size] = arr
    return padded


def _can_use_cuda_per_image_backproject(config: ForwardModelConfig) -> bool:
    return custom_cuda_requested() and jax.default_backend() == "gpu" and core.decide_order(config.disc_type) <= 1


# The per-image CUDA path is ~15-100x faster than the Python loop below. Setting this
# flag (truthy) is the only way to run the slow loop on a GPU with custom CUDA enabled —
# guards against silently crawling because libcuda_backproject.so is stale/unbuilt.
_ALLOW_LOOP_FALLBACK_ENV = "RECOVAR_ALLOW_BACKPROJECT_LOOP_FALLBACK"


def _loop_fallback_allowed(config: ForwardModelConfig) -> bool:
    if os.environ.get(_ALLOW_LOOP_FALLBACK_ENV, "").lower() not in {"", "0", "false", "no", "off"}:
        return True  # explicit opt-in
    if jax.default_backend() != "gpu":
        return True  # no GPU: the loop is the only option (e.g. CPU unit tests)
    if not custom_cuda_requested():
        return True  # user explicitly disabled custom CUDA (RECOVAR_DISABLE_CUDA)
    return False


def backproject_weight_sets_from_fft(
    config: ForwardModelConfig,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    image_weights,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    upsample_ctf: bool = True,
):
    """Backproject one image batch into several weighted accumulators.

    ``image_weights`` has shape ``(n_weight_sets, batch_size)``.  The returned
    arrays have shape ``(n_weight_sets, half_volume_size)`` and can be reshaped
    by callers into estimator-specific coefficient axes.
    """
    if _can_use_cuda_per_image_backproject(config):
        return _backproject_weight_sets_from_fft_cuda(
            config,
            images,
            ctf_params,
            rotation_matrices,
            translations,
            noise_variance,
            image_weights,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            upsample_ctf=upsample_ctf,
        )

    if not _loop_fallback_allowed(config):
        raise RuntimeError(
            "backproject_weight_sets_from_fft fell back to the slow Python loop on a GPU "
            f"with custom CUDA enabled (disc_type={config.disc_type!r}, "
            f"order={core.decide_order(config.disc_type)}). The CUDA per-image path only "
            "supports order<=1; for higher orders, or if libcuda_backproject.so is "
            "stale/unbuilt (rebuild with `recovar build_custom_cuda`), set "
            f"{_ALLOW_LOOP_FALLBACK_ENV}=1 to use the slow loop intentionally."
        )

    image_weights = np.asarray(image_weights)
    y_rows = []
    ctf_rows = []
    for weight_idx in range(image_weights.shape[0]):
        y_row, ctf_row = _backproject_weighted_batch_from_fft(
            config,
            images,
            ctf_params,
            rotation_matrices,
            translations,
            noise_variance,
            image_weights[weight_idx],
            Ft_y=None if Ft_y is None else Ft_y[weight_idx],
            Ft_ctf=None if Ft_ctf is None else Ft_ctf[weight_idx],
            upsample_ctf=upsample_ctf,
        )
        y_rows.append(y_row)
        ctf_rows.append(ctf_row)
    return jnp.stack(y_rows, axis=0), jnp.stack(ctf_rows, axis=0)


@eqx.filter_jit
def _backproject_weight_sets_from_fft_cuda(
    config: ForwardModelConfig,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    image_weights,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    upsample_ctf: bool = False,
):
    from recovar.cuda_backproject import per_image_backproject

    half_images, ctf_half = _image_and_ctf_terms_from_fft(
        config,
        images,
        ctf_params,
        translations,
        noise_variance,
        upsample_ctf=upsample_ctf,
    )
    n_images = half_images.shape[0]
    half_images = half_images.reshape(n_images, -1)
    weights = jnp.asarray(image_weights, dtype=half_images.real.dtype)

    if Ft_y is None:
        vol_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(config.volume_shape)
        Ft_y = jnp.zeros((weights.shape[0], int(np.prod(vol_shape))), dtype=half_images.dtype)

    per_image_y = jnp.zeros((n_images, Ft_y.shape[1]), dtype=half_images.dtype)
    per_image_y = per_image_backproject(
        per_image_y,
        half_images,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        order=core.decide_order(config.disc_type),
        half_image=True,
        half_volume=True,
        max_r=None,
    )
    Ft_y = Ft_y + weights.astype(per_image_y.real.dtype) @ per_image_y

    ctf_half = _broadcast_rows_and_flatten(ctf_half, n_images)

    if Ft_ctf is None:
        vol_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(config.volume_shape)
        Ft_ctf = jnp.zeros((weights.shape[0], int(np.prod(vol_shape))), dtype=ctf_half.real.dtype)

    per_image_ctf = jnp.zeros((n_images, Ft_ctf.shape[1]), dtype=Ft_ctf.dtype)
    per_image_ctf = per_image_backproject(
        per_image_ctf,
        ctf_half.astype(Ft_ctf.dtype),
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        order=core.decide_order(config.disc_type),
        half_image=True,
        half_volume=True,
        max_r=None,
    )
    Ft_ctf = Ft_ctf + weights.astype(Ft_ctf.dtype) @ per_image_ctf
    return Ft_y, Ft_ctf.real


def estimate_standard_kernel_volumes(
    experiment_dataset,
    heterogeneity_distances,
    heterogeneity_bins,
    batch_size=None,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    return_lhs_rhs=False,
    heterogeneity_kernel="parabola",
    upsampling_factor=None,
    return_real_space=False,
    use_fast_rfft=False,
):
    bins = heterogeneity_bins

    if (
        getattr(experiment_dataset, "tilt_series_flag", False)
        and hasattr(experiment_dataset, "tilt_particles")
        and heterogeneity_distances.shape[0] != experiment_dataset.n_images
    ):
        per_image = np.empty(experiment_dataset.n_images, dtype=heterogeneity_distances.dtype)
        for p_idx, tilt_inds in enumerate(experiment_dataset.tilt_particles):
            per_image[tilt_inds] = heterogeneity_distances[p_idx]
        heterogeneity_distances = per_image

    inds = np.digitize(heterogeneity_distances, bins, right=True).astype(np.int32)
    n_bins = bins.size
    half_volume_size = _candidate_half_volume_size(experiment_dataset, upsampling_factor)

    rhs_all = np.zeros((n_bins, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_bins, half_volume_size), dtype=experiment_dataset.dtype_real)

    if batch_size is None:
        accum_gb = utils.get_size_in_gb(rhs_all) + utils.get_size_in_gb(lhs_all)
        avail_gb = _effective_heterogeneity_memory_budget(max(1.0, utils.get_gpu_memory_total() - accum_gb))
        batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, avail_gb))
    logger.info("batch size in heterogeneity kernel: %s", batch_size)

    config = _reconstruction_config(experiment_dataset, disc_type, upsampling_factor)
    image_inds_by_bin = [np.flatnonzero(inds == bin_idx).astype(np.int32) for bin_idx in range(n_bins)]
    Ft_y_acc = jnp.zeros(half_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf_acc = jnp.zeros(half_volume_size, dtype=experiment_dataset.dtype_real)
    for bin_idx, image_inds in enumerate(image_inds_by_bin):
        if image_inds.size == 0:
            continue

        Ft_y_acc = jnp.zeros_like(Ft_y_acc)
        Ft_ctf_acc = jnp.zeros_like(Ft_ctf_acc)
        raw_batches = experiment_dataset.iter_batches(
            batch_size,
            noise_model=experiment_dataset.noise,
            noise_half=False,
            indices=image_inds,
        )
        for (
            raw_images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            _particle_indices,
            _image_indices,
        ) in raw_batches:
            _, images, rotation_matrices, translations, ctf_params, noise_variance = _prepare_half_image_batch(
                experiment_dataset,
                raw_images,
                rotation_matrices,
                translations,
                ctf_params,
                noise_variance,
                batch_size=batch_size,
                use_fast_rfft=use_fast_rfft,
            )
            image_weights = jnp.ones((images.shape[0],), dtype=images.real.dtype)
            Ft_y_acc, Ft_ctf_acc = _backproject_weighted_batch_from_fft(
                config,
                images,
                ctf_params,
                rotation_matrices,
                translations,
                noise_variance,
                image_weights,
                Ft_y=Ft_y_acc,
                Ft_ctf=Ft_ctf_acc,
            )

        rhs_all[bin_idx] = np.asarray(Ft_y_acc)
        lhs_all[bin_idx] = np.asarray(Ft_ctf_acc)

    lhs_all, rhs_all = _apply_standard_kernel_weights(lhs_all, rhs_all, bins, heterogeneity_kernel)
    estimates = _postprocess_candidate_estimates(
        lhs_all,
        rhs_all,
        experiment_dataset,
        upsampling_factor=upsampling_factor,
        tau=tau,
        disc_type=disc_type,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        return_real_space=return_real_space,
    )
    if return_lhs_rhs:
        return estimates, np.asarray(lhs_all), np.asarray(rhs_all)

    return estimates
