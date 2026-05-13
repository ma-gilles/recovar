"""One-dimensional deconvolved kernel regression for compute_state."""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
from recovar.cuda_backproject import custom_cuda_requested
from recovar.heterogeneity import kernel_regression_reconstruction as kernel_recon

logger = logging.getLogger(__name__)

DEFAULT_DECONV_LAMBDA_GRID = np.asarray(
    [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    dtype=np.float32,
)
_DECONV_EPS = 1e-6


def epanechnikov_deconvolution_kernel_1d(u, lambda_val):
    """Gaussian-error Epanechnikov deconvolution kernel for 1D latent variables."""
    u = np.asarray(u, dtype=np.float64)
    lambda_val = np.asarray(lambda_val, dtype=np.float64)
    lambda_val = np.maximum(lambda_val, _DECONV_EPS)
    abs_u = np.abs(u)
    z = (1j * lambda_val * abs_u - 1.0 / lambda_val) / np.sqrt(2.0)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = (
            -lambda_val
            / np.sqrt(2.0 * np.pi)
            * np.exp(1.0 / (2.0 * lambda_val**2))
            * np.imag(np.exp(-1j * abs_u) * special.wofz(z))
        )
    return weights


def deconvolution_weights_1d(latent_diff, latent_precision, h):
    """Return signed deconvolution weights for one target/bandwidth."""
    return deconvolution_weights_1d_many(latent_diff, latent_precision, np.asarray([h], dtype=np.float64))[0]


def deconvolution_weights_1d_many(latent_diff, latent_precision, h_grid):
    """Return signed deconvolution weights for several bandwidths at once.

    Output shape is ``(len(h_grid), n_images)``.
    """
    latent_diff = np.asarray(latent_diff, dtype=np.float64).reshape(-1)
    latent_precision = np.asarray(latent_precision, dtype=np.float64).reshape(-1)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    h_grid = np.asarray(h_grid, dtype=np.float64).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"h_grid must contain finite positive values, got {h_grid}")

    valid_precision = np.isfinite(latent_precision) & (latent_precision > 0)
    latent_noise_variance = np.zeros_like(latent_precision, dtype=np.float64)
    latent_noise_variance[valid_precision] = 1.0 / latent_precision[valid_precision]
    latent_noise_std = np.full_like(latent_precision, np.inf, dtype=np.float64)
    latent_noise_std[valid_precision] = np.sqrt(latent_noise_variance[valid_precision])

    weights = np.zeros((h_grid.size, latent_diff.size), dtype=np.float64)
    valid_inputs = valid_precision & np.isfinite(latent_diff)
    if np.any(valid_inputs):
        u = latent_diff[valid_inputs][None, :] / h_grid[:, None]
        lambda_i = h_grid[:, None] / np.maximum(latent_noise_std[valid_inputs][None, :], _DECONV_EPS)
        weights[:, valid_inputs] = epanechnikov_deconvolution_kernel_1d(u, lambda_i)
    valid_weights = valid_inputs[None, :] & np.isfinite(weights)
    weights = np.where(valid_weights, weights, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        weights32 = weights.astype(np.float32)
    valid_weights32 = np.isfinite(weights32)
    if not np.all(valid_weights32):
        weights32 = np.where(valid_weights32, weights32, 0.0)
        valid_weights = valid_weights & valid_weights32
    if not np.all(valid_weights):
        logger.warning(
            "Zeroing %d invalid deconvolution weights out of %d",
            int(weights.size - np.count_nonzero(valid_weights)),
            int(weights.size),
        )
    return weights32


def _coerce_deconv_lambda_grid(lambda_grid):
    if lambda_grid is None:
        return DEFAULT_DECONV_LAMBDA_GRID.copy()
    arr = np.asarray(lambda_grid, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"lambda_grid must contain finite positive values, got {lambda_grid}")
    return arr


def deconvolution_bandwidths_1d(latent_precision, lambda_grid=None, sigma_ref=None):
    """Return ``(lambda_grid, h_grid, sigma_ref)`` for finite positive precision."""
    lambda_grid = _coerce_deconv_lambda_grid(lambda_grid)
    if sigma_ref is None:
        latent_precision = np.asarray(latent_precision, dtype=np.float64).reshape(-1)
        valid_precision = np.isfinite(latent_precision) & (latent_precision > 0)
        if not np.any(valid_precision):
            raise ValueError("No finite positive latent precision values for deconvolution bandwidth selection.")
        latent_noise_variance = 1.0 / latent_precision[valid_precision]
        latent_noise_std = np.sqrt(latent_noise_variance)
        sigma_ref = float(np.median(latent_noise_std))
    else:
        sigma_ref = float(sigma_ref)
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise ValueError(f"Invalid deconvolution sigma_ref={sigma_ref}")
    return lambda_grid, (lambda_grid.astype(np.float64) * sigma_ref), sigma_ref


def _auto_deconv_lambda_batch_size(n_lambdas, half_volume_size, complex_dtype, real_dtype):
    """Pick how many lambda candidates to backproject together."""
    per_lambda_gb = (
        half_volume_size * (np.dtype(complex_dtype).itemsize + np.dtype(real_dtype).itemsize) / (1024**3)
    )
    if per_lambda_gb <= 0:
        return 1
    target_gb = max(1.0, 0.20 * float(utils.get_gpu_memory_total()))
    return max(1, min(int(n_lambdas), 64, int(target_gb / per_lambda_gb)))


def _auto_deconv_per_image_batch_size(batch_size, n_weight_sets, half_volume_size, complex_dtype, real_dtype):
    """Cap image batch size for dense per-image backprojection temporaries."""
    if not (custom_cuda_requested() and jax.default_backend() == "gpu"):
        return batch_size

    bytes_per_image = half_volume_size * (np.dtype(complex_dtype).itemsize + np.dtype(real_dtype).itemsize)
    bytes_per_weight_set = bytes_per_image
    target_bytes = utils.get_gpu_memory_total() * (1024**3) * 0.45
    target_bytes -= 2 * n_weight_sets * bytes_per_weight_set
    memory_limited = max(1, int(target_bytes // max(1, bytes_per_image)))
    return max(1, min(int(batch_size), int(memory_limited), 256))


def _deconvolved_batch_size(experiment_dataset, lhs_all, rhs_all, lambda_batch_size, half_volume_size, disc_type):
    accum_gb = utils.get_size_in_gb(rhs_all) + utils.get_size_in_gb(lhs_all)
    avail_gb = kernel_recon._effective_heterogeneity_memory_budget(max(1.0, utils.get_gpu_memory_total() - accum_gb))
    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, avail_gb))
    if custom_cuda_requested() and jax.default_backend() == "gpu" and core.decide_order(disc_type) <= 1:
        return _auto_deconv_per_image_batch_size(
            batch_size,
            lambda_batch_size,
            half_volume_size,
            experiment_dataset.dtype,
            experiment_dataset.dtype_real,
        )
    return batch_size


def _broadcast_rows_and_flatten(arr, n_rows):
    """Return ``arr`` as ``(n_rows, -1)``, expanding broadcast rows when needed."""
    return kernel_recon._broadcast_rows_and_flatten(arr, n_rows)


def _coerce_1d_latent_differences(latent_differences):
    arr = np.asarray(latent_differences, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise NotImplementedError(f"Deconvolved kernel regression only supports zdim=1; got shape {arr.shape}")


def _coerce_1d_latent_precision(latent_precision):
    arr = np.asarray(latent_precision, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 3 and arr.shape[1:] == (1, 1):
        return arr[:, 0, 0]
    raise NotImplementedError(f"Deconvolved kernel regression only supports zdim=1 precision; got shape {arr.shape}")


def _expand_tilt_latent_array_to_images(experiment_dataset, values, name):
    values = np.asarray(values)
    if values.shape[0] == experiment_dataset.n_images:
        return values
    if (
        getattr(experiment_dataset, "tilt_series_flag", False)
        and hasattr(experiment_dataset, "tilt_particles")
        and values.shape[0] != experiment_dataset.n_images
    ):
        per_image = np.empty(experiment_dataset.n_images, dtype=values.dtype)
        for p_idx, tilt_inds in enumerate(experiment_dataset.tilt_particles):
            per_image[tilt_inds] = values[p_idx]
        return per_image
    raise ValueError(f"{name} length {values.shape[0]} does not match dataset n_images={experiment_dataset.n_images}")


def _pad_image_weight_matrix_for_fixed_batch(image_weights, current_batch_size, target_batch_size):
    """Pad a ``(n_weights, n_images)`` weight matrix with zero image rows."""
    return kernel_recon._pad_image_weight_matrix_for_fixed_batch(
        image_weights,
        current_batch_size,
        target_batch_size,
    )


def _can_use_cuda_per_image_backproject(config: ForwardModelConfig) -> bool:
    return kernel_recon._can_use_cuda_per_image_backproject(config)


def _backproject_weight_sets_from_fft(
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
    """Backproject one image batch into several weighted accumulators."""
    return kernel_recon.backproject_weight_sets_from_fft(
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

    half_images, ctf_half = kernel_recon._image_and_ctf_terms_from_fft(
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


def estimate_deconvolved_kernel_volumes(
    experiment_dataset,
    latent_differences,
    latent_precision,
    lambda_grid=None,
    batch_size=None,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    return_lhs_rhs=False,
    upsampling_factor=None,
    return_real_space=False,
    use_fast_rfft=False,
    sigma_ref=None,
    lambda_batch_size=None,
):
    latent_differences = _coerce_1d_latent_differences(latent_differences)
    latent_precision = _coerce_1d_latent_precision(latent_precision)
    latent_differences = _expand_tilt_latent_array_to_images(
        experiment_dataset, latent_differences, "latent_differences"
    )
    latent_precision = _expand_tilt_latent_array_to_images(experiment_dataset, latent_precision, "latent_precision")
    lambda_grid, h_grid, sigma_ref = deconvolution_bandwidths_1d(
        latent_precision, lambda_grid=lambda_grid, sigma_ref=sigma_ref
    )
    n_lambdas = lambda_grid.size
    half_volume_size = kernel_recon._candidate_half_volume_size(experiment_dataset, upsampling_factor)
    rhs_all = np.zeros((n_lambdas, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_lambdas, half_volume_size), dtype=experiment_dataset.dtype_real)

    if lambda_batch_size is None:
        lambda_batch_size = _auto_deconv_lambda_batch_size(
            n_lambdas,
            half_volume_size,
            experiment_dataset.dtype,
            experiment_dataset.dtype_real,
        )
    lambda_batch_size = int(max(1, min(n_lambdas, lambda_batch_size)))
    if batch_size is None:
        batch_size = _deconvolved_batch_size(
            experiment_dataset,
            lhs_all,
            rhs_all,
            lambda_batch_size,
            half_volume_size,
            disc_type,
        )
    logger.info("batch size in deconvolved heterogeneity kernel: %s", batch_size)
    logger.info("lambda batch size in deconvolved heterogeneity kernel: %s", lambda_batch_size)
    logger.info("deconvolution lambda_grid=%s sigma_ref=%s", lambda_grid, sigma_ref)

    config = kernel_recon._reconstruction_config(experiment_dataset, disc_type, upsampling_factor)

    for lambda_start in range(0, n_lambdas, lambda_batch_size):
        lambda_stop = min(lambda_start + lambda_batch_size, n_lambdas)
        h_group = h_grid[lambda_start:lambda_stop]
        n_lambda_group = h_group.size
        Ft_y_acc = jnp.zeros((n_lambda_group, half_volume_size), dtype=experiment_dataset.dtype)
        Ft_ctf_acc = jnp.zeros((n_lambda_group, half_volume_size), dtype=experiment_dataset.dtype_real)
        raw_batches = experiment_dataset.iter_batches(
            batch_size,
            noise_model=experiment_dataset.noise,
            noise_half=False,
        )
        for (
            raw_images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            _particle_indices,
            image_indices,
        ) in raw_batches:
            image_indices = np.asarray(image_indices, dtype=np.int32)
            image_weights = deconvolution_weights_1d_many(
                latent_differences[image_indices],
                latent_precision[image_indices],
                h_group,
            )
            current_batch_size, images, rotation_matrices, translations, ctf_params, noise_variance = (
                kernel_recon._prepare_half_image_batch(
                    experiment_dataset,
                    raw_images,
                    rotation_matrices,
                    translations,
                    ctf_params,
                    noise_variance,
                    batch_size=batch_size,
                    use_fast_rfft=use_fast_rfft,
                )
            )
            image_weights = _pad_image_weight_matrix_for_fixed_batch(
                image_weights,
                current_batch_size=current_batch_size,
                target_batch_size=batch_size,
            )
            Ft_y_acc, Ft_ctf_acc = kernel_recon.backproject_weight_sets_from_fft(
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

        rhs_all[lambda_start:lambda_stop] = np.asarray(Ft_y_acc)
        lhs_all[lambda_start:lambda_stop] = np.asarray(Ft_ctf_acc)

    estimates = kernel_recon._postprocess_candidate_estimates(
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
