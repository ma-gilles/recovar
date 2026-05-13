"""Noise-aware one-dimensional local polynomial Fourier regression."""

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.core import mask
from recovar.cuda_backproject import custom_cuda_requested
from recovar.heterogeneity import kernel_regression_reconstruction as kernel_recon
from recovar.reconstruction import relion_functions

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_POLY_DEGREE = 3
DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS = np.asarray(
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    dtype=np.float32,
)
_LOCAL_POLY_EPS = 1e-12


def _coerce_positive_1d_array(values, name):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[1:] == (1, 1):
        out = arr[:, 0, 0]
    else:
        raise NotImplementedError(f"local_poly only supports 1D {name}; got shape {arr.shape}")
    if out.size == 0 or not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return out.astype(np.float32, copy=False)


def coerce_1d_latent_differences(latent_differences):
    """Return latent differences as a flat 1D float32 array."""
    arr = np.asarray(latent_differences, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    else:
        raise NotImplementedError(f"local_poly only supports zdim=1; got shape {arr.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("latent_differences must be finite")
    return out.astype(np.float32, copy=False)


def coerce_1d_latent_coords(zs):
    """Return 1D latent coordinates as a flat float32 array."""
    return coerce_1d_latent_differences(zs)


def coerce_1d_latent_precision(latent_precision):
    """Return 1D latent precision as finite positive float32 values."""
    return _coerce_positive_1d_array(latent_precision, "latent_precision")


def _coerce_bandwidth_multipliers(multipliers):
    if multipliers is None:
        return DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS.copy()
    arr = np.asarray(multipliers, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"local_poly bandwidth multipliers must be finite positive values, got {multipliers}")
    return arr


def local_poly_bandwidth_grid_info_1d(
    latent_diff,
    latent_precision,
    n_min_particles,
    multipliers=None,
):
    """Return ``(multipliers, h_grid, sigma_ref, h_min, r_min)`` for one target."""
    latent_diff = coerce_1d_latent_differences(latent_diff).astype(np.float64)
    latent_precision = coerce_1d_latent_precision(latent_precision).astype(np.float64)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    multipliers = _coerce_bandwidth_multipliers(multipliers)
    latent_std = np.sqrt(1.0 / latent_precision)
    sigma_ref = float(np.median(latent_std))
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise ValueError(f"Invalid local_poly sigma_ref={sigma_ref}")

    n_images = latent_diff.size
    if n_images == 0:
        raise ValueError("No latent points for local_poly bandwidth selection")
    if n_min_particles is None:
        n_min_particles = 1
    closest_idx = max(0, min(int(n_min_particles), n_images) - 1)
    r_min = float(np.partition(np.abs(latent_diff), closest_idx)[closest_idx])
    h_min = max(1.25 * sigma_ref, r_min, _LOCAL_POLY_EPS)
    h_grid = h_min * multipliers.astype(np.float64)
    return multipliers, h_grid.astype(np.float32), sigma_ref, float(h_min), r_min


def local_poly_bandwidth_grid_1d(
    latent_diff,
    latent_precision,
    n_min_particles,
    multipliers=None,
):
    """Return the positive local-polynomial bandwidth grid for one target."""
    return local_poly_bandwidth_grid_info_1d(
        latent_diff,
        latent_precision,
        n_min_particles,
        multipliers=multipliers,
    )[1]


def _gaussian_raw_moments(mean, variance, max_order):
    moments = [np.ones_like(mean, dtype=np.float64)]
    if max_order == 0:
        return moments
    moments.append(mean.astype(np.float64, copy=False))
    for order in range(2, max_order + 1):
        moments.append(mean * moments[order - 1] + (order - 1) * variance * moments[order - 2])
    return moments


def gaussian_window_polynomial_moments_1d(
    latent_diff,
    latent_precision,
    h,
    degree,
    poly_scale=None,
):
    """Closed-form posterior-window moments for 1D local polynomial regression.

    Returns ``(m, M)`` with shapes ``(n_images, degree + 1)`` and
    ``(n_images, degree + 1, degree + 1)``.
    """
    latent_diff = coerce_1d_latent_differences(latent_diff).astype(np.float64)
    latent_precision = coerce_1d_latent_precision(latent_precision).astype(np.float64)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > 4:
        raise ValueError(f"local_poly degree must be between 0 and 4, got {degree}")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be finite and positive, got {h}")
    if poly_scale is None:
        poly_scale = h
    poly_scale = float(poly_scale)
    if not np.isfinite(poly_scale) or poly_scale <= 0:
        raise ValueError(f"poly_scale must be finite and positive, got {poly_scale}")

    variance = 1.0 / latent_precision
    h2 = h * h
    denom = h2 + variance
    alpha = h / np.sqrt(denom) * np.exp(-0.5 * latent_diff**2 / denom)
    mu = latent_diff * h2 / denom
    tau2 = variance * h2 / denom

    t_mean = mu / poly_scale
    t_var = tau2 / (poly_scale * poly_scale)
    raw_moments = _gaussian_raw_moments(t_mean, t_var, 2 * degree)
    factorials = np.asarray([math.factorial(idx) for idx in range(degree + 1)], dtype=np.float64)

    m = np.empty((latent_diff.size, degree + 1), dtype=np.float64)
    M = np.empty((latent_diff.size, degree + 1, degree + 1), dtype=np.float64)
    for r in range(degree + 1):
        m[:, r] = alpha * raw_moments[r] / factorials[r]
        for s in range(degree + 1):
            M[:, r, s] = alpha * raw_moments[r + s] / (factorials[r] * factorials[s])
    return m.astype(np.float32), M.astype(np.float32)


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


def _auto_local_poly_bandwidth_batch_size(n_bandwidths, degree, half_volume_size, complex_dtype, real_dtype):
    n_features = int(degree) + 1
    per_bandwidth_gb = (
        half_volume_size
        * (
            n_features * np.dtype(complex_dtype).itemsize
            + n_features * n_features * np.dtype(real_dtype).itemsize
        )
        / (1024**3)
    )
    if per_bandwidth_gb <= 0:
        return 1
    target_gb = max(1.0, 0.20 * float(utils.get_gpu_memory_total()))
    return max(1, min(int(n_bandwidths), 8, int(target_gb / per_bandwidth_gb)))


def _local_poly_batch_size(experiment_dataset, lhs_all, rhs_all, half_volume_size):
    accum_gb = utils.get_size_in_gb(rhs_all) + utils.get_size_in_gb(lhs_all)
    avail_gb = kernel_recon._effective_heterogeneity_memory_budget(max(1.0, utils.get_gpu_memory_total() - accum_gb))
    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, avail_gb))
    if custom_cuda_requested() and jax.default_backend() == "gpu":
        bytes_per_image = half_volume_size * (
            np.dtype(experiment_dataset.dtype).itemsize + np.dtype(experiment_dataset.dtype_real).itemsize
        )
        target_bytes = utils.get_gpu_memory_total() * (1024**3) * 0.45
        memory_limited = max(1, int(target_bytes // max(1, bytes_per_image)))
        batch_size = max(1, min(batch_size, memory_limited, 256))
    return batch_size


def _local_poly_weight_sets(latent_diff, latent_precision, h_group, degree):
    n_features = int(degree) + 1
    rhs_rows = []
    lhs_rows = []
    for h in h_group:
        m, M = gaussian_window_polynomial_moments_1d(
            latent_diff,
            latent_precision,
            h=float(h),
            degree=degree,
            poly_scale=float(h),
        )
        rhs_rows.extend([m[:, r] for r in range(n_features)])
        lhs_rows.extend([M[:, r, s] for r in range(n_features) for s in range(n_features)])
    return np.asarray(rhs_rows, dtype=np.float32), np.asarray(lhs_rows, dtype=np.float32)


def solve_local_polynomial_fourier_system(
    lhs_all,
    rhs_all,
    experiment_dataset,
    *,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    upsampling_factor=None,
    return_real_space=False,
    solve_chunk_size=262144,
):
    """Solve per-voxel polynomial normal equations and post-process theta_0."""
    lhs_all = np.asarray(lhs_all, dtype=np.float32)
    rhs_all = np.asarray(rhs_all)
    if lhs_all.ndim != 4 or rhs_all.ndim != 3:
        raise ValueError(
            "lhs_all must have shape (n_bandwidths, degree+1, degree+1, half_volume_size) "
            f"and rhs_all (n_bandwidths, degree+1, half_volume_size); got {lhs_all.shape}, {rhs_all.shape}"
        )
    if (
        lhs_all.shape[0] != rhs_all.shape[0]
        or lhs_all.shape[1] != lhs_all.shape[2]
        or lhs_all.shape[1] != rhs_all.shape[1]
    ):
        raise ValueError(f"Incompatible local_poly lhs/rhs shapes: {lhs_all.shape} and {rhs_all.shape}")

    n_bandwidths, n_features, _, half_volume_size = lhs_all.shape
    kernel_type = "triangular" if disc_type == "linear_interp" else "square"
    vol_upsample = kernel_recon._postprocess_upsampling_factor(upsampling_factor)
    upsampled_volume_shape = tuple(3 * [experiment_dataset.volume_shape[0] * vol_upsample])
    expected_half_size = int(
        np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape))
    )
    if half_volume_size != expected_half_size:
        raise ValueError(f"half_volume_size {half_volume_size} does not match expected {expected_half_size}")

    valid_full = (
        mask.get_radial_mask(upsampled_volume_shape, radius=upsampled_volume_shape[0] // 2 - 1)
        .reshape(-1)
        .astype(np.float32)
    )
    valid_half = np.asarray(
        fourier_transform_utils.full_volume_to_half_volume(jnp.asarray(valid_full), upsampled_volume_shape)
    ).reshape(-1).real.astype(np.float32)

    estimates = []
    diag_idx = np.arange(n_features)
    solve_chunk_size = int(max(1, solve_chunk_size))
    for bw_idx in range(n_bandwidths):
        lhs = lhs_all[bw_idx]
        rhs = rhs_all[bw_idx]
        reg_filter = np.asarray(
            relion_functions.adjust_regularization_relion_style(
                jnp.asarray(lhs[0, 0]),
                upsampled_volume_shape,
                tau=None if tau is None else jnp.asarray(tau),
                padding_factor=vol_upsample,
                max_res_shell=None,
                half_volume=True,
            )
        ).reshape(-1).astype(np.float32)
        rho = np.maximum(reg_filter - lhs[0, 0], 0.0).astype(np.float32)
        theta0 = np.zeros(half_volume_size, dtype=rhs.dtype)
        for start in range(0, half_volume_size, solve_chunk_size):
            stop = min(start + solve_chunk_size, half_volume_size)
            gram = np.moveaxis(lhs[:, :, start:stop], -1, 0).astype(np.float32, copy=True)
            gram = 0.5 * (gram + np.swapaxes(gram, 1, 2))
            gram[:, diag_idx, diag_idx] += rho[start:stop, None]
            rhs_chunk = np.moveaxis(rhs[:, start:stop], -1, 0)
            theta = np.linalg.solve(gram, rhs_chunk[..., None])[..., 0]
            theta0[start:stop] = theta[:, 0]
        theta0 = theta0 * valid_half.astype(theta0.real.dtype)
        estimates.append(
            relion_functions.post_process_predivided_fourier_volume(
                jnp.asarray(theta0),
                experiment_dataset.volume_shape,
                vol_upsample,
                kernel=kernel_type,
                use_spherical_mask=use_spherical_mask,
                grid_correct=grid_correct,
                gridding_correct="square",
                kernel_width=1,
                return_real_space=return_real_space,
                input_half_volume=True,
            ).reshape(-1)
        )
    return np.asarray(jnp.stack(estimates, axis=0))


def estimate_local_polynomial_volumes(
    experiment_dataset,
    latent_differences,
    latent_precision,
    h_grid,
    *,
    degree=DEFAULT_LOCAL_POLY_DEGREE,
    batch_size=None,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    return_lhs_rhs=False,
    upsampling_factor=None,
    return_real_space=False,
    use_fast_rfft=False,
    bandwidth_batch_size=None,
):
    """Estimate local-polynomial candidate volumes for one halfset."""
    latent_differences = coerce_1d_latent_differences(latent_differences)
    latent_precision = coerce_1d_latent_precision(latent_precision)
    latent_differences = _expand_tilt_latent_array_to_images(
        experiment_dataset, latent_differences, "latent_differences"
    )
    latent_precision = _expand_tilt_latent_array_to_images(experiment_dataset, latent_precision, "latent_precision")
    if latent_differences.shape != latent_precision.shape:
        raise ValueError(
            "latent_differences and latent_precision must have the same flattened shape, "
            f"got {latent_differences.shape} and {latent_precision.shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > 4:
        raise ValueError(f"local_poly degree must be between 0 and 4, got {degree}")
    h_grid = np.asarray(h_grid, dtype=np.float32).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"h_grid must contain finite positive values, got {h_grid}")

    n_bandwidths = h_grid.size
    n_features = degree + 1
    half_volume_size = kernel_recon._candidate_half_volume_size(experiment_dataset, upsampling_factor)
    rhs_all = np.zeros((n_bandwidths, n_features, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_bandwidths, n_features, n_features, half_volume_size), dtype=experiment_dataset.dtype_real)

    if bandwidth_batch_size is None:
        bandwidth_batch_size = _auto_local_poly_bandwidth_batch_size(
            n_bandwidths,
            degree,
            half_volume_size,
            experiment_dataset.dtype,
            experiment_dataset.dtype_real,
        )
    bandwidth_batch_size = int(max(1, min(n_bandwidths, bandwidth_batch_size)))
    if batch_size is None:
        batch_size = _local_poly_batch_size(experiment_dataset, lhs_all, rhs_all, half_volume_size)

    logger.info("batch size in local_poly heterogeneity kernel: %s", batch_size)
    logger.info("bandwidth batch size in local_poly heterogeneity kernel: %s", bandwidth_batch_size)
    logger.info("local_poly degree=%s h_grid=%s", degree, h_grid)

    config = kernel_recon._reconstruction_config(experiment_dataset, disc_type, upsampling_factor)
    n_rhs_sets = n_features
    n_lhs_sets = n_features * n_features
    for h_start in range(0, n_bandwidths, bandwidth_batch_size):
        h_stop = min(h_start + bandwidth_batch_size, n_bandwidths)
        h_group = h_grid[h_start:h_stop]
        n_h_group = h_group.size
        n_weight_sets = n_h_group * (n_rhs_sets + n_lhs_sets)
        Ft_y_acc = jnp.zeros((n_h_group * n_rhs_sets, half_volume_size), dtype=experiment_dataset.dtype)
        Ft_ctf_acc = jnp.zeros((n_h_group * n_lhs_sets, half_volume_size), dtype=experiment_dataset.dtype_real)
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
            rhs_weights, lhs_weights = _local_poly_weight_sets(
                latent_differences[image_indices],
                latent_precision[image_indices],
                h_group,
                degree,
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
            image_weights = np.concatenate([rhs_weights, lhs_weights], axis=0)
            if image_weights.shape[0] != n_weight_sets:
                raise RuntimeError(
                    f"Unexpected local_poly weight-set count {image_weights.shape[0]} != {n_weight_sets}"
                )
            image_weights = kernel_recon._pad_image_weight_matrix_for_fixed_batch(
                image_weights,
                current_batch_size=current_batch_size,
                target_batch_size=batch_size,
            )
            Ft_all_y, Ft_all_ctf = kernel_recon.backproject_weight_sets_from_fft(
                config,
                images,
                ctf_params,
                rotation_matrices,
                translations,
                noise_variance,
                image_weights,
                Ft_y=jnp.concatenate([Ft_y_acc, jnp.zeros_like(Ft_ctf_acc, dtype=Ft_y_acc.dtype)], axis=0),
                Ft_ctf=jnp.concatenate([jnp.zeros_like(Ft_y_acc, dtype=Ft_ctf_acc.dtype), Ft_ctf_acc], axis=0),
            )
            Ft_y_acc = Ft_all_y[: n_h_group * n_rhs_sets]
            Ft_ctf_acc = Ft_all_ctf[n_h_group * n_rhs_sets :]

        rhs_all[h_start:h_stop] = np.asarray(Ft_y_acc).reshape(n_h_group, n_features, half_volume_size)
        lhs_all[h_start:h_stop] = np.asarray(Ft_ctf_acc).reshape(
            n_h_group,
            n_features,
            n_features,
            half_volume_size,
        )

    estimates = solve_local_polynomial_fourier_system(
        lhs_all,
        rhs_all,
        experiment_dataset,
        tau=tau,
        grid_correct=grid_correct,
        disc_type=disc_type,
        use_spherical_mask=use_spherical_mask,
        upsampling_factor=upsampling_factor,
        return_real_space=return_real_space,
    )
    if return_lhs_rhs:
        return estimates, np.asarray(lhs_all), np.asarray(rhs_all)
    return estimates
