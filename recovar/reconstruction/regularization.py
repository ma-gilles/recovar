"""Fourier-shell regularization priors and FSC computation."""

import functools
import logging
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)

# NVTX domain for regularization operations
NVTX_DOMAIN_REG = "regularization"
_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS = 200_000_000
_LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS = 200_000_000


def _relion_round_away_from_zero(x):
    """Mirror RELION's ``ROUND`` macro for NumPy arrays."""
    x = np.asarray(x)
    return np.trunc(np.where(x > 0, x + 0.5, x - 0.5)).astype(np.int64)


def _unscaled_fft_frequency_grid_np(n):
    half = int(n) // 2
    return np.arange(-half, int(n) - half, dtype=np.int64)


def _unscaled_rfft_frequency_grid_np(n):
    return np.arange(0, int(n) // 2 + 1, dtype=np.int64)


@functools.lru_cache(maxsize=64)
def _low_resolution_join_flat_indices(volume_shape, half_layout, lowres_r2_max):
    """Return flat Fourier voxels inside RELION's low-resolution join sphere."""
    volume_shape = tuple(int(s) for s in volume_shape)
    if half_layout:
        layout_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
        x_coords = _unscaled_rfft_frequency_grid_np(volume_shape[2])
    else:
        layout_shape = volume_shape
        x_coords = _unscaled_fft_frequency_grid_np(volume_shape[2])

    z_coords = _unscaled_fft_frequency_grid_np(volume_shape[0])
    y_coords = _unscaled_fft_frequency_grid_np(volume_shape[1])
    y2 = y_coords * y_coords
    x2 = x_coords * x_coords
    lowres_r2_max = int(lowres_r2_max)
    ny = int(layout_shape[1])
    nx = int(layout_shape[2])

    flat_chunks = []
    for zi, z_coord in enumerate(z_coords):
        remaining_after_z = lowres_r2_max - int(z_coord * z_coord)
        if remaining_after_z < 0:
            continue
        y_indices = np.nonzero(y2 <= remaining_after_z)[0]
        z_offset = zi * ny * nx
        for yi in y_indices:
            remaining_after_y = remaining_after_z - int(y2[yi])
            x_indices = np.nonzero(x2 <= remaining_after_y)[0]
            if x_indices.size:
                flat_chunks.append((z_offset + int(yi) * nx + x_indices).astype(np.int32, copy=False))

    if not flat_chunks:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(flat_chunks).astype(np.int32, copy=False)


def _low_resolution_join_host_fallback_enabled_for_size(values_size, join_size):
    mode = os.environ.get("RECOVAR_LOWRES_JOIN_HOST_FALLBACK", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode != "auto":
        logger.warning(
            "Unrecognised RECOVAR_LOWRES_JOIN_HOST_FALLBACK=%r; using auto",
            mode,
        )
    if int(join_size) >= int(values_size):
        return False
    threshold = int(
        os.environ.get(
            "RECOVAR_LOWRES_JOIN_HOST_FALLBACK_MIN_ELEMENTS",
            _LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS,
        )
    )
    return int(values_size) >= threshold


def _low_resolution_join_host_fallback_enabled(values_0, flat_indices):
    return _low_resolution_join_host_fallback_enabled_for_size(np.size(values_0), np.size(flat_indices))


def _delete_if_jax_array(value):
    delete = getattr(value, "delete", None)
    if callable(delete):
        try:
            delete()
        except RuntimeError:
            pass


def _join_half_pair_at_indices_host(values_0, values_1, flat_indices):
    values_0_np = np.array(jax.device_get(values_0), copy=True)
    values_1_np = np.array(jax.device_get(values_1), copy=True)
    flat_indices_np = np.asarray(jax.device_get(flat_indices), dtype=np.intp)

    values_0_flat = values_0_np.reshape(-1)
    values_1_flat = values_1_np.reshape(-1)
    half_scalar = np.asarray(0.5, dtype=values_0_flat.real.dtype)
    if int(flat_indices_np.size) >= int(values_0_flat.size):
        average = ((values_0_flat + values_1_flat) * half_scalar).astype(values_0_flat.dtype, copy=False)
        values_0_np = average.reshape(values_0_np.shape)
        values_1_np = average.reshape(values_1_np.shape).copy()
    else:
        average_at_join = (
            (values_0_flat[flat_indices_np] + values_1_flat[flat_indices_np]) * half_scalar
        ).astype(values_0_flat.dtype, copy=False)
        values_0_flat[flat_indices_np] = average_at_join
        values_1_flat[flat_indices_np] = average_at_join

    _delete_if_jax_array(values_0)
    _delete_if_jax_array(values_1)
    return values_0_np, values_1_np


def _join_half_pair_at_indices(values_0, values_1, flat_indices):
    values_0_flat = values_0.reshape(-1)
    values_1_flat = values_1.reshape(-1)
    if int(flat_indices.size) >= int(values_0_flat.size):
        average = 0.5 * (values_0_flat + values_1_flat)
        return average.reshape(values_0.shape), average.reshape(values_1.shape)

    if _low_resolution_join_host_fallback_enabled(values_0_flat, flat_indices):
        logger.info(
            "Low-resolution half-join using host fallback: elements=%d joined=%d dtype=%s",
            int(values_0_flat.size),
            int(flat_indices.size),
            values_0.dtype,
        )
        return _join_half_pair_at_indices_host(values_0, values_1, flat_indices)

    average_at_join = 0.5 * (values_0_flat[flat_indices] + values_1_flat[flat_indices])
    joined_0 = values_0_flat.at[flat_indices].set(average_at_join)
    joined_1 = values_1_flat.at[flat_indices].set(average_at_join)
    return joined_0.reshape(values_0.shape), joined_1.reshape(values_1.shape)

## Mean prior computation


def compute_batch_prior_quantities(
    rotation_matrices,
    translations,
    CTF_params,
    noise_variance,
    voxel_size,
    dtype,
    volume_shape,
    image_shape,
    grid_size,
    ctf,
    for_whitening=False,
):
    volume_size = np.prod(np.array(volume_shape))
    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape)
    CTF = ctf(CTF_params, image_shape, voxel_size)
    ctf_sq_over_noise = (CTF**2 / noise_variance[None]).reshape(-1)
    diag_mean = jnp.zeros(volume_size, dtype=dtype).at[grid_point_indices.reshape(-1)].add(ctf_sq_over_noise)

    return diag_mean


def compute_prior_quantites(halfset_datasets, cov_noise, batch_size, for_whitening=False):

    reference_dataset = halfset_datasets[0]
    bottom_of_fraction = jnp.zeros(reference_dataset.volume_size, dtype=reference_dataset.dtype)
    for halfset_dataset in halfset_datasets:
        n_images = halfset_dataset.n_images
        # Match main: each halfset iterates in its own local indexing domain.
        for k in range(0, int(np.ceil(n_images / batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min([(k + 1) * batch_size, n_images]))
            indices = jnp.arange(batch_st, batch_end)
            bottom_of_fraction_this = compute_batch_prior_quantities(
                halfset_dataset.rotation_matrices[indices],
                halfset_dataset.translations[indices],
                halfset_dataset.CTF_params[indices],
                cov_noise,
                halfset_dataset.voxel_size,
                halfset_dataset.dtype,
                halfset_dataset.volume_shape,
                halfset_dataset.image_shape,
                halfset_dataset.grid_size,
                halfset_dataset.ctf_evaluator,
                for_whitening,
            )

            bottom_of_fraction += bottom_of_fraction_this

    bottom_of_fraction = bottom_of_fraction.real / len(halfset_datasets)
    return bottom_of_fraction


def compute_relion_prior(
    halfset_datasets,
    cov_noise,
    image0,
    image1,
    batch_size,
    estimate_merged_SNR=False,
    noise_level=None,
    tau2_fudge=1.0,
):
    """Compute a RELION-style spectral prior from two half-set reconstructions.

    Args:
        halfset_datasets: Pair of half-set datasets.
        cov_noise: Scalar noise variance.
        image0: First half-map (Fourier coefficients).
        image1: Second half-map (Fourier coefficients).
        batch_size: GPU batch size for noise estimation.
        estimate_merged_SNR: Estimate SNR from merged map.
        noise_level: Pre-computed noise level (skips estimation if given).
        tau2_fudge: RELION's ``--tau2_fudge`` parameter (default 1.0).
            Multiplies the SSNR before computing tau2.

    Returns:
        Tuple ``(prior, fsc, prior_avg)`` — the spectral prior, FSC
        curve, and averaged prior.
    """

    if noise_level is not None:
        bottom_of_fraction = noise_level
        from_noise_level = True
    else:
        bottom_of_fraction = compute_prior_quantites(halfset_datasets, cov_noise, batch_size, for_whitening=False)
        from_noise_level = False

    return compute_fsc_prior_gpu(
        halfset_datasets[0].volume_shape,
        image0,
        image1,
        bottom_of_fraction,
        estimate_merged_SNR=estimate_merged_SNR,
        from_noise_level=from_noise_level,
        tau2_fudge=tau2_fudge,
    )


def compute_relion_tau2_from_iref_power_spectrum(
    Iref_padded_fourier,
    volume_shape,
    *,
    padding_factor=1,
    current_size=None,
    return_details=False,
):
    """Compute RELION-style tau2 from a previous Iref Fourier volume.

    Mirrors RELION's ``Projector::computeFourierTransformMap`` path by
    converting the reference back to real space and delegating the power-
    spectrum accumulation to the RELION projector binding. The returned
    spectrum is expanded back into RECOVAR's centered Fourier layout.

    The output is in the same Fourier-amplitude scale RECOVAR uses for
    ``mean_variance`` / Wiener regularization. ``current_size`` optionally
    clips the spectrum to the same resolution limit RELION uses when updating
    the projector map.
    """

    volume_shape = tuple(int(s) for s in volume_shape)
    current_size = None if current_size is None else int(current_size)
    from recovar.core import fourier_transform_utils as _ftu
    from recovar.relion_bind._relion_bind_core import compute_fourier_transform_map
    from recovar.utils.helpers import recovar_volume_to_relion

    vol_ft = jnp.asarray(Iref_padded_fourier).reshape(volume_shape)
    vol_real = np.asarray(_ftu.get_idft3(vol_ft).real, dtype=np.float64)
    relion_volume = recovar_volume_to_relion(vol_real)
    _, relion_power_spectrum, *_ = compute_fourier_transform_map(
        relion_volume,
        ori_size=volume_shape[0],
        padding_factor=int(padding_factor),
        current_size=-1 if current_size is None else current_size,
        do_gridding=True,
    )

    # RELION stores ReferenceTau2 on the projector's shell-average scale:
    # getSpectrum(..., POWER_SPECTRUM) is normalized by the padded FFT volume
    # and then multiplied by the projector's normfft/2 factor. For the
    # padding/FFT convention RECOVAR uses here, that lands on an
    # ``ori_size^2 * padding_factor^3 / 8`` scale.
    norm_scale = float(volume_shape[0] ** 2 * (int(padding_factor) ** 3) / 8.0)
    tau2_shells = (np.asarray(relion_power_spectrum, dtype=np.float64) * norm_scale).astype(jnp.float32)
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1)
    )
    radial_distances = jnp.minimum(radial_distances, volume_shape[0] // 2)
    tau2 = tau2_shells[radial_distances]
    if not return_details:
        return tau2
    details = {
        "tau2_shells": tau2_shells,
        "shell_sum": tau2_shells.astype(jnp.float32),
        "shell_count": jnp.ones_like(tau2_shells, dtype=jnp.float32),
    }
    return tau2, details


def get_fsc(vol1, vol2, volume_shape, substract_shell_mean=False, frequency_shift=0):
    """Compute the Fourier Shell Correlation between two volumes.

    Args:
        vol1: First volume (flattened Fourier coefficients).
        vol2: Second volume (flattened Fourier coefficients).
        volume_shape: Tuple ``(N, N, N)`` giving the 3-D grid dimensions.
        substract_shell_mean: Subtract per-shell mean before correlating.
        frequency_shift: Shift applied to frequency indices.

    Returns:
        1-D array of FSC values, one per radial shell.
    """
    return get_fsc_gpu(vol1, vol2, volume_shape, substract_shell_mean, frequency_shift)


@nvtx.annotate("get_fsc_gpu", color="blue", domain=NVTX_DOMAIN_REG)
def get_fsc_gpu(vol1, vol2, volume_shape, substract_shell_mean=False, frequency_shift=0):

    if substract_shell_mean:
        # Center two volumes.
        vol1_avg = average_over_shells(vol1, volume_shape, frequency_shift=frequency_shift)
        vol2_avg = average_over_shells(vol2, volume_shape, frequency_shift=frequency_shift)
        radial_distances = (
            fourier_transform_utils.get_grid_of_radial_distances(
                volume_shape, scaled=False, frequency_shift=frequency_shift
            )
            .astype(int)
            .reshape(-1)
        )
        vol1 -= vol1_avg[radial_distances].reshape(vol1.shape)
        vol2 -= vol2_avg[radial_distances].reshape(vol2.shape)

    top = jnp.conj(vol1) * vol2
    top_avg = average_over_shells(top.real, volume_shape, frequency_shift=frequency_shift)
    bot1 = average_over_shells(jnp.abs(vol1) ** 2, volume_shape, frequency_shift=frequency_shift)
    bot2 = average_over_shells(jnp.abs(vol2) ** 2, volume_shape, frequency_shift=frequency_shift)
    bot = jnp.sqrt(bot1 * bot2)
    fsc = top_avg / bot
    fsc = jnp.where(~jnp.isfinite(fsc), 0, fsc)
    # The generic RECOVAR estimator extends the first measured shell through
    # DC. RELION-specific estimators set FSC[0] = 1 explicitly in their own
    # helpers.
    if fsc.shape[0] > 1:
        fsc = fsc.at[0].set(fsc[1])
    return fsc


@nvtx.annotate("average_over_shells", color="green", domain=NVTX_DOMAIN_REG)
def average_over_shells(input_vec, volume_shape, frequency_shift=0):
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape, scaled=False, frequency_shift=frequency_shift
        )
        .astype(int)
        .reshape(-1)
    )
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0] // 2 - 1)
    return jax_scipy_nd_image_mean(input_vec.reshape(-1), labels=labels, index=indices)


def jax_scipy_nd_image_mean(input, labels=None, index=None):
    if input.dtype == "complex64":
        input = input.astype("complex128")  # jax.numpy.bincount complex64 version seems to be bugged.
        return jax_scipy_nd_image_mean(input.reshape(-1), labels=labels, index=index).astype("complex64")
    return jax_scipy_nd_image_mean_inner(input, labels=labels, index=index)


def jax_scipy_nd_image_mean_inner(input, labels=None, index=None):
    ## TODO fix this stuff
    numpy = jnp
    unique_labels = index
    new_labels = labels

    # counts = numpy.bincount(new_labels,length = index.size )
    counts = numpy.bincount(new_labels, length=index.size)

    # sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    sums = numpy.bincount(new_labels, weights=input.ravel(), length=index.size)

    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = numpy.where(idxs >= int(unique_labels.size), 0, idxs)

    found = unique_labels[idxs] == index
    counts = counts[idxs]
    counts = numpy.where(found, counts, 0)
    sums = sums[idxs]

    sums = numpy.where(sums, sums, 0)
    valid = counts > 0
    safe_counts = numpy.where(valid, counts, 1)
    return numpy.where(valid, sums / safe_counts, 0)


def sum_over_shells(input_vec, volume_shape, frequency_shift=0):
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape, scaled=False, frequency_shift=frequency_shift
        )
        .astype(int)
        .reshape(-1)
    )
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0] // 2 - 1)
    return jax_scipy_nd_image_sum(input_vec.reshape(-1), labels=labels, index=indices)


def jax_scipy_nd_image_sum(input, labels=None, index=None):
    # A jittable simplified scipy.ndimage.sum method
    numpy = jnp
    unique_labels = index
    new_labels = labels

    counts = numpy.bincount(new_labels, length=index.size)
    sums = numpy.bincount(new_labels, weights=input.ravel(), length=index.size)

    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = jnp.where(idxs >= int(unique_labels.size), 0, idxs)

    found = unique_labels[idxs] == index
    counts = counts[idxs]
    counts = jnp.where(found, counts, 0)
    sums = sums[idxs]

    sums = jnp.where(sums, sums, 0)
    return sums


def compute_fsc_prior_gpu(
    volume_shape,
    image0,
    image1,
    bottom_of_fraction=None,
    estimate_merged_SNR=False,
    substract_shell_mean=False,
    frequency_shift=0,
    from_noise_level=False,
    tau2_fudge=1.0,
):
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    # FSC top:
    fsc = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)

    if substract_shell_mean:
        # Set the first 2 to zeros b/c could run in trouble, since killing all signal
        fsc = fsc.at[0:2].set(1)

    fsc = jnp.where(fsc > epsilon, fsc, epsilon)
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon)
    if estimate_merged_SNR:
        fsc = 2 * fsc / (1 + fsc)

    # RELION: SSNR = myfsc / (1 - myfsc) * tau2_fudge
    SNR = fsc / (1 - fsc) * tau2_fudge

    # Bottom of fraction
    if from_noise_level:
        # bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)
        prior_avg = SNR * bottom_of_fraction  # jnp.where( bottom_avg > 0 , SNR * bottom_avg, epsilon )
        logger.warning("Using outdated prior (from_noise_level=True)")
    else:
        bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)
        prior_avg = jnp.where(bottom_avg > 0, SNR / bottom_avg, jax_config.EPSILON)

    # Put back in array
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape, scaled=False, frequency_shift=frequency_shift
        )
        .astype(int)
        .reshape(-1)
    )
    prior = prior_avg[radial_distances]

    return prior, fsc, prior_avg


def downsample_lhs(lhs, volume_shape, upsampling_factor=1):
    # Downsample lhs by a factor of 2
    # radial_distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = -1)
    # lhs_inp_shape = lhs.shape
    kernel = jnp.ones(3 * [2 * upsampling_factor - 1], dtype=jnp.float32)
    kernel = kernel / jnp.sum(kernel)
    lhs = jax.scipy.signal.fftconvolve(lhs, kernel, mode="same")
    lhs = lhs[::upsampling_factor, ::upsampling_factor, ::upsampling_factor]
    lhs = jnp.where(lhs > 0, lhs, 0)
    return lhs * (2 ** len(volume_shape))


@functools.partial(jax.jit, static_argnums=[0, 6, 7])
@nvtx.annotate("compute_fsc_prior_gpu_v2", color="cyan", domain=NVTX_DOMAIN_REG)
def compute_fsc_prior_gpu_v2(
    volume_shape,
    image0,
    image1,
    lhs,
    prior,
    frequency_shift,
    substract_shell_mean=False,
    upsampling_factor=1,
    tau2_fudge=1.0,
):
    """Compute a RELION-style shell regularization tau from half-set FSC.

    This returns a reconstruction regularizer, not the raw shell signal
    variance. See docs/math/ppca_variance_prior_notes.md.
    """
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    # FSC top:
    fsc_raw = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)

    fsc = jnp.where(fsc_raw > epsilon, fsc_raw, epsilon)
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon)

    # RELION: SSNR = myfsc / (1 - myfsc) * tau2_fudge
    SNR = fsc / (1 - fsc) * tau2_fudge

    # Gotta somehow downsample lhs by a factor of 2
    upsampled_volume_shape = tuple([upsampling_factor * i for i in volume_shape])
    lhs = downsample_lhs(
        lhs.reshape(upsampled_volume_shape), upsampled_volume_shape, upsampling_factor=upsampling_factor
    ).reshape(-1)

    if prior is None:
        top = jnp.ones_like(lhs)
        # Safe division: avoid inf when lhs==0 (no-coverage voxels)
        bot = jnp.where(lhs > epsilon, 1 / lhs, 0)
    else:
        safe_prior = jnp.where(prior > 0, prior, jnp.float32(epsilon))
        denom = (lhs + 1 / safe_prior) ** 2
        safe_denom = jnp.where(denom > 0, denom, jnp.float32(1.0))
        top = lhs**2 / safe_denom
        bot = lhs / safe_denom

    sum_top = average_over_shells(top, volume_shape, frequency_shift)
    sum_bot = average_over_shells(bot, volume_shape, frequency_shift)

    prior_avg = jnp.where(sum_top > 0, SNR * sum_bot / sum_top, jax_config.EPSILON).real

    # Put back in array
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape, scaled=False, frequency_shift=frequency_shift
        )
        .astype(int)
        .reshape(-1)
    )
    prior = prior_avg[radial_distances]

    return prior, fsc_raw, prior_avg


@nvtx.annotate("covariance_update_col", color="yellow", domain=NVTX_DOMAIN_REG)
def covariance_update_col(H, B, prior, epsilon=jax_config.EPSILON):
    # H is not divided by sigma.
    safe_prior = jnp.where(prior > 0, prior, jnp.float32(epsilon))
    cov = jnp.where(jnp.abs(H) < epsilon, 0, B / (H + (1 / safe_prior)))
    return cov


def covariance_update_col_with_mask(H, B, prior, volume_mask, valid_idx, volume_shape, epsilon=jax_config.EPSILON):
    # H is not divided by sigma.
    safe_prior = jnp.where(prior > 0, prior, jnp.float32(epsilon))
    cov = (jnp.where(jnp.abs(H) < epsilon, 0, B / (H + (1 / safe_prior))) * valid_idx).reshape(volume_shape)
    cov = fourier_transform_utils.get_dft3(fourier_transform_utils.get_idft3(cov) * volume_mask).reshape(-1)
    return cov


from recovar.reconstruction import relion_functions


@functools.partial(jax.jit, static_argnums=[6, 7, 8, 9, 10, 12, 13, 15])
@nvtx.annotate("prior_iteration_relion_style", color="red", domain=NVTX_DOMAIN_REG)
def prior_iteration_relion_style(
    H0,
    H1,
    B0,
    B1,
    frequency_shift,
    init_regularization,
    substract_shell_mean,
    volume_shape,
    kernel="triangular",
    use_spherical_mask=True,
    grid_correct=True,
    volume_mask=None,
    prior_iterations=3,
    downsample_from_fsc_flag=False,
    tau2_fudge=1.0,
    volume_upsampling_factor=1,
):
    # assert substract_shell_mean == False
    # assert jnp.linalg.norm(frequency_shift) < 1e-8

    H_comb = (H0 + H1) / 2
    prior = init_regularization.real

    def body_fun(prior, fsc):
        cov_col0 = relion_functions.post_process_from_filter_v2(
            H0,
            B0,
            volume_shape,
            volume_upsampling_factor=volume_upsampling_factor,
            tau=prior,
            kernel=kernel,
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct="square",
            kernel_width=1,
            volume_mask=volume_mask,
            tau2_fudge=tau2_fudge,
        )
        cov_col1 = relion_functions.post_process_from_filter_v2(
            H1,
            B1,
            volume_shape,
            volume_upsampling_factor=volume_upsampling_factor,
            tau=prior,
            kernel=kernel,
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct="square",
            kernel_width=1,
            volume_mask=volume_mask,
            tau2_fudge=tau2_fudge,
        )
        prior, fsc, _ = compute_fsc_prior_gpu_v2(
            volume_shape,
            cov_col0,
            cov_col1,
            H_comb,
            prior,
            frequency_shift=frequency_shift,
            substract_shell_mean=substract_shell_mean,
            tau2_fudge=tau2_fudge,
            upsampling_factor=volume_upsampling_factor,
        )
        return prior, fsc

    # Run body_fun without FSC for prior_iterations-1, then one final step with FSC
    def body_fun_no_fsc(i, prior):
        prior, _ = body_fun(prior, None)
        return prior

    if prior_iterations > 0:
        prior = jax.lax.fori_loop(0, prior_iterations, body_fun_no_fsc, prior)
        _, fsc = body_fun(prior, None)
    elif prior_iterations == -1:
        prior = None
        _, fsc = body_fun(prior, None)
    elif prior_iterations == 0:
        _, fsc = body_fun(prior, None)
    else:
        raise ValueError("Prior iterations must be a non-negative integer or -1 (no reg)")

    if downsample_from_fsc_flag:
        B = downsample_from_fsc(B0 + B1, fsc, volume_shape)
    else:
        B = B0 + B1

    cov_col0 = relion_functions.post_process_from_filter_v2(
        H0 + H1,
        B,
        volume_shape,
        volume_upsampling_factor=volume_upsampling_factor,
        tau=prior,
        kernel=kernel,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct="square",
        kernel_width=1,
        volume_mask=volume_mask,
        tau2_fudge=tau2_fudge,
    )

    return cov_col0.reshape(-1), prior, fsc


def _compute_relion_weight_shell_stats(
    weight,
    volume_shape,
    *,
    padding_factor=1,
    r_max=None,
    shell_rounding="round",
    full_half_axis=-1,
    accumulator_volume_shape=None,
):
    """Match RELION's shell-wise weight averaging for tau2 diagnostics.

    Parameters
    ----------
    weight : array-like
        Combined Fourier weight volume (typically ``(Ft_ctf_0 + Ft_ctf_1) / 2``).
        Accepts flat or grid-shaped centered-full arrays, or packed
        half-volume arrays on the same grid.
    volume_shape : tuple[int, int, int]
        Native reconstruction shape ``(N, N, N)``.
    padding_factor : int
        Fourier padding factor. When ``> 1``, ``weight`` must live on the
        padded grid ``(pf*N)^3``.
    r_max : float or None
        RELION reconstruction support radius in native Fourier pixels.  When
        provided, match ``BackProjector::updateSSNRarrays`` by averaging only
        padded voxels with ``r2 < ROUND(r_max * padding_factor)^2``.
    shell_rounding : {"round", "floor"}
        Shell binning rule. RELION's SSNR/tau2 update path uses ``round``
        while the Wiener reconstruct / current-size path uses ``floor``.
    full_half_axis : {-3, -2, -1, 0, 1, 2}
        Axis that corresponds to the RELION half-complex packed dimension
        when ``weight`` is a full Hermitian-expanded volume. Native RECOVAR
        full volumes use the last axis (default). Full volumes expanded from
        RELION x-half storage and transposed into RECOVAR public layout use
        axis 0.

    Returns
    -------
    dict
        ``shell_sum``, ``shell_count``, and ``avg_weight_shells`` arrays with
        RELION-matching shell indexing.
    """
    volume_shape = tuple(int(s) for s in volume_shape)
    ori_half = volume_shape[0] // 2
    n_shells = ori_half + 1

    grid_shape = (
        tuple(d * padding_factor for d in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    native_full_size = int(np.prod(volume_shape))
    half_grid_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(grid_shape)
    full_size = int(np.prod(grid_shape))
    half_size = int(np.prod(half_grid_shape))
    weight_size = int(np.size(weight))
    force_host_shell_stats = (
        padding_factor > 1
        and weight_size in {full_size, half_size}
        and weight_size > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS
    )
    weight_arr = np.asarray(weight).real if force_host_shell_stats else jnp.asarray(weight).real.astype(jnp.float64)
    native_layout = False
    if weight_size == full_size:
        is_half_layout = False
        # Keep shell-stat inputs in RECOVAR's centered grid convention so a
        # full volume and its packed-half view average the same voxels.
        # RELION backprojector FSC has its own axis conversion below.
        weight = weight_arr.reshape(-1)
        relion_grid_shape = grid_shape
    elif weight_size == half_size:
        is_half_layout = True
        weight = weight_arr.reshape(-1)
        relion_grid_shape = grid_shape
    elif padding_factor > 1 and weight_size == native_full_size:
        # Some callers still pass native-grid weights while requesting a
        # padded-shell scaling factor. Treat those as native full-layout input
        # and only apply the oversampling correction to the output.
        native_layout = True
        is_half_layout = False
        weight = weight_arr.reshape(-1)
        relion_grid_shape = volume_shape
    else:
        raise ValueError(
            f"Expected full or half Fourier weight with {full_size} or {half_size} voxels for "
            f"volume_shape={volume_shape} and padding_factor={padding_factor}, got {weight_size}"
        )

    if shell_rounding not in {"round", "floor"}:
        raise ValueError(f"shell_rounding must be 'round' or 'floor', got {shell_rounding!r}")
    full_half_axis = int(full_half_axis)
    if full_half_axis < 0:
        full_half_axis += 3
    if full_half_axis not in {0, 1, 2}:
        raise ValueError(f"full_half_axis must identify one Fourier axis, got {full_half_axis!r}")

    round_fn = (lambda values: jnp.floor(values + 0.5)) if shell_rounding == "round" else jnp.floor
    shell_sum_np = None
    shell_count_np = None

    def _centered_full_half_axis_mask_np(shape, axis):
        axis = int(axis)
        size = int(shape[axis])
        coords = np.arange(-(size // 2), size - size // 2, dtype=np.int64)
        keep = coords >= 0
        if size % 2 == 0:
            keep = keep | (coords == -(size // 2))
        mask_shape = [1] * len(shape)
        mask_shape[axis] = size
        return np.broadcast_to(keep.reshape(mask_shape), shape)

    def _centered_full_half_axis_mask_jax(shape, axis, dtype):
        axis = int(axis)
        size = int(shape[axis])
        idx = jnp.arange(size)
        keep = idx >= size // 2
        if size % 2 == 0:
            keep = keep | (idx == 0)
        keep = keep.astype(dtype)
        mask_shape = [1] * len(shape)
        mask_shape[axis] = size
        return jnp.broadcast_to(keep.reshape(mask_shape), shape)

    def _numpy_bincount_shell_stats(labels, values, mask):
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        values_np = np.asarray(values).reshape(-1)
        mask_np = np.asarray(mask, dtype=bool).reshape(-1)
        labels_included = labels_np[mask_np]
        return (
            np.bincount(labels_included, weights=values_np[mask_np], minlength=n_shells)[:n_shells],
            np.bincount(labels_included, minlength=n_shells).astype(np.float64)[:n_shells],
        )

    if padding_factor > 1 and not native_layout:
        if is_half_layout:
            radial_shape = half_grid_shape
            radial_fn = fourier_transform_utils.get_grid_of_radial_distances_real
            radial_volume_shape = grid_shape
        else:
            radial_shape = relion_grid_shape
            radial_fn = fourier_transform_utils.get_grid_of_radial_distances
            radial_volume_shape = relion_grid_shape

        if int(np.prod(radial_shape)) > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS:
            coords = [
                np.arange(-(int(s) // 2), int(s) - int(s) // 2, dtype=np.float32)
                for s in radial_volume_shape[:-1]
            ]
            if is_half_layout:
                coords.append(np.arange(0, int(radial_volume_shape[-1]) // 2 + 1, dtype=np.float32))
            else:
                last_dim = int(radial_volume_shape[-1])
                coords.append(np.arange(-(last_dim // 2), last_dim - last_dim // 2, dtype=np.float32))
            radial_sq = np.zeros(tuple(radial_shape), dtype=np.float32)
            for axis, grid_axis in enumerate(coords):
                shape = [1] * len(radial_shape)
                shape[axis] = grid_axis.shape[0]
                radial_sq += grid_axis.reshape(shape) ** 2
            padded_dist_np = np.sqrt(radial_sq).astype(np.float32, copy=False)
            del radial_sq
            if r_max is None:
                radius_included_np = np.ones_like(padded_dist_np, dtype=bool)
            else:
                max_r_pad = int(_relion_round_away_from_zero(np.asarray(float(r_max) * padding_factor)))
                radius_included_np = padded_dist_np * padded_dist_np < float(max_r_pad * max_r_pad)
            if shell_rounding == "round":
                shell_index_np = np.floor(padded_dist_np / padding_factor + 0.5).astype(np.int32)
            else:
                shell_index_np = np.floor(padded_dist_np / padding_factor).astype(np.int32)
            shell_index_np = np.minimum(shell_index_np, ori_half)
            if not is_half_layout:
                radius_included_np = radius_included_np & _centered_full_half_axis_mask_np(
                    radial_shape,
                    full_half_axis,
                )
            shell_sum_np, shell_count_np = _numpy_bincount_shell_stats(
                shell_index_np,
                np.asarray(weight).reshape(radial_shape),
                radius_included_np,
            )
        else:
            padded_dist = radial_fn(
                radial_volume_shape,
                scaled=False,
                frequency_shift=0,
                rounded=False,
            ).reshape(-1)
            if r_max is None:
                radius_included = jnp.ones_like(padded_dist, dtype=jnp.float64)
            else:
                max_r_pad = int(_relion_round_away_from_zero(np.asarray(float(r_max) * padding_factor)))
                radius_included = (padded_dist * padded_dist < float(max_r_pad * max_r_pad)).astype(jnp.float64)
            shell_index = jnp.minimum(
                round_fn(padded_dist / padding_factor).astype(jnp.int32),
                ori_half,
            )
            if is_half_layout:
                included = radius_included
            else:
                # RELION iterates the stored half-complex axis only. For
                # native RECOVAR full volumes that axis is last; for full
                # volumes expanded from RELION x-half storage and transposed
                # to public layout it is axis 0.
                half_complex_included = _centered_full_half_axis_mask_jax(
                    radial_shape,
                    full_half_axis,
                    jnp.float64,
                ).reshape(-1)
                included = half_complex_included * radius_included
    else:
        radial_fn = (
            fourier_transform_utils.get_grid_of_radial_distances_real
            if is_half_layout
            else fourier_transform_utils.get_grid_of_radial_distances
        )
        radial_raw = radial_fn(
            volume_shape,
            scaled=False,
            frequency_shift=0,
            rounded=False,
        ).reshape(-1)
        shell_index = round_fn(radial_raw).astype(jnp.int32)
        shell_index = jnp.minimum(shell_index, ori_half)
        if r_max is None:
            included = jnp.ones(weight.shape[0], dtype=jnp.float64)
        else:
            max_r_native = int(_relion_round_away_from_zero(np.asarray(float(r_max))))
            included = (radial_raw * radial_raw < float(max_r_native * max_r_native)).astype(jnp.float64)
        if not is_half_layout:
            included = included * _centered_full_half_axis_mask_jax(
                relion_grid_shape,
                full_half_axis,
                jnp.float64,
            ).reshape(-1)

    if shell_sum_np is None:
        shell_sum = jnp.bincount(shell_index, weights=weight * included, length=n_shells).astype(jnp.float64)
        shell_count = jnp.bincount(shell_index, weights=included, length=n_shells).astype(jnp.float64)
    else:
        shell_sum = jnp.asarray(shell_sum_np, dtype=jnp.float64)
        shell_count = jnp.asarray(shell_count_np, dtype=jnp.float64)
    avg_weight = jnp.where(shell_count > 0, shell_sum / shell_count, 0.0)
    return {
        "shell_sum": shell_sum,
        "shell_count": shell_count,
        "avg_weight_shells": avg_weight,
    }


def compute_relion_tau2_from_weights(
    Ft_ctf_0,
    Ft_ctf_1,
    fsc,
    volume_shape,
    *,
    tau2_fudge=1.0,
    padding_factor=1,
    r_max=None,
    is_whole_instead_of_half=False,
    return_details=False,
    full_half_axis=-1,
    accumulator_volume_shape=None,
    weight_combination="average",
):
    """Compute tau2 from CTF weights and external FSC (RELION's updateSSNRarrays).

    RELION computes tau2 = SSNR * sigma2 where:
    - SSNR = fsc / (1 - fsc) * tau2_fudge
    - sigma2 = count_per_shell / (pf³ * sum_weight_per_shell)
      which is the inverse of the average weight per shell

    When padding_factor > 1, Ft_ctf arrays are at (pf*N)³ or the packed
    half-volume equivalent, while volume_shape is the native (N,N,N).
    Shell averages are computed at native resolution (clamping padded radial
    indices to ori_size/2), matching RELION's updateSSNRarrays which uses
    ``ires = MIN(ires, ori_size/2)``. Output tau2 is at native N³ resolution.
    r_max : float or None
        Reconstruction support radius. RELION ignores padded weights with
        ``r2 >= ROUND(r_max * padding_factor)^2`` in
        ``BackProjector::updateSSNRarrays``; pass the current iteration
        ``current_size // 2`` for auto-refine parity.
    weight_combination : {"average", "sum"}
        How to combine the two input weight arrays before shell averaging.
        Numbered split-half iterations pass one half twice, so the default
        average preserves legacy behavior. RELION's final joined all-data
        iteration first adds the two half BackProjectors, then calls
        ``updateSSNRarrays`` on the combined BackProjector; pass ``"sum"`` for
        that path.
    """
    prior_dtype = jnp.float32
    if weight_combination not in {"average", "sum"}:
        raise ValueError(
            "weight_combination must be 'average' or 'sum', "
            f"got {weight_combination!r}"
        )

    padded_shape = (
        tuple(int(s) * int(padding_factor) for s in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    half_padded_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    large_weight_size = max(int(np.prod(padded_shape)), int(np.prod(half_padded_shape)))
    use_host_shell_stats = (
        int(padding_factor) > 1
        and max(int(np.size(Ft_ctf_0)), int(np.size(Ft_ctf_1))) > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS
        and large_weight_size > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS
    )
    if use_host_shell_stats:
        H0 = np.asarray(Ft_ctf_0).real.astype(np.float32, copy=False)
        H1 = np.asarray(Ft_ctf_1).real.astype(np.float32, copy=False)
        H_comb = H0 + H1
        if weight_combination == "average":
            H_comb = (H_comb * np.float32(0.5)).astype(np.float32, copy=False)
    else:
        H0 = jnp.asarray(Ft_ctf_0).real.astype(prior_dtype)
        H1 = jnp.asarray(Ft_ctf_1).real.astype(prior_dtype)
        H_comb = H0 + H1
        if weight_combination == "average":
            H_comb = H_comb / jnp.asarray(2.0, dtype=prior_dtype)
    shell_stats = _compute_relion_weight_shell_stats(
        H_comb,
        volume_shape,
        padding_factor=padding_factor,
        r_max=r_max,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    shell_sum = shell_stats["shell_sum"]
    shell_count = shell_stats["shell_count"]
    bottom_avg = shell_stats["avg_weight_shells"]

    n_shells = bottom_avg.shape[0]

    # Compute SSNR in float64 to avoid catastrophic cancellation in
    # 1 - fsc when fsc is clamped near 0.999 (float32 loses ~3 digits).
    fsc_raw = jnp.asarray(fsc, dtype=jnp.float64)
    fsc_indices = jnp.minimum(jnp.arange(n_shells), fsc_raw.shape[0] - 1)
    fsc_arr = fsc_raw[fsc_indices]
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    fsc_clamped = jnp.maximum(fsc_arr, epsilon)
    if is_whole_instead_of_half:
        fsc_clamped = jnp.sqrt(2.0 * fsc_clamped / (fsc_clamped + 1.0))
    fsc_clamped = jnp.minimum(fsc_clamped, 1.0 - epsilon)
    ssnr = fsc_clamped / (1.0 - fsc_clamped) * tau2_fudge

    # RELION backprojector.cpp:1061,1075 — updateSSNRarrays multiplies each
    # weight by oversampling_correction = pf³ before shell-averaging, because
    # padding dilutes the per-voxel weight by that factor.  Match here.
    oversampling_correction = padding_factor**3
    sigma2_shells = jnp.where(bottom_avg > 0, 1.0 / (oversampling_correction * bottom_avg), 0.0)
    prior_avg = jnp.where(bottom_avg > 0, ssnr * sigma2_shells, jax_config.EPSILON)
    prior_avg = prior_avg.astype(prior_dtype)

    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled=False, frequency_shift=0)
        .astype(int)
        .reshape(-1)
    )
    prior = prior_avg[radial_distances]
    if not return_details:
        return prior, fsc_clamped

    details = {
        "prior_shells": prior_avg,
        "sigma2_shells": sigma2_shells.astype(prior_dtype),
        "avg_weight_shells": bottom_avg.astype(prior_dtype),
        "shell_sum": shell_sum,
        "shell_count": shell_count,
        "fsc_shells": fsc_clamped,
        "ssnr_shells": ssnr.astype(prior_dtype),
        "oversampling_correction": jnp.asarray(oversampling_correction, dtype=prior_dtype),
        "is_whole_instead_of_half": jnp.asarray(bool(is_whole_instead_of_half)),
        "weight_combination": np.asarray(str(weight_combination)),
    }
    return prior, fsc_clamped, details


def compute_relion_fsc_from_backprojector(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    volume_shape,
    *,
    padding_factor=1,
    r_max=None,
    accumulator_volume_shape=None,
):
    """Compute RELION's gold-standard FSC from backprojector accumulators.

    RELION does not compute the auto-refine FSC used by ``updateSSNRarrays``
    from reconstructed unregularized maps. In
    ``BackProjector::getDownsampledAverage`` it rounds each padded
    backprojector Fourier voxel onto the native grid, divides accumulated
    complex data by accumulated weight, and then
    ``calculateDownSampledFourierShellCorrelation`` bins the two half averages
    with ``ROUND(R)``. RELION's ``ROUND`` is round-half-away-from-zero, not
    NumPy's banker rounding. This helper mirrors that path for centered full
    Fourier arrays used by RECOVAR's dense single-volume M-step.
    """

    volume_shape = tuple(int(s) for s in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(f"Expected cubic 3-D volume_shape, got {volume_shape}")
    n = volume_shape[0]
    pf = int(padding_factor)
    if pf <= 0:
        raise ValueError(f"padding_factor must be positive, got {padding_factor}")
    padded_shape = (
        tuple(d * pf for d in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    full_size = int(np.prod(padded_shape))
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    half_size = int(np.prod(half_shape))

    def _packed_half_to_full_numpy(arr_np):
        """Expand RECOVAR's centered packed half-volume layout on host."""
        half_grid = np.asarray(arr_np).reshape(half_shape)
        n0, n1, n2 = padded_shape
        ic2 = n2 // 2
        if n2 % 2 == 0:
            packed_idx = np.concatenate([np.arange(ic2, n2, dtype=np.int64), np.asarray([0], dtype=np.int64)])
            redundant = np.arange(1, ic2, dtype=np.int64)
        else:
            packed_idx = np.arange(ic2, n2, dtype=np.int64)
            redundant = np.arange(0, ic2, dtype=np.int64)

        full_grid = np.zeros(padded_shape, dtype=half_grid.dtype)
        full_grid[:, :, packed_idx] = half_grid
        if redundant.size:
            partner_i0 = (n0 - (n0 % 2) - np.arange(n0, dtype=np.int64)) % n0
            partner_i1 = (n1 - (n1 % 2) - np.arange(n1, dtype=np.int64)) % n1
            conj_partner = np.conj(half_grid[partner_i0[:, None], partner_i1[None, :], :])
            source_cols = ic2 - redundant
            full_grid[:, :, redundant] = conj_partner[:, :, source_cols]
        return full_grid

    def _as_padded_full(arr, name):
        arr_np = np.asarray(arr)
        if arr_np.size == full_size:
            return arr_np.reshape(padded_shape)
        if arr_np.size == half_size:
            return _packed_half_to_full_numpy(arr_np)
        raise ValueError(
            f"{name} must be centered full Fourier data with {full_size} entries or packed half "
            f"Fourier data with {half_size} entries for volume_shape={volume_shape}, "
            f"padding_factor={padding_factor}; got {arr_np.size}"
        )

    data0 = _as_padded_full(Ft_y_0, "Ft_y_0")
    data1 = _as_padded_full(Ft_y_1, "Ft_y_1")
    weight0 = _as_padded_full(Ft_ctf_0, "Ft_ctf_0").real
    weight1 = _as_padded_full(Ft_ctf_1, "Ft_ctf_1").real

    axes = [
        np.asarray(fourier_transform_utils.get_1d_frequency_grid(s, scaled=False), dtype=np.float64)
        for s in padded_shape
    ]
    # RECOVAR stores centered Fourier volumes as (z, y, x), but the RELION
    # BackProjector's compact half-axis is its logical x coordinate.  In the
    # dense EM rotation convention that logical RELION x corresponds to
    # RECOVAR axis 0 after the CUDA rotation-row swap.  Interpret the saved
    # full accumulator as (relion_y, relion_x, relion_z) before mirroring
    # BackProjector::getDownsampledAverage.  This is source-level layout
    # emulation; the shell bins are rotationally invariant, but the compact
    # half-axis selection is not.
    relion_z, relion_y, relion_x = np.meshgrid(axes[1], axes[2], axes[0], indexing="ij")
    dz = _relion_round_away_from_zero(relion_z / pf)
    dy = _relion_round_away_from_zero(relion_y / pf)
    dx = _relion_round_away_from_zero(relion_x / pf)
    data0 = np.transpose(data0, (1, 2, 0))
    data1 = np.transpose(data1, (1, 2, 0))
    weight0 = np.transpose(weight0, (1, 2, 0))
    weight1 = np.transpose(weight1, (1, 2, 0))

    half = n // 2
    max_shell = half if r_max is None else int(r_max)
    down_radius = max_shell + 1
    down_size = 2 * down_radius + 1
    down_xsize = down_size // 2 + 1
    valid = (
        (dz >= -down_radius)
        & (dz <= down_radius)
        & (dy >= -down_radius)
        & (dy <= down_radius)
        & (dx >= 0)
        & (dx < down_xsize)
    )
    labels = ((dz[valid] + down_radius) * down_size + (dy[valid] + down_radius)) * down_xsize + dx[valid]
    labels = labels.reshape(-1)
    minlength = down_size * down_size * down_xsize

    def _downsample_average(data, weight):
        weight_flat = weight[valid].reshape(-1)
        data_flat = data[valid].reshape(-1)
        sum_weight = np.bincount(labels, weights=weight_flat, minlength=minlength)
        sum_real = np.bincount(labels, weights=data_flat.real, minlength=minlength)
        sum_imag = np.bincount(labels, weights=data_flat.imag, minlength=minlength)
        avg = sum_real + 1j * sum_imag
        nonzero = sum_weight > 0.0
        avg[nonzero] /= sum_weight[nonzero]
        avg[~nonzero] = 0.0
        return avg.reshape((down_size, down_size, down_xsize)), sum_weight.reshape((down_size, down_size, down_xsize))

    avg0, down_weight0 = _downsample_average(data0, weight0)
    avg1, down_weight1 = _downsample_average(data1, weight1)

    z_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    y_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    x_axis = np.arange(0, down_xsize, dtype=np.float64)
    rz, ry, rx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    radius = np.sqrt(rz * rz + ry * ry + rx * rx)
    shell = _relion_round_away_from_zero(radius)
    shell_count = half + 1
    # RELION's calculateDownSampledFourierShellCorrelation bins by ROUND(R),
    # but first skips samples with exact native radius R > r_max.
    shell_valid = radius <= float(max_shell)
    shell_labels = shell[shell_valid].reshape(-1)
    avg0_flat = avg0[shell_valid].reshape(-1)
    avg1_flat = avg1[shell_valid].reshape(-1)

    numerator = np.bincount(shell_labels, weights=(np.conj(avg0_flat) * avg1_flat).real, minlength=shell_count)
    denom0 = np.bincount(shell_labels, weights=np.abs(avg0_flat) ** 2, minlength=shell_count)
    denom1 = np.bincount(shell_labels, weights=np.abs(avg1_flat) ** 2, minlength=shell_count)
    fsc = np.zeros(shell_count, dtype=np.float64)
    nonzero = (denom0 * denom1) > 0.0
    fsc[nonzero] = numerator[nonzero] / np.sqrt(denom0[nonzero] * denom1[nonzero])
    fsc[0] = 1.0

    dump_dir = os.environ.get("RECOVAR_MSTEP_FSC_DUMP_DIR")
    if dump_dir:
        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        tag = os.environ.get("RECOVAR_MSTEP_FSC_DUMP_TAG", "recovar")
        np.savetxt(
            pathlib.Path(dump_dir) / f"{tag}_downsampled_fsc.txt",
            np.column_stack([np.arange(shell_count), numerator, denom0, denom1, fsc]),
            header="shell num den1 den2 fsc",
        )
        if os.environ.get("RECOVAR_MSTEP_FSC_DUMP_AVG", "").lower() in {"1", "true", "yes", "on"}:
            coords = np.column_stack(
                [
                    rz[shell_valid].reshape(-1).astype(np.int64),
                    ry[shell_valid].reshape(-1).astype(np.int64),
                    rx[shell_valid].reshape(-1).astype(np.int64),
                ]
            )
            for suffix, avg, down_weight in (
                ("half1", avg0, down_weight0),
                ("half2", avg1, down_weight1),
            ):
                avg_flat = avg[shell_valid].reshape(-1)
                np.savetxt(
                    pathlib.Path(dump_dir) / f"{tag}_downsampled_avg_{suffix}.txt",
                    np.column_stack(
                        [
                            coords,
                            avg_flat.real,
                            avg_flat.imag,
                            down_weight[shell_valid].reshape(-1),
                        ]
                    ),
                    header=(
                        f"xsize {down_xsize} ysize {down_size} zsize {down_size} "
                        f"xinit 0 yinit {-down_radius} zinit {-down_radius}\n"
                        "k i j real imag weight"
                    ),
                )
    return jnp.asarray(fsc, dtype=jnp.float32)


@functools.lru_cache(maxsize=16)
def _relion_rfft_shell_grid(volume_shape):
    """Return RELION-style shell labels for NumPy ``rfftn`` volume spectra."""
    volume_shape = tuple(int(s) for s in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(f"Expected cubic 3-D volume_shape, got {volume_shape}")
    z = np.fft.fftfreq(volume_shape[0]) * volume_shape[0]
    y = np.fft.fftfreq(volume_shape[1]) * volume_shape[1]
    x = np.fft.rfftfreq(volume_shape[2]) * volume_shape[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    radius_sq = zz * zz + yy * yy + xx * xx
    shell = _relion_round_away_from_zero(np.sqrt(radius_sq))
    shell_count = volume_shape[2] // 2 + 1
    valid = shell < shell_count
    return shell.astype(np.int64), valid, radius_sq


def _relion_fsc_from_real_maps_numpy(map0, map1):
    """Mirror RELION ``getFSC`` for two real-space maps using NumPy FFTs."""
    map0 = np.asarray(map0, dtype=np.float64)
    map1 = np.asarray(map1, dtype=np.float64)
    if map0.shape != map1.shape:
        raise ValueError(f"map shapes must match, got {map0.shape} and {map1.shape}")
    shell, valid, _ = _relion_rfft_shell_grid(tuple(map0.shape))
    shell_count = map0.shape[-1] // 2 + 1

    fft_axes = (0, 1, 2)
    ft0 = np.fft.rfftn(map0, axes=fft_axes)
    ft1 = np.fft.rfftn(map1, axes=fft_axes)
    labels = shell[valid].reshape(-1)
    ft0_flat = ft0[valid].reshape(-1)
    ft1_flat = ft1[valid].reshape(-1)
    numerator = np.bincount(labels, weights=(np.conj(ft0_flat) * ft1_flat).real, minlength=shell_count)
    denom0 = np.bincount(labels, weights=np.abs(ft0_flat) ** 2, minlength=shell_count)
    denom1 = np.bincount(labels, weights=np.abs(ft1_flat) ** 2, minlength=shell_count)
    fsc = np.zeros(shell_count, dtype=np.float64)
    nonzero = (denom0 * denom1) > 0.0
    fsc[nonzero] = numerator[nonzero] / np.sqrt(denom0[nonzero] * denom1[nonzero])
    fsc = np.where(np.isfinite(fsc), fsc, 0.0)
    if fsc.size:
        fsc[0] = 1.0
    return fsc


def _relion_randomize_phases_beyond_numpy(map_real, index, rng):
    """Mirror RELION ``randomizePhasesBeyond`` for a real-space map."""
    map_real = np.asarray(map_real, dtype=np.float64)
    _, _, radius_sq = _relion_rfft_shell_grid(tuple(map_real.shape))
    fft_axes = (0, 1, 2)
    ft = np.fft.rfftn(map_real, axes=fft_axes)
    randomize = radius_sq >= int(index) * int(index)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=ft.shape)
    randomized = np.abs(ft) * (np.cos(phases) + 1j * np.sin(phases))
    ft = np.where(randomize, randomized, ft)
    return np.fft.irfftn(ft, s=map_real.shape, axes=fft_axes).real


def compute_relion_solvent_corrected_true_fsc(
    half_map0_real,
    half_map1_real,
    solvent_mask,
    *,
    current_size=None,
    randomize_fsc_at=0.8,
    rng_seed=0,
    return_details=False,
):
    """Compute RELION's solvent-corrected gold-standard FSC for auto-refine.

    RELION writes unfiltered split-half maps, computes their unmasked FSC,
    masks both maps with the solvent mask, randomizes phases beyond the first
    unmasked shell below 0.8, and then applies Richard Henderson's corrected
    FSC formula before feeding the curve to ``updateSSNRarrays``.
    """
    half_map0_real = np.asarray(half_map0_real, dtype=np.float64)
    half_map1_real = np.asarray(half_map1_real, dtype=np.float64)
    if half_map0_real.shape != half_map1_real.shape:
        raise ValueError(
            f"half-map shapes must match, got {half_map0_real.shape} and {half_map1_real.shape}"
        )
    if solvent_mask is not None:
        solvent_mask = np.asarray(solvent_mask, dtype=np.float64)
        if solvent_mask.shape != half_map0_real.shape:
            raise ValueError(
                f"solvent_mask shape {solvent_mask.shape} does not match half-map shape {half_map0_real.shape}"
            )

    fsc_unmasked = _relion_fsc_from_real_maps_numpy(half_map0_real, half_map1_real)
    randomize_at = -1
    for idx in range(1, fsc_unmasked.size):
        if fsc_unmasked[idx] < float(randomize_fsc_at):
            randomize_at = idx
            break

    if solvent_mask is None or randomize_at <= 0:
        fsc_masked = fsc_unmasked.copy()
        fsc_random_masked = np.zeros_like(fsc_unmasked)
        fsc_true = fsc_unmasked.copy()
    else:
        masked0 = half_map0_real * solvent_mask
        masked1 = half_map1_real * solvent_mask
        fsc_masked = _relion_fsc_from_real_maps_numpy(masked0, masked1)

        rng = np.random.default_rng(int(rng_seed))
        randomized0 = _relion_randomize_phases_beyond_numpy(half_map0_real, randomize_at, rng)
        randomized1 = _relion_randomize_phases_beyond_numpy(half_map1_real, randomize_at, rng)
        fsc_random_masked = _relion_fsc_from_real_maps_numpy(randomized0 * solvent_mask, randomized1 * solvent_mask)

        if fsc_masked[0] <= 0.0:
            fsc_masked[0] = 1.0
        if fsc_unmasked[0] <= 0.0:
            fsc_unmasked[0] = 1.0
        if fsc_random_masked[0] <= 0.0:
            fsc_random_masked[0] = 1.0

        fsc_true = np.empty_like(fsc_masked)
        handoff = int(randomize_at) + 2
        fsc_true[:handoff] = fsc_masked[:handoff]
        fsct = fsc_masked[handoff:]
        fscn = fsc_random_masked[handoff:]
        denom = 1.0 - fscn
        corrected = np.where((fscn > fsct) | (denom <= 0.0), 0.0, (fsct - fscn) / denom)
        fsc_true[handoff:] = corrected

    fsc_true = np.where(np.isfinite(fsc_true), fsc_true, 0.0)
    if current_size is not None:
        zero_start = int(current_size) // 2 + 1
        if zero_start < fsc_true.size:
            fsc_true[zero_start:] = 0.0

    out = jnp.asarray(fsc_true, dtype=jnp.float32)
    if not return_details:
        return out
    details = {
        "randomize_at": int(randomize_at),
        "fsc_unmasked": fsc_unmasked.astype(np.float32),
        "fsc_masked": fsc_masked.astype(np.float32),
        "fsc_random_masked": fsc_random_masked.astype(np.float32),
        "fsc_true": fsc_true.astype(np.float32),
    }
    return out, details


def downsample_from_fsc(array, fsc, volume_shape):
    from recovar.heterogeneity import locres

    # Accept both NumPy and JAX arrays.
    fsc = jnp.asarray(fsc)
    array = jnp.asarray(array)
    fsc_above_threshold = fsc >= 0.0001
    # Sometimes the FSC dips at low resolution. We want to avoid that case.
    fsc_above_threshold = fsc_above_threshold.at[:16].set(True)
    ires_max = locres.find_first_zero_in_bool(fsc_above_threshold)

    downsample_ar = jnp.where(jnp.arange(fsc.size) < ires_max, fsc, 0)
    distances = fourier_transform_utils.get_grid_of_radial_distances(volume_shape)
    fsc_mask = downsample_ar[distances]
    return array * fsc_mask.reshape(-1)


# ---------------------------------------------------------------------------
# RELION-style data_vs_prior resolution criterion (C4)
# ---------------------------------------------------------------------------


def compute_data_vs_prior(
    Ft_ctf,
    tau2,
    volume_shape,
    padding_factor=1,
    tau2_fudge=1.0,
    current_size=None,
    full_half_axis=-1,
    accumulator_volume_shape=None,
):
    """Compute RELION's data_vs_prior ratio per radial shell.

    RELION determines the effective resolution from the shell where
    ``data_vs_prior`` drops below 1.0, rather than from FSC < 0.143.

    The ratio is defined as::

        data_vs_prior[ires] = avg_Fweight[ires] * tau2_fudge * tau2[ires] * padding_factor**3

    where ``avg_Fweight`` is the shell-averaged Fourier weight from
    backprojection (the real part of ``Ft_ctf``), and ``tau2`` is the
    spectral signal prior.

    Parameters
    ----------
    Ft_ctf : jnp.ndarray
        Fourier-space CTF weight array in either centered full-volume or
        packed half-volume layout. The real part gives the per-voxel
        weight (sum of CTF^2 / noise).
    tau2 : jnp.ndarray, shape (n_shells,)
        Spectral signal prior (one value per radial shell).
    volume_shape : tuple of int
        3-D volume dimensions, e.g. ``(N, N, N)``.
    padding_factor : int or float
        Oversampling / padding factor (1 for no padding).
    tau2_fudge : float
        RELION's ``--tau2_fudge`` parameter (default 1.0).
    current_size : int or None
        Optional current image size. When provided, shells beyond
        ``current_size // 2`` are zeroed to match RELION's current-resolution
        truncation during growth updates.

    Returns
    -------
    jnp.ndarray, shape (n_shells,)
        Per-shell data_vs_prior ratio.
    """
    avg_weight = _compute_relion_weight_shell_stats(
        Ft_ctf,
        volume_shape,
        padding_factor=padding_factor,
        r_max=current_size // 2 if current_size is not None else None,
        shell_rounding="round",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
    )["avg_weight_shells"].astype(jnp.asarray(tau2).dtype)
    tau2 = jnp.asarray(tau2)
    if tau2.shape[0] != avg_weight.shape[0]:
        shell_count = min(int(tau2.shape[0]), int(avg_weight.shape[0]))
        avg_weight = avg_weight[:shell_count]
        tau2 = tau2[:shell_count]
    oversampling_correction = padding_factor**3
    data_vs_prior = avg_weight * tau2_fudge * tau2 * oversampling_correction
    if current_size is not None:
        shell_limit = min(int(current_size) // 2, int(data_vs_prior.shape[0]) - 1)
        if shell_limit + 1 < int(data_vs_prior.shape[0]):
            data_vs_prior = data_vs_prior.at[shell_limit + 1 :].set(0)
    return data_vs_prior


def resolution_from_data_vs_prior(
    data_vs_prior,
    *,
    allow_high_res_recovery=False,
    recovery_margin_shells=3,
):
    """Find the resolution shell where data_vs_prior drops below 1.0.

    Scans from shell 1 outward (skipping DC) and returns the last shell
    where ``data_vs_prior >= 1.0``.

    When ``allow_high_res_recovery`` is enabled, mimic RELION's
    ``updateCurrentResolution()`` behavior for split-half auto-refine: if the
    curve dips below 1.0 and then rises again at substantially higher shells,
    keep the later shell instead of the first crossing. This handles the
    phase-randomization / tight-mask artefact check in RELION.

    Parameters
    ----------
    data_vs_prior : array-like, shape (n_shells,)
        Per-shell data_vs_prior ratio from :func:`compute_data_vs_prior`.

    allow_high_res_recovery : bool, optional
        Enable RELION's high-resolution recheck.
    recovery_margin_shells : int, optional
        Minimum number of shells by which the recovered high-resolution shell
        must exceed the first crossing.

    Returns
    -------
    int
        Shell index of the resolution limit.  Returns ``len(data_vs_prior) - 1``
        if data_vs_prior never drops below 1.0.
    """
    dvp = np.asarray(data_vs_prior)
    for ires in range(1, len(dvp)):
        if dvp[ires] < 1.0:
            maxres = ires - 1
            break
    else:
        maxres = len(dvp) - 1

    if allow_high_res_recovery:
        recovered = maxres
        for ires2 in range(len(dvp) - 1, maxres - 1, -1):
            if dvp[ires2] > 1.0:
                recovered = ires2
                break
        if recovered > maxres + int(recovery_margin_shells):
            maxres = recovered

    return maxres


# ---------------------------------------------------------------------------
# RELION auto-refine resolution / current-size helpers
# ---------------------------------------------------------------------------


def fsc_to_relion_ssnr(fsc, tau2_fudge=1.0, is_whole_instead_of_half=False):
    """Convert an FSC curve to RELION's data-vs-prior / SSNR curve.

    In gold-standard auto-refine, RELION updates ``data_vs_prior`` from the
    half-map FSC by converting each shell's FSC into an SSNR value. The shell
    where this curve drops below ``1`` is the same shell where the FSC drops
    below ``0.5``.
    """
    fsc = jnp.asarray(fsc)
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    myfsc = jnp.maximum(fsc, epsilon)
    if is_whole_instead_of_half:
        myfsc = jnp.sqrt(2.0 * myfsc / (myfsc + 1.0))
    myfsc = jnp.minimum(myfsc, 1.0 - epsilon)
    return tau2_fudge * myfsc / (1.0 - myfsc)


def first_shell_below_threshold(values, threshold):
    """Return the first shell index below ``threshold``.

    RELION's shell scans start at shell 1 (skipping DC). If no shell drops
    below the threshold, return the last available shell.
    """
    arr = np.asarray(values)
    for i in range(1, len(arr)):
        if arr[i] < threshold:
            return i
    return len(arr) - 1


def compute_relion_incr_size_from_fsc(fsc, default=10):
    """RELION auto-refine shell-growth heuristic from the current FSC curve.

    RELION enlarges ``incr_size`` to at least ``fsc0143 - fsc05 + 5`` after the
    half-map comparison, where ``fsc05`` and ``fsc0143`` are the first shells
    where the FSC drops below 0.5 and 0.143, respectively.
    """
    fsc05 = first_shell_below_threshold(fsc, 0.5)
    fsc0143 = first_shell_below_threshold(fsc, 0.143)
    return max(int(default), int(fsc0143 - fsc05 + 5))


def update_relion_growth_state_from_fsc(
    fsc,
    current_size,
    *,
    incr_size=10,
    has_high_fsc_at_limit=False,
):
    """Update RELION's sticky current-size growth state from the FSC curve.

    RELION keeps ``incr_size`` as a non-decreasing value across iterations and
    only flips ``has_high_fsc_at_limit`` from false to true once. This helper
    mirrors the MPI auto-refine update in ``ml_optimiser_mpi.cpp``.
    """
    fsc = np.asarray(fsc)
    next_incr_size = compute_relion_incr_size_from_fsc(fsc, default=int(incr_size))

    if len(fsc) == 0:
        return next_incr_size, bool(has_high_fsc_at_limit)

    limit_shell = min(max(int(current_size) // 2 - 1, 0), len(fsc) - 1)
    high_fsc_now = bool(float(fsc[limit_shell]) > 0.2)
    return next_incr_size, bool(has_high_fsc_at_limit or high_fsc_now)


# ---------------------------------------------------------------------------
# RELION-style current_size growth logic (C5)
# ---------------------------------------------------------------------------


def compute_current_size_relion(resolution_shell, ori_size, ave_Pmax=0.0, has_high_fsc_at_limit=False, incr_size=10):
    """Compute the next current_size using RELION's growth logic.

    RELION grows current_size beyond the current resolution limit.  If
    the average maximum posterior probability (``ave_Pmax``) exceeds 0.1
    AND the FSC is still high at the resolution limit, the jump is 25%
    of ``ori_size / 2`` (aggressive growth).  Otherwise the jump is
    ``incr_size`` shells (conservative).

    The result is clamped to ``ori_size``.

    Parameters
    ----------
    resolution_shell : int
        Current resolution shell index (e.g. from
        :func:`resolution_from_data_vs_prior` or FSC-based estimate).
    ori_size : int
        Original image size in pixels (diameter, e.g. 128).
    ave_Pmax : float
        Average of the per-image maximum posterior probability.
        Typical range 0-1; early iterations have low values.
    has_high_fsc_at_limit : bool
        True if the FSC is still significantly above 0 at the current
        resolution limit (indicating the data supports higher resolution).
    incr_size : int
        Default shell increment when conditions for aggressive growth
        are not met.

    Returns
    -------
    int
        New current_size in pixels (diameter).
    """
    maxres = resolution_shell
    if ave_Pmax > 0.1 and has_high_fsc_at_limit:
        maxres += round(0.25 * ori_size / 2)
    else:
        maxres += incr_size
    return min(2 * maxres, ori_size)


prior_iteration_relion_style_batch = jax.vmap(
    prior_iteration_relion_style,
    # 14 positional args from
    # ``compute_covariance_regularization_relion_style``: H0, H1, B0, B1,
    # frequency_shift, init_regularization (all batched: 0), then
    # substract_shell_mean, volume_shape, kernel, use_spherical_mask,
    # grid_correct, volume_mask, prior_iterations, downsample_from_fsc_flag
    # (all broadcast: None). The trailing ``tau2_fudge`` and
    # ``volume_upsampling_factor`` are taken from their defaults and are
    # NOT passed positionally.
    in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
)

batch_average_over_shells = jax.vmap(average_over_shells, in_axes=(0, None, None))


def join_halves_at_low_resolution(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    volume_shape,
    voxel_size,
    grid_size,
    low_resol_join_halves_angstrom,
    current_resolution_angstrom=None,
    padding_factor=None,
):
    """RELION's ``--low_resol_join_halves`` operation on Fourier accumulators.

    Mirrors ``MlOptimiserMpi::joinTwoHalvesAtLowResolution`` in
    ``relion/src/ml_optimiser_mpi.cpp:3112-3219``: at low resolutions where
    the half-set reconstructions are unreliably independent, RELION
    averages the **backprojection accumulators** (``data`` ↔ ``Ft_y`` and
    ``weight`` ↔ ``Ft_ctf``) of the two halves at every Fourier voxel
    inside a low-resolution sphere, then writes the average back into both
    halves before doing the Wiener solve. This forces the iter's two
    half-maps to share their low-frequency content, preventing the
    half-sets from diverging in orientation space at SNR-poor low shells.

    The joining radius (in shells) is set by the LARGER (lower-frequency)
    of:
        - ``low_resol_join_halves_angstrom`` (the user/GUI default 40 Å), and
        - ``current_resolution_angstrom`` (the iter's resolution estimate)
    so that the joining radius never exceeds the actual resolution of the
    map (which would join shells where the FSC is genuinely high).
    Concretely:

    .. code-block:: text

        myres = max(low_resol_join_halves_angstrom, current_resolution_angstrom)
        lowres_r_max = ceil(grid_size * voxel_size / myres)

    matching ``ml_optimiser_mpi.cpp:3122-3123``:

    .. code-block:: cpp

        RFLOAT myres = XMIPP_MAX(low_resol_join_halves, 1./mymodel.current_resolution);
        int lowres_r_max = CEIL(mymodel.ori_size * mymodel.pixel_size / myres);

    Parameters
    ----------
    Ft_y_0, Ft_y_1 : array (volume_size,) complex
        Per-half ``Pᵀy`` (numerator of the Wiener filter) accumulators
        from the M-step, in centered Fourier order, flattened.
    Ft_ctf_0, Ft_ctf_1 : array (volume_size,) float
        Per-half ``Pᵀ(CTF² / σ²)`` (denominator) accumulators.
    volume_shape : tuple of 3 ints
        Shape of the centered Fourier volume.
    voxel_size : float
        Voxel size in Angstroms (image pixel size in real space).
    grid_size : int
        Real-space grid edge length, ``ori_size`` in RELION terms.
    low_resol_join_halves_angstrom : float
        The user-set joining resolution (RELION's ``--low_resol_join_halves``).
        Pass ``<= 0`` to disable; the function then returns the inputs
        unchanged.
    current_resolution_angstrom : float or None
        The current iteration's resolution estimate in Angstroms. The
        joining radius is the LOWER frequency (LARGER Å) of this and
        ``low_resol_join_halves_angstrom``. Pass ``None`` (the default)
        to ignore (equivalent to passing ``+inf``).
    padding_factor : int or None
        RELION backprojector padding factor used to convert native-shell
        join radii to accumulator-space coordinates. If omitted, falls back
        to the legacy shape-based inference, which is only reliable for full
        padded accumulators and not current-size BPref grids.

    Returns
    -------
    (Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined)
        New accumulators with the low-resolution shells averaged. Outside
        the joining sphere they are identical to the inputs.
    """
    if low_resol_join_halves_angstrom is None or low_resol_join_halves_angstrom <= 0:
        return Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1

    # Effective joining resolution: the larger (lower-frequency) of
    # low_resol_join_halves and current_resolution.
    myres = float(low_resol_join_halves_angstrom)
    if current_resolution_angstrom is not None and np.isfinite(current_resolution_angstrom):
        myres = max(myres, float(current_resolution_angstrom))

    lowres_r_max = int(np.ceil(grid_size * voxel_size / myres))
    if lowres_r_max <= 0:
        return Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1

    # RELION BackProjector::getLowResDataAndWeight / setLowResDataAndWeight
    # uses squared coordinates, not rounded shell labels:
    #   lowres_r2_max = ROUND(padding_factor * lowres_r_max)^2
    #   if (k*k + i*i + j*j <= lowres_r2_max) ...
    # Using rounded radial shells joins extra boundary voxels and changes the
    # downsampled half-map FSC near the 40 A join cutoff.
    if padding_factor is None:
        pf = volume_shape[0] // grid_size if volume_shape[0] > grid_size else 1
    else:
        pf = int(padding_factor)
        if pf <= 0:
            raise ValueError(f"padding_factor must be positive, got {padding_factor}")
    lowres_r_max_padded = int(_relion_round_away_from_zero(float(pf) * lowres_r_max))
    lowres_r2_max = lowres_r_max_padded * lowres_r_max_padded

    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    full_size = int(np.prod(volume_shape))
    half_size = int(np.prod(half_shape))
    ft_y_size = int(np.size(Ft_y_0))
    if ft_y_size == full_size:
        half_layout = False
    elif ft_y_size == half_size:
        half_layout = True
    else:
        raise ValueError(
            f"Could not infer Fourier layout for join_halves_at_low_resolution with shape {np.shape(Ft_y_0)} "
            f"and volume_shape={volume_shape}"
        )

    join_indices_np = _low_resolution_join_flat_indices(volume_shape, half_layout, lowres_r2_max)
    if join_indices_np.size == 0:
        return Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1

    max_input_size = max(
        ft_y_size,
        int(np.size(Ft_y_1)),
        int(np.size(Ft_ctf_0)),
        int(np.size(Ft_ctf_1)),
    )
    if _low_resolution_join_host_fallback_enabled_for_size(max_input_size, join_indices_np.size):
        logger.info(
            "Low-resolution half join using host fallback: size=%d join_voxels=%d",
            max_input_size,
            int(join_indices_np.size),
        )
        Ft_y_0_joined, Ft_y_1_joined = _join_half_pair_at_indices_host(Ft_y_0, Ft_y_1, join_indices_np)
        Ft_ctf_0_joined, Ft_ctf_1_joined = _join_half_pair_at_indices_host(Ft_ctf_0, Ft_ctf_1, join_indices_np)
        return Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined

    Ft_y_0_arr = jnp.asarray(Ft_y_0)
    Ft_y_1_arr = jnp.asarray(Ft_y_1)
    Ft_ctf_0_arr = jnp.asarray(Ft_ctf_0)
    Ft_ctf_1_arr = jnp.asarray(Ft_ctf_1)
    join_indices = jnp.asarray(join_indices_np)

    Ft_y_0_joined, Ft_y_1_joined = _join_half_pair_at_indices(Ft_y_0_arr, Ft_y_1_arr, join_indices)
    Ft_ctf_0_joined, Ft_ctf_1_joined = _join_half_pair_at_indices(Ft_ctf_0_arr, Ft_ctf_1_arr, join_indices)

    return Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined
