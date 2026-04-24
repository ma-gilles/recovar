"""Fourier-shell regularization priors and FSC computation."""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)

# NVTX domain for regularization operations
NVTX_DOMAIN_REG = "regularization"

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
    # RELION sets fsc(0) = 1.0 in calculateDownSampledFourierShellCorrelation
    fsc = fsc.at[0].set(1.0)
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


def _compute_relion_weight_shell_stats(weight, volume_shape, *, padding_factor=1):
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

    Returns
    -------
    dict
        ``shell_sum``, ``shell_count``, and ``avg_weight_shells`` arrays with
        RELION-matching shell indexing.
    """
    volume_shape = tuple(int(s) for s in volume_shape)
    ori_half = volume_shape[0] // 2
    n_shells = ori_half + 1

    weight = jnp.asarray(weight).real.reshape(-1).astype(jnp.float64)
    grid_shape = tuple(d * padding_factor for d in volume_shape) if padding_factor > 1 else volume_shape
    half_grid_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(grid_shape)
    full_size = int(np.prod(grid_shape))
    half_size = int(np.prod(half_grid_shape))
    if weight.size == full_size:
        is_half_layout = False
    elif weight.size == half_size:
        is_half_layout = True
    else:
        raise ValueError(
            f"Expected full or half Fourier weight with {full_size} or {half_size} voxels for "
            f"volume_shape={volume_shape} and padding_factor={padding_factor}, got {weight.size}"
        )

    if padding_factor > 1:
        padded_dist = fourier_transform_utils.get_grid_of_radial_distances(
            grid_shape,
            scaled=False,
            frequency_shift=0,
            rounded=False,
        ).reshape(-1)
        shell_index = jnp.minimum(jnp.floor(padded_dist / padding_factor + 0.5).astype(jnp.int32), ori_half)
        if is_half_layout:
            shell_index = jnp.minimum(
                jnp.floor(
                    fourier_transform_utils.get_grid_of_radial_distances_real(
                        grid_shape,
                        scaled=False,
                        frequency_shift=0,
                        rounded=False,
                    ).reshape(-1)
                    / padding_factor
                    + 0.5
                ).astype(jnp.int32),
                ori_half,
            )
            included = jnp.ones(weight.shape[0], dtype=jnp.float64)
        else:
            # RELION iterates the half-complex x-axis (kx >= 0) only.
            pf_n = grid_shape[-1]
            kx_idx = jnp.arange(weight.size) % pf_n
            included = ((kx_idx == 0) | (kx_idx >= pf_n // 2)).astype(jnp.float64)
    else:
        radial_fn = (
            fourier_transform_utils.get_grid_of_radial_distances_real
            if is_half_layout
            else fourier_transform_utils.get_grid_of_radial_distances
        )
        shell_index = radial_fn(
            volume_shape,
            scaled=False,
            frequency_shift=0,
        ).astype(jnp.int32).reshape(-1)
        shell_index = jnp.minimum(shell_index, ori_half)
        included = jnp.ones(weight.shape[0], dtype=jnp.float64)

    shell_sum = jnp.zeros(n_shells, dtype=jnp.float64)
    shell_count = jnp.zeros(n_shells, dtype=jnp.float64)
    shell_sum = shell_sum.at[shell_index].add(weight * included)
    shell_count = shell_count.at[shell_index].add(included)
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
    return_details=False,
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
    """
    prior_dtype = jnp.float32

    H0 = jnp.asarray(Ft_ctf_0).real.astype(prior_dtype)
    H1 = jnp.asarray(Ft_ctf_1).real.astype(prior_dtype)
    H_comb = (H0 + H1) / jnp.asarray(2.0, dtype=prior_dtype)
    shell_stats = _compute_relion_weight_shell_stats(
        H_comb,
        volume_shape,
        padding_factor=padding_factor,
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
    fsc_clamped = jnp.clip(fsc_arr, epsilon, 1.0 - epsilon)
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
    }
    return prior, fsc_clamped, details


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


def compute_data_vs_prior(Ft_ctf, tau2, volume_shape, padding_factor=1, tau2_fudge=1.0):
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

    Returns
    -------
    jnp.ndarray, shape (n_shells,)
        Per-shell data_vs_prior ratio.
    """
    avg_weight = _compute_relion_weight_shell_stats(
        Ft_ctf,
        volume_shape,
        padding_factor=padding_factor,
    )["avg_weight_shells"].astype(jnp.asarray(tau2).dtype)
    oversampling_correction = padding_factor**3
    return avg_weight * tau2_fudge * tau2 * oversampling_correction


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
    myfsc = jnp.clip(fsc, epsilon, 1.0 - epsilon)
    if is_whole_instead_of_half:
        myfsc = jnp.sqrt(2.0 * myfsc / (myfsc + 1.0))
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

    # RELION ml_optimiser_mpi.cpp:3161 scales the comparison radius by
    # padding_factor when operating on the padded backprojector grid:
    #   if (kp*kp + ip*ip + jp*jp <= lowres_r_max*lowres_r_max * pf*pf)
    # Since volume_shape here is the padded grid, scale lowres_r_max.
    pf = volume_shape[0] // grid_size if volume_shape[0] > grid_size else 1
    lowres_r_max_padded = lowres_r_max * pf

    Ft_y_0_arr = jnp.asarray(Ft_y_0)
    Ft_y_1_arr = jnp.asarray(Ft_y_1)
    Ft_ctf_0_arr = jnp.asarray(Ft_ctf_0)
    Ft_ctf_1_arr = jnp.asarray(Ft_ctf_1)

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    full_size = int(np.prod(volume_shape))
    half_size = int(np.prod(half_shape))
    if Ft_y_0_arr.size == full_size:
        radial_dist_3d = fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape,
            voxel_size=1,
            scaled=False,
            frequency_shift=0,
            rounded=True,
        )
    elif Ft_y_0_arr.size == half_size:
        radial_dist_3d = fourier_transform_utils.get_grid_of_radial_distances_real(
            volume_shape,
            voxel_size=1,
            scaled=False,
            frequency_shift=0,
            rounded=True,
        )
    else:
        raise ValueError(
            f"Could not infer Fourier layout for join_halves_at_low_resolution with shape {Ft_y_0_arr.shape} "
            f"and volume_shape={volume_shape}"
        )

    join_mask_3d = radial_dist_3d <= lowres_r_max_padded
    join_mask_flat = join_mask_3d.reshape(-1)

    avg_Ft_y = 0.5 * (Ft_y_0_arr + Ft_y_1_arr)
    avg_Ft_ctf = 0.5 * (Ft_ctf_0_arr + Ft_ctf_1_arr)

    Ft_y_0_joined = jnp.where(join_mask_flat, avg_Ft_y, Ft_y_0_arr)
    Ft_y_1_joined = jnp.where(join_mask_flat, avg_Ft_y, Ft_y_1_arr)
    Ft_ctf_0_joined = jnp.where(join_mask_flat, avg_Ft_ctf, Ft_ctf_0_arr)
    Ft_ctf_1_joined = jnp.where(join_mask_flat, avg_Ft_ctf, Ft_ctf_1_arr)

    return Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined
