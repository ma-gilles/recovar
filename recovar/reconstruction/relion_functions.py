"""RELION-compatible reconstruction and Wiener filtering routines."""

import functools
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config, utils
from recovar.core import mask, padding
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise, regularization

logger = logging.getLogger(__name__)

_RELION_PROJECTION_PAD_HOST_FFT_MIN_VOXELS = 200_000_000


# ---------------------------------------------------------------------------
# Fourier volume zero-padding for reconstruction with padding_factor > 1
# ---------------------------------------------------------------------------


def _gridding_correct_trilinear(vol_real, ori_size, padding_factor):
    """Apply RELION-style gridding correction for trilinear interpolation.

    Divides each real-space voxel by sinc²(r / (N·pf)) where
    r = sqrt(x² + y² + z²) is the radial distance from the volume center.
    Matches RELION's ``Projector::griddingCorrect()`` (projector.cpp:27-60)
    for ``interpolator == TRILINEAR``.

    RELION uses a RADIAL sinc² correction (not separable per-axis).

    Parameters
    ----------
    vol_real : jnp.ndarray, shape (N, N, N)
        Real-space volume with origin at array center.
    ori_size : int
        Original box size (N).
    padding_factor : int
        Padding factor (1 or 2).

    Returns
    -------
    vol_corrected : jnp.ndarray, shape (N, N, N)
    """
    N = vol_real.shape[0]
    coords = jnp.arange(N, dtype=jnp.float64) - N / 2.0
    # 3D radial distance
    r = jnp.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    arg = r / (ori_size * padding_factor)
    # sinc(x) = sin(πx)/(πx), sinc(0) = 1
    sinc_r = jnp.where(arg < 1e-15, 1.0, jnp.sin(jnp.pi * arg) / (jnp.pi * arg))
    sinc2_r = sinc_r**2
    return vol_real / sinc2_r


def _gridding_correct_trilinear_np(vol_real, ori_size, padding_factor):
    """NumPy equivalent of ``_gridding_correct_trilinear`` for host padding."""

    N = vol_real.shape[0]
    coords = np.arange(N, dtype=np.float64) - N / 2.0
    r = np.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    arg = r / (ori_size * padding_factor)
    sinc_r = np.ones_like(arg)
    nz = arg >= 1e-15
    sinc_r[nz] = np.sin(np.pi * arg[nz]) / (np.pi * arg[nz])
    return vol_real / (sinc_r**2)


def _get_dft3_np(img, norm=fourier_transform_utils.DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = np.fft.fftshift(img, axes=axes)
    img = np.fft.fftn(img, axes=axes, norm=norm)
    img = np.fft.fftshift(img, axes=axes)
    return img


def _get_idft3_np(img, norm=fourier_transform_utils.DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = np.fft.ifftshift(img, axes=axes)
    img = np.fft.ifftn(img, axes=axes, norm=norm)
    img = np.fft.ifftshift(img, axes=axes)
    return img


def _pad_volume_for_projection_host(
    vol_ft_flat,
    volume_shape,
    padding_factor,
    *,
    do_gridding_correction=False,
    current_size=None,
):
    """Host-side projection padding for grids whose cuFFT workspace is too large."""

    N = int(volume_shape[0])
    padded_shape = tuple(int(s) * int(padding_factor) for s in volume_shape)
    vol_real = _get_idft3_np(np.asarray(vol_ft_flat, dtype=np.complex64).reshape(volume_shape))
    if do_gridding_correction:
        vol_real = _gridding_correct_trilinear_np(vol_real, N, int(padding_factor))
    pad_amount = N * (int(padding_factor) - 1)
    pad_before = pad_amount // 2
    pad_after = pad_amount - pad_before
    vol_real_padded = np.pad(vol_real, [(pad_before, pad_after)] * 3, mode="constant")
    vol_ft_padded = _get_dft3_np(vol_real_padded).astype(np.complex64, copy=False)

    if current_size is not None:
        r_max_ref = int(padding_factor) * (int(current_size) // 2)
        pN = int(padded_shape[0])
        coords = np.arange(pN, dtype=np.float32) - pN / 2.0
        r2_3d = coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2
        vol_ft_padded = np.where(r2_3d <= r_max_ref**2, vol_ft_padded, 0.0)

    return jnp.asarray(vol_ft_padded.reshape(-1)), padded_shape


def pad_volume_for_projection(
    vol_ft_flat, volume_shape, padding_factor, do_gridding_correction=False, current_size=None
):
    """Pad a Fourier volume via real-space zero-padding for smoother projection.

    RELION pads volumes in REAL SPACE before FFT so that trilinear
    interpolation operates on a (pf*N)³ grid.  This is NOT the same as
    Fourier zero-padding (which leaves stride-pf gaps that degrade
    interpolation).

    When ``do_gridding_correction=True``, applies RELION's gridding correction
    (``Projector::griddingCorrect``) to the real-space volume before padding.
    This compensates for the smoothing inherent in trilinear Fourier-slice
    interpolation, matching RELION's projector behaviour.

    When ``current_size`` is provided, applies a spherical mask at
    ``r_max_ref = padding_factor * current_size // 2`` to the padded Fourier
    volume.  This matches RELION's ``Projector::computeFourierTransformMap``
    which calls ``decenter(data, Faux, max_r2)`` — only copying Fourier
    coefficients within ``r_max_ref`` and leaving everything beyond as zero.
    Without this mask, JAX's trilinear interpolation near the scoring-window
    boundary produces different (non-attenuated) values compared to RELION,
    because RELION's trilinear blends real data with zeros at the boundary.

    Parameters
    ----------
    vol_ft_flat : jnp.ndarray, shape (N³,)
        Flat centered Fourier volume at native resolution.
    volume_shape : tuple of int, (N, N, N)
        Native volume shape.
    padding_factor : int
        Padding factor (typically 2).
    do_gridding_correction : bool, optional
        If True, apply gridding correction before padding (default False).
    current_size : int, optional
        RELION's current resolution size.  When set, applies a spherical
        Fourier mask at radius ``padding_factor * current_size // 2`` to
        match RELION's projector data boundary.

    Returns
    -------
    padded_ft_flat : jnp.ndarray, shape ((pf*N)³,)
        Fourier volume on the (pf*N)³ grid, ready for slice_volume.
    padded_shape : tuple of int
        (pf*N, pf*N, pf*N).
    """
    if padding_factor == 1:
        return vol_ft_flat, volume_shape

    padded_shape = tuple(int(s) * int(padding_factor) for s in volume_shape)
    if int(np.prod(padded_shape)) > _RELION_PROJECTION_PAD_HOST_FFT_MIN_VOXELS:
        return _pad_volume_for_projection_host(
            vol_ft_flat,
            volume_shape,
            padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )

    N = volume_shape[0]
    vol_real = fourier_transform_utils.get_idft3(jnp.asarray(vol_ft_flat).reshape(volume_shape))
    if do_gridding_correction:
        vol_real = _gridding_correct_trilinear(vol_real, N, padding_factor)
    pad_amount = N * (padding_factor - 1)
    vol_real_padded = padding.pad_volume_spatial_domain(vol_real, pad_amount)
    vol_ft_padded = fourier_transform_utils.get_dft3(vol_real_padded)

    if current_size is not None:
        # Match RELION's Projector::computeFourierTransformMap decenter behaviour:
        # zero all Fourier coefficients beyond r_max_ref = pf * (cs // 2).
        r_max_ref = padding_factor * (current_size // 2)
        pN = padded_shape[0]
        coords = jnp.arange(pN, dtype=jnp.float32) - pN / 2.0
        r2_3d = coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2
        sphere_mask = (r2_3d <= r_max_ref**2).astype(vol_ft_padded.dtype)
        vol_ft_padded = vol_ft_padded.reshape(padded_shape) * sphere_mask
        vol_ft_padded = vol_ft_padded.reshape(-1)

    return vol_ft_padded.reshape(-1), padded_shape


def zero_pad_fourier_volume(vol_flat, native_shape, padding_factor):
    """Zero-pad a flat centered Fourier volume to a larger grid.

    RELION's padded reconstruction grid uses the same physical frequencies on a
    finer lattice. In centered / fftshift layout, native frequency bin ``k``
    therefore lands at ``padding_factor * k`` on the padded grid, not at the
    same array index inside a larger centered cube.

    Parameters
    ----------
    vol_flat : jnp.ndarray, shape (N^3,)
        Flat centered Fourier volume at native resolution.
    native_shape : tuple of int, (N, N, N)
        Native volume shape.
    padding_factor : int
        Padding factor (typically 2).  Output size is (N*pf)^3.

    Returns
    -------
    padded_flat : jnp.ndarray, shape ((N*pf)^3,)
        Zero-padded Fourier volume in centered layout.
    """
    if padding_factor == 1:
        return vol_flat

    native_shape = tuple(int(s) for s in native_shape)
    padded_shape = tuple(s * padding_factor for s in native_shape)
    vol_3d = jnp.asarray(vol_flat).reshape(native_shape)
    padded = jnp.zeros(padded_shape, dtype=vol_3d.dtype)

    padded_indices = []
    for native_dim, padded_dim in zip(native_shape, padded_shape):
        start = padded_dim // 2 - padding_factor * (native_dim // 2)
        padded_indices.append(np.arange(native_dim, dtype=np.int32) * padding_factor + start)

    padded = padded.at[np.ix_(*padded_indices)].set(vol_3d)

    return padded.reshape(-1)


def griddingCorrect(vol_in, ori_size, padding_factor, order=0):
    """Radial sinc gridding correction."""
    og_shape = vol_in.shape
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False).astype(jnp.float64)
    r = jnp.sqrt(jnp.sum(pixels**2, axis=-1))
    safe_rval = jnp.where(r > 0, r / (ori_size * padding_factor), 1.0)
    sinc = jnp.where(r > 0, jnp.sin(jnp.pi * safe_rval) / (jnp.pi * safe_rval), 1.0)
    if order == 0:
        kernel = sinc
    elif order == 1:
        kernel = sinc**2
    else:
        raise ValueError("Order not implemented")
    return (vol_in.reshape(-1) / kernel).reshape(og_shape), kernel.reshape(og_shape)


def griddingCorrect_square(vol_in, ori_size, padding_factor, order=0):
    """Per-axis sinc product gridding correction (Fourier transform of trilinear interpolator)."""
    og_shape = vol_in.shape
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False).astype(np.float64)
    pixels_rescaled = pixels / (ori_size * padding_factor)

    def sinc(ar):
        return jnp.where(jnp.abs(ar) < 1e-8, 1.0, jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

    if order == 0:
        kernel_fn = sinc
    elif order == 1:
        kernel_fn = lambda x: sinc(x) ** 2
    else:
        raise ValueError("Order not implemented")

    kernel = kernel_fn(pixels_rescaled[:, 0]) * kernel_fn(pixels_rescaled[:, 1]) * kernel_fn(pixels_rescaled[:, 2])
    return (vol_in / kernel.reshape(og_shape)).reshape(og_shape), kernel.reshape(og_shape)


def relion_style_triangular_kernel(
    experiment_dataset,
    cov_noise,
    batch_size,
    disc_type="linear_interp",
    index_subset=None,
    upsampling_factor=None,
    by_image=True,
):
    """RELION-style triangular kernel reconstruction.

    Accumulates weighted back-projection (Ft_y) and CTF weight sum (Ft_ctf)
    in half-volume layout, then expands to full volume before returning.

    Parameters
    ----------
    experiment_dataset : CryoEMDataset
    cov_noise : array or None
        Radial shell variances or pre-expanded noise.  When None, noise is
        drawn per-batch from ``experiment_dataset.noise``.
    batch_size : int
    disc_type : str
    index_subset : array-like, optional
        If given, only iterate over this subset in the domain selected by
        ``by_image``.
    upsampling_factor : int, optional
        Defaults to ``1``.
    by_image : bool, default True
        When ``True``, interpret ``index_subset`` as image indices.
        When ``False``, interpret it as group / particle indices and iterate
        with the grouped dataset iterator. Needed for tilt-series particle
        subsets such as junk-detection halfmap reconstruction.
    """
    uf = upsampling_factor if upsampling_factor is not None else 1
    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, upsampling_factor=uf)
    noise_model = (
        noise.as_noise_model(cov_noise, config.image_shape) if cov_noise is not None else experiment_dataset.noise
    )

    Ft_y, Ft_ctf = None, None
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
        noise_model=noise_model,
        indices=index_subset,
        by_image=by_image,
    ):
        Ft_y, Ft_ctf = relion_kernel_batch(
            config,
            images,
            ctf_params,
            rotation_matrices,
            translations,
            noise_variance,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
        )

    if Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(Ft_y, config.volume_shape).reshape(-1)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, config.volume_shape).reshape(-1)

    return Ft_ctf, Ft_y


def relion_kernel_batch(
    config: ForwardModelConfig,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
):
    """RELION-style triangular kernel batch for raw real-space images.

    Applies pad + rfft2 internally, then backprojects with half_image and
    half_volume layouts for maximum memory efficiency.
    """
    half_images = padding.padded_rfft(images * config.data_multiplier, config.grid_size, config.padding)
    return _relion_kernel_batch_half(
        config,
        half_images,
        ctf_params,
        rotation_matrices,
        translations,
        noise_variance,
        Ft_y,
        Ft_ctf,
    )


def relion_kernel_batch_from_fft(
    config: ForwardModelConfig,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
):
    """RELION-style triangular kernel batch for pre-FFTed complex images.

    Extracts the half-spectrum from full-spectrum images, then backprojects
    using half_image and half_volume layouts.
    """
    half_images = fourier_transform_utils.full_image_to_half_image(images, config.image_shape)
    return _relion_kernel_batch_half(
        config,
        half_images,
        ctf_params,
        rotation_matrices,
        translations,
        noise_variance,
        Ft_y,
        Ft_ctf,
    )


@eqx.filter_jit
def _relion_kernel_batch_half(
    config,
    half_images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variances,
    Ft_y,
    Ft_ctf,
):
    """Backproject half-spectrum images into half-volume accumulators."""
    half_images = core.translate_images(half_images, translations, config.image_shape, half_image=True)
    noise_half = noise.to_batched_half_pixel_noise(noise_variances, config.image_shape, batch_size=half_images.shape[0])
    ctf_half = config.compute_ctf_half(ctf_params)

    images_weighted = (ctf_half * half_images if not config.premultiplied_ctf else half_images) / noise_half
    Ft_y = core.adjoint_slice_volume(
        images_weighted,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        config.disc_type,
        volume=Ft_y,
        half_image=True,
        half_volume=True,
    )
    Ft_ctf = core.adjoint_slice_volume(
        ctf_half**2 / noise_half,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        config.disc_type,
        volume=Ft_ctf,
        half_image=True,
        half_volume=True,
    )
    return Ft_y, Ft_ctf


@eqx.filter_jit
def residual_relion_kernel_trilinear(
    config: ForwardModelConfig,
    mean_estimate: jax.Array,
    images: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    translations: jax.Array,
    cov_noise: jax.Array,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
):
    """Residual RELION-style kernel (trilinear), accumulating into half-volume layout.

    Parameters
    ----------
    Ft_y, Ft_ctf : optional accumulator volumes (half-volume layout) to add into.
    """
    CTF = config.compute_ctf(ctf_params)
    images = core.translate_images(images, translations, config.image_shape)
    images = (
        images
        - core.slice_volume(
            mean_estimate,
            rotation_matrices,
            config.image_shape,
            config.volume_shape,
            "linear_interp",
        )
        * CTF
    )
    images_squared = jnp.abs(images) ** 2 - cov_noise
    CTF_fourth = CTF**4

    images_squared_half = fourier_transform_utils.full_image_to_half_image(images_squared, config.image_shape)
    Ft_y = core.adjoint_slice_volume(
        images_squared_half,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        "linear_interp",
        volume=Ft_y,
        half_image=True,
        half_volume=True,
    )
    CTF_fourth_half = fourier_transform_utils.full_image_to_half_image(CTF_fourth, config.image_shape)
    Ft_ctf = core.adjoint_slice_volume(
        CTF_fourth_half,
        rotation_matrices,
        config.image_shape,
        config.volume_shape,
        "linear_interp",
        volume=Ft_ctf,
        half_image=True,
        half_volume=True,
    )
    return Ft_y, Ft_ctf


def residual_relion_style_triangular_kernel(
    experiment_dataset, mean_estimate, cov_noise, batch_size, index_subset=None
):
    """Residual RELION-style triangular kernel reconstruction."""
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type="linear_interp",
        upsampling_factor=1,
    )

    Ft_y, Ft_ctf = None, None
    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        _image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        indices=index_subset,
        by_image=False,
    ):
        images = experiment_dataset.process_images(images)
        Ft_y, Ft_ctf = residual_relion_kernel_trilinear(
            config,
            mean_estimate,
            images,
            ctf_params,
            rotation_matrices,
            translations,
            cov_noise,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
        )

    if Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(Ft_y, config.volume_shape).reshape(-1)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, config.volume_shape).reshape(-1)
    else:
        vol_size = int(np.prod(config.volume_shape))
        Ft_y = jnp.zeros(vol_size, dtype=experiment_dataset.dtype)
        Ft_ctf = jnp.zeros(vol_size, dtype=experiment_dataset.dtype_real)

    return Ft_ctf, Ft_y


def upscale_tau(tau, padding_factor, volume_shape, tau_is_1d=False):
    if not tau_is_1d:
        tau = regularization.average_over_shells(tau, volume_shape)
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(
        np.array(volume_shape) * padding_factor, 1, scaled=False
    )
    radius = jnp.round(jnp.linalg.norm(pixels, axis=-1) / padding_factor).astype(jnp.int32)
    return tau[radius]


def _upscale_tau_half(tau, padding_factor, volume_shape, tau_is_1d=False):
    if not tau_is_1d:
        tau = regularization.average_over_shells(tau, volume_shape)
    radius = (
        fourier_transform_utils.get_grid_of_radial_distances_real(
            np.array(volume_shape) * padding_factor,
            scaled=False,
            frequency_shift=0,
        )
        / padding_factor
    )
    radius = jnp.round(radius).astype(jnp.int32).reshape(-1)
    return tau[radius]


def _as_flat_single_volume(arr, volume_shape):
    """Accept flat or grid volume array and return flat vector + was_grid flag."""
    arr = jnp.asarray(arr)
    flat_size = int(np.prod(volume_shape))
    if arr.ndim == len(volume_shape) and tuple(arr.shape) == tuple(volume_shape):
        return arr.reshape(-1), True
    if arr.ndim == 1 and int(arr.shape[0]) == flat_size:
        return arr, False
    raise ValueError(f"Expected array with shape {volume_shape} or ({flat_size},), got {arr.shape}")


def _average_over_shells_half(input_vec, volume_shape, frequency_shift=0):
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances_real(
            volume_shape,
            scaled=False,
            frequency_shift=frequency_shift,
        )
        .astype(int)
        .reshape(-1)
    )
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0] // 2 - 1)
    return regularization.jax_scipy_nd_image_mean(input_vec.reshape(-1), labels=labels, index=indices)


def adjust_regularization_relion_style(
    filter,
    volume_shape,
    tau=None,
    padding_factor=1,
    max_res_shell=None,
    half_volume=False,
    tau2_fudge=1.0,
    minres_map=0,
):
    """Adjust the RELION-style regularization filter.

    Adds 1/tau to the filter (Wiener denominator) and floors small values at
    1/1000 of the spherically-averaged filter to avoid division by zero.
    See RELION backprojector.cpp for the original algorithm.

    The ``tau2_fudge`` parameter mirrors RELION's ``--tau2_fudge`` flag
    (default 1.0).  It enters the Wiener denominator as::

        inv_tau = 1 / (padding_factor**3 * tau2_fudge * tau)

    ``minres_map`` mirrors RELION's ``--minres_map``: the Wiener prior term is
    only added for shells ``ires >= minres_map``.
    """
    volume_shape = tuple(int(s) for s in volume_shape)
    packed_shape = (
        fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
    )
    filter_flat, input_is_grid = _as_flat_single_volume(filter, packed_shape)

    # Exact half-volume behavior: reuse full-volume implementation and repack.
    if half_volume:
        if tau is not None:
            oversampling_factor = padding_factor**3
            og_volume_shape = tuple(s // padding_factor for s in volume_shape)
            tau = _upscale_tau_half(tau, padding_factor, og_volume_shape, tau_is_1d=False)
            safe_tau = jnp.where(tau > 1e-20, tau, jnp.float32(1.0))
            inv_tau = 1 / (oversampling_factor * tau2_fudge * safe_tau)
            inv_tau = jnp.where((tau < 1e-20) & (filter_flat > 1e-20), 1.0 / (0.001 * filter_flat), inv_tau)
            inv_tau = jnp.where((tau < 1e-20) & (filter_flat <= 1e-20), 0, inv_tau)
            if int(minres_map) > 0:
                shell = fourier_transform_utils.get_grid_of_radial_distances_real(
                    volume_shape,
                    scaled=False,
                    frequency_shift=0,
                ) / float(padding_factor)
                shell = jnp.round(shell).astype(jnp.int32).reshape(-1)
                inv_tau = jnp.where(shell >= int(minres_map), inv_tau, 0)
            regularized_filter = filter_flat + inv_tau
        else:
            regularized_filter = filter_flat

        if max_res_shell is None:
            max_res_shell = volume_shape[0] // 2 - 1

        avged_reg = _average_over_shells_half(regularized_filter, volume_shape, frequency_shift=0) / 1000
        avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
        avged_reg_volume = utils.make_radial_image_half(avged_reg, volume_shape).reshape(regularized_filter.shape)

        regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume)
        regularized_filter = jnp.maximum(regularized_filter, jax_config.EPSILON)
        if input_is_grid:
            return regularized_filter.reshape(packed_shape)
        return regularized_filter

    if tau is not None:
        # RELION: invtau2 = 1 / (padding_factor^3 * tau2_fudge * tau2[ires])
        oversampling_factor = padding_factor**3
        og_volume_shape = tuple(s // padding_factor for s in volume_shape)
        tau = upscale_tau(tau, padding_factor, og_volume_shape, tau_is_1d=False)
        safe_tau = jnp.where(tau > 1e-20, tau, jnp.float32(1.0))
        inv_tau = 1 / (oversampling_factor * tau2_fudge * safe_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter_flat > 1e-20), 1.0 / (0.001 * filter_flat), inv_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter_flat <= 1e-20), 0, inv_tau)
        if int(minres_map) > 0:
            pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(
                np.array(volume_shape),
                1,
                scaled=False,
            )
            shell = jnp.round(jnp.linalg.norm(pixels, axis=-1) / float(padding_factor)).astype(jnp.int32)
            shell = shell.reshape(-1)
            inv_tau = jnp.where(shell >= int(minres_map), inv_tau, 0)
        regularized_filter = filter_flat + inv_tau
    else:
        regularized_filter = filter_flat

    if max_res_shell is None:
        max_res_shell = volume_shape[0] // 2 - 1

    avged_reg = regularization.average_over_shells(regularized_filter, volume_shape, frequency_shift=0) / 1000
    avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
    avged_reg_volume = utils.make_radial_image(avged_reg, volume_shape).reshape(regularized_filter.shape)

    regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume)
    regularized_filter = jnp.maximum(regularized_filter, jax_config.EPSILON)
    if input_is_grid:
        return regularized_filter.reshape(packed_shape)
    return regularized_filter


def _infer_half_volume_layout(arr, volume_shape):
    """Infer whether a Fourier volume is packed half layout from its shape."""
    full_shape = tuple(int(s) for s in volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_shape)
    full_size = int(np.prod(full_shape))
    half_size = int(np.prod(half_shape))
    arr = jnp.asarray(arr)
    if arr.ndim == 3 and tuple(arr.shape) == half_shape:
        return True
    if arr.ndim == 3 and tuple(arr.shape) == full_shape:
        return False
    if arr.ndim == 1 and int(arr.shape[0]) == half_size:
        return True
    if arr.ndim == 1 and int(arr.shape[0]) == full_size:
        return False
    raise ValueError(f"Could not infer half/full Fourier layout for shape {arr.shape} and volume_shape={volume_shape}")


def post_process_from_filter(
    cryo,
    Ft_ctf,
    F_ty,
    tau=None,
    disc_type="nearest",
    use_spherical_mask=True,
    grid_correct=True,
    gridding_correct="square",
    kernel_width=1,
    tau2_fudge=1.0,
    padding_factor=1,
    minres_map=0,
):
    """Post-process RELION-style reconstruction from filter weights.

    Thin wrapper around ``post_process_from_filter_v2`` that extracts the
    necessary geometry from a dataset object (*cryo*).
    """
    kernel = "triangular" if disc_type == "linear_interp" else "square"
    return post_process_from_filter_v2(
        Ft_ctf,
        F_ty,
        cryo.volume_shape,
        padding_factor,
        tau=tau,
        kernel=kernel,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=kernel_width,
        tau2_fudge=tau2_fudge,
        minres_map=minres_map,
    )


@functools.partial(jax.jit, static_argnums=[2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 17, 18])
def post_process_from_filter_v2(
    Ft_ctf,
    F_ty,
    og_volume_shape,
    volume_upsampling_factor,
    tau=None,
    kernel="triangular",
    use_spherical_mask=True,
    grid_correct=True,
    gridding_correct="square",
    kernel_width=1,
    volume_mask=None,
    return_real_space=False,
    return_half_volume=False,
    input_half_volume=None,
    tau2_fudge=1.0,
    gridding_padding_factor=None,
    gridding_order=None,
    minres_map=0,
    current_size=None,
):
    """Post-process RELION-style reconstruction from filter weights.

    Steps: regularize -> iDFT -> crop -> spherical mask -> grid correct -> DFT.

    Supports both full Fourier inputs ``(N0*N1*N2,)`` and packed half-volume
    inputs ``(N0*N1*(N2//2+1),)``.

    The ``tau2_fudge`` parameter (default 1.0) is forwarded to
    :func:`adjust_regularization_relion_style` and mirrors RELION's
    ``--tau2_fudge`` flag.

    ``current_size`` (when given) limits the Wiener filter's spatial mask
    to the padded sphere ``r <= padding_factor * (current_size // 2)``,
    matching RELION's ``BackProjector::reconstruct`` which skips voxels
    with ``r2 >= max_r2 = ROUND(r_max * padding_factor)^2`` (line 1264).
    Without this, recovar's Wiener filter operates on every padded voxel
    up to ``upsampled_volume_shape[0]//2 - 1``, producing residual
    high-shell content from the regularization floor that RELION omits.
    """
    upsampled_volume_shape = tuple(3 * [og_volume_shape[0] * volume_upsampling_factor])
    if input_half_volume is None:
        input_half_volume = _infer_half_volume_layout(Ft_ctf, upsampled_volume_shape)

    # Wiener spatial mask: match RELION's max_r2 skip when current_size given.
    if current_size is not None and current_size > 0:
        wiener_radius = volume_upsampling_factor * (int(current_size) // 2)
    else:
        wiener_radius = upsampled_volume_shape[0] // 2 - 1

    if input_half_volume:
        packed_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape)
        Ft_ctf_flat, _ = _as_flat_single_volume(Ft_ctf, packed_shape)
        F_ty_flat, _ = _as_flat_single_volume(F_ty, packed_shape)
        valid_indices = (
            (
                fourier_transform_utils.full_volume_to_half_volume(
                    mask.get_radial_mask(upsampled_volume_shape, radius=wiener_radius),
                    upsampled_volume_shape,
                )
            )
            .reshape(-1)
            .astype(Ft_ctf_flat.real.dtype)
        )
    else:
        Ft_ctf_flat, _ = _as_flat_single_volume(Ft_ctf, upsampled_volume_shape)
        F_ty_flat, _ = _as_flat_single_volume(F_ty, upsampled_volume_shape)
        valid_indices = (
            mask.get_radial_mask(upsampled_volume_shape, radius=wiener_radius)
            .reshape(-1)
            .astype(Ft_ctf_flat.real.dtype)
        )

    Ft_ctf2 = adjust_regularization_relion_style(
        Ft_ctf_flat.real,
        upsampled_volume_shape,
        tau=tau,
        padding_factor=volume_upsampling_factor,
        max_res_shell=None,
        half_volume=input_half_volume,
        tau2_fudge=tau2_fudge,
        minres_map=minres_map,
    )
    vol = (F_ty_flat * valid_indices) / Ft_ctf2

    # iDFT → crop to original size
    if input_half_volume:
        vol = fourier_transform_utils.get_idft3_real(vol.reshape(packed_shape), volume_shape=upsampled_volume_shape)
    else:
        vol = fourier_transform_utils.get_idft3(vol.reshape(upsampled_volume_shape))
    vol = padding.unpad_volume_spatial_domain(vol, upsampled_volume_shape[0] - og_volume_shape[0])

    if use_spherical_mask:
        vol, _ = mask.soft_mask_outside_map(vol, cosine_width=3)

    if volume_mask is not None:
        vol = vol * volume_mask

    if grid_correct:
        order = gridding_order if gridding_order is not None else (1 if kernel == "triangular" else 0)
        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        gc_pf = gridding_padding_factor if gridding_padding_factor is not None else volume_upsampling_factor
        vol, _ = grid_fn(vol.reshape(og_volume_shape), og_volume_shape[0], gc_pf / kernel_width, order=order)

    if return_real_space:
        return vol.real.astype(Ft_ctf2.real.dtype)

    if input_half_volume:
        vol = fourier_transform_utils.get_dft3_real(vol.reshape(og_volume_shape))
    else:
        vol = fourier_transform_utils.get_dft3(vol.reshape(og_volume_shape))
    if return_half_volume:
        if not input_half_volume:
            vol = fourier_transform_utils.full_volume_to_half_volume(vol, og_volume_shape)
        return vol.reshape(-1).astype(F_ty_flat.dtype)
    if input_half_volume:
        vol = fourier_transform_utils.half_volume_to_full_volume(vol, og_volume_shape)
    return vol.astype(F_ty_flat.dtype)


def relion_reconstruct(
    cryo,
    noise_variance,
    batch_size=100,
    disc_type="linear_interp",
    use_spherical_mask=True,
    upsampling_factor=2,
    grid_correct=True,
    gridding_correct="square",
    tau=None,
    tau2_fudge=1.0,
):
    """Full mean reconstruction pipeline: accumulate → post-process."""
    Ft_ctf, F_ty = relion_style_triangular_kernel(
        cryo,
        noise_variance.astype(np.float32),
        batch_size,
        disc_type=disc_type,
        upsampling_factor=upsampling_factor,
    )
    kernel = "triangular" if disc_type == "linear_interp" else "square"
    estimate = post_process_from_filter_v2(
        Ft_ctf,
        F_ty,
        cryo.volume_shape,
        upsampling_factor,
        tau=tau,
        kernel=kernel,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=1,
        tau2_fudge=tau2_fudge,
    )
    return estimate, Ft_ctf
