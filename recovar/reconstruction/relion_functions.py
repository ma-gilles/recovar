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


def griddingCorrect(vol_in, ori_size, padding_factor, order=0):
    """Radial sinc gridding correction."""
    og_shape = vol_in.shape
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False) + 0.0
    r = np.linalg.norm(pixels, axis=-1)
    safe_rval = np.where(r > 0, r / (ori_size * padding_factor), 1.0)
    sinc = np.where(r > 0, np.sin(np.pi * safe_rval) / (np.pi * safe_rval), 1.0)
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
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False)
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


def _as_flat_single_volume(arr, volume_shape):
    """Accept flat or grid volume array and return flat vector + was_grid flag."""
    arr = jnp.asarray(arr)
    flat_size = int(np.prod(volume_shape))
    if arr.ndim == len(volume_shape) and tuple(arr.shape) == tuple(volume_shape):
        return arr.reshape(-1), True
    if arr.ndim == 1 and int(arr.shape[0]) == flat_size:
        return arr, False
    raise ValueError(f"Expected array with shape {volume_shape} or ({flat_size},), got {arr.shape}")


def adjust_regularization_relion_style(
    filter, volume_shape, tau=None, padding_factor=1, max_res_shell=None, half_volume=False,
    tau2_fudge=1.0,
):
    """Adjust the RELION-style regularization filter.

    Adds 1/tau to the filter (Wiener denominator) and floors small values at
    1/1000 of the spherically-averaged filter to avoid division by zero.
    See RELION backprojector.cpp for the original algorithm.

    The ``tau2_fudge`` parameter mirrors RELION's ``--tau2_fudge`` flag
    (default 1.0).  It enters the Wiener denominator as::

        inv_tau = 1 / (padding_factor**3 * tau2_fudge * tau)
    """
    volume_shape = tuple(int(s) for s in volume_shape)
    packed_shape = (
        fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
    )
    filter_flat, input_is_grid = _as_flat_single_volume(filter, packed_shape)

    # Exact half-volume behavior: reuse full-volume implementation and repack.
    if half_volume:
        filter_full = fourier_transform_utils.half_volume_to_full_volume(filter_flat, volume_shape).reshape(-1).real
        reg_full = adjust_regularization_relion_style(
            filter_full,
            volume_shape,
            tau=tau,
            padding_factor=padding_factor,
            max_res_shell=max_res_shell,
            half_volume=False,
            tau2_fudge=tau2_fudge,
        )
        reg_half = fourier_transform_utils.full_volume_to_half_volume(reg_full, volume_shape).reshape(-1)
        if input_is_grid:
            return reg_half.reshape(packed_shape)
        return reg_half

    if tau is not None:
        # RELION: invtau2 = 1 / (padding_factor^3 * tau2_fudge * tau2[ires])
        oversampling_factor = padding_factor**3
        og_volume_shape = tuple(s // padding_factor for s in volume_shape)
        tau = upscale_tau(tau, padding_factor, og_volume_shape, tau_is_1d=False)
        safe_tau = jnp.where(tau > 1e-20, tau, jnp.float32(1.0))
        inv_tau = 1 / (oversampling_factor * tau2_fudge * safe_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter_flat > 1e-20), 1.0 / (0.001 * filter_flat), inv_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter_flat <= 1e-20), 0, inv_tau)
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
    )


@functools.partial(jax.jit, static_argnums=[2, 3, 5, 6, 7, 8, 9, 11, 12, 13])
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
):
    """Post-process RELION-style reconstruction from filter weights.

    Steps: regularize -> iDFT -> crop -> spherical mask -> grid correct -> DFT.

    Supports both full Fourier inputs ``(N0*N1*N2,)`` and packed half-volume
    inputs ``(N0*N1*(N2//2+1),)``.

    The ``tau2_fudge`` parameter (default 1.0) is forwarded to
    :func:`adjust_regularization_relion_style` and mirrors RELION's
    ``--tau2_fudge`` flag.
    """
    upsampled_volume_shape = tuple(3 * [og_volume_shape[0] * volume_upsampling_factor])
    if input_half_volume is None:
        input_half_volume = _infer_half_volume_layout(Ft_ctf, upsampled_volume_shape)

    if input_half_volume:
        packed_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape)
        Ft_ctf_flat, _ = _as_flat_single_volume(Ft_ctf, packed_shape)
        F_ty_flat, _ = _as_flat_single_volume(F_ty, packed_shape)
        # Expand canonical half-volume to full before regularization/iDFT.
        Ft_ctf_flat = (
            fourier_transform_utils.half_volume_to_full_volume(Ft_ctf_flat, upsampled_volume_shape).reshape(-1).real
        )
        F_ty_flat = fourier_transform_utils.half_volume_to_full_volume(F_ty_flat, upsampled_volume_shape).reshape(-1)
    else:
        Ft_ctf_flat, _ = _as_flat_single_volume(Ft_ctf, upsampled_volume_shape)
        F_ty_flat, _ = _as_flat_single_volume(F_ty, upsampled_volume_shape)

    valid_indices = (
        mask.get_radial_mask(upsampled_volume_shape, radius=upsampled_volume_shape[0] // 2 - 1)
        .reshape(-1)
        .astype(Ft_ctf_flat.real.dtype)
    )

    Ft_ctf2 = adjust_regularization_relion_style(
        Ft_ctf_flat.real,
        upsampled_volume_shape,
        tau=tau,
        padding_factor=volume_upsampling_factor,
        max_res_shell=None,
        half_volume=False,
        tau2_fudge=tau2_fudge,
    )
    vol = (F_ty_flat * valid_indices) / Ft_ctf2

    # iDFT → crop to original size
    vol = fourier_transform_utils.get_idft3(vol.reshape(upsampled_volume_shape))
    vol = padding.unpad_volume_spatial_domain(vol, upsampled_volume_shape[0] - og_volume_shape[0])

    if use_spherical_mask:
        vol, _ = mask.soft_mask_outside_map(vol, cosine_width=3)

    if volume_mask is not None:
        vol = vol * volume_mask

    if grid_correct:
        order = 1 if kernel == "triangular" else 0
        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        vol, _ = grid_fn(
            vol.reshape(og_volume_shape), og_volume_shape[0], volume_upsampling_factor / kernel_width, order=order
        )

    if return_real_space:
        return vol.real.astype(Ft_ctf2.real.dtype)

    vol = fourier_transform_utils.get_dft3(vol.reshape(og_volume_shape))
    if return_half_volume:
        vol = fourier_transform_utils.full_volume_to_half_volume(vol, og_volume_shape)
        return vol.reshape(-1).astype(F_ty_flat.dtype)
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
