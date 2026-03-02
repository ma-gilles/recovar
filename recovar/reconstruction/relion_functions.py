"""RELION-compatible reconstruction and Wiener filtering routines."""

import functools
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, jax_config, utils
from recovar.core import mask, padding
from recovar.core.configs import BatchData, ForwardModelConfig
from recovar.reconstruction import noise, regularization

logger = logging.getLogger(__name__)


def griddingCorrect(vol_in, ori_size, padding_factor, order=0):
    """Radial sinc gridding correction."""
    og_shape = vol_in.shape
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False) + 0.
    r = np.linalg.norm(pixels, axis=-1)
    safe_rval = np.where(r > 0, r / (ori_size * padding_factor), 1.0)
    sinc = np.where(r > 0, np.sin(np.pi * safe_rval) / (np.pi * safe_rval), 1.0)
    if order == 0:
        kernel = sinc
    elif order == 1:
        kernel = sinc ** 2
    else:
        raise ValueError("Order not implemented")
    return (vol_in.reshape(-1) / kernel).reshape(og_shape), kernel.reshape(og_shape)


def griddingCorrect_square(vol_in, ori_size, padding_factor, order=0):
    """Per-axis sinc product gridding correction (Fourier transform of trilinear interpolator)."""
    og_shape = vol_in.shape
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(og_shape, 1, scaled=False)
    pixels_rescaled = pixels / (ori_size * padding_factor)

    def sinc(ar):
        return jnp.where(jnp.abs(ar) < 1e-8, 1., jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

    if order == 0:
        kernel_fn = sinc
    elif order == 1:
        kernel_fn = lambda x: sinc(x) ** 2
    else:
        raise ValueError("Order not implemented")

    kernel = kernel_fn(pixels_rescaled[:, 0]) * kernel_fn(pixels_rescaled[:, 1]) * kernel_fn(pixels_rescaled[:, 2])
    return (vol_in / kernel.reshape(og_shape)).reshape(og_shape), kernel.reshape(og_shape)


def relion_style_triangular_kernel(
    experiment_dataset, cov_noise, batch_size=None, disc_type='linear_interp',
    data_generator=None, upsampling_factor=None,
):
    """RELION-style triangular kernel reconstruction.

    Loops over batches from *experiment_dataset*, accumulating the weighted
    back-projection (Ft_y) and CTF weight sum (Ft_ctf).

    When CUDA is available, accumulation uses half-volume layout for ~2x
    memory savings. The half-volumes are expanded to full volumes before
    returning.

    Parameters
    ----------
    experiment_dataset : CryoEMDataset
    cov_noise : array or None
        Per-image or radial noise variance. When None, per-batch noise is
        fetched from ``experiment_dataset.noise``.
    batch_size : int, optional
        Used to create an image generator internally. Mutually exclusive
        with *data_generator*.
    disc_type : str
    data_generator : iterable, optional
        Pre-built generator of ``(batch, particles_ind, indices)`` tuples.
    upsampling_factor : int, optional
        Volume oversampling factor. Defaults to the dataset's own
        ``volume_upsampling_factor``.
    """
    if batch_size is not None:
        data_generator = experiment_dataset.get_image_generator(batch_size=batch_size)

    uf = upsampling_factor if upsampling_factor is not None else experiment_dataset.volume_upsampling_factor
    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, upsampling_factor=uf)

    # Pre-expand 1D radial noise to half-image format once, outside the loop.
    if cov_noise is not None:
        cov_noise_arr = np.asarray(cov_noise)
        half_pixel_count = int(config.image_shape[0] * (config.image_shape[1] // 2 + 1))
        pixel_count = int(np.prod(config.image_shape))
        if cov_noise_arr.ndim == 1 and cov_noise_arr.size not in (pixel_count, half_pixel_count):
            cov_noise = noise.make_radial_noise_half(cov_noise_arr, config.image_shape).reshape(1, -1)

    Ft_y, Ft_ctf = None, None
    for batch, particles_ind, indices in data_generator:
        if cov_noise is not None:
            batch_noise = cov_noise
        elif hasattr(experiment_dataset.noise, 'get_half'):
            batch_noise = experiment_dataset.noise.get_half(indices)
        else:
            batch_noise = experiment_dataset.noise.get(indices)
        batch_data = BatchData(
            images=batch,
            ctf_params=experiment_dataset.CTF_params[indices],
            rotation_matrices=experiment_dataset.rotation_matrices[indices],
            translations=experiment_dataset.translations[indices],
            noise_variance=batch_noise,
        )
        Ft_y, Ft_ctf = relion_kernel_batch(config, batch_data, Ft_y=Ft_y, Ft_ctf=Ft_ctf)

    if Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(Ft_y, config.volume_shape).reshape(-1)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(Ft_ctf, config.volume_shape).reshape(-1)

    return Ft_ctf, Ft_y


@eqx.filter_jit
def relion_kernel_batch(
    config: ForwardModelConfig,
    batch: BatchData,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
):
    """RELION-style triangular kernel batch for raw real-space images.

    Applies pad + rfft2 internally, then backprojects with half_image and
    half_volume layouts for maximum memory efficiency.

    Parameters
    ----------
    batch : BatchData with real-valued ``(batch, H, W)`` images.
    Ft_y, Ft_ctf : optional accumulator volumes (half-volume layout) to add into.
    """
    half_images = padding.padded_rfft(
        batch.images * config.data_multiplier, config.grid_size, config.padding
    )
    return _relion_kernel_batch_half(
        config, half_images, batch.ctf_params, batch.rotation_matrices, batch.translations,
        batch.noise_variance, False, Ft_y, Ft_ctf,
    )


@eqx.filter_jit
def relion_kernel_batch_from_fft(
    config: ForwardModelConfig,
    batch: BatchData,
    use_upsampled_ctf: bool = False,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
):
    """RELION-style triangular kernel batch for pre-FFTed complex images.

    Extracts the half-spectrum from full-spectrum images, then backprojects
    using half_image and half_volume layouts.

    Parameters
    ----------
    batch : BatchData with complex-valued ``(batch, H*W)`` pre-processed images.
    use_upsampled_ctf : if True, compute CTF on 2x grid then downsample
        (used by the heterogeneity pipeline for aliasing suppression).
    Ft_y, Ft_ctf : optional accumulator volumes (half-volume layout) to add into.
    """
    half_images = fourier_transform_utils.full_image_to_half_image(batch.images, config.image_shape)
    return _relion_kernel_batch_half(
        config, half_images, batch.ctf_params, batch.rotation_matrices, batch.translations,
        batch.noise_variance, use_upsampled_ctf, Ft_y, Ft_ctf,
    )


def _relion_kernel_batch_half(
    config, half_images, ctf_params, rotation_matrices, translations,
    noise_variances, use_upsampled_ctf, Ft_y, Ft_ctf,
):
    """Shared implementation: backproject half-spectrum images into half-volume accumulators."""
    from recovar.core.geometry import translate_half_images

    half_images = translate_half_images(half_images, translations, config.image_shape)
    noise_half = noise.to_batched_half_pixel_noise(
        noise_variances, config.image_shape, batch_size=half_images.shape[0]
    )
    half_images = half_images / noise_half

    Ft_y = core_forward.adjoint_forward_model(
        config, half_images, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf,
        volume=Ft_y, half_image=True, half_volume=True,
    )

    if use_upsampled_ctf:
        # Compute CTF on 2x-upsampled grid and box-filter back to native
        # resolution before backprojecting. Used by the heterogeneity pipeline.
        upsample_factor = 2
        upsampled_shape = tuple(np.array(config.image_shape) * upsample_factor)
        ctf_up = config.CTF_fun(ctf_params, upsampled_shape, config.voxel_size) ** 2
        batch_size = ctf_up.shape[0]
        kernel_size = upsample_factor + upsample_factor // 2
        kernel = jnp.ones((1, 1, kernel_size, kernel_size), dtype=ctf_up.dtype) / (kernel_size ** 2)
        ctf_up = jax.lax.conv_general_dilated(
            ctf_up.reshape(batch_size, 1, *upsampled_shape),
            kernel, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NCHW', 'IOHW', 'NCHW'),
        ).squeeze(1)[:, ::upsample_factor, ::upsample_factor]
        ctf_half = fourier_transform_utils.full_image_to_half_image(
            ctf_up.reshape(batch_size, -1), config.image_shape
        ) / noise_half
        Ft_ctf = core_forward.adjoint_forward_model(
            config, ctf_half, ctf_params, rotation_matrices, skip_ctf=True,
            volume=Ft_ctf, half_image=True, half_volume=True,
        )
    else:
        ctf_half = config.compute_ctf_half(ctf_params) / noise_half
        Ft_ctf = core_forward.adjoint_forward_model(
            config, ctf_half, ctf_params, rotation_matrices,
            volume=Ft_ctf, half_image=True, half_volume=True,
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
    images = images - core.slice_volume_by_trilinear(
        mean_estimate, rotation_matrices, config.image_shape, config.volume_shape,
    ) * CTF
    images_squared = jnp.abs(images) ** 2 - cov_noise
    CTF_fourth = CTF ** 4

    images_squared_half = fourier_transform_utils.full_image_to_half_image(images_squared, config.image_shape)
    Ft_y = core.adjoint_slice_volume_by_trilinear_from_half_images(
        images_squared_half, rotation_matrices, config.image_shape, config.volume_shape,
        volume=Ft_y, half_volume=True,
    )
    CTF_fourth_half = fourier_transform_utils.full_image_to_half_image(CTF_fourth, config.image_shape)
    Ft_ctf = core.adjoint_slice_volume_by_trilinear_from_half_images(
        CTF_fourth_half, rotation_matrices, config.image_shape, config.volume_shape,
        volume=Ft_ctf, half_volume=True,
    )
    return Ft_y, Ft_ctf


def residual_relion_style_triangular_kernel(experiment_dataset, mean_estimate, cov_noise, batch_size, index_subset=None):
    """Residual RELION-style triangular kernel reconstruction."""
    if index_subset is None:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    else:
        data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices=index_subset)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type='linear_interp',
        upsampling_factor=experiment_dataset.volume_upsampling_factor,
    )

    Ft_y, Ft_ctf = None, None
    for batch, particles_ind, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask=False)
        Ft_y, Ft_ctf = residual_relion_kernel_trilinear(
            config, mean_estimate, batch,
            experiment_dataset.CTF_params[indices],
            experiment_dataset.rotation_matrices[indices],
            experiment_dataset.translations[indices],
            cov_noise,
            Ft_y=Ft_y, Ft_ctf=Ft_ctf,
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


def adjust_regularization_relion_style(filter, volume_shape, tau=None, padding_factor=1, max_res_shell=None):
    """Adjust the RELION-style regularization filter.

    Adds 1/tau to the filter (Wiener denominator) and floors small values at
    1/1000 of the spherically-averaged filter to avoid division by zero.
    See RELION backprojector.cpp for the original algorithm.
    """
    if tau is not None:
        oversampling_factor = padding_factor ** 3
        og_volume_shape = tuple(s // padding_factor for s in volume_shape)
        tau = upscale_tau(tau, padding_factor, og_volume_shape, tau_is_1d=False)
        safe_tau = jnp.where(tau > 1e-20, tau, jnp.float32(1.0))
        inv_tau = 1 / (oversampling_factor * safe_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter > 1e-20), 1. / (0.001 * filter), inv_tau)
        inv_tau = jnp.where((tau < 1e-20) & (filter <= 1e-20), 0, inv_tau)
        regularized_filter = filter + inv_tau
    else:
        regularized_filter = filter

    if max_res_shell is None:
        max_res_shell = volume_shape[0] // 2 - 1

    avged_reg = regularization.average_over_shells(regularized_filter, volume_shape, frequency_shift=0) / 1000
    avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
    avged_reg_volume = utils.make_radial_image(avged_reg, volume_shape).reshape(regularized_filter.shape)

    regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume)
    regularized_filter = jnp.maximum(regularized_filter, jax_config.EPSILON)
    return regularized_filter


def post_process_from_filter(cryo, Ft_ctf, F_ty, tau=None, disc_type='nearest', use_spherical_mask=True, grid_correct=True, gridding_correct="square", kernel_width=1):
    """Post-process RELION-style reconstruction from filter weights.

    Thin wrapper around ``post_process_from_filter_v2`` that extracts the
    necessary geometry from a dataset object (*cryo*).
    """
    kernel = 'triangular' if disc_type == 'linear_interp' else 'square'
    return post_process_from_filter_v2(
        Ft_ctf, F_ty,
        cryo.volume_shape, cryo.volume_upsampling_factor,
        tau=tau, kernel=kernel,
        use_spherical_mask=use_spherical_mask, grid_correct=grid_correct,
        gridding_correct=gridding_correct, kernel_width=kernel_width,
    )


@functools.partial(jax.jit, static_argnums=[2, 3, 5, 6, 7, 8, 9])
def post_process_from_filter_v2(
    Ft_ctf, F_ty, og_volume_shape, volume_upsampling_factor,
    tau=None, kernel='triangular', use_spherical_mask=True,
    grid_correct=True, gridding_correct="square", kernel_width=1,
    volume_mask=None,
):
    """Post-process RELION-style reconstruction from filter weights.

    Steps: regularize → iDFT → crop → spherical mask → grid correct → DFT.
    """
    upsampled_volume_shape = tuple(3 * [og_volume_shape[0] * volume_upsampling_factor])
    valid_indices = mask.get_radial_mask(
        upsampled_volume_shape, radius=upsampled_volume_shape[0] // 2 - 1
    ).reshape(-1).astype(Ft_ctf.real.dtype)

    Ft_ctf2 = adjust_regularization_relion_style(
        Ft_ctf.real, upsampled_volume_shape, tau=tau,
        padding_factor=volume_upsampling_factor, max_res_shell=None,
    )
    vol = (F_ty * valid_indices) / Ft_ctf2

    # iDFT → crop to original size
    vol = fourier_transform_utils.get_idft3(vol.reshape(upsampled_volume_shape))
    vol = padding.unpad_volume_spatial_domain(vol, upsampled_volume_shape[0] - og_volume_shape[0])

    if use_spherical_mask:
        vol, _ = mask.soft_mask_outside_map(vol, cosine_width=3)

    if volume_mask is not None:
        vol = vol * volume_mask

    if grid_correct:
        order = 1 if kernel == 'triangular' else 0
        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        vol, _ = grid_fn(vol.reshape(og_volume_shape), og_volume_shape[0], volume_upsampling_factor / kernel_width, order=order)

    vol = fourier_transform_utils.get_dft3(vol.reshape(og_volume_shape))
    return vol.astype(F_ty.dtype)


def relion_reconstruct(cryo, noise_variance, batch_size=100, disc_type='linear_interp', use_spherical_mask=True, upsampling_factor=2, grid_correct=True, gridding_correct="square", tau=None):
    """Full mean reconstruction pipeline: accumulate → post-process."""
    Ft_ctf, F_ty = relion_style_triangular_kernel(
        cryo, noise_variance.astype(np.float32), batch_size, disc_type=disc_type,
        upsampling_factor=upsampling_factor,
    )
    kernel = 'triangular' if disc_type == 'linear_interp' else 'square'
    estimate = post_process_from_filter_v2(
        Ft_ctf, F_ty, cryo.volume_shape, upsampling_factor,
        tau=tau, kernel=kernel,
        use_spherical_mask=use_spherical_mask, grid_correct=grid_correct,
        gridding_correct=gridding_correct, kernel_width=1,
    )
    return estimate, Ft_ctf
