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
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise, regularization

logger = logging.getLogger(__name__)


def griddingCorrect(vol_in, ori_size, padding_factor, order = 0,):

    # Correct real-space map by dividing it by the Fourier transform of the interpolator(s)
    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) + 0.
    og_shape = vol_in.shape
    r = np.linalg.norm(pixels, axis = -1)
    vol_in = vol_in.reshape(-1)

    mask = r > 0.
    
    rval = r / (ori_size * padding_factor)
    rval[~mask] = 1.
    sinc = np.sin(np.pi * rval) / (np.pi * rval)
    sinc[~mask] = 1.

    if order ==0:
        vol_out = vol_in/ sinc
    elif order ==1:
        vol_out = vol_in/ (sinc**2)
        sinc = sinc**2
    else:
        raise ValueError("Order not implemented")
    
    return vol_out.reshape(og_shape), sinc.reshape(og_shape)

# I think this is the correct Fourier transform of the trilinear interpolator: sinc(x) * sinc(y) * sinc(z)
def griddingCorrect_square(vol_in, ori_size, padding_factor, order = 0,):
    og_shape = vol_in.shape

    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) 
    pixels_rescaled = pixels / (ori_size * padding_factor)

    def sinc(ar):
        # ar_scaled = ar / (ori_size * padding_factor)
        return jnp.where(jnp.abs(ar) < 1e-8, 1., jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

    if order ==0:
        kernel = sinc
    elif order ==1:
        kernel = lambda x : sinc(x)**2
    else:
        raise ValueError("Order not implemented")

    kernel_ar = kernel(pixels_rescaled[:,0]) * kernel(pixels_rescaled[:,1]) * kernel(pixels_rescaled[:,2])
    vol_out = vol_in / kernel_ar.reshape(og_shape)

    return vol_out.reshape(og_shape), kernel_ar.reshape(og_shape)


def relion_style_triangular_kernel(experiment_dataset, cov_noise, batch_size=None, disc_type='linear_interp', data_generator=None, upsampling_factor=None):
    """RELION-style triangular kernel reconstruction.

    Loops over batches from *experiment_dataset*, accumulating the weighted
    back-projection (Ft_y) and CTF sum (Ft_ctf).

    Raw real-space images are passed directly to :func:`relion_kernel_batch`
    which handles pad + rfft → half-spectrum processing inside JIT.

    When CUDA is available, accumulation uses half-volume layout for better
    memory efficiency and cache utilization.  The half-volumes are expanded
    to full volumes before returning.

    Parameters
    ----------
    upsampling_factor : int, optional
        Volume upsampling factor.  When given, the config is created with
        this factor directly (no dataset mutation needed).  When ``None``,
        falls back to ``use_upsampled=True`` which reads the dataset's
        current ``volume_upsampling_factor``.
    """
    if batch_size is None and data_generator is None:
        raise ValueError("Either batch_size or data_generator must be provided")
    if batch_size is not None and data_generator is not None:
        raise ValueError("Either batch_size or data_generator must be provided, not both")

    if batch_size is not None:
        data_generator = experiment_dataset.get_image_generator(batch_size=batch_size)

    if upsampling_factor is not None:
        config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, upsampling_factor=upsampling_factor)
    else:
        config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type, use_upsampled=True)

    # Expand 1D radial noise to half-image format directly
    if cov_noise is not None:
        cov_noise_arr = np.asarray(cov_noise)
        half_pixel_count = int(config.image_shape[0] * (config.image_shape[1] // 2 + 1))
        pixel_count = np.prod(config.image_shape)
        if cov_noise_arr.ndim == 1 and cov_noise_arr.size != pixel_count and cov_noise_arr.size != half_pixel_count:
            # Radial noise → half-pixel expansion (native)
            cov_noise = noise.make_radial_noise_half(cov_noise_arr, config.image_shape).reshape(1, -1)

    # Use half-volume accumulation when CUDA is available for ~2x less memory
    try:
        from recovar.cuda_backproject import cuda_available
        use_half_vol = cuda_available()
        logger.info("CUDA backproject/project kernels enabled")
    except (ImportError, OSError):
        use_half_vol = False
        logger.info("CUDA backproject/project kernels disabled")
        
    Ft_y, Ft_ctf = None, None
    for batch, particles_ind, indices in data_generator:
        if cov_noise is not None:
            batch_noise = cov_noise
        elif hasattr(experiment_dataset.noise, 'get_half'):
            batch_noise = experiment_dataset.noise.get_half(indices)
        else:
            batch_noise = experiment_dataset.noise.get(indices)
        Ft_y, Ft_ctf = relion_kernel_batch(
            config, batch,
            experiment_dataset.CTF_params[indices],
            experiment_dataset.rotation_matrices[indices],
            experiment_dataset.translations[indices],
            batch_noise,
            Ft_y=Ft_y, Ft_ctf=Ft_ctf,
            half_volume=use_half_vol,
        )

    # Convert half-volume → full-volume for downstream consumers
    if use_half_vol and Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(
            Ft_y, config.volume_shape
        ).reshape(-1)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(
            Ft_ctf, config.volume_shape
        ).reshape(-1)

    return Ft_ctf, Ft_y


@eqx.filter_jit
def relion_kernel_batch(
    config: ForwardModelConfig,
    images: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    translations: jax.Array,
    noise_variances: jax.Array,
    use_upsampled_ctf: bool = False,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    half_volume: bool = False,
):
    """RELION-style triangular kernel batch for raw real-space images.

    Takes raw real-space images ``(batch, H, W)``, applies pad + rfft2 to
    get half-spectrum, then backprojects with ``half_image=True``.

    Parameters
    ----------
    images : real-valued ``(batch, H, W)`` raw images.
    Ft_y, Ft_ctf : optional accumulator volumes to add into directly.
    half_volume : if True, output volumes use rfft-packed layout.
    """
    half_images = padding.padded_rfft(
        images * config.data_multiplier, config.grid_size, config.padding
    )
    return _relion_kernel_batch_half(
        config, half_images, ctf_params, rotation_matrices, translations,
        noise_variances, use_upsampled_ctf, Ft_y, Ft_ctf, half_volume,
    )


@eqx.filter_jit
def relion_kernel_batch_from_fft(
    config: ForwardModelConfig,
    images: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    translations: jax.Array,
    noise_variances: jax.Array,
    use_upsampled_ctf: bool = False,
    Ft_y: jax.Array = None,
    Ft_ctf: jax.Array = None,
    half_volume: bool = False,
):
    """RELION-style triangular kernel batch for pre-FFTed complex images.

    Takes full-spectrum complex images ``(batch, H*W)``, extracts the
    half-spectrum, then backprojects with ``half_image=True``.

    Parameters
    ----------
    images : complex-valued ``(batch, H*W)`` pre-processed images.
    Ft_y, Ft_ctf : optional accumulator volumes to add into directly.
    half_volume : if True, output volumes use rfft-packed layout.
    """
    half_images = fourier_transform_utils.full_image_to_half_image(
        images, config.image_shape
    )
    return _relion_kernel_batch_half(
        config, half_images, ctf_params, rotation_matrices, translations,
        noise_variances, use_upsampled_ctf, Ft_y, Ft_ctf, half_volume,
    )


def _relion_kernel_batch_half(
    config, half_images, ctf_params, rotation_matrices, translations,
    noise_variances, use_upsampled_ctf, Ft_y, Ft_ctf, half_volume,
):
    """Shared implementation: backproject half-spectrum images."""
    from recovar.core.geometry import translate_half_images

    half_images = translate_half_images(half_images, translations, config.image_shape)

    noise_half = noise.to_batched_half_pixel_noise(
        noise_variances, config.image_shape, batch_size=half_images.shape[0]
    )
    half_images = half_images / noise_half

    Ft_y = core_forward.adjoint_forward_model(
        config, half_images, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf,
        volume=Ft_y, half_image=True, half_volume=half_volume,
    )

    ctf_half = config.compute_ctf_half(ctf_params) / noise_half

    if use_upsampled_ctf:
        upsample_factor = 2
        upsampled_shape = tuple(np.array(config.image_shape) * upsample_factor)
        upsampled_CTF_squared = config.CTF_fun(ctf_params, upsampled_shape, config.voxel_size) ** 2
        batch_size = upsampled_CTF_squared.shape[0]
        ctf = upsampled_CTF_squared.reshape(batch_size, *upsampled_shape)
        kernel_size = upsample_factor + upsample_factor // 2
        kernel = jnp.ones((kernel_size, kernel_size), dtype=ctf.dtype) / (kernel_size * kernel_size)
        ctf = jnp.expand_dims(ctf, 1)
        kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
        ctf = jax.lax.conv_general_dilated(
            ctf, kernel, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NCHW', 'IOHW', 'NCHW'),
        )
        ctf = jnp.squeeze(ctf, axis=1)[:, ::upsample_factor, ::upsample_factor]
        CTF_squared = ctf.reshape(batch_size, -1)
        CTF_squared_half = fourier_transform_utils.full_image_to_half_image(
            CTF_squared, config.image_shape
        ) / noise_half
        Ft_ctf = core_forward.adjoint_forward_model(
            config, CTF_squared_half, ctf_params, rotation_matrices, skip_ctf=True,
            volume=Ft_ctf, half_image=True, half_volume=half_volume,
        )
        return Ft_y, Ft_ctf

    Ft_ctf = core_forward.adjoint_forward_model(
        config, ctf_half, ctf_params, rotation_matrices,
        volume=Ft_ctf, half_image=True, half_volume=half_volume,
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
    half_volume: bool = False,
):
    """Residual RELION-style kernel (trilinear) — Equinox API.

    Parameters
    ----------
    Ft_y, Ft_ctf : optional accumulator volumes to add into directly.
    half_volume : if True, accumulate in rfft-packed half-volume layout.
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
        volume=Ft_y, half_volume=half_volume,
    )
    CTF_fourth_half = fourier_transform_utils.full_image_to_half_image(CTF_fourth, config.image_shape)
    Ft_ctf = core.adjoint_slice_volume_by_trilinear_from_half_images(
        CTF_fourth_half, rotation_matrices, config.image_shape, config.volume_shape,
        volume=Ft_ctf, half_volume=half_volume,
    )
    return Ft_y, Ft_ctf


def residual_relion_style_triangular_kernel(experiment_dataset, mean_estimate, cov_noise, batch_size, index_subset=None):
    """Residual RELION-style triangular kernel reconstruction."""
    if index_subset is None:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    else:
        data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices=index_subset)

    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type='linear_interp', use_upsampled=True)

    try:
        from recovar.cuda_backproject import cuda_available
        use_half_vol = cuda_available()
    except (ImportError, OSError):
        use_half_vol = False

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
            half_volume=use_half_vol,
        )

    # Convert half-volume → full-volume for downstream consumers
    if use_half_vol and Ft_y is not None:
        Ft_y = fourier_transform_utils.half_volume_to_full_volume(
            Ft_y, config.volume_shape
        ).reshape(-1)
        Ft_ctf = fourier_transform_utils.half_volume_to_full_volume(
            Ft_ctf, config.volume_shape
        ).reshape(-1)

    if Ft_y is None:
        vol_size = int(np.prod(config.volume_shape))
        Ft_y = jnp.zeros(vol_size, dtype=experiment_dataset.dtype)
        Ft_ctf = jnp.zeros(vol_size, dtype=experiment_dataset.dtype_real)

    return Ft_ctf, Ft_y


def upscale_tau(tau, padding_factor, volume_shape, tau_is_1d = False):

    if not tau_is_1d:
        tau = regularization.average_over_shells(tau, volume_shape)

    # int ires = ROUND(sqrt((RFLOAT)r2) / padding_factor);
    # RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

    # RFLOAT invtau2;
    # if (DIRECT_A1D_ELEM(tau2, ires) > 0.)

    pixels = fourier_transform_utils.get_k_coordinate_of_each_pixel(np.array(volume_shape)*padding_factor, 1, scaled = False)
    radius = jnp.round(jnp.linalg.norm(pixels, axis = -1) / padding_factor).astype(jnp.int32)
    upscaled_tau = tau[radius]

    return upscaled_tau

def adjust_regularization_relion_style(filter, volume_shape, tau = None, padding_factor = 1, max_res_shell = None):

    # Original code here https://github.com/3dem/relion/blob/e5c4835894ea7db4ad4f5b0f4861b33269dbcc77/src/backprojector.cpp#L1082

    # There is an "oversampling" factor of 8 in the FSC, I guess due to the fact that they swap back and forth between a padded and unpadded grid

    if tau is not None:
        oversampling_factor = padding_factor ** (3)
        og_volume_shape = (volume_shape[0]//padding_factor, volume_shape[1]//padding_factor, volume_shape[2]//padding_factor)
        tau = upscale_tau(tau, padding_factor, og_volume_shape, tau_is_1d = False)
        inv_tau = 1 / (oversampling_factor * tau)
        inv_tau = jnp.where( (tau < 1e-20) * (filter > 1e-20 ),  1./ ( 0.001 * filter), inv_tau)
        inv_tau = jnp.where( (tau < 1e-20) * (filter <= 1e-20 ),  0, inv_tau)

        regularized_filter = filter + inv_tau
    else:
        regularized_filter = filter

    # This may be a little different b/c I keep things scaled slightly differently. Perhaps should be fixed in fourier_transform_utils
        
    # Take max of weight of 1/1000 of spherically averaged weight 
    # const RFLOAT weight =  XMIPP_MAX(DIRECT_A3D_ELEM(Fweight, k, i, j), DIRECT_A1D_ELEM(radavg_weight, (ires < r_max) ? ires : (r_max - 1)));
    # Compute spherically averaged 
    avged_reg = regularization.average_over_shells(regularized_filter, volume_shape, frequency_shift = 0) / 1000
    # For the things below that frequency, set them to averaged.
    if max_res_shell is not None:
        avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
    else:
        max_res_shell = volume_shape[0]//2 - 1
        # avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])

    avged_reg_volume_shape = utils.make_radial_image(avged_reg, volume_shape).reshape(regularized_filter.shape)

    regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume_shape)
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



@functools.partial(jax.jit, static_argnums=[2,3,5,6,7,8,9])
def post_process_from_filter_v2(Ft_ctf, F_ty, og_volume_shape, volume_upsampling_factor, tau = None, kernel = 'triangular', use_spherical_mask = True, grid_correct = True, gridding_correct = "square", kernel_width = 1, volume_mask = None ):
    
    Ft_ctf= Ft_ctf.real
    upsampled_volume_shape = tuple(3*[(og_volume_shape[0]*volume_upsampling_factor)])
    valid_indices = mask.get_radial_mask(upsampled_volume_shape, radius = upsampled_volume_shape[0]//2-1).reshape(-1).astype(Ft_ctf.dtype)
    F_ty =  F_ty * valid_indices # Zero-out FT outside sphere

    # Adjust reg for small values
    Ft_ctf2 = adjust_regularization_relion_style(Ft_ctf, upsampled_volume_shape, tau = tau, padding_factor = volume_upsampling_factor, max_res_shell = None)
    
    myreliontest = F_ty / Ft_ctf2
    
    # Window real space
    myreliontest = fourier_transform_utils.get_idft3(myreliontest.reshape(upsampled_volume_shape))

    myreliontest = padding.unpad_volume_spatial_domain(myreliontest, (upsampled_volume_shape[0] - og_volume_shape[0]) )
    

    # Soft Spherical mask
    if use_spherical_mask:
        myreliontest, mask2 = mask.soft_mask_outside_map(myreliontest, cosine_width = 3)
    
    if volume_mask is not None:
        logger.warning("Applying mask in post_proces_from_filter_v2") 
        myreliontest = myreliontest * volume_mask

    # Correct gridding effect
    if grid_correct:

        if kernel == 'triangular':
            order = 1
        elif kernel == 'square':
            order = 0
        else:
            raise ValueError("Kernel not implemented")
        # order = 1 if disc_type == 'linear_interp' else 0

        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        myreliontest, sinc = grid_fn(myreliontest.reshape(og_volume_shape), og_volume_shape[0], volume_upsampling_factor/kernel_width, order = order)

    myreliontest = fourier_transform_utils.get_dft3(myreliontest.reshape(og_volume_shape))


    return myreliontest.astype(F_ty.dtype)


def relion_reconstruct(cryo, noise_variance, batch_size=100, disc_type='linear_interp', use_spherical_mask=True, upsampling_factor=2, grid_correct=True, gridding_correct="square", tau=None):
    Ft_ctf, F_ty = relion_style_triangular_kernel(
        cryo, noise_variance.astype(np.float32), batch_size, disc_type=disc_type,
        upsampling_factor=upsampling_factor,
    )
    kernel = 'triangular' if disc_type == 'linear_interp' else 'square'
    estimate = post_process_from_filter_v2(
        Ft_ctf, F_ty,
        cryo.volume_shape, upsampling_factor,
        tau=tau, kernel=kernel,
        use_spherical_mask=use_spherical_mask, grid_correct=grid_correct,
        gridding_correct=gridding_correct, kernel_width=1,
    )
    return estimate, Ft_ctf
