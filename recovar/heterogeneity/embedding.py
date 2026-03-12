"""Per-image latent coordinate estimation via linear projection."""

import logging
import time
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as ftu
from recovar import core, jax_config, utils
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig, BatchData, ModelState, EmbeddingOpts
from recovar.heterogeneity import covariance_core

logger = logging.getLogger(__name__)

# NVTX domain for embedding/latent variable operations
NVTX_DOMAIN_EMBED = "embedding"
USE_CUBIC = True


_rfft2_hermitian_weights = linalg.rfft2_hermitian_weights


def split_weights(weight, cryos):
    start_idx = 0
    weights = []
    for cryo in cryos:
        end_idx = start_idx + cryo.n_images
        weights.append(weight[start_idx:end_idx])
        start_idx = end_idx
    return weights

def generate_conformation_from_reprojection(xs, mean, u ):
    return ((mean[...,None] + u @ xs.T)[0]).T


def _volume_layout_sizes(volume_shape):
    full_size = int(np.prod(volume_shape))
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    return full_size, half_size


def _mean_is_half_volume(mean_estimate, volume_shape):
    _, half_size = _volume_layout_sizes(volume_shape)
    return int(np.prod(mean_estimate.shape)) == half_size


def _basis_is_half_volume(basis, volume_shape):
    _, half_size = _volume_layout_sizes(volume_shape)
    if basis.ndim < 1:
        return False
    per_vec_size = int(np.prod(basis.shape[1:])) if basis.shape[0] != 0 else 0
    return per_vec_size == half_size


def _prepare_model_half_volumes(config, mean_estimate, basis):
    """Convert full Fourier volumes to half-volume layout when supported."""
    full_size, half_size = _volume_layout_sizes(config.volume_shape)

    mean_size = int(np.prod(mean_estimate.shape))
    if mean_size == full_size:
        mean_estimate = ftu.full_volume_to_half_volume(
            mean_estimate.reshape(config.volume_shape), config.volume_shape,
        ).reshape(-1)
    elif mean_size != half_size:
        logger.warning(
            "Unexpected mean_estimate size %d for volume_shape %s; expected %d (full) or %d (half).",
            mean_size, config.volume_shape, full_size, half_size,
        )

    if basis.ndim >= 1 and basis.shape[0] > 0:
        basis_size = int(np.prod(basis.shape[1:]))
        if basis_size == full_size:
            n_basis = basis.shape[0]
            basis = ftu.full_volume_to_half_volume(
                basis.reshape(n_basis, *config.volume_shape), config.volume_shape,
            ).reshape(n_basis, -1)
        elif basis_size != half_size:
            logger.warning(
                "Unexpected basis vector size %d for volume_shape %s; expected %d (full) or %d (half).",
                basis_size, config.volume_shape, full_size, half_size,
            )

    return mean_estimate, basis


def _noise_get_half_or_full(noise_model, image_indices, prefer_half=True):
    """Fetch noise variance, optionally preferring half-spectrum layout."""
    if noise_model is None:
        return None
    get_half = getattr(noise_model, "get_half", None) if prefer_half else None
    if callable(get_half):
        return get_half(image_indices)
    return noise_model.get(image_indices)


def _embedding_hermitian_weights(config: ForwardModelConfig):
    """Half-spectrum weights for embedding inner products, or ``None``."""
    return _rfft2_hermitian_weights(config.image_shape)


def _particle_ids_per_image(particles_ind, n_images):
    """Normalize batch particle ids to one id per image."""
    particle_ids = np.asarray(particles_ind).reshape(-1)
    if particle_ids.size == n_images:
        return particle_ids
    if particle_ids.size == 1:
        return np.full(n_images, particle_ids[0], dtype=particle_ids.dtype)
    raise ValueError(
        f"Unexpected particles_ind size {particle_ids.size} for batch with {n_images} images"
    )


@nvtx.annotate("get_per_image_embedding", color="purple", domain=NVTX_DOMAIN_EMBED)
def get_per_image_embedding(mean, u, s, basis_size, cryos, volume_mask, gpu_memory, disc_type = 'linear_interp',  contrast_grid = None, contrast_option = "contrast", to_real = True, compute_covariances = True, ignore_zero_frequency = False, contrast_mean = 1, contrast_variance = np.inf, compute_bias = False, image_subset_in_tilt_series = None):
    """Compute per-image latent coordinates by projecting onto principal components.

    For each image, estimates the linear coefficients (latent embedding)
    that best explain the image given the mean volume and eigenvectors,
    optionally estimating per-image contrast and covariance.

    Args:
        mean: Mean volume in Fourier space, shape ``(volume_size,)``.
        u: Eigenvectors, shape ``(volume_size, n_components)``.
        s: Eigenvalues, shape ``(n_components,)``.
        basis_size: Number of principal components to use.
        cryos: Half-set datasets (``CryoEMHalfsets``).
        volume_mask: Binary mask selecting valid voxels.
        gpu_memory: Available GPU memory in GB.
        disc_type: Discretization type (``'linear_interp'`` or ``'cubic'``).
        contrast_grid: Grid of contrast values to search over.
        contrast_option: Contrast estimation mode (``'contrast'``,
            ``'contrast_shared'``, or ``'none'``).
        to_real: Convert output to real-valued coordinates.
        compute_covariances: Compute per-image latent covariance matrices.
        ignore_zero_frequency: Exclude the DC component.
        contrast_mean: Prior mean for contrast estimation.
        contrast_variance: Prior variance for contrast estimation.
        compute_bias: Compute per-image bias terms.
        image_subset_in_tilt_series: Subset of tilt images to use.

    Returns:
        Tuple ``(zs, precision_zs, est_contrasts, bias)`` where *zs* has shape
        ``(n_images, basis_size)``, *precision_zs* is the per-image posterior
        precision matrix (inverse covariance) with shape
        ``(n_images, basis_size, basis_size)`` (or ``None``),
        *est_contrasts* has shape ``(n_images,)``, and *bias* is
        ``None`` unless *compute_bias* is ``True``.
    """

    if u.shape[0] != cryos.volume_size:
        raise ValueError(f"input u should be volume_size x basis_size, got {u.shape[0]} != {cryos.volume_size}")
    st_time = time.time()    
    basis = np.asarray(u[:, :basis_size]).T
    eigenvalues = (s + jax_config.ROOT_EPSILON)
    use_contrast = "contrast" in contrast_option
    contrast_shared_across_tilt_series = ("shared" in contrast_option) #and not use_contrast
    logger.info("using contrast? %s", use_contrast)

    if use_contrast:
        contrast_grid = np.linspace(0, 2, 51)[1:] if contrast_grid is None else contrast_grid
    else:
        contrast_grid = np.ones([1])
    
    basis_size = u.shape[-1] if basis_size == -1 else basis_size

    batch_size = utils.get_embedding_batch_size(basis, cryos.image_size, contrast_grid, basis_size, gpu_memory)
    # JIT trace uses ~10x more peak memory than the raw array estimate
    _EMBEDDING_BATCH_SAFETY_FACTOR = 10
    batch_size = utils.safe_batch_size(batch_size // _EMBEDDING_BATCH_SAFETY_FACTOR)
    logger.info("embedding batch size: %s", batch_size)

    # It is not so clear whether this step should ever use the mask. But when using the options['ignore_zero_frequency'] option, there is a good reason not to do it
    if ignore_zero_frequency:
        volume_mask = np.ones_like(volume_mask)

    logger.info("ignore_zero_frequency? %s", ignore_zero_frequency)

    if USE_CUBIC:
        disc_type = 'cubic'
        from recovar.core import cubic_interpolation
        mean = cubic_interpolation.calculate_spline_coefficients(mean.reshape(cryos.volume_shape))
        from recovar.heterogeneity import covariance_estimation
        basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, cryos.volume_shape, gpu_memory= None)

    ## TODO: I don't love the way this is handled either. Perhaps should be stored in a more clever consistent way
    ## E.g. instantly resort to the right order, when CryoDataset also gets cleaned up
    ## Perhaps 
    n_cryos = len(cryos)
    zs = [None] * n_cryos
    cov_zs = [None] * n_cryos
    est_contrasts = [None] * n_cryos
    bias = [None] * n_cryos
    for cryo_idx,cryo in enumerate(cryos):
        zs[cryo_idx], cov_zs[cryo_idx], est_contrasts[cryo_idx], bias[cryo_idx] = get_coords_in_basis_and_contrast_3(
            cryo, mean, basis, eigenvalues[:basis.shape[0]], volume_mask,
             contrast_grid, batch_size, disc_type, 
            compute_covariances = compute_covariances, contrast_mean = contrast_mean, contrast_variance = contrast_variance , compute_bias = compute_bias, image_subset_in_tilt_series = image_subset_in_tilt_series, contrast_shared_across_tilt_series= contrast_shared_across_tilt_series)

    
    zs = np.concatenate(zs, axis = 0)
    est_contrasts = np.concatenate(est_contrasts)
    end_time = time.time()
    logger.info("time to compute xs %s", end_time - st_time)
    
    if compute_covariances:
        cov_zs = np.concatenate(cov_zs, axis = 0)
        if to_real:
            cov_zs = cov_zs.real

    if compute_bias:
        bias = np.concatenate(bias, axis = 0)
        if to_real:
            bias = bias.real

    if to_real:
        zs = zs.real
    
    return zs, cov_zs, est_contrasts, bias
    
## TODO: a lot of implementations of same thing. It shoudl be refactored better and the names as well.
## Also it should be benchmarked. Is my whole "precompute matrices, then do fast contrast with precompute" make sense
## Also these functoins have too many inputs
@nvtx.annotate("get_coords_in_basis_and_contrast", color="blue", domain=NVTX_DOMAIN_EMBED)
def get_coords_in_basis_and_contrast_3(experiment_dataset, mean_estimate, basis, eigenvalues, volume_mask, contrast_grid, batch_size, disc_type, compute_covariances = True, contrast_mean = 1, contrast_variance = np.inf, compute_bias = False, image_subset_in_tilt_series = None, force_not_shared_label = False, contrast_shared_across_tilt_series = False):

    shared_label = experiment_dataset.tilt_series_flag and not force_not_shared_label
    n_units = experiment_dataset.n_units if not force_not_shared_label else experiment_dataset.n_images

    # Transfer arrays to GPU once before the batch loop
    basis = jnp.asarray(basis, dtype=experiment_dataset.dtype)
    volume_mask = jnp.asarray(volume_mask, dtype=experiment_dataset.dtype_real)
    mean_estimate = jnp.asarray(mean_estimate, dtype=experiment_dataset.dtype)
    eigenvalues = jnp.asarray(eigenvalues, dtype=experiment_dataset.dtype)
    contrast_grid = contrast_grid.astype(experiment_dataset.dtype_real)

    # Construct structured parameters once outside the loop
    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.image_stack.process_images,
    )
    # Embedding uses half-spectrum inner products by default; pre-convert model
    # Fourier volumes once so forward passes can use native half-volume kernels.
    mean_estimate, basis = _prepare_model_half_volumes(config, mean_estimate, basis)
    model = ModelState(
        mean_estimate=mean_estimate,
        volume_mask=volume_mask,
        basis=basis,
        eigenvalues=eigenvalues[:basis.shape[0]],
    )
    opts = EmbeddingOpts(
        compute_covariances=compute_covariances,
        compute_bias=compute_bias,
        shared_label=shared_label,
        contrast_shared_across_tilt_series=contrast_shared_across_tilt_series,
    )

    basis_size = basis.shape[0]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)

    xs = np.zeros((n_units, basis_size), dtype=basis.dtype)
    image_latent_precisions = np.zeros((n_units, basis_size, basis_size), dtype=basis.dtype) if compute_covariances else None
    image_latent_bias = np.zeros((n_units, basis_size, basis_size), dtype=basis.dtype) if compute_bias else None

    contrast_units = n_units if contrast_shared_across_tilt_series else experiment_dataset.n_images
    estimated_contrasts = np.zeros(contrast_units, dtype=basis.dtype).real

    # Precompute half-spectrum Hermitian weights once — halves memory and
    # compute cost of the inner-product step inside compute_batch_coords.
    hermitian_weights = _embedding_hermitian_weights(config)
    prefer_half_noise = hermitian_weights is not None

    noise_model = experiment_dataset.noise

    for batch, particles_ind, batch_image_ind in data_generator:
        batch = jnp.asarray(batch)
        batch_image_ind = np.asarray(batch_image_ind).reshape(-1)
        particle_ids = _particle_ids_per_image(particles_ind, batch.shape[0])

        # Handle tilt series image subsetting
        if image_subset_in_tilt_series is not None:
            subset = np.asarray(image_subset_in_tilt_series, dtype=np.int32).reshape(-1)
            subset = subset[(subset >= 0) & (subset < batch.shape[0])]
            if subset.size == 0:
                continue
            batch = batch[subset]
            batch_image_ind = batch_image_ind[subset]
            particle_ids = particle_ids[subset]

        if shared_label:
            # Some iterators may return multiple particles in one image batch.
            # Shared-label solves must be computed per particle (tilt series).
            unique_particles, particle_group_ids = np.unique(particle_ids, return_inverse=True)

            # Common case with the current tilt-series loader: one particle per
            # iterator batch. Skip boolean-mask splitting to avoid extra gathers.
            if unique_particles.size == 1:
                particle_ind = np.asarray(unique_particles, dtype=np.int32)
                batch_data = BatchData(
                    images=batch,
                    ctf_params=experiment_dataset.CTF_params[batch_image_ind],
                    rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
                    translations=experiment_dataset.translations[batch_image_ind],
                    noise_variance=_noise_get_half_or_full(noise_model, batch_image_ind, prefer_half=prefer_half_noise),
                )

                xs_single, contrast_single, cov_batch, bias = compute_batch_coords(
                    config, batch_data, model, opts,
                    experiment_dataset.image_stack.mask, contrast_grid,
                    contrast_mean, contrast_variance,
                    hermitian_weights,
                )

                xs[particle_ind] = xs_single
                if not contrast_shared_across_tilt_series:
                    estimated_contrasts[batch_image_ind] = contrast_single
                else:
                    estimated_contrasts[particle_ind] = contrast_single

                if compute_covariances:
                    image_latent_precisions[particle_ind] = cov_batch
                if compute_bias:
                    image_latent_bias[particle_ind] = bias
                continue

            # Fast path: when contrast is shared across tilts and a batch contains
            # multiple particles, solve all particle groups in one batched call.
            if contrast_shared_across_tilt_series and unique_particles.size > 1:
                batch_data = BatchData(
                    images=batch,
                    ctf_params=experiment_dataset.CTF_params[batch_image_ind],
                    rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
                    translations=experiment_dataset.translations[batch_image_ind],
                    noise_variance=_noise_get_half_or_full(noise_model, batch_image_ind, prefer_half=prefer_half_noise),
                )

                xs_group, contrast_group, cov_group, bias_group = compute_grouped_shared_batch_coords(
                    config,
                    batch_data,
                    model,
                    experiment_dataset.image_stack.mask,
                    contrast_grid,
                    contrast_mean,
                    contrast_variance,
                    jnp.asarray(particle_group_ids, dtype=jnp.int32),
                    int(unique_particles.size),
                    compute_covariances,
                    compute_bias,
                    hermitian_weights,
                )

                xs[unique_particles] = np.asarray(xs_group)
                estimated_contrasts[unique_particles] = np.asarray(contrast_group)

                if compute_covariances:
                    image_latent_precisions[unique_particles] = np.asarray(cov_group)
                if compute_bias:
                    image_latent_bias[unique_particles] = np.asarray(bias_group)
                continue

            for pid in unique_particles:
                mask = (particle_ids == pid)
                if not np.any(mask):
                    continue

                local_image_ind = batch_image_ind[mask]
                local_batch = batch[mask]
                local_particle_ind = np.asarray([pid], dtype=np.int32)

                batch_data = BatchData(
                    images=local_batch,
                    ctf_params=experiment_dataset.CTF_params[local_image_ind],
                    rotation_matrices=experiment_dataset.rotation_matrices[local_image_ind],
                    translations=experiment_dataset.translations[local_image_ind],
                    noise_variance=_noise_get_half_or_full(noise_model, local_image_ind, prefer_half=prefer_half_noise),
                )

                xs_single, contrast_single, cov_batch, bias = compute_batch_coords(
                    config, batch_data, model, opts,
                    experiment_dataset.image_stack.mask, contrast_grid,
                    contrast_mean, contrast_variance,
                    hermitian_weights,
                )

                xs[local_particle_ind] = xs_single
                if not contrast_shared_across_tilt_series:
                    estimated_contrasts[local_image_ind] = contrast_single
                else:
                    estimated_contrasts[local_particle_ind] = contrast_single

                if compute_covariances:
                    image_latent_precisions[local_particle_ind] = cov_batch
                if compute_bias:
                    image_latent_bias[local_particle_ind] = bias
            continue

        batch_data = BatchData(
            images=batch,
            ctf_params=experiment_dataset.CTF_params[batch_image_ind],
            rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
            translations=experiment_dataset.translations[batch_image_ind],
            noise_variance=_noise_get_half_or_full(noise_model, batch_image_ind, prefer_half=prefer_half_noise),
        )

        xs_single, contrast_single, cov_batch, bias = compute_batch_coords(
            config, batch_data, model, opts,
            experiment_dataset.image_stack.mask, contrast_grid,
            contrast_mean, contrast_variance,
            hermitian_weights,
        )

        target_ind = batch_image_ind if force_not_shared_label else np.asarray(particle_ids)
        xs[target_ind] = xs_single

        if not contrast_shared_across_tilt_series:
            estimated_contrasts[batch_image_ind] = contrast_single
        else:
            estimated_contrasts[target_ind] = contrast_single

        if compute_covariances:
            image_latent_precisions[target_ind] = cov_batch

        if compute_bias:
            image_latent_bias[target_ind] = bias

    return xs, image_latent_precisions, estimated_contrasts, image_latent_bias


def slice_ar(indx, arr):
    return arr[indx]
# Vectorized index selection via vmap
batch_slice_ar = jax.jit(jax.vmap(slice_ar, in_axes =(0, 0)))
batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))


# ============================================================================
# New Equinox-based embedding API
# ============================================================================

## 
def _compute_batch_coords_p1(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    hermitian_weights=None,
):
    """Phase 1: compute inner products AU^T images, AU^T Amean, etc.

    Internal helper used by :func:`compute_batch_coords`.

    Parameters
    ----------
    hermitian_weights : jax.Array or None
        Precomputed ``sqrt(w)`` weights of shape ``(H*(W//2+1),)`` for
        half-spectrum inner products (see :func:`_rfft2_hermitian_weights`).
        When provided, all frequency arrays are converted to half-spectrum and
        weighted so that the plain half-spectrum inner product equals the
        correct full-spectrum inner product.  When ``None``, the computation
        stays in full Fourier space.
    """
    batch = batch_data.images
    ctf_params = batch_data.ctf_params
    rotation_matrices = batch_data.rotation_matrices
    translations = batch_data.translations
    noise_variance = batch_data.noise_variance

    if config.process_fn is not None:
        batch = config.process_fn(batch)
    batch = batch.reshape(batch.shape[0], -1)

    # --- Project volumes to image space ---
    # Multiply projected arrays by sqrt(w) so that the plain half-spectrum inner
    # product equals the correct Hermitian-weighted full-spectrum inner product:
    #   <A_w, B_w>_half = sum_{k in half} w[k]*conj(A[k])*B[k] = <A, B>_full
    half = hermitian_weights is not None
    full_image_size = int(np.prod(config.image_shape))
    half_image_size = int(np.prod(ftu.image_shape_to_half_image_shape(config.image_shape)))

    if half:
        if batch.shape[-1] == full_image_size:
            batch = ftu.full_image_to_half_image(batch, config.image_shape)
        elif batch.shape[-1] != half_image_size:
            raise ValueError(
                f"Expected batch image size {full_image_size} (full) or {half_image_size} (half), got {batch.shape[-1]}"
            )
        batch = core.translate_images(batch, translations, config.image_shape, half_image=True)
    else:
        if batch.shape[-1] == half_image_size:
            batch = ftu.half_image_to_full_image(batch, config.image_shape)
        elif batch.shape[-1] != full_image_size:
            raise ValueError(
                f"Expected batch image size {full_image_size} (full) or {half_image_size} (half), got {batch.shape[-1]}"
            )
        batch = core.translate_images(batch, translations, config.image_shape)

    mean_half_volume = half and _mean_is_half_volume(model.mean_estimate, config.volume_shape)
    basis_half_volume = half and _basis_is_half_volume(model.basis, config.volume_shape)
    projected_mean = core_forward.forward_model(
        config, model.mean_estimate, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf, half_image=half, half_volume=mean_half_volume,
    )
    # AUs: (n_basis, n_images, n_pix[_half])
    AUs = covariance_core.batch_vol_forward_from_map(
        config, model.basis, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf, half_image=half, half_volume=basis_half_volume,
    )

    if half:
        # Apply sqrt(w) so plain half-space inner products equal full-space ones.
        batch = batch * hermitian_weights
        projected_mean = projected_mean * hermitian_weights
        AUs = AUs * hermitian_weights[None, None, :]

    if noise_variance is None:
        noise_variance = jnp.ones(batch.shape, dtype=batch.real.dtype)
    else:
        noise_variance = jnp.asarray(noise_variance)
        if noise_variance.ndim > 2:
            noise_variance = noise_variance.reshape(noise_variance.shape[0], -1)
        elif noise_variance.ndim == 1:
            noise_variance = noise_variance.reshape(1, -1)
        if noise_variance.shape[-1] != batch.shape[-1]:
            if half:
                noise_variance = ftu.full_image_to_half_image(noise_variance, config.image_shape)
            else:
                noise_variance = ftu.half_image_to_full_image(noise_variance, config.image_shape)

    AUs = AUs.transpose(1, 2, 0)  # (n_images, n_pix[_half], n_basis)

    # Noise normalization (clamp to avoid division by zero)
    safe_noise_std = jnp.sqrt(jnp.maximum(noise_variance, jnp.finfo(noise_variance.dtype).tiny))
    batch /= safe_noise_std
    projected_mean /= safe_noise_std
    AUs /= safe_noise_std[..., None]

    if config.premultiplied_ctf:
        AU_t_images = batch_x_T_y(AUs, batch)
        image_T_A_mean = batch_x_T_y(batch, projected_mean)
        # Use half-spectrum CTF when operating in half space
        CTF = config.compute_ctf_half(ctf_params) if hermitian_weights is not None else config.compute_ctf(ctf_params)
        AUs *= CTF[..., None]
        projected_mean *= CTF
    else:
        AU_t_images = batch_x_T_y(AUs, batch)
        image_T_A_mean = batch_x_T_y(batch, projected_mean)

    AU_t_Amean = batch_x_T_y(AUs, projected_mean)
    AU_t_AU = batch_x_T_y(AUs, AUs)
    A_mean_norm_sq = jnp.linalg.norm(projected_mean, axis=-1) ** 2
    image_norms_sq = jnp.linalg.norm(batch, axis=-1) ** 2

    # Cross inner products are real for Hermitian cryo-EM data.  CUDA trilinear
    # interpolation breaks exact Hermitian symmetry in float32, producing spurious
    # ~1e-1 imaginary parts even in the full-spectrum path; discard unconditionally.
    AU_t_images = AU_t_images.real
    AU_t_Amean = AU_t_Amean.real
    AU_t_AU = AU_t_AU.real
    image_T_A_mean = image_T_A_mean.real

    return AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq


@eqx.filter_jit
def compute_batch_coords(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    opts: EmbeddingOpts,
    image_mask: jax.Array,
    contrast_grid: jax.Array,
    contrast_mean: float = 1.0,
    contrast_variance: float = np.inf,
    hermitian_weights=None,
):
    """Compute latent coordinates for a batch — Equinox API.

    Replaces the 27-param ``compute_single_batch_coords_split``.

    Parameters
    ----------
    hermitian_weights : jax.Array or None
        Precomputed ``sqrt(w)`` weights for half-spectrum inner products
        (see :func:`_rfft2_hermitian_weights`).  When provided, all inner
        products are computed in the compressed half-spectrum representation,
        roughly halving memory use and compute for the inner-product step.
    """
    contrast_grid = jnp.array(contrast_grid)
    eigenvalues = model.eigenvalues

    AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq = \
        _compute_batch_coords_p1(config, batch_data, model, hermitian_weights)

    if opts.shared_label and not opts.contrast_shared_across_tilt_series:
        # Save unsummed copies for per-image contrast refinement.
        # No .copy() needed — JAX arrays are immutable; rebinding below
        # creates new arrays without mutating the originals.
        AU_t_images_unsummed = AU_t_images
        AU_t_Amean_unsummed = AU_t_Amean
        AU_t_AU_unsummed = AU_t_AU
        image_T_A_mean_unsummed = image_T_A_mean
        A_mean_norm_sq_unsummed = A_mean_norm_sq
        image_norms_sq_unsummed = image_norms_sq

    if opts.shared_label:
        AU_t_images = jnp.sum(AU_t_images, axis=0, keepdims=True)
        AU_t_Amean = jnp.sum(AU_t_Amean, axis=0, keepdims=True)
        AU_t_AU = jnp.sum(AU_t_AU, axis=0, keepdims=True)
        image_T_A_mean = jnp.sum(image_T_A_mean, axis=0, keepdims=True)
        A_mean_norm_sq = jnp.sum(A_mean_norm_sq, axis=0, keepdims=True)
        image_norms_sq = jnp.sum(image_norms_sq, axis=0, keepdims=True)

    xs_batch_contrast = batch_over_images_and_contrast_solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast_grid
    )
    residuals_fit, residuals_prior = batch_compute_contrast_residual_fast_2(
        xs_batch_contrast, AU_t_images, image_norms_sq, AU_t_Amean,
        A_mean_norm_sq, image_T_A_mean, AU_t_AU, eigenvalues, contrast_grid,
    )
    contrast_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
    res_sum1 = residuals_fit + residuals_prior + contrast_prior
    best_idx = jnp.argmin(res_sum1, axis=1).astype(int)

    xs_single = batch_slice_ar(best_idx, xs_batch_contrast)
    contrast_single = contrast_grid[best_idx]

    if opts.shared_label and not opts.contrast_shared_across_tilt_series:
        # Per-image contrast refinement via iterative optimization
        contrast_est = jnp.ones(batch_data.images.shape[0], dtype=contrast_single.dtype) * contrast_single

        def refine_contrast(i, contrast_est):
            _AU_t_images = jnp.sum(AU_t_images_unsummed * contrast_est[:, None], axis=0, keepdims=True)
            _AU_t_Amean = jnp.sum(AU_t_Amean_unsummed * contrast_est[:, None] ** 2, axis=0, keepdims=True)
            _AU_t_AU = jnp.sum(AU_t_AU_unsummed * contrast_est[:, None, None] ** 2, axis=0, keepdims=True)

            _xs = solve_contrast_linear_system(_AU_t_images, _AU_t_Amean, _AU_t_AU, eigenvalues, 1)[None]
            _xs_repeat = jnp.repeat(_xs, axis=0, repeats=batch_data.images.shape[0])
            _xs_repeat = jnp.repeat(_xs_repeat, axis=1, repeats=contrast_grid.shape[0])

            _res_fit, _res_prior = batch_compute_contrast_residual_fast_2(
                _xs_repeat, AU_t_images_unsummed, image_norms_sq_unsummed,
                AU_t_Amean_unsummed, A_mean_norm_sq_unsummed,
                image_T_A_mean_unsummed, AU_t_AU_unsummed, eigenvalues, contrast_grid,
            )
            _contrast_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
            _res_sum = _res_fit + _res_prior + _contrast_prior[None]
            _best_idx = jnp.argmin(_res_sum, axis=1).astype(int)
            return contrast_grid[_best_idx]

        contrast_single = jax.lax.fori_loop(0, 10, refine_contrast, contrast_est)

    # Covariance computation
    if opts.compute_covariances:
        if opts.shared_label and not opts.contrast_shared_across_tilt_series:
            gram = jnp.sum(AU_t_AU_unsummed * contrast_single[:, None, None] ** 2, axis=0, keepdims=True)
        else:
            gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        cov_batch = gram + jnp.diag(1 / eigenvalues)
        cov_batch = cov_batch @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ cov_batch
    else:
        cov_batch = None

    # Bias computation
    if opts.compute_bias:
        if opts.shared_label and not opts.contrast_shared_across_tilt_series:
            gram = jnp.sum(AU_t_AU_unsummed * contrast_single[:, None, None] ** 2, axis=0, keepdims=True)
        else:
            gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        _cov = gram + jnp.diag(1 / eigenvalues)
        bias = jnp.linalg.pinv(_cov, rcond=1e-6, hermitian=True) @ gram
    else:
        bias = None

    return xs_single, contrast_single, cov_batch, bias


@eqx.filter_jit
def compute_grouped_shared_batch_coords(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    image_mask: jax.Array,
    contrast_grid: jax.Array,
    contrast_mean: float,
    contrast_variance: float,
    group_ids: jax.Array,
    n_groups: int,
    compute_covariances: bool,
    compute_bias: bool,
    hermitian_weights=None,
):
    """Solve shared-label, shared-contrast batches for multiple particles at once.

    This path is used when a single iterator batch contains images from
    multiple particles. It computes the AU statistics once over all images,
    then segment-sums by particle id and solves all particles in one batched
    call. This avoids per-particle Python/JIT dispatch overhead while
    preserving shared-label semantics.
    """
    _ = image_mask  # Kept for API parity with compute_batch_coords.
    contrast_grid = jnp.asarray(contrast_grid)
    group_ids = jnp.asarray(group_ids, dtype=jnp.int32)
    eigenvalues = model.eigenvalues

    AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq = \
        _compute_batch_coords_p1(config, batch_data, model, hermitian_weights)

    AU_t_images = jax.ops.segment_sum(AU_t_images, group_ids, num_segments=n_groups)
    AU_t_Amean = jax.ops.segment_sum(AU_t_Amean, group_ids, num_segments=n_groups)
    AU_t_AU = jax.ops.segment_sum(AU_t_AU, group_ids, num_segments=n_groups)
    image_norms_sq = jax.ops.segment_sum(image_norms_sq, group_ids, num_segments=n_groups)
    image_T_A_mean = jax.ops.segment_sum(image_T_A_mean, group_ids, num_segments=n_groups)
    A_mean_norm_sq = jax.ops.segment_sum(A_mean_norm_sq, group_ids, num_segments=n_groups)

    xs_batch_contrast = batch_over_images_and_contrast_solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast_grid
    )
    residuals_fit, residuals_prior = batch_compute_contrast_residual_fast_2(
        xs_batch_contrast, AU_t_images, image_norms_sq, AU_t_Amean,
        A_mean_norm_sq, image_T_A_mean, AU_t_AU, eigenvalues, contrast_grid,
    )
    contrast_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
    res_sum = residuals_fit + residuals_prior + contrast_prior
    best_idx = jnp.argmin(res_sum, axis=1).astype(int)

    xs_single = batch_slice_ar(best_idx, xs_batch_contrast)
    contrast_single = contrast_grid[best_idx]

    if compute_covariances:
        gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        cov_batch = gram + jnp.diag(1 / eigenvalues)
        cov_batch = cov_batch @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ cov_batch
    else:
        cov_batch = None

    if compute_bias:
        gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        _cov = gram + jnp.diag(1 / eigenvalues)
        bias = jnp.linalg.pinv(_cov, rcond=1e-6, hermitian=True) @ gram
    else:
        bias = None

    return xs_single, contrast_single, cov_batch, bias


@functools.partial(
    jax.jit,
    static_argnums=[3, 4, 5, 6, 7, 8],
    static_argnames=["skip_ctf"],
)
def _legacy_forward_model_from_map(
    volume,
    ctf_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_fun,
    disc_type,
    skip_ctf=False,
):
    slices = core.slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    if not skip_ctf:
        slices = slices * ctf_fun(ctf_params, image_shape, voxel_size)
    return slices


def _legacy_batch_over_vol_forward_model_from_map(
    volumes,
    ctf_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_fun,
    disc_type,
    skip_ctf=False,
):
    return jax.vmap(
        _legacy_forward_model_from_map,
        in_axes=(0, None, None, None, None, None, None, None, None),
    )(
        volumes,
        ctf_params,
        rotation_matrices,
        image_shape,
        volume_shape,
        voxel_size,
        ctf_fun,
        disc_type,
        skip_ctf,
    )


@functools.partial(
    jax.jit,
    static_argnums=[9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 23, 24, 25, 26],
)
def _compute_single_batch_coords_split_legacy(
    batch,
    mean_estimate,
    volume_mask,
    basis,
    eigenvalues,
    ctf_params,
    rotation_matrices,
    translations,
    image_mask,
    volume_mask_threshold,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    padding,
    disc_type,
    compute_covariances,
    noise_variance,
    process_fn,
    ctf_fun,
    contrast_grid,
    contrast_mean=1.0,
    contrast_variance=np.inf,
    compute_bias=False,
    shared_label=False,
    contrast_shared_across_tilt_series=True,
    premultiplied_ctf=False,
):
    contrast_grid = jnp.asarray(contrast_grid)

    (AU_t_images, AU_t_Amean, AU_t_AU,
     image_norms_sq, image_T_A_mean, A_mean_norm_sq) = _compute_single_batch_coords_p1_legacy(
        batch, mean_estimate, volume_mask, basis, eigenvalues,
        ctf_params, rotation_matrices, translations, image_mask,
        volume_mask_threshold, image_shape, volume_shape, grid_size,
        voxel_size, padding, disc_type, noise_variance, process_fn,
        ctf_fun, premultiplied_ctf,
    )

    if shared_label and not contrast_shared_across_tilt_series:
        AU_t_images_unsummed = AU_t_images
        AU_t_Amean_unsummed = AU_t_Amean
        AU_t_AU_unsummed = AU_t_AU
        image_T_A_mean_unsummed = image_T_A_mean
        A_mean_norm_sq_unsummed = A_mean_norm_sq
        image_norms_sq_unsummed = image_norms_sq

    if shared_label:
        AU_t_images = jnp.sum(AU_t_images, axis=0, keepdims=True)
        AU_t_Amean = jnp.sum(AU_t_Amean, axis=0, keepdims=True)
        AU_t_AU = jnp.sum(AU_t_AU, axis=0, keepdims=True)
        image_T_A_mean = jnp.sum(image_T_A_mean, axis=0, keepdims=True)
        A_mean_norm_sq = jnp.sum(A_mean_norm_sq, axis=0, keepdims=True)
        image_norms_sq = jnp.sum(image_norms_sq, axis=0, keepdims=True)

    xs_batch_contrast = batch_over_images_and_contrast_solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast_grid
    )
    residuals_fit, residuals_prior = batch_compute_contrast_residual_fast_2(
        xs_batch_contrast, AU_t_images, image_norms_sq, AU_t_Amean,
        A_mean_norm_sq, image_T_A_mean, AU_t_AU, eigenvalues, contrast_grid,
    )
    contrast_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
    res_sum = residuals_fit + residuals_prior + contrast_prior
    best_idx = jnp.argmin(res_sum, axis=1).astype(int)
    xs_single = batch_slice_ar(best_idx, xs_batch_contrast)
    contrast_single = contrast_grid[best_idx]

    if shared_label and not contrast_shared_across_tilt_series:
        contrast_est = jnp.ones(batch.shape[0], dtype=contrast_single.dtype) * contrast_single

        def refine_contrast(_, current):
            _AU_t_images = jnp.sum(AU_t_images_unsummed * current[:, None], axis=0, keepdims=True)
            _AU_t_Amean = jnp.sum(AU_t_Amean_unsummed * current[:, None] ** 2, axis=0, keepdims=True)
            _AU_t_AU = jnp.sum(AU_t_AU_unsummed * current[:, None, None] ** 2, axis=0, keepdims=True)

            xs = solve_contrast_linear_system(_AU_t_images, _AU_t_Amean, _AU_t_AU, eigenvalues, 1)[None]
            xs = jnp.repeat(xs, axis=0, repeats=batch.shape[0])
            xs = jnp.repeat(xs, axis=1, repeats=contrast_grid.shape[0])

            fit, prior = batch_compute_contrast_residual_fast_2(
                xs, AU_t_images_unsummed, image_norms_sq_unsummed, AU_t_Amean_unsummed,
                A_mean_norm_sq_unsummed, image_T_A_mean_unsummed, AU_t_AU_unsummed,
                eigenvalues, contrast_grid,
            )
            c_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
            best = jnp.argmin(fit + prior + c_prior[None], axis=1).astype(int)
            return contrast_grid[best]

        contrast_single = jax.lax.fori_loop(0, 10, refine_contrast, contrast_est)

    if compute_covariances:
        if shared_label and not contrast_shared_across_tilt_series:
            gram = jnp.sum(AU_t_AU_unsummed * contrast_single[:, None, None] ** 2, axis=0, keepdims=True)
        else:
            gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        cov_batch = gram + jnp.diag(1 / eigenvalues)
        cov_batch = cov_batch @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ cov_batch
    else:
        cov_batch = None

    if compute_bias:
        if shared_label and not contrast_shared_across_tilt_series:
            gram = jnp.sum(AU_t_AU_unsummed * contrast_single[:, None, None] ** 2, axis=0, keepdims=True)
        else:
            gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
        cov = gram + jnp.diag(1 / eigenvalues)
        bias = jnp.linalg.pinv(cov, rcond=1e-6, hermitian=True) @ gram
    else:
        bias = None

    return xs_single, contrast_single, cov_batch, bias


def _compute_single_batch_coords_p1_legacy(
    batch,
    mean_estimate,
    volume_mask,
    basis,
    eigenvalues,
    ctf_params,
    rotation_matrices,
    translations,
    image_mask,
    volume_mask_threshold,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    padding,
    disc_type,
    noise_variance,
    process_fn,
    ctf_fun,
    premultiplied_ctf,
):
    _ = volume_mask
    _ = eigenvalues
    _ = image_mask
    _ = volume_mask_threshold
    _ = grid_size
    _ = padding

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations, image_shape)

    projected_mean = _legacy_forward_model_from_map(
        mean_estimate, ctf_params, rotation_matrices, image_shape, volume_shape,
        voxel_size, ctf_fun, disc_type, skip_ctf=premultiplied_ctf,
    )
    AUs = _legacy_batch_over_vol_forward_model_from_map(
        basis, ctf_params, rotation_matrices, image_shape, volume_shape,
        voxel_size, ctf_fun, disc_type, skip_ctf=premultiplied_ctf,
    )
    AUs = AUs.transpose(1, 2, 0)

    batch = batch / jnp.sqrt(noise_variance)
    projected_mean = projected_mean / jnp.sqrt(noise_variance)
    AUs = AUs / jnp.sqrt(noise_variance)[..., None]

    if premultiplied_ctf:
        AU_t_images = batch_x_T_y(AUs, batch)
        image_T_A_mean = batch_x_T_y(batch, projected_mean)
        ctf = ctf_fun(ctf_params, image_shape, voxel_size)
        AUs = AUs * ctf[..., None]
        projected_mean = projected_mean * ctf
    else:
        AU_t_images = batch_x_T_y(AUs, batch)
        image_T_A_mean = batch_x_T_y(batch, projected_mean)

    AU_t_Amean = batch_x_T_y(AUs, projected_mean)
    AU_t_AU = batch_x_T_y(AUs, AUs)
    A_mean_norm_sq = jnp.linalg.norm(projected_mean, axis=-1) ** 2
    image_norms_sq = jnp.linalg.norm(batch, axis=-1) ** 2

    AU_t_images = AU_t_images.real
    AU_t_Amean = AU_t_Amean.real
    AU_t_AU = AU_t_AU.real
    image_T_A_mean = image_T_A_mean.real

    return AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq

# Naive functions, without precompute
def compute_contrast_residual_naive(image, AU, projected_mean, xs, eigenvalues, noise_variance, contrast_grid):
    fit_residual =  jnp.linalg.norm( (contrast_grid * (AU @ xs.T + projected_mean[...,None]) - image[...,None])) #/ jnp.sqrt(noise_variance), axis =0)**2 what is this???
    prior_residual = batch_x_T_y( xs, xs / eigenvalues) #jnp.conj(xs).T @ ( xs /  eigenvalues )
    return fit_residual,  prior_residual.real
batch_compute_contrast_residual_naive = jax.vmap(compute_contrast_residual_naive, in_axes = (0,0,0,0, None, None, None) )


@jax.jit
def compute_contrast_residual_fast_2(xs, AU_t_images, image_norms_sq, AU_t_Amean, Amean_norms_sq, image_T_A_mean ,  AU_t_AU, eigenvalues, contrast):
    
    # x^T (AU)^T AU x
    # Square terms
    p1 = contrast**2 * batch_x_T_y(xs, (AU_t_AU @ xs.T).T ).real

    p2 = contrast **2 * Amean_norms_sq
    
    ## Now passed as argument
    p3 = image_norms_sq
    
    # Cross terms
    p4 = 2 * contrast**2 * (jnp.conj(AU_t_Amean).T @ xs.T).real
    # 
    p5 = - 2 * contrast * (jnp.conj(AU_t_images).T @ xs.T).real
    
    p6 = - 2 * contrast * (image_T_A_mean).real

    return ((p1 + p2 + p3 + p4 + p5 + p6) ).real, batch_x_T_y( xs, xs / eigenvalues).real

batch_compute_contrast_residual_fast_2 = jax.vmap(compute_contrast_residual_fast_2, in_axes = (0,0,0,0,0,0,0,None, None))


@jax.jit
def solve_contrast_linear_system(AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast):
    A = (contrast **2) * AU_t_AU  +  jnp.diag(1 / eigenvalues )
    b = contrast * ( AU_t_images - contrast * AU_t_Amean ) 
    sol = linalg.batch_hermitian_linear_solver(A,b)
    return sol

batch_over_contrast_solve_contrast_linear_system = jax.vmap(solve_contrast_linear_system, in_axes = ( None, None, None, None, 0) )
batch_over_images_and_contrast_solve_contrast_linear_system = jax.vmap(batch_over_contrast_solve_contrast_linear_system, in_axes = ( 0, 0,0,None, None) )


# ============================================================================
# Multi-zdim embedding: one data pass for all n_pcs values
# ============================================================================

@eqx.filter_jit
def _collect_batch_stats(config, batch_data, model, hermitian_weights):
    """Forward-model pass only — no solving.

    Returns the six inner-product statistics consumed by :func:`_solve_batch_from_stats`.
    Reuses :func:`_compute_batch_coords_p1` under the equinox JIT boundary.
    """
    return _compute_batch_coords_p1(config, batch_data, model, hermitian_weights)


@jax.jit
def _solve_batch_from_stats(
    AU_t_images, AU_t_Amean, AU_t_AU,
    image_norms_sq, image_T_A_mean, A_mean_norm_sq,
    eigenvalues, contrast_grid, contrast_mean, contrast_variance,
):
    """Solve for xs, best contrast, and posterior covariance from precomputed statistics.

    Parameters
    ----------
    AU_t_images : (n_images, n_pcs)
    AU_t_Amean  : (n_images, n_pcs)
    AU_t_AU     : (n_images, n_pcs, n_pcs)
    image_norms_sq, image_T_A_mean, A_mean_norm_sq : (n_images,)
    eigenvalues : (n_pcs,)  — jax array
    contrast_grid : (n_contrasts,)  — jax array
    contrast_mean, contrast_variance : scalar jax arrays

    Returns
    -------
    xs_single      : (n_images, n_pcs)
    contrast_single: (n_images,)
    cov_batch      : (n_images, n_pcs, n_pcs)
    """
    xs_batch_contrast = batch_over_images_and_contrast_solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast_grid)
    residuals_fit, residuals_prior = batch_compute_contrast_residual_fast_2(
        xs_batch_contrast, AU_t_images, image_norms_sq, AU_t_Amean,
        A_mean_norm_sq, image_T_A_mean, AU_t_AU, eigenvalues, contrast_grid,
    )
    contrast_prior = (contrast_grid - contrast_mean) ** 2 / contrast_variance
    res_sum1 = residuals_fit + residuals_prior + contrast_prior
    best_idx = jnp.argmin(res_sum1, axis=1).astype(int)
    xs_single = batch_slice_ar(best_idx, xs_batch_contrast)
    contrast_single = contrast_grid[best_idx]
    gram = (contrast_single ** 2)[:, None, None] * AU_t_AU
    cov_batch = gram + jnp.diag(1 / eigenvalues)
    cov_batch = cov_batch @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ cov_batch
    return xs_single, contrast_single, cov_batch


def get_per_image_embedding_multi_zdim(
    mean, u, s, n_pcs_list, cryos, volume_mask, gpu_memory,
    disc_type='linear_interp', contrast_grid=None, contrast_option='none',
    ignore_zero_frequency=False, contrast_mean=1, contrast_variance=np.inf,
):
    """Compute per-image embeddings for multiple n_pcs values in a single data pass.

    Replaces N separate calls to :func:`get_per_image_embedding` with a single
    forward-model pass (at ``max(n_pcs_list)`` basis vectors) that accumulates
    sufficient statistics, then solves lightweight linear systems for each
    ``(n_pcs, regularized/unregularized)`` pair.  Typically 5–10× faster than
    calling :func:`get_per_image_embedding` independently for each zdim.

    Not supported for tilt-series data.

    Args:
        mean: Mean volume in Fourier space, shape ``(volume_size,)``.
        u: Eigenvectors, shape ``(volume_size, n_max_components)``.
        s: Eigenvalues, shape ``(n_max_components,)``.
        n_pcs_list: List of int — n_pcs values to compute (need not be sorted).
        cryos: Half-set datasets.
        volume_mask: Binary mask selecting valid voxels.
        gpu_memory: Available GPU memory in GB.
        disc_type: Discretization type (overridden to ``'cubic'`` when ``USE_CUBIC``).
        contrast_grid: Contrast values to search (default: 50-point grid).
        contrast_option: Contrast mode (``'none'``, ``'contrast'``, …).
        ignore_zero_frequency: Exclude DC component.
        contrast_mean: Prior mean for contrast estimation.
        contrast_variance: Prior variance for contrast estimation.

    Returns:
        ``(zs_reg, zs_noreg)`` — two dicts keyed by *n_pcs* (int).  Each value
        is a tuple ``(xs, cov_zs, contrasts)`` where *xs* has shape
        ``(n_total_images, n_pcs)``, *cov_zs* has shape
        ``(n_total_images, n_pcs, n_pcs)``, and *contrasts* has shape
        ``(n_total_images,)``.
    """
    if cryos.tilt_series_flag:
        raise ValueError(
            "get_per_image_embedding_multi_zdim does not support tilt-series data. "
            "Use get_per_image_embedding instead."
        )

    n_pcs_list = sorted(set(n_pcs_list))
    max_n_pcs = n_pcs_list[-1]

    if ignore_zero_frequency:
        volume_mask = np.ones_like(volume_mask)

    use_contrast = 'contrast' in contrast_option
    if use_contrast:
        cg = np.linspace(0, 2, 51)[1:].astype(np.float32) if contrast_grid is None else contrast_grid
    else:
        cg = np.ones([1], dtype=np.float32)
    cg_jax = jnp.array(cg)
    contrast_mean_jax     = jnp.array(contrast_mean,     dtype=jnp.float32)
    contrast_variance_jax = jnp.array(contrast_variance, dtype=jnp.float32)

    # Cubic spline precompute — ONCE for max_n_pcs, not once per zdim.
    actual_disc_type = disc_type
    full_basis = np.asarray(u[:, :max_n_pcs]).T  # (max_n_pcs, volume_size)
    mean_out = mean
    if USE_CUBIC:
        actual_disc_type = 'cubic'
        from recovar.core import cubic_interpolation
        from recovar.heterogeneity import covariance_estimation as _cov_est
        mean_out = cubic_interpolation.calculate_spline_coefficients(mean.reshape(cryos.volume_shape))
        logger.info("Computing spline coefficients for %d basis vectors (multi-zdim)", max_n_pcs)
        full_basis = _cov_est.compute_spline_coeffs_in_batch(full_basis, cryos.volume_shape, gpu_memory=None)

    dtype = full_basis.dtype

    # Precompute eigenvalue arrays for each n_pcs — reg uses prior, noreg uses inf.
    s_reg   = {k: jnp.array((s[:k] + jax_config.ROOT_EPSILON).astype(dtype)) for k in n_pcs_list}
    s_noreg = {k: jnp.full(k, jnp.inf, dtype=dtype) for k in n_pcs_list}

    # Allocate per-halfset output arrays, concatenated at the end.
    def _alloc(n_pcs):
        return {
            'zs':        [np.zeros((cryo.n_images, n_pcs),        dtype=dtype)    for cryo in cryos],
            'cov_zs':    [np.zeros((cryo.n_images, n_pcs, n_pcs), dtype=dtype)    for cryo in cryos],
            'contrasts': [np.zeros(cryo.n_images,                  dtype=np.float32) for cryo in cryos],
        }

    zs_reg   = {k: _alloc(k) for k in n_pcs_list}
    zs_noreg = {k: _alloc(k) for k in n_pcs_list}

    for cryo_idx, cryo in enumerate(cryos):
        config = ForwardModelConfig.from_dataset(
            cryo, disc_type=actual_disc_type,
            process_fn=cryo.image_stack.process_images,
        )
        # eigenvalues field not used by _collect_batch_stats; set placeholder.
        model = ModelState(
            mean_estimate=jnp.asarray(mean_out, dtype=cryo.dtype),
            volume_mask=jnp.asarray(volume_mask, dtype=cryo.dtype_real),
            basis=jnp.asarray(full_basis, dtype=cryo.dtype),
            eigenvalues=jnp.ones(max_n_pcs, dtype=cryo.dtype),
        )
        mean_hv, basis_hv = _prepare_model_half_volumes(
            config, model.mean_estimate, model.basis,
        )
        model = ModelState(
            mean_estimate=mean_hv,
            volume_mask=model.volume_mask,
            basis=basis_hv,
            eigenvalues=model.eigenvalues,
        )

        batch_size = utils.get_embedding_batch_size(full_basis, cryo.image_size, cg, max_n_pcs, gpu_memory)
        _EMBEDDING_BATCH_SAFETY_FACTOR = 10
        batch_size = utils.safe_batch_size(batch_size // _EMBEDDING_BATCH_SAFETY_FACTOR)
        logger.info("multi-zdim embedding batch size (halfset %d): %d", cryo_idx, batch_size)

        hermitian_weights = _embedding_hermitian_weights(config)
        prefer_half_noise = hermitian_weights is not None
        noise_model = cryo.noise
        data_generator = cryo.get_dataset_generator(batch_size=batch_size)

        for batch, particles_ind, batch_image_ind in data_generator:
            batch_data = BatchData(
                images=batch,
                ctf_params=cryo.CTF_params[batch_image_ind],
                rotation_matrices=cryo.rotation_matrices[batch_image_ind],
                translations=cryo.translations[batch_image_ind],
                noise_variance=_noise_get_half_or_full(noise_model, batch_image_ind, prefer_half=prefer_half_noise),
            )

            # ── Single forward-model pass at max_n_pcs ──────────────────
            AU_t_im, AU_t_Am, AU_t_AU, im_norms_sq, im_T_Am, Am_norm_sq = \
                _collect_batch_stats(config, batch_data, model, hermitian_weights)

            # ── Solve for each n_pcs (reg and noreg) ────────────────────
            for n_pcs in n_pcs_list:
                sub_AI  = AU_t_im[:, :n_pcs]
                sub_AM  = AU_t_Am[:, :n_pcs]
                sub_AAU = AU_t_AU[:, :n_pcs, :n_pcs]

                xs_r, c_r, cov_r = _solve_batch_from_stats(
                    sub_AI, sub_AM, sub_AAU,
                    im_norms_sq, im_T_Am, Am_norm_sq,
                    s_reg[n_pcs], cg_jax, contrast_mean_jax, contrast_variance_jax,
                )
                zs_reg[n_pcs]['zs']      [cryo_idx][particles_ind] = np.asarray(xs_r).real
                zs_reg[n_pcs]['cov_zs']  [cryo_idx][particles_ind] = np.asarray(cov_r).real
                zs_reg[n_pcs]['contrasts'][cryo_idx][batch_image_ind] = np.asarray(c_r)

                xs_n, c_n, cov_n = _solve_batch_from_stats(
                    sub_AI, sub_AM, sub_AAU,
                    im_norms_sq, im_T_Am, Am_norm_sq,
                    s_noreg[n_pcs], cg_jax, contrast_mean_jax, contrast_variance_jax,
                )
                zs_noreg[n_pcs]['zs']      [cryo_idx][particles_ind] = np.asarray(xs_n).real
                zs_noreg[n_pcs]['cov_zs']  [cryo_idx][particles_ind] = np.asarray(cov_n).real
                zs_noreg[n_pcs]['contrasts'][cryo_idx][batch_image_ind] = np.asarray(c_n)

    # Concatenate across halfsets; return flat (xs, cov_zs, contrasts) tuples per n_pcs.
    result_reg   = {}
    result_noreg = {}
    for n_pcs in n_pcs_list:
        result_reg[n_pcs] = (
            np.concatenate(zs_reg[n_pcs]['zs'],        axis=0),
            np.concatenate(zs_reg[n_pcs]['cov_zs'],    axis=0),
            np.concatenate(zs_reg[n_pcs]['contrasts'],  axis=0),
        )
        result_noreg[n_pcs] = (
            np.concatenate(zs_noreg[n_pcs]['zs'],       axis=0),
            np.concatenate(zs_noreg[n_pcs]['cov_zs'],   axis=0),
            np.concatenate(zs_noreg[n_pcs]['contrasts'], axis=0),
        )

    logger.info("multi-zdim embedding complete for n_pcs=%s", n_pcs_list)
    return result_reg, result_noreg


def set_contrasts_in_cryos(cryos, contrasts):

    if cryos.tilt_series_flag:

        # If it's a per image assignment
        if contrasts.shape[0] == cryos.n_total_images:
            running_idx = 0 
            for i in range(2): # Untested
                cryos[i].CTF_params[:,core.CTFParamIndex.CONTRAST] *= contrasts[running_idx:running_idx+cryos[i].n_images]
                running_idx += cryos[i].n_images
        else:
            # If it's a per tilt series assignment
            running_idx = 0 
            for i in range(2):
                for p in cryos[i].image_stack.particles:
                    cryos[i].CTF_params[p,core.CTFParamIndex.CONTRAST] *= contrasts[running_idx]
                    running_idx+=1
    else:
        running_idx = 0 
        for i in range(2): 
            cryos[i].CTF_params[:,core.CTFParamIndex.CONTRAST] *= contrasts[running_idx:running_idx+cryos[i].n_images]
            running_idx += cryos[i].n_images
