"""Per-image latent coordinate estimation via linear projection."""

import logging
import time

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
    hermitian_weights = _rfft2_hermitian_weights(config.image_shape)

    noise_model = experiment_dataset.noise

    for batch, particles_ind, batch_image_ind in data_generator:

        # Handle tilt series image subsetting
        if image_subset_in_tilt_series is not None:
            subset = image_subset_in_tilt_series
            if np.max(image_subset_in_tilt_series) >= batch.shape[0]:
                subset = image_subset_in_tilt_series[image_subset_in_tilt_series < batch.shape[0]]
            batch = jnp.array(batch[subset])
            batch_image_ind = batch_image_ind[subset]

        batch_data = BatchData(
            images=batch,
            ctf_params=experiment_dataset.CTF_params[batch_image_ind],
            rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
            translations=experiment_dataset.translations[batch_image_ind],
            noise_variance=noise_model.get(batch_image_ind),
        )

        xs_single, contrast_single, cov_batch, bias = compute_batch_coords(
            config, batch_data, model, opts,
            experiment_dataset.image_stack.mask, contrast_grid,
            contrast_mean, contrast_variance,
            hermitian_weights,
        )

        if force_not_shared_label:
            particles_ind = batch_image_ind

        xs[particles_ind] = xs_single

        if not contrast_shared_across_tilt_series:
            estimated_contrasts[batch_image_ind] = contrast_single
        else:
            estimated_contrasts[particles_ind] = contrast_single

        if compute_covariances:
            image_latent_precisions[np.array(particles_ind)] = cov_batch

        if compute_bias:
            image_latent_bias[np.array(particles_ind)] = bias

    return xs, image_latent_precisions, estimated_contrasts, image_latent_bias


def slice_ar(indx, arr):
    return arr[indx]
# Vectorized index selection via vmap
batch_slice_ar = jax.jit(jax.vmap(slice_ar, in_axes =(0, 0)))
batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))


# ============================================================================
# New Equinox-based embedding API
# ============================================================================


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
    batch = core.translate_images(batch, translations, config.image_shape)

    # --- Project volumes to image space ---
    # Multiply projected arrays by sqrt(w) so that the plain half-spectrum inner
    # product equals the correct Hermitian-weighted full-spectrum inner product:
    #   <A_w, B_w>_half = sum_{k in half} w[k]*conj(A[k])*B[k] = <A, B>_full
    half = hermitian_weights is not None
    projected_mean = core_forward.forward_model(
        config, model.mean_estimate, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf, half_image=half,
    )
    # AUs: (n_basis, n_images, n_pix[_half])
    AUs = covariance_core.batch_vol_forward_from_map(
        config, model.basis, ctf_params, rotation_matrices,
        skip_ctf=config.premultiplied_ctf, half_image=half,
    )

    if half:
        # Observed images arrive full-spectrum; convert to half and weight.
        batch = ftu.full_image_to_half_image(batch, config.image_shape) * hermitian_weights
        projected_mean = projected_mean * hermitian_weights
        AUs = AUs * hermitian_weights[None, None, :]
        noise_variance = ftu.full_image_to_half_image(noise_variance, config.image_shape)

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
