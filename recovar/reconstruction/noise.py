"""Noise variance estimation from cryo-EM image residuals."""

import functools
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import nvtx

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core, utils, jax_config
from recovar.core.configs import ForwardModelConfig, BatchData, ModelState
from recovar.heterogeneity import covariance_core
from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)

# NVTX domain for noise operations
NVTX_DOMAIN_NOISE = "noise"
## There is currently two ways to estimate noise:
# White and radial
# From my observations, white seems fine for most datasets but some need some other noise distribution
# Neither solution implemented here are very satisfying. Guessing noise in presence of heterogeneity is not trivial, since the residual doesn't seem like the correct way to do it.
# It makes me think we should have "noise pickers".

# 

class RadialNoiseModel():
    def __init__(self, noise_variance_radial, image_shape = None):
        self.noise_variance_radial = noise_variance_radial
        self.image_shape = image_shape if image_shape is not None else (2 * (len(noise_variance_radial)+1) , 2 * (len(noise_variance_radial)+1) )
    def get(self, *args, **kwargs):
        return make_radial_noise(self.noise_variance_radial, self.image_shape).reshape(1, -1)
    
    def set_variance(self, noise_variance_radial):
        self.noise_variance_radial = noise_variance_radial

    def get_average_radial_noise(self, *args, **kwargs):
        return self.noise_variance_radial


class VariableRadialNoiseModel():
    def __init__(self, noise_variance_radials, dose_indices, image_shape = None):
        self.noise_variance_radials = noise_variance_radials
        self.dose_indices = dose_indices
        self.image_shape = image_shape if image_shape is not None else (2 * (len(noise_variance_radials[0])+1) , 2 * (len(noise_variance_radials[0])+1) )

    def get(self, indices, *args, **kwargs):
        dose_indices_batch = self.dose_indices[indices]
        return batch_make_radial_noise(self.noise_variance_radials[dose_indices_batch,:], self.image_shape)
    
    def get_average_radial_noise(self, *args, **kwargs):
        counts = jnp.bincount(self.dose_indices) / self.dose_indices.size      
        return jnp.sum(self.noise_variance_radials * counts[:,None], axis=0)
    
    def set_variance(self, noise_variance_radials):
        self.noise_variance_radials = noise_variance_radials


def to_batched_pixel_noise(noise_variances, image_shape, batch_size=None):
    """Normalize noise variance into shape (B, D*D).

    Accepts common forms used in the codebase:
    - (D, D)
    - (1, D, D) or (B, D, D)
    - (D*D,)
    - (1, D*D) or (B, D*D)
    """
    arr = jnp.asarray(noise_variances)

    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2 and tuple(arr.shape) == tuple(image_shape):
        arr = arr.reshape(1, -1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if batch_size is not None and arr.ndim == 2 and arr.shape[0] == 1 and batch_size > 1:
        arr = jnp.repeat(arr, batch_size, axis=0)
    return arr

@functools.partial(jax.jit, static_argnums=(3,4,6))
def get_image_masks(volume_mask, rotation_matrices, volume_mask_threshold, volume_shape, image_shape, image_mask, invert_mask):
    if volume_mask is not None:
        image_masks = covariance_core.get_per_image_tight_mask(volume_mask, 
                                            rotation_matrices,
                                            image_mask, 
                                            volume_mask_threshold,
                                            image_shape, 
                                            volume_shape, 
                                            image_shape[0], 
                                            0, 
                                            'linear_interp',
                                              soften =5)
    elif image_mask is not None:
        image_masks = image_mask
    else:
        image_masks = jnp.ones(image_shape, dtype = np.float32)

    if invert_mask:
        image_masks = 1 - image_masks
    return image_masks


# From a given noise variance model, predict the observed noise variance in a possibly CTFed + masked image
def predict_noise_variance(noise_variance, CTF_params, voxel_size, CTF_fun, image_masks, image_shape, radial=True, premultiplied_ctf=False, upsample_factor=1):
    """Predict noise variance in images, optionally handling upsampling.
    
    Args:
        noise_variance: Base noise variance (radial or scalar)
        CTF_params: CTF parameters
        voxel_size: Voxel size
        CTF_fun: Function to compute CTF
        image_masks: Image masks
        image_shape: Image shape
        radial: Whether noise is radial
        premultiplied_ctf: Whether CTF is premultiplied
        upsample_factor: Factor to upsample by (default 1 for no upsampling)
    """
    if upsample_factor > 1:
        # Interpolate noise_variance onto a finer grid using JAX operations
        n_orig = len(noise_variance)
        n_new = n_orig * upsample_factor
        new_indices = jnp.linspace(0, n_orig - 1, n_new)
        
        # Get indices for interpolation
        idx_lo = jnp.floor(new_indices).astype(int)
        idx_hi = jnp.minimum(idx_lo + 1, n_orig - 1)
        alpha = new_indices - idx_lo
        
        # Linear interpolation
        noise_variance = (1 - alpha) * noise_variance[idx_lo] + alpha * noise_variance[idx_hi]
        
        # Scale noise by upsample factor squared to maintain variance
        noise_variance = noise_variance * upsample_factor**2
        upsampled_shape = tuple(np.array(image_shape) * upsample_factor)
        noise_variance = make_radial_noise(noise_variance, upsampled_shape)

        # Apply CTF on upsampled grid if needed
        if premultiplied_ctf:
            upsampled_CTF = CTF_fun(CTF_params, upsampled_shape, voxel_size)
            noise_variance = noise_variance * upsampled_CTF**2
    else:
        noise_variance = make_radial_noise(noise_variance, image_shape)
        if premultiplied_ctf:
            CTF = CTF_fun(CTF_params, image_shape, voxel_size)
            noise_variance = noise_variance * CTF**2

    predicted_noise_variance = get_masked_noise_variance_from_noise_variance(image_masks, noise_variance, image_shape)

    return predicted_noise_variance


def noise_variance_loss(images, noise_variance, translations, CTF_params, voxel_size, CTF_fun, image_masks, image_shape, radial , premultiplied_ctf):
    # Compute the predicted noise variance

    images = core.translate_images(images, translations , image_shape)

    masked_images = covariance_core.apply_image_masks(images, image_masks, image_shape).reshape(-1, *image_shape)
    predicted_noise_variance = predict_noise_variance(noise_variance, CTF_params, voxel_size, CTF_fun, image_masks, image_shape, radial, premultiplied_ctf)

    loss = jnp.sum( jnp.abs( jnp.abs(masked_images)**2 - predicted_noise_variance )**2  ) / np.prod(image_shape)

    return loss


@nvtx.annotate("fit_noise_model_to_images", color="red", domain=NVTX_DOMAIN_NOISE)
def fit_noise_model_to_images(experiment_dataset, volume_mask, mean_estimate, image_subset, batch_size, invert_mask, disc_type='linear_interp', use_batch_solver=True, tilt_dose_inner=False, image_n_iter=1e4):
    """Fit noise model to images, handling tilt series data specially.
    
    Args:
        experiment_dataset: Dataset containing images and metadata
        volume_mask: Mask for the volume
        mean_estimate: Estimate of mean volume
        image_subset: Subset of images to use, or None for all
        batch_size: Batch size for processing
        invert_mask: Whether to invert the mask
        disc_type: Type of discretization to use
        use_batch_solver: Whether to use batch solver vs full dataset
        tilt_dose_inner: Whether this is an inner call for tilt series
        
    Returns:
        For tilt series: Array of noise variances per tilt
        Otherwise: Single noise variance and initial estimate
    """
    # Import optimization libraries
    from jaxopt import ScipyBoundedMinimize, OptaxSolver
    import optax

    # Special handling for tilt series data
    if isinstance(experiment_dataset.noise, VariableRadialNoiseModel) and not tilt_dose_inner:
        logger.info("Fitting noise model for each tilt")
        # Initialize array to store noise variances for each tilt
        noise_variance_radials = []
        initial_noise_variance_radials = []

        # Get max tilt index
        max_noise_index = jnp.max(experiment_dataset.noise.dose_indices) + 1
        
        # Fit noise model separately for each tilt
        for tilt_idx in range(max_noise_index):
            # Get images for this tilt
            tilt_mask = experiment_dataset.noise.dose_indices == tilt_idx
            tilt_images = np.where(tilt_mask)[0]
            
            # Intersect with provided image subset if any
            images_to_use = (tilt_images if image_subset is None 
                           else np.intersect1d(image_subset, tilt_images))
            images_to_use = np.array(images_to_use)

            # Fit noise model for this tilt
            noise_variance, initial_noise_variance_radial = fit_noise_model_to_images(
                experiment_dataset,
                volume_mask, 
                mean_estimate,
                images_to_use,
                batch_size,
                invert_mask,
                disc_type,
                use_batch_solver,
                tilt_dose_inner=True
            )
            noise_variance_radials.append(noise_variance)
            initial_noise_variance_radials.append(initial_noise_variance_radial)
            logger.info("Done fitting noise model for tilt %d", tilt_idx)
        noise_variance_radials = jnp.stack(noise_variance_radials)
        initial_noise_variance_radials = jnp.stack(initial_noise_variance_radials)
        return noise_variance_radials, initial_noise_variance_radials
    
    if image_subset is not None:
        batch_size = int(np.min([batch_size, image_subset.size]))

    # ---------- initial guess (keep the real estimate, don't overwrite) ----------
    initial_noise_variance = estimate_noise_level_no_masks(
        experiment_dataset,
        image_subset,
        mean_estimate,
        batch_size,
        disc_type="linear_interp",
    )


    def infinite_data_iterator():
            while True:                              # repeat forever (or add a break after N epochs)

                for batch, _, batch_ind in \
                    experiment_dataset.get_image_subset_generator(
                        batch_size=batch_size,
                        subset_indices=image_subset):
                    yield (
                        batch,
                        experiment_dataset.rotation_matrices[batch_ind],
                        experiment_dataset.translations[batch_ind],
                        experiment_dataset.CTF_params[batch_ind],
                    )


    # ---------- choose optimiser ----------

    if use_batch_solver:
        # ---------- mini-batch loss (Optax expects signature: (params, data)) ----------
        #@jax.jit  # compiles once, regardless of batch size
        logger.info("Fitting noise model to images using batch solver")
        @jax.jit  # compiles once, regardless of batch size
        def loss_fn_batch(noise_variance, data):
            batch, rot, trans, ctf_p = data

            # preprocess images
            batch = experiment_dataset.image_stack.process_images(batch)

            img_masks = get_image_masks(
                volume_mask,
                rot,
                experiment_dataset.volume_mask_threshold,
                experiment_dataset.volume_shape,
                experiment_dataset.image_shape,
                experiment_dataset.image_stack.mask,
                invert_mask,
            )

            # compute per-image loss so gradients have a stable scale
            loss = noise_variance_loss(
                batch,
                noise_variance,
                trans,
                ctf_p,
                experiment_dataset.voxel_size,
                experiment_dataset.CTF_fun,
                img_masks,
                experiment_dataset.image_shape,
                True,
                experiment_dataset.premultiplied_ctf,
            )
            return loss / batch.shape[0]          # divide by mini-batch size
        
        opt = optax.adam(1e-3)

        maxiter = int(image_n_iter / batch_size)

        solver = OptaxSolver(opt=opt,
                            fun=loss_fn_batch,
                            maxiter=maxiter)

        # run the streaming optimiser; run_iterator returns (params, state)
        optimized_noise_variance, state = solver.run_iterator(
            init_params=initial_noise_variance,
            iterator=infinite_data_iterator(),           # ← infinite iterator
        )

        for k in [15, 10, 5]:
            def loss_function_partial(noise_subset, data, k=k):
                noise_v_updated = optimized_noise_variance.at[:k].set(noise_subset)
                return loss_fn_batch(noise_v_updated, data)

            lower_bounds = jnp.zeros(k)
            upper_bounds = jnp.ones(k) * jnp.inf
            bounds = (lower_bounds, upper_bounds)

            solver = OptaxSolver(opt=opt,
                                fun=loss_function_partial,
                                maxiter=maxiter, tol = 1e-5)

            # run the streaming optimiser; run_iterator returns (params, state)
            optimized_noise_variance_k, state = solver.run_iterator(
                init_params=optimized_noise_variance[:k],
                iterator=infinite_data_iterator()
            )
            optimized_noise_variance = optimized_noise_variance.at[:k].set(optimized_noise_variance_k)

            # optimized_noise_variance, state = solver.run_iterator(
            logger.info("Optimizer finished. Final state: %s", state)

    else:
        def loss_function(noise_variance_to_opt, only_up_to_k= None, noise_variance = None):
            if only_up_to_k is not None:
                noise_variance = noise_variance.at[:only_up_to_k].set(noise_variance_to_opt)
            else:
                noise_variance = noise_variance_to_opt
            data_generator = experiment_dataset.get_image_subset_generator(
                batch_size=batch_size,
                subset_indices=image_subset
            )
            total_loss = 0.0
            total_grad = 0.0
            n_images = (image_subset.size if image_subset is not None else experiment_dataset.n_images)
            for batch, _, batch_ind in data_generator:
                batch = experiment_dataset.image_stack.process_images(batch)
                image_masks = get_image_masks(
                    volume_mask,
                    experiment_dataset.rotation_matrices[batch_ind],
                    experiment_dataset.volume_mask_threshold,
                    experiment_dataset.volume_shape,
                    experiment_dataset.image_shape,
                    None,
                    invert_mask
                )
                loss_val, grad_val = jax.value_and_grad(
                    noise_variance_loss, argnums=1
                )(
                    batch,
                    noise_variance,
                    experiment_dataset.translations[batch_ind],
                    experiment_dataset.CTF_params[batch_ind],
                    experiment_dataset.voxel_size,
                    experiment_dataset.CTF_fun,
                    image_masks,
                    experiment_dataset.image_shape,
                    True,
                    experiment_dataset.premultiplied_ctf
                )
                total_loss += loss_val / n_images
                total_grad += grad_val / n_images

            return total_loss, total_grad


        lbfgsb = ScipyBoundedMinimize(
            fun=loss_function,
            method="L-BFGS-B",
            maxiter = 50,
            value_and_grad=True,
            jit=False,
        )

        lower_bounds = jnp.zeros_like(initial_noise_variance)

        upper_bounds = jnp.ones_like(initial_noise_variance) * jnp.inf
        bounds = (lower_bounds, upper_bounds)

        lbfgsb_sol = lbfgsb.run(init_params=initial_noise_variance, bounds=bounds)
        optimized_noise_variance = lbfgsb_sol.params


        for k in [15, 10, 5]:
            def loss_function_partial(noise_subset, k=k):
                noise_v_updated = optimized_noise_variance.at[:k].set(noise_subset)
                return loss_function(noise_v_updated)

            lower_bounds = jnp.zeros(k)
            upper_bounds = jnp.ones(k) * jnp.inf
            bounds = (lower_bounds, upper_bounds)

            lbfgsb_solver = ScipyBoundedMinimize(
                fun=loss_function_partial,
                method="L-BFGS-B",
                maxiter=50,
                value_and_grad=True,
                jit=False,
            )

            lbfgsb_sol = lbfgsb_solver.run(init_params=optimized_noise_variance[:k], bounds=bounds)
            optimized_noise_variance = optimized_noise_variance.at[:k].set(lbfgsb_sol.params)
            logger.info("Noise optimization complete")


    return optimized_noise_variance, initial_noise_variance
    

def update_noise_variance(noise_variance, cryos):
    for cryo in cryos:
        if isinstance(cryo.noise, RadialNoiseModel):
            cryo.set_radial_noise_model(noise_variance)
        elif isinstance(cryo.noise, VariableRadialNoiseModel):
            cryo.set_variable_radial_noise_model(noise_variance)


def upper_bound_noise_by_signal_p_noise_dispatched(noise_var_used, cryos, means, batch_size, dilated_volume_mask):

    if isinstance(cryos[0].noise, VariableRadialNoiseModel):
        # Get max tilt index
        experiment_dataset = cryos[0]
        max_noise_index = jnp.max(experiment_dataset.noise.dose_indices) + 1
        ub_noise_var_by_var_ests = []
        # Fit noise model separately for each tilt
        for tilt_idx in range(max_noise_index):
            # noise_var_used may be 2D (per-tilt) or 1D (single radial profile)
            noise_for_tilt = noise_var_used[tilt_idx] if np.ndim(noise_var_used) >= 2 else noise_var_used
            variance, ub_noise_var_by_var_est = upper_bound_noise_by_signal_p_noise(noise_for_tilt, cryos, means, batch_size, dilated_volume_mask, noise_ind_subset = tilt_idx)
            ub_noise_var_by_var_ests.append(ub_noise_var_by_var_est)
        return variance, np.stack(ub_noise_var_by_var_ests)
    else:
        return upper_bound_noise_by_signal_p_noise(noise_var_used, cryos, means, batch_size, dilated_volume_mask, noise_ind_subset = None)


@nvtx.annotate("upper_bound_noise_by_signal_p_noise", color="green", domain=NVTX_DOMAIN_NOISE)
def upper_bound_noise_by_signal_p_noise(noise_var_used, cryos, means, batch_size, dilated_volume_mask, noise_ind_subset = None):
        # Now, estimate the variance of the signal. If the variance estimate ends up negative, we have overestimated the noise variance.
        for noise_repeat in range(2):
            # Compute variance estimate (tilt series uses same path currently)
            variance_time = time.time()
            from recovar.heterogeneity import covariance_estimation
            # //2: variance computation with cubic disc_type needs ~2x memory per image (spline coefficients)
            variance_est, variance_prior, variance_fsc, lhs, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], utils.safe_batch_size(batch_size//2), dilated_volume_mask, noise_ind_subset = noise_ind_subset, use_regularization = True, disc_type = 'cubic')
            logger.info("variance estimation time: %s", time.time() - variance_time)
            utils.report_memory_device(logger=logger)

            rad_grid = np.array(fourier_transform_utils.get_grid_of_radial_distances(cryos.volume_shape).reshape(-1))
            # Often low frequency noise will be overestiated. This can be bad for the covariance estimation. This is a way to upper bound noise in the low frequencies by noise + variance .
            n_shell_to_ub = np.min([32, cryos.grid_size//2 -1])
            ub_noise_var_by_var_est = np.zeros(n_shell_to_ub, dtype = np.float32)
            variance_est_low_res_5_pc = np.zeros(n_shell_to_ub, dtype = np.float32)
            variance_est_low_res_median = np.zeros(n_shell_to_ub, dtype = np.float32)

            for k in range(n_shell_to_ub):
                if np.sum(rad_grid==k) >0:
                    ub_noise_var_by_var_est[k] = np.percentile(noise_p_variance_est[rad_grid==k], 5)
                    ub_noise_var_by_var_est[k] = np.max([0, ub_noise_var_by_var_est[k]])
                    variance_est_low_res_5_pc[k] = np.percentile(variance_est['combined'][rad_grid==k], 5)
                    variance_est_low_res_median[k] = np.median(variance_est['combined'][rad_grid==k])

            if np.any(ub_noise_var_by_var_est >  noise_var_used[:n_shell_to_ub]):
                logger.info("Estimated noise greater than upper bound. Bounding noise using estimated upper bound")

            if np.any(variance_est_low_res_5_pc < 0):
                logger.info("Estimated variance resolution is < 0. Noise was likely incorrectly estimated. Recomputing noise")
                logger.info("5 percentile: %s", variance_est_low_res_5_pc)
                logger.info("5 percentile/median over low shells: %s", variance_est_low_res_5_pc/variance_est_low_res_median)

                # This is a bit of a hack. We are using the variance estimate to bound the noise variance
                # This is not correct, but it is better than nothing
            noise_var_used[:n_shell_to_ub] = np.where( noise_var_used[:n_shell_to_ub] > ub_noise_var_by_var_est, ub_noise_var_by_var_est, noise_var_used[:n_shell_to_ub])

            noise_var_used = noise_var_used.astype(cryos[0].dtype_real)
            if noise_ind_subset is None:
                update_noise_variance(noise_var_used, cryos)
            else:
                new_noise_variance = cryos[0].noise.noise_variance_radials.copy()
                new_noise_variance[noise_ind_subset] = noise_var_used
                update_noise_variance(new_noise_variance, cryos)
            
        return variance_est, ub_noise_var_by_var_est


@nvtx.annotate("estimate_noise_level_no_masks", color="orange", domain=NVTX_DOMAIN_NOISE)
def estimate_noise_level_no_masks(experiment_dataset, image_subset, mean_estimate, batch_size, disc_type='linear_interp'):
    lhs = 0
    rhs = 0 
    
    # Print debug info about input parameters
    
    config = ForwardModelConfig.from_dataset(experiment_dataset, disc_type=disc_type)

    data_generator = experiment_dataset.get_image_subset_generator(
        batch_size=batch_size,
        subset_indices=image_subset
    )

    total_loss = 0.0
    n_images = (image_subset.size if image_subset is not None else experiment_dataset.n_images)
    for batch, _, batch_ind in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch)
        batch = core.translate_images(batch, experiment_dataset.translations[batch_ind], experiment_dataset.image_shape)
        CTF = experiment_dataset.CTF_fun(experiment_dataset.CTF_params[batch_ind], experiment_dataset.image_shape, experiment_dataset.voxel_size)

        if mean_estimate is not None:
            projected_mean = core_forward.forward_model(
                config, mean_estimate,
                experiment_dataset.CTF_params[batch_ind],
                experiment_dataset.rotation_matrices[batch_ind],
            )
            if experiment_dataset.premultiplied_ctf:
                batch = batch - projected_mean * CTF
            else:
                batch = batch - projected_mean

        averaged_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, experiment_dataset.image_shape, 0)
        lhs += jnp.sum(averaged_PS, axis=0)

        if experiment_dataset.premultiplied_ctf:
            averaged_CTF_square = regularization.batch_average_over_shells(jnp.abs(CTF)**2, experiment_dataset.image_shape, 0) 
            rhs += jnp.sum(averaged_CTF_square, axis=0)
        else:
            rhs += batch.shape[0] 


    logger.info("Finished processing all batches")
    estimated_noise = lhs / rhs
    # Replace any inf entries with the last non-inf value. Inf value can happen when the CTF is 0, because of weight dosing.
    non_inf_mask = ~jnp.isinf(estimated_noise)
    if not jnp.all(non_inf_mask):
        # Find the last non-inf value
        last_valid_idx = jnp.where(non_inf_mask)[0][-1]
        last_valid_value = estimated_noise[last_valid_idx]
        # Replace inf values with the last valid value
        estimated_noise = jnp.where(non_inf_mask, estimated_noise, last_valid_value)
    return estimated_noise


def batch_make_radial_noise(average_image_PS, image_shape):
    return jax.vmap(lambda amp: make_radial_noise(amp, image_shape))(average_image_PS)

    
    # Perhaps it should be mean at low freq and median at high freq?
mean_fn = np.mean

@nvtx.annotate("estimate_noise_variance", color="yellow", domain=NVTX_DOMAIN_NOISE)
def estimate_noise_variance(experiment_dataset, batch_size, max_images = 10000):
    """Estimate per-image noise variance from corner pixels.

    Computes the noise power spectrum from image regions outside the
    particle mask, subsampling to at most *max_images* for efficiency.

    Args:
        experiment_dataset: A ``CryoEMDataset`` instance.
        batch_size: Number of images to process per GPU batch.
        max_images: Maximum number of images to use for estimation.

    Returns:
        Tuple ``(cov_noise, radial_noise_profile)`` where *cov_noise*
        is a scalar noise variance and *radial_noise_profile* is the
        averaged radial power spectrum of the noise.
    """
    sum_sq = 0

    # Subsample at most 10000 images
    if experiment_dataset.n_images > max_images:
        # Calculate subsampling ratio
        subsample_ratio = max_images / experiment_dataset.n_images
        # Create subset indices for subsampling
        subset_indices = np.random.choice(
            experiment_dataset.n_images, 
            size=max_images, 
            replace=False
        )
        data_generator = experiment_dataset.get_image_subset_generator(
            batch_size=batch_size, 
            subset_indices=subset_indices
        )
        n_images_used = max_images
    else:
        data_generator = experiment_dataset.get_image_generator(batch_size=batch_size)
        n_images_used = experiment_dataset.n_images
    

    for batch, _, _ in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch)
        sum_sq += jnp.sum(np.abs(batch)**2, axis =0)

    mean_PS =  sum_sq / n_images_used
    cov_noise_mask = jnp.median(mean_PS)

    average_image_PS = regularization.average_over_shells(mean_PS, experiment_dataset.image_shape)

    return np.asarray(cov_noise_mask, dtype=experiment_dataset.dtype_real), np.asarray(average_image_PS, dtype=experiment_dataset.dtype_real)
    

def estimate_white_noise_variance_from_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):
    _, predicted_pixel_variances, _ = estimate_noise_variance_from_outside_mask_v2(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp')
    return np.median(predicted_pixel_variances)


def estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_image_generator(batch_size=batch_size) 
    image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    masked_image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    for batch, particles_ind, batch_ind in data_generator:
        masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask_inner(batch, 
                    volume_mask, experiment_dataset.rotation_matrices[batch_ind], 
                    experiment_dataset.translations[batch_ind], 
                    image_mask, 
                    experiment_dataset.volume_mask_threshold, 
                    experiment_dataset.image_shape, 
                    experiment_dataset.volume_shape, 
                    experiment_dataset.grid_size, 
                    experiment_dataset.padding, 
                    disc_type, 
                    experiment_dataset.image_stack.process_images)
        image_PSs[batch_ind] = np.array(image_PS)
        masked_image_PSs[batch_ind] = np.array(masked_image_PS)

    return masked_image_PSs, image_PSs


def estimate_noise_variance_from_outside_mask_v2(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_image_generator(batch_size=batch_size) 

    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    top_fraction = 0
    kernel_sq_sum =0 
    for batch, particles_ind, batch_ind in data_generator:
        top_fraction_this, kernel_sq_sum_this, per_image_est = estimate_noise_variance_from_outside_mask_inner_v2(batch, 
                    volume_mask, experiment_dataset.rotation_matrices[batch_ind], 
                    experiment_dataset.translations[batch_ind], 
                    image_mask, 
                    experiment_dataset.volume_mask_threshold, 
                    experiment_dataset.image_shape, 
                    experiment_dataset.volume_shape, 
                    experiment_dataset.grid_size, 
                    experiment_dataset.padding, 
                    disc_type, 
                    experiment_dataset.image_stack.process_images)
        top_fraction += top_fraction_this
        kernel_sq_sum+= kernel_sq_sum_this
    predicted_pixel_variances= top_fraction / kernel_sq_sum
    predicted_pixel_variances = jnp.fft.ifft2( predicted_pixel_variances).real * experiment_dataset.image_size


    pred_noise = regularization.average_over_shells(predicted_pixel_variances, experiment_dataset.image_shape, 0) 
    return pred_noise, predicted_pixel_variances, per_image_est


@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10,11])    
def estimate_noise_variance_from_outside_mask_inner_v2(batch, volume_mask, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, process_fn):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften =5 )
    
    # Invert mask
    image_mask = 1 - image_mask

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    return get_masked_image_noise_fractions(batch, image_mask, image_shape)


def get_masked_image_noise_fractions(images, image_masks, image_shape):
    images = covariance_core.apply_image_masks(images, image_masks, image_shape)

    masked_variance = jnp.abs(images.reshape([-1, *image_shape]))**2
    masked_variance_ft = jnp.fft.fft2(masked_variance)

    f_mask = jnp.fft.fft2(image_masks)
    kernels = jnp.fft.ifft2(jnp.abs(f_mask)**2)
    kernel_sq_sum = jnp.sum(jnp.abs(kernels)**2, axis=0)
    top_fraction= jnp.sum(masked_variance_ft * jnp.conj(kernels), axis=0) 

    # get a per image one
    kernels_bad = jnp.abs(kernels)  < jax_config.EPSILON
    kernels = jnp.where(kernels_bad, jnp.ones_like(kernels_bad) , kernels )
    per_image_estimate = jnp.where( kernels_bad, jnp.zeros_like(masked_variance_ft),  masked_variance_ft / kernels )

    return top_fraction, kernel_sq_sum, jnp.fft.ifft2(per_image_estimate).real * np.prod(image_shape)

def get_masked_noise_variance_from_noise_variance(image_masks, unmasked_noise_variance, image_shape):

    f_mask = jnp.fft.ifft2(image_masks)
    f_mask = jnp.fft.fft2(jnp.abs(f_mask)**2)

    image_cov_noise_ft = jnp.fft.fft2(unmasked_noise_variance.reshape(-1, *image_shape))
    masked_noise_variance = jnp.fft.ifft2( f_mask * image_cov_noise_ft )

    return masked_noise_variance.real


@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10,11])    
def estimate_noise_variance_from_outside_mask_inner(batch, volume_mask, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, process_fn):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften =10)
    
    # Invert mask
    image_mask = 1 - image_mask

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    image_size = batch.shape[-1]
    # Integral of mask:
    image_mask_2 = fourier_transform_utils.get_dft2(image_mask)
    image_mask_sums = jnp.sum(jnp.abs(image_mask_2)**2, axis =(-2, -1)) / image_size**2 
    masked_image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0) / image_mask_sums[:,None]


    return masked_image_PS, image_PS
    

def estimate_radial_noise_upper_bound_from_inside_mask_v2(experiment_dataset, mean_estimate, volume_mask, batch_size):
    noise_dist, per_pixel, aa = get_average_residual_square_just_mean(experiment_dataset, volume_mask, mean_estimate, batch_size, disc_type = 'linear_interp')
    return noise_dist, per_pixel, aa


# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_radial_noise_statistic_from_outside_mask(experiment_dataset, volume_mask, batch_size):
    masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp')
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0), mean_fn(image_PS, axis =0), np.std(image_PS, axis =0)


def make_radial_noise(average_image_PS, image_shape):
    # If you pass a scalar, return a constant
    if average_image_PS.size == 1:
        return np.ones(image_shape, dtype =average_image_PS.dtype ) * average_image_PS
    
    return utils.make_radial_image(average_image_PS, image_shape, extend_last_frequency = True)


# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_noise_from_heterogeneity_residuals_inside_mask(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp', subset_indices= None):
    masked_image_PS =  get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type, subset_indices=subset_indices  )
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0)

def get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp', subset_indices = None):
    
    if subset_indices is None:
        n_images = experiment_dataset.n_images
        data_generator = experiment_dataset.get_image_generator(batch_size=batch_size) 
    else:
        n_images = subset_indices.size
        data_generator = experiment_dataset.get_image_subset_generator(batch_size=batch_size, subset_indices = subset_indices) 

    residual_squared = jnp.zeros(experiment_dataset.image_stack.image_size, dtype = basis.dtype)
    all_averaged_residual_squared = np.empty((n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)
    basis = jnp.asarray(basis.T)
    for batch, _, batch_image_ind in data_generator:
        averaged_residual_squared = get_average_residual_square_inner(batch, mean_estimate, volume_mask, 
                                                                        basis,
                                                                        experiment_dataset.CTF_params[batch_image_ind],
                                                                        experiment_dataset.rotation_matrices[batch_image_ind],
                                                                        experiment_dataset.translations[batch_image_ind],
                                                                        experiment_dataset.image_stack.mask,
                                                                        experiment_dataset.volume_mask_threshold,
                                                                        experiment_dataset.image_shape, 
                                                                        experiment_dataset.volume_shape, 
                                                                        experiment_dataset.grid_size, 
                                                                        experiment_dataset.voxel_size, 
                                                                        experiment_dataset.padding, 
                                                                        disc_type, 
                                                                        experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun, 
                                                                        contrasts[batch_image_ind], basis_coordinates[batch_image_ind])
        all_averaged_residual_squared[batch_image_ind] = np.array(averaged_residual_squared)

    return all_averaged_residual_squared


def get_average_residual_square_inner(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, process_fn, CTF_fun, contrasts,basis_coordinates):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften = 5 )
    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=grid_size, voxel_size=voxel_size,
        padding=padding, disc_type=disc_type, CTF_fun=CTF_fun,
        premultiplied_ctf=False, volume_mask_threshold=volume_mask_threshold,
    )

    projected_mean = core_forward.forward_model(
        config, mean_estimate, CTF_params, rotation_matrices,
    )

    projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
    AUs = covariance_core.batch_vol_forward_from_map(
        config, basis, CTF_params, rotation_matrices,
    ) 
    # Apply mask on operator
    AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)
    image_mask_sums = jnp.sum(image_mask, axis =(-2, -1)) / batch.shape[-1]

    predicted_images = contrasts[...,None] * (jax.lax.batch_matmul(AUs, basis_coordinates[...,None])[...,0] + projected_mean)
    residual_squared = jnp.abs(batch - predicted_images)**2    / image_mask_sums[...,None]
    averaged_residual_squared = regularization.batch_average_over_shells(residual_squared, image_shape,0) 

    return averaged_residual_squared


def get_average_residual_square_just_mean(experiment_dataset, volume_mask, mean_estimate, batch_size, disc_type = 'linear_interp', subset_indices = None, subset_fn = None):
    contrasts = np.ones(experiment_dataset.n_images, dtype = experiment_dataset.dtype_real)
    basis = np.zeros((experiment_dataset.volume_size, 0))
    zs = np.zeros((experiment_dataset.n_images, 0))

    return get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,zs, batch_size, disc_type = disc_type, subset_indices=subset_indices, subset_fn = subset_fn)


def estimate_noise_from_heterogeneity_residuals_inside_mask_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    return get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type )


def get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp', subset_indices = None, subset_fn = None):


    assert basis.shape[0] == experiment_dataset.volume_size, "input u should be volume_size x basis_size"
    st_time = time.time()    
    basis = np.asarray(basis[:, :basis_coordinates.shape[-1]]).T

    if disc_type == 'cubic':
        st_time = time.time()
        from recovar.core import cubic_interpolation
        from recovar.heterogeneity import covariance_estimation
        mean_estimate = cubic_interpolation.calculate_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))
        basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)
        logger.info("Time to compute spline coefficients: %f", time.time() - st_time)


    if subset_indices is None:
        n_images = experiment_dataset.n_images
        data_generator = experiment_dataset.get_image_generator(batch_size=batch_size) 

    else:
        n_images = subset_indices.size
        data_generator = experiment_dataset.get_image_subset_generator(batch_size=batch_size, subset_indices = subset_indices) 


    # Construct structured parameters once outside the loop
    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.image_stack.process_images,
    )
    model = ModelState(
        mean_estimate=mean_estimate,
        volume_mask=volume_mask,
        basis=jnp.asarray(basis.T),
    )

    top_fraction = 0
    kernel_sq_sum = 0

    for batch, _, batch_image_ind in data_generator:
        if subset_fn is not None:
            idx = subset_fn(batch_image_ind)
            batch = batch[idx]
            batch_image_ind = batch_image_ind[idx]

        batch_data = BatchData(
            images=batch,
            ctf_params=experiment_dataset.CTF_params[batch_image_ind],
            rotation_matrices=experiment_dataset.rotation_matrices[batch_image_ind],
            translations=experiment_dataset.translations[batch_image_ind],
        )
        top_fraction_this, kernel_sq_sum_this, per_image_est = average_residual_square(
            config, batch_data, model,
            experiment_dataset.image_stack.mask,
            contrasts[batch_image_ind], basis_coordinates[batch_image_ind],
        )

        top_fraction += top_fraction_this
        kernel_sq_sum += kernel_sq_sum_this

    predicted_pixel_variances= top_fraction / kernel_sq_sum
    predicted_pixel_variances = jnp.fft.ifft2( predicted_pixel_variances).real * experiment_dataset.image_size

    pred_noise = regularization.average_over_shells(predicted_pixel_variances, experiment_dataset.image_shape, 0) 
    return pred_noise, predicted_pixel_variances, None


def basis_times_coords(basis, coords):
    assert basis.shape[-1] == coords.shape[-1]
    return jnp.sum(basis * coords, axis=-1)
def batch_basis_times_coords2(basis, coords):
    """Compute basis @ coords.T reshaped for batched images, memory-efficiently."""
    assert basis.shape[-1] == coords.shape[-1]
    basis_shape_inp = basis.shape

    basis = basis.transpose(-1, *np.arange(basis.ndim-1) )
    basis = basis.reshape((coords.shape[-1], np.prod(basis_shape_inp[:-1])))

    summed = basis.T @ coords.T

    summed = summed.T
    summed = summed.reshape(coords.shape[0], *basis_shape_inp[:-1])
    return summed


# ============================================================================
# New Equinox-based noise estimation API
# ============================================================================


@eqx.filter_jit
def average_residual_square(
    config: ForwardModelConfig,
    batch_data: BatchData,
    model: ModelState,
    image_mask: jax.Array,
    contrasts: jax.Array,
    basis_coordinates: jax.Array,
):
    """Compute average residual squared — Equinox API.

    Replaces the 19-param ``get_average_residual_square_inner_v2``.
    """
    batch = batch_data.images
    ctf_params = batch_data.ctf_params
    rotation_matrices = batch_data.rotation_matrices
    translations = batch_data.translations

    if model.volume_mask is not None:
        image_mask = covariance_core.get_per_image_tight_mask(
            model.volume_mask, rotation_matrices, image_mask,
            config.volume_mask_threshold, config.image_shape, config.volume_shape,
            config.grid_size, config.padding, config.disc_type, soften=5,
        )
    else:
        image_mask = jnp.ones_like(batch).real

    if model.basis.shape[-1] == 0:
        predicted_vols = contrasts.reshape(
            (contrasts.shape[0], *np.ones(model.mean_estimate.ndim, dtype=int))
        ) * model.mean_estimate[None]
    else:
        predicted_vols = contrasts.reshape(
            (contrasts.shape[0], *np.ones(model.mean_estimate.ndim, dtype=int))
        ) * (batch_basis_times_coords2(model.basis, basis_coordinates) + model.mean_estimate[None])

    # Per-image forward model: project volume[i] with CTF[i] and rotation[i]
    projected_vols = jax.vmap(
        lambda vol, ctf, rot: core_forward.forward_model(
            config, vol, ctf[None], rot[None]
        )[0],
    )(predicted_vols, ctf_params, rotation_matrices)

    if config.process_fn is not None:
        batch = config.process_fn(batch)
    batch = core.translate_images(batch, translations, config.image_shape)
    subtracted = batch - projected_vols

    return get_masked_image_noise_fractions(subtracted, image_mask, config.image_shape)
