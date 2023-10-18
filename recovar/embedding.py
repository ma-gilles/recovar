import logging
import jax.numpy as jnp
import numpy as np
import functools, time, jax

from recovar import core, covariance_core, latent_density, homogeneous, constants, utils
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)


def split_weights(weight, cryos):
    start_idx = 0
    weights = []
    for cryo in cryos:
        end_idx = start_idx + cryo.n_images
        weights.append(weight[start_idx:end_idx])
        start_idx = end_idx
    return weights

def generate_conformation_from_reweighting(cryos, means, cov_noise, zs, cov_zs, latent_points, batch_size, disc_type, likelihood_threshold = None, recompute_prior = True, volume_mask = None ):    
    
    likelihood_threshold = latent_density.get_log_likelihood_threshold(k = zs.shape[-1])
    weights = latent_density.compute_weights_of_conformation_2(latent_points, zs, cov_zs,likelihood_threshold = likelihood_threshold )

    all_weights_0 = []
    all_weights_1 = []

    for w in weights.T:
        weight_this = split_weights(w,cryos)
        all_weights_0.append(weight_this[0])
        all_weights_1.append(weight_this[1])

    image_weights = [np.array(all_weights_0),np.array(all_weights_1)] 

    reconstructions, fscs = homogeneous.get_multiple_conformations(cryos, cov_noise, disc_type, batch_size, means['prior'], means['combined']*0 , image_weights, recompute_prior = recompute_prior, volume_mask = volume_mask)
    return reconstructions, fscs
    
def generate_conformation_from_reprojection(xs, mean, u ):
    return ((mean[...,None] + u @ xs.T)[0]).T
    

def get_per_image_embedding(mean, u, s, basis_size, cov_noise, cryos, volume_mask, gpu_memory, disc_type = 'linear_interp',  contrast_grid = None, contrast_option = "contrast", to_real = True, parallel_analysis = False, compute_covariances = True ):
    
    st_time = time.time()    
    basis = np.array(u[:, :basis_size]) 
    eigenvalues = (s + constants.ROOT_EPSILON)
    use_contrast = "contrast" in contrast_option
    logger.info(f"using contrast? {use_contrast}")

    if use_contrast:
        contrast_grid = np.linspace(0, 2, 50) if contrast_grid is None else contrast_grid
    else:
        contrast_grid = np.ones([1])
    
    basis_size = u.shape[-1] if basis_size == -1 else basis_size

    left_over_memory = ( utils.get_gpu_memory_total() - utils.get_size_in_gb(basis))
    # batch_size = int(left_over_memory/ 
    #                 ((cryos[0].grid_size**2 * contrast_grid.size * basis_size
    #                 + cryos[0].grid_size * contrast_grid.size * basis_size**2) * utils.get_size_in_gb(cryos[0].get_image(0)) )/3)
    assert(left_over_memory > 0, "GPU memory too small?")
    batch_size = int(left_over_memory/ ( 
        (cryos[0].grid_size**2 * contrast_grid.size * basis_size
        + contrast_grid.size * basis_size**2)
        *8/1e9 )/ 20)

    batch_size_old = int((2**24)*8 /( cryos[0].grid_size**2 * np.max([basis_size, 8]) ) * gpu_memory / 38 ) 
    print("new batch:",batch_size, "old batch:", batch_size_old)
    logger.info(f"z batch size? {batch_size}")
    # logger.info(f"z batch size old {batch_size_old}")
    # import pdb; pdb.set_trace()

    zs = [None]*2; cov_zs = [None]*2; est_contrasts = [None]*2
    for cryo_idx,cryo in enumerate(cryos):
        zs[cryo_idx], cov_zs[cryo_idx], est_contrasts[cryo_idx] = get_coords_in_basis_and_contrast_3(
            cryo, mean, basis, eigenvalues[:basis.shape[-1]], volume_mask,
            jnp.array(cov_noise) , contrast_grid, batch_size, disc_type, 
            parallel_analysis = parallel_analysis, compute_covariances = compute_covariances )

    
    zs = np.concatenate(zs, axis = 0)
    est_contrasts = np.concatenate(est_contrasts)
    end_time = time.time()
    logger.info(f"time to compute xs {end_time - st_time}")
    
    if compute_covariances:
        cov_zs = np.concatenate(cov_zs, axis = 0)
        if to_real:
            cov_zs = cov_zs.real

    if to_real:
        zs = zs.real
    
    return zs, cov_zs, est_contrasts
    

# @functools.partial(jax.jit, static_argnums = [5])    
def get_coords_in_basis_and_contrast_3(experiment_dataset, mean_estimate, basis, eigenvalues, volume_mask, noise_variance, contrast_grid, batch_size, disc_type, parallel_analysis = False, compute_covariances = True ):
    
    basis = basis.T
        
    # Make sure variables used in every iteration are on gpu.
    basis = jnp.array(basis).astype(experiment_dataset.dtype)
    volume_mask = jnp.array(volume_mask).astype(experiment_dataset.dtype_real)
    mean_estimate = jnp.array(mean_estimate).astype(experiment_dataset.dtype)
    eigenvalues = jnp.array(eigenvalues).astype(experiment_dataset.dtype)
    contrast_grid = contrast_grid.astype(experiment_dataset.dtype_real)

    no_mask = covariance_core.check_mask(volume_mask)    
    
    basis_size = basis.shape[0]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    
    xs = np.zeros((experiment_dataset.n_images, basis_size), dtype = basis.dtype)
    estimated_contrasts = np.zeros(experiment_dataset.n_images, dtype = basis.dtype).real
    image_latent_covariances = np.zeros((experiment_dataset.n_images, basis_size, basis_size), dtype = basis.dtype) if compute_covariances else None
        
    batch_idx =0 
    for batch, batch_image_ind in data_generator:
        
        # batch = experiment_dataset.image_stack.process_images(batch)

        xs_single, contrast_single, cov_batch = compute_single_batch_coords_split(batch, mean_estimate, volume_mask, 
                                                                        basis, eigenvalues,
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
                                                                        compute_covariances, np.array(noise_variance),
                                                                        experiment_dataset.image_stack.process_images,
                                                                       experiment_dataset.CTF_fun, contrast_grid)
        
        xs[batch_image_ind] = xs_single
        estimated_contrasts[batch_image_ind] = contrast_single
        if compute_covariances:
            image_latent_covariances[np.array(batch_image_ind)] = cov_batch
            
        if (batch_idx % 50 == 49) and (batch_idx > 0):
            compute_single_batch_coords_split._clear_cache() 
        
        batch_idx +=1
    return xs, image_latent_covariances, estimated_contrasts



def compute_single_batch_coords_p1(batch, mean_estimate, volume_mask, basis, eigenvalues, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, compute_covariances, noise_variance, process_fn, CTF_fun, contrast_grid):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type )

    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    projected_mean = covariance_core.get_projected_image(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                           
                                          )


    ## DO MASK BUSINESS HERE.

    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
    projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)
    AUs = covariance_core.batch_over_vol_forward_model(basis,
                                         CTF_params, 
                                         rotation_matrices,
                                         image_shape, 
                                         volume_shape, 
                                         grid_size, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type )    
    # Apply mask on operator
    AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)

    AU_t_images = batch_x_T_y(AUs, batch)#.block_until_ready()
    AU_t_Amean = batch_x_T_y(AUs, projected_mean)#.block_until_ready()
    AU_t_AU = batch_x_T_y(AUs,AUs)#.block_until_ready()


    # Compute everything that is needed, before a low dimension contrast search
    image_norms_sq = jnp.linalg.norm(batch, axis =-1)**2
    image_T_A_mean =  batch_x_T_y(batch, projected_mean) #jnp.conj(images).T @ projected_mean
    A_mean_norm_sq = jnp.linalg.norm(projected_mean, axis =-1)**2
    
    return AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq
    
@functools.partial(jax.jit, static_argnums = [9,10,11,12,13,14,15,16,18, 19])    
def compute_single_batch_coords_split(batch, mean_estimate, volume_mask, basis, eigenvalues, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, compute_covariances, noise_variance, process_fn, CTF_fun, contrast_grid):
    
    # This should scale as O( batch_size * (n^2 * basis_size + n^3 + basis_size**2))
    AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq = compute_single_batch_coords_p1(batch, mean_estimate, volume_mask, basis, eigenvalues, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, compute_covariances, noise_variance, process_fn, CTF_fun, contrast_grid)
    
    masked_noises = noise_variance * jnp.ones(batch.shape[0], dtype = noise_variance.dtype) 

    # This should scale as O( contrast_grid_size * (n^2 * batch_size * basis_size +  )
    xs_batch_contrast = batch_over_images_and_contrast_solve_contrast_linear_system(AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, masked_noises, contrast_grid)

    # Compute residual
    residuals_fit, residuals_prior = batch_compute_contrast_residual_fast_2(xs_batch_contrast, AU_t_images, image_norms_sq, AU_t_Amean, A_mean_norm_sq, image_T_A_mean,  AU_t_AU, eigenvalues, masked_noises, contrast_grid)

    # Pick best contrast
    res_sum1 = residuals_fit + residuals_prior
    best_idx = jnp.argmin(res_sum1, axis = 1).astype(int)
    
    xs_single = batch_slice_ar(best_idx, xs_batch_contrast)
    contrast_single = contrast_grid[best_idx]

    # covariance
    if compute_covariances:
        cov_batch = (contrast_single**2 / masked_noises)[:,None,None] * AU_t_AU  + jnp.diag(1/eigenvalues)
    else:
        cov_batch = None
    
    return xs_single, contrast_single, cov_batch


def slice_ar(indx, arr):
    return arr[indx]

batch_slice_ar = jax.jit(jax.vmap(slice_ar, in_axes =(0, 0)))
batch_x_T_y = jax.vmap(  lambda x,y : jnp.conj(x).T @ y, in_axes = (0,0))

# Naive functions, without precompute
def compute_contrast_residual_naive(image, AU, projected_mean, xs, eigenvalues, noise_variance, contrast_grid):
    fit_residual =  jnp.linalg.norm( (contrast_grid * (AU @ xs.T + projected_mean[...,None]) - image[...,None]) / jnp.sqrt(noise_variance), axis =0)**2
    prior_residual = batch_x_T_y( xs, xs / eigenvalues) #jnp.conj(xs).T @ ( xs /  eigenvalues )
    return fit_residual,  prior_residual.real
batch_compute_contrast_residual_naive = jax.vmap(compute_contrast_residual_naive, in_axes = (0,0,0,0, None, None, None) )


@jax.jit
def compute_contrast_residual_fast_2(xs, AU_t_images, image_norms_sq, AU_t_Amean, Amean_norms_sq, image_T_A_mean ,  AU_t_AU, eigenvalues, noise_variance, contrast):
    
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

    return ((p1 + p2 + p3 + p4 + p5 + p6) / noise_variance).real, batch_x_T_y( xs, xs / eigenvalues).real

batch_compute_contrast_residual_fast_2 = jax.vmap(compute_contrast_residual_fast_2, in_axes = (0,0,0,0,0,0,0,None, 0, None))


@jax.jit
def solve_contrast_linear_system(AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, noise_variance, contrast):
    A = (contrast **2) * AU_t_AU / noise_variance +  jnp.diag(1 / eigenvalues )
    b = contrast * ( AU_t_images - contrast * AU_t_Amean ) / noise_variance
    sol = jnp.linalg.solve(A, b)    
    return sol

batch_over_contrast_solve_contrast_linear_system = jax.vmap(solve_contrast_linear_system, in_axes = ( None, None, None, None, None, 0) )
batch_over_images_and_contrast_solve_contrast_linear_system = jax.vmap(batch_over_contrast_solve_contrast_linear_system, in_axes = ( 0, 0,0,None,0, None) )

