import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
import pandas as pd
from recovar import core, covariance_core, regularization, utils, constants, noise, homogeneous, linalg, embedding, adaptive_kernel_discretization
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

## HAVEN'T FINISHED THIS YET

batch_over_vol_slice_volume_by_map = jax.vmap(core.slice_volume_by_map, in_axes = (1, None, None, None, None), out_axes = 1 )


def check_imaginary_part(x, image_shape, name, skip_ft = False ):
    if not skip_ft:
        if len(image_shape) == 2:
            y = ftu.get_idft2(x.reshape(-1, *image_shape))
        else:
            y = ftu.get_idft3(x.reshape(-1, *image_shape))
    else:
        y = x
    z = np.linalg.norm(y.real)/ np.linalg.norm(y.imag)
    print('imaginary part ratio', name, z)
    return np.linalg.norm(y.real)/ np.linalg.norm(y.imag)

@functools.partial(jax.jit, static_argnums = [8, 9, 13, 14 ])
def E_M_step_batch(images, lhs_summed, rhs_summed, mean, W, CTF_params, rotation_matrices, translations, image_shape, volume_shape, grid_size, voxel_size, noise_variance,  CTF_fun, compute_ll):
    disc_type = "nearest"
    basis_size = W.shape[1]
    volume_size = np.prod(volume_shape)
    
    # Precomp piece
    images = core.translate_images(images, translations, image_shape) / jnp.sqrt(noise_variance)
    # Just "whiten" the images and the projected mean, and include noise in CTF to simplify
    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / jnp.sqrt(noise_variance)
    projected_mean = core.forward_model_from_map(mean, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf = False) / jnp.sqrt(noise_variance)

    # check_imaginary_part(projected_mean, image_shape, 'projected_mean' )
    # check_imaginary_part(images, image_shape, 'images' )

    ctf_squared_over_noise_variance = CTF**2 
    # 
    PW = batch_over_vol_slice_volume_by_map(W, rotation_matrices, image_shape, volume_shape, disc_type)
    # n_images x n_basis_functions x image_size
    PW *= CTF[...,None,:]

    # check_imaginary_part(PW, image_shape, 'PW' )
    # check_imaginary_part(W.T, volume_shape, 'W' )


    # Swap axes to get n_images x n_basis_functions x image_size
    # PW = PW.transpose(2,0,1)
     
    # P W .T @ P W
    M_n = (jnp.conj(PW) @ PW.transpose(0,2,1)).real + jnp.eye(basis_size)
    
    centered_images = images - projected_mean
    b_n = (jnp.conj(PW) @ centered_images[...,None]).real
    # check_imaginary_part(b_n, volume_shape, 'bn', skip_ft = True )

    M_n_inv = jax.numpy.linalg.pinv(M_n, hermitian=True)
    expected_zs = (M_n_inv @ b_n).squeeze(-1)
    # check_imaginary_part(expected_zs, volume_shape, '<z>', skip_ft = True )
    # check_imaginary_part(M_n_inv, volume_shape, 'Var(z)', skip_ft = True )

    # print('np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0)', np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0))

     
    second_moment_zs = M_n_inv + linalg.broadcast_outer(expected_zs, jnp.conj(expected_zs)) #expected_zs[...,None] * jnp.conj(expected_zs)[...,None]

    # grid_point_indices = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size)
    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )

    # Should be size n_images x image_size x basis_size x basis_size
    before_backproj_second_moments = ctf_squared_over_noise_variance[...,None,None] * second_moment_zs[:,None,:,:]
    before_backproj_first_moments = CTF[...,None] * centered_images[...,None] * jnp.conj(expected_zs)[:,None,:]
     

    lhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_second_moments.reshape(*before_backproj_second_moments.shape[:-2], -1), grid_point_vec_indices, lhs_summed)
    rhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_first_moments, grid_point_vec_indices, rhs_summed)
     
    # return lhs_summed, rhs_summed, expected_zs, second_moment_zs
    # --- Optional log-likelihood (observed-data) ---
    #   ell_n = -0.5 * [ d_n log(2π) + ||r||^2 - u^T M^{-1} u + logdet M ]
    if compute_ll:
        # u = b_n.squeeze(-1)
        u = b_n.squeeze(-1)  # (b, q)

        # quadratic term via M_n_inv (already computed):
        quad = jnp.real(jnp.sum(jnp.conj(u) * (M_n_inv @ u[..., None]).squeeze(-1), axis=-1))  # (b,)

        # ||r||^2
        r2 = jnp.real(jnp.sum(jnp.conj(centered_images) * centered_images, axis=-1))  # (b,)

        # logdet M via Cholesky (more stable than slogdet on Hermitian PD)
        L = jnp.linalg.cholesky(M_n)  # (b, q, q)
        logdetM = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=1, axis2=2))), axis=-1)  # (b,)
        
        
        d_n         = images.shape[-1]  # image dimensionality (pixels)
        const = d_n * jnp.log(2.0 * jnp.pi)
        ll_per_image = -0.5 * (const + r2 - quad + logdetM)  # (b,)
        ll_sum = jnp.sum(ll_per_image)
    else:
        ll_sum = jnp.array(0.0, dtype=images.dtype)
        ll_per_image = jnp.zeros((0,), dtype=images.dtype)

    return lhs_summed, rhs_summed, expected_zs, second_moment_zs, ll_sum, ll_per_image



batch1_symmetrize_ft_volume = jax.vmap(utils.symmetrize_ft_volume, in_axes = (1, None), out_axes = 1)

# @functools.partial(jax.jit, static_argnums = [5])    
def EM_step(experiment_datasets, mean_estimate, W_estimate, batch_size, W_prior, sparse_PCA = False):
    
            
    basis_size = W_estimate.shape[-1]
    rhs_summed = jnp.zeros((experiment_datasets[0].volume_size, basis_size), dtype = experiment_datasets[0].dtype)
    lhs_summed = jnp.zeros((experiment_datasets[0].volume_size, basis_size *  basis_size), dtype = experiment_datasets[0].dtype_real)

    ll_sum = jnp.array(0.0, dtype=experiment_datasets[0].dtype)
    expected_zs = []
    second_moment_zs = []
    for experiment_dataset in experiment_datasets:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
        for batch, particles_ind, batch_image_ind in data_generator:
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
            lhs_summed, rhs_summed, expected_zs_batch, second_moment_zs_batch, ll_sum_batch, _ = E_M_step_batch(batch, lhs_summed, rhs_summed, mean_estimate, W_estimate,
                                                experiment_dataset.CTF_params[batch_image_ind],
                                                experiment_dataset.rotation_matrices[batch_image_ind],
                                                experiment_dataset.translations[batch_image_ind],
                                                experiment_dataset.image_shape, 
                                                experiment_dataset.volume_shape, 
                                                experiment_dataset.grid_size, 
                                                experiment_dataset.voxel_size, 
                                                noise_variance,
                                                experiment_dataset.CTF_fun,
                                                compute_ll = True)
            expected_zs.append(np.array(expected_zs_batch))
            second_moment_zs.append(np.array(second_moment_zs_batch))
            ll_sum += ll_sum_batch


    expected_zs = np.concatenate(expected_zs, axis=0)
    second_moment_zs = np.concatenate(second_moment_zs, axis=0)
    
    # Calculate statistics for reporting
    expected_zs_mean = np.mean(expected_zs, axis=0)
    expected_zs_var = np.var(expected_zs, axis=0)

    # Solve least squares
    # V = jax.vmap(jnp.diag)(1 / (W_prior + 1e-16 ))
     

    if sparse_PCA:
        cryos = experiment_datasets
        volume_size = cryos[0].volume_size
        volume_shape = cryos[0].volume_shape
        normal_size = W_estimate.shape

        from recovar.ppca.admm_test import WaveletL1
        ll_prior = WaveletL1(normal_size, volume_shape, 'db1', sigma=W_prior)(W_estimate)


        lhs_summed = batch1_symmetrize_ft_volume(lhs_summed, volume_shape)
        lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)

        rhs_summed = batch1_symmetrize_ft_volume(rhs_summed, volume_shape)
        W_estimate = batch1_symmetrize_ft_volume(W_estimate, volume_shape)

        from recovar.ppca.admm_test import admm_wavelet
        W, Z_rec = admm_wavelet( lhs_summed, rhs_summed, W_prior, 0.9, 20, volume_shape, normal_size, W_estimate)

    else:
        lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)

        lhs_summed = lhs_summed  + jax.vmap(jnp.diag)(1 / (W_prior + 1e-16 ) )
         
        # W = linalg.batch_hermitian_linear_solver(lhs_summed, rhs_summed)
        W = linalg.batch_linear_solver(lhs_summed, rhs_summed[...,None])[...,0]

        # Note that this is the log likelihood at the previous W_estimate
        ll_prior = jnp.linalg.norm(W_estimate / jnp.sqrt(W_prior + 1e-16 ))**2

    # Calculate log-likelihood statistics
    neg_ll_total = float(-ll_sum.real + ll_prior.real)
    neg_ll_data = float(-ll_sum.real)
    neg_ll_prior = float(ll_prior.real)

    if jnp.isnan(W).any():
        import pdb; pdb.set_trace()
    return W, expected_zs, second_moment_zs, expected_zs_mean, expected_zs_var, neg_ll_total, neg_ll_data, neg_ll_prior



def batch_vec(x):
    return x.swapaxes(-1,-2).reshape(-1, x.shape[-1]**2)

def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1,n,n).swapaxes(-1,-2)

import matplotlib.pyplot as plt

def EM(experiment_dataset, mean_estimate, W_initial, W_prior, EM_iter = 20, sparse_PCA = False, U_gt = None, S_gt = None, make_plots = False):

    # Initialize
    # import jax.random as jr
    # matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    # W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype_real)
    # W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    # eigenvalue = np.ones(basis_size)
    volume_mask = np.ones(experiment_dataset[0].volume_shape)
    basis_size = W_initial.shape[-1]
    contrast_grid = np.ones([1])
    batch_size = 1000
    disc_type = 'nearest'
    W = W_initial
    
    # Initialize table for collecting iteration data
    iteration_data = []
    
    # Print table header
    print("\n" + "="*120)
    print("EM ALGORITHM CONVERGENCE TABLE")
    print("="*120)
    header = f"{'Iter':>4} | {'Neg_LL_Total':>12} | {'Neg_LL_Data':>12} | {'Neg_LL_Prior':>12} | {'Exp_ZS_Mean':>12} | {'Exp_ZS_Var':>12} | {'Rel_Var_Expl':>12}"
    if U_gt is not None:
        header += f" | {'Top_5_Rel_Var':>20}"
    print(header)
    print("-" * len(header))
    
    for iter_i in range(EM_iter):
        W, expected_zs, second_moment_zs, expected_zs_mean, expected_zs_var, neg_ll_total, neg_ll_data, neg_ll_prior = EM_step(experiment_dataset, mean_estimate, W, batch_size, W_prior, sparse_PCA)

        #Make real
        W = W.T.reshape(basis_size, *experiment_dataset[0].volume_shape)
        W = ftu.get_idft3(W).real
        W = W.reshape(W.shape[0], -1).T

        # SVD?
        # U, S, _ = jnp.linalg.svd(W, full_matrices=False)
        # W = U * S 

        W = W.T
        W = ftu.get_dft3(W.reshape(W.shape[0], *experiment_dataset[0].volume_shape))
        W = W.reshape(W.shape[0], -1).T

        # plt.figure()
        # plt.imshow(experiment_dataset[0].get_proj(W[:,0].reshape(-1)))

        logger.info(f"Done with EM step {iter_i}")

        # Collect iteration data
        iter_info = {
            'Iteration': iter_i,
            'Neg_LL_Total': f"{neg_ll_total:.6e}",
            'Neg_LL_Data': f"{neg_ll_data:.6e}",
            'Neg_LL_Prior': f"{neg_ll_prior:.6e}",
            'Expected_ZS_Mean': f"{np.mean(expected_zs_mean):.6e}",
            'Expected_ZS_Var': f"{np.mean(expected_zs_var):.6e}"
        }

        if U_gt is not None:
            U, S, _ = jnp.linalg.svd(W, full_matrices=False)
            from recovar import metrics
            variance, rel_var, norm_var = metrics.get_all_variance_scores(U, U_gt, S_gt)
            iter_info['Rel_Var_Explained'] = f"{(rel_var[-1]):.6e}"
            iter_info['Top_5_Rel_Var'] = f"{rel_var[:5]}"
        else:
            iter_info['Rel_Var_Explained'] = 'N/A'
            iter_info['Top_5_Rel_Var'] = 'N/A'

        iteration_data.append(iter_info)
        
        # Print current iteration row
        row = f"{iter_i:>4} | {neg_ll_total:12.6e} | {neg_ll_data:12.6e} | {neg_ll_prior:12.6e} | {np.mean(expected_zs_mean):12.6e} | {np.mean(expected_zs_var):12.6e}"
        if U_gt is not None:
            row += f" | {rel_var[-1]:12.6e} | {str(rel_var[:min(5, len(rel_var))]):>20}"
        else:
            row += f" | {'N/A':>12}"
        print(row)

        if (make_plots or iter_i == EM_iter - 1) and U_gt is not None:
            max_size_this = np.min([20, U.shape[-1]])
            plt.figure()
            plt.plot(rel_var)
            plt.title('relative variance expained at iteration ' + str(iter_i))
            plt.show()
            u = { 'ppca': U, 'gt': U_gt}
            cryos = experiment_dataset
            ppca_key = 'ppca'
            n_rows = np.max([2, u[ppca_key].shape[-1]])
            n_cols = len(u.keys())
            fig_size = (n_cols * 4, n_rows * 4)
            fig, axes = plt.subplots(  n_rows, n_cols, figsize=(fig_size))
            for i, u_key in enumerate(u.keys()):
                # Plot PPCA components
                for j in range(u[ppca_key].shape[-1]):
                    axes[j, i].imshow(cryos[0].get_proj(u[u_key][:,j].reshape(-1)))
                    axes[j, i].set_title(f'{u_key} PC{j+1}')

            plt.tight_layout()
            plt.show()

    # Print final summary
    print("="*120)
    print("EM ALGORITHM COMPLETED")
    print("="*120)

    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    return U, S**2, W, expected_zs, second_moment_zs


