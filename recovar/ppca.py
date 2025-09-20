import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants, noise, homogeneous, linalg, embedding, adaptive_kernel_discretization
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

## HAVEN'T FINISHED THIS YET

batch_over_vol_slice_volume_by_map = jax.vmap(core.slice_volume_by_map, in_axes = (1, None, None, None, None), out_axes = 1 )


def check_imaginary_part(x, image_shape, name, skip_ft = False ):
    return 0
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

@functools.partial(jax.jit, static_argnums = [8, 9, 13 ])
def E_M_step_batch(images, lhs_summed, rhs_summed, mean, W, CTF_params, rotation_matrices, translations, image_shape, volume_shape, grid_size, voxel_size, noise_variance,  CTF_fun):
    disc_type = "nearest"
    basis_size = W.shape[1]
    volume_size = np.prod(volume_shape)
    
    # Precomp piece
    images = core.translate_images(images, translations, image_shape) / jnp.sqrt(noise_variance)
    # Just "whiten" the images and the projected mean, and include noise in CTF to simplify
    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / jnp.sqrt(noise_variance)
    projected_mean = core.forward_model_from_map(mean, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf = False) / jnp.sqrt(noise_variance)

    check_imaginary_part(projected_mean, image_shape, 'projected_mean' )
    check_imaginary_part(images, image_shape, 'images' )

    ctf_squared_over_noise_variance = CTF**2 
    # 
    PW = batch_over_vol_slice_volume_by_map(W, rotation_matrices, image_shape, volume_shape, disc_type)
    # n_images x n_basis_functions x image_size
    PW *= CTF[...,None,:]

    check_imaginary_part(PW, image_shape, 'PW' )
    check_imaginary_part(W.T, volume_shape, 'W' )


    # Swap axes to get n_images x n_basis_functions x image_size
    # PW = PW.transpose(2,0,1)
    # import pdb; pdb.set_trace()
    # P W .T @ P W
    M_n = jnp.conj(PW) @ PW.transpose(0,2,1) + jnp.eye(basis_size)
    
    centered_images = images - projected_mean
    b_n = jnp.conj(PW) @ centered_images[...,None]
    check_imaginary_part(b_n, volume_shape, 'bn', skip_ft = True )

    M_n_inv = jax.numpy.linalg.pinv(M_n, hermitian=True)
    expected_zs = (M_n_inv @ b_n).squeeze(-1)
    check_imaginary_part(expected_zs, volume_shape, '<z>', skip_ft = True )
    check_imaginary_part(M_n_inv, volume_shape, 'Var(z)', skip_ft = True )

    # print('np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0)', np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0))

    # import pdb; pdb.set_trace()
    second_moment_zs = M_n_inv + linalg.broadcast_outer(expected_zs, jnp.conj(expected_zs)) #expected_zs[...,None] * jnp.conj(expected_zs)[...,None]

    # grid_point_indices = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size)
    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )

    # Should be size n_images x image_size x basis_size x basis_size
    before_backproj_second_moments = ctf_squared_over_noise_variance[...,None,None] * second_moment_zs[:,None,:,:]
    before_backproj_first_moments = CTF[...,None] * centered_images[...,None] * jnp.conj(expected_zs)[:,None,:]
    # import pdb; pdb.set_trace()

    lhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_second_moments.reshape(*before_backproj_second_moments.shape[:-2], -1), grid_point_vec_indices, lhs_summed)
    rhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_first_moments, grid_point_vec_indices, rhs_summed)
    # import pdb; pdb.set_trace()
    return lhs_summed, rhs_summed, expected_zs, second_moment_zs



# @functools.partial(jax.jit, static_argnums = [5])    
def EM_step(experiment_datasets, mean_estimate, W_estimate, batch_size, W_prior ):
    
            
    basis_size = W_estimate.shape[-1]
    rhs_summed = jnp.zeros((experiment_datasets[0].volume_size, basis_size), dtype = experiment_datasets[0].dtype)
    lhs_summed = jnp.zeros((experiment_datasets[0].volume_size, basis_size *  basis_size), dtype = experiment_datasets[0].dtype)

    expected_zs = []
    second_moment_zs = []
    for experiment_dataset in experiment_datasets:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
        for batch, particles_ind, batch_image_ind in data_generator:
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
            lhs_summed, rhs_summed, expected_zs_batch, second_moment_zs_batch = E_M_step_batch(batch, lhs_summed, rhs_summed, mean_estimate, W_estimate,
                                                experiment_dataset.CTF_params[batch_image_ind],
                                                experiment_dataset.rotation_matrices[batch_image_ind],
                                                experiment_dataset.translations[batch_image_ind],
                                                experiment_dataset.image_shape, 
                                                experiment_dataset.volume_shape, 
                                                experiment_dataset.grid_size, 
                                                experiment_dataset.voxel_size, 
                                                noise_variance,
                                                experiment_dataset.CTF_fun)
            expected_zs.append(np.array(expected_zs_batch))
            second_moment_zs.append(np.array(second_moment_zs_batch))
            
    expected_zs = np.concatenate(expected_zs, axis=0)
    second_moment_zs = np.concatenate(second_moment_zs, axis=0)
    print('np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0)', np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0))
    # Solve least squares
    lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)
    # V = jax.vmap(jnp.diag)(1 / (W_prior + 1e-16 ))
    # import pdb; pdb.set_trace()
    lhs_summed = lhs_summed  + jax.vmap(jnp.diag)(1 / (W_prior + 1e-16 ) )
    # import pdb; pdb.set_trace()
    # W = linalg.batch_hermitian_linear_solver(lhs_summed, rhs_summed)
    W = linalg.batch_linear_solver(lhs_summed, rhs_summed[...,None])[...,0]

    if jnp.isnan(W).any():
        import pdb; pdb.set_trace()
    return W, expected_zs, second_moment_zs



def batch_vec(x):
    return x.swapaxes(-1,-2).reshape(-1, x.shape[-1]**2)

def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1,n,n).swapaxes(-1,-2)

import matplotlib.pyplot as plt

def EM(experiment_dataset, mean_estimate, W_initial, W_prior, EM_iter = 20):

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
    for iter_i in range(EM_iter):
        W, expected_zs, second_moment_zs = EM_step(experiment_dataset, mean_estimate, W, batch_size, W_prior)

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
        print(f"Done with EM step {iter_i}")
        
    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    return U, S**2, W, expected_zs, second_moment_zs


