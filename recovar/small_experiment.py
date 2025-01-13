import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax.experimental import sparse
from recovar import core, simulator, fourier_transform_utils, mask, utils, covariance_estimation
import logging
logger = logging.getLogger(__name__)

def indices_to_coo(grid_indices, n, data= None):
    if data is None:
        data = np.ones(grid_indices.size)
    image_index = np.arange(grid_indices.shape[0])
    image_index = np.repeat(image_index, grid_indices.shape[1])
    grid_indices = grid_indices.flatten()
    return sparse.BCOO((data.flatten(), np.array([image_index, grid_indices]).T), shape=(grid_indices.shape[0], n))


def subsample_coo_columns(sparse_mat, right_indices):
    good_indices = jnp.where(jnp.isin(sparse_mat.indices[:,1], right_indices))[0]
    new_indices = sparse_mat.indices[good_indices,1]

    # Find positions of new_indices inside right_indices
    for new_col, old_col in enumerate(right_indices):
        new_indices = new_indices.at[new_indices == old_col].set(new_col)
        # import pdb; pdb.set_trace()
    # Update column indices
    # new_indices = sparse_mat.indices[good_indices].at[:,1].set(mapped_indices)
    new_indices_all = sparse_mat.indices[good_indices]
    new_indices_all = new_indices_all.at[:,1].set(new_indices)

    return sparse.BCOO((sparse_mat.data[good_indices], new_indices_all), shape=(sparse_mat.shape[0], right_indices.size))

def covar_estimate_sparse(y, P, noise_variance, right_indices, covar_regularization = None):

    # import pdb; pdb.set_trace() 
    y_sliced = subsample_coo_columns(y, right_indices).todense()

    P_sliced = subsample_coo_columns(P, right_indices).todense()

    rhs = (y.T @ y_sliced)
    lhs = (P.T @ P_sliced)
    noise_variance_term = jnp.diag(jnp.diag(lhs[right_indices]))
    # Substract noise variance
    rhs = rhs.at[right_indices].add(-noise_variance * noise_variance_term)

    return rhs, lhs


batch_outer = jax.vmap(jnp.outer, in_axes=(0,0))
batch_kron = jax.vmap(jnp.kron, in_axes = (0,0))
batch_diag = jax.vmap(jnp.diag)

def vec(X):
    return X.T.reshape(-1)

## Inverse of vec function.
def unvec(x):
    n = np.sqrt(x.size).astype(int)
    return x.reshape(n,n).T




def covar_estimate_batch_sparse_U(y, indices, U, noise_variance, regularization =None, chunk_size=1024):

    num_batches = y.shape[0]
    lhs_acc = 0.0
    rhs_acc = 0.0
    utils.report_memory_device(logger=logger)
    # Process the data in chunks to avoid exceeding GPU memory limits
    for batch_idx in utils.index_batch_iter(y.shape[0], chunk_size):
    
        y_chunk = to_gpu(y[np.array(batch_idx)])
        indices_chunk = to_gpu(indices[np.array(batch_idx)])
        AUs = batch_slice(U, indices_chunk)


        AU_t_images = covariance_estimation.batch_x_T_y(AUs, y_chunk)
        AU_t_AU = covariance_estimation.batch_x_T_y(AUs,AUs)
        AUs *= jnp.sqrt(noise_variance)[...,None]
        UALambdaAUs = jnp.sum(covariance_estimation.batch_x_T_y(AUs,AUs), axis=0)

        outer_products = covariance_estimation.summed_outer_products(AU_t_images)

        rhs = outer_products - UALambdaAUs
        lhs = jnp.sum(batch_kron(AU_t_AU, AU_t_AU), axis=0)

        # # Accumulate results
        lhs_acc = lhs_acc + lhs
        rhs_acc = rhs_acc + vec(rhs)

    if regularization is not None:
        lhs_acc += jnp.eye(lhs_acc.shape[0]) * regularization
    utils.report_memory_device(logger=logger)
    # Solve for covariance after processing all chunks
    covar = jnp.linalg.solve(lhs_acc, rhs_acc)
    covar = unvec(covar)
    return covar

from recovar import core, simulator, fourier_transform_utils, mask, utils
ftu = fourier_transform_utils.fourier_transform_utils()
from jax import device_put

@jax.jit
def slice_array(volume_vec, plane_indices_on_grid):
    return volume_vec.at[plane_indices_on_grid].get(mode="fill", fill_value=0)

# Used to project the mean
batch_slice_array = jax.vmap(slice_array, (None, 0))
batch_slice = jax.vmap(batch_slice_array, in_axes = (1, None), out_axes = -1)
                                                                                              
def make_random_sampling_scheme(grid_size, m, seed = 0):
    image_rads = ftu.get_grid_of_radial_distances((grid_size, grid_size)).flatten()
    volume_rads = ftu.get_grid_of_radial_distances((grid_size, grid_size, grid_size)).flatten()
    # running_idx = 0
    # indices = np.zeros((m, grid_size*grid_size), dtype=np.int32)  
    # for rad in range(0, 2*grid_size):
    #     num_points = np.sum(image_rads == rad)
    #     indices_at_rad = np.flatnonzero(volume_rads == rad)
    #     sampled_indices = np.random.choice(indices_at_rad, size=m * num_points, replace=False)
    #     indices[:, running_idx:running_idx + num_points] = sampled_indices.reshape(m, num_points)
    #     running_idx += num_points

    key = jax.random.PRNGKey(seed)
    sampled_indices_list = []
    for rad in jnp.unique(image_rads):
        num_points = jnp.sum(image_rads == rad)
        indices_at_rad = jnp.where(volume_rads == rad)[0]
        total_samples = m * num_points

        key, subkey = jax.random.split(key)
        # if total_samples > indices_at_rad.size:
            # Sample with replacement when total_samples > indices_at_rad.size
        sampled_indices_flat = jax.random.choice(subkey, indices_at_rad, shape=(total_samples,), replace=True)
        # else:
        #     # Sample without replacement
        #     shuffled_indices = jax.random.permutation(subkey, indices_at_rad)
        #     sampled_indices_flat = shuffled_indices[:total_samples]
        sampled_indices = sampled_indices_flat.reshape(m, num_points)
        sampled_indices_list.append(sampled_indices)
        # import pdb; pdb.set_trace()

    indices = jnp.concatenate(sampled_indices_list, axis=1)

    return indices



        # image_rads[image_rads == rad] = rad / (2*grid_size)
        # volume_rads[volume_rads == rad] = rad / (2*grid_size)



def generate_cryo_like_experiment(grid_size, m, b, snr, eig_decay = 0.75, random_sampling = False, voxel_size = 4, Bfactor = 60, return_mat = False):

    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = np.prod(volume_shape)
    # rotation_matrices = simulator.uniform_rotation_sampling(m, grid_size, seed = 0 )
    # indices = core.get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape)

    radial_distances = ftu.get_grid_of_radial_distances(volume_shape).reshape(-1)
    signal_decay_power = 2
    U = np.random.randn(volume_size,b) * ((1.0 /(1+radial_distances[:,None]) ) ** signal_decay_power)
    from recovar import simulator
    B_fac = simulator.get_B_factor_scaling(volume_shape, voxel_size, B_factor = Bfactor)
    U *= B_fac[...,None]

    radial_mask = mask.get_radial_mask(volume_shape).reshape(-1)
    U *= radial_mask[...,None]
    U,_ = np.linalg.qr(U)
    eigs = (eig_decay)**np.arange(b)
    noise_variance = eigs[0] / snr

    batch_size = 1024
    y_list = []
    indices_list = []
    for i in range(0, m, batch_size):
        m_batch = min(batch_size, m - i)
        if random_sampling == "cryoemlike":
            indices_batch = make_random_sampling_scheme(grid_size, m_batch, seed = i)
        elif random_sampling == "cryoem":
            rotation_matrices_batch = simulator.uniform_rotation_sampling(m_batch, grid_size, seed=i)
            indices_batch = core.get_nearest_gridpoint_indices(rotation_matrices_batch, image_shape, volume_shape)
        elif random_sampling == "full":
            indices_batch = np.arange(volume_size).reshape(1, -1).repeat(m_batch, axis=0)
        elif isinstance(random_sampling, float):
            samples_per_image = int(random_sampling * volume_size)
            key = jax.random.PRNGKey(i)
            sampled_indices_batch = []
            for _ in range(m_batch):
                key, subkey = jax.random.split(key)
                sampled_indices = jax.random.choice(subkey, jnp.arange(volume_size), shape=(samples_per_image,), replace=False)
                sampled_indices_batch.append(sampled_indices)
            indices_batch = jnp.stack(sampled_indices_batch)

            # total_samples = int(random_sampling * volume_size)
            
            # key = jax.random.PRNGKey(0)
            # key, subkey = jax.random.split(key)
            # sampled_indices_flat = jax.random.choice(subkey, jnp.arange(volume_size), shape=(total_samples,), replace=False)
            # indices_batch = sampled_indices_flat.reshape(m_batch, -1)

        Z_batch = np.sqrt(eigs)[..., None] * np.random.randn(b, m_batch)
        Z_batch = Z_batch.T[:, :, None]
        Pu_batch = batch_slice(U, indices_batch)
        y_batch = Pu_batch @ Z_batch
        y_batch = y_batch[..., 0]
        y_batch += np.random.randn(*y_batch.shape) * np.sqrt(noise_variance)
        y_list.append(to_cpu(y_batch))
        indices_list.append(to_cpu(indices_batch))

    if return_mat:
        mat = (U @ Z_batch[...,0].T).T
        return U, eigs, y_batch, indices_batch, noise_variance, mat

    y = np.concatenate(y_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)

    logger.info('done with sim')
    utils.report_memory_device( logger=logger)

    return U, eigs, y, indices, noise_variance

from jax import device_put
def to_cpu(x):
    return device_put(x, jax.devices('cpu')[0])

def to_gpu(x):
    return device_put(x, jax.devices('gpu')[0])


def covar_estimate_again(y, indices, noise_variance, n, right_indices, covar_regularization = 1e-6, chunk_size = 1024):
    # Process y and indices in batches
    batch_size = chunk_size  # Adjust batch size as needed

    rhs_acc = 0.0
    lhs_acc = 0.0

    for i in range(0, y.shape[0], batch_size):
        y_batch = to_gpu(y[i:i + batch_size])
        indices_batch = to_gpu(indices[i:i + batch_size])

        y_mat_batch = indices_to_coo(indices_batch, n, y_batch)
        p_batch = indices_to_coo(indices_batch, n)

        rhs_batch, lhs_batch = covar_estimate_sparse(
            y_mat_batch, p_batch, noise_variance, right_indices, covar_regularization
        )

        rhs_acc += rhs_batch
        lhs_acc += lhs_batch

    covar = rhs_acc / (lhs_acc + covar_regularization )
    return covar


# covar_estimate(y, P, noise_covar, diag = False)
def high_d_PCA_by_nystrom_covar(y, indices, noise_variance, covar_indices, n,  eig_threshold = 1e-4, rank_threshold = None, covar_regularization = None, chunk_size = 1024):

    
    if type(covar_indices) == int:
        covar_indices = np.random.permutation(np.arange(n))[:covar_indices]

    C = covar_estimate_again(y, indices, noise_variance, n, covar_indices, covar_regularization = covar_regularization, chunk_size = chunk_size)
    logger.info('done with C')
    utils.report_memory_device( logger=logger)

    W = C[covar_indices]

    sw,uw = np.linalg.eigh(W)
    sw = np.flip(sw)
    uw = np.fliplr(uw)
    good_indices = (sw >=  eig_threshold * sw[0])
    # To shut up the warning
    sw = np.where(good_indices, sw, 1 )
    sw_sqrt = np.where(good_indices, 1/np.sqrt(sw), 0 )
    if rank_threshold is not None:
        sw_sqrt[rank_threshold:] = 0

    Nys_mat = C @ (uw * sw_sqrt @ uw.T)
    U, S, _ = jnp.linalg.svd(Nys_mat, full_matrices= False)
    logger.info('done with Nys')
    utils.report_memory_device( logger=logger)
    return to_cpu(U), S**2


def high_d_PCA_by_projected_covar(y, indices, noise_variance, covar_indices, n, covar_regularization = None, chunk_size = 1024):

    if type(covar_indices) == int:
        covar_indices = np.random.permutation(np.arange(n)[:covar_indices])
                                              
    C = covar_estimate_again(y, indices, noise_variance, n, covar_indices, covar_regularization = covar_regularization, chunk_size = chunk_size)
    logger.info('done with C')
    utils.report_memory_device( logger=logger)

    Q,_ = jnp.linalg.qr(C)
    covar  = covar_estimate_batch_sparse_U(y, indices, Q, noise_variance, chunk_size=chunk_size, regularization = None)
    logger.info('done with projected covar')
    utils.report_memory_device( logger=logger)

    eigs,small_us = np.linalg.eigh(covar)

    eigs = np.flip(eigs)
    small_us = np.fliplr(small_us)
    U2 = Q@ small_us
    logger.info('done with all')
    utils.report_memory_device( logger=logger)

    return U2, eigs 


def high_d_PCA_by_low_rank_completion(y, indices, n, mu = .8):
    # Restriction operator
    # sub = 0.4
    # nsub = int(ny*nx*sub)
    # nsub
    # iava = np.random.permutation(np.arange(ny*nx))[:nsub]
    # n = U.shape[0]
    mat_shape = (n, y.shape[0])
    mat_size = np.prod(mat_shape)
    coo = indices_to_coo(indices, n, data= None)
    #Flattened coo.indices
    coo_ind = coo.indices[:,0] * mat_shape[0] + coo.indices[:,1]
    del coo
    # coo.indices = coo.indices[:,0] * y.shape[1] + coo.indices
    import pylops, pyproximal
    Rop = pylops.Restriction(mat_size, coo_ind)

    f = pyproximal.L2(Rop, y.T.flatten())
    g = pyproximal.Nuclear(mat_shape, mu)

    Xpg = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(mat_size), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=True)
    Xpg = Xpg.reshape(*mat_shape)

    # Recompute SVD and see how the singular values look like
    Upg, Spg, _ = np.linalg.svd(Xpg, full_matrices=False)

    return Upg, Spg**2 / y.shape[0], Xpg

def simulate_C(U, eigs, indices, noise_variance, volume_shape, decay=1):
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape).reshape(-1) ** decay
    U = to_gpu(U)
    covar = (U * eigs ) @ U[indices].T
    noise_variance_decay = radial_distances @ radial_distances[indices].T * noise_variance
    covar += np.random.randn(*covar.shape) * np.sqrt(noise_variance_decay)
    
    return covar


