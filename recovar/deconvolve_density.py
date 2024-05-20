from recovar import latent_density
import numpy as np
import jax, jaxopt
from jaxopt import ScipyBoundedMinimize
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm

def get_raw_density(pipeline_output, zdim = 10, pca_dim_max = 5, percentile_reject = 10, num_points = 50):

    zs = pipeline_output.get('zs')[zdim]
    cov_zs =  pipeline_output.get('cov_zs')[zdim]

    cov_zs_norm = np.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm >np.percentile(cov_zs_norm, percentile_reject)
    zdim = pca_dim_max
    zs = zs[good_zs][:,:zdim]
    cov_zs = cov_zs[good_zs][:,:zdim,:zdim]
    gauss_kde = jax.scipy.stats.gaussian_kde(zs.T, 'silverman')
    covar_data = np.mean(jnp.linalg.inv(cov_zs), axis=0)
    total_covar = covar_data + gauss_kde.covariance

    density, bds = latent_density.compute_latent_space_density_kde(zs, pca_dim_max = pca_dim_max, num_points = num_points, percentile = 0.1)
    grids_flat = latent_density.make_latent_space_grid_from_bounds(bds, num_points)
    grids = grids_flat.reshape(*density.shape, grids_flat.shape[-1])

    return density, total_covar, grids, bds

def get_deconvolved_density(pipeline_output, zdim = 10, pca_dim_max = 4, percentile_reject = 10, num_points = 50):
    density, total_covar, grids, bounds = get_raw_density(pipeline_output, zdim = zdim, pca_dim_max = pca_dim_max, percentile_reject = percentile_reject, num_points = num_points)
    lbfgsb_sols, cost, reg_cost, alphas = compute_deconvolved_density(density, total_covar, grids)
    return lbfgsb_sols, alphas, cost, reg_cost, density, total_covar, grids, bounds

def compute_deconvolved_density( density, total_covar, grids):

    def compute_kernel_on_grid_nd(grids_inp):
        grid_size = jnp.max(grids_inp, axis = np.arange(grids_inp.ndim-1))  - jnp.min(grids_inp, axis = np.arange(grids_inp.ndim-1)) 
        coord_pca_1D = []
        num_points = grids_inp.shape[0]
        # print(grids.shape)
        # FIND BOUNDS ON SPACE TO DISCRETIZE
        
        pca_dim_max = grids_inp.shape[-1]
        for pca_dim in range(pca_dim_max):
            coord_pca = jnp.flip(jnp.linspace(- grid_size[pca_dim]/2, grid_size[pca_dim]/2, num_points, endpoint = False))
            # coord_pca = np.linspace(latent_space_bounds[pca_dim][0], latent_space_bounds[pca_dim][1], num_points)
            coord_pca_1D.append(coord_pca)
        grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
        grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 
        kernel_on_grid = jax.scipy.stats.multivariate_normal.pdf(grids_flat, np.zeros(total_covar.shape[0]), total_covar)
        kernel_on_grid = kernel_on_grid/jnp.sum(kernel_on_grid)
        return kernel_on_grid.reshape(grids_inp.shape[:-1])

    kernel_on_grid = compute_kernel_on_grid_nd(grids).astype(np.float32)

    density = density.astype(np.float32) / np.mean(density) #*circ_mask
    def forward_model_grid(fun_on_grid):
        convolve_fun = convolve_with_pad_nd(fun_on_grid, kernel_on_grid)
        return convolve_fun

    def ridge_reg_objective_grid(fun_on_grid, alpha = 0.0):
        # fun_on_grid = circ_mask * fun_on_grid
        residuals = forward_model_grid(fun_on_grid) - density #/ jnp.sum(y)
        
        if fun_on_grid.ndim ==2:
            dx = grids[1,1,:] - grids[0,0,:] 
        elif fun_on_grid.ndim ==3:
            dx = grids[1,1,1,:] - grids[0,0,0,:] 
        elif fun_on_grid.ndim ==4:
            dx = grids[1,1,1,1,:] - grids[0,0,0,0,:] 
        elif fun_on_grid.ndim ==5:
            dx = grids[1,1,1,1,1,:] - grids[0,0,0,0,0,:] 
        else:
            assert False

        dx/= jnp.mean(dx)
        return jnp.mean((residuals * 1e4) ** 2)  + alpha * jnp.linalg.norm(jnp.array(jnp.gradient(fun_on_grid, *dx)))**2

    alphas = np.flip(np.logspace(-3, 2, 5))
    cost = np.zeros_like(alphas)
    reg_cost = np.zeros_like(alphas)
    lbfgsb_sols = []
    for alpha_idx, alpha in enumerate(alphas):
        w_init = density# * 0 +1
        lbfgsb = ScipyBoundedMinimize(fun=ridge_reg_objective_grid, method="l-bfgs-b", maxiter = 5000)
        lower_bounds = jnp.zeros_like(w_init)
        upper_bounds = jnp.ones_like(w_init) * jnp.inf
        bounds = (lower_bounds, upper_bounds)
        lbfgsb_sol_p = lbfgsb.run(w_init, alpha = alpha, bounds=bounds )
        lbfgsb_sol = lbfgsb_sol_p.params
        cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha = 0)
        reg_cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha = alpha)
        lbfgsb_sols.append(lbfgsb_sol)
        print(alpha_idx)

    return lbfgsb_sols, cost, reg_cost, alphas

def plot_density(lbfgsb_sols, density, alphas):
    # plt.axis('square');
    from recovar.output import sum_over_other


    def plot_dens(density, title):
        if density.ndim ==2:
            plt.figure()
            plt.title(title)
            plt.imshow(density)#.sum(axis=-1))
            plt.show()
        else:
            for k in range(1, density.ndim):
                to_plot = sum_over_other(density, [0,k])
                plt.figure()
                plt.title(title)
                plt.imshow(to_plot)#.sum(axis=-1))
                plt.show()
            to_plot = sum_over_other(density, [0,2])
            plt.figure()
            plt.title(title)
            plt.imshow(to_plot)#.sum(axis=-1))
            plt.show()

    plot_dens(density, 'raw density')

    for alpha_idx, alpha in enumerate(alphas):
        lbfgsb_sol = lbfgsb_sols[alpha_idx]
        plot_dens(lbfgsb_sol, f'deconvolved density alpha = {alpha}')
                    
def convolve_with_pad_nd(ar1, ar2):
    full_shape = tuple(2*np.array(ar1.shape)-1)
    if ar1.ndim <=3:
        fft, ifft = jnp.fft.fftn, jnp.fft.ifftn
    else:
        fft, ifft = fftn, ifftn
    conv = ifft(fft(ar1, full_shape)* fft(ar2, full_shape)).real
    return _centered(conv, ar1.shape)

def convolve_with_pad_nd_np(ar1, ar2):
    full_shape = tuple(2*np.array(ar1.shape)-1)
    fft, ifft = np.fft.fftn, np.fft.ifftn
    
    conv = ifft(fft(ar1, full_shape)* fft(ar2, full_shape)).real
    return _centered(conv, ar1.shape)


# @partial(jax.jit, static_argnames=['newshape'])
def _centered(arr, newshape):
    startind = [(s1 - s2) // 2 for s1, s2 in zip(arr.shape, newshape)]
    return jax.lax.dynamic_slice(arr, startind, newshape)


def ifftn(arr, s= None, axes = None):
    axes = np.arange(arr.ndim)
    s = arr.shape if s is None else s
    assert len(axes) <= 6, "only implemented up to dim 6"
    if len(axes) <= 3:
        arr = jnp.fft.ifftn(arr, s, axes)
    else:
        arr = jnp.fft.ifftn(arr, s[:3], axes[:3])
        arr = jnp.fft.ifftn(arr, s[3:], axes[3:])
    return arr


def fftn(arr, s= None, axes = None):
    axes = np.arange(arr.ndim)
    s = arr.shape if s is None else s
    assert len(axes) <= 6, "only implemented up to dim 6"
    if len(axes) <= 3:
        arr = jnp.fft.fftn(arr, s, axes)
    else:
        arr = jnp.fft.fftn(arr, s[:3], axes[:3])
        arr = jnp.fft.fftn(arr, s[3:], axes[3:])
    return arr 

