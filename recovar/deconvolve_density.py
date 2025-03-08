from recovar import latent_density
import numpy as np
import jax, jaxopt
from jaxopt import ScipyBoundedMinimize
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm

def get_raw_density(pipeline_output, zdim = 10, pca_dim_max = 5, percentile_reject = 10, num_points = 50, percentile_bound = 0.1):

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

    density, bds = latent_density.compute_latent_space_density_kde(zs, pca_dim_max = pca_dim_max, num_points = num_points, percentile = percentile_bound)
    grids_flat = latent_density.make_latent_space_grid_from_bounds(bds, num_points)
    grids = grids_flat.reshape(*density.shape, grids_flat.shape[-1])

    kernel = estimate_kernel_by_sampling(grids, cov_zs, gauss_kde.covariance, num_samples = 5000)

    return density, kernel, total_covar, grids, bds

def get_deconvolved_density(pipeline_output, zdim = '4_noreg', pca_dim_max = 4, percentile_reject = 10, num_points = 50, kernel_option = 'sampling', alphas = None, percentile_bound = 1, save_to_file = None):
    alphas = np.flip(np.logspace(-4, 1, 10)) if alphas is None else alphas

    density, kernel, total_covar, grids, bounds = get_raw_density(pipeline_output, zdim = zdim, pca_dim_max = pca_dim_max, percentile_reject = percentile_reject, num_points = num_points, percentile_bound=percentile_bound)
    lbfgsb_sols, cost, reg_cost, alphas = compute_deconvolved_density(density, kernel, total_covar, grids, kernel_option = kernel_option, alphas = alphas)
    
    if save_to_file is not None:
        from recovar import utils
        utils.pickle_dump({'deconv_densities': lbfgsb_sols, 'alphas': alphas, 'cost': cost, 'reg_cost': reg_cost,  'density': density, 'kernel': kernel, 'total_covar': total_covar, 'grids': grids, 'bounds': bounds, 'zdim': zdim}, save_to_file)


    return lbfgsb_sols, alphas, cost, reg_cost, density, total_covar, grids, bounds


def estimate_kernel_by_sampling(grids_inp, cov_zs, gauss_kde_covariance,  num_samples = 5000):

    grid_size = jnp.max(grids_inp, axis = np.arange(grids_inp.ndim-1))  - jnp.min(grids_inp, axis = np.arange(grids_inp.ndim-1)) 
    coord_pca_1D = []
    num_points = grids_inp.shape[0]
    
    pca_dim_max = grids_inp.shape[-1]
    for pca_dim in range(pca_dim_max):
        coord_pca = jnp.flip(jnp.linspace(- grid_size[pca_dim]/2, grid_size[pca_dim]/2, num_points, endpoint = False))
        coord_pca_1D.append(coord_pca)
    grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
    grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 

    kernel_on_grid =0
    for k in range(num_samples):
        idx = np.random.choice(cov_zs.shape[0], 1)[0]
        covar_data = jnp.linalg.pinv(cov_zs[idx])
        total_covar = covar_data + gauss_kde_covariance
        kernel_on_grid += jax.scipy.stats.multivariate_normal.pdf(grids_flat, np.zeros(total_covar.shape[0]), total_covar)

    kernel_on_grid = kernel_on_grid/jnp.sum(kernel_on_grid)

    return kernel_on_grid.reshape(grids_inp.shape[:-1])

def compute_deconvolved_density( density, kernel, total_covar, grids, kernel_option = 'sampling', alphas = None):
    alphas = np.flip(np.logspace(-3, 2, 5)) if alphas is None else alphas

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
    
    if kernel_option == 'sampling':
        kernel_on_grid = kernel
    elif kernel_option == 'avg_cov':
        kernel_on_grid = compute_kernel_on_grid_nd(grids).astype(np.float32)
    else:
        NotImplementedError()

    # kernel_on_grid = kernel #compute_kernel_on_grid_nd(grids).astype(np.float32)

    density = density.astype(np.float32) / np.mean(density) #*circ_mask

    density = jnp.array(density)
    kernel_on_grid = jnp.array(kernel_on_grid)
    # from recovar import utils
    # utils.pickle_dump(kernel_on_grid, '/home/mg6942/kernel_on_grid.pkl')
    # import pdb; pdb.set_trace()
    def forward_model_grid(fun_on_grid):
        convolve_fun = convolve_with_pad_nd(fun_on_grid, kernel_on_grid)
        return convolve_fun
    
    @jax.jit
    def ridge_reg_objective_grid(fun_on_grid, alpha = 0.0):
        # fun_on_grid = circ_mask * fun_on_grid
        residuals = forward_model_grid(fun_on_grid) - density #/ jnp.sum(y)
        ## YIKES
        if fun_on_grid.ndim ==1:
            dx = grids[1,:] - grids[0,:] 
        elif fun_on_grid.ndim ==2:
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

        return 1e8 * (jnp.mean((residuals * 1e0) ** 2)  +  alpha * jnp.mean(jnp.array(jnp.gradient(fun_on_grid, *dx))**2 ))

    cost = np.zeros_like(alphas)
    reg_cost = np.zeros_like(alphas)
    lbfgsb_sols = []
    for alpha_idx, alpha in enumerate(alphas):
        # w_init = density# * 0 +1
        w_init = jnp.array(density)# * 0 +1
        use_scipy = True
        if use_scipy:
            lbfgsb = ScipyBoundedMinimize(fun=ridge_reg_objective_grid, method="l-bfgs-b", maxiter = 500)
        else:   
            lbfgsb = jaxopt.LBFGSB(fun=ridge_reg_objective_grid, maxiter = 500)
        # # lbfgsb = ScipyBoundedMinimize(fun=ridge_reg_objective_grid, method="l-bfgs-b", maxiter = 500)

        # lbfgsb = jaxopt.LBFGSB(fun=ridge_reg_objective_grid, maxiter = 500)
        lower_bounds = jnp.zeros_like(w_init)
        upper_bounds = jnp.ones_like(w_init) * jnp.inf
        bounds = (lower_bounds, upper_bounds)

        # lbfgsb.init_state(init_params=w_init, bounds=bounds, alpha = alpha)
        # lbfgsb_sol_p = lbfgsb.run(w_init, alpha = alpha, LB = lower_bounds, UB = upper_bounds)

        lbfgsb_sol_p = lbfgsb.run(init_params=w_init,bounds=bounds, alpha = alpha )#, bounds=bounds)
        # lbfgsb_sol_p = lbfgsb.run(w_init, alpha = alpha, bounds=bounds )
        # print(alpha_idx, alpha, lbfgsb_sol_p.state)
        # import pdb; pdb.set_trace()

        lbfgsb_sol = lbfgsb_sol_p.params
        cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha = 0) / ridge_reg_objective_grid(lbfgsb_sol * 0, alpha = 0) 
        reg_cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha = alpha) / ridge_reg_objective_grid(lbfgsb_sol * 0, alpha = 0)
        lbfgsb_sols.append(np.array(lbfgsb_sol))

    return lbfgsb_sols, cost, reg_cost, alphas

def plot_density(lbfgsb_sols, density, alphas, function = None, cmap = 'inferno'):
    # plt.axis('square');
    from recovar.output import sum_over_other

    def half_slice_other(density, axes):
        axes = [i for i in range(density.ndim) if i not in axes]
        axes = np.sort(axes)
        for i in range(len(axes)-1, -1, -1):
            density = np.take(density, density.shape[0]//2, axis = axes[i])
        return density
    
    function = half_slice_other if function == 'slice' else function
    function = sum_over_other if function is None else function


    plt.rcParams.update({
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    import matplotlib
    matplotlib.rc('font', **font)

    n_plots = len(lbfgsb_sols)+1
    n_cols = density.ndim +1 if density.ndim < 2 else density.ndim 

    fig, axs = plt.subplots( n_plots, n_cols, figsize = ( n_cols *5, n_plots*5 ))#, 6*3))
    global is_first
    is_first = True

    def plot_dens(density, title, n_plot, first = False):
        density = np.asarray(density)
        global is_first
        if density.ndim ==1:
            axs[n_plot,0].plot(density)
            axs[n_plot,0].set_title(title)
            axs[n_plot,0].set_xticklabels([])
            axs[n_plot,0].set_yticklabels([])
            if is_first:            
                axs[n_plot,0].set_title(f"PC x={0}")
            axs[n_plot,0].set_ylabel(title)
            return

        for k in range(1, density.ndim):

            if k ==1:
                axs[n_plot,k-1].set_ylabel(title)
            axs[n_plot,k-1].set_xticklabels([])
            axs[n_plot,k-1].set_yticklabels([])

            to_plot = function(density, [0,k])
            # plt.figure()
            # plt.title(title)
            axs[n_plot,k-1].imshow(to_plot.T, cmap =cmap)#.sum(axis=-1))
            # plt.show()
            if is_first:            
                axs[n_plot,k-1].set_title(f"PC x={0}, y={k}")


        if density.ndim > 2:
            to_plot = function(density, [1,2])
            # plt.figure()
            # plt.title(title)
            axs[n_plot,k].imshow(to_plot.T, cmap =cmap)#.sum(axis=-1))
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:            
                axs[n_plot,k].set_title(f"PC x={1}, y={2}")
        is_first = False
            # plt.show()

    # axs[n_plot,0].set_ylabel(name)

    plot_dens(density, 'raw density', 0)

    for alpha_idx, alpha in enumerate(alphas):
        lbfgsb_sol = lbfgsb_sols[alpha_idx]
        plot_dens(lbfgsb_sol, r'a'+f'[{alpha_idx}]={alpha:.1e}', alpha_idx+1)
    plt.subplots_adjust(wspace=0, hspace=0)



def plot_density_centers(density, centers):
    # plt.axis('square');
    from recovar.output import sum_over_other

    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    import matplotlib
    matplotlib.rc('font', **font)

    n_plots = 1
    fig, axs = plt.subplots( n_plots, density.ndim, figsize = ( density.ndim *10, n_plots*6 ))#, 6*3))
    global is_first
    is_first = True

    def plot_dens(density, title, n_plot):

        # if is_first:            
        #     axs[n_plot,k].set_title(f"projection {k}")

        if density.ndim ==2:
            # plt.figure()
            # plt.title(title)
            axs[n_plot,k].imshow(density)#.sum(axis=-1))
            # plt.show()
        else:

            for k in range(1, density.ndim):
                if k ==1:
                    axs[n_plot,k-1].set_ylabel(title)
                axs[n_plot,k-1].set_xticklabels([])
                axs[n_plot,k-1].set_yticklabels([])

                to_plot = sum_over_other(density, [0,k])
                # plt.figure()
                # plt.title(title)
                axs[n_plot,k-1].imshow(to_plot)#.sum(axis=-1))
                axs[n_plot,k-1].scatter(centers[:,0], centers[:,k] )
                for i in range(centers.shape[0]):
                    axs[n_plot,k-1].annotate(str(i), centers[i, [0,k]] + np.array([0.1, 0.1]))

                # plt.show()
            to_plot = sum_over_other(density, [0,2])
            # plt.figure()
            # plt.title(title)
            axs[n_plot,k].imshow(to_plot)#.sum(axis=-1))
            axs[n_plot,k].scatter(centers[:,0], centers[:,2] )
            for i in range(centers.shape[0]):
                axs[n_plot,k].annotate(str(i), centers[i, [0,2]] + np.array([0.1, 0.1]))

            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            # plt.show()
    # axs[n_plot,0].set_ylabel(name)

    plot_dens(density, 'raw density', 0)
    plt.subplots_adjust(wspace=0, hspace=0)

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

# TODO: I think this was fixed in newer versions of JAX? 
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



def find_local_maxs_of_density(density_deconv, latent_space_bounds, percent_top = 1, n_local_maxs = 3, plot_folder = None):
    ## Attempts to find local maxima of the density by clustering the top percent_top of the density on the grid. 
    ## Then finds the maximum within each cluster.
    ## Not a very good method, but has worked fine for a couple of datasets....

    zz = np.argsort(density_deconv.reshape(-1))
    large_dens_indices_raveled = np.unravel_index(zz, density_deconv.shape)
    n_top_points = int(percent_top/100 * np.prod(density_deconv.shape))
    large_dens_indices = (np.array(large_dens_indices_raveled)[:,-n_top_points:]).T

    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, HDBSCAN
    import matplotlib.pyplot as plt
    # clustering = KMeans(n_clusters=n_local_maxs,  n_init=10).fit(X)


    if n_local_maxs >= 1:
        
        clustering = AgglomerativeClustering(n_clusters = n_local_maxs, linkage = 'complete').fit(large_dens_indices)
        
        if clustering == 'kmeans':
            clustering = KMeans(n_clusters=n_local_maxs,  n_init=10).fit(large_dens_indices)
    else:
        clustering = HDBSCAN().fit(large_dens_indices)



    def find_max_within_clusters(density, labels, indices):
        n_clusters = np.max(labels)+1
        best_indices = []
        for k in range(n_clusters):
            sub_indices = indices[labels==k]
            zz = np.argmax(density[tuple(sub_indices.T)])
            best_indices.append(sub_indices[zz])
                                
        return np.array(best_indices)
    max_within_cluster = find_max_within_clusters(density_deconv, clustering.labels_, large_dens_indices)

    ndim = density_deconv.ndim
    if ndim > 2:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(large_dens_indices[:,0], large_dens_indices[:,1], large_dens_indices[:,2], c = clustering.labels_)
        ax.scatter(max_within_cluster[:,0], max_within_cluster[:,1], max_within_cluster[:,2], s = 300, c = 'k', alpha = 1, marker='x', linewidths=30)
        if plot_folder is not None:
            plt.savefig(plot_folder + 'local_max_viz.png')
            plt.close()

    if ndim == 2:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot()
        ax.scatter(large_dens_indices[:,0], large_dens_indices[:,1],  c = clustering.labels_)
        ax.scatter(max_within_cluster[:,0], max_within_cluster[:,1], s = 300, c = 'k', alpha = 1, marker='x', linewidths=30)
        if plot_folder is not None:
            plt.savefig(plot_folder + 'local_max_viz.png')
            plt.close()


    # plt.figure()
    # plt.scatter(large_dens_indices[:,0], large_dens_indices[:,3], large_dens_indices[:,2], c = kmeans.labels_)

    grid_to_z, _ = latent_density.get_grid_z_mappings(latent_space_bounds, density_deconv.shape[0])
    max_within_cluster_z = grid_to_z(max_within_cluster)

    return max_within_cluster_z, max_within_cluster
