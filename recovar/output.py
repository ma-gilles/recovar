import logging
logger = logging.getLogger(__name__)
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib, pickle, os, json
import recovar.latent_density as ld
import recovar.fourier_transform_utils as fourier_transform_utils
from recovar import embedding
from recovar import linalg
from recovar import trajectory
from recovar import utils
from recovar import dataset
from recovar import regularization
from recovar import noise
import matplotlib.patheffects as pe
from packaging.version import parse as parse_version
import time
import recovar.utils as utils

def get_resampled_distances(gt_vols):
    return trajectory.get_cum_curvelength(gt_vols)

def resample_trajectory(gt_vols, n_vols_along_path = 6):
    distances_between_volumes = get_resampled_distances(gt_vols)
    
    # n_volumes at approximately equispaced points 
    x = np.linspace(0, distances_between_volumes[-1], n_vols_along_path)
    gt_vols_x= np.interp(x, distances_between_volumes, np.arange(gt_vols.shape[0]), left=None, right=None, period=None)
    
    indices_along_path = np.round(gt_vols_x).astype(int)
    return indices_along_path



def mkdir_safe(folder):
    os.makedirs(folder, exist_ok = True)
    
def save_volume(vol, path, volume_shape = None, from_ft = True, voxel_size = None):
    volume_shape = 3*[utils.guess_grid_size_from_vol_size(vol.size)] if volume_shape is None else volume_shape
    if from_ft:
        vol =  np.real(fourier_transform_utils.get_idft3(vol.reshape(volume_shape)))
    else:
        vol = np.real(vol.reshape(volume_shape))
    utils.write_mrc(path + '.mrc', vol.astype(np.float32), voxel_size = voxel_size)
    
def save_volumes(volumes,  save_path , volume_shape = None, from_ft = True, index_offset=0, voxel_size = None):
    grid_size = np.round((volumes[0].shape[0])**(1/3)).astype(int)
    volume_shape = 3*[grid_size] if volume_shape is None else volume_shape
    for v_idx, vol in enumerate(volumes):
        save_volume(vol, save_path + format(index_offset + v_idx, '04d') , volume_shape, from_ft = from_ft, voxel_size = voxel_size)


def plot_on_same_scale(cs, xs, labels,plot_folder, ):
    plt.figure(figsize = (7,5))
    k = 0 

    for curve in cs:
        plt.plot(np.linspace(0,1, curve.size), curve / np.max(curve), label = labels[k], lw = 4)
        k+=1
    plt.legend()
    save_filepath = plot_folder + 'path_density.png'
    #if save_to_file:
    plt.savefig(save_filepath, bbox_inches='tight')


def plot_two_twings_with_diff_scale(cs, xs, labels,plot_folder= None): 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 

    
    # Create some mock data
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel(labels[0], color=color, fontsize = 20)
    x = np.linspace(0,1, cs[0].size) if xs[0] is None else xs[0]/ np.max(xs[0])
    ax1.plot(x, cs[0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, np.max(cs[0]))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(labels[1], color=color, fontsize = 20)  # we already handled the x-label with ax1
    x = np.linspace(0,1, cs[1].size) if xs[1] is None else xs[1] / np.max(xs[1])
    ax2.plot(x, cs[1], color=color)
    ax2.set_ylim(0, np.max(cs[1]))

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if plot_folder is not None:
        save_filepath = plot_folder + 'path_density_t.png'
        #if save_to_file:
        plt.savefig(save_filepath, bbox_inches='tight')
    plt.show()

    
        
def sum_over_other(x, use_axis = [0,1], *args, **kwargs):
    other_axes = []
    for k in range(x.ndim):
        if k not in use_axis:
            other_axes.append(k)
            
    xk = np.sum(x, axis = tuple(other_axes))
    return xk

def half_slice_other(density, axes, *args, **kwargs):
    axes = [i for i in range(density.ndim) if i not in axes]
    axes = np.sort(axes)
    for i in range(len(axes)-1, -1, -1):
        density = np.take(density, density.shape[axes[i]] // 2, axis = axes[i])
    return density

def slice_at_point(density, axes, point, *args, **kwargs):
    axes = [i for i in range(density.ndim) if i not in axes]
    axes = np.sort(axes)
    for i in range(len(axes)-1, -1, -1):
        density = np.take(density, point[axes[i]], axis = axes[i])
    return density


# def plot_trajectories_over_density_from_result(results, trajectories, subsampled, zdim ):
#     latent_space_bounds = ld.compute_latent_space_bounds(results['zs'][zdim])
#     plot_over_density(results['density'], trajectories, latent_space_bounds,  subsampled = subsampled, colors = None, plot_folder = None, cmap = 'inferno', same_st_end = True, zs = results['zs'][zdim], cov_zs = results['cov_zs'][zdim] )
#     return

def plot_over_density(density, trajectories = None, latent_space_bounds = None,  subsampled = None, colors = None, plot_folder = None, cmap = 'inferno', same_st_end = True, zs = None, cov_zs = None, points = None, projection_function = None, annotate = False, slice_point = None):

    colors = ['k', 'cornflowerblue', 'g' , 'r', 'b', 'w', 'c'] if colors is None else colors
    path_exists = trajectories is not None
        

    # assert projection_function is None or slice_point is None, "Can't have both projection function and slice point"
    assert projection_function in ['slice', 'slice_point', 'sum', None], "Unknown projection function"
    projection_function = half_slice_other if projection_function == 'slice' else projection_function
    projection_function = slice_at_point if projection_function == 'slice_point' else projection_function


    projection_function = sum_over_other if projection_function is None else projection_function

    if slice_point is None:
        slice_point = points[0] if points is not None else None

    compute_density = False
    if density is None:
        assert zs is not None
        assert cov_zs is not None 
        compute_density = True
        
    if compute_density:
        num_points = 200
    else:
        num_points= density.shape[0]

    if path_exists:
        assert latent_space_bounds is not None, "Need latent space bounds to plot trajectories"
        grid_to_z, z_to_grid = ld.get_grid_z_mappings(latent_space_bounds, num_points)
    
    def plot_traj_along_axes(axes, points = None ):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize = (8,8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]
        if compute_density:
            density_pl, _= ld.compute_latent_space_density_on_2_axes(zs, cov_zs, axes = axes, num_points = num_points)
        else:
            density_pl = projection_function(density, axes, slice_point)
            
        if axis_x > axis_y:
            density_pl = density_pl.T
            
        ax.imshow((density_pl.T), origin='lower', cmap = cmap, interpolation = 'bilinear')#[...,25,25])
        # plt.colorbar()
        if points is not None:
            points = points.copy()
            # project points in bound:
            # out_of_bounds_points = ((points < 0) * (points > np.array(density.shape)[None])).any(axis=-1)
            # points = np.where(points < 0, 0, points)
            out_of_bounds_points = np.zeros(points.shape[0], dtype = bool)
            for k in range(points.shape[1]):
                out_of_bounds_points = out_of_bounds_points | (points[:,k] > density.shape[k])
                out_of_bounds_points = out_of_bounds_points | (points[:,k] < 0)

                points[:,k] = np.where(points[:,k] > density.shape[k], density.shape[k], points[:,k])
                points[:,k] = np.where(points[:,k] < 0, 0, points[:,k])

            #     in_bound_points[k] = z_to_grid(in_bound_points[k])
            # in_bound_points = np.where(points > (points > np.array(density.shape)[None]), 0, points)
            ax.scatter(points[out_of_bounds_points,axis_x], points[out_of_bounds_points,axis_y], 
                      color = 'red', s = 100, edgecolors= 'black', linewidth=1, alpha=0.8)
            ax.scatter(points[~out_of_bounds_points,axis_x], points[~out_of_bounds_points,axis_y], 
                      color = 'cornflowerblue', s = 100, edgecolors= 'black', linewidth=1, alpha=0.8)
            if annotate:
                for i in range(points.shape[0]):
                    plt.annotate(str(i), points[i, axes] + np.array([0.1, 0.1]), color='white', path_effects=[pe.withStroke(linewidth=4, foreground="black")])

        if path_exists:
            # path_grid = z_to_grid(path)
            for traj_idx, traj in enumerate(trajectories):
                traj = z_to_grid(traj)
                
                plt.plot(traj[:,axis_x], traj[:,axis_y], '-', c='w', linewidth=6)
                plt.plot(traj[:,axis_x], traj[:,axis_y], '--', c=colors[traj_idx], dashes=[3], linewidth=6)
                if subsampled is not None:
                    # subs = subsampled[traj_idx]
                    subs = z_to_grid(subsampled[traj_idx].copy())
                    ax.scatter(subs[:,axis_x], subs[:,axis_y], marker = 'o', c=colors[traj_idx], 
                             edgecolors = 'w', s = 500, zorder=2, linewidth=1)
                
                if not same_st_end or traj_idx ==0:
                    g_st = traj[0]
                    g_end = traj[-1]
                    ax.scatter(g_st[axis_x], g_st[axis_y], marker = '*', c='w', 
                             edgecolors = colors[traj_idx], s = 1800, zorder=2, linewidth=2)
                    ax.scatter(g_end[axis_x], g_end[axis_y], marker = 's', c='w', 
                             edgecolors = colors[traj_idx], s = 600, zorder=2, linewidth=2)
                            
        ax.axis("off")
        if plot_folder is not None:
            save_filepath = plot_folder  + 'density_' + str(axes[0]) + str(axes[1]) + '.png'    
            plt.savefig(save_filepath, bbox_inches='tight')
            

    if density is not None:
        traj_dim = density.ndim
    else:
        traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2], points = points)




def plot_kmeans_over_density(density, centers, plot_folder = None, cmap = 'inferno' ):
    # colors = ['k', 'cornflowerblue'] if colors is None else colors
    
    compute_density = False
        
    
    if compute_density:
        num_points = 200
    else:
        num_points= density.shape[0]
    
    # path_grid = z_to_grid(path_z[:,:low_dim])
    # g_st = z_to_grid(z_st[:low_dim])
    # g_end = z_to_grid(z_end[:low_dim])

    def plot_traj_along_axes(axes, save_to_file= False ):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize = (8,8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]
        if compute_density:
            density_pl, _= ld.compute_latent_space_density_on_2_axes(zs, cov_zs, axes = axes, num_points = num_points)
        else:
            density_pl = sum_over_other(density, axes)
            
        if axis_x > axis_y:
            density_pl = density_pl.T
            
        ax.imshow((density_pl.T), origin='lower', cmap = cmap, interpolation = 'bilinear')#[...,25,25])
        # plt.colorbar()
        
        ax.scatter(centers[:,axis_x], centers[:,axis_y], c='red', edgecolor='black', s=100, zorder=3, linewidth=1)
        for i in range(centers.shape[0]):
            ax.annotate(str(i), centers[i, axes] + np.array([0.1, 0.1]), c = 'white', 
                       fontsize=12, fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        

        ax.axis("off")
            
        if plot_folder is not None:
            save_filepath = plot_folder  + 'density_' + str(axes[0]) + str(axes[1]) + '.png'    
            plt.savefig(save_filepath, bbox_inches='tight')
            
    traj_dim = centers.shape[-1]
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2])




def save_covar_output_volumes(output_folder, mean, u, s, mask, volume_shape,  us_to_save = 50, us_to_var = [4,10,20], voxel_size = None):
     
    mkdir_safe(output_folder + 'volumes/')
    n_available = int(u.shape[-1])
    n_to_save = max(0, min(int(us_to_save), n_available))
    save_volumes([u[..., k] for k in range(n_to_save)], output_folder + 'volumes/' + 'eigen_pos', volume_shape=volume_shape, voxel_size=voxel_size)
    # save_volumes([ -u[...,k] for k in range (us_to_save)], output_folder + 'volumes/' +  'eigen_neg', volume_shape = volume_shape,   voxel_size = voxel_size)
    save_volume(mean, output_folder + 'volumes/' + 'mean', volume_shape = volume_shape,   voxel_size = voxel_size)
    
    grid_size = int(volume_shape[0])
    # 2^24 / grid_size^3: ~1 volume at 256^3, scales up for smaller grids
    vol_batch_size = utils.safe_batch_size((2**24) / (grid_size**3))
    n_svals = int(np.asarray(s['rescaled']).shape[0])
    for n_eigs in us_to_var:
        n_eigs_eff = min(int(n_eigs), n_available, n_svals)
        if n_eigs_eff <= 0:
            continue
        u_real = linalg.batch_idft3(u[..., :n_eigs_eff], volume_shape, vol_batch_size )
        variance_real = utils.estimate_variance(u_real.T, s['rescaled'][:n_eigs_eff])
        save_volume(variance_real, output_folder + 'volumes/' + 'variance' + str(n_eigs), volume_shape, from_ft = False,   voxel_size = voxel_size)

# def kmeans_analysis_from_dict(output_folder, pipeline_output, cryos, likelihood_threshold,  n_clusters = 20, generate_volumes = True, zdim =-1, compute_reproj = False):
#     from recovar import dataset

#     if cryos is None:

#         cryos = pipeline_output.get('dataset')#(results['input_args']) if cryos is None else cryos
#         embedding.set_contrasts_in_cryos(cryos, pipeline_output.get('contrasts')[zdim])

#     return kmeans_analysis(output_folder, cryos, pipeline_output.get('mean'), results['u']['rescaled'], results['zs'][zdim], results['cov_zs'][zdim], results['cov_noise'], likelihood_threshold,  n_clusters = n_clusters, generate_volumes = generate_volumes, compute_reproj = compute_reproj)


def kmeans_analysis(output_folder, zs, n_clusters = 20):

    #key = 'zs12'
    #zs = results[key]
    #cov_zs = results['cov_' + key]
    reorder = zs.shape[1] != 1
    labels, centers = cluster_kmeans(zs, n_clusters, reorder = reorder)
    
    import os
    try:
        os.mkdir(output_folder)
    except:
        pass

    def plot_axes(axes = [0,1]):
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=True, labels=None, alpha=0.1, s=1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'PC_'+str(axes[0]) + str(axes[1])+'.png' )
        
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=False, labels=None, alpha=0.1, s=2)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'PC_'+str(axes[0]) + str(axes[1])+'no_annotate.png' )

    for k in range(1,zs.shape[-1]):
        plot_axes(axes = [0,k])
    if zs.shape[-1] > 2:
        plot_axes(axes = [1,2])
    
    return centers, labels


def move_to_one_folder(path_folder, n_vols, string_name = 'locres_filtered.mrc', new_stringname = 'vol' ):
    mkdir_safe(path_folder + '/all_volumes/')
    output_folder = path_folder + '/all_volumes/'
    import shutil
    for k in range(n_vols):
        input_file = path_folder + "/vol" + format(k, '04d') + '/' + string_name
        output_file = output_folder + "/" + new_stringname + format(k, '04d') + ".mrc"
        shutil.copyfile(input_file, output_file)
    return




def plot_umap(output_folder, zs, centers):
    # import cryodrgn.analysis as cryodrgn_analysis

    def plot_axes(axes = [0,1]):
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=True, labels=None, alpha=0.1, s=1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'kmeans_centers.png' )
        
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=False, labels=None, alpha=0.1, s=2)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'kmeans_centers_no_annotate.png' )

        import seaborn as sns

        g = sns.jointplot(x=zs[:,0], y=zs[:,1], alpha=.1, s=1)
        g.set_axis_labels('UMAP1', 'UMAP2')
        if output_folder is not None:

            plt.savefig(output_folder + 'sns.png' )


        g = sns.jointplot(x=zs[:,0], y=zs[:,1], kind='hex')
        g.set_axis_labels('UMAP1', 'UMAP2')
        if output_folder is not None:
            plt.savefig(output_folder + 'sns_hex.png' )

    plot_axes(axes = [0,1])




def compute_and_save_reweighted(cryos, path_subsampled, zs, cov_zs,  output_folder, B_factor, n_bins = 30, n_min_particles = 100, embedding_option = 'cov_dist', save_all_estimates = False, maskrad_fraction= 20, apply_global_filtering=False, fsc_mask = None, fsc_mask_radius = None, fsc_mask_edgewidth = None):

    #batch_size = 

    if n_min_particles is None:
        n_min_particles =100

    mkdir_safe(output_folder)
    new_volume_generation = True
    if new_volume_generation:
        from recovar import heterogeneity_volume, latent_density
        for k in range(path_subsampled.shape[0]):
            output_folder_this = output_folder + "/vol" + format(k, '04d') + "/"
            mkdir_safe(output_folder_this)
            ndim = zs.shape[-1]
            # n_bins = 30
            latent_points = path_subsampled[k][None]
            np.savetxt(output_folder_this + 'latent_coords.txt', latent_points)

            if embedding_option == 'llh':
                log_likelihoods = latent_density.compute_latent_log_likelihood(latent_points, zs, cov_zs)[...,0]
                heterogeneity_distances = log_likelihoods - np.min(log_likelihoods)
            elif embedding_option == 'cov_dist':
                cov_zs = cov_zs#*0 + np.eye(dim)
                heterogeneity_distances = latent_density.compute_latent_quadratic_forms_in_batch(latent_points, zs, cov_zs)[...,0]
            elif embedding_option == 'dist':
                cov_zs = cov_zs*0 + np.eye(ndim)
                heterogeneity_distances = latent_density.compute_latent_log_likelihood(latent_points, zs, cov_zs)[...,0]
            else:
                raise ValueError("Unknown embed option")


            # latent_points = path_subsampled[k][None]
            # log_likelihoods = latent_density.compute_latent_quadratic_forms_in_batch(latent_points[:,:ndim], zs, cov_zs)[...,0]
            heterogeneity_distances = cryos.split_units_array(heterogeneity_distances)

            from recovar import noise

            locres_maskrad = cryos.grid_size * cryos.voxel_size / maskrad_fraction
            logger.info(f"Mask radius fraction = {maskrad_fraction}. Setting locres_maskrac = locres_sampling = box_size * voxel_size / {maskrad_fraction} = {locres_maskrad:.1f} Angstroms. Using {n_min_particles} particles for template.")
            heterogeneity_volume.make_volumes_kernel_estimate_local(heterogeneity_distances, cryos,  output_folder_this, ndim, n_bins, B_factor, tau = None, n_min_particles = n_min_particles, locres_sampling = locres_maskrad, locres_maskrad = locres_maskrad, locres_edgwidth = 0, upsampling_for_ests = 1, use_mask_ests =False, grid_correct_ests = False, save_all_estimates=save_all_estimates, metric_used= 'locshellmost_likely')

            # Apply global filtering to the generated halfmaps if requested
            if apply_global_filtering:
                from recovar import locres
                import glob
                
                # Find the halfmap files that were just generated
                halfmap_files = glob.glob(output_folder_this + "half*_unfil.mrc")
                if len(halfmap_files) >= 2:
                    # Load the halfmaps
                    halfmap1_path = output_folder_this + "halfmap1.mrc"
                    halfmap2_path = output_folder_this + "halfmap2.mrc"
                    
                    if os.path.exists(halfmap1_path) and os.path.exists(halfmap2_path):
                        halfmap1 = utils.load_mrc(halfmap1_path)
                        halfmap2 = utils.load_mrc(halfmap2_path)
                        
                        # Apply global filtering
                        filtered_combined, fsc, global_resol = locres.filter_maps_with_global_fsc(
                            halfmap1, halfmap2, cryos.voxel_size, fsc_mask = fsc_mask
                        )
                        
                        # Save filtered halfmaps with _filtered suffix
                        filtered_half1_path = output_folder_this + "halfmap1_filtered.mrc"
                        filtered_half2_path = output_folder_this + "halfmap2_filtered.mrc"
                        
                        utils.write_mrc(filtered_half1_path, filtered_combined, voxel_size=cryos.voxel_size)
                        utils.write_mrc(filtered_half2_path, filtered_combined, voxel_size=cryos.voxel_size)
                        
                        logger.info(f"Applied global filtering to volume {k}. Resolution: {global_resol:.2f} Angstroms")
                        logger.info(f"Saved filtered halfmaps: {filtered_half1_path}, {filtered_half2_path}")
                    else:
                        logger.warning(f"Halfmap files not found for volume {k}, skipping global filtering")
                else:
                    logger.warning(f"Not enough halfmap files found for volume {k}, skipping global filtering")

            # filter the volumes with the global FSC
            logger.info(f"Done with volume generation {k} stored in {output_folder_this}")
        move_to_one_folder(output_folder, path_subsampled.shape[0], string_name = 'locres_filtered.mrc', new_stringname = 'vol' )
        move_to_one_folder(output_folder, path_subsampled.shape[0], string_name = 'locres.mrc', new_stringname = 'locres' )
        np.savetxt(output_folder + 'latent_coords.txt', path_subsampled)

    # memory_to_use = utils.get_gpu_memory_total() - path_subsampled.shape[0] * cryos[0].volume_size * 8 / 1e9 * 8
    # assert memory_to_use > 0, "reduce number of volumes computed at once"
    # batch_size = 2 * utils.get_image_batch_size(cryos[0].grid_size, memory_to_use)
    # logger.info(f"batch size in reweighting: {batch_size}")

    # else:
    #     trajectory_prior, halfmaps = embedding.generate_conformation_from_reweighting(cryos, means, noise_variance, zs, cov_zs, path_subsampled, batch_size = batch_size, disc_type = 'linear_interp', likelihood_threshold = likelihood_threshold, recompute_prior = recompute_prior, volume_mask = volume_mask, adaptive=adaptive)
    #     save_volumes(trajectory_prior, output_folder +  'reweight_')
    #     save_volumes(halfmaps[0], output_folder +  'halfmap0_reweight_')
    #     save_volumes(halfmaps[1], output_folder +  'halfmap1_reweight_')

    
# def compute_and_save_volumes_from_z(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, adaptive = False ):
#     mkdir_safe(output_folder)
#     compute_and_save_reweighted(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, adaptive = adaptive)
    
    # if compute_reproj:
    #     n_eigs = zs.shape[1]
    #     trajectory_reproj = embedding.generate_conformation_from_reprojection(path_subsampled, means['combined'], u[:,:n_eigs] )
    #     save_volumes(trajectory_reproj, output_folder +  'reproj_')


    
def plot_loglikelihood_over_scatter(path_subsampled, zs, cov_zs, save_path, likelihood_threshold = 1e-5 ):
    scale_zs = ld.compute_det_cov_xs(cov_zs)
    likelihoods = ld.compute_likelihood_of_latent_given_image(path_subsampled, zs, cov_zs, scale_zs)
    
    likelihoods = ld.compute_latent_quadratic_forms(path_subsampled, zs, cov_zs)
    vmax = np.max(likelihoods)
    vmin = np.max([np.min(likelihoods),  1e-8 *np.max(likelihoods)])
    # print(vmax, vmin)
    plt.ioff()
    for k in range(likelihoods.shape[1]):
        fig, ax = plt.subplots(figsize = (8,8))
        
        # Create hexbin density plot for background
        try:
            ax.hexbin(zs[:,0], zs[:,1], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
        except:
            pass
        
        greater_x = likelihoods[:,k] > vmin
        ax.scatter(zs[~greater_x,0], zs[~greater_x,1], c='black', alpha = 0.3, s = 2, edgecolors='none')  

        scatter = ax.scatter(zs[greater_x,0], zs[greater_x,1], c= likelihoods[greater_x,k], cmap='rainbow', 
                           s = 5, alpha = 0.8, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip = True), 
                           edgecolors='none')

        # Add grid and improve styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        ax.set_xlabel('PC1', fontweight='bold')
        ax.set_ylabel('PC2', fontweight='bold')
        ax.set_title(f'Likelihood Analysis - Component {k}', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Likelihood')

        ax.set_xticks([])
        ax.set_yticks([])
        save_filepath = save_path +  format(k, '04d') + '.png' # output_folder + 'plots/' + 'vol_weights' +str(k) + '.png'
        plt.savefig(save_filepath, bbox_inches='tight', dpi=300)
        if k > 0:
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


def load_results(datadir, option_str = 'from_halfmap'):
    model_folder = datadir +'model' + option_str + '/'
    output_folder = datadir +'output' + option_str + '/'
    with open(model_folder + 'results.pkl', 'rb') as f:
        results = pickle.load( f)
    results['dataset_loader_dict']['compute_ground_truth'] = False
    # dataset_loader = dataset.get_dataset_loader_from_dict(results['dataset_loader_dict'])
    grid_to_z, z_to_grid = ld.get_grid_z_mappings(results['latent_space_bounds'], results['density'].shape[0])
    return results, results['dataset_loader_dict'], grid_to_z, z_to_grid, output_folder #['mean'], results['u'], results['s'], results['volume_mask'], results['dilated_volume_mask'], 


def load_results_new(datadir):
    model_folder = datadir +'model'  + '/'
    output_folder = datadir +'output'  + '/'
    with open(model_folder + 'results.pkl', 'rb') as f:
        results = pickle.load( f)
    # results['dataset_loader_dict']['compute_ground_truth'] = False
    # dataset_loader = dataset.get_dataset_loader_from_dict(results['dataset_loader_dict'])
    # grid_to_z, z_to_grid = ld.get_grid_z_mappings(results['latent_space_bounds'], results['density'].shape[0])
    return results#, results['dataset_loader_dict'], grid_to_z, z_to_grid


class PipelineOutput:
    def __init__(self, result_path):
        self.params = utils.pickle_load(result_path + '/model/params.pkl')
        self.embedding = None
        self.embedding_loaded = False
        self._embedding_key_cache = {}
        self._embedding_halfsets_cache = None
        self.result_path = result_path + '/'
        self.version = str(self.params['version']) if 'version' in self.params else '0'

    def _ensure_embedding_raw_loaded(self):
        if self.embedding is None:
            self.embedding = utils.pickle_load(self.result_path + 'model/' + 'embeddings' + '.pkl')

    def _get_input_args_compat(self):
        if not hasattr(self.params, "get") or "input_args" not in self.params:
            return None
        input_args = self.params["input_args"]
        try:
            return input_args.item()
        except Exception:
            return input_args

    def _use_image_halfsets_for_unshared_tilt_contrast(self):
        input_args = self._get_input_args_compat()
        if input_args is None:
            return False
        if isinstance(input_args, dict):
            tilt_series = bool(input_args.get("tilt_series", False))
            shared = bool(input_args.get("shared_contrast_across_tilts", True))
        else:
            tilt_series = bool(getattr(input_args, "tilt_series", False))
            shared = bool(getattr(input_args, "shared_contrast_across_tilts", True))
        return tilt_series and not shared

    def _get_embedding_halfsets(self):
        if self._embedding_halfsets_cache is not None:
            return self._embedding_halfsets_cache
        if self.version == '0':
            self._embedding_halfsets_cache = (None, None)
            return self._embedding_halfsets_cache
        if self.version == '0.1':
            particle_halfsets = np.concatenate(self.get('halfsets'))
        else:
            particle_halfsets = np.concatenate(self.get('particles_halfsets'))
        image_halfsets = np.concatenate(self.get('halfsets'))
        self._embedding_halfsets_cache = (particle_halfsets, image_halfsets)
        return self._embedding_halfsets_cache

    def get_embedding_keys(self, entry):
        self._ensure_embedding_raw_loaded()
        return list(self.embedding[entry].keys())

    def has_embedding_key(self, entry, key):
        return key in self.get_embedding_keys(entry)

    def get_embedding_component(self, entry, key):
        if self.embedding_loaded:
            return self.embedding[entry][key]
        cache_key = (entry, key)
        if cache_key in self._embedding_key_cache:
            return self._embedding_key_cache[cache_key]

        self._ensure_embedding_raw_loaded()
        if key not in self.embedding[entry]:
            raise KeyError(f"Embedding key {key} not found in {entry}.")

        values = self.embedding[entry][key]
        if self.version != '0':
            particle_halfsets, image_halfsets = self._get_embedding_halfsets()
            if entry == 'contrasts' and self._use_image_halfsets_for_unshared_tilt_contrast():
                values = values[image_halfsets]
            else:
                values = values[particle_halfsets]
        self._embedding_key_cache[cache_key] = values
        return values

    def load_embedding(self):
        self._ensure_embedding_raw_loaded()

        if self.version != '0':
            halfsets, image_halfsets = self._get_embedding_halfsets()

            # embedding_dict = self.embedding
            for entry in self.embedding:
                for key in self.embedding[entry]:
                    # Handling the case where the contrasts are not shared across tilts...
                    if entry == 'contrasts' and self._use_image_halfsets_for_unshared_tilt_contrast():
                        self.embedding[entry][key] = self.embedding[entry][key][image_halfsets]
                    else:
                        self.embedding[entry][key] = self.embedding[entry][key][halfsets]
                # for key in self.embedding[entry]:
                #     self.embedding[entry][key] = self.embedding[entry][key][halfsets]
        self.embedding_loaded = True
        self._embedding_key_cache = {}
        return 

    def _list_saved_eigenvector_indices(self):
        vols_dir = self.result_path + 'output/volumes/'
        if not os.path.isdir(vols_dir):
            return []
        prefix = 'eigen_pos'
        suffix = '.mrc'
        indices = []
        for name in os.listdir(vols_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            num_str = name[len(prefix):-len(suffix)]
            if num_str.isdigit():
                indices.append(int(num_str))
        indices.sort()
        return indices

    def _select_saved_eigenvector_indices(self, n_pcs=50):
        saved_indices = self._list_saved_eigenvector_indices()
        if len(saved_indices) == 0:
            raise ValueError("No eigenvector volumes found in output/volumes (expected files like eigen_pos0000.mrc).")
        if n_pcs is None:
            n_pcs = 50
        n_pcs = int(n_pcs)
        if n_pcs <= 0:
            raise ValueError(f"n_pcs must be positive, got {n_pcs}")
        return saved_indices[: min(n_pcs, len(saved_indices))]

    def get_u_real(self, n_pcs=50):
        selected_indices = self._select_saved_eigenvector_indices(n_pcs)
        out = np.empty([len(selected_indices), *(self.params['volume_shape'])], dtype=np.float32)
        for i, eig_idx in enumerate(selected_indices):
            out[i] = utils.load_mrc(self.result_path + 'output/volumes/' + 'eigen_pos' + format(eig_idx, '04d') + '.mrc')
        return out

    def get_u(self, n_pcs=50):
        selected_indices = self._select_saved_eigenvector_indices(n_pcs)
        vol_size = int(np.prod(self.params['volume_shape']))
        out = np.empty((len(selected_indices), vol_size), dtype=np.complex64)
        for i, eig_idx in enumerate(selected_indices):
            vol = utils.load_mrc(self.result_path + 'output/volumes/' + 'eigen_pos' + format(eig_idx, '04d') + '.mrc')
            out[i] = np.asarray(fourier_transform_utils.get_dft3(vol), dtype=np.complex64).reshape(-1)
        return out

    def get(self, key):

        if key in ['zs', 'cov_zs', 'contrasts', 'zs_cont', 'cov_zs_cont', 'est_contrasts_cont']:
            if not self.embedding_loaded:
                self.load_embedding()
            return self.embedding[key]

        elif key == 'unsorted_embedding':
            return utils.pickle_load(self.result_path + 'model/embeddings.pkl')

        elif key in ('u', 'u_real'):
            return self.get_u_real(50) if key == 'u_real' else self.get_u(50)

        elif key == 'mean':
            return fourier_transform_utils.get_dft3(utils.load_mrc(self.result_path + 'output/volumes/mean.mrc')).reshape(-1)

        elif key == 'mean_halfmaps':
            half1 = fourier_transform_utils.get_dft3(utils.load_mrc(self.result_path + 'output/volumes/mean_half1_unfil.mrc')).reshape(-1)
            half2 = fourier_transform_utils.get_dft3(utils.load_mrc(self.result_path + 'output/volumes/mean_half2_unfil.mrc')).reshape(-1)
            return half1, half2

        elif key == 'image_snr':
            vol_shape = self.get('volume_shape')
            PS = regularization.average_over_shells(np.abs(self.get('mean').reshape(vol_shape)) ** 2, vol_shape)
            return utils.make_radial_image(PS / self.get('noise_var_used'), tuple(vol_shape[:2]))

        elif key == 'image_snr_radial':
            vol_shape = self.get('volume_shape')
            PS = regularization.average_over_shells(np.abs(self.get('mean').reshape(vol_shape)) ** 2, vol_shape)
            return PS / self.get('noise_var_used')

        elif key == 'variance':
            return utils.load_mrc(self.result_path + 'output/volumes/variance10.mrc')
        elif key == 'variance20':
            return utils.load_mrc(self.result_path + 'output/volumes/variance20.mrc')
        elif key == 'focus_mask':
            return utils.load_mrc(self.result_path + 'output/volumes/focus_mask.mrc')
        elif key == 'volume_mask':
            return utils.load_mrc(self.result_path + 'output/volumes/mask.mrc')
        elif key == 'dilated_volume_mask':
            return utils.load_mrc(self.result_path + 'output/volumes/dilated_mask.mrc')
        elif key == 'covariance_cols':
            return utils.pickle_load(self.result_path + 'model/covariance_cols.pkl')

        elif key in ('dataset', 'lazy_dataset'):
            cryos = dataset.load_dataset_from_args(self.get('input_args'), lazy='lazy' in key, ind_split=self.get('halfsets'))
            add_noise_to_loaded_dataset(cryos, self.get('noise_var_used'))
            return cryos

        elif key == 'halfsets':
            return utils.pickle_load(self.result_path + 'model/halfsets.pkl')
        elif key == 'particles_halfsets':
            if self.version == '0.1':
                return utils.pickle_load(self.result_path + 'model/halfsets.pkl')
            else:
                return utils.pickle_load(self.result_path + 'model/particles_halfsets.pkl')

        elif key == 'input_args':
            try:
                return self.params['input_args'].item()
            except Exception:
                return self.params['input_args']

        # Backward compat: fields removed in v0.6 (were always None or redundant)
        elif key == 's_all':
            if 's_all' in self.params:
                return self.params['s_all']
            return {'rescaled': self.params['s'], 'real': None}
        elif key in ('density', 'latent_space_bounds', 'pc_metric', 'contrasts_for_second'):
            return self.params.get(key, None)
        elif key in ('std_image_PS', 'std_masked_image_PS'):
            return self.params.get(key, None)

        elif key in self.params:
            return self.params[key]
        else:
            raise KeyError(f"key '{key}' not found in PipelineOutput")

    def keys(self):
        keys = list(self.params.keys())
        keys += ['zs', 'cov_zs', 'contrasts', 'u', 'u_real', 'mean', 'volume_mask', 'dilated_volume_mask', 'covariance_cols', 'dataset', 'lazy_dataset', 'variance', 'variance20', 'focus_mask', 'image_snr', 'mean_halfmaps', 'halfsets', 'input_args', 'unsorted_embedding']
        return keys


def add_noise_to_loaded_dataset(cryos, noise_variance):
    for cryo in cryos:
        if noise_variance.ndim == 1:
            cryo.set_radial_noise_model(noise_variance)
        else:
            cryo.set_variable_radial_noise_model(noise_variance)


def make_trajectory_plots_from_results(pipeline_output, basis_size, output_folder, cryos = None, z_st = None, z_end = None, gt_volumes= None, n_vols_along_path = 6, plot_llh = False,  input_density = None, latent_space_bounds = None):

    assert (((z_st is not None) and (z_end is not None)) or (gt_volumes is not None)), 'either z_st and z_end should be passed, or gt_volumes'

    if input_density is not None:
        assert latent_space_bounds is not None, 'need latent_space_bounds if providing density'

    latent_space_bounds = ld.compute_latent_space_bounds(pipeline_output.get('zs')[basis_size]) if latent_space_bounds is None else latent_space_bounds

    if cryos is None:
        cryos = pipeline_output.get('dataset') if cryos is None else cryos
        embedding.set_contrasts_in_cryos(cryos, pipeline_output.get('contrasts')[basis_size])

    density = input_density if input_density is not None else pipeline_output.get('density')

    return make_trajectory_plots(
        density, 
        pipeline_output.get('zs')[basis_size], pipeline_output.get('cov_zs')[basis_size], 
        z_st, z_end, latent_space_bounds, output_folder, 
        gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = plot_llh, use_input_density = input_density is not None)


def make_trajectory_plots(density, zs, cov_zs, z_st, z_end, latent_space_bounds, output_folder, gt_volumes= None, n_vols_along_path = 6, plot_llh = False, use_input_density =False):

    latent_space_bounds = ld.compute_latent_space_bounds(zs) if latent_space_bounds is None else latent_space_bounds

    # latent_space_bounds = ld.compute_latent_space_bounds(zs) if latent_space_bounds is None else latent_space_bounds


    st_time = time.time()
    gt_volumes = None    
    basis_size = zs.shape[1]
    if use_input_density and (density.ndim < zs.shape[-1]):
        logger.warning("density dimension is less than zs dimension, truncate zs dimension")
        basis_size = density.ndim

    zs = zs[:,:basis_size]
    cov_zs = cov_zs[:,:basis_size,:basis_size]

    mkdir_safe(output_folder + 'density/')
    if basis_size >1:
        if not use_input_density:
            path_z = trajectory.compute_high_dimensional_path(zs, cov_zs, z_st, z_end, density_low_dim=density,
                                                    density_eps = 1e-5, max_dim = basis_size, percentile_bound = 1, num_points = 50, 
                                                    use_log_density = False)
        else:
            path_z = trajectory.compute_fixed_dimensional_path(z_st, z_end, density, latent_space_bounds, density_eps = 1e-5, debug_plot = False, density_option = "kde", use_log_density = False)

        path_subsampled = trajectory.subsample_path(path_z, n_pts = n_vols_along_path)    

        logger.info(f"after path {time.time() - st_time}")
        #trajectory.subsample_path(path_z, n_pts = n_vols_along_path)
        # plot_over_density(density, None,latent_space_bounds,  colors = None, plot_folder = output_folder + 'density_nopath/', cmap = 'inferno')
        inp_dens = density if use_input_density else None

        if gt_volumes is not None:
            plot_over_density(inp_dens, [gt_volumes_z, path_z], latent_space_bounds, subsampled = [gt_volumes_z[gt_subs_idx][1:-1], path_subsampled[1:-1] ] , colors = ['k', 'cornflowerblue'], plot_folder = output_folder, cmap = 'inferno', zs = zs, cov_zs = cov_zs) 
        else:
            plot_over_density(inp_dens, [path_z],latent_space_bounds,  subsampled = [path_subsampled[1:-1] ] , colors = ['cornflowerblue'], plot_folder = output_folder + 'density/', cmap = 'inferno', same_st_end = False, zs = zs, cov_zs = cov_zs)
    else:
        path_z = np.linspace(z_st, z_end, n_vols_along_path)[...,0]
        path_subsampled = path_z
        path_subsampled = path_subsampled

    st_time = time.time()
    # compute_and_save_reweighted(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, B_factor, n_bins = 30)
    # logger.info(f"vol time {time.time() - st_time}")
    
    if use_input_density:
        # grid_to_z, z_to_grid = ld.get_grid_z_mappings(latent_space_bounds, num_points = density.shape[0])

        # path_grid = z_to_grid(path_z)
        # path_grid_subs = z_to_grid(path_subsampled)

        # import jax.scipy
        density_on_path = density_on_grid(path_z, density, latent_space_bounds) 
        density_on_path_subs = density_on_grid(path_subsampled, density, latent_space_bounds) 

        #jax.scipy.ndimage.map_coordinates(density, path_grid.T, order=1)
        # density_on_path_subs = jax.scipy.ndimage.map_coordinates(density, path_grid_subs.T, order=1)

        # density_on_path = ld.compute_latent_space_density_at_pts(path_z, zs, cov_zs)
    else:
        logger.warning("density on path not computed")
        density_on_path = ld.compute_latent_space_density_at_pts(path_z, zs, cov_zs) + np.nan
        density_on_path_subs = ld.compute_latent_space_density_at_pts(path_subsampled, zs, cov_zs) + np.nan

    densities = { 'density' :  density_on_path.tolist(), 'path' : path_z.tolist(), 'density_subsampled': density_on_path_subs.tolist(), 'path_subsampled' : path_subsampled.tolist(), } 
    json.dump(densities, open(output_folder + '/path.json', 'w'))

    if plot_llh:
        plot_loglikelihood_over_scatter(path_subsampled, zs, cov_zs, save_path = output_folder, likelihood_threshold = None  )

    logger.info(f"after all plots {time.time() - st_time}")
    return path_z, path_subsampled


def density_on_grid(points, density, bounds):
    import jax.scipy
    _, z_to_grid = ld.get_grid_z_mappings(bounds, num_points = density.shape[0])
    path_grid = z_to_grid(points)
    return jax.scipy.ndimage.map_coordinates(density, path_grid.T, order=1)


def vol_to_z(gt_volumes, u, mean, basis_size):
    coords = ((np.conj(u[:,:basis_size].T) @ (gt_volumes - mean).T).T).real
    return coords



def plot_trajectories_over_scatter(trajectories,  subsampled = None, colors = None, plot_folder = None, cmap = 'inferno', same_st_end = True, zs = None, cov_zs = None ):
    colors = ['k', 'cornflowerblue'] if colors is None else colors
    path_exists = trajectories is not None
        
    def plot_traj_along_axes(axes, save_to_file= False ):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize = (8,8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]
        
        # Create hexbin density plot for background
        try:
            ax.hexbin(zs[:,axis_x], zs[:,axis_y], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
        except:
            pass
        
        # Main scatter plot with improved styling
        ax.scatter(zs[:,axis_x], zs[:,axis_y], s=1, alpha=0.6, c='cornflowerblue', edgecolors='none')
        
        if path_exists:
            # path_grid = z_to_grid(path)
            for traj_idx, traj in enumerate(trajectories):
                                
                ax.plot(traj[:,axis_x], traj[:,axis_y], '-o', c=colors[traj_idx], linewidth=3, zorder=3, markersize=4)
                # plt.plot(traj[:,axis_x], traj[:,axis_y], '--', c=colors[traj_idx], dashes=[3], linewidth=6)

                if subsampled is not None:
                    subs = subsampled[traj_idx]
                    if subs is not None:
                        ax.scatter(subs[:,axis_x], subs[:,axis_y], marker = '>', c=colors[traj_idx], 
                                 edgecolors = 'w', s = 200, zorder=3, linewidth=1)

            # for traj_idx, traj in enumerate(trajectories):
                if not same_st_end or traj_idx ==0:
                    g_st = traj[0]
                    g_end = traj[-1]
                    ax.scatter(g_st[axis_x], g_st[axis_y], marker = '*', c='w', 
                             edgecolors = colors[traj_idx], s = 1800, zorder=2, linewidth=2)
                    ax.scatter(g_end[axis_x], g_end[axis_y], marker = 's', c='w', 
                             edgecolors = colors[traj_idx], s = 600, zorder=2, linewidth=2)

        # Add grid and improve styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        ax.set_xlabel(f'PC{axis_x+1}', fontweight='bold')
        ax.set_ylabel(f'PC{axis_y+1}', fontweight='bold')
        ax.set_title(f'PC{axis_x+1} vs PC{axis_y+1}', fontweight='bold')
            
        if plot_folder is not None:
            save_filepath = plot_folder  + 'density_' + str(axes[0]) + str(axes[1]) + '.png'    
            plt.savefig(save_filepath, bbox_inches='tight', dpi=300)
            
    traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    # print(traj_dim)
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2])


def umap_latent_space(zs):
    import umap
    st_time = time.time()
    n_components = np.min([zs.shape[1], 2])
    mapper = umap.UMAP(n_components = n_components).fit(zs)
    # umap.plot.points(mapper) 
    logger.info(f"time to umap: {time.time() - st_time}")
    return mapper



def standard_pipeline_plots(po, zdim_key, output_folder):
    from recovar import plot_utils
    mkdir_safe(output_folder)
    plot_utils.plot_summary_t(po, n_eigs = 10, filename = output_folder + "mean_variance_eigenvolume_plots.png")

    import matplotlib.pyplot as plt
    plt.figure(figsize = (10,10))
    plt.hist(po.get('contrasts')[zdim_key],bins =50)
    plt.xlabel('Contrast')
    plt.ylabel('Number of particles')
    plt.savefig(output_folder + 'contrast_histogram.png')
    plt.title(f'contrast histogram using zdim={zdim_key}')
    plt.close()


    plt.figure(figsize = (8,8))
    plt.semilogy(po.get('s')[:40], '-o')
    plt.xlabel('eigenvalue index')
    plt.ylabel('eigenvalue')
    plt.savefig(output_folder + 'eigenvalues.png')
    plt.close()


    plt.figure(figsize = (8,8))
    ax = plot_utils.plot_mean_fsc(po,None)
    plt.savefig(output_folder + 'mean_fsc.png')



    # Load latent coordinates with robust error handling
    try:
        zs_data = po.get('zs')
        if zs_data is None:
            logger.warning("No latent coordinates found in pipeline output. Skipping PC analysis.")
            return
            
        # Try to get 4D latent space, fallback to available dimensions
        if isinstance(zs_data, dict):
            if 4 in zs_data:
                z = zs_data[4]
            elif len(zs_data) > 0:
                # Use the first available key
                first_key = list(zs_data.keys())[0]
                z = zs_data[first_key]
                logger.info(f"Using latent space with key {first_key} instead of 4")
            else:
                logger.warning("No valid latent coordinates found. Skipping PC analysis.")
                return
        elif isinstance(zs_data, (list, tuple)) and len(zs_data) > 4:
            z = zs_data[4]
        elif isinstance(zs_data, (list, tuple)) and len(zs_data) > 0:
            z = zs_data[0]
            logger.info("Using first available latent space instead of 4D")
        else:
            logger.warning("Unexpected format for latent coordinates. Skipping PC analysis.")
            return
            
        if z is None or not hasattr(z, 'shape'):
            logger.warning("Invalid latent coordinates data. Skipping PC analysis.")
            return
            
        logger.info("Latent space shape: %s", z.shape)
        logger.info("Number of particles: %d", z.shape[0])
        logger.info("Latent dimensions: %d", z.shape[1])
        
        # Validate data quality
        if z.shape[0] < 10:
            logger.warning(f"Too few particles ({z.shape[0]}) for meaningful PC analysis. Skipping.")
            return
            
        if z.shape[1] < 2:
            logger.warning(f"Too few dimensions ({z.shape[1]}) for PC analysis. Skipping.")
            return
            
        # Check for NaN or infinite values
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            logger.warning("Latent coordinates contain NaN or infinite values. Attempting to clean data.")
            z = z[~np.any(np.isnan(z) | np.isinf(z), axis=1)]
            if z.shape[0] < 10:
                logger.warning("Too few valid particles after cleaning. Skipping PC analysis.")
                return
                
    except Exception as e:
        logger.error(f"Error loading latent coordinates: {e}")
        return

    # Determine number of PCs to plot based on available dimensions
    max_pcs = min(4, z.shape[1])
    if max_pcs < 2:
        logger.warning(f"Only {max_pcs} dimensions available, cannot create pairwise plots.")
        return
        
    # Calculate number of subplots needed
    n_combinations = max_pcs * (max_pcs - 1) // 2
    if n_combinations == 0:
        logger.warning("No valid PC combinations to plot.")
        return
        
    # Create appropriate subplot layout
    if n_combinations <= 3:
        n_rows, n_cols = 1, n_combinations
    elif n_combinations <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
        
    try:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_combinations == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    except Exception as e:
        logger.error(f"Error creating subplots: {e}")
        return

    # Generate pairwise combinations
    combinations = [(i, j) for i in range(max_pcs) for j in range(i+1, max_pcs)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(combinations)))

    for idx, (i, j) in enumerate(combinations):
        if idx >= len(axes):
            break
            
        try:
            # Validate data for this combination
            x_data = z[:, i]
            y_data = z[:, j]
            
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                logger.warning(f"Skipping PC{i+1} vs PC{j+1} due to NaN values")
                continue
                
            if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
                logger.warning(f"Skipping PC{i+1} vs PC{j+1} due to infinite values")
                continue
            
            # Scatter plot
            axes[idx].scatter(x_data, y_data, alpha=0.6, s=1, c=colors[idx])
            axes[idx].set_xlabel(f'PC{i+1}')
            axes[idx].set_ylabel(f'PC{j+1}')
            axes[idx].set_title(f'PC{i+1} vs PC{j+1}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add density contours with error handling
            try:
                if len(x_data) > 50:  # Only add hexbin for sufficient data points
                    axes[idx].hexbin(x_data, y_data, gridsize=min(30, len(x_data)//10), 
                                   alpha=0.3, cmap='Blues', mincnt=1)
            except Exception as e:
                logger.debug(f"Could not add density contours for PC{i+1} vs PC{j+1}: {e}")
                
        except Exception as e:
            logger.error(f"Error plotting PC{i+1} vs PC{j+1}: {e}")
            # Hide the problematic subplot
            axes[idx].set_visible(False)

    # Hide unused subplots
    for idx in range(len(combinations), len(axes)):
        axes[idx].set_visible(False)

    try:
        plt.suptitle(f'Principal Component Space Analysis ({max_pcs}D)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with error handling
        output_path = os.path.join(output_folder, 'principal_component_space_analysis.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"PC analysis plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving PC analysis plot: {e}")
    finally:
        plt.close()


    return



from typing import Optional, Union, Tuple
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.figure import Figure, Axes
import numpy.typing as npt


def scatter_annotate(
    x: np.ndarray,
    y: np.ndarray,
    centers: Optional[np.ndarray] = None,
    centers_ind: Optional[np.ndarray] = None,
    annotate: bool = True,
    labels: Optional[np.ndarray] = None,
    alpha: Union[float, np.ndarray, None] = 0.6,
    s: Union[float, np.ndarray, None] = 1,
    colors: Union[list, str, None] = None,
) -> Tuple[Figure, Axes]:
    """Scatter plot with optional cluster-center markers and annotations."""
    fig, ax = plt.subplots(figsize=(8, 8))

    try:
        ax.hexbin(x, y, gridsize=30, alpha=0.3, cmap="Blues", mincnt=1)
    except Exception:
        pass

    ax.scatter(x, y, alpha=alpha, s=s, c="cornflowerblue", edgecolors="none", rasterized=True)

    if centers_ind is not None:
        assert centers is None
        centers = np.column_stack([x[centers_ind], y[centers_ind]])
    if centers is not None:
        c = "red" if colors is None else colors
        ax.scatter(centers[:, 0], centers[:, 1], c=c, edgecolor="black", s=100, zorder=3)
    if annotate:
        assert centers is not None
        lbl = np.arange(len(centers)) if labels is None else labels
        for i in lbl:
            ax.annotate(
                str(i),
                centers[i, :2] + np.array([0.1, 0.1]),
                fontsize=12,
                fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")],
            )

    ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")
    return fig, ax


def get_nearest_point(
    data: np.ndarray, query: np.ndarray
) -> Tuple[npt.NDArray[np.float32], np.ndarray]:
    """For each row of query, return the closest row of data and its index."""
    dists = np.linalg.norm(data[np.newaxis, :] - query[:, np.newaxis, :], axis=-1)
    ind = dists.argmin(axis=1)
    return data[ind], ind


def cluster_kmeans(
    z: np.ndarray, K: int, on_data: bool = True, reorder: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering of z into K clusters.

    Returns (labels, centers). If reorder=True, clusters are sorted by
    agglomerative linkage of their centers.
    """
    km = KMeans(n_clusters=K, random_state=0, max_iter=10)
    labels = km.fit_predict(z)
    centers = km.cluster_centers_

    centers_ind = None
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)

    if reorder:
        cg = sns.clustermap(centers)
        order = cg.dendrogram_row.reordered_ind
        centers = centers[order]
        if centers_ind is not None:
            centers_ind = centers_ind[order]
        remap = {old_k: new_k for new_k, old_k in enumerate(order)}
        labels = np.array([remap[lbl] for lbl in labels])

    return labels, centers
