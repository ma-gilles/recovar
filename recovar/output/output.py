"""Volume I/O, result saving, and MRC file writing utilities."""

import json
import logging
import os
import pickle
import time

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.heterogeneity.latent_density as ld
from recovar import utils
from recovar.core import linalg
from recovar.data_io import cryoem_dataset, halfsets
from recovar.heterogeneity import embedding, trajectory
from recovar.output.output_paths import AnalysisPaths, ResultPaths
from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)
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


def sum_over_other(x, use_axis = None, *args, **kwargs):
    if use_axis is None:
        use_axis = [0, 1]
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


def plot_over_density(density, trajectories = None, latent_space_bounds = None,  subsampled = None, colors = None, plot_folder = None, cmap = 'inferno', same_st_end = True, zs = None, cov_zs = None, points = None, projection_function = None, annotate = False, slice_point = None):
    """Plot 2-D density projections with optional trajectories and cluster centers.

    For each pair of latent dimensions, creates a density heatmap and overlays
    trajectory paths, subsampled volume positions, and/or cluster center markers.

    Args:
        density: N-D density array on a regular grid (or None to compute on the fly).
        trajectories: List of trajectory arrays, each of shape (n_pts, n_dims).
        latent_space_bounds: Array of shape (n_dims, 2) giving [min, max] per axis.
        subsampled: List of subsampled trajectory arrays for volume markers.
        colors: Color list for trajectories.
        plot_folder: Directory to save PNG files. If None, plots are not saved.
        cmap: Matplotlib colormap name for the density.
        same_st_end: If True, only draw start/end markers for the first trajectory.
        zs: Latent coordinates, shape (n_particles, n_dims).
        cov_zs: Per-particle covariance matrices for on-the-fly density computation.
        points: Extra points (e.g. cluster centers) to scatter on top.
        projection_function: Callable to project the density onto 2 axes.
        annotate: Whether to annotate *points* with integer labels.
        slice_point: Slice coordinate for the projection function.
    """
    colors = ['k', 'cornflowerblue', 'g' , 'r', 'b', 'w', 'c'] if colors is None else colors
    path_exists = trajectories is not None

    if projection_function not in ('slice', 'slice_point', 'sum', None):
        raise ValueError(f"Unknown projection function: {projection_function}")
    projection_function = half_slice_other if projection_function == 'slice' else projection_function
    projection_function = slice_at_point if projection_function == 'slice_point' else projection_function


    projection_function = sum_over_other if projection_function is None else projection_function

    if slice_point is None:
        slice_point = points[0] if points is not None else None

    compute_density = False
    if density is None:
        if zs is None or cov_zs is None:
            raise ValueError("zs and cov_zs are required when density is not provided")
        compute_density = True
        
    if compute_density:
        num_points = 200
    else:
        num_points= density.shape[0]

    if path_exists:
        if latent_space_bounds is None:
            raise ValueError("Need latent space bounds to plot trajectories")
        _, z_to_grid = ld.get_grid_z_mappings(latent_space_bounds, num_points)
    
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
            
        ax.imshow((density_pl.T), origin='lower', cmap = cmap, interpolation = 'bilinear')
        if points is not None:
            points = points.copy()
            out_of_bounds_points = np.zeros(points.shape[0], dtype = bool)
            for k in range(points.shape[1]):
                out_of_bounds_points = out_of_bounds_points | (points[:,k] > density.shape[k])
                out_of_bounds_points = out_of_bounds_points | (points[:,k] < 0)

                points[:,k] = np.where(points[:,k] > density.shape[k], density.shape[k], points[:,k])
                points[:,k] = np.where(points[:,k] < 0, 0, points[:,k])

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
            plt.close()

    if density is not None:
        traj_dim = density.ndim
    else:
        traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2], points = points)




def plot_kmeans_over_density(density, centers, plot_folder = None, cmap = 'inferno' ):
    compute_density = False
        
    
    if compute_density:
        num_points = 200
    else:
        num_points= density.shape[0]
    
    def plot_traj_along_axes(axes):
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
            
        ax.imshow((density_pl.T), origin='lower', cmap = cmap, interpolation = 'bilinear')
        ax.scatter(centers[:,axis_x], centers[:,axis_y], c='red', edgecolor='black', s=100, zorder=3, linewidth=1)
        for i in range(centers.shape[0]):
            ax.annotate(str(i), centers[i, axes] + np.array([0.1, 0.1]), c = 'white', 
                       fontsize=12, fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        

        ax.axis("off")
            
        if plot_folder is not None:
            save_filepath = plot_folder  + 'density_' + str(axes[0]) + str(axes[1]) + '.png'
            plt.savefig(save_filepath, bbox_inches='tight')
            plt.close()

    traj_dim = centers.shape[-1]
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2])




def save_covar_output_volumes(output_folder, mean, u, s, mask, volume_shape,  us_to_save = 50, us_to_var = None, voxel_size = None):
    if us_to_var is None:
        us_to_var = [4, 10, 20]

    vol_dir = os.path.join(output_folder, 'volumes')
    mkdir_safe(vol_dir)
    n_available = int(u.shape[-1])
    n_to_save = max(0, min(int(us_to_save), n_available))
    save_volumes([u[..., k] for k in range(n_to_save)], os.path.join(vol_dir, 'eigen_pos'), volume_shape=volume_shape, voxel_size=voxel_size)
    save_volume(mean, os.path.join(vol_dir, 'mean'), volume_shape = volume_shape,   voxel_size = voxel_size)

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
        save_volume(variance_real, os.path.join(vol_dir, 'variance' + str(n_eigs)), volume_shape, from_ft = False,   voxel_size = voxel_size)


# ---------------------------------------------------------------------------
# Pipeline result building and saving
# ---------------------------------------------------------------------------

def build_params_dict(
    *,
    volume_shape,
    voxel_size,
    s_rescaled,
    noise_var_from_hf,
    noise_var_from_het_residual,
    noise_var_used,
    noise_result,
    ub_noise_var_by_var_est,
    variance_est,
    variance_fsc,
    noise_p_variance_est,
    covariance_options,
    column_fscs,
    picked_frequencies,
    input_args,
):
    """Build the params dict saved as ``model/params.pkl``.

    This is the authoritative schema for the params dict (v0.7).

    Schema
    ------
    version : str
        Format version (currently ``'0.7'``).
    volume_shape : tuple of int
        3-D grid dimensions, e.g. ``(128, 128, 128)``.
    voxel_size : float
        Angstroms per voxel.
    s : ndarray, shape (n_pcs,)
        Rescaled eigenvalues from PCA of the covariance.
    noise_var_from_hf : ndarray
        Noise variance estimated from half-map differences.
    noise_var_from_het_residual : ndarray or None
        Noise variance estimated from heterogeneity residuals (None for tilt series).
    noise_var_used : ndarray
        The noise variance actually used during estimation.
    radial_noise_var_outside_mask : ndarray
        Radial noise profile estimated from outside the solvent mask.
    radial_ub_noise_var : ndarray
        Upper-bound radial noise variance from inside the mask.
    white_noise_var_outside_mask : float
        Scalar white noise variance (median of radial profile).
    ub_noise_var_by_var_est : ndarray
        Upper-bound noise variance from signal+noise variance estimation.
    image_PS : ndarray
        Radial image power spectrum.
    masked_image_PS : ndarray
        Radial power spectrum of masked images.
    variance_est : dict
        Per-halfset variance estimates.
    variance_fsc : ndarray
        FSC of variance half-maps.
    noise_p_variance_est : ndarray
        Noise-plus-variance estimate.
    covariance_options : dict
        Options used for covariance estimation.
    column_fscs : ndarray
        Per-column FSC values.
    picked_frequencies : ndarray
        Frequency indices selected for covariance columns.
    input_args : Namespace
        The full command-line arguments used to run the pipeline.
    """
    return {
        'version': '0.7',
        'volume_shape': volume_shape,
        'voxel_size': voxel_size,
        's': s_rescaled,
        # Noise estimates
        'noise_var_from_hf': np.asarray(noise_var_from_hf),
        'noise_var_from_het_residual': (
            np.asarray(noise_var_from_het_residual)
            if noise_var_from_het_residual is not None else None
        ),
        'noise_var_used': np.asarray(noise_var_used),
        'radial_noise_var_outside_mask': np.asarray(noise_result['radial_noise_var_outside_mask']),
        'radial_ub_noise_var': np.asarray(noise_result['radial_ub_noise_var']),
        'white_noise_var_outside_mask': np.asarray(noise_result['white_noise_var_outside_mask']),
        'ub_noise_var_by_var_est': np.asarray(ub_noise_var_by_var_est),
        'image_PS': np.asarray(noise_result['image_PS']),
        'masked_image_PS': np.asarray(noise_result['masked_image_PS']),
        # Variance and covariance
        'variance_est': variance_est,
        'variance_fsc': variance_fsc,
        'noise_p_variance_est': noise_p_variance_est,
        'covariance_options': covariance_options,
        'column_fscs': column_fscs,
        'picked_frequencies': picked_frequencies,
        # Input
        'input_args': input_args,
    }


def build_embedding_dict(latent_coords, latent_coords_noreg,
                         latent_precision, latent_precision_noreg,
                         contrasts, contrasts_noreg):
    """Build the embedding dict saved as ``model/embeddings.pkl``.

    All six sub-dicts are keyed by zdim (int).

    Schema
    ------
    latent_coords : dict[int, ndarray]
        Regularized latent coordinates.
    latent_coords_noreg : dict[int, ndarray]
        Unregularized latent coordinates.
    latent_precision : dict[int, ndarray]
        Posterior covariance of regularized latent coordinates.
    latent_precision_noreg : dict[int, ndarray]
        Posterior covariance of unregularized latent coordinates.
    contrasts : dict[int, ndarray]
        Per-image contrast estimates (regularized).
    contrasts_noreg : dict[int, ndarray]
        Per-image contrast estimates (unregularized).
    """
    return {
        'latent_coords': latent_coords,
        'latent_coords_noreg': latent_coords_noreg,
        'latent_precision': latent_precision,
        'latent_precision_noreg': latent_precision_noreg,
        'contrasts': contrasts,
        'contrasts_noreg': contrasts_noreg,
    }


def save_pipeline_results(
    paths,
    result,
    embedding_dict,
    covariance_cols,
    particles_ind_split,
    ind_split,
    zs_full=None,
):
    """Save all pipeline results to disk.

    Parameters
    ----------
    paths : ResultPaths
        Centralized output paths.
    result : dict
        The params dict built by :func:`build_params_dict`.
    embedding_dict : dict
        The embedding dict built by :func:`build_embedding_dict`.
    covariance_cols : ndarray or None
        Covariance columns (None if ``--keep-intermediate`` is off).
    particles_ind_split : list of ndarray
        Per-particle halfset indices.
    ind_split : list of ndarray
        Per-image halfset indices.
    zs_full : dict or None
        Full latent coordinates before complement-mask trimming (if applicable).
    """
    paths.ensure_dirs()

    utils.pickle_dump(particles_ind_split, paths.particles_halfsets)
    utils.pickle_dump(ind_split, paths.halfsets)
    utils.pickle_dump(result, paths.params)
    utils.pickle_dump(covariance_cols, paths.covariance_cols)

    # Save embeddings as per-zdim directories with individual .npy files
    _save_embeddings_per_zdim(paths, embedding_dict)

    if zs_full is not None:
        utils.pickle_dump(zs_full, paths.zs_with_complement)

    write_metadata_json(paths, result)

    logger.info("Saved pipeline results to %s", paths.model_dir)


# Embedding field names that are saved as .npy files per zdim
_EMBEDDING_FIELDS = [
    'latent_coords', 'latent_coords_noreg',
    'latent_precision', 'latent_precision_noreg',
    'contrasts', 'contrasts_noreg',
]


def _save_embeddings_per_zdim(paths, embedding_dict):
    """Save embeddings as per-zdim directories with individual .npy files.

    Creates ``model/zdim_{N}/`` for each computed zdim, containing::

        latent_coords.npy
        latent_coords_noreg.npy
        latent_precision.npy
        latent_precision_noreg.npy
        contrasts.npy
        contrasts_noreg.npy

    Parameters
    ----------
    paths : ResultPaths
    embedding_dict : dict
        Built by :func:`build_embedding_dict`.
    """
    # Collect all zdims from any field
    all_zdims = set()
    for field in _EMBEDDING_FIELDS:
        if field in embedding_dict and isinstance(embedding_dict[field], dict):
            all_zdims.update(embedding_dict[field].keys())

    for zdim in sorted(all_zdims):
        zdim_dir = paths.embedding_zdim_dir(zdim)
        os.makedirs(zdim_dir, exist_ok=True)
        for field in _EMBEDDING_FIELDS:
            if field in embedding_dict and zdim in embedding_dict[field]:
                arr = np.asarray(embedding_dict[field][zdim])
                np.save(os.path.join(zdim_dir, f"{field}.npy"), arr)

    logger.info("Saved embeddings for zdims %s to per-zdim directories", sorted(all_zdims))


def _load_embeddings_per_zdim(paths):
    """Load embeddings from per-zdim .npy directories.

    Returns the same dict structure as the legacy ``embeddings.pkl``::

        {
            'latent_coords': {2: array, 4: array, ...},
            'latent_coords_noreg': {2: array, 4: array, ...},
            ...
        }

    Returns None if no zdim directories are found.
    """
    import re
    model_dir = paths.model_dir
    if not os.path.isdir(model_dir):
        return None

    # Find zdim_* directories
    zdim_dirs = {}
    pattern = re.compile(r"^zdim_(\d+)$")
    for name in os.listdir(model_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(model_dir, name)):
            zdim_dirs[int(m.group(1))] = os.path.join(model_dir, name)

    if not zdim_dirs:
        return None

    # Build the embedding dict
    embedding = {field: {} for field in _EMBEDDING_FIELDS}
    for zdim in sorted(zdim_dirs):
        zdim_dir = zdim_dirs[zdim]
        for field in _EMBEDDING_FIELDS:
            npy_path = os.path.join(zdim_dir, f"{field}.npy")
            if os.path.isfile(npy_path):
                embedding[field][zdim] = np.load(npy_path)

    logger.info("Loaded embeddings from per-zdim directories: zdims=%s", sorted(zdim_dirs))
    return embedding


def write_metadata_json(paths, result):
    """Write a human-readable JSON manifest alongside the pickle files.

    This file is not loaded by the pipeline -- it exists for users to quickly
    inspect run parameters without unpickling.
    """
    import datetime
    try:
        from recovar import __version__
    except ImportError:
        __version__ = "unknown"

    zdims = []
    input_args = result.get('input_args')
    if input_args is not None:
        zdims_raw = getattr(input_args, 'zdim', None)
        if zdims_raw is not None:
            zdims = [int(z) for z in zdims_raw]

    # Downsample info (set by pipeline.py before swapping to downsampled stack)
    downsample_applied = None
    original_particles = None
    if input_args is not None:
        downsample_applied = getattr(input_args, '_downsample_applied', None)
        original_particles = getattr(input_args, '_original_particles', None)

    metadata = {
        'recovar_version': str(__version__),
        'params_version': result.get('version', 'unknown'),
        'saved_at': datetime.datetime.now().isoformat(),
        'volume_shape': list(result.get('volume_shape', [])),
        'voxel_size': float(result.get('voxel_size', 0)),
        'zdims_computed': zdims,
        'downsample_applied': downsample_applied,
        'original_particles_file': original_particles,
        'files': {
            'params': 'model/params.pkl',
            'embeddings': 'model/embeddings.pkl',
            'covariance_cols': 'model/covariance_cols.pkl',
            'halfsets': 'model/halfsets.pkl',
            'particles_halfsets': 'model/particles_halfsets.pkl',
            'mean_volume': 'output/volumes/mean.mrc',
            'mask': 'output/volumes/mask.mrc',
        },
    }

    try:
        with open(paths.metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
    except (IOError, OSError, TypeError) as e:
        logger.warning("Could not write metadata.json: %s", e)


def kmeans_analysis(output_folder, zs, n_clusters = 20):
    """Run k-means clustering on latent coordinates and save scatter plots.

    Generates annotated and unannotated scatter plots for all pairwise
    combinations of the first few latent dimensions.

    Args:
        output_folder: Directory to save PCA scatter PNGs and center data.
        zs: Latent coordinates, shape (n_particles, n_dims).
        n_clusters: Number of k-means clusters.

    Returns:
        Tuple of (labels, centers) from k-means clustering.
    """
    reorder = zs.shape[1] != 1
    labels, centers = cluster_kmeans(zs, n_clusters, reorder = reorder)
    mkdir_safe(output_folder)

    def plot_axes(axes = [0,1]):
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=True, labels=None, alpha=0.1, s=1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'PC_'+str(axes[0]) + str(axes[1])+'.png' )
            plt.close()

        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=False, labels=None, alpha=0.1, s=2)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'PC_'+str(axes[0]) + str(axes[1])+'no_annotate.png' )
            plt.close()

    for k in range(1,zs.shape[-1]):
        plot_axes(axes = [0,k])
    if zs.shape[-1] > 2:
        plot_axes(axes = [1,2])
    
    return centers, labels



def plot_umap(output_folder, zs, centers):
    """Generate UMAP embedding plots with cluster center overlay.

    Creates scatter and hexbin UMAP projections saved as PNGs in
    ``output_folder/umap/``.

    Args:
        output_folder: Parent directory; a ``umap/`` subdirectory is created.
        zs: Latent coordinates, shape (n_particles, n_dims).
        centers: Cluster centers, shape (n_clusters, n_dims).
    """
    def plot_axes(axes = [0,1]):
        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=True, labels=None, alpha=0.1, s=1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'kmeans_centers.png' )
            plt.close()

        fig,ax = scatter_annotate(zs[:,axes[0]], zs[:,axes[1]], centers=centers[:,axes], centers_ind=None, annotate=False, labels=None, alpha=0.1, s=2)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + 'kmeans_centers_no_annotate.png' )
            plt.close()

        import seaborn as sns

        g = sns.jointplot(x=zs[:,0], y=zs[:,1], alpha=.1, s=1)
        g.set_axis_labels('UMAP1', 'UMAP2')
        if output_folder is not None:
            plt.savefig(output_folder + 'sns.png' )
            plt.close()

        g = sns.jointplot(x=zs[:,0], y=zs[:,1], kind='hex')
        g.set_axis_labels('UMAP1', 'UMAP2')
        if output_folder is not None:
            plt.savefig(output_folder + 'sns_hex.png' )
            plt.close()

    plot_axes(axes = [0,1])



def compute_and_save_reweighted(
    dataset,
    path_subsampled,
    zs,
    cov_zs,
    output_folder,
    B_factor,
    n_bins=30,
    n_min_particles=100,
    embedding_option="cov_dist",
    save_all_estimates=False,
    maskrad_fraction=20,
    apply_global_filtering=False,
    fsc_mask=None,
    fsc_mask_radius=None,
    fsc_mask_edgewidth=None,
    vol_prefix="state",
    kernel_regression_mode="standard",
    deconv_lambda_grid=None,
    local_poly_degree=3,
    local_poly_bandwidth_multipliers=None,
):
    """Compute reweighted volume estimates and save with standardized organization.

    Primary volumes (filtered, half-maps) are placed directly in
    *output_folder*.  Diagnostics (params, local resolution, etc.) go
    into ``output_folder/diagnostics/{prefix}{idx:03d}/``.

    Parameters
    ----------
    dataset : CryoEMDataset
        Dataset with ``halfset_indices`` set.  Halfset datasets are
        obtained lazily via ``dataset.get_halfset(k)``.
    """
    from recovar.output.output_paths import VolumeOutputPaths
    ds = dataset

    if n_min_particles is None:
        n_min_particles = 100

    mkdir_safe(output_folder)
    from recovar.heterogeneity import (
        deconvolved_kernel_regression,
        heterogeneity_volume,
        latent_density,
        local_polynomial_regression,
    )
    n_vols = path_subsampled.shape[0]
    if kernel_regression_mode not in ("standard", "deconvolved", "local_poly"):
        raise ValueError(f"Unknown kernel_regression_mode={kernel_regression_mode!r}")

    for k in range(n_vols):
        vol_paths = VolumeOutputPaths(output_folder, vol_prefix, k)
        vol_paths.ensure_dirs()

        ndim = zs.shape[-1]
        latent_points = path_subsampled[k][None]
        np.savetxt(vol_paths.latent_coords, latent_points)

        if embedding_option == 'llh':
            log_likelihoods = latent_density.compute_latent_log_likelihood(latent_points, zs, cov_zs)[...,0]
            heterogeneity_distances = log_likelihoods - np.min(log_likelihoods)
        elif embedding_option == 'cov_dist':
            heterogeneity_distances = latent_density.compute_latent_quadratic_forms_in_batch(latent_points, zs, cov_zs)[...,0]
        elif embedding_option == 'dist':
            cov_zs = cov_zs*0 + np.eye(ndim)
            heterogeneity_distances = latent_density.compute_latent_log_likelihood(latent_points, zs, cov_zs)[...,0]
        else:
            raise ValueError("Unknown embed option")

        heterogeneity_distances = ds.split_halfset_array(
            heterogeneity_distances, per_particle=ds.tilt_series_flag)
        deconv_latent_differences = None
        deconv_latent_precision = None
        local_poly_latent_differences = None
        local_poly_latent_precision = None
        if kernel_regression_mode == "deconvolved":
            if ndim != 1:
                raise NotImplementedError(f"Deconvolved kernel regression only supports zdim=1; got ndim={ndim}")
            zs_1d = np.asarray(zs, dtype=np.float32)
            if zs_1d.ndim == 1:
                zs_1d = zs_1d[:, None]
            latent_differences = np.asarray(zs_1d[:, :1] - latent_points[:, :1], dtype=np.float32)
            latent_precision = deconvolved_kernel_regression._coerce_1d_latent_precision(cov_zs)
            deconv_latent_differences = ds.split_halfset_array(
                latent_differences[:, 0],
                per_particle=ds.tilt_series_flag,
            )
            deconv_latent_precision = ds.split_halfset_array(
                latent_precision,
                per_particle=ds.tilt_series_flag,
            )
        elif kernel_regression_mode == "local_poly":
            if ndim != 1:
                raise NotImplementedError(f"local_poly kernel regression only supports zdim=1; got ndim={ndim}")
            zs_1d = np.asarray(zs, dtype=np.float32)
            if zs_1d.ndim == 1:
                zs_1d = zs_1d[:, None]
            latent_differences = np.asarray(zs_1d[:, :1] - latent_points[:, :1], dtype=np.float32)
            latent_precision = local_polynomial_regression.coerce_1d_latent_precision(cov_zs)
            local_poly_latent_differences = ds.split_halfset_array(
                latent_differences[:, 0],
                per_particle=ds.tilt_series_flag,
            )
            local_poly_latent_precision = ds.split_halfset_array(
                latent_precision,
                per_particle=ds.tilt_series_flag,
            )

        locres_maskrad = ds.grid_size * ds.voxel_size / maskrad_fraction
        logger.info("Mask radius fraction = %s. Setting locres_maskrad = locres_sampling = box_size * voxel_size / %s = %.1f Angstroms. Using %d particles for template.", maskrad_fraction, maskrad_fraction, locres_maskrad, n_min_particles)
        heterogeneity_volume.make_volumes_kernel_estimate_local(
            heterogeneity_distances,
            ds,
            vol_paths,
            ndim,
            n_bins,
            B_factor,
            tau=None,
            n_min_particles=n_min_particles,
            locres_sampling=locres_maskrad,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=0,
            upsampling_for_ests=1,
            use_mask_ests=False,
            grid_correct_ests=False,
            save_all_estimates=save_all_estimates,
            metric_used="locshellmost_likely",
            use_fast_rfft=True,
            kernel_regression_mode=kernel_regression_mode,
            deconv_latent_differences=deconv_latent_differences,
            deconv_latent_precision=deconv_latent_precision,
            deconv_lambda_grid=deconv_lambda_grid,
            local_poly_degree=local_poly_degree,
            local_poly_bandwidth_multipliers=local_poly_bandwidth_multipliers,
            local_poly_latent_differences=local_poly_latent_differences,
            local_poly_latent_precision=local_poly_latent_precision,
        )

        logger.info("Done with volume %d: %s", k, vol_paths.stem)

    np.savetxt(os.path.join(output_folder, 'latent_coords.txt'), path_subsampled)


def plot_loglikelihood_over_scatter(path_subsampled, zs, cov_zs, save_path):
    likelihoods = ld.compute_latent_quadratic_forms(path_subsampled, zs, cov_zs)
    vmax = np.max(likelihoods)
    vmin = np.max([np.min(likelihoods),  1e-8 *np.max(likelihoods)])
    plt.ioff()
    for k in range(likelihoods.shape[1]):
        fig, ax = plt.subplots(figsize = (8,8))
        
        # Create hexbin density plot for background
        try:
            ax.hexbin(zs[:,0], zs[:,1], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
        except (ValueError, TypeError):
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
        save_filepath = save_path +  format(k, '04d') + '.png'
        plt.savefig(save_filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)


def load_results_new(datadir):
    model_folder = datadir +'model'  + '/'
    output_folder = datadir +'output'  + '/'
    with open(model_folder + 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    return results


class PipelineOutput:
    def __init__(self, result_path):
        self.result_path = result_path.rstrip('/') + '/'
        self.paths = ResultPaths(self.result_path.rstrip('/'))
        self.params = utils.pickle_load(self.paths.params)
        self.embedding = None
        self.embedding_loaded = False
        self._embedding_key_cache = {}
        self._embedding_halfsets_cache = None
        self.version = str(self.params['version']) if 'version' in self.params else '0'

    def _ensure_embedding_raw_loaded(self):
        if self.embedding is not None:
            return
        # Try new per-zdim .npy format first
        self.embedding = _load_embeddings_per_zdim(self.paths)
        if self.embedding is not None:
            return
        # Fall back to legacy monolithic embeddings.pkl
        self.embedding = utils.pickle_load(self.paths.embeddings)

    def _get_input_args(self):
        if not hasattr(self.params, "get") or "input_args" not in self.params:
            return None
        input_args = self.params["input_args"]
        try:
            return input_args.item()
        except (ValueError, AttributeError):
            return input_args

    def _use_image_halfsets_for_unshared_tilt_contrast(self):
        input_args = self._get_input_args()
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
        if entry not in self.embedding:
            available = [k for k in self.embedding.keys()] if self.embedding else []
            raise KeyError(
                f"Embedding entry '{entry}' not found. "
                f"Available entries: {available}"
            )
        return list(self.embedding[entry].keys())

    def has_embedding_entry(self, entry):
        """Check whether a top-level embedding entry exists."""
        self._ensure_embedding_raw_loaded()
        return self.embedding is not None and entry in self.embedding

    def has_embedding_key(self, entry, key):
        return key in self.get_embedding_keys(entry)

    def get_embedding_component(self, entry, key):
        """Return one embedding array in **dataset-local order**.

        The stored embeddings are NaN-padded in original-file space.
        This method selects only computed entries (identified by
        ``particles_halfsets``) and returns them in sorted original-index
        order, which matches the unified dataset's local ordering.
        """
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
            if entry.startswith('contrasts') and self._use_image_halfsets_for_unshared_tilt_contrast():
                values = values[np.sort(image_halfsets)]
            else:
                values = values[np.sort(particle_halfsets)]
        self._embedding_key_cache[cache_key] = values
        return values

    def get_unsorted_embedding_component(self, entry, key):
        """Return raw embedding values in original dataset order, without halfset reindexing."""
        self._ensure_embedding_raw_loaded()
        if key not in self.embedding[entry]:
            raise KeyError(f"Embedding key {key} not found in {entry}.")
        return np.asarray(self.embedding[entry][key])

    def load_embedding(self):
        """Load all embedding arrays into dataset-local order.

        Selects only computed entries (no NaN) and sorts by original
        index, matching the unified dataset's local ordering.
        """
        self._ensure_embedding_raw_loaded()

        if self.version != '0':
            halfsets, image_halfsets = self._get_embedding_halfsets()
            sorted_halfsets = np.sort(halfsets)
            sorted_image_halfsets = np.sort(image_halfsets)

            for entry in self.embedding:
                for key in self.embedding[entry]:
                    if entry.startswith('contrasts') and self._use_image_halfsets_for_unshared_tilt_contrast():
                        self.embedding[entry][key] = self.embedding[entry][key][sorted_image_halfsets]
                    else:
                        self.embedding[entry][key] = self.embedding[entry][key][sorted_halfsets]
        self.embedding_loaded = True
        self._embedding_key_cache = {}
        return

    def _list_saved_eigenvector_indices(self):
        vols_dir = self.paths.volumes_dir
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
            out[i] = utils.load_mrc(self.paths.eigenvector(eig_idx))
        return out

    def get_u(self, n_pcs=50):
        selected_indices = self._select_saved_eigenvector_indices(n_pcs)
        vol_size = int(np.prod(self.params['volume_shape']))
        out = np.empty((len(selected_indices), vol_size), dtype=np.complex64)
        for i, eig_idx in enumerate(selected_indices):
            vol = utils.load_mrc(self.paths.eigenvector(eig_idx))
            out[i] = np.asarray(fourier_transform_utils.get_dft3(vol), dtype=np.complex64).reshape(-1)
        return out

    _EMBEDDING_ENTRIES = (
        'latent_coords', 'latent_coords_noreg',
        'latent_precision', 'latent_precision_noreg',
        'contrasts', 'contrasts_noreg',
    )

    def get(self, key):
        if key in self._EMBEDDING_ENTRIES:
            if not self.embedding_loaded:
                self.load_embedding()
            return self.embedding[key]

        elif key == 'unsorted_embedding':
            return utils.pickle_load(self.paths.embeddings)

        elif key in ('u', 'u_real'):
            return self.get_u_real(50) if key == 'u_real' else self.get_u(50)

        elif key == 'mean':
            return fourier_transform_utils.get_dft3(utils.load_mrc(self.paths.mean_volume)).reshape(-1)

        elif key == 'mean_halfmaps':
            half1 = fourier_transform_utils.get_dft3(utils.load_mrc(self.paths.mean_half1_unfil)).reshape(-1)
            half2 = fourier_transform_utils.get_dft3(utils.load_mrc(self.paths.mean_half2_unfil)).reshape(-1)
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
            return utils.load_mrc(self.paths.variance(10))
        elif key == 'variance20':
            return utils.load_mrc(self.paths.variance(20))
        elif key == 'focus_mask':
            return utils.load_mrc(self.paths.focus_mask_volume)
        elif key == 'volume_mask':
            return utils.load_mrc(self.paths.mask_volume)
        elif key == 'dilated_volume_mask':
            return utils.load_mrc(self.paths.dilated_mask_volume)
        elif key == 'covariance_cols':
            return utils.pickle_load(self.paths.covariance_cols)

        elif key in ('dataset', 'lazy_dataset'):
            ds = halfsets.load_halfset_dataset_from_args(
                self.get('input_args'),
                lazy='lazy' in key,
                ind_split=self.get('halfsets'),
            )
            add_noise_to_loaded_dataset(ds, self.get('noise_var_used'))
            return ds

        elif key == 'halfsets':
            return utils.pickle_load(self.paths.halfsets)
        elif key == 'particles_halfsets':
            if self.version == '0.1':
                return utils.pickle_load(self.paths.halfsets)
            else:
                return utils.pickle_load(self.paths.particles_halfsets)

        elif key == 'input_args':
            return self._get_input_args()

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
        keys += list(self._EMBEDDING_ENTRIES)
        keys += ['u', 'u_real', 'mean', 'volume_mask', 'dilated_volume_mask', 'covariance_cols', 'dataset', 'lazy_dataset', 'variance', 'variance20', 'focus_mask', 'image_snr', 'mean_halfmaps', 'halfsets', 'input_args', 'unsorted_embedding']
        return keys


def add_noise_to_loaded_dataset(ds, noise_variance):
    if noise_variance.ndim == 1:
        ds.set_radial_noise_model(noise_variance)
    else:
        ds.set_variable_radial_noise_model(noise_variance)

def make_trajectory_plots_from_results(pipeline_output, basis_size, output_folder, cryos = None, z_st = None, z_end = None, gt_volumes= None, n_vols_along_path = 6, plot_llh = False,  input_density = None, latent_space_bounds = None):
    """Compute minimum-energy trajectories and generate volume/density plots.

    Finds optimal paths between start and end latent coordinates (or between
    ground-truth volume endpoints), generates volumes along the path, and
    saves density overlay plots.

    Args:
        pipeline_output: Pipeline output object with embeddings and covariance.
        basis_size: Number of PCA dimensions for trajectory computation.
        output_folder: Directory to write trajectory volumes and plots.
        cryos: CryoEMDataset (optional; loaded from pipeline_output if None).
        z_st: Start point in latent space, shape (n_dims,).
        z_end: End point in latent space, shape (n_dims,).
        gt_volumes: Ground-truth volumes for automatic endpoint selection.
        n_vols_along_path: Number of volumes to generate along the trajectory.
        plot_llh: Whether to generate per-volume likelihood scatter plots.
        input_density: Pre-computed density array (or None to compute).
        latent_space_bounds: Bounds for the latent space grid.
    """
    if not (((z_st is not None) and (z_end is not None)) or (gt_volumes is not None)):
        raise ValueError("either z_st and z_end should be passed, or gt_volumes")

    if input_density is not None:
        if latent_space_bounds is None:
            raise ValueError("need latent_space_bounds if providing density")

    cryos, zs, cov_zs, density, latent_space_bounds = _resolve_trajectory_plot_inputs(
        pipeline_output,
        basis_size,
        cryos,
        input_density,
        latent_space_bounds,
    )

    return make_trajectory_plots(
        density,
        zs,
        cov_zs,
        z_st, z_end, latent_space_bounds, output_folder,
        gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = plot_llh, use_input_density = input_density is not None)


def _resolve_trajectory_plot_inputs(pipeline_output, basis_size, cryos, input_density, latent_space_bounds):
    """Resolve trajectory plotting inputs from PipelineOutput."""
    zs = pipeline_output.get_embedding_component("latent_coords", basis_size)
    cov_zs = pipeline_output.get_embedding_component("latent_precision", basis_size)

    if cryos is None:
        cryos = pipeline_output.get("dataset")
        try:
            contrasts = pipeline_output.get_embedding_component("contrasts", basis_size)
        except KeyError:
            contrasts = None
        if contrasts is not None:
            embedding.set_contrasts_in_cryos(cryos, contrasts)

    if latent_space_bounds is None:
        latent_space_bounds = ld.compute_latent_space_bounds(zs)
    density = input_density if input_density is not None else pipeline_output.get("density")
    return cryos, zs, cov_zs, density, latent_space_bounds


def make_trajectory_plots(density, zs, cov_zs, z_st, z_end, latent_space_bounds, output_folder, gt_volumes= None, n_vols_along_path = 6, plot_llh = False, use_input_density =False):

    latent_space_bounds = ld.compute_latent_space_bounds(zs) if latent_space_bounds is None else latent_space_bounds

    st_time = time.time()
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

        logger.info("after path %.1fs", time.time() - st_time)
        inp_dens = density if use_input_density else None
        plot_over_density(inp_dens, [path_z],latent_space_bounds,  subsampled = [path_subsampled[1:-1] ] , colors = ['cornflowerblue'], plot_folder = output_folder + 'density/', cmap = 'inferno', same_st_end = False, zs = zs, cov_zs = cov_zs)
    else:
        path_z = np.linspace(z_st, z_end, n_vols_along_path)[...,0]
        path_subsampled = path_z

    st_time = time.time()

    if use_input_density:
        density_on_path = density_on_grid(path_z, density, latent_space_bounds)
        density_on_path_subs = density_on_grid(path_subsampled, density, latent_space_bounds)
    else:
        logger.warning("density on path not computed")
        density_on_path = ld.compute_latent_space_density_at_pts(path_z, zs, cov_zs) + np.nan
        density_on_path_subs = ld.compute_latent_space_density_at_pts(path_subsampled, zs, cov_zs) + np.nan

    densities = { 'density' :  density_on_path.tolist(), 'path' : path_z.tolist(), 'density_subsampled': density_on_path_subs.tolist(), 'path_subsampled' : path_subsampled.tolist(), } 
    with open(output_folder + '/path.json', 'w') as f:
        json.dump(densities, f)

    if plot_llh:
        plot_loglikelihood_over_scatter(path_subsampled, zs, cov_zs, save_path = output_folder)

    logger.info("after all plots %.1fs", time.time() - st_time)
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
        
    def plot_traj_along_axes(axes):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize = (8,8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]

        # Create hexbin density plot for background
        try:
            ax.hexbin(zs[:,axis_x], zs[:,axis_y], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
        except (ValueError, TypeError):
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
            plt.close()

    traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    for k1 in range(np.min([traj_dim,3])):
        for k2 in range(k1+1, traj_dim):
            plot_traj_along_axes([k1, k2])


def umap_latent_space(zs):
    import umap
    st_time = time.time()
    n_components = np.min([zs.shape[1], 2])
    mapper = umap.UMAP(n_components=n_components, random_state=42).fit(zs)
    logger.info("time to umap: %.1fs", time.time() - st_time)
    return mapper



def standard_pipeline_plots(po, zdim_key, output_folder):
    """Generate standard pipeline output plots.

    Produces individual plots (eigenvolumes, contrast histogram, eigenvalues,
    FSC, PC scatter) plus a consolidated ``pipeline_summary.png``.

    Args:
        po: Pipeline output object.
        zdim_key: Latent dimension key for embeddings/contrasts.
        output_folder: Directory to save plots into.
    """
    from recovar.output import plot_utils
    mkdir_safe(output_folder)
    plot_utils.plot_summary_t(po, n_eigs = 10, filename = os.path.join(output_folder, "mean_variance_eigenvolume_plots.png"))

    import matplotlib.pyplot as plt

    # Contrast histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_utils.plot_contrast_histogram(
        po.get_embedding_component('contrasts', zdim_key),
        ax=ax,
        zdim_key=zdim_key,
    )
    plt.savefig(os.path.join(output_folder, 'contrast_histogram.png'), bbox_inches='tight')
    plt.close()

    # Eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_utils.plot_eigenvalues(po.get('s'), ax=ax)
    plt.savefig(os.path.join(output_folder, 'eigenvalues.png'), bbox_inches='tight')
    plt.close()

    # Mean FSC
    plt.figure(figsize = (8,8))
    ax = plot_utils.plot_mean_fsc(po,None)
    plt.savefig(os.path.join(output_folder, 'mean_fsc.png'), bbox_inches='tight')
    plt.close()



    # Load latent coordinates with robust error handling
    try:
        latent_keys = list(po.get_embedding_keys('latent_coords'))
        if len(latent_keys) == 0:
            logger.warning("No latent coordinates found in pipeline output. Skipping PC analysis.")
            return
        latent_key = 4 if 4 in latent_keys else latent_keys[0]
        if latent_key != 4:
            logger.info("Using latent space with key %s instead of 4", latent_key)

        z = np.asarray(po.get_embedding_component('latent_coords', latent_key))
        if z.ndim != 2:
            logger.warning("Invalid latent coordinates data. Skipping PC analysis.")
            return
            
        logger.info("Latent space shape: %s", z.shape)
        logger.info("Number of particles: %d", z.shape[0])
        logger.info("Latent dimensions: %d", z.shape[1])
        
        # Validate data quality
        if z.shape[0] < 10:
            logger.warning("Too few particles (%d) for meaningful PC analysis. Skipping.", z.shape[0])
            return
            
        if z.shape[1] < 2:
            logger.warning("Too few dimensions (%d) for PC analysis. Skipping.", z.shape[1])
            return
            
        # Check for NaN or infinite values
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            logger.warning("Latent coordinates contain NaN or infinite values. Attempting to clean data.")
            z = z[~np.any(np.isnan(z) | np.isinf(z), axis=1)]
            if z.shape[0] < 10:
                logger.warning("Too few valid particles after cleaning. Skipping PC analysis.")
                return
                
    except (KeyError, FileNotFoundError, ValueError) as e:
        logger.error("Error loading latent coordinates: %s", e)
        return

    # Determine number of PCs to plot based on available dimensions
    max_pcs = min(4, z.shape[1])
    if max_pcs < 2:
        logger.warning("Only %d dimensions available, cannot create pairwise plots.", max_pcs)
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
    except (ValueError, TypeError) as e:
        logger.error("Error creating subplots: %s", e)
        return

    # Generate pairwise combinations
    combinations = [(i, j) for i in range(max_pcs) for j in range(i+1, max_pcs)]

    for idx, (i, j) in enumerate(combinations):
        if idx >= len(axes):
            break

        try:
            x_data = z[:, i]
            y_data = z[:, j]

            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                logger.warning("Skipping PC%d vs PC%d due to NaN values", i+1, j+1)
                continue

            if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
                logger.warning("Skipping PC%d vs PC%d due to infinite values", i+1, j+1)
                continue

            # Hexbin density + scatter (consistent with scatter_annotate)
            try:
                if len(x_data) > 50:
                    axes[idx].hexbin(x_data, y_data, gridsize=30,
                                   alpha=0.3, cmap='Blues', mincnt=1)
            except (ValueError, TypeError) as e:
                logger.debug("Could not add hexbin for PC%d vs PC%d: %s", i+1, j+1, e)
            axes[idx].scatter(x_data, y_data, alpha=0.3, s=0.5,
                            c='cornflowerblue', edgecolors='none', rasterized=True)
            axes[idx].set_xlabel(f'PC{i+1}')
            axes[idx].set_ylabel(f'PC{j+1}')
            axes[idx].set_title(f'PC{i+1} vs PC{j+1}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        except (ValueError, TypeError, IndexError) as e:
            logger.error("Error plotting PC%d vs PC%d: %s", i+1, j+1, e)
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
        logger.info("PC analysis plot saved to %s", output_path)
        
    except (OSError, ValueError) as e:
        logger.error("Error saving PC analysis plot: %s", e)
    finally:
        plt.close()

    # Consolidated summary plot
    try:
        plot_utils.plot_pipeline_summary(po, zdim_key, output_folder)
        logger.info("Pipeline summary plot saved to %s", os.path.join(output_folder, 'pipeline_summary.png'))
    except (KeyError, FileNotFoundError, ValueError, TypeError) as e:
        logger.warning("Could not generate pipeline summary plot: %s", e)


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
    except (ValueError, TypeError):
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
    data: np.ndarray, query: np.ndarray, chunk_size: Optional[int] = None
) -> Tuple[npt.NDArray[np.float32], np.ndarray]:
    """For each row of query, return the closest row of data and its index.

    The computation is chunked over query rows to avoid materializing a full
    ``(n_query, n_data, n_dim)`` distance tensor for larger inputs.
    """
    data = np.asarray(data)
    query = np.asarray(query)
    if data.ndim != 2 or query.ndim != 2:
        raise ValueError("data and query must both be 2D arrays")
    if data.shape[1] != query.shape[1]:
        raise ValueError("data and query must have the same feature dimension")
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one point")

    if chunk_size is None:
        target_bytes = 256 * 1024 * 1024
        bytes_per_value = np.dtype(np.result_type(data.dtype, query.dtype, np.float32)).itemsize
        per_query_bytes = max(1, data.shape[0] * data.shape[1] * bytes_per_value)
        chunk_size = max(1, target_bytes // per_query_bytes)
    else:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    ind = np.empty(query.shape[0], dtype=np.int64)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        dists = np.linalg.norm(data[np.newaxis, :] - query[start:end, np.newaxis, :], axis=-1)
        ind[start:end] = dists.argmin(axis=1)
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
