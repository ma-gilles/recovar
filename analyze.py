import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding, plot_utils
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)
from recovar import parser_args

def add_args(parser: argparse.ArgumentParser):

    parser = parser_args.standard_downstream_args(parser, analyze = True)

    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable (a single int, not a list)"
    )

    parser.add_argument(
        "--n-clusters", dest= "n_clusters", type=int, default=40, help="number of k-means clusters (default 40)"
    )

    parser.add_argument(
        "--normalize-kmeans",
        dest="normalize_kmeans",
        action="store_true",
        help="whether to normalize the zs before computing k-means"
    )

    parser.add_argument(
        "--n-trajectories", type=int, default=0, dest="n_trajectories", help="number of trajectories to compute between k-means clusters (default 0)"
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)"
    )

    parser.add_argument(
        "--skip-centers",
        dest="skip_centers",
        action="store_true",
        help="whether to generate the volume of the k-means centers"
    )

    parser.add_argument(
        "--n-vols-along-path", type=int, default=6, dest="n_vols_along_path", help="number of volumes to compute along each trajectory (default 6)"
    )

    parser.add_argument(
        "--density",
        type=os.path.abspath,
        required=False,
        help="density saved in .pkl file, with keys 'density' and 'latent_space_bounds'",
    )

    return parser


def analyze(recovar_result_dir, output_folder = None, zdim = 4, n_clusters = 40, n_paths = 0, skip_umap = False, q = None, n_std = None, B_factor=0, n_bins=30, n_vols_along_path = 6, skip_centers = False, normalize_kmeans = False, density_path = None, no_z_reg = False, lazy = False):


    po = o.PipelineOutput(recovar_result_dir + '/')

    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    
    elif zdim is None:
        zdim = list(po.get('zs').keys())[0]
        logger.info(f"using zdim={zdim}")
    zdim_key = f"{zdim}_noreg" if no_z_reg else zdim

    if output_folder is None:
        output_folder = recovar_result_dir + f'/analysis_{zdim_key}/' 

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    zs = po.get('zs')[zdim_key]
    cov_zs = po.get('cov_zs')[zdim_key]

    # if pipeline_output.get('unsorted_embedding') is not None:
    #     zs_unsort 
    if lazy:
        cryos = po.get('lazy_dataset')
    else:
        cryos = po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim_key])

    if density_path is not None:
        dens_pkl = utils.pickle_load(density_path)
        input_density = dens_pkl['density']
        latent_space_bounds = dens_pkl['latent_space_bounds']
        logger.warning(f"density dimension is less than zs dimension, truncate zs dimension to match density dimension = {input_density.ndim}")
        zdim = input_density.ndim
        zdim_key = f"{zdim}_noreg" if no_z_reg else zdim
        zs = zs[:,:zdim]
        cov_zs = cov_zs[:,:zdim,:zdim]
    else:
        density, latent_space_bounds  = latent_density.compute_latent_space_density(zs, cov_zs, pca_dim_max = np.min([4,zs.shape[-1]]), num_points = 50, density_option = 'kde')
        po.params['density'] = density
        # latent_space_bounds = None
        input_density = None
        latent_space_bounds = None
        
    # if zdim is None and len(po.get('zs']) > 1:
    #     logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

    # cryos = po.get('dataset')
    # embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim])
    
    noise_variance = po.get('noise_var_used')
    B_factor = args.Bfactor
    n_bins = args.n_bins

    def reorder(array):
        return dataset.reorder_to_original_indexing_from_halfsets(array, po.get('particles_halfsets'))

    output_folder_kmeans = output_folder + '/' #+ '/kmeans'+'_'+ str(n_clusters) + '/'    
    o.mkdir_safe(output_folder_kmeans)    
    utils.basic_config_logger(output_folder)
    plot_utils.plot_summary_t(po,cryos, n_eigs = 5, filename = output_folder_kmeans + "mean_variance_eigenvolume_plots.png")

    # logging.basicConfig(
    #     format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    #                     level=logging.INFO,
    #                     force = True, 
    #                     handlers=[
    #     logging.FileHandler(f"{output_folder}/run.log"),
    #     logging.StreamHandler()])



    logger.info(args)
    # zs_unsort = po.get('unsorted_embedding')['zs'][zdim_key]
    zs_unsort = zs
    if normalize_kmeans:
        # zs_unsort = po.get('unsorted_embedding')['zs'][zdim_key]
        std = np.std(zs_unsort, axis = 0)
        centers, labels = o.kmeans_analysis(output_folder_kmeans + '/PCA/', zs_unsort/std, n_clusters = n_clusters)
        centers = centers * std
    else:
        centers, labels = o.kmeans_analysis(output_folder_kmeans  + '/PCA/', zs_unsort, n_clusters = n_clusters)

    output_folder_kmeans_centers = output_folder_kmeans + '/kmeans_center_volumes/'
    o.mkdir_safe(output_folder_kmeans_centers)    
    kmeans_result = { 'centers' : centers, 'labels': reorder(labels)  }
    pickle.dump(kmeans_result, open(output_folder_kmeans + 'kmeans_result.pkl', 'wb'))
    np.savetxt(output_folder_kmeans + 'kmeans_center_coords.txt', centers)
    
    if not skip_centers:
        o.compute_and_save_reweighted(cryos, centers, zs, cov_zs, noise_variance, output_folder_kmeans_centers, B_factor, n_bins, n_min_images = args.n_min_images,  maskrad_fraction = args.maskrad_fraction)
        # move_to_one_folder(output_folder_kmeans_centers, n_clusters )

    if (not skip_umap) and (zdim > 1):
        # zs_unsort = po.get('unsorted_embedding')['zs'][zdim_key]
        mapper = o.umap_latent_space(zs_unsort)
        o.mkdir_safe(output_folder + '/umap/')    
        utils.pickle_dump(reorder(mapper.embedding_), output_folder + '/umap/umap_embedding.pkl')
        from cryodrgn import analysis
        _, kmeans_ind = analysis.get_nearest_point(zs_unsort, centers)

        o.plot_umap(output_folder + '/umap/', mapper.embedding_, mapper.embedding_[kmeans_ind])

    del zs_unsort
    # num_images = 
    # for entry in embedding_dict:
    #     for key in embedding_dict[entry]:
    #         embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(embedding_dict[entry][key], ind_split)

    if zdim > 1:
        # Recompute density
        # logger.info("recomputing density. Take out?")
        pairs = pick_pairs(centers, n_paths)
        for pair_idx in range(len(pairs)):
            pair = pairs[pair_idx]
            z_st = centers[pair[0],:]
            z_end = centers[pair[1],:]

            path_folder = output_folder_kmeans + 'path' + str(pair_idx) + '/'        
            o.mkdir_safe(path_folder)
            full_path, subsampled_path = o.make_trajectory_plots_from_results(po, zdim_key, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, input_density = input_density, latent_space_bounds = latent_space_bounds)

            logger.info(f"path {pair_idx} done")
            o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, n_min_images = args.n_min_images,  maskrad_fraction = args.maskrad_fraction)

    else:
        path_folder = output_folder_kmeans + 'path' + str(0) + '/'        
        o.mkdir_safe(path_folder)
        q = 0.03
        pairs = np.percentile(po.get('zs')[zdim], [q, 100-q])
        z_st = pairs[0]
        z_end = pairs[1]
        # n_vols_along_path = 80
        # z_points = np.linspace(z_st, z_end, n_vols_along_path)
        # pairs = [ [z_points[0], z_points[40-1]], [z_points[40], z_points[80-1]] ]
        subsampled_path = np.linspace(z_st, z_end, n_vols_along_path)[:,None]
        o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, save_all_estimates = False, n_min_images = args.n_min_images,  maskrad_fraction = args.maskrad_fraction)

    # for pair_idx in range(len(pairs)):
    #     pair = pairs[pair_idx]
    #     z_st = centers[pair[0],:]
    #     z_end = centers[pair[1],:]

    #     path_folder = output_folder_kmeans + 'path' + str(pair_idx) + '/'        
    #     o.mkdir_safe(path_folder)
    #     print("HERE")
    #     full_path, subsampled_path = o.make_trajectory_plots_from_results(po, zdim, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, compute_reproj = compute_reproj, likelihood_threshold = likelihood_threshold)        
    #     logger.info(f"path {pair_idx} done")
    #     o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins)

    kmeans_res = { 'centers': centers.tolist(), 'pairs' : pairs }
    pickle.dump(kmeans_res, open(output_folder_kmeans + 'trajectory_endpoints.pkl', 'wb'))

from recovar.output import move_to_one_folder


def pick_pairs(centers, n_pairs):
    # We try to pick some pairs that cover the latent space in some way.
    # This probably could be improved
    #     
    # Pick some pairs that are far away from each other.
    pairs = []
    X = distance_matrix(centers[:,:], centers[:,:])

    for _ in range(n_pairs//2):
        i_idx,j_idx = np.unravel_index(np.argmax(X), X.shape)
        X[i_idx, :] = 0 
        X[:, i_idx] = 0 
        X[j_idx, :] = 0 
        X[:, j_idx] = 0 
        pairs.append([i_idx, j_idx])

    # Pick some pairs that are far in the first few principal components.
    zdim = centers.shape[-1]
    max_k = np.min([(n_pairs - n_pairs//2), zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:,k])
        j_idx = np.argmin(centers[:,k])
        pairs.append([i_idx, j_idx])

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    analyze(args.result_dir, output_folder = args.outdir, zdim=  args.zdim, n_clusters = args.n_clusters, n_paths= args.n_trajectories, skip_umap = args.skip_umap, B_factor = args.Bfactor, n_bins = args.n_bins, n_vols_along_path = args.n_vols_along_path, skip_centers = args.skip_centers, normalize_kmeans = args.normalize_kmeans, density_path = args.density, no_z_reg = args.no_z_regularization, lazy = args.lazy)
