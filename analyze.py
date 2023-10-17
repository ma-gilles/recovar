## something
import recovar_config
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "result_dir",
        # dest="result_dir",
        type=os.path.abspath,
        help="result dir (output dir of pipeline)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=False,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--zdim", type=int, help="Dimension of latent variable (a single int, not a list)"
    )

    parser.add_argument(
        "--n-clusters", metavar=int, default=40, help="mask options: from_halfmaps (default), input, sphere, none"
    )

    parser.add_argument(
        "--n-trajectories", type=int, default=6, dest="n_trajectories", help="how many trajectories to compute between k-means clusters"
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)"
    )

    parser.add_argument(
        "--q", metavar=float, default=None, help="quantile used for reweighting (default = 0.95)"
    )

    parser.add_argument(
        "--n-std", metavar=float, default=None, help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)"
    )

    return parser


def analyze(recovar_result_dir, output_folder = None, zdim = 4, n_clusters = 40, n_paths = 6, skip_umap = False, q = None, n_std = None):

    results = o.load_results_new(recovar_result_dir + '/')

    if zdim is None and len(results['zs']) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    elif zdim is None:
        zdim = next(iter(results['zs']))
        logger.info(f"using zdim={zdim}")

    if q is None and n_std is None:
        likelihood_threshold = latent_density.get_log_likelihood_threshold( k = zdim)
    elif q is not None and n_std is not None:
        logger.error("either q or n_std should be set, not both")
    elif n_std is not None:
        likelihood_threshold = n_std
    else: 
        likelihood_threshold = latent_density.get_log_likelihood_threshold(q=q, k = zdim)

    if output_folder is None:
        output_folder = recovar_result_dir + '/output/'

    if zdim is None and len(results['zs']) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

    if zdim not in results['zs']:
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in results['zs'].keys()))

    cryos = dataset.load_dataset_from_args(results['input_args'])

    # DO THIS
    # for cryos_idx,cryo in enumerate(cryos):
    #     cryo.CTF_params[:,-1] = results['est_contrasts'][zdim][]

    output_folder_kmeans = output_folder + 'kmeans'+str(zdim)+'_'+ str(n_clusters) + '/'        
    centers = o.kmeans_analysis_from_dict(output_folder_kmeans, results, cryos, likelihood_threshold,  n_clusters = 40, generate_volumes = True, zdim =zdim)
    pickle.dump(centers, open(output_folder_kmeans + 'centers.pkl', 'wb'))

    pairs = pick_pairs(centers, n_paths)
    for pair_idx in range(len(pairs)):

        pair = pairs[pair_idx]
        z_st = centers[pair[0],:]
        z_end = centers[pair[1],:]

        path_folder = output_folder_kmeans + 'path' + str(pair_idx) + '/'        
        o.mkdir_safe(path_folder)

        o.make_trajectory_plots_from_results(results, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = 6, plot_llh = False, basis_size =zdim, compute_reproj = False, likelihood_threshold = likelihood_threshold)        
        print('path ', pair_idx)
        
    kmeans_res = { 'centers': centers.tolist(), 'pairs' : pairs }
    pickle.dump(kmeans_res, open(output_folder + 'trajectory_endpoints.pkl', 'wb'))

    if not skip_umap:
        mapper = o.umap_latent_space(results['zs'][zdim])
        pickle.dump(mapper.embedding_, open(output_folder + 'umap_embedding.pkl', 'wb'))


def pick_pairs(centers, n_pairs):
    # We try to pick some pairs that cover the latent space in some way.
    # This probably could be improved

    # Pick some pairs that are far in the first few principal components.
    zdim = centers.shape[-1]
    pairs = []
    max_k = np.max([n_pairs//2, zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:,k])
        j_idx = np.argmin(centers[:,k])
        pairs.append([i_idx, j_idx])
    
    # Pick some pairs that are far away from each other.
    X = distance_matrix(centers[:,:], centers[:,:])
    for _ in range(n_pairs//2):
        i_idx,j_idx = np.unravel_index(np.argmax(X), X.shape)
        X[i_idx, :] = 0 
        X[:, i_idx] = 0 
        X[j_idx, :] = 0 
        X[:, j_idx] = 0 
        pairs.append([i_idx, j_idx])

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    analyze(args.result_dir, args.outdir, args.zdim, args.n_clusters, args.n_trajectories, args.skip_umap, args.q, args.n_std )
