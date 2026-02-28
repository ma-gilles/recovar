import recovar.jax_config
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import os, argparse
from recovar.utils_core import cleanup_temp_files, copy_data_from_pipeline_output
logger = logging.getLogger(__name__)
from recovar import parser_args

def add_args(parser: argparse.ArgumentParser):

    parser = parser_args.standard_downstream_args(parser, analyze = True)

    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable (a single int, not a list)"
    )

    parser.add_argument(
        "--n-clusters", dest= "n_clusters", type=int, default=20, help="number of k-means clusters (default 20). The clustering is only used to sample the latent space, and the actual cluster labels are not used to generate volumes."
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


def analyze(recovar_result_dir, output_folder = None, zdim = 4, n_clusters = 40, n_paths = 0, skip_umap = False, q = None, n_std = None, B_factor=0, n_bins=30, n_vols_along_path = 6, skip_centers = False, normalize_kmeans = False, density_path = None, no_z_reg = False, lazy = False, n_min_particles = 0, maskrad_fraction = 0.5, apply_global_filtering = False, fsc_mask_radius = None, fsc_mask_edgewidth = None, args = None):

    po = o.PipelineOutput(recovar_result_dir)

    # Copy data to temp folder if requested
    path_mapping = None
    if args is not None and hasattr(args, 'copy_to_folder') and args.copy_to_folder is not None:
        path_mapping = copy_data_from_pipeline_output(po, args.copy_to_folder)

    try:
        if hasattr(po, "get_embedding_keys"):
            zs_keys = list(po.get_embedding_keys("zs"))
            cov_keys = list(po.get_embedding_keys("cov_zs"))
            contrast_keys = list(po.get_embedding_keys("contrasts"))
            zs_all = cov_zs_all = contrasts_all = None
        else:
            zs_all = po.get('zs')
            cov_zs_all = po.get('cov_zs')
            contrasts_all = po.get('contrasts')
            zs_keys = list(zs_all.keys())
            cov_keys = list(cov_zs_all.keys())
            contrast_keys = list(contrasts_all.keys())

        if zdim is None and len(zs_keys) > 1:
            logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
            raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

        elif zdim is None:
            zdim = zs_keys[0]
            logger.info(f"using zdim={zdim}")
        zdim_key = f"{zdim}_noreg" if no_z_reg else zdim
        contrast_key = zdim_key

        if output_folder is None:
            output_folder = recovar_result_dir + f'/analysis_{zdim_key}/'

        if zdim not in zs_keys:
            logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in zs_keys))
            raise ValueError("Requested zdim was not found in embedding outputs.")

        if zdim_key not in zs_keys or zdim_key not in cov_keys or zdim_key not in contrast_keys:
            raise ValueError(
                f"Requested embedding key {zdim_key} is missing in pipeline output zs/contrasts/cov_zs."
            )

        if hasattr(po, "get_embedding_component"):
            zs = po.get_embedding_component('zs', zdim_key)
            cov_zs = po.get_embedding_component('cov_zs', zdim_key)
            contrasts = po.get_embedding_component('contrasts', contrast_key)
        else:
            zs = zs_all[zdim_key]
            cov_zs = cov_zs_all[zdim_key]
            contrasts = contrasts_all[contrast_key]

        # Keep memory footprint low for downstream JAX kernels.
        zs = np.asarray(zs).astype(np.float32, copy=False)
        cov_zs = np.asarray(cov_zs).astype(np.float32, copy=False)
        contrasts = np.asarray(contrasts).astype(np.float32, copy=False)

        if lazy:
            cryos = po.get('lazy_dataset')
        else:
            cryos = po.get('dataset')
        embedding.set_contrasts_in_cryos(cryos, contrasts)

        # Get the mask from pipeline output for FSC filtering
        fsc_mask = None
        if apply_global_filtering:
            try:
                fsc_mask = po.get('volume_mask')
                logger.info("Using pipeline output volume_mask for FSC filtering")
            except Exception:
                logger.warning("Could not load volume_mask from pipeline output, proceeding without FSC mask")

        if density_path is not None:
            dens_pkl = utils.pickle_load(density_path)
            input_density = dens_pkl['density']
            latent_space_bounds = dens_pkl['latent_space_bounds']
            logger.warning(f"density dimension is less than zs dimension, truncate zs dimension to match density dimension = {input_density.ndim}")
            zdim = input_density.ndim
            zdim_key = f"{zdim}_noreg" if no_z_reg else zdim
            zs = zs[:, :zdim]
            cov_zs = cov_zs[:, :zdim, :zdim]
        else:
            density, latent_space_bounds = latent_density.compute_latent_space_density(
                zs, cov_zs, pca_dim_max=np.min([4, zs.shape[-1]]), num_points=50, density_option='kde'
            )
            po.params['density'] = density
            input_density = None
            latent_space_bounds = None

        def reorder(array):
            return dataset.reorder_to_original_indexing_from_halfsets(array, po.get('particles_halfsets'))

        o.mkdir_safe(output_folder)
        utils.basic_config_logger(output_folder)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.hist(contrasts, bins=50)
        plt.xlabel('Contrast')
        plt.ylabel('Number of particles')
        plt.savefig(os.path.join(output_folder, 'contrast_histogram.png'))
        plt.close()

        zs_unsort = zs
        if normalize_kmeans:
            std = np.std(zs_unsort, axis=0)
            centers, labels = o.kmeans_analysis(os.path.join(output_folder, 'PCA/'), zs_unsort / std, n_clusters=n_clusters)
            centers = centers * std
        else:
            centers, labels = o.kmeans_analysis(os.path.join(output_folder, 'PCA/'), zs_unsort, n_clusters=n_clusters)

        # Kmeans volumes go into kmeans/ subdirectory
        kmeans_dir = os.path.join(output_folder, 'kmeans/')
        o.mkdir_safe(kmeans_dir)
        kmeans_result = {'centers': centers, 'labels': reorder(labels)}
        utils.pickle_dump(kmeans_result, os.path.join(output_folder, 'kmeans_result.pkl'))
        np.savetxt(os.path.join(kmeans_dir, 'centers.txt'), centers)

        if density_path is not None:
            _, z_to_grid = latent_density.get_grid_z_mappings(latent_space_bounds, input_density.shape[0])
            centers_grid = z_to_grid(centers)

            o.mkdir_safe(os.path.join(output_folder, 'density_plots/'))
            o.plot_over_density(input_density, points=centers_grid, annotate=True, plot_folder=os.path.join(output_folder, 'density_plots/'))

            o.mkdir_safe(os.path.join(output_folder, 'density_plots_sliced/'))
            o.plot_over_density(input_density, points=centers_grid, annotate=True, plot_folder=os.path.join(output_folder, 'density_plots_sliced/'), projection_function='slice')

        if (not skip_umap) and (zdim > 1):
            mapper = o.umap_latent_space(zs_unsort)
            o.mkdir_safe(os.path.join(output_folder, 'umap/'))
            utils.pickle_dump(reorder(mapper.embedding_), os.path.join(output_folder, 'umap/umap_embedding.pkl'))
            from recovar import output
            _, kmeans_ind = output.get_nearest_point(zs_unsort, centers)

            o.plot_umap(os.path.join(output_folder, 'umap/'), mapper.embedding_, mapper.embedding_[kmeans_ind])

        if not skip_centers:
            o.compute_and_save_reweighted(
                cryos, centers, zs, cov_zs, kmeans_dir, B_factor, n_bins,
                n_min_particles=n_min_particles, maskrad_fraction=maskrad_fraction,
                apply_global_filtering=apply_global_filtering,
                fsc_mask=fsc_mask,
                fsc_mask_radius=fsc_mask_radius,
                fsc_mask_edgewidth=fsc_mask_edgewidth,
                vol_prefix="center"
            )

        del zs_unsort

        if zdim > 1:
            pairs = pick_pairs(centers, n_paths)
            for pair_idx in range(len(pairs)):
                pair = pairs[pair_idx]
                z_st = centers[pair[0], :]
                z_end = centers[pair[1], :]

                traj_folder = os.path.join(output_folder, f'traj{pair_idx:03d}/')
                o.mkdir_safe(traj_folder)
                full_path, subsampled_path = o.make_trajectory_plots_from_results(
                    po, zdim_key, traj_folder, cryos=cryos, z_st=z_st, z_end=z_end, gt_volumes=None,
                    n_vols_along_path=n_vols_along_path, plot_llh=False, input_density=input_density,
                    latent_space_bounds=latent_space_bounds
                )

                logger.info(f"trajectory {pair_idx} done")
                o.compute_and_save_reweighted(
                    cryos, subsampled_path, zs, cov_zs, traj_folder, B_factor, n_bins,
                    n_min_particles=n_min_particles, maskrad_fraction=maskrad_fraction,
                    apply_global_filtering=apply_global_filtering,
                    fsc_mask=fsc_mask,
                    fsc_mask_radius=fsc_mask_radius,
                    fsc_mask_edgewidth=fsc_mask_edgewidth,
                    vol_prefix="state"
                )

        else:
            traj_folder = os.path.join(output_folder, 'traj000/')
            o.mkdir_safe(traj_folder)
            q = 0.03
            zs_1d = np.asarray(zs).reshape(-1)
            pairs = np.percentile(zs_1d, [q, 100 - q])
            z_st = pairs[0]
            z_end = pairs[1]
            subsampled_path = np.linspace(z_st, z_end, n_vols_along_path)[:, None]
            o.compute_and_save_reweighted(
                cryos, subsampled_path, zs, cov_zs, traj_folder, B_factor, n_bins,
                save_all_estimates=False, n_min_particles=n_min_particles, maskrad_fraction=maskrad_fraction,
                apply_global_filtering=apply_global_filtering,
                fsc_mask=fsc_mask,
                fsc_mask_radius=fsc_mask_radius,
                fsc_mask_edgewidth=fsc_mask_edgewidth,
                vol_prefix="state"
            )

        kmeans_res = {'centers': centers.tolist(), 'pairs': pairs}
        utils.pickle_dump(kmeans_res, os.path.join(output_folder, 'trajectory_endpoints.pkl'))
    finally:
        # Clean up temp files at the end (including failures).
        if path_mapping is not None and args is not None and not getattr(args, "no_cleanup", False):
            cleanup_temp_files(path_mapping)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    logger.info(args)
    analyze(args.result_dir, output_folder = args.outdir, zdim=  args.zdim, n_clusters = args.n_clusters, n_paths= args.n_trajectories, skip_umap = args.skip_umap, B_factor = args.Bfactor, n_bins = args.n_bins, n_vols_along_path = args.n_vols_along_path, skip_centers = args.skip_centers, normalize_kmeans = args.normalize_kmeans, density_path = args.density, no_z_reg = args.no_z_regularization, lazy = args.lazy, n_min_particles = args.n_min_particles, maskrad_fraction = args.maskrad_fraction, apply_global_filtering = args.apply_global_filtering, fsc_mask_radius = args.fsc_mask_radius, fsc_mask_edgewidth = args.fsc_mask_edgewidth, args = args)

if __name__ == "__main__":
    main()
