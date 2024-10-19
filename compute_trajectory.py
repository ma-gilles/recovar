import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import pickle
import os, argparse
from recovar import parser_args
logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):


    parser = parser_args.standard_downstream_args(parser)

    parser.add_argument(
        "--zdim", type=int, help="Dimension of latent variable (a single int, not a list)"
    )

    # parser.add_argument(
    #     "--no_z_reg", type=int, help="Dimension of latent variable (a single int, not a list)"
    # )
    parser.add_argument(
        "--override_z_regularization", action="store_true", help= "Whether to override z regularization. It probably does not make sense to use this option, because the deconvolved density uses the UNREGULARIZED z's (see paper for why)."
    )

    parser.add_argument(
        "--n-vols-along-path", type=int, default=6, dest="n_vols_along_path", help="number of volumes to compute along each trajectory (default 6)"
    )

    parser.add_argument(
        "--density",
        type=os.path.abspath,
        required=False,
        help="density saved in pkl file, key is 'density' and 'latent_space_bounds",
    )

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument(
        "--ind",
        dest="ind",
        type=list_of_ints,
        default=None,
        help="indices of in list of coords to use as endpoints",
    )

    parser.add_argument(
        "--endpts",
        dest="endpts_file",
        default=None,
        help="end points file (txt). It it has more than 2 lines, it will use the first two lines as endpoints. If that's not the case, use --ind to specify them instead",
    )

    parser.add_argument(
        "--z_st",
        dest="z_st_file",
        default=None,
        help="z_st file (txt)",
    )

    parser.add_argument(
        "--z_end",
        dest="z_end_file",
        default=None,
        help="z_end file (txt)",
    )

    return parser


def compute_trajectory(recovar_result_dir, output_folder = None, zdim = 4,  B_factor=0, n_bins=30, n_vols_along_path = 6, density_path = None, no_z_reg = False, z_st = None, z_end = None, args = None):
    # I kind of like the idea of not passing args, but I'm getting lazy.
    # TODO dont pass args, pass options

    po = o.PipelineOutput(recovar_result_dir + '/')

    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
    
    elif zdim is None:
        zdim = list(po.get('zs').keys())[0]
        logger.info(f"using zdim={zdim}")
    zdim_key = f"{zdim}_noreg" if no_z_reg else zdim
    logger.info(f"using zdim_key={zdim_key}")
    assert output_folder is not None
    # if output_folder is None:
    #     output_folder = recovar_result_dir + f'/output/analysis_{zdim_key}/' 

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    zs = po.get('zs')[zdim_key]
    cov_zs = po.get('cov_zs')[zdim_key]
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
    
    if zs.shape[1] > z_st.shape[0]:
        z_st = np.concatenate([z_st, np.zeros(zs.shape[1] - z_st.shape[0])])
        z_end = np.concatenate([z_end, np.zeros(zs.shape[1] - z_end.shape[0])])
        logger.warning(f"endpoints are padded with 0 to match zs dimension = {zs.shape[1]}")
    elif zs.shape[1] < z_st.shape[0]:
        z_st = z_st[:zs.shape[1]]
        z_end = z_end[:zs.shape[1]]
        logger.warning(f"endpoints are truncated to match zs dimension = {zs.shape[1]}")

    noise_variance = po.get('noise_var_used')
    B_factor = args.Bfactor
    n_bins = args.n_bins
    output_folder_kmeans = output_folder + '/' #+ '/kmeans'+'_'+ str(n_clusters) + '/'    
    o.mkdir_safe(output_folder_kmeans)    
    # logger.addHandler(logging.FileHandler(f"{output_folder_kmeans}/run.log"))
    logger.info(args)

    if zdim > 1:
        path_folder = output_folder_kmeans       
        o.mkdir_safe(path_folder)
        full_path, subsampled_path = o.make_trajectory_plots_from_results(po, zdim_key, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, input_density = input_density, latent_space_bounds = latent_space_bounds)
        logger.info(f"path done")
        # o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, maskrad_fraction = args.maskrad_fraction, n_min_images = args.n_min_images, save_all_estimates = False)
        # move_to_one_folder(path_folder, n_vols_along_path )

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
        # o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, save_all_estimates = False)
        # move_to_one_folder(path_folder, n_vols_along_path )
    o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, maskrad_fraction = args.maskrad_fraction, n_min_images = args.n_min_images, save_all_estimates = False)


from recovar.output import move_to_one_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    if args.ind is not None:
        z_st_ind = args.ind[0]
        z_end_ind = args.ind[1]
    else:
        z_st_ind = 0
        z_end_ind = 1

    if args.endpts_file is not None:
        end_points = np.loadtxt(args.endpts_file)
        z_st = end_points[z_st_ind]
        z_end = end_points[z_end_ind]
    elif args.z_st_file is not None and args.z_end_file is not None:
        z_st = np.loadtxt(args.z_st_file)
        z_end = np.loadtxt(args.z_end_file)
    else:
        raise Exception("end point format wrong. Either pass end points file or z_st_file and z_end_file")

    compute_trajectory(args.result_dir, output_folder = args.outdir, zdim= args.zdim, B_factor = args.Bfactor, n_bins = args.n_bins, n_vols_along_path = args.n_vols_along_path, density_path = args.density, no_z_reg = not args.override_z_regularization, z_st = z_st, z_end = z_end, args = args)

