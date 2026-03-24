import recovar.jax_config
import logging
import numpy as np
from recovar.output import output as o
from recovar import utils
from recovar.data_io import cryoem_dataset
from recovar.heterogeneity import latent_density, embedding
import os, argparse
from recovar.utils import parser_args
logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):


    parser = parser_args.standard_downstream_args(parser)

    parser.add_argument(
        "--zdim", type=int, help="Dimension of latent variable (a single int, not a list)"
    )

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
        help="end points file (txt). It it has more than 2 lines, it will use the first two lines as endpoints. If that's not the case, use --ind to specify them instead. Alternatively, use --z_st and --z_end to specify each one separately.",
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
    po = o.PipelineOutput(recovar_result_dir)

    # Auto-remap stored paths when filesystem has been migrated
    params = getattr(po, "params", None)
    input_args = params.get('input_args') if hasattr(params, "get") else None
    if input_args is not None:
        from recovar.commands.compute_state import _auto_remap_paths
        if args is not None:
            if getattr(args, "particles", None) is not None:
                input_args.particles = args.particles
            if getattr(args, "datadir", None) is not None:
                input_args.datadir = args.datadir
            if getattr(args, "strip_prefix", None) is not None:
                input_args.strip_prefix = args.strip_prefix
        _auto_remap_paths(input_args, recovar_result_dir)

    lazy = bool(getattr(args, "lazy", False))
    # Select reg vs noreg entry names
    coords_entry = 'latent_coords_noreg' if no_z_reg else 'latent_coords'
    precision_entry = 'latent_precision_noreg' if no_z_reg else 'latent_precision'
    contrast_entry = 'contrasts_noreg' if no_z_reg else 'contrasts'

    if hasattr(po, "get_embedding_keys"):
        zs_keys = list(po.get_embedding_keys(coords_entry))
    else:
        zs_keys = list(po.get(coords_entry).keys())

    if zdim is None and len(zs_keys) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")

    elif zdim is None:
        zdim = zs_keys[0]
        logger.info("using zdim=%s", zdim)
    noreg_suffix = '_noreg' if no_z_reg else ''
    logger.info("using zdim=%s%s", zdim, noreg_suffix)
    if output_folder is None:
        raise ValueError("output_folder is required")

    if zdim not in zs_keys:
        logger.error("z-dim not found in results. Options are: %s", ','.join(str(e) for e in zs_keys))
        raise ValueError("Requested zdim was not found in embedding outputs.")

    if hasattr(po, "get_embedding_component"):
        zs = po.get_embedding_component(coords_entry, zdim)
        cov_zs = po.get_embedding_component(precision_entry, zdim)
        contrasts = po.get_embedding_component(contrast_entry, zdim)
    else:
        zs = po.get(coords_entry)[zdim]
        cov_zs = po.get(precision_entry)[zdim]
        contrasts = po.get(contrast_entry)[zdim]

    # Keep memory footprint low for downstream JAX kernels.
    zs = np.asarray(zs, dtype=np.float32)
    cov_zs = np.asarray(cov_zs, dtype=np.float32)
    contrasts = np.asarray(contrasts, dtype=np.float32)

    cryos = po.get('lazy_dataset') if lazy else po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, contrasts)

    if density_path is not None:
        dens_pkl = utils.pickle_load(density_path)
        input_density = dens_pkl['density']
        latent_space_bounds = dens_pkl['latent_space_bounds']
        logger.warning("density dimension is less than zs dimension, truncate zs dimension to match density dimension = %s", input_density.ndim)
        zdim = input_density.ndim
        zs = zs[:, :zdim]
        cov_zs = cov_zs[:, :zdim, :zdim]
    else:
        density, latent_space_bounds = latent_density.compute_latent_space_density(
            zs, cov_zs, pca_dim_max=np.min([4, zs.shape[-1]]), num_points=50, density_option='kde'
        )
        po.params['density'] = density
        input_density = None
        latent_space_bounds = None

    if zs.shape[1] > z_st.shape[0]:
        z_st = np.concatenate([z_st, np.zeros(zs.shape[1] - z_st.shape[0])])
        z_end = np.concatenate([z_end, np.zeros(zs.shape[1] - z_end.shape[0])])
        logger.warning("endpoints are padded with 0 to match zs dimension = %s", zs.shape[1])
    elif zs.shape[1] < z_st.shape[0]:
        z_st = z_st[:zs.shape[1]]
        z_end = z_end[:zs.shape[1]]
        logger.warning("endpoints are truncated to match zs dimension = %s", zs.shape[1])

    if args is not None:
        B_factor = args.Bfactor
        n_bins = args.n_bins
        maskrad_fraction = args.maskrad_fraction
        n_min_particles = args.n_min_particles
    else:
        maskrad_fraction = 20
        n_min_particles = 100

    o.mkdir_safe(output_folder)
    logger.info(args)

    if zdim > 1:
        path_folder = output_folder
        o.mkdir_safe(path_folder)
        full_path, subsampled_path = o.make_trajectory_plots_from_results(
            po, zdim, path_folder, cryos=cryos, z_st=z_st, z_end=z_end, gt_volumes=None,
            n_vols_along_path=n_vols_along_path, plot_llh=False, input_density=input_density,
            latent_space_bounds=latent_space_bounds
        )
        logger.info("path done")

    else:
        path_folder = os.path.join(output_folder, 'path0')
        o.mkdir_safe(path_folder)
        q = 0.03
        zs_1d = np.asarray(zs).reshape(-1)
        pairs = np.percentile(zs_1d, [q, 100 - q])
        z_st = pairs[0]
        z_end = pairs[1]
        subsampled_path = np.linspace(z_st, z_end, n_vols_along_path)[:, None]
    o.compute_and_save_reweighted(
        cryos, subsampled_path, zs, cov_zs, path_folder, B_factor, n_bins,
        maskrad_fraction=maskrad_fraction, n_min_particles=n_min_particles, save_all_estimates=False
    )

def main():
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

    from recovar.output.job import JobDir
    job = JobDir.create(
        outdir=args.outdir,
        command_name="compute_trajectory",
        parent_result_dir=args.result_dir,
        auto_number=(args.outdir is None),
    )
    args.outdir = job.root
    job.start(args)
    try:
        compute_trajectory(args.result_dir, output_folder = args.outdir, zdim= args.zdim, B_factor = args.Bfactor, n_bins = args.n_bins, n_vols_along_path = args.n_vols_along_path, density_path = args.density, no_z_reg = not args.override_z_regularization, z_st = z_st, z_end = z_end, args = args)
        job.complete()
    except Exception:
        job.complete(status="failed")
        raise


if __name__ == "__main__":
    main()
