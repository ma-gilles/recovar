import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "result_dir",
        type=os.path.abspath,
        help="result dir (output dir of pipeline)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )

    parser.add_argument(
        "--latent-points", type=os.path.abspath,
        required=True,
        help="path to latent points (.txt file)",
    )

    parser.add_argument(
        "--Bfactor",  type =float, default=0, help="0"
    )

    parser.add_argument(
        "--n-bins",  type =float, default=50, dest="n_bins",help="number of bins for kernel regression"
    )

    parser.add_argument(
        "--maskrad-fraction",  type =float, default=20, dest="maskrad_fraction",help="Radius of mask used in kernel regression. Default = 20, which means radius = grid_size/20 pixels, or grid_size * voxel_size / 20 angstrom"
    )

    parser.add_argument(
        "--n-min-images",  type =int, default=None, dest="n_min_images",help="minimum number of images to compute kernel regression. Default = 100 for SPA, and 10 particles for tilt series"
    )


    parser.add_argument(
        "--zdim1",  action="store_true", help="Whether dimension 1 is used. This is an annoying corner case for np.loadtxt..."
    )

    parser.add_argument(
        "--no-z-regularization",  action="store_true", dest="no_z_regularization", help="Whether to use z regularization"
    )

    parser.add_argument(
        "--lazy",  action="store_true", help="Whether to use lazy loading")

    parser.add_argument(
        "--particles",  default=None, help="particle stack dataset. In case you want to use a higher resolution stack")

    parser.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )



    return parser


def compute_state(args):

    po = o.PipelineOutput(args.result_dir + '/')

    if args.particles is not None:
        po.params['input_args'].particles = args.particles

    if args.datadir is not None:
        po.params['input_args'].datadir = args.datadir


    if args.latent_points.endswith('.pkl'):
        target_zs = pickle.load(open(args.latent_points, 'rb'))
    elif args.latent_points.endswith('.txt'):
        target_zs = np.loadtxt(args.latent_points)
    else:
        raise ValueError("Target zs should be a .txt or .pkl file")

    output_folder = args.outdir + '/'

    if args.zdim1:
        zdim =1
        target_zs = target_zs[:,None]
    else:
        zdim = target_zs.shape[-1]
        if target_zs.ndim ==1:
            logger.warning("Did you mean to use --zdim1?")
            target_zs = target_zs[None]

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    zdim_key = f"{zdim}_noreg" if args.no_z_regularization else zdim

    cryos = po.get('lazy_dataset') if args.lazy else po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim_key])
    zs = po.get('zs')[zdim_key]
    cov_zs = po.get('cov_zs')[zdim_key]
    noise_variance = po.get('noise_var_used')
    n_bins = args.n_bins
    o.mkdir_safe(output_folder)    
    # logger.addHandler(logging.FileHandler(f"{output_folder}/run.log"))
    logger.info(args)
    o.compute_and_save_reweighted(cryos, target_zs, zs, cov_zs, noise_variance, output_folder, args.Bfactor, n_bins =n_bins, maskrad_fraction = args.maskrad_fraction, n_min_images = args.n_min_images)
    o.move_to_one_folder(output_folder, target_zs.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)
