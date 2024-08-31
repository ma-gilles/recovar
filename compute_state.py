import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)
from recovar import parser_args


def add_args(parser: argparse.ArgumentParser):
    parser = parser_args.standard_downstream_args(parser)

    parser.add_argument(
        "--latent-points", type=os.path.abspath,
        required=True,
        help="path to latent points (.txt file). E.g., you can use the output of k-means and input output/analysis_2/centers.txt from analyze.py. Or you can make your own latent points. It should be a .txt file with shape (n_points, zdim).",
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
        logger.error("zdim of provided latent points are not found in embedding results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

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
