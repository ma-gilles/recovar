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
from recovar.utils_core import copy_data_to_temp_folder, cleanup_temp_files, copy_data_from_pipeline_output


def add_args(parser: argparse.ArgumentParser):
    parser = parser_args.standard_downstream_args(parser)

    parser.add_argument(
        "--latent-points", type=os.path.abspath,
        required=True,
        help="path to latent points (.txt file). E.g., you can use the output of k-means and input output/analysis_2/centers.txt from analyze.py. Or you can make your own latent points. It should be a .txt file with shape (n_points, zdim).",
    )
    parser.add_argument(
        "--save-all-estimates", action="store_true",
        help="Save all estimates. This is useful for debugging.",
    )

    return parser


def compute_state(args):
    po = o.PipelineOutput(args.result_dir + '/')
    
    if args.particles is not None:
        po.params['input_args'].particles = args.particles

    if args.datadir is not None:
        po.params['input_args'].datadir = args.datadir

    if args.strip_prefix is not None:
        po.params['input_args'].strip_prefix = args.strip_prefix

    # Copy data to temp folder if requested
    path_mapping = None
    if hasattr(args, 'copy_to_folder') and args.copy_to_folder is not None:
        path_mapping = copy_data_from_pipeline_output(po, args.copy_to_folder)


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

    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")

    # for cryo in cryos:
    #     cryo.premultiplied_ctf = True
    # [ cryo.premultiplied_ctf = False for cryo in cryos ] 

    embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim_key])
    zs = po.get('zs')[zdim_key]
    cov_zs = po.get('cov_zs')[zdim_key]
    noise_variance = po.get('noise_var_used')
    n_bins = args.n_bins
    o.mkdir_safe(output_folder)    
    logger.info(args)
    
    # Get the mask from pipeline output for FSC filtering
    fsc_mask = None
    if args.apply_global_filtering:
        try:
            fsc_mask = po.get('volume_mask')
            logger.info("Using pipeline output volume_mask for FSC filtering")
        except:
            logger.warning("Could not load volume_mask from pipeline output, proceeding without FSC mask")
    
    o.compute_and_save_reweighted(
        cryos, target_zs, zs, cov_zs, output_folder, args.Bfactor, 
        n_bins=n_bins, maskrad_fraction=args.maskrad_fraction, 
        n_min_particles=args.n_min_particles, save_all_estimates=args.save_all_estimates,
        apply_global_filtering=args.apply_global_filtering,
        fsc_mask=fsc_mask,
        fsc_mask_radius=args.fsc_mask_radius,
        fsc_mask_edgewidth=args.fsc_mask_edgewidth
    )
    o.move_to_one_folder(output_folder, target_zs.shape[0])

    # Clean up temp files at the end
    if path_mapping is not None and not args.no_cleanup:
        cleanup_temp_files(path_mapping)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)

if __name__ == "__main__":
    main()