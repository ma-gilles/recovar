import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import embedding
import pickle
import os, argparse
logger = logging.getLogger(__name__)
from recovar import parser_args
from recovar.utils_core import cleanup_temp_files, copy_data_from_pipeline_output


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
    zs_all = po.get('zs')
    cov_zs_all = po.get('cov_zs')
    contrasts_all = po.get('contrasts')
    
    input_args = po.params.get('input_args') if hasattr(po, "params") else None
    if input_args is not None:
        if args.particles is not None:
            input_args.particles = args.particles
        if args.datadir is not None:
            input_args.datadir = args.datadir
        if args.strip_prefix is not None:
            input_args.strip_prefix = args.strip_prefix
    elif args.particles is not None or args.datadir is not None or args.strip_prefix is not None:
        logger.warning("Pipeline output is missing input_args; ignoring particles/datadir/strip-prefix overrides.")

    # Copy data to temp folder if requested
    path_mapping = None
    if hasattr(args, 'copy_to_folder') and args.copy_to_folder is not None:
        path_mapping = copy_data_from_pipeline_output(po, args.copy_to_folder)


    if args.latent_points.endswith('.pkl'):
        with open(args.latent_points, 'rb') as f:
            target_zs = pickle.load(f)
    elif args.latent_points.endswith('.txt'):
        target_zs = np.loadtxt(args.latent_points)
    else:
        raise ValueError("Target zs should be a .txt or .pkl file")
    target_zs = np.asarray(target_zs)

    output_folder = args.outdir + '/'

    if args.zdim1:
        if target_zs.ndim > 1 and target_zs.shape[-1] != 1:
            raise ValueError(
                f"--zdim1 expects scalar/1D latent points or Nx1 arrays; got shape {target_zs.shape}"
            )
        zdim =1
        target_zs = np.atleast_1d(target_zs).reshape(-1, 1)
    else:
        if target_zs.ndim == 0:
            raise ValueError("Scalar latent point requires --zdim1.")
        zdim = target_zs.shape[-1]
        if target_zs.ndim ==1:
            logger.warning("Did you mean to use --zdim1?")
            target_zs = target_zs[None]

    if zdim not in zs_all:
        options = ','.join(str(e) for e in zs_all.keys())
        raise ValueError(f"zdim {zdim} from provided latent points is not found in embedding results. Options are: {options}")

    zdim_key = f"{zdim}_noreg" if args.no_z_regularization else zdim
    if zdim_key not in zs_all or zdim_key not in contrasts_all or zdim_key not in cov_zs_all:
        raise ValueError(
            f"Requested embedding key {zdim_key} is missing in pipeline output zs/contrasts/cov_zs."
        )

    cryos = po.get('lazy_dataset') if args.lazy else po.get('dataset')

    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")
    # print("CHANGE THIS STUFF")

    # for cryo in cryos:
    #     cryo.premultiplied_ctf = True
    # [ cryo.premultiplied_ctf = False for cryo in cryos ] 

    embedding.set_contrasts_in_cryos(cryos, contrasts_all[zdim_key])
    zs = zs_all[zdim_key]
    cov_zs = cov_zs_all[zdim_key]
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
