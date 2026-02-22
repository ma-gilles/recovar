import recovar.config 
import logging
import numpy as np
import warnings
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
    # Support programmatic callers that may pass a minimal args namespace.
    zdim1 = bool(getattr(args, "zdim1", False))
    no_z_regularization = bool(getattr(args, "no_z_regularization", False))
    lazy = bool(getattr(args, "lazy", False))
    n_bins = getattr(args, "n_bins", 50)
    bfactor = getattr(args, "Bfactor", 0.0)
    maskrad_fraction = getattr(args, "maskrad_fraction", 20)
    n_min_particles = getattr(args, "n_min_particles", None)
    save_all_estimates = bool(getattr(args, "save_all_estimates", False))
    apply_global_filtering = bool(getattr(args, "apply_global_filtering", False))
    fsc_mask_radius = getattr(args, "fsc_mask_radius", None)
    fsc_mask_edgewidth = getattr(args, "fsc_mask_edgewidth", None)

    result_dir = os.fspath(args.result_dir)
    outdir = os.fspath(args.outdir)
    po = o.PipelineOutput(os.path.join(result_dir, ""))
    
    params = getattr(po, "params", None)
    input_args = params.get('input_args') if hasattr(params, "get") else None
    particles_override = getattr(args, "particles", None)
    datadir_override = getattr(args, "datadir", None)
    strip_prefix_override = getattr(args, "strip_prefix", None)
    if input_args is not None:
        if particles_override is not None:
            input_args.particles = particles_override
        if datadir_override is not None:
            input_args.datadir = datadir_override
        if strip_prefix_override is not None:
            input_args.strip_prefix = strip_prefix_override
    elif particles_override is not None or datadir_override is not None or strip_prefix_override is not None:
        logger.warning("Pipeline output is missing input_args; ignoring particles/datadir/strip-prefix overrides.")

    # Copy data to temp folder if requested
    path_mapping = None
    if hasattr(args, 'copy_to_folder') and args.copy_to_folder is not None:
        path_mapping = copy_data_from_pipeline_output(po, args.copy_to_folder)

    try:
        latent_points_path = os.fspath(args.latent_points)
        if latent_points_path.endswith('.pkl'):
            if not os.path.isfile(latent_points_path):
                raise FileNotFoundError(f"Latent points file not found: {latent_points_path}")
            with open(latent_points_path, 'rb') as f:
                target_zs = pickle.load(f)
        elif latent_points_path.endswith('.txt'):
            if not os.path.isfile(latent_points_path):
                raise FileNotFoundError(f"Latent points file not found: {latent_points_path}")
            with warnings.catch_warnings():
                # Empty text files produce a UserWarning; we handle emptiness explicitly below.
                warnings.simplefilter("ignore", category=UserWarning)
                target_zs = np.loadtxt(latent_points_path)
        else:
            raise ValueError("Target zs should be a .txt or .pkl file")
        target_zs = np.asarray(target_zs)
        try:
            target_zs = target_zs.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Target zs must be numeric.") from exc
        if target_zs.size == 0:
            raise ValueError("Target zs file is empty.")
        if not np.all(np.isfinite(target_zs)):
            raise ValueError("Target zs contains non-finite values (NaN/Inf).")

        output_folder = os.path.join(outdir, "")

        if zdim1:
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

        if hasattr(po, "get_embedding_keys"):
            zs_keys = po.get_embedding_keys("zs")
            cov_keys = po.get_embedding_keys("cov_zs")
            contrast_keys = po.get_embedding_keys("contrasts")
        else:
            zs_all = po.get('zs')
            cov_zs_all = po.get('cov_zs')
            contrasts_all = po.get('contrasts')
            zs_keys = list(zs_all.keys())
            cov_keys = list(cov_zs_all.keys())
            contrast_keys = list(contrasts_all.keys())

        if zdim not in zs_keys:
            options = ','.join(str(e) for e in zs_keys)
            raise ValueError(f"zdim {zdim} from provided latent points is not found in embedding results. Options are: {options}")

        zdim_key = f"{zdim}_noreg" if no_z_regularization else zdim
        if zdim_key not in zs_keys or zdim_key not in contrast_keys or zdim_key not in cov_keys:
            raise ValueError(
                f"Requested embedding key {zdim_key} is missing in pipeline output zs/contrasts/cov_zs."
            )

        if hasattr(po, "get_embedding_component"):
            contrasts_key = po.get_embedding_component('contrasts', zdim_key)
            zs_key = po.get_embedding_component('zs', zdim_key)
            cov_zs_key = po.get_embedding_component('cov_zs', zdim_key)
        else:
            contrasts_key = contrasts_all[zdim_key]
            zs_key = zs_all[zdim_key]
            cov_zs_key = cov_zs_all[zdim_key]

        # Keep memory footprint low for downstream JAX kernels.
        contrasts_key = np.asarray(contrasts_key).astype(np.float32, copy=False)
        zs_key = np.asarray(zs_key).astype(np.float32, copy=False)
        cov_zs_key = np.asarray(cov_zs_key).astype(np.float32, copy=False)

        cryos = po.get('lazy_dataset') if lazy else po.get('dataset')
        embedding.set_contrasts_in_cryos(cryos, contrasts_key)
        zs = zs_key
        cov_zs = cov_zs_key
        o.mkdir_safe(output_folder)
        logger.info(args)

        # Get the mask from pipeline output for FSC filtering
        fsc_mask = None
        if apply_global_filtering:
            try:
                fsc_mask = po.get('volume_mask')
                logger.info("Using pipeline output volume_mask for FSC filtering")
            except Exception:
                logger.warning("Could not load volume_mask from pipeline output, proceeding without FSC mask")

        o.compute_and_save_reweighted(
            cryos, target_zs, zs, cov_zs, output_folder, bfactor,
            n_bins=n_bins, maskrad_fraction=maskrad_fraction,
            n_min_particles=n_min_particles, save_all_estimates=save_all_estimates,
            apply_global_filtering=apply_global_filtering,
            fsc_mask=fsc_mask,
            fsc_mask_radius=fsc_mask_radius,
            fsc_mask_edgewidth=fsc_mask_edgewidth
        )
        o.move_to_one_folder(output_folder, target_zs.shape[0])
    finally:
        # Clean up temp files at the end (including failures).
        if path_mapping is not None and not getattr(args, "no_cleanup", False):
            cleanup_temp_files(path_mapping)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)

if __name__ == "__main__":
    main()
