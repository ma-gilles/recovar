import recovar.jax_config 
import logging
import numpy as np
import warnings
from recovar.output import output as o
from recovar.heterogeneity import embedding
import pickle
import os, argparse
logger = logging.getLogger(__name__)
from recovar.utils import parser_args

_PATH_REMAP_ATTRS = (
    "particles",
    "ctf",
    "poses",
    "ind",
    "halfsets",
    "focus_mask",
    "tilt_series_ctf",
)


def _auto_remap_paths(input_args, actual_result_dir: str):
    """Remap stored absolute paths when the data has moved.

    Detects if ``input_args.outdir`` differs from *actual_result_dir* and
    derives an old→new prefix mapping.  Then rewrites all file-path
    attributes (particles, ctf, poses, etc.) using that mapping.
    """
    stored_outdir = getattr(input_args, "outdir", None)
    if stored_outdir is None:
        return
    stored_outdir = os.path.realpath(stored_outdir)
    actual = os.path.realpath(actual_result_dir)
    if stored_outdir == actual:
        return
    # Walk from the end to find the common suffix
    sp = stored_outdir.split(os.sep)
    ap = actual.split(os.sep)
    common = 0
    for i in range(1, min(len(sp), len(ap)) + 1):
        if sp[-i] == ap[-i]:
            common = i
        else:
            break
    if common == 0:
        return
    old_prefix = os.sep.join(sp[:-common])
    new_prefix = os.sep.join(ap[:-common])
    if not old_prefix or not new_prefix or old_prefix == new_prefix:
        return
    logger.info("Auto-remapping data paths: %s -> %s", old_prefix, new_prefix)
    for attr in _PATH_REMAP_ATTRS:
        val = getattr(input_args, attr, None)
        if isinstance(val, str) and val.startswith(old_prefix):
            new_val = new_prefix + val[len(old_prefix):]
            if os.path.exists(new_val):
                setattr(input_args, attr, new_val)
                logger.info("  %s: %s -> %s", attr, val, new_val)


def _load_latent_points(latent_points_path):
    """Load latent points from txt/pkl and validate numeric finite contents."""
    latent_points_path = os.fspath(latent_points_path)
    if latent_points_path.endswith(".pkl"):
        if not os.path.isfile(latent_points_path):
            raise FileNotFoundError(f"Latent points file not found: {latent_points_path}")
        with open(latent_points_path, "rb") as f:
            target_zs = pickle.load(f)
    elif latent_points_path.endswith(".txt"):
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
    return target_zs


def _get_embedding_keys(pipeline_output, coords_entry):
    """Return available z dimensions for the requested embedding entry."""
    if hasattr(pipeline_output, "get_embedding_keys"):
        return pipeline_output.get_embedding_keys(coords_entry)
    return list(pipeline_output.get(coords_entry).keys())


def _get_embedding_components(pipeline_output, zdim, coords_entry, precision_entry, contrast_entry):
    """Fetch embedding arrays for a specific zdim with API fallback support."""
    if hasattr(pipeline_output, "get_embedding_component"):
        return (
            pipeline_output.get_embedding_component(contrast_entry, zdim),
            pipeline_output.get_embedding_component(coords_entry, zdim),
            pipeline_output.get_embedding_component(precision_entry, zdim),
        )
    return (
        pipeline_output.get(contrast_entry)[zdim],
        pipeline_output.get(coords_entry)[zdim],
        pipeline_output.get(precision_entry)[zdim],
    )


def _build_reweighted_halfset_datasets(dataset, *, lazy):
    """Build direct halfset datasets for the reweighted-volume hot path."""
    if not hasattr(dataset, "materialize_halfset_datasets"):
        return None
    if getattr(dataset, "halfset_indices", None) is None:
        return None
    can_reload = getattr(dataset, "can_reload_from_original_images", None)
    if can_reload is None or not can_reload():
        return None
    return dataset.materialize_halfset_datasets(independent=True, lazy=lazy)


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
    po = o.PipelineOutput(result_dir)
    
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
        # Auto-remap stored paths when they no longer exist (filesystem migration)
        _auto_remap_paths(input_args, result_dir)
    elif particles_override is not None or datadir_override is not None or strip_prefix_override is not None:
        logger.warning("Pipeline output is missing input_args; ignoring particles/datadir/strip-prefix overrides.")

    target_zs = _load_latent_points(args.latent_points)

    output_folder = outdir

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

    # Select reg vs noreg entry names
    coords_entry = 'latent_coords_noreg' if no_z_regularization else 'latent_coords'
    precision_entry = 'latent_precision_noreg' if no_z_regularization else 'latent_precision'
    contrast_entry = 'contrasts_noreg' if no_z_regularization else 'contrasts'

    zs_keys = _get_embedding_keys(po, coords_entry)

    if zdim not in zs_keys:
        options = ','.join(str(e) for e in zs_keys)
        raise ValueError(f"zdim {zdim} from provided latent points is not found in embedding results. Options are: {options}")

    contrasts_key, zs_key, cov_zs_key = _get_embedding_components(
        po, zdim, coords_entry, precision_entry, contrast_entry
    )

    # Keep memory footprint low for downstream JAX kernels.
    contrasts_key = np.asarray(contrasts_key, dtype=np.float32)
    zs_key = np.asarray(zs_key, dtype=np.float32)
    cov_zs_key = np.asarray(cov_zs_key, dtype=np.float32)

    cryos = po.get('lazy_dataset') if lazy else po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, contrasts_key)
    reweighted_halfset_datasets = _build_reweighted_halfset_datasets(cryos, lazy=lazy)
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
        except (KeyError, FileNotFoundError):
            logger.warning("Could not load volume_mask from pipeline output, proceeding without FSC mask")

    o.compute_and_save_reweighted(
        cryos, target_zs, zs, cov_zs, output_folder, bfactor,
        n_bins=n_bins, maskrad_fraction=maskrad_fraction,
        n_min_particles=n_min_particles, save_all_estimates=save_all_estimates,
        apply_global_filtering=apply_global_filtering,
        fsc_mask=fsc_mask,
        fsc_mask_radius=fsc_mask_radius,
        fsc_mask_edgewidth=fsc_mask_edgewidth,
        halfset_datasets=reweighted_halfset_datasets,
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)

if __name__ == "__main__":
    main()
