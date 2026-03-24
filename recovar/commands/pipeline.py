"""Main recovar pipeline command: mean reconstruction, covariance, PCA, embedding."""

import logging
logger = logging.getLogger(__name__)
import recovar.jax_config
import numpy as np
import os, argparse, time, sys
from recovar import utils
from recovar.reconstruction import homogeneous, noise
from recovar.output import output
from recovar.data_io import cryoem_dataset, halfsets
from recovar.core import mask
from recovar.heterogeneity import embedding, principal_components, covariance_estimation
from recovar.output.output_paths import ResultPaths
import recovar.core.fourier_transform_utils as fourier_transform_utils


from recovar.utils.helpers import RobustFileHandler as _RobustFileHandler
from recovar.utils.helpers import RobustStreamHandler as _RobustStreamHandler


def add_args(parser: argparse.ArgumentParser):

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    # ── Required / positional ──────────────────────────────────────────────
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "-o", "--outdir",
        type=os.path.abspath,
        help="Output directory to save model. Required unless --project is used.",
    )
    from recovar.utils.parser_args import add_project_arg
    add_project_arg(parser)
    parser.add_argument(
        "--mask", metavar="mrc", required=True,
        help="Solvent mask (.mrc). Special values: from_halfmaps, sphere, none",
    )

    # ── Dataset loading ────────────────────────────────────────────────────
    data = parser.add_argument_group("Dataset loading")
    data.add_argument(
        "--poses", type=os.path.abspath, default=None,
        help="Image poses (.pkl). Auto-extracted from .star/.cs if omitted",
    )
    data.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, default=None,
        help="CTF parameters (.pkl). Auto-extracted from .star/.cs if omitted",
    )
    data.add_argument(
        "--ind", type=os.path.abspath, metavar="PKL",
        help="Filter images by these indices",
    )
    data.add_argument(
        "--particle-ind", dest="tilt_ind", type=os.path.abspath, metavar="PKL",
        help="Filter particles by these indices (tilt-series only)",
    )
    data.add_argument(
        "--halfsets", default=None, type=os.path.abspath,
        help="Halfset indices (.pkl). If omitted, reads _rlnRandomSubset from star file, or splits randomly",
    )
    data.add_argument(
        "--datadir", type=os.path.abspath,
        help="Path prefix to particle stack for relative paths in .star/.cs files",
    )
    data.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file. "
             "E.g. --strip-prefix Extract/job193 --datadir /your/path/to/",
    )
    data.add_argument(
        "--n-images", default=-1, dest="n_images", type=int,
        help="Number of images to use (for quick runs)",
    )
    data.add_argument(
        "--lazy", action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    data.add_argument(
        "--uninvert-data", dest="uninvert_data", default="automatic",
        help="Invert data sign: true, false, automatic (default)",
    )
    # ── Downsampling ───────────────────────────────────────────────────────
    ds = parser.add_argument_group("Downsampling")
    ds.add_argument(
        "--downsample", type=int, default=256,
        help="Downsample images to this box size (default 256). Skipped if images are already near this size. Use --no-downsample to disable",
    )
    ds.add_argument(
        "--no-downsample", action="store_true", dest="no_downsample", default=False,
        help="Disable automatic downsampling (process at original resolution)",
    )

    # ── Masking ────────────────────────────────────────────────────────────
    mk = parser.add_argument_group("Masking")
    mk.add_argument(
        "--focus-mask", metavar="mrc", dest="focus_mask", default=None, type=os.path.abspath,
        help="Focus mask (.mrc)",
    )
    mk.add_argument(
        "--keep-input-mask", action="store_true", dest="keep_input_mask",
        help="Use input mask as-is (skip thresholding/softening)",
    )
    mk.add_argument(
        "--use-complement-mask", action="store_true", dest="use_complement_mask",
        help="Use complement of focus mask",
    )
    mk.add_argument(
        "--mask-dilate-iter", type=int, default=0, dest="mask_dilate_iter",
        help="Iterations to dilate solvent and focus mask",
    )
    mk.add_argument(
        "--dilated-mask-dilation-iters", type=int, default=None,
        help="How many times to dilate the mask. Default = 6 * volume_shape / 128",
    )
    mk.add_argument(
        "--dont-use-image-mask", dest="dont_use_image_mask", action="store_true",
    )

    # ── Algorithm options ──────────────────────────────────────────────────
    algo = parser.add_argument_group("Algorithm")
    algo.add_argument(
        '--zdim', type=list_of_ints, default=[1, 2, 4, 10, 20],
        help="Dimensions of latent variable (comma-separated). Default=1,2,4,10,20",
    )
    algo.add_argument(
        "--correct-contrast", dest="correct_contrast",
        action=argparse.BooleanOptionalAction, default=False,
        help="Estimate and correct for amplitude scaling variation across images",
    )
    algo.add_argument(
        "--only-mean", action="store_true", dest="only_mean",
        help="Only compute mean (skip covariance/PCA/embedding)",
    )

    # ── Tilt series (cryo-ET) ──────────────────────────────────────────────
    tilt = parser.add_argument_group("Tilt series (cryo-ET)")
    tilt.add_argument(
        "--tomograms", type=os.path.abspath, default=None,
        help="RELION5 tomograms.star file. When provided, automatically converts "
             "RELION5 tomo data to 2D tilt format (via parse_relion5_tomo) before "
             "running the pipeline. Implies --tilt-series.",
    )
    tilt.add_argument(
        "--tilt-series", action="store_true", dest="tilt_series",
        help="Enable tilt-series mode",
    )
    tilt.add_argument(
        "--tilt-series-ctf", default=None, dest="tilt_series_ctf",
        help="CTF model for tilt series: cryoem, relion5, warp. Default = cryoem (SPA) or relion5 (tilt series)",
    )
    tilt.add_argument(
        "--dose-per-tilt", default=None, type=float, dest="dose_per_tilt",
        help="Dose per tilt (default: read from starfile)",
    )
    tilt.add_argument(
        "--angle-per-tilt", default=None, type=float, dest="angle_per_tilt",
        help="Angle per tilt (default: estimated from starfile)",
    )
    tilt.add_argument(
        "--ntilts", default=None, type=int,
        help="Number of tilts per tilt series. Default = all",
    )
    tilt.add_argument(
        '--shared_contrast_across_tilts', action=argparse.BooleanOptionalAction, default=False,
        help="Share contrast across tilts in cryo-ET",
    )
    tilt.add_argument(
        "--premultiplied-ctf", dest='premultiplied_ctf', action="store_true",
        help="Use premultiplied CTF (images already multiplied by CTF)",
    )

    # ── Performance / GPU ──────────────────────────────────────────────────
    perf = parser.add_argument_group("Performance")
    perf.add_argument(
        "--gpu-gb", default=None, type=float, dest="gpu_memory",
        help="GPU memory to use in GB (default: all)",
    )
    perf.add_argument(
        "--multi-gpu", action="store_true", dest="multi_gpu",
        help="Enable multi-GPU parallelization for covariance computation",
    )
    perf.add_argument(
        "--n-gpus", type=int, default=None, dest="n_gpus",
        help="Number of GPUs to use (default: all available)",
    )
    perf.add_argument(
        "--accept-cpu", dest="accept_cpu", action="store_true",
        help="Accept running on CPU if no GPU is found",
    )
    perf.add_argument(
        "--low-memory-option", dest="low_memory_option", action="store_true",
        help="Use lower memory options for covariance estimation",
    )
    perf.add_argument(
        "--very-low-memory-option", dest="very_low_memory_option", action="store_true",
        help="Use lowest memory options for covariance estimation",
    )

    # ── Advanced / debugging ─────────────────────────────────────────────
    adv = parser.add_argument_group("Advanced")
    adv.add_argument(
        "--noise-model", dest="noise_model", default="radial",
        help="Noise model: radial (default) or white",
    )
    adv.add_argument(
        "--mean-fn", dest="mean_fn", default="triangular",
        help="Mean function: triangular (default) or triangular_reg",
    )
    adv.add_argument(
        "--do-over-with-contrast", dest="do_over_with_contrast",
        action=argparse.BooleanOptionalAction, default=None,
        help="Re-run once contrast is estimated. Default = same as --correct-contrast",
    )
    adv.add_argument(
        "--ignore-zero-frequency", dest="ignore_zero_frequency", action="store_true",
        help="Ignore zero frequency (useful when images are normalized to 0 mean). "
             "Experimental: inflates DC noise by 1e16",
    )
    adv.add_argument(
        "--padding", type=int, default=0,
        help="Real-space padding",
    )
    adv.add_argument(
        "--new-noise-est", dest='new_noise_est', action="store_true",
        help="Use new noise estimation",
    )
    adv.add_argument(
        '--use_reg_mean_in_contrast', action=argparse.BooleanOptionalAction, default=True,
    )
    adv.add_argument(
        "--multi-zdim-embedding", dest="multi_zdim_embedding",
        action=argparse.BooleanOptionalAction, default=False,
        help="Experimental: single-pass embedding for all zdims (can be slower on some datasets)",
    )
    adv.add_argument(
        "--keep-intermediate", dest="keep_intermediate", action="store_true",
        help="Save intermediate results (for debugging)",
    )
    adv.add_argument(
        "--test-covar-options", dest="test_covar_options", action="store_true",
        help="Test different covariance estimation options (development only)",
    )
    return parser


# ---------------------------------------------------------------------------
# Helper functions — extracted from standard_recovar_pipeline for clarity
# ---------------------------------------------------------------------------
# NOTE(refactor): _peek_image_size and _resolve_downsample could live in
# data_io, but they depend on argparse args and are only used here, so
# moving them would require an intermediate data class.  Low priority.

def _peek_image_size(particles_file: str, datadir: str = '', strip_prefix=None) -> int:
    """Get the image box size D without constructing a full ImageLoader.

    For RELION 3.1 .star files ``_rlnImageSize`` lives in the optics block at
    the very top of the file (typically fewer than 50 lines).  We read only
    until that value is found and return immediately, skipping the potentially
    huge particle data table and avoiding the MRC-header opens that
    ``StarLoader`` performs for every unique stack file.

    For .mrc/.mrcs files we open with ``header_only=True`` (~1 kB read).
    For .cs files we load just the numpy structured array header.
    Falls back to a full ``load_images()`` call for anything not handled above.
    """
    ext = particles_file.rsplit('.', 1)[-1].lower()

    if ext == 'star':
        # Fast path: scan only the optics block for _rlnImageSize.
        in_optics = False
        cols: list[str] = []
        with open(particles_file) as fh:
            for line in fh:
                s = line.strip()
                if s == 'data_optics':
                    in_optics = True
                    continue
                if not in_optics:
                    continue
                if s.startswith('data_') and s != 'data_optics':
                    break  # left optics block without finding _rlnImageSize
                if s.startswith('_'):
                    cols.append(s.split()[0])
                elif s and not s.startswith(('#', 'loop_')) and cols:
                    vals = s.split()
                    if '_rlnImageSize' in cols and len(vals) >= len(cols):
                        return int(float(vals[cols.index('_rlnImageSize')]))
        # RELION 3.0 / fallback: open the first MRC file referenced
        # (just one header open instead of all unique stacks)
        with open(particles_file) as fh:
            for line in fh:
                if '@' in line:
                    parts = line.strip().split('@')
                    if len(parts) == 2:
                        mrcs_path = parts[1].split()[0]
                        if strip_prefix and mrcs_path.startswith(strip_prefix):
                            mrcs_path = mrcs_path[len(strip_prefix):].lstrip('/')
                        full = os.path.join(datadir or os.path.dirname(particles_file), mrcs_path)
                        if os.path.exists(full):
                            import mrcfile
                            with mrcfile.open(full, mode='r', header_only=True) as mrc:
                                return int(mrc.header.ny)
                        break

    elif ext in ('mrc', 'mrcs'):
        import mrcfile
        with mrcfile.open(particles_file, mode='r', header_only=True) as mrc:
            return int(mrc.header.ny)

    elif ext == 'cs':
        cs_data = np.load(particles_file)
        if 'blob/shape' in cs_data.dtype.names:
            return int(cs_data['blob/shape'][0][1])

    elif ext == 'txt':
        with open(particles_file) as fh:
            first = fh.readline().strip()
        if first:
            import mrcfile
            with mrcfile.open(first, mode='r', header_only=True) as mrc:
                return int(mrc.header.ny)

    # Generic fallback — expensive for .star (opens all MRC headers)
    from recovar.data_io.image_loader import load_images
    loader = load_images(particles_file, lazy=True,
                         datadir=datadir or '', strip_prefix=strip_prefix)
    return loader.image_size

def _resolve_downsample(args):
    """Decide whether downsampling is actually needed.

    Skips downsampling if --no-downsample was passed, if the original image
    size is already <= the target, or if it is within 12.5% of the target
    (not worth the overhead).  Sets ``args.downsample`` to ``None`` when
    skipping.
    """
    if getattr(args, 'no_downsample', False):
        logger.info("Downsampling disabled by --no-downsample")
        args.downsample = None
        return

    if args.downsample is None:
        return

    orig_D = _peek_image_size(
        args.particles,
        datadir=getattr(args, 'datadir', None) or '',
        strip_prefix=getattr(args, 'strip_prefix', None),
    )
    target_D = args.downsample

    if orig_D <= target_D:
        logger.info(
            "Image size %d <= downsample target %d, skipping downsampling",
            orig_D, target_D,
        )
        args.downsample = None
        return

    # Skip if within 12.5% — not worth the overhead
    threshold = int(target_D * 1.125)
    if orig_D <= threshold:
        logger.info(
            "Image size %d is close to target %d (threshold %d), "
            "skipping downsampling",
            orig_D, target_D, threshold,
        )
        args.downsample = None
        return

    logger.info("Will downsample images from %d to %d", orig_D, target_D)


def _check_uninvert_data(means, dataset, args):
    """Check if data needs sign inversion based on the mean estimate.

    In cryo-EM, the convention is that protein density is positive in real
    space. If the estimated mean has negative density in the protein region,
    the sign of the data (and means) is flipped.
    """
    mean_real = fourier_transform_utils.get_idft3(means.combined.reshape(dataset.volume_shape))
    radial_mask = dataset.get_volume_radial_mask(dataset.grid_size // 3).reshape(dataset.volume_shape)
    uninvert_check = np.sum(mean_real.real ** 3 * radial_mask) < 0

    if args.uninvert_data == 'automatic':
        if uninvert_check:
            means.negate()
            dataset.data_multiplier = -1 * dataset.data_multiplier
            args.uninvert_data = "true"
            logger.warning('sum(mean) < 0! Swapping sign of data (uninvert-data = true). If this is not what you want, explicitely set uninvert-data=False ')
        else:
            logger.info('setting (uninvert-data = false)')
            args.uninvert_data = "false"
    elif uninvert_check:
        logger.warning('sum(mean) < 0! Data probably needs to be inverted! set --uninvert-data=true (or automatic)')

## TODO I'd like to simplify the logic here in this function, seems confusing
def _estimate_noise(dataset, means, dilated_volume_mask, batch_size, args, noise_model):
    """Estimate radial noise variance from outside-mask and upper-bound methods.

    Returns a dict with all noise-related quantities needed by the pipeline.
    """
    use_new_noise_fn = args.new_noise_est or args.premultiplied_ctf
    logger.info("Using new noise estimation function?: %s", use_new_noise_fn)

    noise_time = time.time()
    if use_new_noise_fn:
        masked_image_PS, image_PS = noise.fit_noise_model_to_images(
            dataset, dilated_volume_mask, means.combined, None,
            batch_size=batch_size, invert_mask=True, disc_type='linear_interp')
        logger.info("Using new noise estimation with linear_interp discretization")
    elif args.mask.endswith(".mrc"):
        masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(
            dataset, dilated_volume_mask, batch_size)
        white_noise_var_outside_mask = noise.estimate_white_noise_variance_from_mask(
            dataset, dilated_volume_mask, batch_size)
        _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
            dataset, dilated_volume_mask, batch_size)
    else:
        masked_image_PS, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
            dataset, dilated_volume_mask, batch_size)

    radial_noise_var_outside_mask = masked_image_PS
    white_noise_var_outside_mask_val = np.median(masked_image_PS)

    if use_new_noise_fn and noise_model not in ("radial", "radial_per_tilt"):
        raise ValueError(f"new noise fn only works with radial noise model, got {noise_model}")

    logger.info("time to estimate noise is %s", time.time() - noise_time)
    utils.report_memory_device(logger=logger)

    noise_time = time.time()
    if use_new_noise_fn:
        radial_ub_noise_var, _ = noise.fit_noise_model_to_images(
            dataset, dilated_volume_mask, means.combined, None,
            batch_size=batch_size, invert_mask=False, disc_type='linear_interp')
    else:
        radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
            dataset, means.combined, dilated_volume_mask, batch_size)
    logger.info("time to upper bound noise is %s", time.time() - noise_time)

    utils.report_memory_device(logger=logger)
    radial_noise_var_ubed = np.where(
        radial_noise_var_outside_mask > radial_ub_noise_var,
        radial_ub_noise_var, radial_noise_var_outside_mask)

    if noise_model == "white":
        noise_var_used = np.ones_like(radial_noise_var_ubed) * white_noise_var_outside_mask_val
    else:
        noise_var_used = radial_noise_var_ubed

    if (noise_var_used < 0).any():
        logger.info("Negative noise variance detected. Setting to image power spectrum / 10")
    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

    utils.report_memory_device(logger=logger)
    return {
        'noise_var_used': noise_var_used,
        'radial_noise_var_outside_mask': radial_noise_var_outside_mask,
        'radial_ub_noise_var': radial_ub_noise_var,
        'white_noise_var_outside_mask': white_noise_var_outside_mask_val,
        'image_PS': image_PS,
        'masked_image_PS': masked_image_PS,
    }

## TODO perhaps should move, and complement mask should be better documented in the parse args/documentation
## The principal goes is to have "two embeddings", to "factor" out the heterogeneity in a part of molecule we don't care about
## Rather than ignoring it which may cause interference with the one we care about
def _build_focus_masks(args, means, volume_mask, volume_shape, dataset):
    """Build focus masks and optional complement mask."""
    if args.focus_mask is not None:
        focus_mask, _ = mask.masking_options(
            args.focus_mask, means, volume_shape, dataset.dtype_real,
            args.mask_dilate_iter, args.keep_input_mask)
    else:
        focus_mask = volume_mask

    if args.use_complement_mask:
        complement_mask = (volume_mask > 0.90) * 1.0 - (focus_mask > 0.9) * 1.0
        complement_mask = (complement_mask > 0)
        from recovar.core import mask as mask_fn
        complement_mask = np.array(mask_fn.soften_volume_mask(complement_mask, 3).astype(np.float32))
        return [complement_mask, focus_mask]
    else:
        return [focus_mask]


def _compute_embeddings(means, u, s, dataset, volume_mask, options, gpu_memory,
                        focus_masks, zdim_for_rest, args):
    """Compute per-image embeddings for all requested zdim values.

    By default uses the legacy per-zdim embedding loops (reg + noreg), which are
    currently more robust performance-wise across datasets. The experimental
    single-pass multi-zdim path can be enabled with ``--multi-zdim-embedding``.

    Returns six dicts, all keyed by zdim (int):
        (latent_coords, latent_coords_noreg,
         latent_precision, latent_precision_noreg,
         contrasts, contrasts_noreg)
    """
    num_foc_masks = len(focus_masks)
    n_pcs_list = [(num_foc_masks - 1) * zdim_for_rest + zdim
                  for zdim in options.zs_dim_to_test]

    emb_time = time.time()

    use_multi_zdim = (not args.tilt_series) and bool(getattr(args, "multi_zdim_embedding", False))
    if use_multi_zdim:
        logger.info("Embedding mode: single-pass multi-zdim (experimental)")
        # Fast path: single data pass for all zdims
        zs_reg, zs_noreg = embedding.get_per_image_embedding_multi_zdim(
            means.combined, u['rescaled'], s['rescaled'], n_pcs_list,
            dataset, volume_mask, gpu_memory,
            contrast_option=options.contrast,
            ignore_zero_frequency=options.ignore_zero_frequency,
        )
        latent_coords         = {zdim: zs_reg[n_pcs][0]   for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
        latent_precision      = {zdim: zs_reg[n_pcs][1]   for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
        contrasts             = {zdim: zs_reg[n_pcs][2]   for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
        latent_coords_noreg   = {zdim: zs_noreg[n_pcs][0] for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
        latent_precision_noreg= {zdim: zs_noreg[n_pcs][1] for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
        contrasts_noreg       = {zdim: zs_noreg[n_pcs][2] for zdim, n_pcs in zip(options.zs_dim_to_test, n_pcs_list)}
    else:
        if args.tilt_series:
            logger.info("Embedding mode: per-zdim loops (tilt-series)")
        else:
            logger.info("Embedding mode: per-zdim loops (default)")
        latent_coords = {}
        latent_coords_noreg = {}
        latent_precision = {}
        latent_precision_noreg = {}
        contrasts = {}
        contrasts_noreg = {}
        for zdim, n_pcs_to_use in zip(options.zs_dim_to_test, n_pcs_list):
            z_time = time.time()
            latent_coords[zdim], latent_precision[zdim], contrasts[zdim], _ = embedding.get_per_image_embedding(
                means.combined, u['rescaled'], s['rescaled'], n_pcs_to_use,
                dataset, volume_mask, gpu_memory, 'linear_interp',
                contrast_grid=None, contrast_option=options.contrast,
                ignore_zero_frequency=options.ignore_zero_frequency)
            logger.info("embedding time for zdim=%s: %s", zdim, time.time() - z_time)
            z_time = time.time()
            latent_coords_noreg[zdim], latent_precision_noreg[zdim], contrasts_noreg[zdim], _ = embedding.get_per_image_embedding(
                means.combined, u['rescaled'], s['rescaled'] * 0 + np.inf, n_pcs_to_use,
                dataset, volume_mask, gpu_memory, 'linear_interp',
                contrast_grid=None, contrast_option=options.contrast,
                ignore_zero_frequency=options.ignore_zero_frequency)
            logger.info("embedding time for zdim=%s_noreg: %s", zdim, time.time() - z_time)

    logger.info("total embedding time (all zdims): %s", time.time() - emb_time)
    return (latent_coords, latent_coords_noreg,
            latent_precision, latent_precision_noreg,
            contrasts, contrasts_noreg)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def standard_recovar_pipeline(args):
    st_time = time.time()

    # --- Auto-convert RELION5 tomo data if --tomograms is provided ---
    if getattr(args, 'tomograms', None) is not None:
        from recovar.commands import parse_relion5_tomo
        args.tilt_series = True
        converted_star = os.path.join(args.outdir, "particles_2d.star")
        os.makedirs(args.outdir, exist_ok=True)
        logger.info("Converting RELION5 tomo data: %s + %s -> %s",
                     args.tomograms, args.particles, converted_star)
        parse_relion5_tomo.convert(
            args.tomograms, args.particles, converted_star,
        )
        args.particles = converted_star
        logger.info("Conversion complete. Proceeding with pipeline.")

    # --- Validate poses/ctf availability ---
    if args.poses is None or args.ctf is None:
        ext = args.particles.rsplit('.', 1)[-1].lower()
        if ext not in ('star', 'cs'):
            raise ValueError(
                "--poses and --ctf are required when particles file is not .star or .cs. "
                "Provide --poses and --ctf, or use a .star/.cs particles file."
            )
        if args.poses is None:
            logger.info("No --poses provided; will auto-extract from %s", args.particles)
        if args.ctf is None:
            logger.info("No --ctf provided; will auto-extract from %s", args.particles)

    # --- Setup ---
    if args.mask.endswith(".mrc"):
        args.mask = os.path.abspath(args.mask)

    if (not args.accept_cpu) and (not utils.jax_has_gpu()):
        raise ValueError("No GPU found. Set --accept-cpu if you really want to run on CPU (probably not). More likely, you want to check that JAX has been properly installed with GPU support.")

    paths = ResultPaths(args.outdir)
    paths.ensure_dirs()
    with open(paths.command_txt, "w") as text_file:
        text_file.write('python ' + ' '.join(sys.argv))

    # CTF defaults
    if args.tilt_series_ctf is None:
        args.tilt_series_ctf = 'relion5' if args.tilt_series else 'cryoem'
        logger.info("Setting tilt_series_ctf to %s", args.tilt_series_ctf)

    if args.tilt_series and args.dose_per_tilt is not None:
        logger.warning("dose_per_tilt is provided, but tilt_series_ctf is set to using starfile = -B_fac/4 (by default). Thus, dose_per_tilt will not be used.")

    if (args.tilt_series_ctf == 'v2_scale_from_star') and (args.angle_per_tilt is not None):
        logger.warning("angle_per_tilt is provided, but tilt_series_ctf is set to using scale from inputfile (by default). Thus, angle_per_tilt will not be used.")

    if args.do_over_with_contrast is None:
        args.do_over_with_contrast = args.correct_contrast

    # Use RobustFileHandler to tolerate stale NFS/GPFS file handles
    # (common on HPC clusters where SLURM output files go stale).
    log_fmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s'
    logging.basicConfig(
        format=log_fmt,
        level=logging.INFO,
        force=True,
        handlers=[
            _RobustFileHandler(paths.run_log),
            _RobustStreamHandler(),
        ])

    logger.info(args)

    # --- Resolve downsample target ---
    _resolve_downsample(args)

    # --- Auto pre-downsample to disk if requested ---
    if getattr(args, 'downsample', None) is not None:
        from recovar.commands.downsample import downsample_to_disk

        # Save original values for metadata before swapping
        args._original_particles = args.particles
        args._downsample_applied = args.downsample

        ds_dir = os.path.join(args.outdir, "downsampled")
        ds_mrcs = os.path.join(ds_dir, f"particles.{args.downsample}.mrcs")

        if os.path.exists(ds_mrcs):
            logger.info("Using cached downsampled images: %s", ds_mrcs)
        else:
            logger.info("Pre-downsampling images to D=%d ...", args.downsample)
            downsample_to_disk(
                particles_file=args.particles,
                target_D=args.downsample,
                outdir=ds_dir,
                datadir=getattr(args, 'datadir', None) or "",
                strip_prefix=getattr(args, 'strip_prefix', None),
            )

        ds_star = os.path.join(ds_dir, f"particles.{args.downsample}.star")

        # Swap to downsampled data (STAR has full metadata for both CS and STAR input)
        args.particles = ds_star
        args.downsample = None
        args.datadir = None
        if hasattr(args, 'strip_prefix'):
            args.strip_prefix = None

    # --- Load dataset ---
    ind_split = halfsets.resolve_halfset_indices(args)
    dataset_spec = halfsets.HalfsetDatasetSpec.from_args(args)
    logger.info("Data loading configuration:")
    logger.info("  Particles file: %s", dataset_spec.particles_file)
    logger.info("  Poses file: %s", dataset_spec.poses_file or "(auto-extract from particles)")
    logger.info("  CTF file: %s", dataset_spec.ctf_file or "(auto-extract from particles)")
    if dataset_spec.downsample_D:
        logger.info("  Downsample to: %s", dataset_spec.downsample_D)
    if dataset_spec.datadir:
        logger.info("  Datadir: %s", dataset_spec.datadir)

    options = utils.make_algorithm_options(args)

    ## TODO this is a big one, so do with care. I wonder if there is a better way to handle this logic.
    ## Could we instead store 'one' dataset and the indices instead of two different objects, then do a clevery use of iterators
    ## The current way to just have two of these objects around which is not great.
    ds = halfsets.load_halfset_dataset(dataset_spec, ind_split=ind_split, lazy=args.lazy)

    ## TODO: log this. Also document it better. Also I'd like a warning or something if I say "allocate this much" and peak gpu memory ends up being more than that
    ## So that it can be fixed in the future
    if args.gpu_memory is not None:
        utils.set_gpu_memory_limit(args.gpu_memory)
        logger.info("GPU memory limited to %.1f GB (requested via --gpu-gb)", args.gpu_memory)
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = ds.volume_shape

    ## TODO: a big one: I want batch size logic to be separated from rest of utils,
    # and have a better/more transparent way this is done, to make future development easier
    ## Right now, all these numbers were basically found "by hand" by trying to maximize gpu memory, without
    ## breaking, but it is  not robust
    ## Also, I would like some slightly better formatting/more informational logging here
    batch_size = utils.get_image_batch_size(ds.grid_size, gpu_memory)
    logger.info("image batch size: %s", batch_size)
    logger.info("volume batch size: %s", utils.get_vol_batch_size(ds.grid_size, gpu_memory))
    logger.info("column batch size: %s", utils.get_column_batch_size(ds.grid_size, gpu_memory))
    logger.info("number of images: %s", ds.n_images)
    logger.info("image size: %sx%s", ds.grid_size, ds.grid_size)
    utils.report_memory_device(logger=logger)

    # --- Initial noise estimate from half-map 0 ---
    # Preserve main-branch behavior exactly: the initial scalar noise estimate
    # comes from the first halfset dataset, not from the unified full cryoem_dataset.
    noise_estimate_dataset = ds.get_halfset_dataset(
        0,
        independent=bool(ds.tilt_series_flag),
        lazy=args.lazy,
    )
    noise_var_from_hf, _ = noise.estimate_noise_variance(noise_estimate_dataset, batch_size)
    valid_idx = ds.get_valid_frequency_indices()
    noise_model = args.noise_model

    # --- Contrast correction repeat logic ---
    n_repeats = 2 if args.do_over_with_contrast else 1
    if args.do_over_with_contrast and not args.correct_contrast:
        logger.warning("Do over with contrast, but contrast correction is off. Setting contrast correction to on")
        args.correct_contrast = True
        # "_qr" = covariance columns orthogonalized against mean when
        # contrast correction is on (paper S3.2).
        options.contrast = "contrast_qr"

    # Per-image amplitude scaling correction (called "contrast" internally,
    # "per_image_scale" in cryoSPARC).
    if args.shared_contrast_across_tilts:
        options.contrast += '_shared'
        logger.info("Setting contrast to shared")

    # Initialize noise model on the single dataset
    if noise_model == "radial":
        ds.set_radial_noise_model(None)
        logger.info("Setting noise model to radial")
    elif noise_model in ('radial_per_tilt', 'radial-per-tilt'):
        ds.set_variable_radial_noise_model(None)
        logger.info("Setting noise model to radial_per_tilt")
    else:
        raise ValueError(f"noise model {noise_model} not recognized")

    contrasts_for_second = None
    for repeat in range(n_repeats):

        if repeat == 1:
            ndim = 10 if 10 in options.zs_dim_to_test else int(np.median(options.zs_dim_to_test))
            logger.warning("repeating with contrast of zdim=%s", ndim)
            contrasts_for_second = est_contrasts[ndim]
            contrasts_for_second /= np.mean(contrasts_for_second)
            # est_contrasts is in halfset-concatenated order (per-image);
            # reindex to original dataset order before applying to the
            # unified cryoem_dataset.  Always image-level, even for tilt-series.
            contrasts_local = cryoem_dataset.reorder_to_dataset_indexing(
                contrasts_for_second, ds,
                use_tilt_indices=False)
            ds.set_contrasts(contrasts_local)
            options.contrast = "contrast"

        ##TODO: mean functions return a dict with volume sized arrays.
        ## I think none of them are on gpu which is good (nothing should be on gpu that absolutely has to be)
        ## But still can build on cpu memory. I would like any ararys which is not used downstream to be 
        ## deallocated right away, or after they're saved to disk.
        ## Also I don't like the use of dicts, to return arguments, this should probably be removed

        # --- Compute mean ---
        if args.mean_fn == 'triangular':
            means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
                ds, 2 * batch_size, noise_variance=noise_var_from_hf, use_regularization=False)
        elif args.mean_fn == 'triangular_reg':
            means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
                ds, 5 * batch_size, noise_variance=noise_var_from_hf, use_regularization=True)
        else:
            raise ValueError(f"mean function {args.mean_fn} not recognized")
        utils.report_memory_device(logger=logger)

        # --- Check sign and uninvert if needed ---
        _check_uninvert_data(means, ds, args)

        if means.combined.dtype != ds.dtype:
            logger.warning("mean estimate is in type: %s", means.combined.dtype)
            means.combined = means.combined.astype(ds.dtype)

        logger.info("mean computed in %s", time.time() - st_time)
        utils.report_memory_device(logger=logger)

        # --- Compute mask ---
        volume_mask, dilated_volume_mask = mask.masking_options(
            args.mask, means, volume_shape, ds.dtype_real,
            args.mask_dilate_iter, args.keep_input_mask, args.dilated_mask_dilation_iters)

        # --- Save mean and mask volumes ---
        paths.ensure_volumes_dir()
        # save_volume appends .mrc, so strip the extension from the path
        output.save_volume(means.combined, os.path.splitext(paths.mean_volume)[0], volume_shape,
                      from_ft=True, voxel_size=ds.voxel_size)
        output.save_volume(means.corrected0, os.path.splitext(paths.mean_half1_unfil)[0], volume_shape,
                      from_ft=True, voxel_size=ds.voxel_size)
        output.save_volume(means.corrected1, os.path.splitext(paths.mean_half2_unfil)[0], volume_shape,
                      from_ft=True, voxel_size=ds.voxel_size)
        output.save_volume(volume_mask, os.path.splitext(paths.mask_volume)[0], volume_shape,
                      from_ft=False, voxel_size=ds.voxel_size)

        # Filter and save mean via local resolution
        from recovar.heterogeneity import locres
        half1 = fourier_transform_utils.get_idft3(means.corrected0.reshape(volume_shape))
        half2 = fourier_transform_utils.get_idft3(means.corrected1.reshape(volume_shape))
        best_filtered_nob, _, _, _, _ = locres.local_resolution(
            half1, half2, 0, ds.voxel_size, use_filter=True, fsc_threshold=1/7, use_v2=True)
        output.save_volume(best_filtered_nob, os.path.splitext(paths.mean_filtered)[0], volume_shape,
                      from_ft=False, voxel_size=ds.voxel_size)

        if args.only_mean:
            return

        # --- Build focus masks ---
        focus_masks = _build_focus_masks(args, means, volume_mask, volume_shape, ds)

        # --- Estimate noise ---
        noise_result = _estimate_noise(ds, means, dilated_volume_mask, batch_size, args, noise_model)
        noise_var_used = noise_result['noise_var_used']
        noise.update_noise_variance(noise_var_used, ds)

        # Upper bound noise using variance estimate
        variance_est, ub_noise_var_by_var_est = noise.upper_bound_noise_by_signal_p_noise_dispatched(
            noise_var_used, ds, means, batch_size, dilated_volume_mask)

        # Compute variance with regularization
        # //2: variance computation with cubic disc_type needs ~2x memory per image (spline coefficients)
        variance_est, _, variance_fsc, _, noise_p_variance_est = covariance_estimation.compute_variance(
            ds, means.combined, utils.safe_batch_size(batch_size // 2), dilated_volume_mask,
            use_regularization=True, disc_type='cubic')

        utils.report_memory_device(logger=logger)

        # --- Covariance options ---
        covariance_options = covariance_estimation.get_default_covariance_computation_options(ds.grid_size)

        if args.low_memory_option:
            logger.info("Using low-memory covariance options (reduced sampling)")
            covariance_options['sampling_n_cols'] = 50
            covariance_options['randomized_sketch_size'] = 100
            covariance_options['n_pcs_to_compute'] = 100
            covariance_options['sampling_avoid_in_radius'] = 3

        if args.very_low_memory_option:
            logger.warning(
                "Using very-low-memory covariance options — results will be "
                "degraded (fewer columns, smaller sketch).  Use only for "
                "quick tests or CPU runs."
            )
            covariance_options['sampling_n_cols'] = 25
            covariance_options['randomized_sketch_size'] = 35
            covariance_options['n_pcs_to_compute'] = 30
            covariance_options['sampling_avoid_in_radius'] = 3

        if args.dont_use_image_mask:
            covariance_options['mask_images_in_proj'] = False
            covariance_options['mask_images_in_H_B'] = False

        # --- Compute principal components ---
        num_foc_masks = len(focus_masks)
        u = []
        s = []
        zdim_for_rest = 20
        n_pcs_to_keep = np.max(np.append(options.zs_dim_to_test, 50))

        ignore_zero_frequency = options.ignore_zero_frequency
        for idx, focus_mask in enumerate(focus_masks):
            u_this, s_this, covariance_cols, picked_frequencies, column_fscs = \
                principal_components.estimate_principal_components(
                    ds, options, means, mean_prior, focus_mask, dilated_volume_mask,
                    valid_idx, batch_size, gpu_memory_to_use=gpu_memory,
                    covariance_options=covariance_options,
                    variance_estimate=variance_est['combined'],
                    use_reg_mean_in_contrast=args.use_reg_mean_in_contrast,
                    use_multi_gpu=args.multi_gpu, n_gpus=args.n_gpus,
                    )
            if idx == num_foc_masks - 1:
                s.append(s_this['rescaled'][:n_pcs_to_keep].copy())
                u.append(u_this['rescaled'][:, :n_pcs_to_keep].copy())
            else:
                s.append(s_this['rescaled'][:zdim_for_rest].copy())
                u.append(u_this['rescaled'][:, :zdim_for_rest].copy())
            del u_this, s_this

        u = {'rescaled': np.concatenate(u, axis=1), 'real': None}
        s = {'rescaled': np.concatenate(s, axis=0), 'real': None}
        options.ignore_zero_frequency = ignore_zero_frequency


        # Validate PCA results
        if not np.all(np.isfinite(u['rescaled'])):
            raise ValueError("u contains non-finite values")
        if not np.all(np.isfinite(s['rescaled'])):
            raise ValueError("s contains non-finite values")
        if not np.all(s['rescaled'] > 0):
            raise ValueError("s contains non-positive values")
        if u['rescaled'].dtype not in [np.float32, np.complex64]:
            raise TypeError(f"u is not of dtype float32 or complex64, but {u['rescaled'].dtype}")
        if s['rescaled'].dtype not in [np.float32, np.complex64]:
            raise TypeError(f"s is not of dtype float32 or complex64, but {s['rescaled'].dtype}")
        if volume_mask.dtype != np.float32:
            raise TypeError(f"volume_mask is not of dtype float32, but {volume_mask.dtype}")

        if options.ignore_zero_frequency:
            logger.warning(
                "ignore_zero_frequency is ON — inflating DC noise by 1e16. "
                "This is experimental and may degrade mean/variance estimates."
            )
            noise_var_used[0] *= 1e16

        if not args.keep_intermediate:
            del u['real']
            if 'rescaled_no_contrast' in u:
                del u['rescaled_no_contrast']
            covariance_cols = None

        # --- Compute embeddings ---
        (latent_coords, latent_coords_noreg,
         latent_precision, latent_precision_noreg,
         est_contrasts, est_contrasts_noreg) = _compute_embeddings(
            means, u, s, ds, volume_mask, options, gpu_memory,
            focus_masks, zdim_for_rest, args)
        
        ## Make sure this makes sense
        if repeat == 1:
            for key in est_contrasts:
                est_contrasts[key] = est_contrasts[key] * contrasts_for_second
            for key in est_contrasts_noreg:
                est_contrasts_noreg[key] = est_contrasts_noreg[key] * contrasts_for_second

    # --- Post-embedding: noise residual estimate ---
    zdim = np.max(options.zs_dim_to_test)
    if not args.tilt_series:
        n_pcs_to_use = (num_foc_masks - 1) * zdim_for_rest + zdim
        try:
            # Reorder halfset-concatenated arrays (per-image) to original
            # dataset order for iteration over the unified cryoem_dataset.
            contrasts_local_resid = cryoem_dataset.reorder_to_dataset_indexing(
                est_contrasts[zdim], ds, use_tilt_indices=False)
            coords_local_resid = cryoem_dataset.reorder_to_dataset_indexing(
                latent_coords[zdim], ds, use_tilt_indices=False)
            noise_var_from_het_residual, _, _ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(
                ds, dilated_volume_mask, means.combined, u['rescaled'][:, :n_pcs_to_use],
                # //10: heterogeneity residual estimation is memory-intensive (holds full embedding + projections)
                contrasts_local_resid, coords_local_resid, utils.safe_batch_size(batch_size // 10),
                disc_type=covariance_options['disc_type'])
        except Exception as exc:
            # Some CPU/mixed backend traces can hit CUDA FFI host-registration
            # errors in this optional post-embedding diagnostic path.
            if "No FFI handler registered for cuda_project" in str(exc):
                logger.warning(
                    "Skipping heterogeneity residual noise estimate due CUDA FFI host-platform mismatch: %s",
                    exc,
                )
                noise_var_from_het_residual = None
            else:
                raise
    else:
        noise_var_from_het_residual = None

    logger.info("embedding time: %s", time.time() - st_time)
    utils.report_memory_device()
    peak_gpu_gb = utils.get_peak_gpu_memory_used(device=0)
    logger.info("peak GPU memory used: %s", peak_gpu_gb)
    if args.gpu_memory is not None and peak_gpu_gb is not None:
        try:
            peak_val = float(peak_gpu_gb) if not isinstance(peak_gpu_gb, (int, float)) else peak_gpu_gb
            if peak_val > args.gpu_memory:
                logger.warning(
                    "Peak GPU usage (%.1f GB) exceeded --gpu-gb limit (%.1f GB). "
                    "Consider increasing --gpu-gb or reducing batch sizes.",
                    peak_val, args.gpu_memory,
                )
        except (TypeError, ValueError):
            pass  # peak_gpu_gb may not be numeric on all backends

    # --- Handle complement mask trimming ---
    if args.use_complement_mask:
        import copy
        zs_full = copy.deepcopy(latent_coords)
        for key in latent_coords:
            latent_coords[key] = latent_coords[key][:, zdim_for_rest:]
        for key in latent_coords_noreg:
            latent_coords_noreg[key] = latent_coords_noreg[key][:, zdim_for_rest:]
        for key in latent_precision:
            latent_precision[key] = latent_precision[key][:, zdim_for_rest:, zdim_for_rest:]
        for key in latent_precision_noreg:
            latent_precision_noreg[key] = latent_precision_noreg[key][:, zdim_for_rest:, zdim_for_rest:]
        u['rescaled'] = u['rescaled'][:, zdim_for_rest:]
        s['rescaled'] = s['rescaled'][zdim_for_rest:]

    # --- Save volumes ---
    paths.ensure_volumes_dir()
    output.save_covar_output_volumes(paths.output_dir, means.combined, u['rescaled'], s,
                                volume_mask, volume_shape, voxel_size=ds.voxel_size)
    output.save_volume(volume_mask, os.path.splitext(paths.mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=ds.voxel_size)
    output.save_volume(dilated_volume_mask, os.path.splitext(paths.dilated_mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=ds.voxel_size)

    focus_mask = focus_masks[-1]
    output.save_volume(focus_mask, os.path.splitext(paths.focus_mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=ds.voxel_size)
    if args.use_complement_mask:
        output.save_volume(focus_masks[0], os.path.splitext(paths.complement_mask_volume)[0], volume_shape,
                      from_ft=False, voxel_size=ds.voxel_size)

    # --- Build result dict and save ---
    if args.tilt_series:
        particles_ind_split = [
            ds.halfset_original_group_indices(halfset_id)
            for halfset_id in range(2)
        ]
    else:
        particles_ind_split = ind_split

    embedding_dict = output.build_embedding_dict(
        latent_coords, latent_coords_noreg,
        latent_precision, latent_precision_noreg,
        est_contrasts, est_contrasts_noreg)

    # Reorder embeddings from halfset ordering to original particle ordering
    for entry in embedding_dict:
        for key in embedding_dict[entry]:
            if entry.startswith('contrasts') and args.tilt_series and ('shared' not in options.contrast):
                embedding_dict[entry][key] = cryoem_dataset.reorder_to_original_indexing_from_halfsets(
                    embedding_dict[entry][key], ind_split)
            else:
                embedding_dict[entry][key] = cryoem_dataset.reorder_to_original_indexing_from_halfsets(
                    embedding_dict[entry][key], particles_ind_split)

    args.halfsets = paths.particles_halfsets

    result = output.build_params_dict(
        volume_shape=volume_shape,
        voxel_size=ds.voxel_size,
        s_rescaled=s['rescaled'],
        noise_var_from_hf=noise_var_from_hf,
        noise_var_from_het_residual=noise_var_from_het_residual,
        noise_var_used=noise_var_used,
        noise_result=noise_result,
        ub_noise_var_by_var_est=ub_noise_var_by_var_est,
        variance_est=variance_est,
        variance_fsc=variance_fsc,
        noise_p_variance_est=noise_p_variance_est,
        covariance_options=covariance_options,
        column_fscs=column_fscs,
        picked_frequencies=picked_frequencies,
        input_args=args,
    )

    output.save_pipeline_results(
        paths,
        result,
        embedding_dict,
        covariance_cols,
        particles_ind_split,
        ind_split,
        zs_full=zs_full if args.use_complement_mask else None,
    )

    logger.info("total time: %s", time.time() - st_time)

    if hasattr(args, '_downsample_applied') and args._downsample_applied:
        logger.info(
            "Pipeline ran at D=%d (downsampled from original). "
            "Downstream commands (analyze, density, etc.) will "
            "automatically use the downsampled data.",
            args._downsample_applied,
        )
    else:
        logger.info("Pipeline ran at original image resolution (D=%d).", ds.grid_size)

    # Generate standard plots
    po = output.PipelineOutput(args.outdir)
    zdims = np.array(args.zdim)
    zdim_choose = np.argmin(np.abs(zdims - 10))
    zdim = zdims[zdim_choose]
    output.standard_pipeline_plots(po, zdim, paths.plots_dir)

    return means, u, s, volume_mask, dilated_volume_mask, noise_var_used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    from recovar.project.job_context import job_context
    with job_context(args, "pipeline") as ctx:
        args.outdir = ctx.output_dir
        standard_recovar_pipeline(args)


if __name__ == "__main__":
    main()
