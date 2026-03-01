import logging
logger = logging.getLogger(__name__)
import recovar.jax_config
import jax
import numpy as np
import os, argparse, time, sys
from recovar.output import output as o
from recovar import utils
from recovar.reconstruction import homogeneous, noise
from recovar.output import output
from recovar.data_io import dataset
from recovar.core import mask
from recovar.heterogeneity import embedding, principal_components, covariance_estimation
from recovar.output.output_paths import ResultPaths
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.utils import copy_data_to_temp_folder, save_original_paths_info, cleanup_temp_files


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument('--zdim', type=list_of_ints, default=[1,2,4,10,20], help="Dimensions of latent variable. Default=1,2,4,10,20")

    parser.add_argument(
        "--poses", type=os.path.abspath, required=False, default=None,
        help="Image poses (.pkl). If not provided, auto-extracted from particles file (.star or .cs)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=False, default=None,
        help="CTF parameters (.pkl). If not provided, auto-extracted from particles file (.star or .cs)"
    )
    parser.add_argument(
        "--downsample", type=int, default=None,
        help="Downsample images to this box size via Fourier cropping (must be even, <= original size)"
    )

    parser.add_argument(
        "--mask", metavar="mrc", required=True, help="solvent mask (.mrc).Can solve provide: from_halfmaps, sphere, none"
    )

    parser.add_argument(
        "--focus-mask", metavar="mrc", dest = "focus_mask", default=None, type=os.path.abspath, help="focus mask (.mrc)"
    )

    parser.add_argument(
        "--keep-input-mask", action="store_true", dest="keep_input_mask", help="By default, the software thresholds and then softens mask. If this option is on, the input mask is used as is."
    )

    parser.add_argument(
        "--use-complement-mask", action="store_true", dest = "use_complement_mask", help="Use complement of focus mask"
    )

    parser.add_argument(
        "--copy-to-folder", dest="copy_to_folder", default = None, type=os.path.abspath, help="Copy all input data files to this temporary folder before processing. Original paths will be saved in output."
    )

    parser.add_argument(
        "--mask-dilate-iter", type=int, default=0, dest="mask_dilate_iter", help="mask options how many iters to dilate solvent and focus mask"
    )

    parser.add_argument(
        "--correct-contrast",
        dest = "correct_contrast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="estimate and correct for amplitude scaling (contrast) variation across images. Default = false "
    )

    parser.add_argument(
        "--ignore-zero-frequency",
        dest = "ignore_zero_frequency",
        action="store_true",
        help="use if you want zero frequency to be ignored. If images have been normalized to 0 mean, this is probably a good idea"
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter images by these indices",
    )

    group.add_argument(
        "--particle-ind",
        dest="tilt_ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices (only for tilt-series/cryo-ET)",
    )

    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "automatic",
        help="Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0",
    )

    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )

    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file. If not specified, uses the directory of the star file.",
    )

    parser.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file (used in starfile input ONLY). \
        Useful when star file contains longer paths than available on the system. By default, it strips the full path (except the filename). E.g, if you starfile path is Extract/job193/Subtomograms/XXX/XXX.mrcs, \
        and your directory looks like /your/path/to/Subtomograms, then you can use --strip-prefix Extract/job193 --datadir /your/path/to/.",
    )

    group.add_argument(
            "--n-images",
            default = -1,
            dest="n_images",
            type=int,
            help="Number of images to use (should only use for quick run)",
        )

    group.add_argument(
            "--padding",
            type=int,
            default = 0,
            help="Real-space padding",
        )

    group.add_argument(
            "--halfsets",
            default = None,
            type=os.path.abspath,
            help="Path to a file with indices of split dataset (.pkl).",
        )

    group.add_argument(
            "--keep-intermediate",
            dest = "keep_intermediate",
            action="store_true",
            help="saves some intermediate result. Probably only useful for debugging"
        )

    group.add_argument(
            "--noise-model",
            dest = "noise_model",
            default = "radial",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
        )

    group.add_argument(
            "--mean-fn",
            dest = "mean_fn",
            default = "triangular",
            help="which mean function to use. Options are triangular (default), old, triangular_reg"
        )

    group.add_argument(
        "--accept-cpu",
        dest="accept_cpu",
        action="store_true",
        help="Accept running on CPU if no GPU is found",
    )

    group.add_argument(
            "--test-covar-options",
            dest = "test_covar_options",
            action="store_true",
            help="Only for development. Test different covariance estimation options"
        )

    group.add_argument(
            "--low-memory-option",
            help = "Use lower memory options for covariance estimation",
            dest = "low_memory_option",
            action="store_true",
        )

    group.add_argument(
            "--very-low-memory-option",
            help = "Use lowest memory options for covariance estimation",
            dest = "very_low_memory_option",
            action="store_true",
        )

    group.add_argument(
            "--dont-use-image-mask",
            dest = "dont_use_image_mask",
            action="store_true",
        )

    parser.add_argument(
        "--tilt-series", action="store_true",  dest="tilt_series", help="Whether to use tilt_series."
    )

    parser.add_argument(
        "--tilt-series-ctf", default = None,  dest="tilt_series_ctf", help="What CTF to use for tilt series. Options : cryoem, relion5, warp (windows). Warptools is not yet supported. Default = cryoem if tilt series is False, relion5 if tilt series is True"
    )

    parser.add_argument(
        "--dose-per-tilt", default =None, type = float, dest="dose_per_tilt", help="Default = None, read from starfile"
    )

    parser.add_argument(
        "--angle-per-tilt", default =None,  type = float, dest="angle_per_tilt", help="Default = None, estimated from starfile"
    )

    parser.add_argument(
        "--only-mean", action="store_true", dest = "only_mean", help="Only compute mean"
    )

    parser.add_argument(
        "--ntilts", default = None, type=int, help="Number of tilts to use per tilt series. None = all (default)"
    )

    parser.add_argument(
        "--gpu-gb", default =None,  type = float, dest="gpu_memory", help="How much GPU memory to use. Default = all"
    )

    parser.add_argument(
        "--premultiplied-ctf", dest = 'premultiplied_ctf', action="store_true", help="Whether to use premultiplied CTF. Default = False"
    )

    parser.add_argument(
        "--new-noise-est", dest = 'new_noise_est', action="store_true", help="Whether to use new noise estimation. Default = False"
    )

    parser.add_argument('--shared_contrast_across_tilts', action=argparse.BooleanOptionalAction, default =False,
                        help="Whether to share contrast (amplitude scale) across tilts in cryoET. Default = False")

    parser.add_argument('--use_reg_mean_in_contrast', action=argparse.BooleanOptionalAction, default =True)

    parser.add_argument(
            "--do-over-with-contrast",
            dest = "do_over_with_contrast",
            action=argparse.BooleanOptionalAction,
            default = None,
            help="Whether to run again once constrast is estimated. By default == correct_contrast. Can enter --no-do-over-with-contrast to turn off",
        )

    parser.add_argument('--dilated-mask-dilation-iters',
                        type = int,
                        default = None,
                        help = "How many times to dilate the mask. Default = 6 * volume_shape[0] / 128"
                        )

    parser.add_argument("--no-cleanup", action="store_true", help="Do not clean up temporary files after processing (useful for chaining multiple pipeline calls)")

    parser.add_argument("--multi-gpu", action="store_true", dest="multi_gpu", help="Enable multi-GPU parallelization for covariance computation")

    parser.add_argument("--n-gpus", type=int, default=None, dest="n_gpus", help="Number of GPUs to use (default: use all available GPUs)")

    return parser


# ---------------------------------------------------------------------------
# Helper functions — extracted from standard_recovar_pipeline for clarity
# ---------------------------------------------------------------------------

def _check_uninvert_data(means, cryos, args):
    """Check if data needs sign inversion based on the mean estimate.

    In cryo-EM, the convention is that protein density is positive in real
    space. If the estimated mean has negative density in the protein region,
    the sign of the data (and means) is flipped.
    """
    mean_real = fourier_transform_utils.get_idft3(means['combined'].reshape(cryos.volume_shape))
    radial_mask = cryos.get_volume_radial_mask(cryos.grid_size // 3).reshape(cryos.volume_shape)
    uninvert_check = np.sum(mean_real.real ** 3 * radial_mask) < 0

    if args.uninvert_data == 'automatic':
        if uninvert_check:
            for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
                if key in means:
                    means[key] = -means[key]
            for cryo in cryos:
                cryo.image_stack.mult = -1 * cryo.image_stack.mult
            args.uninvert_data = "true"
            logger.warning('sum(mean) < 0! Swapping sign of data (uninvert-data = true)')
        else:
            logger.info('setting (uninvert-data = false)')
            args.uninvert_data = "false"
    elif uninvert_check:
        logger.warning('sum(mean) < 0! Data probably needs to be inverted! set --uninvert-data=true (or automatic)')


def _estimate_noise(cryos, means, dilated_volume_mask, batch_size, args, noise_model):
    """Estimate radial noise variance from outside-mask and upper-bound methods.

    Returns a dict with all noise-related quantities needed by the pipeline.
    """
    use_new_noise_fn = args.new_noise_est or args.premultiplied_ctf
    logger.info(f"Using new noise estimation function?: {use_new_noise_fn}")

    noise_time = time.time()

    # Estimate noise outside the mask
    if use_new_noise_fn:
        masked_image_PS, image_PS = noise.fit_noise_model_to_images(
            cryos[0], dilated_volume_mask, means['combined'], None,
            batch_size=batch_size, invert_mask=True, disc_type='linear_interp')
        logger.info("Using new noise estimation with linear_interp discretization")
    elif args.mask.endswith(".mrc"):
        masked_image_PS, _, _ = noise.estimate_noise_variance_from_outside_mask_v2(
            cryos[0], dilated_volume_mask, batch_size)
        white_noise_var_outside_mask = noise.estimate_white_noise_variance_from_mask(
            cryos[0], dilated_volume_mask, batch_size)
        _, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
            cryos[0], dilated_volume_mask, batch_size)
    else:
        masked_image_PS, _, image_PS, _ = noise.estimate_radial_noise_statistic_from_outside_mask(
            cryos[0], dilated_volume_mask, batch_size)

    radial_noise_var_outside_mask = masked_image_PS
    white_noise_var_outside_mask_val = np.median(masked_image_PS)

    if use_new_noise_fn:
        assert (noise_model == "radial" or noise_model == "radial_per_tilt"), \
            f"new noise fn only works with radial noise model. You set {noise_model}"

    logger.info(f"time to estimate noise is {time.time() - noise_time}")
    utils.report_memory_device(logger=logger)

    # Upper bound on noise variance from inside the mask
    noise_time = time.time()
    if use_new_noise_fn:
        radial_ub_noise_var, _ = noise.fit_noise_model_to_images(
            cryos[0], dilated_volume_mask, means['combined'], None,
            batch_size=batch_size, invert_mask=False, disc_type='linear_interp')
    else:
        radial_ub_noise_var, _, _ = noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(
            cryos[0], means['combined'], dilated_volume_mask, batch_size)
    logger.info(f"time to upper bound noise is {time.time() - noise_time}")

    # Bound the noise variance
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

    return {
        'noise_var_used': noise_var_used,
        'radial_noise_var_outside_mask': radial_noise_var_outside_mask,
        'radial_ub_noise_var': radial_ub_noise_var,
        'white_noise_var_outside_mask': white_noise_var_outside_mask_val,
        'image_PS': image_PS,
        'masked_image_PS': masked_image_PS,
    }


def _build_focus_masks(args, means, volume_mask, volume_shape, cryos):
    """Build focus masks and optional complement mask."""
    if args.focus_mask is not None:
        focus_mask, _ = mask.masking_options(
            args.focus_mask, means, volume_shape, cryos.dtype_real,
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


def _compute_embeddings(means, u, s, cryos, volume_mask, options, gpu_memory,
                        focus_masks, zdim_for_rest, args, mean_cubic=None):
    """Compute per-image embeddings for all requested zdim values.

    Returns (zs, cov_zs, est_contrasts) dicts keyed by zdim (int or str).
    """
    num_foc_masks = len(focus_masks)
    zs = {}
    cov_zs = {}
    est_contrasts = {}

    for zdim in options['zs_dim_to_test']:
        n_pcs_to_use = (num_foc_masks - 1) * zdim_for_rest + zdim
        z_time = time.time()
        zs[zdim], cov_zs[zdim], est_contrasts[zdim], _ = embedding.get_per_image_embedding(
            means['combined'], u['rescaled'], s['rescaled'], n_pcs_to_use,
            cryos, volume_mask, gpu_memory, 'linear_interp',
            contrast_grid=None, contrast_option=options['contrast'],
            ignore_zero_frequency=options['ignore_zero_frequency'],
            mean_cubic=mean_cubic)
        logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")

    # Also compute unregularized embeddings (s -> inf means no prior)
    for zdim in options['zs_dim_to_test']:
        z_time = time.time()
        n_pcs_to_use = (num_foc_masks - 1) * zdim_for_rest + zdim
        key = f"{zdim}_noreg"
        zs[key], cov_zs[key], est_contrasts[key], _ = embedding.get_per_image_embedding(
            means['combined'], u['rescaled'], s['rescaled'] * 0 + np.inf, n_pcs_to_use,
            cryos, volume_mask, gpu_memory, 'linear_interp',
            contrast_grid=None, contrast_option=options['contrast'],
            ignore_zero_frequency=options['ignore_zero_frequency'],
            mean_cubic=mean_cubic)
        logger.info(f"embedding time for zdim={zdim}_noreg: {time.time() - z_time}")

    return zs, cov_zs, est_contrasts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def standard_recovar_pipeline(args):
    st_time = time.time()

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
    path_mapping = copy_data_to_temp_folder(args)

    if args.mask.endswith(".mrc"):
        args.mask = os.path.abspath(args.mask)

    if (not args.accept_cpu) and (not utils.jax_has_gpu()):
        raise ValueError("No GPU found. Set --accept-cpu if you really want to run on CPU (probably not). More likely, you want to check that JAX has been properly installed with GPU support.")

    paths = ResultPaths(args.outdir)
    paths.ensure_dirs()
    with open(paths.command_txt, "w") as text_file:
        text_file.write('python ' + ' '.join(sys.argv))

    save_original_paths_info(path_mapping, args.outdir)

    # CTF defaults
    if args.tilt_series_ctf is None:
        args.tilt_series_ctf = 'relion5' if args.tilt_series else 'cryoem'
        logger.info(f"Setting tilt_series_ctf to {args.tilt_series_ctf}")

    if args.tilt_series and args.dose_per_tilt is not None:
        logger.warning("dose_per_tilt is provided, but tilt_series_ctf is set to using starfile = -B_fac/4 (by default). Thus, dose_per_tilt will not be used.")

    if (args.tilt_series_ctf == 'v2_scale_from_star') and (args.angle_per_tilt is not None):
        logger.warning("angle_per_tilt is provided, but tilt_series_ctf is set to using scale from inputfile (by default). Thus, angle_per_tilt will not be used.")

    if args.do_over_with_contrast is None:
        args.do_over_with_contrast = args.correct_contrast

    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        force=True,
        handlers=[
            logging.FileHandler(paths.run_log),
            logging.StreamHandler(),
        ])

    logger.info(args)

    # --- Auto pre-downsample to disk if requested ---
    if getattr(args, 'downsample', None) is not None:
        from recovar.commands.downsample import downsample_to_disk

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
    ind_split = dataset.figure_out_halfsets(args)
    dataset_loader_dict = dataset.make_dataset_loader_dict(args)
    logger.info("Data loading configuration:")
    logger.info(f"  Particles file: {dataset_loader_dict['particles_file']}")
    logger.info(f"  Poses file: {dataset_loader_dict.get('poses_file', '(auto-extract from particles)')}")
    logger.info(f"  CTF file: {dataset_loader_dict.get('ctf_file', '(auto-extract from particles)')}")
    if dataset_loader_dict.get('downsample_D'):
        logger.info(f"  Downsample to: {dataset_loader_dict['downsample_D']}")
    if dataset_loader_dict.get('datadir'):
        logger.info(f"  Datadir: {dataset_loader_dict['datadir']}")

    options = utils.make_algorithm_options(args)
    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split, args.lazy)

    if args.gpu_memory is not None:
        utils.GPU_MEMORY_LIMIT = args.gpu_memory
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = cryos.volume_shape

    batch_size = utils.get_image_batch_size(cryos.grid_size, gpu_memory)
    logger.info(f"image batch size: {batch_size}")
    logger.info(f"volume batch size: {utils.get_vol_batch_size(cryos.grid_size, gpu_memory)}")
    logger.info(f"column batch size: {utils.get_column_batch_size(cryos.grid_size, gpu_memory)}")
    logger.info(f"number of images: {cryos.n_total_images}")
    utils.report_memory_device(logger=logger)

    # --- Initial noise estimate from half-maps ---
    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)
    valid_idx = cryos.get_valid_frequency_indices()
    noise_model = args.noise_model

    # --- Contrast correction repeat logic ---
    n_repeats = 2 if args.do_over_with_contrast else 1
    if args.do_over_with_contrast and not args.correct_contrast:
        logger.warning("Do over with contrast, but contrast correction is off. Setting contrast correction to on")
        args.correct_contrast = True
        options["contrast"] = "contrast_qr"

    if args.shared_contrast_across_tilts:
        options['contrast'] += '_shared'
        logger.info("Setting contrast to shared")

    # Initialize noise model
    for cryo in cryos:
        if noise_model == "radial":
            cryo.set_radial_noise_model(None)
            logger.info("Setting noise model to radial")
        elif noise_model in ('radial_per_tilt', 'radial-per-tilt'):
            cryo.set_variable_radial_noise_model(None)
            logger.info("Setting noise model to radial_per_tilt")
        else:
            raise ValueError(f"noise model {noise_model} not recognized")

    contrasts_for_second = None
    for repeat in range(n_repeats):

        if repeat == 1:
            ndim = 10 if 10 in options['zs_dim_to_test'] else np.median(options['zs_dim_to_test'])
            logger.warning(f"repeating with contrast of zdim={ndim}")
            contrasts_for_second = est_contrasts[ndim]
            contrasts_for_second /= np.mean(contrasts_for_second)
            embedding.set_contrasts_in_cryos(cryos, contrasts_for_second)
            options["contrast"] = "contrast"

        # --- Compute mean ---
        if args.mean_fn == 'triangular':
            means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
                cryos, 2 * batch_size, noise_variance=noise_var_from_hf, use_regularization=False)
        elif args.mean_fn == 'triangular_reg':
            means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
                cryos, 5 * batch_size, noise_variance=noise_var_from_hf, use_regularization=True)
        else:
            raise ValueError(f"mean function {args.mean_fn} not recognized")
        utils.report_memory_device(logger=logger)

        # --- Check sign and uninvert if needed ---
        _check_uninvert_data(means, cryos, args)

        if means['combined'].dtype != cryos.dtype:
            logger.warning(f"mean estimate is in type: {means['combined'].dtype}")
            means['combined'] = means['combined'].astype(cryos.dtype)

        logger.info(f"mean computed in {time.time() - st_time}")
        utils.report_memory_device(logger=logger)

        # --- Pre-compute cubic spline coefficients for mean (once) ---
        from recovar.core import cubic_interpolation
        mean_cubic = cubic_interpolation.calculate_spline_coefficients(
            means['combined'].reshape(volume_shape))
        logger.info("Pre-computed cubic spline coefficients for mean estimate")

        # --- Compute mask ---
        volume_mask, dilated_volume_mask = mask.masking_options(
            args.mask, means, volume_shape, cryos.dtype_real,
            args.mask_dilate_iter, args.keep_input_mask, args.dilated_mask_dilation_iters)

        # --- Save mean and mask volumes ---
        paths.ensure_volumes_dir()
        # save_volume appends .mrc, so strip the extension from the path
        o.save_volume(means['combined'], os.path.splitext(paths.mean_volume)[0], volume_shape,
                      from_ft=True, voxel_size=cryos.voxel_size)
        o.save_volume(means['corrected0'], os.path.splitext(paths.mean_half1_unfil)[0], volume_shape,
                      from_ft=True, voxel_size=cryos.voxel_size)
        o.save_volume(means['corrected1'], os.path.splitext(paths.mean_half2_unfil)[0], volume_shape,
                      from_ft=True, voxel_size=cryos.voxel_size)
        o.save_volume(volume_mask, os.path.splitext(paths.mask_volume)[0], volume_shape,
                      from_ft=False, voxel_size=cryos.voxel_size)

        # Filter and save mean
        from recovar.heterogeneity import locres
        half1 = fourier_transform_utils.get_idft3(means['corrected0'].reshape(volume_shape))
        half2 = fourier_transform_utils.get_idft3(means['corrected1'].reshape(volume_shape))
        best_filtered_nob, _, _, _, _ = locres.local_resolution(
            half1, half2, 0, cryos.voxel_size, use_filter=True, fsc_threshold=1/7, use_v2=True)
        o.save_volume(best_filtered_nob, os.path.splitext(paths.mean_filtered)[0], volume_shape,
                      from_ft=False, voxel_size=cryos.voxel_size)

        if args.only_mean:
            return

        # --- Build focus masks ---
        focus_masks = _build_focus_masks(args, means, volume_mask, volume_shape, cryos)

        # --- Estimate noise ---
        noise_result = _estimate_noise(cryos, means, dilated_volume_mask, batch_size, args, noise_model)
        noise_var_used = noise_result['noise_var_used']
        noise.update_noise_variance(noise_var_used, cryos)

        # Upper bound noise using variance estimate
        variance_est, ub_noise_var_by_var_est = noise.upper_bound_noise_by_signal_p_noise_dispatched(
            noise_var_used, cryos, means, batch_size, dilated_volume_mask)

        # Compute variance with regularization
        # //2: variance computation with cubic disc_type needs ~2x memory per image (spline coefficients)
        variance_est, _, variance_fsc, _, noise_p_variance_est = covariance_estimation.compute_variance(
            cryos, means['combined'], utils.safe_batch_size(batch_size // 2), dilated_volume_mask,
            use_regularization=True, disc_type='cubic', mean_cubic=mean_cubic)

        utils.report_memory_device(logger=logger)

        # --- Covariance options ---
        covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos.grid_size)
        if args.low_memory_option:
            covariance_options['sampling_n_cols'] = 50
            covariance_options['randomized_sketch_size'] = 100
            covariance_options['n_pcs_to_compute'] = 100
            covariance_options['sampling_avoid_in_radius'] = 3

        if args.very_low_memory_option:
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
        n_pcs_to_keep = np.max(np.append(options['zs_dim_to_test'], 50))

        ignore_zero_frequency = options['ignore_zero_frequency']
        for idx, focus_mask in enumerate(focus_masks):
            u_this, s_this, covariance_cols, picked_frequencies, column_fscs = \
                principal_components.estimate_principal_components(
                    cryos, options, means, mean_prior, focus_mask, dilated_volume_mask,
                    valid_idx, batch_size, gpu_memory_to_use=gpu_memory,
                    covariance_options=covariance_options,
                    variance_estimate=variance_est['combined'],
                    use_reg_mean_in_contrast=args.use_reg_mean_in_contrast,
                    use_multi_gpu=args.multi_gpu, n_gpus=args.n_gpus,
                    mean_cubic=mean_cubic)
            if idx == num_foc_masks - 1:
                s.append(s_this['rescaled'][:n_pcs_to_keep].copy())
                u.append(u_this['rescaled'][:, :n_pcs_to_keep].copy())
            else:
                s.append(s_this['rescaled'][:zdim_for_rest].copy())
                u.append(u_this['rescaled'][:, :zdim_for_rest].copy())
            del u_this, s_this

        u = {'rescaled': np.concatenate(u, axis=1), 'real': None}
        s = {'rescaled': np.concatenate(s, axis=0), 'real': None}
        options['ignore_zero_frequency'] = ignore_zero_frequency

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

        if options['ignore_zero_frequency']:
            logger.info('ignoring zero frequency')
            noise_var_used[0] *= 1e16

        if not args.keep_intermediate:
            del u['real']
            if 'rescaled_no_contrast' in u:
                del u['rescaled_no_contrast']
            covariance_cols = None

        # --- Compute embeddings ---
        zs, cov_zs, est_contrasts = _compute_embeddings(
            means, u, s, cryos, volume_mask, options, gpu_memory,
            focus_masks, zdim_for_rest, args, mean_cubic=mean_cubic)

        if repeat == 1:
            for key in est_contrasts:
                est_contrasts[key] = est_contrasts[key] * contrasts_for_second

    # --- Post-embedding: noise residual estimate ---
    zdim = np.max(options['zs_dim_to_test'])
    if not args.tilt_series:
        n_pcs_to_use = (num_foc_masks - 1) * zdim_for_rest + zdim
        noise_var_from_het_residual, _, _ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(
            cryos[0], dilated_volume_mask, means['combined'], u['rescaled'][:, :n_pcs_to_use],
            # //10: heterogeneity residual estimation is memory-intensive (holds full embedding + projections)
            est_contrasts[zdim], zs[zdim], utils.safe_batch_size(batch_size // 10),
            disc_type=covariance_options['disc_type'])
    else:
        noise_var_from_het_residual = None

    logger.info(f"embedding time: {time.time() - st_time}")
    utils.report_memory_device()
    logger.info(f"peak gpu memory use {utils.get_peak_gpu_memory_used(device=0)}")

    # --- Handle complement mask trimming ---
    if args.use_complement_mask:
        import copy
        zs_full = copy.deepcopy(zs)
        for key in zs:
            zs[key] = zs[key][:, zdim_for_rest:]
        for key in cov_zs:
            cov_zs[key] = cov_zs[key][:, zdim_for_rest:, zdim_for_rest:]
        u['rescaled'] = u['rescaled'][:, zdim_for_rest:]
        s['rescaled'] = s['rescaled'][zdim_for_rest:]

    # --- Save volumes ---
    paths.ensure_volumes_dir()
    o.save_covar_output_volumes(paths.output_dir, means['combined'], u['rescaled'], s,
                                volume_mask, volume_shape, voxel_size=cryos.voxel_size)
    o.save_volume(volume_mask, os.path.splitext(paths.mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=cryos.voxel_size)
    o.save_volume(dilated_volume_mask, os.path.splitext(paths.dilated_mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=cryos.voxel_size)

    focus_mask = focus_masks[-1]
    o.save_volume(focus_mask, os.path.splitext(paths.focus_mask_volume)[0], volume_shape,
                  from_ft=False, voxel_size=cryos.voxel_size)
    if args.use_complement_mask:
        o.save_volume(focus_masks[0], os.path.splitext(paths.complement_mask_volume)[0], volume_shape,
                      from_ft=False, voxel_size=cryos.voxel_size)

    # --- Build result dict and save ---
    if args.tilt_series:
        particles_ind_split = [cryo.image_stack.dataset_tilt_indices for cryo in cryos]
    else:
        particles_ind_split = ind_split

    embedding_dict = o.build_embedding_dict(zs, cov_zs, est_contrasts)

    # Reorder embeddings from halfset ordering to original particle ordering
    for entry in embedding_dict:
        for key in embedding_dict[entry]:
            if entry == 'contrasts' and args.tilt_series and ('shared' not in options['contrast']):
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(
                    embedding_dict[entry][key], ind_split)
            else:
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(
                    embedding_dict[entry][key], particles_ind_split)

    args.halfsets = paths.particles_halfsets

    # Restore original paths before saving args
    if path_mapping is not None:
        logger.info("Restoring original paths in input_args before saving...")
        paths_to_restore = [
            ('original_particles', 'temp_particles', 'particles'),
            ('original_poses', 'temp_poses', 'poses'),
            ('original_ctf', 'temp_ctf', 'ctf'),
            ('original_mask', 'temp_mask', 'mask'),
            ('original_focus_mask', 'temp_focus_mask', 'focus_mask'),
            ('original_ind', 'temp_ind', 'ind'),
            ('original_tilt_ind', 'temp_particle_ind', 'tilt_ind'),
            ('original_halfsets', 'temp_halfsets', 'halfsets'),
        ]
        for orig_key, temp_key, attr_name in paths_to_restore:
            if orig_key in path_mapping and temp_key in path_mapping:
                setattr(args, attr_name, path_mapping[orig_key])
                logger.info(f"Restored {attr_name} path: {path_mapping[orig_key]}")
            elif orig_key in path_mapping:
                if attr_name == 'datadir' and path_mapping[orig_key]:
                    setattr(args, attr_name, path_mapping[orig_key])
                    logger.info(f"Restored {attr_name} path: {path_mapping[orig_key]}")

    result = o.build_params_dict(
        volume_shape=volume_shape,
        voxel_size=cryos.voxel_size,
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

    if path_mapping is not None:
        result['original_paths'] = path_mapping

    o.save_pipeline_results(
        paths,
        result,
        embedding_dict,
        covariance_cols,
        particles_ind_split,
        ind_split,
        zs_full=zs_full if args.use_complement_mask else None,
    )

    logger.info(f"total time: {time.time() - st_time}")

    # Clean up temp files
    if path_mapping is not None and not args.no_cleanup:
        cleanup_temp_files(path_mapping)

    # Generate standard plots
    from recovar.output import output
    po = output.PipelineOutput(args.outdir)
    zdims = np.array(args.zdim)
    zdim_choose = np.argmin(np.abs(zdims - 10))
    zdim = zdims[zdim_choose]
    output.standard_pipeline_plots(po, zdim, paths.plots_dir)

    return means, u, s, volume_mask, dilated_volume_mask, noise_var_used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)


if __name__ == "__main__":
    main()
