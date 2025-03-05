# If you want to extend and use recovar, you should import this first
import logging
# It is important to import cryodrgn before the setting basicConfig which is why it is imported here (but not used)
# import cryodrgn
logger = logging.getLogger(__name__)
import recovar.config 
import jax
import jax.numpy as jnp
import numpy as np
import os, argparse, time, pickle, sys
from recovar import output as o
from recovar import dataset, homogeneous, embedding, principal_components, latent_density, mask, utils, constants, noise, output, covariance_estimation
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
# logger.setLevel(logger.info)
logger = logging.getLogger(__name__)


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

    # parser.add_argument(
    #     "--zdim", type=list, help="Dimension of latent variable"
    # )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=True, help="CTF parameters (.pkl)"
    )

    # parser.add_argument(
    #     "--mask", metavar="mrc", default=None, type=os.path.abspath, help="mask (.mrc)"
    # )

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


    # parser.add_argument(
    #     "--mask-option", metavar=str, default="input", help="mask options: from_halfmaps , input (default), sphere, none"
    # )

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

    # parser.add_argument(
    #     "--no-z-regularization",
    #     dest = "no_z_regularization",
    #     action="store_true",
    # )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter images by these indices",
    )

    group.add_argument(
        "--tilt-ind",
        dest="tilt_ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter tilts (particles) by these indices",
    )


    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "automatic",
        help="Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0",
    )

    # group.add_argument(
    #     "--rerescale",
    #     dest = "rerescale",
    #     action="store_true",
    # )

    # Should probably add these options
    # group.add_argument(
    #     "--no-window",
    #     dest="window",
    #     action="store_false",
    #     help="Turn off real space windowing of dataset",
    # )
    # group.add_argument(
    #     "--window-r",
    #     type=float,
    #     default=0.85,
    #     help="Windowing radius (default: %(default)s)",
    # )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )

    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
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

    ### CHANGE THESE TWO BACK!?!?!?!
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

    # group.add_argument(
    #         "--do-over-with-contrast",
    #         dest = "do_over_with_contrast",
    #         default = "True",
    #         help="Whether to run again once constrast is estimated",
    #     )
    
    parser.add_argument(
        "--tilt-series", action="store_true",  dest="tilt_series", help="Whether to use tilt_series."
    )

    parser.add_argument(
        "--tilt-series-ctf", default = None,  dest="tilt_series_ctf", help="What CTF to use for tilt series. Default = cryoem if tilt series is False, dose weighting + ctfFromStar if tilt series is True"
    )

    parser.add_argument(
        "--dose-per-tilt", default =None, type = float, dest="dose_per_tilt",
    )

    parser.add_argument(
        "--angle-per-tilt", default =None,  type = float, dest="angle_per_tilt", 
    )

    # parser.add_argument(
    #     "--per-image-bfac-scale", action="store_true",
    # )

    parser.add_argument(
        "--only-mean", action="store_true", dest = "only_mean", help="Only compute mean"
    )

    parser.add_argument(
        "--ntilts", default = None, type=int, help="Number of tilts to use per tilt series. None = all (default)"
    )

    parser.add_argument(
        "--gpu-gb", default =None,  type = float, dest="gpu_memory", help="How much GPU memory to use. Default = all" 
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
    

    return parser
    

def standard_recovar_pipeline(args):
    st_time = time.time()

    if args.mask.endswith(".mrc"):
        args.mask = os.path.abspath(args.mask)

    if (not args.accept_cpu) and (not utils.jax_has_gpu()):
        raise ValueError("No GPU found. Set --accept-cpu if you really want to run on CPU (probably not). More likely, you want to check that JAX has been properly installed with GPU support.")

    # Dump input arguments
    o.mkdir_safe(args.outdir)
    with open(f"{args.outdir}/command.txt", "w") as text_file:
        command = 'python ' + ' '.join((sys.argv))
        text_file.write(command)

    # Set CTF function here
    if args.tilt_series_ctf is None and args.tilt_series is False:
        args.tilt_series_ctf = 'cryoem'
        logger.info("Setting tilt_series_ctf to cryoem")
    elif args.tilt_series_ctf is None and args.tilt_series is True:
        args.tilt_series_ctf = 'v2_scale_from_star'
        logger.info("Setting tilt_series_ctf to v2_scale_from_star")

    if args.tilt_series and args.dose_per_tilt is not None:
        logger.warning("dose_per_tilt is provided, but tilt_series_ctf is set to using starfile = -B_fac/4 (by default). Thus, dose_per_tilt will not be used.")
        # logger.warning("angle_per_tilt is provided, but tilt_series_ctf is set to using scale from inputfile (by default). Thus, angle_per_tilt will not be used.")
        # raise ValueError("dose_per_tilt must be provided for tilt series")

    if (args.tilt_series_ctf == 'v2_scale_from_star') and (args.angle_per_tilt is not None):
        logger.warning("angle_per_tilt is provided, but tilt_series_ctf is set to using scale from inputfile (by default). Thus, angle_per_tilt will not be used.")

    if args.do_over_with_contrast is None:
        args.do_over_with_contrast = args.correct_contrast

    # The force interaction has something to do with cryodrgn interaction which is breaking the logger...
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        level=logging.INFO,
                        force = True, 
                        handlers=[
        logging.FileHandler(f"{args.outdir}/run.log"),
        logging.StreamHandler()])

    logger.info(args)
    ind_split = dataset.figure_out_halfsets(args)

    dataset_loader_dict = dataset.make_dataset_loader_dict(args)
    options = utils.make_algorithm_options(args)

    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split, args.lazy)
    cryo = cryos[0]
    if args.gpu_memory is not None:
        utils.GPU_MEMORY_LIMIT = args.gpu_memory
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = cryo.volume_shape
    disc_type = "linear_interp"

    batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    logger.info(f"image batch size: {batch_size}")
    logger.info(f"volume batch size: {utils.get_vol_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"column batch size: {utils.get_column_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"number of images: {cryos[0].n_images + cryos[1].n_images}")
    utils.report_memory_device(logger=logger)

    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    valid_idx = cryo.get_valid_frequency_indices()
    noise_model = args.noise_model

    
    if args.do_over_with_contrast:
        n_repeats = 2
        if not args.correct_contrast:
            logger.warning("Do over with contrast, but contrast correction is off. Setting contrast correction to on") 
            args.correct_contrast = True
            options["contrast"] = "contrast_qr"

    else:
        n_repeats = 1

    if args.shared_contrast_across_tilts:
        options['contrast'] += '_shared'
        logger.info("Setting contrast to shared")


    for repeat in range(n_repeats):
        
        if repeat == 1:
            if 10 in options['zs_dim_to_test']:
                ndim = 10
            else:
                ndim = np.median(options['zs_dim_to_test'])
            logger.warning(f"repeating with contrast of zdim={ndim}")
            contrasts_for_second = est_contrasts[ndim]
            contrasts_for_second /= np.mean(contrasts_for_second) # normalize to have mean 1
            embedding.set_contrasts_in_cryos(cryos, contrasts_for_second)
            options["contrast"] = "contrast"
        else:
            contrasts_for_second = None

        # Compute mean
        if args.mean_fn == 'old':
            means, mean_prior, _, _ = homogeneous.get_mean_conformation(cryos, 5*batch_size, noise_var_from_hf , valid_idx, disc_type, use_noise_level_prior = False, grad_n_iter = 5)
        elif args.mean_fn == 'triangular':
            means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 2*batch_size, noise_variance = noise_var_from_hf,  use_regularization = False)

        elif args.mean_fn == 'triangular_reg':
            means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = noise_var_from_hf,  use_regularization = True)
        else:
            raise ValueError(f"mean function {args.mean_fn} not recognized")
        utils.report_memory_device(logger=logger)


        mean_real = ftu.get_idft3(means['combined'].reshape(cryos[0].volume_shape))

        ## DECIDE IF WE SHOULD UNINVERT DATA
        uninvert_check = np.sum((mean_real.real**3 * cryos[0].get_volume_radial_mask(cryos[0].grid_size//3).reshape(cryos[0].volume_shape))) < 0
        if args.uninvert_data == 'automatic':
            # Check if in real space, things towards the middle are mostly positive or negative
            if uninvert_check:
            # if np.sum(mean_real.real**3 * cryos[0].get_volume_mask() ) < 0:
                for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
                    if key in means:
                        means[key] =- means[key]
                for cryo in cryos:
                    cryo.image_stack.mult = -1 * cryo.image_stack.mult
                args.uninvert_data = "true"
                logger.warning('sum(mean) < 0! swapping sign of data (uninvert-data = true)')
            else:
                logger.info('setting (uninvert-data = false)')
                args.uninvert_data = "false"
        elif uninvert_check:
            logger.warning('sum(mean) < 0! Data probably needs to be inverted! set --uninvert-data=true (or automatic)')
        ## END OF THIS - maybe move this block of code somewhere else?


        if means['combined'].dtype != cryo.dtype:
            logger.warning(f"mean estimate is in type: {means['combined'].dtype}")
            means['combined'] = means['combined'].astype(cryo.dtype)

        logger.info(f"mean computed in {time.time() - st_time}")
        utils.report_memory_device(logger=logger)

        # Compute mask
        volume_mask, dilated_volume_mask= mask.masking_options(args.mask, means, volume_shape, cryo.dtype_real, args.mask_dilate_iter, args.keep_input_mask)

        ## Always dump mean?
        # if args.only_mean:
        if True:
            output_folder = args.outdir + '/output/' 
            o.mkdir_safe(output_folder)
            o.mkdir_safe(output_folder + 'volumes/')

            o.save_volume(means['combined'], output_folder + 'volumes/' + 'mean', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
            o.save_volume(means['corrected0'], output_folder + 'volumes/' + 'mean_half1_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
            o.save_volume(means['corrected1'], output_folder + 'volumes/' + 'mean_half2_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
            o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)

            from recovar import locres
            half1 = ftu.get_idft3(means['corrected0'].reshape(cryos[0].volume_shape))
            half2 = ftu.get_idft3(means['corrected1'].reshape(cryos[0].volume_shape))

            best_filtered_nob, _, _, _, _ = locres.local_resolution(half1, half2, 0, cryos[0].voxel_size, use_filter = True, fsc_threshold = 1/7, use_v2 = True)

            o.save_volume(best_filtered_nob, output_folder + 'volumes/' + 'mean_filt', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)

        if args.only_mean:
            return

        if args.focus_mask is not None:
            focus_mask, _= mask.masking_options(args.focus_mask, means, volume_shape, cryo.dtype_real, args.mask_dilate_iter, args.keep_input_mask)
        else:
            focus_mask = volume_mask
        
        if args.use_complement_mask:
            complement_mask = (volume_mask > 0.90)*1.0 - (focus_mask > 0.9)*1.0
            complement_mask = (complement_mask > 0)
            from recovar import mask as mask_fn
            complement_mask = np.array(mask_fn.soften_volume_mask(complement_mask, 3).astype(np.float32))
            focus_masks = [complement_mask, focus_mask]
        else:
            focus_masks = [focus_mask]

        noise_time = time.time()
        # Probably should rename all of this...
        masked_image_PS, std_masked_image_PS, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)
        
        if args.mask.endswith(".mrc"):
            radial_noise_var_outside_mask, _,_ =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)

            white_noise_var_outside_mask = noise.estimate_white_noise_variance_from_mask(cryo, dilated_volume_mask, batch_size)
            # white_noise_var_outside_mask = white_noise_var_outside_mask.copy()
        else:
            radial_noise_var_outside_mask = masked_image_PS#noise_var_from_hf * np.ones(cryos[0].grid_size//2 -1, dtype = np.float32)
            white_noise_var_outside_mask = np.median(masked_image_PS)

            # radial_noise_var_outside_mask = noise_var_from_hf * np.ones_like(noise_var_outside_mask)

        logger.info(f"time to estimate noise is {time.time() - noise_time}")
        utils.report_memory_device(logger=logger)

        noise_time = time.time()
        radial_ub_noise_var, _,_ =  noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(cryo, means['combined'], dilated_volume_mask, batch_size)
        logger.info(f"time to upper bound noise is {time.time() - noise_time}")


        utils.report_memory_device(logger=logger)
        radial_noise_var_ubed = np.where(radial_noise_var_outside_mask >  radial_ub_noise_var, radial_ub_noise_var, radial_noise_var_outside_mask)
        # logger.warning("doing funky noise business")
        # noise_var = np.where(noise_var_outside_mask >  noise_var_from_hf, noise_var_outside_mask, np.ones_like(noise_var_from_hf))
        # Noise statistic
        if noise_model == "white":
            noise_var_used = np.ones_like(radial_noise_var_ubed) * white_noise_var_outside_mask
        else:
            noise_var_used = radial_noise_var_ubed
        
        if (noise_var_used <0).any():
            logger.warning("Negative noise variance detected. Setting to image power spectrum / 10")

        noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        ## TODO Does Tilt series for anything for variance??
        variance_time = time.time()
        variance_est, variance_prior, variance_fsc, lhs, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], batch_size//2, dilated_volume_mask, noise_variance = image_cov_noise,  use_regularization = True, disc_type = 'cubic')
        # print('using regul in variance est?!?')
        logger.info(f"variance estimation time: {time.time() - variance_time}")
        utils.report_memory_device(logger=logger)


        rad_grid = np.array(ftu.get_grid_of_radial_distances(cryos[0].volume_shape).reshape(-1))
        # Often low frequency noise will be overestiated. This can be bad for the covariance estimation. This is a way to upper bound noise in the low frequencies by noise + variance .
        n_shell_to_ub = np.min([32, cryos[0].grid_size//2 -1])
        ub_noise_var_by_var_est = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_5_pc = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_median = np.zeros(n_shell_to_ub, dtype = np.float32)

        for k in range(n_shell_to_ub):
            if np.sum(rad_grid==k) >0:
                ub_noise_var_by_var_est[k] = np.percentile(noise_p_variance_est[rad_grid==k], 5)
                ub_noise_var_by_var_est[k] = np.max([0, ub_noise_var_by_var_est[k]])
                variance_est_low_res_5_pc[k] = np.percentile(variance_est['combined'][rad_grid==k], 5)
                variance_est_low_res_median[k] = np.median(variance_est['combined'][rad_grid==k])

        if np.any(ub_noise_var_by_var_est >  noise_var_used[:n_shell_to_ub]):
            logger.warning("Estimated noise greater than upper bound. Bounding noise using estimated upper obund")

        if np.any(variance_est_low_res_5_pc < 0):
            logger.warning("Estimated variance resolutino is < 0. This probably means that the noise was incorrectly estimated. Recomputing noise")
            print("5 percentile:", variance_est_low_res_5_pc)
            print("5 percentile/median over low shells:", variance_est_low_res_5_pc/variance_est_low_res_median)

        noise_var_used[:n_shell_to_ub] = np.where( noise_var_used[:n_shell_to_ub] > ub_noise_var_by_var_est, ub_noise_var_by_var_est, noise_var_used[:n_shell_to_ub])

        noise_var_used = noise_var_used.astype(cryos[0].dtype_real)

        if noise_model == "mixed":
            # Noise at very low resolution is difficult to estimate. This is a heuristic to avoid some issues.
            fixed_resolution_shell = 32
            # Take min of PS and noise variance at fixed shell
            noise_var_used[:fixed_resolution_shell] = np.where(image_PS[:fixed_resolution_shell]> noise_var_used[fixed_resolution_shell], noise_var_used[fixed_resolution_shell], image_PS[:fixed_resolution_shell])

        ## DELETE FROM HERE?
        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        variance_est, _, variance_fsc, _, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], batch_size//2, dilated_volume_mask, noise_variance = image_cov_noise,  use_regularization = True, disc_type = 'cubic')
        utils.report_memory_device(logger=logger)


        rad_grid = np.array(ftu.get_grid_of_radial_distances(cryos[0].volume_shape).reshape(-1))
        # Often low frequency noise will be overestiated. This can be bad for the covariance estimation. This is a way to upper bound noise in the low frequencies by noise + variance .
        n_shell_to_ub = np.min([32, cryos[0].grid_size//2 -1])
        ub_noise_var_by_var_est = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_5_pc = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_median = np.zeros(n_shell_to_ub, dtype = np.float32)

        for k in range(n_shell_to_ub):
            if np.sum(rad_grid==k) >0:
                ub_noise_var_by_var_est[k] = np.percentile(noise_p_variance_est[rad_grid==k], 5)
                ub_noise_var_by_var_est[k] = np.max([0, ub_noise_var_by_var_est[k]])
                variance_est_low_res_5_pc[k] = np.percentile(variance_est['combined'][rad_grid==k], 5)
                variance_est_low_res_median[k] = np.median(variance_est['combined'][rad_grid==k])

        if np.any(ub_noise_var_by_var_est >  noise_var_used[:n_shell_to_ub]):
            logger.warning("Estimated noise greater than upper bound. Bounding noise using estimated upper obund")


        if np.any(variance_est_low_res_5_pc < 0):
            logger.warning("Estimated variance resolution is < 0.")
            print("5 percentile:", variance_est_low_res_5_pc)
            print("5 percentile/median over low shells:", variance_est_low_res_5_pc/variance_est_low_res_median)


        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))


        # test_covar_options = False
        if args.test_covar_options:
            tests = [ ]
            idx = 0
            for test in tests:
                output_folder = args.outdir + '/output/' 
                # Compute principal components
                covariance_options = covariance_estimation.get_default_covariance_computation_options()
                for key in test:
                    covariance_options[key] = test[key]
        
                u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, noise_var_used, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model, covariance_options = covariance_options, variance_estimate = variance_est['combined'])
                from recovar import output
                output.mkdir_safe(output_folder)
                utils.pickle_dump({
                    'options':test, 'u' :u['rescaled'][:,:20], 's' :s['rescaled'][:20], 'picked_frequencies':picked_frequencies
                }, output_folder + f'test_{idx}.pkl')
                del u, s, covariance_cols, picked_frequencies, column_fscs
                idx = idx + 1
                print('done with', idx, test)


        utils.report_memory_device(logger=logger)

        covariance_options = covariance_estimation.get_default_covariance_computation_options()
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



        # Compute principal components
        # Only focus_mask[-1] will do a zdim search, rest will do zdim_for_rest

        # if len(focus_mask) > 1:
        num_foc_masks = len(focus_masks)
        u = []
        s = []
        # This could be sped up by a factor of len(focus_masks)
        zdim_for_rest = 20 # Maybe should make this an option
        n_pcs_to_keep = np.max( np.append(options['zs_dim_to_test'], 50))

        ## FIXME
        ignore_zero_frequency = options['ignore_zero_frequency']
        # mean_for_contrast_correction = means['combined_regularized'] if args.contrast_use_reg_mean else means['combined']

        # options['ignore_zero_frequency'] = False
        for idx, focus_mask in enumerate(focus_masks):
            u_this,s_this, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, noise_var_used, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model, covariance_options = covariance_options, variance_estimate = variance_est['combined'], use_reg_mean_in_contrast = args.use_reg_mean_in_contrast)
            if idx == num_foc_masks -1:
                s.append(s_this['rescaled'][:n_pcs_to_keep].copy())
                u.append(u_this['rescaled'][:,:n_pcs_to_keep].copy())
            else:
                s.append(s_this['rescaled'][:zdim_for_rest].copy())
                u.append(u_this['rescaled'][:,:zdim_for_rest].copy())
            del u_this, s_this
        u = { 'rescaled' : np.concatenate(u, axis = 1), 'real' : None}
        s =  { 'rescaled' : np.concatenate(s, axis = 0), 'real': None}
        options['ignore_zero_frequency'] = ignore_zero_frequency

        # Check if u and s are finite and not NaN
        if not np.all(np.isfinite(u['rescaled'])):
            raise ValueError("u contains non-finite values")
        if not np.all(np.isfinite(s['rescaled'])):
            raise ValueError("s contains non-finite values")

        # Check if s is positive
        if not np.all(s['rescaled'] > 0):
            raise ValueError("s contains non-positive values")

        # Check if u and s are of dtype float32/complex64
        if u['rescaled'].dtype not in [np.float32, np.complex64]:
            raise TypeError(f"u is not of dtype float32 or complex64, but {u['rescaled'].dtype}")
        if s['rescaled'].dtype not in [np.float32, np.complex64]:
            raise TypeError(f"s is not of dtype float32 or complex64, but {s['rescaled'].dtype}")

        # Check if mask is of dtype float32
        if volume_mask.dtype != np.float32:
            raise TypeError(f"volume_mask is not of dtype float32, but {volume_mask.dtype}")

        if options['ignore_zero_frequency']:
            # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
            logger.info('ignoring zero frequency')
            noise_var_used[0] *=1e16

        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        if not args.keep_intermediate:
            del u['real']
            if 'rescaled_no_contrast' in u:
                del u['rescaled_no_contrast']
            covariance_cols = None

        # Compute embeddings
        zs = {}; cov_zs = {}; est_contrasts = {}
        for zdim in options['zs_dim_to_test']:
            # Now we keep num_foc_masks-1*zdim_rest + zdim
            n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
            z_time = time.time()
            zs[zdim], cov_zs[zdim], est_contrasts[zdim], _ = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] , n_pcs_to_use,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )
            logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")


        for zdim in options['zs_dim_to_test']:
            z_time = time.time()
            n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
            key = f"{zdim}_noreg"
            zs[key], cov_zs[key], est_contrasts[key], _ = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled']* 0 + np.inf , n_pcs_to_use,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )
            logger.info(f"embedding time for zdim={zdim}_noreg: {time.time() - z_time}")


        if repeat == 1:
            for key in est_contrasts:
                est_contrasts[key] = est_contrasts[key] * contrasts_for_second

    zs_cont = {}; cov_zs_cont = {}; est_contrasts_cont = {}        
    var_metrics = {'filt_var': None}

    zdim = np.max(options['zs_dim_to_test'])

    if not args.tilt_series:
        n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
        noise_var_from_het_residual, _,_ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(cryos[0], dilated_volume_mask, means['combined'], u['rescaled'][:,:n_pcs_to_use], est_contrasts[zdim], zs[zdim], batch_size//10, disc_type = covariance_options['disc_type'] )
    else:
        noise_var_from_het_residual = None
    # ### END OF DEL

    logger.info(f"embedding time: {time.time() - st_time}")

    utils.report_memory_device()

    # Compute latent space density. Now just done in analyze or elsewhere
    # Precompute the density on the 4D grid. This is the most expensive part of computing trajectories, and can be reused across trajectories. 
    # logger.info(f"starting density computation")
    # density_z = 10 if 10 in zs else options['zs_dim_to_test'][0]
    # density, latent_space_bounds = None, None
    # density, latent_space_bounds  = latent_density.compute_latent_space_density(zs[density_z], cov_zs[density_z], pca_dim_max = 4, num_points = 50, density_option = 'kde')
    # logger.info(f"ending density computation")

    # Dump results to file
    output_model_folder = args.outdir + '/model/'
    o.mkdir_safe(args.outdir)
    o.mkdir_safe(output_model_folder)

    logger.info(f"peak gpu memory use {utils.get_peak_gpu_memory_used(device =0)}")

    
    # For now, maybe just dump the rest?
    
    if args.use_complement_mask:
        import copy
        zs_full = copy.deepcopy(zs)
        for key in zs:
            zs[key] = zs[key][:,zdim_for_rest:]
        for key in cov_zs:
            cov_zs[key] = cov_zs[key][:,zdim_for_rest:,zdim_for_rest:]
        u['rescaled'] = u['rescaled'][:,zdim_for_rest:]
        s['rescaled'] = s['rescaled'][zdim_for_rest:]


    result = { 's' : s['rescaled'],'s_all': s,
                'input_args' : args,
                'latent_space_bounds' : None, #np.array(latent_space_bounds), 
                'density': None,
                'noise_var_from_hf': noise_var_from_hf,
                'radial_noise_var_outside_mask' : np.array(radial_noise_var_outside_mask),
                'radial_ub_noise_var' : np.array(radial_ub_noise_var),
                'white_noise_var_outside_mask' : np.array(white_noise_var_outside_mask),
                'image_PS' : np.array(image_PS),
                'std_image_PS' : np.array(std_image_PS),
                'masked_image_PS' : np.array(masked_image_PS),
                'std_masked_image_PS' : np.array(std_masked_image_PS),
                'noise_var_from_het_residual' : np.array(noise_var_from_het_residual),
                'noise_var_used' : np.array(noise_var_used),
                'column_fscs': column_fscs, 
                'covariance_cols': None, 
                'picked_frequencies' : picked_frequencies, 'volume_shape': volume_shape, 'voxel_size': cryos[0].voxel_size, 'pc_metric' : var_metrics['filt_var'],
                'variance_est': variance_est, 'variance_fsc': variance_fsc, 'noise_p_variance_est': noise_p_variance_est, 'ub_noise_var_by_var_est': ub_noise_var_by_var_est, 'covariance_options': covariance_options,
                'contrasts_for_second': contrasts_for_second, 
                'version': '0.4'}
    
    # Make sure nothing is a JAX array...
    for entry in result:
        if result[entry] is not None:
            try:
                result[entry] = np.array(result[entry])
            except:
                pass


    output_folder = args.outdir + '/output/' 
    o.mkdir_safe(output_folder)
    o.save_covar_output_volumes(output_folder, means['combined'], u['rescaled'], s, volume_mask, volume_shape,  voxel_size = cryos[0].voxel_size)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    o.save_volume(dilated_volume_mask, output_folder + 'volumes/' + 'dilated_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    o.save_volume(focus_mask, output_folder + 'volumes/' + 'focus_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    if args.use_complement_mask:
        o.save_volume(focus_masks[0], output_folder + 'volumes/' + 'complement_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)


    utils.pickle_dump(covariance_cols, output_model_folder + 'covariance_cols.pkl')
    utils.pickle_dump(result, output_model_folder + 'params.pkl')

    embedding_dict = { 'zs': zs, 'cov_zs' : cov_zs , 'contrasts': est_contrasts, 'zs_cont' : zs_cont, 'cov_zs_cont' : cov_zs_cont, 'contrasts_cont' : est_contrasts_cont}

    if args.tilt_series:
        particles_ind_split = [ cryo.image_stack.dataset_tilt_indices for cryo in cryos]
    else:
        particles_ind_split = ind_split

    utils.pickle_dump(particles_ind_split, output_model_folder + 'particles_halfsets.pkl')
    # Always dump to know where to load... Maybe wasteful?
    if True:#args.halfsets is None:
        pickle.dump(ind_split, open(output_model_folder + 'halfsets.pkl', 'wb'))
        args.halfsets = output_model_folder + 'halfsets.pkl'


    for entry in embedding_dict:
        for key in embedding_dict[entry]:
            if entry == 'contrasts' and args.tilt_series and ('shared' not in options['contrast']):
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(embedding_dict[entry][key], ind_split)
            else:
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(embedding_dict[entry][key], particles_ind_split)

            # for k in range(num_foc_masks-1):
            #     embedding_dict[entry][key][k] = embedding_dict[entry][key][k].astype(np.float32)
    utils.pickle_dump(embedding_dict, output_model_folder + 'embeddings.pkl')
    if args.use_complement_mask:
        utils.pickle_dump(zs_full, output_model_folder + 'zs_with_complement.pkl')

    logger.info(f"Dumped results to file:, {output_model_folder}results.pkl")
    
    logger.info(f"total time: {time.time() - st_time}")
    
    # from analyze import analyze
    # analyze(args.outdir, output_folder = None, zdim=  np.max(options['zs_dim_to_test']), n_clusters = 40, n_paths= 2, skip_umap = False, q=None, n_std=None )

    return means, u, s, volume_mask, dilated_volume_mask, noise_var_used 




def main():
    # import jax
    parser = argparse.ArgumentParser(description=__doc__)

    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)

    ## Make plots
    from recovar import output
    po = output.PipelineOutput(args.outdir + '/')
    zdims = np.array(args.zdim)
    zdim_choose = np.argmin(np.abs(zdims - 10))
    zdim = zdims[zdim_choose]
    output.standard_pipeline_plots(po, zdim, args.outdir + '/output/plots/')


if __name__ == "__main__":
    main()