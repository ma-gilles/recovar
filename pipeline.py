# If you want to extend and use recovar, you should import this first
import recovar.config 
import jax.numpy as jnp
import numpy as np

import os, argparse, time, pickle, logging
from recovar import output as o
from recovar import dataset, homogeneous, embedding, principal_components, latent_density, mask, utils, constants, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

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

    parser.add_argument('--zdim', type=list_of_ints, default=[4,10,20], help="Dimension of latent variable")

    # parser.add_argument(
    #     "--zdim", type=list, help="Dimension of latent variable"
    # )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=True, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--mask", metavar="mrc", default=None, type=os.path.abspath, help="mask (.mrc)"
    )

    parser.add_argument(
        "--mask-option", metavar=str, default="from_halfmaps", help="mask options: from_halfmaps (default), input, sphere, none"
    )

    parser.add_argument(
        "--mask-dilate-iter", type=int, default=0, dest="mask_dilate_iter", help="mask options how many iters to dilate input mask (only used for input mask)"
    )

    parser.add_argument(
        "--correct-contrast",
        dest = "correct_contrast",
        action="store_true",
        help="estimate and correct for amplitude scaling (contrast) variation across images "
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
        help="Filter particles by these indices",
    )

    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "automatic",
        help="Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0",
    )

    group.add_argument(
        "--rerescale",
        dest = "rerescale",
        action="store_true",
    )


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
    # group.add_argument(
    #     "--lazy",
    #     action="store_true",
    #     help="Lazy loading if full dataset is too large to fit in memory",
    # )

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
            default = "white",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
        )

    group.add_argument(
            "--mean-fn",
            dest = "mean_fn",
            default = "old",
            help="which mean function to use. Options are old (default), triangular, triangular_reg"
        )

    group = parser.add_argument_group("Covariance estimation options")


    group.add_argument(
            "--covariance-fn",
            dest = "covariance_fn",
            default = "noisemask",
            help="noisemask (default), kernel"
        )

    group.add_argument(
            "--covariance-reg-fn",
            dest = "covariance_reg_fn",
            default = "old",
            help="old (default), new"
        )

    group.add_argument(
            "--covariance-left-kernel",
            dest = "covariance_left_kernel",
            default = "triangular",
            help="triangular (default), square"
        )

    group.add_argument(
            "--covariance-right-kernel",
            dest = "covariance_right_kernel",
            default = "triangular",
            help="triangular (default), square"
        )

    group.add_argument(
            "--covariance-left-kernel-width",
            dest = "covariance_left_kernel_width",
            default = 1,
            type=int,
        )

    group.add_argument(
            "--covariance-right-kernel-width",
            dest = "covariance_right_kernel_width",
            default = 2,
            type=int,
        )
    
    # options = {
    #     "covariance_fn": "noisemask",
    #     "reg_fn": "old",
    #     "left_kernel": "triangular",
    #     "right_kernel": "triangular",
    #     "left_kernel_width": 1,
    #     "right_kernel_width": 2,
    #     "shift_fsc": False,
    #     "substract_shell_mean": False,
    #     "grid_correct": True,
    #     "use_spherical_mask": True,
    #     "use_mask_in_fsc": False,
    #     "column_radius": 5,
    # }

    group.add_argument(
            "--covariance-shift-fsc",
            dest = "covariance_shift_fsc",
            action="store_true",
        )


    group.add_argument(
            "--covariance-substract-shell-mean",
            dest = "covariance_substract_shell_mean",
            action="store_true",
        )

    group.add_argument(
            "--covariance-grid-correct",
            dest = "covariance_substract_shell_mean",
            action="store_true",
        )



    group.add_argument(
            "--covariance-mask-in-fsc",
            dest = "covariance_mask_in_fsc",
            action="store_true",
        )



    group.add_argument(
            "--n-covariance-columns",
            dest = "covariance_reg_fn",
            default = "old",
            help="old (default), new"
        )


    return parser
    


def standard_recovar_pipeline(args):
    # import pdb; pdb.set_trace()
    st_time = time.time()

    o.mkdir_safe(args.outdir)
    logger.addHandler(logging.FileHandler(f"{args.outdir}/run.log"))
    logger.info(args)
    ind_split = dataset.figure_out_halfsets(args)

    dataset_loader_dict = dataset.make_dataset_loader_dict(args)
    options = utils.make_algorithm_options(args)

    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split)
    cryo = cryos[0]
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = cryo.volume_shape
    disc_type = "linear_interp"

    batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    logger.info(f"image batch size: {batch_size}")
    logger.info(f"volume batch size: {utils.get_vol_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"column batch size: {utils.get_column_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"number of images: {cryos[0].n_images + cryos[1].n_images}")
    utils.report_memory_device(logger=logger)

    cov_noise, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    # I need to rewrite the reweighted so it can use the more general noise distribution, but for now I'll go with that. 
    cov_noise_init = cov_noise
    valid_idx = cryo.get_valid_frequency_indices()
    noise_model = args.noise_model

    # Compute mean
    if args.mean_fn == 'old':
        means, mean_prior, _, _ = homogeneous.get_mean_conformation(cryos, 5*batch_size, cov_noise , valid_idx, disc_type, use_noise_level_prior = False, grad_n_iter = 5)
        use_adaptive = False
    elif args.mean_fn == 'triangular':
        means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = cov_noise,  use_regularization = False)
    elif args.mean_fn == 'triangular_reg':
        means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = cov_noise,  use_regularization = True)
    else:
        raise ValueError(f"mean function {args.mean_fn} not recognized")
    

    # if use_adaptive:
    #     for cryo_idx, cryo in enumerate(cryos):
    #         means['adaptive' + str(cryo_idx)], means['adaptive' + str(cryo_idx)+'_h'] = homogeneous.compute_with_adaptive_discretization(cryo, means['lhs'], means['prior'], means['combined'], cov_noise, 1*batch_size)
    #     means['combined'] = (means['adaptive' + str(0)] + means['adaptive' + str(1)])/2

    means['indices'] = [cryo.dataset_indices for cryo in cryos ]
    utils.pickle_dump(means, args.outdir + '/means.pkl')
    
    mean_real = ftu.get_idft3(means['combined'].reshape(cryos[0].volume_shape))

    ## DECIDE IF WE SHOULD UNINVERT DATA
    uninvert_check = np.sum((mean_real.real**3 * cryos[0].get_volume_radial_mask(cryos[0].grid_size//3).reshape(cryos[0].volume_shape))) < 0
    if args.uninvert_data == 'automatic':
        # Check if in real space, things towards the middle are mostly positive or negative
        if uninvert_check:
        # if np.sum(mean_real.real**3 * cryos[0].get_volume_mask() ) < 0:
            for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
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

    # Compute mask
    volume_mask, dilated_volume_mask= mask.masking_options(args.mask_option, means, volume_shape, args.mask, cryo.dtype_real, args.mask_dilate_iter)

    # Let's see?
    
    noise_time = time.time()
    # Probably should rename all of this...
    noise_var_outside_mask, std_noise_var, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)

    if args.mask_option is not None:
        noise_var_outside_mask, per_pixel_noise,_ =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)

        cov_noise = noise.estimate_white_noise_variance_from_mask(cryo, dilated_volume_mask, batch_size)
        cov_noise_white_second = cov_noise.copy()
    else:
        cov_noise_white_second = cov_noise_init
        noise_var_outside_mask = cov_noise_init * np.ones_like(noise_var_outside_mask)

    logger.info(f"time to estimate noise is {time.time() - noise_time}")

    # I believe that some versino of this is how relion/cryosparc infer the noise, but it seems like it would only be correct for homogeneous datasets
    ub_noise_var, std_ub_noise_var, _, _ =  noise.estimate_radial_noise_upper_bound_from_inside_mask(cryo, means['combined'], dilated_volume_mask, batch_size)

    ub_noise_var, _,_ =  noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(cryo, means['combined'], dilated_volume_mask, batch_size)

    # noise_var_outside_mask, per_pixel_noise =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)

    noise_time = time.time()
    logger.info(f"time to upper bound noise is {time.time() - noise_time}")
    noise_var = np.where(noise_var_outside_mask >  ub_noise_var, ub_noise_var, noise_var_outside_mask)

    logger.warning("doing funky noise business")
    noise_var = np.where(noise_var_outside_mask >  cov_noise_init, noise_var_outside_mask, np.ones_like(cov_noise_init))

    # noise_var_ = np.where(noise_var_outside_mask >  ub_noise_var, ub_noise_var, noise_var_outside_mask)


    noise_var = noise_var_outside_mask
    # Noise statistic
    if noise_model == "white":
        cov_noise = np.ones_like(noise_var)*cov_noise
    else:
        cov_noise = noise_var


    # Compute principal components
    u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model)

    if options['ignore_zero_frequency']:
        # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
        logger.info('ignoring zero frequency')
        noise_var[0] *=1e16

    image_cov_noise = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].image_shape))

    if not args.keep_intermediate:
        del u['real']
        if 'rescaled_no_contrast' in u:
            del u['rescaled_no_contrast']
        covariance_cols = None

    # Compute embeddings
    zs = {}; cov_zs = {}; est_contrasts = {}        
    for zdim in options['zs_dim_to_test']:
        z_time = time.time()
        zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] , zdim,
                                                                image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                contrast_grid = None, contrast_option = options['contrast'],
                                                                ignore_zero_frequency = options['ignore_zero_frequency'] )
        logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")

    ndim = np.max(options['zs_dim_to_test'])
    cov_noise, _,_ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(cryo, dilated_volume_mask, means['combined'], u['rescaled'][:,:ndim], est_contrasts[zdim], zs[zdim], batch_size//10, disc_type = 'linear_interp')
    cov_noise_second = cov_noise.copy()
    rerun = True
    # ### END OF DEL

    logger.info(f"embedding time: {time.time() - st_time}")

    utils.report_memory_device()

    # Compute latent space density
    # Precompute the density on the 4D grid. This is the most expensive part of computing trajectories, and can be reused across trajectories. 
    logger.info(f"starting density computation")
    density_z = 10 if 10 in zs else options['zs_dim_to_test'][0]
    density, latent_space_bounds  = latent_density.compute_latent_space_density(zs[density_z], cov_zs[density_z], pca_dim_max = 4, num_points = 50)
    logger.info(f"ending density computation")

    # Dump results to file
    output_model_folder = args.outdir + '/model/'
    o.mkdir_safe(args.outdir)
    o.mkdir_safe(output_model_folder)

    logger.info(f"peak gpu memory use {utils.get_peak_gpu_memory_used(device =0)}")

    if args.halfsets is None:
        pickle.dump(ind_split, open(output_model_folder + 'halfsets.pkl', 'wb'))
        args.halfsets = output_model_folder + 'halfsets.pkl'

    result = { 'means':means, 'u': u, 's':s, 'volume_mask' : volume_mask,
               'dilated_volume_mask': dilated_volume_mask,
                'zs': zs, 'cov_zs' : cov_zs , 'contrasts': est_contrasts, 'cov_noise': cov_noise_init,
                'input_args' : args,
                'latent_space_bounds' : np.array(latent_space_bounds), 
                'density': np.array(density),
                'noise_var_outside_mask' : np.array(noise_var_outside_mask),
                'ub_noise_var' : np.array(ub_noise_var),
                'noise_var' : np.array(noise_var),
                'std_noise_var' : np.array(std_noise_var),
                'image_PS' : np.array(image_PS),
                'std_image_PS' : np.array(std_image_PS),
                'column_fscs': column_fscs, 
                'covariance_cols': covariance_cols, 
                'picked_frequencies' : picked_frequencies,
                 'cov_noise_white_second' : cov_noise_white_second }
    
    if rerun:
        result['cov_noise_second'] = cov_noise_second

    with open(output_model_folder + 'results.pkl', 'wb') as f :
        pickle.dump(result, f)
    logger.info(f"Dumped results to file:, {output_model_folder}results.pkl")

    output_folder = args.outdir + '/output/' 
    o.mkdir_safe(output_folder)
    o.save_covar_output_volumes(output_folder, means['combined'], u['rescaled'], s, volume_mask, volume_shape)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False)
    o.save_volume(dilated_volume_mask, output_folder + 'volumes/' + 'dilated_mask', volume_shape, from_ft = False)
    logger.info(f"total time: {time.time() - st_time}")
    
    # from analyze import analyze
    # analyze(args.outdir, output_folder = None, zdim=  np.max(options['zs_dim_to_test']), n_clusters = 40, n_paths= 2, skip_umap = False, q=None, n_std=None )

    return means, u, s, volume_mask, dilated_volume_mask, cov_noise 


if __name__ == "__main__":
    # import jax
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)
