# If you want to extend and use recovar, you should import this first
import recovar.config 
import jax.numpy as jnp
import numpy as np

import os, argparse, time, pickle, logging
from recovar import output as o
from recovar import dataset, homogeneous, embedding, principal_components, latent_density, mask, plot_utils, utils, constants, noise
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
            action="store_false",
            help="saves some intermediate result. Probably only useful for debugging"
        )


    group.add_argument(
            "--noise-model",
            dest = "noise_model",
            default = "white",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
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

    cov_noise, _ = homogeneous.estimate_noise_variance(cryos[0], batch_size)

    # I need to rewrite the reweighted so it can use the more general noise distribution, but for now I'll go with that. 
    cov_noise_init = cov_noise
    valid_idx = cryo.get_valid_frequency_indices()
    noise_model = args.noise_model

    # Compute mean
    means, mean_prior, _, _ = homogeneous.get_mean_conformation(cryos, 5*batch_size, cov_noise , valid_idx, disc_type, use_noise_level_prior = False, grad_n_iter = 5)
    
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
    noise_var, std_noise_var, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)
    logger.info(f"noise time estimate is {time.time() - noise_time}")

    # Noise statistic
    if noise_model == "white":
        cov_noise = cov_noise
    else:
        cov_noise = noise_var

    # noise_var, std_noise_var, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)

    # Compute principal components
    u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model, disc_type = 'linear_interp', radius = constants.COLUMN_RADIUS) 

    if options['ignore_zero_frequency']:
        # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
        logger.info('ignoring zero frequency')
        noise_var[0] *=1e16

    # 
    if noise_model == "white":
        image_cov_noise = cov_noise
    else:
        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var, cryos[0].image_shape))

    if not args.keep_intermediate:
        del u['real']
        if 'rescaled_no_contrast' in u:
            del u['rescaled_no_contrast']
        covariance_cols = None


    # Compute embeddings
    zs = {}; cov_zs = {}; est_contrasts = {}        
    for zdim in options['zs_dim_to_test']:
        z_time = time.time()
        if zdim ==1:
            ss =  s['rescaled'] + 1e16
            print('BUMP EIGS BIG TIME')
        else:
            ss = s['rescaled']

        #re-rescale
        # rescale_eigs
        rerescale = False
        print('rerescale option is off')
        #rerescale = args.rerescale
        if rerescale:
            u[f"rescaled_{zdim}"],s[f"rescaled_{zdim}"], _ = principal_components.rescale_eigs(cryos, u['rescaled'],s['rescaled'], means['combined'], volume_mask, image_cov_noise, basis_size = zdim, gpu_memory_to_use = utils.get_gpu_memory_total(), use_mask = True, ignore_zero_frequency = options['ignore_zero_frequency'])

            zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(means['combined'], u[f"rescaled_{zdim}"], s[f"rescaled_{zdim}"] , zdim,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )
            
            var_zs = np.var(zs[zdim], axis =0)
            mean_cov_zs = np.diag(np.mean(cov_zs[zdim], axis=0))
            mean_cov_zs_unregged = np.diag(np.mean(cov_zs[zdim] - np.diag(1/s[f"rescaled_{zdim}"]) , axis=0))
            # predicted

            pred_eigs = var_zs/mean_cov_zs_unregged**2 * mean_cov_zs**2
            s[f"rescaled_{zdim}_guess"] = pred_eigs
            # Estimate eigenvalues:

            zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(means['combined'], u[f"rescaled_{zdim}"], s[f"rescaled_{zdim}"] , zdim,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )

        else:
            zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(means['combined'], u['rescaled'], ss , zdim,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )
        logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")

        # if zdim ==1:
        #     zs[-1], cov_zs[-1], est_contrasts[-1] = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] + 1e16 , zdim,
        #                                                             image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
        #                                                             contrast_grid = None, contrast_option = options['contrast'],
        #                                                             ignore_zero_frequency = options['ignore_zero_frequency'] )


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
                'noise_var' : np.array(noise_var),
                'std_noise_var' : np.array(std_noise_var),
                'image_PS' : np.array(image_PS),
                'std_image_PS' : np.array(std_image_PS),
                'column_fscs': column_fscs, 
                'covariance_cols': covariance_cols, 
                'picked_frequencies' : picked_frequencies }
    

    with open(output_model_folder + 'results.pkl', 'wb') as f :
        pickle.dump(result, f)
    logger.info(f"Dumped results to file:, {output_model_folder}results.pkl")

    output_folder = args.outdir + '/output/' 
    o.mkdir_safe(output_folder)
    o.save_covar_output_volumes(output_folder, means['combined'], u['rescaled'], s, volume_mask, volume_shape)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False)
    o.save_volume(dilated_volume_mask, output_folder + 'volumes/' + 'dilated_mask', volume_shape, from_ft = False)
    logger.info(f"total time: {time.time() - st_time}")

    return means, u, s, volume_mask, dilated_volume_mask, cov_noise 

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)
