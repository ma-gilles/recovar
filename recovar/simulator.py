import jax.numpy as jnp
import numpy as np
import jax, functools
from recovar import core, dataset, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils
import jax.scipy.spatial.transform
import os
import mrcfile
# import matplotlib.pyplot as plt
# from cryodrgn import ctf
# from cryodrgn.pose import PoseTracker
from recovar import cryodrgn_load
# xx = Path(__file__).resolve()
import recovar.simulate_scattering_potential as gsm
import logging
CONSTANT_CTF=False
logger = logging.getLogger(__name__)

data_path = os.path.join(os.path.dirname(__file__),'data/')

# Two generators that load ctf and poses from real datasets
def get_dataset_params(n_images, grid_size, ctf_file, poses_file):
    ctf_params = np.array(cryodrgn_load.load_ctf_for_training(grid_size, ctf_file))
    
    # Initialize bfactor == 0
    ctf_params = np.concatenate( [ctf_params, np.zeros_like(ctf_params[:,0][...,None])], axis =-1)
    
    # Initialize constrast == 1
    ctf_params = np.concatenate( [ctf_params, np.ones_like(ctf_params[:,0][...,None])], axis =-1)
    rots, trans, _ = cryodrgn_load.load_poses(poses_file, n_images, grid_size, ind = None) 
    return ctf_params, np.array(rots), np.array(trans)

def load_first_dataset_params(grid_size):
    n_images = 131899
    ctf_file = os.path.join(data_path, 'ctf_10076.pkl')
    poses_file = os.path.join(data_path, 'poses_10076.pkl')
    return get_dataset_params(n_images, grid_size, ctf_file, poses_file)

def load_second_dataset_params(grid_size):
    n_images = 327490
    ctf_file = os.path.join(data_path, 'ctf_10180.pkl')
    poses_file = os.path.join(data_path, 'poses_10180.pkl')
    return get_dataset_params(n_images, grid_size, ctf_file, poses_file)


def set_constant_ctf(ctf_params_data):
    # Set constant CTF
    ctf_params_data[:,0] = 0
    ctf_params_data[:,1] = 0
    # ctf_params_data[:,2] = 0.0
    # ctf_params_data[:,3] = 0.0
    ctf_params_data[:,4] = 0.0
    ctf_params_data[:,5] = -1
    # ctf_params_data[:,4] = 0.0
    return ctf_params_data

def generate_simulated_params_from_real(n_images, dataset_params_fn, grid_size  ):
    
    ctf_params_data, rots_data, trans_data = dataset_params_fn(grid_size)

    # Sample rotations from them:
    ind = np.random.choice(ctf_params_data.shape[0], n_images)
    # Choose random CTFs
    sample_ctf_params = ctf_params_data[ind]


    # Sample rotations independently?
    ind = np.random.choice(ctf_params_data.shape[0], n_images)
    sample_rots = rots_data[ind]

    ind = np.random.choice(ctf_params_data.shape[0], n_images)
    sample_trans = trans_data[ind]
    # if not sample_trans:
    #     sample_trans = 0 * sample_trans # in theory, distribution of rotations doesn't matter, but it can make the particle leave the image and such
        # However, when using masks, it is unclear that this is necessarily true, since the mask may exit the image or something.
    # import pdb; pdb.set_trace()

    sample_ctf_params = sample_ctf_params[...,1:]
    # import pdb; pdb.set_trace()
    if CONSTANT_CTF:
        sample_ctf_params = set_constant_ctf(sample_ctf_params)
    # import pdb; pdb.set_trace()
    return sample_ctf_params, sample_rots, sample_trans

def get_params_generator(dataset_params_fn ):
    def params_generator(n_images, grid_size):
        return generate_simulated_params_from_real(n_images, dataset_params_fn, grid_size )
    return params_generator

## A uniform pose generator
def random_sampling_scheme(n_images, grid_size, seed =0, uniform = True ):
    
    np.random.seed(seed)
    dataset_params_fn = load_first_dataset_params
    ctf_params, _, _ = generate_simulated_params_from_real(n_images, dataset_params_fn, grid_size  )
    if uniform:
        rotations = uniform_rotation_sampling(n_images, grid_size, seed = seed )
    else:
        rotations = nonuniform_rotation_sampling(n_images, grid_size, seed = seed )
    translations = np.zeros([n_images,2])
    return ctf_params, rotations, translations





def uniform_rotation_sampling(n_images, grid_size, seed = 0 ):
    from scipy.spatial.transform import Rotation
    # Rotation.random(type cls, num=None, random_state=None)
    # key = jax.random.PRNGKey(seed)
    # rotations = np.array(jax.scipy.spatial.transform.Rotation.random(key, n_images))
    rotations = Rotation.random(n_images).as_matrix()
    return rotations

def nonuniform_rotation_sampling(n_images, grid_size, seed = 0 ):
    from scipy.spatial.transform import Rotation 
    rotation_matrices = [] 
    xs = []
    for rot_idx in range(n_images):

        y = np.random.randn(1)*np.pi * 0.02 + np.pi/2
        x = np.random.randn(1)*np.pi * 0.1 + np.pi/2

        random_rot = Rotation.from_euler('xyz', [y[0],x[0],0 ] )
        rotation_matrices.append(random_rot.as_matrix())
        xs.append(x)

    # rotation_matrices[0] = Rotation.from_euler('xyz', [0,0,0] ).as_matrix()
    # rotation_matrices[1] = Rotation.from_euler('xyz', [0,np.pi/2,0] ).as_matrix()
    # rotation_matrices[2] = Rotation.from_euler('xyz', [np.pi/2,0,0] ).as_matrix()
    # rotation_matrices[3] = Rotation.from_euler('xyz', [0,0,np.pi/2] ).as_matrix()
    # rotation_matrices[4] = Rotation.from_euler('xyz', [0,0,0] ).as_matrix()
    # rotation_matrices[5] = Rotation.from_euler('xyz', [0,np.pi,0] ).as_matrix()
    # rotation_matrices[6] = Rotation.from_euler('xyz', [np.pi,0,0] ).as_matrix()
    # rotation_matrices[7] = Rotation.from_euler('xyz', [0,0,np.pi] ).as_matrix()

    rotation_matrices = np.array(rotation_matrices)
    # uniform_rots = uniform_rotation_sampling(n_images, grid_size, seed = seed)
    # rand_pick = np.random.randint(0, n_images, n_images//10)
    # rotation_matrices[rand_pick,:] = uniform_rots[rand_pick,:]

    return rotation_matrices


## The two main generators
def get_pose_ctf_generator(option):
    if option == "uniform":
        return random_sampling_scheme
    elif option == "dataset1":
        return get_params_generator(load_first_dataset_params)
    elif option == "nonuniform":
        f = lambda x,y=0,z=0: random_sampling_scheme(x, y, z, uniform = False )
        return f
    else:
        return get_params_generator(load_second_dataset_params)


def generate_contrast_params(n_images,noise_scale_std, contrast_std ):
    contrast = 1 + np.random.randn(n_images) * contrast_std
    noise_scale = 1 + np.random.randn(n_images) * noise_scale_std
    return contrast, noise_scale



def generate_volumes_from_mrcs(mrc_names, grid_size_i = None, padding= 0 ):
    ## TODO rewrite this function.
    Xs_vec = []
    first_vol = True
    idx = 0 
    ftu = fourier_transform_utils(jnp)
    for mrc_name in mrc_names:
        if idx % 100 == 0:
            logger.info(f"Loading volume {idx}")
        idx+=1
        data, voxel_size = utils.load_mrc(mrc_name, return_voxel_size = True)
        voxel_size = voxel_size.x
        # data, header =  mrc.parse_mrc(mrc_name)
        # data = np.transpose(data, (2,1,0))
        # voxel_size = header.fields['xlen'] / header.fields['nx']
        mrc_grid_size = data.shape[0]
        
        if grid_size_i is None:
            grid_size = mrc_grid_size
        else:
            grid_size = grid_size_i - padding
            
        if first_vol:
            first_vol = False
            first_voxel_size = voxel_size
        else:
            assert( first_voxel_size == voxel_size)
        

        if mrc_grid_size == grid_size:
            # if return_ft:
            #     X_padded = ftu.get_dft3(data)
            X_padded = ftu.get_dft3(data)
        else:
            # Zero out grid_sizes outside radius
            X = ftu.get_dft3(data)
            # Downsample/Upsample
            if mrc_grid_size > grid_size:
                diff_size = mrc_grid_size - grid_size 
                half_diff_size = diff_size//2
                X = X[half_diff_size:-half_diff_size,half_diff_size:-half_diff_size,half_diff_size:-half_diff_size ]
                X = np.array(X)
                X[0,:,:] = 0 
                X[:,0,:] = 0 
                X[:,:,0] = 0 
                X = jnp.asarray(X)
            elif mrc_grid_size < grid_size:
                diff_size = grid_size - mrc_grid_size 
                half_diff_size = diff_size//2            
                X_new = np.zeros( 3*[grid_size] , dtype = X.dtype )
                X_new[half_diff_size:-half_diff_size,half_diff_size:-half_diff_size,half_diff_size:-half_diff_size ] = X
                X = jnp.asarray(X_new)

            # Pad in real space.
            X = ftu.get_idft3(X)
            X_padded = np.zeros_like(X, shape = np.array(X.shape) + padding )
            half_pad = padding//2
            X_padded[half_pad:X.shape[0] + half_pad,half_pad:X.shape[1] + half_pad,half_pad:X.shape[2] + half_pad] = X
            
            # Back to FT space
            X_padded = ftu.get_dft3(X_padded)

        Xs_vec.append(np.array(X_padded.reshape(-1)))
    voxel_size = voxel_size / grid_size * mrc_grid_size
    return np.stack(Xs_vec), voxel_size

def get_noise_model(option, grid_size):
    if option == "white":
        return np.ones(grid_size//2-1) 
    elif option == "radial1":
        noise_file = os.path.join(data_path, 'noise_10076.pkl')
        return utils.pickle_load(noise_file)


def generate_synthetic_dataset(output_folder, voxel_size,  volumes_path_root, n_images, outlier_file_input = None, grid_size = 128,
                               volume_distribution = None,  dataset_params_option = "dataset1", noise_level = 1, 
                               noise_model = "radial1", put_extra_particles = True, percent_outliers = 0.1, 
                               volume_radius = 0.9, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.3, contrast_std =0.3, disc_type = 'linear_interp', n_tilts = -1, dose_per_tilt = 3, angle_per_tilt = 3, image_dtype = np.float16, image_offset_n_std = 0.0, per_particle_contrast=True ):
    from recovar import output
    output.mkdir_safe(output_folder)
    volumes = load_volumes_from_folder(volumes_path_root, grid_size, trailing_zero_format_in_vol_name, normalize = False )
    scale_vol = 1 / np.mean(np.linalg.norm(volumes, axis =(-1)))
    volumes *= scale_vol

    vol_shape = utils.guess_vol_shape_from_vol_size(volumes.shape[-1])
    # import matplotlib.pyplot as plt
    # plt.imshow(ftu.get_idft3(volumes[0].reshape(vol_shape)).real.sum(axis=0))
    # Maybe normalize volumes?
    # volumes /= np.linalg.norm(volumes, axis =(-1))
    volume_distribution = np.ones(volumes.shape[0]) / volumes.shape[0] if volume_distribution is None else volume_distribution

    outlier_volume = ftu.get_dft3(utils.load_mrc(outlier_file_input)).reshape(-1) if outlier_file_input is not None else None

    dataset_param_generator = get_pose_ctf_generator(dataset_params_option)
    noise_variance = get_noise_model(noise_model, grid_size) / 50000 * noise_level

    ## TODO
    # Scale noise and volumes so that images have approximately std =1?
    # noise_image = noise.make_radial_noise(noise_variance, experiment_dataset.image_shape).reshape(experiment_dataset.image_shape)
    # key, subkey = jax.random.split(key)
    # noise_batch = make_noise_batch(subkey, noise_image, images_batch.shape)

    # mrcf = mrcfile.new(output_folder + '/particles.'+str(grid_size)+'.mrcs',overwrite=True)
    mrc_file = None# mrcfile.new_mmap( output_folder + '/particles.'+str(grid_size)+'.mrcs', shape=(n_images, grid_size, grid_size), mrc_mode=2, overwrite = True)

    rescale_noise = True
    if rescale_noise:
        main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, _ = generate_simulated_dataset(volumes, voxel_size, volume_distribution, 10, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = volume_radius, outlier_volume = outlier_volume, disc_type = disc_type, mrc_file = mrc_file, image_offset_n_std= image_offset_n_std, per_particle_contrast=per_particle_contrast )
        norm_image_square = np.mean(main_image_stack**2)
        norm_image = (norm_image_square)

        # print(norm_image)
        noise_variance = noise_variance / (norm_image)
        volumes = volumes / np.sqrt(norm_image)
        scale_vol =  scale_vol / np.sqrt(norm_image)

        # main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size = generate_simulated_dataset(volumes, voxel_size, volume_distribution, 10, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = volume_radius, outlier_volume = outlier_volume, disc_type = disc_type, mrc_file = mrc_file )
        # norm_image = np.mean(np.linalg.norm(main_image_stack, axis = (-1,-2)))
        # Scale noise and volumes so that images have approximately std =1?

    # First make some dataset to figure out a good scaling?
    main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, tilt_groups = generate_simulated_dataset(volumes, voxel_size, volume_distribution, n_images, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = volume_radius, outlier_volume = outlier_volume, disc_type = disc_type, mrc_file = mrc_file, n_tilts = n_tilts, 
    dose_per_tilt = dose_per_tilt, angle_per_tilt = angle_per_tilt, image_offset_n_std= image_offset_n_std , per_particle_contrast=per_particle_contrast)


    simulation_info['volumes_path_root'] = volumes_path_root
    simulation_info['grid_size'] = grid_size
    simulation_info['trailing_zero_format_in_vol_name'] = trailing_zero_format_in_vol_name
    simulation_info['disc_type'] = disc_type
    simulation_info['scale_vol'] = scale_vol
    simulation_info['dose_per_tilt'] = dose_per_tilt
    simulation_info['angle_per_tilt'] = angle_per_tilt

    particles_file = output_folder + '/particles.'+str(grid_size)+'.mrcs'

    with mrcfile.new(particles_file ,overwrite=True) as mrc:
        mrc.set_data(main_image_stack.astype(image_dtype))
        mrc.voxel_size = voxel_size
    poses = (rots.astype(np.float32), trans.astype(np.float32))
    utils.pickle_dump(poses, output_folder + '/poses.pkl')
    save_ctf_params(output_folder, grid_size, ctf_params, voxel_size)
    utils.pickle_dump(simulation_info, output_folder + '/simulation_info.pkl' )

    utils.write_starfile(ctf_params, rots.astype(np.float32), trans.astype(np.float32), voxel_size, grid_size, particles_file, output_folder + '/particles.star', halfset_indices = None, tilt_groups = tilt_groups )

    return main_image_stack, simulation_info

def load_volumes_from_folder(volumes_path_root, grid_size, trailing_zero_format_in_vol_name = False, normalize = True):

    if trailing_zero_format_in_vol_name:
        def make_file(k):
            # return os.path.join(volumes_path_root,   format(k, '04d')+".mrc")
            return volumes_path_root + format(k, '04d')+".mrc"

    else:
        def make_file(k):
            return volumes_path_root + f"{k}.mrc"
            # return os.path.join(volumes_path_root,  f"{k}.mrc")

    
    idx =0 
    files = []
    while(os.path.isfile(make_file(idx))):
        files.append(make_file(idx))
        # import pdb; pdb.set_trace()
        idx+=1
    volumes, voxel_size = generate_volumes_from_mrcs(files, grid_size, padding= 0 )
    if normalize:
        volumes /= np.mean(np.linalg.norm(volumes, axis =(-1)))
    return volumes


def generate_simulated_dataset(volumes, voxel_size, volume_distribution, n_images, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = 0.95, outlier_volume = None, disc_type = 'linear_interp', mrc_file = None, n_tilts = -1, dose_per_tilt = None, angle_per_tilt = None, voltage = 100, image_offset_n_std = 0.0, per_particle_contrast= True ):
    
    # voxel_size = 
    volume_shape = utils.guess_vol_shape_from_vol_size(volumes[0].size)
    grid_size = volume_shape[0]

    ctf_params, rots, trans = dataset_param_generator(n_images, grid_size)

    if "ewald" in disc_type:
        phase_shift = np.arcsin(ctf_params[:,core.w_ind]) / np.pi * 180
        ctf_params[:,core.w_ind] = 0
        ctf_params[:,core.phase_shift_ind] = phase_shift
        ctf_params[:,core.volt_ind] = 100
        # import pdb; pdb.set_trace()

    trans *=0 
    per_image_contrast, per_image_noise_scale = generate_contrast_params(n_images, noise_scale_std, contrast_std )

    image_assignments = np.random.choice(np.arange(volumes.shape[0]), size = n_images,  p = volume_distribution)

    if n_tilts >0:
        # Define tilt groups
        tilt_groups = np.arange(n_images) // n_tilts

        n_tilt_groups = np.max(tilt_groups)+1

        # Assign each tilt group to a particle
        image_assignments_tilt = np.random.choice(np.arange(volumes.shape[0]), size = n_tilt_groups,  p = volume_distribution)

        # Assign each image a tilt number
        tilt_numbers = np.arange(n_images) % n_tilts
        # Put contrast in the tilt groups?
        # This is how it is done in cryoDRGN. tilt number == the ranking in the tilt series by contrast? Seems a bit sus
        # TODO check this
        # logger.warning("A very arbitrary contrast per tilt number! FIX?")
        # ctf_params[:,core.contrast_ind] =  np.cos(  np.ceil(tilt_numbers/2) * angle_per_tilt )

        # Make a tilt series with symmetric angles
        x_angles_half = np.arange(n_tilts//2+1) * angle_per_tilt
        x_angles = np.zeros(n_tilts)
        x_angles[::2] = -x_angles_half[:-1]   
        x_angles[1::2] = x_angles_half[1:]   

        # ctf_params[:,core.contrast_ind] =  np.cos(  x_angles / 180 * np.pi )


        x_angles_zz = np.concatenate([x_angles[:,None], np.zeros([n_tilts,2])], axis = -1)
        from scipy.spatial.transform import Rotation    
        x_rotations = Rotation.from_euler('xyz', x_angles_zz, degrees=True).as_matrix()
        B_facs = -4 * (tilt_numbers + 0.5) * dose_per_tilt 

        for i in range(n_tilt_groups):
            image_assignments[tilt_groups == i] = image_assignments_tilt[i]
            ctf_params[tilt_groups == i,core.contrast_ind] =  np.cos(  x_angles / 180 * np.pi )

            ind = np.where(tilt_groups == i)[0]
            # sort_idxs = ctf_params[ind,core.contrast_ind].argsort()
            # ranks = np.empty_like(sort_idxs)
            # ranks[sort_idxs[::-1]] = np.arange(len(ind))
            # tilt_numbers[ind] = ranks
            zero_tilt_rot = rots[ind[0]]
            these_rot = zero_tilt_rot @ x_rotations
            rots[ind] = these_rot

            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # Tag it to the end
        if per_particle_contrast:
            per_tilt_contrast, _ = generate_contrast_params(n_tilt_groups, noise_scale_std, contrast_std )
            per_image_contrast = per_tilt_contrast[tilt_groups]
        else:
            per_tilt_contrast = None
            per_image_contrast, _ = generate_contrast_params(n_images, noise_scale_std, contrast_std )


        ## Set the ctf_scale_params to angle correction (this is how it is saved from WARP supposedly)
        angle_scale_correction = jnp.cos(x_angles * np.pi / 180)
        ctf_params[:,core.contrast_ind] = angle_scale_correction[tilt_numbers]

        # Angle ind is just set to 0 in this version
        dose =  (tilt_numbers + 0.5) * dose_per_tilt

        # The cryo-EM / angle (unsused?) / dose (used)
        # ctf_params_big = np.zeros([n_images, 11])
        ctf_params = np.concatenate([ctf_params, dose[:,None], np.zeros_like(tilt_numbers[:,None]) ], axis = -1)

    else:
        per_tilt_contrast = None
        tilt_groups = None
        image_assignments_tilt = None

    if n_tilts >0:
        # CTF_fun = core.get_cryo_ET_CTF_fun(dose_per_tilt, angle_per_tilt)
        CTF_fun = core.evaluate_ctf_wrapper_tilt_series_v2
    else:
        CTF_fun = core.evaluate_ctf_wrapper

    main_dataset = dataset.CryoEMDataset( None, voxel_size,
                              rots, trans, ctf_params, CTF_fun = CTF_fun, dataset_indices = None, grid_size = grid_size)
    
    mult = 1 if 'cubic' in disc_type else 5
    batch_size = mult * utils.get_image_batch_size(grid_size, utils.get_gpu_memory_total())

    # plt.imshow(main_dataset.get_CTF_image(0)); plt.colorbar()
    # plt.show();
    # import pdb; pdb.set_trace()
    # simulate_data(experiment_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None )
    # logger.warning("USING NEAREST")
    # disc_type = 'nearest'
    # disc_type = 'linear_interp'

    main_image_stack = simulate_data(main_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = disc_type, mrc_file = mrc_file )

    image_means = np.mean(main_image_stack, axis = (-1,-2))
    # image_means_mean = np.mean(main_image_stack, axis = (-1,-2))
    image_mean_std = np.std(image_means)
    logger.info(f"Image mean mean {np.mean(image_means)}, image mean std: {image_mean_std}")
    per_image_offset = np.random.randn(n_images) * image_mean_std * image_offset_n_std
    main_image_stack += per_image_offset[:,None,None]




    if put_extra_particles:
        # Make other particles with same ctf but different rots
        _, rots_2, trans_2 = dataset_param_generator(n_images, grid_size)
        _, per_image_noise_scale_2 = generate_contrast_params(n_images, noise_scale_std, contrast_std )

        # To first approximation, maybe it is good enough to move them radially 
        # random number on the circle
        trans_2 = np.random.randn(n_images, 2)
        trans_2/= np.linalg.norm(trans_2, axis =-1,keepdims=True)
        # Move the center by around twice the radius
        trans_2 *= 2 * volume_radius * volume_shape[0] / 2
        # print(np.mean(np.linalg.norm(trans_2, axis =-1)))
        # import pdb; pdb.set_trace()
        other_particles_dataset = dataset.CryoEMDataset( None, voxel_size,
                                rots_2, trans_2, ctf_params, CTF_fun = CTF_fun, dataset_indices = None, grid_size = grid_size)
        # No noise in this stack.
        extra_particles_image_stack = simulate_data(other_particles_dataset, volumes,  noise_variance * 0 ,  batch_size, image_assignments, per_image_noise_scale_2, per_image_noise_scale, seed =0, disc_type = disc_type, mrc_file = None, pad_before_translate= True )

        main_image_stack += extra_particles_image_stack

    if percent_outliers > 0:
        # Perhaps a reasonable way to throw in outliers is to put a different structure entirely with wrong angles

        assert outlier_volume is not None, "if you want outliers, need to provide a structure"
        # Make sure they are on the same scale

        outlier_volume  = outlier_volume / np.linalg.norm(outlier_volume) * np.mean(np.linalg.norm(volumes,axis=(-1)))
        n_outlier_images = np.round(percent_outliers * n_images).astype(int)
        ctf_params_3, rots_3, trans_3 = dataset_param_generator(n_outlier_images, grid_size)
        outlier_contrast, outlier_noise_scale = generate_contrast_params(n_outlier_images, noise_scale_std, contrast_std )

        outlier_particle_dataset = dataset.CryoEMDataset( None, voxel_size,
                                rots_3, trans_3, ctf_params_3, CTF_fun = CTF_fun, dataset_indices = None, grid_size = grid_size)
        
        outlier_particle_image_stack = simulate_data(outlier_particle_dataset, outlier_volume[None],  noise_variance ,  batch_size, np.zeros(n_outlier_images, dtype = int), outlier_noise_scale, outlier_contrast, seed =1, disc_type = disc_type, mrc_file = None )

        ind_outliers = np.random.choice(n_images, n_outlier_images, replace =False)
        main_image_stack[ind_outliers] = outlier_particle_image_stack
        image_assignments[ind_outliers] = -1


    if n_tilts > 0:
        # Note that b_facs are stored here just so that the get saved in the starfile in WARP style...
        ctf_params[:,core.bfactor_ind] = B_facs

    simulation_info = { 
        "ctf_params" : ctf_params,
        "rots" : rots,
        "trans" : trans,
        "per_image_contrast" : per_image_contrast,
        "per_image_noise_scale" : per_image_noise_scale,
        "per_image_offset" : per_image_offset,
        "image_assignment" : image_assignments,
        "noise_variance": noise_variance.astype(np.float32),
        "voxel_size": voxel_size,
        "tilt_series_assignment": image_assignments_tilt,
        "tilt_groups": tilt_groups,
        "per_tilt_contrast": per_tilt_contrast,
    }

    return main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, tilt_groups


def make_small_dataset(output_path = ".", grid_size=128, n_images = 1000):
    from recovar import simulate_scattering_potential
    import os
    this_file_path = os.path.dirname(__file__)
    atom_coeff_path = 'data/5nrl.cif'
    splice_path = os.path.join(this_file_path, atom_coeff_path)
    voxel_size = 2/256 * grid_size
    volume = simulate_scattering_potential.generate_molecule_spectrum_from_pdb_id(splice_path, voxel_size, grid_size)
    from recovar import output
    volume_path_root = output_path + "/gt_volumes/"
    output.save_volumes(volume[None], volume_path_root)

    generate_synthetic_dataset(output_path + "/dataset/", voxel_size,  volume_path_root, n_images = 1000, grid_size = grid_size)

    # output_dir = os.path.join(this_file_path, atom_coeff_path)


    return 

# def generate_data_and_save_to_file(outdir):

#     main_image_stack, ctf_params, rots, trans, simulation_info = generate_simulated_dataset(volumes, volume_distribution, voxel_size, n_images, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_data, volume_radius = 0.9, outlier_volume = None )
#     grid_size = main_image_stack.shape[-1]

#     #
#     with mrcfile.new(outdir + 'particles.'+str(grid_size)+'.mrcs') as mrc:
#         mrc.set_data(main_image_stack)
#         mrc.voxel_size = voxel_size

#     poses = (rots, trans)
#     pickle.dump(poses, open(outdir + '/poses.pkl', "wb"))
#     save_ctf_params(outdir, grid_size, ctf_params, voxel_size)
#     pickle.dump(simulation_info, open(outdir + '/simulation_info.pkl', "wb"))
#     return 

def save_ctf_params(outdir, D: int, ctf_params, voxel_size):

    assert D % 2 == 0
    # assert ctf_params.shape[1] == 9
    ctf_params_all = np.zeros([ctf_params.shape[0], ctf_params.shape[1] + 2])
    ctf_params_all[:,2:] = ctf_params
    ctf_params_all[:,0] = D
    ctf_params_all[:,1] = voxel_size
    # Throw away B factor and contrast, because that's what cryodrgn loader wants. Probably should change this.
    utils.pickle_dump(ctf_params_all[:,:9].astype(np.float32), outdir + '/ctf.pkl')
    return 


roll_batch = jax.vmap(lambda x,y,z: jax.numpy.roll(x,y,axis = z), in_axes = (0, 0, None))


# Solves the linear system Dx = b.
def simulate_data(experiment_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None, pad_before_translate = False, Bfactor=100 ):

    if disc_type == "pdb":
        gt_vols = [gsm.generate_volume_from_atoms(vol, voxel_size = experiment_dataset.voxel_size,  grid_size = experiment_dataset.grid_size,  freq_coords = None, jax_backend = False).reshape(-1) for vol in volumes ]
        B_fac_vols = [Bfactorize_vol(volume, experiment_dataset.voxel_size, Bfactor, experiment_dataset.volume_shape) for volume in gt_vols]
        gt_vols_norm = np.mean(np.linalg.norm(B_fac_vols, axis =(-1)))
        print(gt_vols_norm)

    key = jax.random.PRNGKey(seed)
    # A little bit of a hack to account for the fact that noise is complex but goes to real
    noise_variance_mod = noise_variance.copy()
    # noise_variance_mod[1:] = 2 * noise_variance_mod[1:] 
    noise_image = noise.make_radial_noise(noise_variance_mod, experiment_dataset.image_shape).reshape(experiment_dataset.image_shape)

    if mrc_file is None:
        output_array = np.empty([experiment_dataset.n_images, *experiment_dataset.image_shape], dtype = experiment_dataset.dtype_real )
    else:
        output_array = mrc_file.data

    n_images_done =0 
    for vol_idx in range(len(volumes)):
        img_indices = np.nonzero(image_assignments == vol_idx)[0]
        n_images = img_indices.size
        
        if disc_type == "nufft":
            vol_real = ftu.get_idft3(volumes[vol_idx].reshape(experiment_dataset.volume_shape))
        elif 'cubic' in disc_type:
            from recovar import cryojax_map_coordinates
            volume = cryojax_map_coordinates.compute_spline_coefficients(volumes[vol_idx].reshape(experiment_dataset.volume_shape))
        else:
            volume = volumes[vol_idx]

        for k in range(0, int(np.ceil(n_images/batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min( [(k+1) * batch_size, n_images]))
            indices = img_indices[batch_st:batch_end]

            translations = np.zeros_like(experiment_dataset.translations[indices]) if pad_before_translate else experiment_dataset.translations[indices]

            if disc_type == "nufft":
                images_batch = simulate_nufft_data_batch(vol_real, 
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 translations, 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
            elif disc_type == "pdb":
                images_batch = simulate_nufft_data_batch_from_pdb(volumes[vol_idx],
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 translations, 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun) / gt_vols_norm
            elif "ewald" in disc_type:
                # disc_type_e = disc_type.split("_")[1]
                disc_type_e = disc_type[6:]

                from recovar import ewald
                # lam = ewald.volt_to_wavelength(experiment_dataset.CTF_params[0,3])
                images_batch_real, images_batch_real_imag = ewald.ewald_sphere_forward_model(
                        volume.real, 
                        volume.imag, 
                        experiment_dataset.rotation_matrices[indices], 
                        experiment_dataset.CTF_params[indices],
                        experiment_dataset.image_shape,
                        experiment_dataset.volume_shape, 
                        experiment_dataset.voxel_size, disc_type_e )
                images_batch = images_batch_real + 1j * images_batch_real_imag

                images_batch = core.translate_images(images_batch,
                        -translations,
                        experiment_dataset.image_shape)
                
                
            elif disc_type == "linear_interp" or disc_type == "nearest" or disc_type == "cubic":
                images_batch = simulate_data_batch(volume,
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 translations,
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
            else:
                raise ValueError("Invalid disc_type")

            images_batch = ftu.get_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape]))
            # import pdb; pdb.set_trace()
            images_batch = images_batch.real
            # images_batch = ftu.get_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape])).real



            if pad_before_translate:
                plotting = True
                # for k in range(images_batch.shape[0]):
                #     if k > 3:
                #         break
                from recovar import padding
                padded_images = padding.pad_images_spatial_domain(images_batch,experiment_dataset.grid_size)
                if plotting:
                    import matplotlib.pyplot as plt
                    plt.imshow(padded_images[0])
                    plt.show()
                padded_images = roll_batch(padded_images, -np.round(experiment_dataset.translations[indices]).astype(int)[:,0], -1 )
                padded_images = roll_batch(padded_images, -np.round(experiment_dataset.translations[indices]).astype(int)[:,1], -2 )
                if plotting:
                    plt.imshow(padded_images[0])
                    plt.show()
                images_batch2 = padding.unpad_images_spatial_domain(padded_images, experiment_dataset.grid_size)

                if plotting:
                    plt.imshow(images_batch2[0]); plt.show()
                    plt.show()
                # import pdb; pdb.set_trace()
                images_batch = images_batch2


            key, subkey = jax.random.split(key)
            noise_batch = make_noise_batch(subkey, noise_image, images_batch.shape)
            noise_batch *= per_image_noise_scale[indices][...,None,None]
            images_batch *= per_image_contrast[indices][...,None,None]
            output_array[indices] = np.array(images_batch + noise_batch)


            n_images_done += indices.size
            # if n_images_done % 1000 == 0:
            logger.info(f"Batch {k}: Generated {n_images_done} images so far")

            # import pdb; pdb.set_trace()
    logger.info("Discretizing with: " + disc_type)
    logger.info("Done generating data")

    if mrc_file is not None:
        return mrc_file
    else:
        return output_array 


def make_noise_batch(subkey, noise_image, images_batch_shape):
    image_size = images_batch_shape[-1] * images_batch_shape[-2]
    # 

    noise_batch = jax.random.normal(subkey, images_batch_shape ) / jnp.sqrt(image_size)
    
    # import recovar.fourier_transform_utils
    # if recovar.fourier_transform_utils.DEFAULT_FFT_NORM == "backward":
    #     noise_batch = noise_batch /  jnp.sqrt(image_size)

    noise_batch_ft = ftu.get_dft2(noise_batch.reshape(images_batch_shape))
    noise_batch_ft *= jnp.sqrt(noise_image)
    noise_batch = ftu.get_idft2(noise_batch_ft.reshape(images_batch_shape)).real
    return noise_batch



@functools.partial(jax.jit, static_argnums = [4,5,6,7,8,9])    
def simulate_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = core.slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type) * CTF
    # import pdb; pdb.set_trace()
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    
    return translated_images


def simulate_nufft_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = get_nufft_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size) * CTF
    
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    return translated_images


# MOVED HERE BECAUSE ONLY USED HERE
@functools.partial(jax.jit, static_argnums=[1,2,3])
def get_rotated_plane_coords(rotation_matrix, image_shape, voxel_size, scaled = True):
    unrotated_plane_indices = core.get_unrotated_plane_coords(image_shape, voxel_size, scaled = scaled)
    rotated_plane = unrotated_plane_indices @ rotation_matrix
    return rotated_plane

batch_get_rotated_plane_coords = jax.vmap(get_rotated_plane_coords, in_axes = (0, None, None, None))


def simulate_nufft_data_batch_from_pdb(atom_group, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun, Bfactor=0 ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)    
    plane_coords_mol = batch_get_rotated_plane_coords(rotation_matrices, image_shape, voxel_size, True) 
    slices = compute_projections_with_nufft(atom_group, plane_coords_mol, voxel_size)

    corrected_images = slices * CTF
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    translated_images = Bfactorize_images(translated_images, Bfactor, plane_coords_mol)
    return translated_images


def compute_projections_with_nufft(atom_group, plane_coords, voxel_size):
    # plane_coords = cu.get_unrotated_plane_coords(image_shape, voxel_size, scaled =True )

    plane_coords_vec = np.array(plane_coords.reshape(-1, 3)).astype(np.float64)
    X_ims = np.array(gsm.generate_potential_at_freqs_from_atoms(atom_group, voxel_size, plane_coords_vec).astype(np.complex64))
    # print(np.max(np.abs(atom_group.getCoords())))
    if np.isnan(np.sum(X_ims)):
        import pdb; pdb.set_trace()
    X_ims = X_ims.reshape(plane_coords.shape[:-1])
    return X_ims


# This can only run on CPU.
def get_nufft_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size ):
    plane_coords_mol = batch_get_rotated_plane_coords(rotation_matrices, image_shape, voxel_size, True) 
    clean_image_mol = compute_volume_projections_with_nufft(volume, plane_coords_mol, voxel_size)
    return clean_image_mol

def compute_volume_projections_with_nufft(volume, plane_coords, voxel_size):
    # This is here because I don't want to impose the dependencies for nufft. If you want to run this, you should 
    # pip install finufft
    # plane_coords = cu.get_unrotated_plane_coords(image_shape, voxel_size, scaled =True )
    plane_coords_vec = np.array(plane_coords.reshape(-1, 3)).astype(np.float64)
    X_ims = gsm.get_fourier_transform_of_volume_at_freq_coords(np.array(volume).astype(np.complex128), plane_coords_vec, voxel_size )
    X_ims = X_ims.reshape(plane_coords.shape[:-1])

    return X_ims
    

def get_B_factor_scaling(volume_shape, voxel_size, B_factor):
    vol_idx = np.arange(np.prod(volume_shape))
    freqs = core.vec_indices_to_frequencies(vol_idx, volume_shape) / ( volume_shape[0] * voxel_size)
    freq_norms = np.linalg.norm(freqs, axis =-1)**2
    B_fac_scaling = np.exp(-B_factor*freq_norms/4) 
    return B_fac_scaling

def Bfactorize_vol(volume, voxel_size, Bfactor, volume_shape):
    B_fac_scaling = get_B_factor_scaling(volume_shape, voxel_size, Bfactor)
    return  volume * B_fac_scaling

def Bfactorize_images(images, Bfactor, plane_coords_mol):
    freq_norms = jnp.linalg.norm(plane_coords_mol, axis =-1)**2
    B_fac_scaling = jnp.exp(-Bfactor*freq_norms/4) 
    return  images * B_fac_scaling