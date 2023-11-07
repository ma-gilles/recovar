import jax.numpy as jnp
import numpy as np
import jax, functools
from recovar import core, dataset, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils
import jax.scipy.spatial.transform
import os
import mrcfile, pickle
import matplotlib.pyplot as plt
from cryodrgn import mrc, ctf
from cryodrgn.pose import PoseTracker
from pathlib import Path
xx = Path(__file__).resolve()

data_path = os.path.join(os.path.dirname(__file__),'data/')

# Two generators that load ctf and poses from real datasets
def get_dataset_params(n_images, grid_size, ctf_file, poses_file):
    ctf_params = np.array(ctf.load_ctf_for_training(grid_size, ctf_file))
    
    # Initialize bfactor == 0
    ctf_params = np.concatenate( [ctf_params, np.zeros_like(ctf_params[:,0][...,None])], axis =-1)
    
    # Initialize constrast == 1
    ctf_params = np.concatenate( [ctf_params, np.ones_like(ctf_params[:,0][...,None])], axis =-1)
    posetracker = PoseTracker.load(poses_file, n_images, grid_size, ind = None) 
    return ctf_params, np.array(posetracker.rots), np.array(posetracker.trans)

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
    return sample_ctf_params[...,1:], sample_rots, sample_trans

def get_params_generator(dataset_params_fn ):
    def params_generator(n_images, grid_size):
        return generate_simulated_params_from_real(n_images, dataset_params_fn, grid_size )
    return params_generator

## A uniform pose generator
def random_sampling_scheme(n_images, grid_size, seed =0 ):
    dataset_params_fn = load_first_dataset_params
    ctf_params, _, _ = generate_simulated_params_from_real(n_images, dataset_params_fn, grid_size  )
    rotations = uniform_rotation_sampling(n_images, grid_size, seed = seed )
    translations = np.zeros([n_images,2])
    return ctf_params, rotations, translations

def uniform_rotation_sampling(n_images, grid_size, seed = 0 ):
    from scipy.spatial.transform import Rotation
    # Rotation.random(type cls, num=None, random_state=None)
    # key = jax.random.PRNGKey(seed)
    # rotations = np.array(jax.scipy.spatial.transform.Rotation.random(key, n_images))
    rotations = Rotation.random(n_images).as_matrix()
    return rotations

## The two main generators
def get_pose_ctf_generator(option):
    if option == "uniform":
        return random_sampling_scheme
    elif option == "dataset1":
        return get_params_generator(load_first_dataset_params)
    else:
        return get_params_generator(load_second_dataset_params)


def generate_contrast_params(n_images,noise_scale_std, contrast_std ):
    contrast = 1 + np.random.randn(n_images) * contrast_std
    noise_scale = 1 + np.random.randn(n_images) * noise_scale_std
    return contrast, noise_scale


def generate_volumes_from_mrcs(mrc_names, grid_size_i = None, padding= 0 ):
    Xs_vec = []
    first_vol = True
    for mrc_name in mrc_names:
        data, header =  mrc.parse_mrc(mrc_name)
        data = np.transpose(data, (2,1,0))
        voxel_size = header.fields['xlen'] / header.fields['nx']
        mrc_grid_size = header.fields['nx']
        
        if grid_size_i is None:
            grid_size = mrc_grid_size
        else:
            grid_size = grid_size_i - padding

            
        if first_vol:
            first_vol = False
            first_voxel_size = voxel_size
        else:
            assert( first_voxel_size == voxel_size)

        # Zero out grid_sizes outside radius
        X = ftu.get_dft3(data)
        # Downsample
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


def generate_synthetic_dataset(output_folder, voxel_size,  volume_folder_input, outlier_file_input, n_images, grid_size = 128,
                               volume_distribution = None,  dataset_params_option = "dataset1", noise_level = 1, 
                               noise_model = "radial1", put_extra_particles = True, percent_outliers = 0.1, 
                               volume_radius = 0.9, volume_file_root = "", noise_scale_std = 0.3, contrast_std =0.3 ):
    
    volumes = load_volumes_from_folder(volume_folder_input, grid_size, volume_file_root )
    vol_shape = utils.guess_vol_shape_from_vol_size(volumes.shape[-1])
    # import matplotlib.pyplot as plt
    # plt.imshow(ftu.get_idft3(volumes[0].reshape(vol_shape)).real.sum(axis=0))
    # Maybe normalize volumes?
    # volumes /= np.linalg.norm(volumes, axis =(-1))
    volume_distribution = np.ones(volumes.shape[0]) / volumes.shape[0] if volume_distribution is None else volume_distribution
    outlier_volume = ftu.get_dft3(utils.load_mrc(outlier_file_input)).reshape(-1) if outlier_file_input is not None else None

    dataset_param_generator = get_pose_ctf_generator(dataset_params_option)
    noise_variance = get_noise_model(noise_model, grid_size) / 50000 * noise_level

    # import pdb; pdb.set_trace()
    main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size = generate_simulated_dataset(volumes, voxel_size, volume_distribution, n_images, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = volume_radius, outlier_volume = outlier_volume )
    simulation_info['volume_folder_input'] = volume_folder_input
    simulation_info['grid_size'] = grid_size

    #
    with mrcfile.new(output_folder + '/particles.'+str(grid_size)+'.mrcs',overwrite=True) as mrc:
        mrc.set_data(main_image_stack.astype(np.float32))
        mrc.voxel_size = voxel_size

    poses = (rots, trans)
    utils.pickle_dump(poses, output_folder + '/poses.pkl')
    save_ctf_params(output_folder, grid_size, ctf_params, voxel_size)
    utils.pickle_dump(simulation_info, output_folder + '/simulation_info.pkl' )
    return main_image_stack, simulation_info

def load_volumes_from_folder(input_folder, grid_size, file_root = ""):
    def make_file(k):
        return os.path.join(input_folder,  f"{file_root}{k}.mrc")
    idx =0 
    files = []
    while(os.path.isfile(make_file(idx))):
        files.append(make_file(idx))
        # import pdb; pdb.set_trace()
        idx+=1
    volumes, voxel_size = generate_volumes_from_mrcs(files, grid_size, padding= 0 )
    volumes /= np.mean(np.linalg.norm(volumes, axis =(-1)))
    return volumes


def generate_simulated_dataset(volumes, voxel_size, volume_distribution, n_images, noise_variance, noise_scale_std, contrast_std, put_extra_particles, percent_outliers, dataset_param_generator, volume_radius = 0.95, outlier_volume = None ):
    
    # voxel_size = 
    volume_shape = utils.guess_vol_shape_from_vol_size(volumes[0].size)
    grid_size = volume_shape[0]

    ctf_params, rots, trans = dataset_param_generator(n_images, grid_size)

    # ctf_params[:,:2] *=0
    # ctf_params[:,4] *=0 
    # voxel_size = ctf_params[0,0]
    trans *=0 
    per_image_contrast, per_image_noise_scale = generate_contrast_params(n_images, noise_scale_std, contrast_std )

    image_assignments = np.random.choice(np.arange(volumes.shape[0]), size = n_images,  p = volume_distribution)

    CTF_fun = core.compute_ctf_cryodgrn_wrapper
    main_dataset = dataset.CryoEMDataset( volume_shape, None, voxel_size,
                              rots, trans, ctf_params, CTF_fun = CTF_fun, dataset_indices = None)
    batch_size = 5 * utils.get_image_batch_size(grid_size, utils.get_gpu_memory_total())

    # plt.imshow(main_dataset.get_CTF_image(0)); plt.colorbar()
    # plt.show();
    # import pdb; pdb.set_trace()
    # simulate_data(experiment_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None )

    main_image_stack = simulate_data(main_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None )

    if put_extra_particles:
        # Make other particles with same ctf but different rots
        _, rots_2, trans_2 = dataset_param_generator(n_images, grid_size)
        _, per_image_noise_scale_2 = generate_contrast_params(n_images, noise_scale_std, contrast_std )

        # To first approximation, maybe it is good enough to move them radially 
        # random number on the circle
        trans_2 = np.random.randn(n_images, 2)
        trans_2/= np.linalg.norm(trans_2, axis =-1,keepdims=True)
    
        # Move the center by around twice the radius
        trans_2 *= 2 * volume_radius * volume_shape[0]//2
        other_particles_dataset = dataset.CryoEMDataset( volume_shape, None, voxel_size,
                                rots_2, trans_2, ctf_params[:,1:], CTF_fun = CTF_fun, dataset_indices = None)

        # No noise in this stack.
        extra_particles_image_stack = simulate_data(other_particles_dataset, volumes,  noise_variance * 0 ,  batch_size, image_assignments, per_image_noise_scale_2, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None, pad_before_translate= True )

        main_image_stack += extra_particles_image_stack

    if percent_outliers > 0:
        # Perhaps a reasonable way to throw in outliers is to put a different structure entirely with wrong angles

        assert outlier_volume is not None, "if you want outliers, need to provide a structure"
        # Make sure they are on the same scale

        outlier_volume  = outlier_volume / np.linalg.norm(outlier_volume) * np.mean(np.linalg.norm(volumes,axis=(-1)))
        # import pdb; pdb.set_trace()
        n_outlier_images = np.round(percent_outliers * n_images).astype(int)
        ctf_params_3, rots_3, trans_3 = dataset_param_generator(n_outlier_images, grid_size)
        outlier_contrast, outlier_noise_scale = generate_contrast_params(n_outlier_images, noise_scale_std, contrast_std )

        outlier_particle_dataset = dataset.CryoEMDataset( volume_shape, None, voxel_size,
                                rots_3, trans_3, ctf_params_3[:,1:], CTF_fun = CTF_fun, dataset_indices = None)
        
        outlier_particle_image_stack = simulate_data(outlier_particle_dataset, outlier_volume[None],  noise_variance ,  batch_size, np.zeros(n_outlier_images, dtype = int), outlier_noise_scale, outlier_contrast, seed =1, disc_type = 'linear_interp', mrc_file = None )

        ind_outliers = np.random.choice(n_images, n_outlier_images, replace =False)
        main_image_stack[ind_outliers] = outlier_particle_image_stack
        image_assignments[ind_outliers] = -1

    simulation_info = { 
        "ctf_params" : ctf_params,
        "rots" : rots,
        "trans" : trans,
        "per_image_contrast" : per_image_contrast,
        "per_image_noise_scale" : per_image_noise_scale,
        "image_assignment" : image_assignments,
        "noise_variance": noise_variance,
        "voxel_size": voxel_size,
    }

    return main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size


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
    utils.pickle_dump(ctf_params_all[:,:9], outdir + '/ctf.pkl')
    return 


# Solves the linear system Dx = b.
def simulate_data(experiment_dataset, volumes,  noise_variance,  batch_size, image_assignments, per_image_contrast, per_image_noise_scale, seed =0, disc_type = 'linear_interp', mrc_file = None, pad_before_translate = False ):
    
    key = jax.random.PRNGKey(seed)
    # A little bit of a hack to account for the fact that noise is complex but goes to real
    noise_variance_mod = noise_variance.copy()
    # noise_variance_mod[1:] = 2 * noise_variance_mod[1:] 
    noise_image = noise.make_radial_noise(noise_variance_mod, experiment_dataset.image_shape).reshape(experiment_dataset.image_shape)

    if mrc_file is None:
        output_array = np.empty([experiment_dataset.n_images, *experiment_dataset.image_shape] )
    else:
        output_array = mrc_file.data

    for vol_idx in range(len(volumes)):
        img_indices = np.nonzero(image_assignments == vol_idx)[0]
        n_images = img_indices.size
        
        if disc_type == "nufft":
            vol_real = ftu.get_idft3(volumes[vol_idx].reshape(experiment_dataset.volume_shape))

        for k in range(0, int(np.ceil(n_images/batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min( [(k+1) * batch_size, n_images]))
            indices = img_indices[batch_st:batch_end]

            translations = np.zeros_like(experiment_dataset.translations[indices]) if pad_before_translate else experiment_dataset.translations[indices]

            if disc_type == "nufft":
                images_batch = simulate_nufft_data_batch(vol_real, 
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 experiment_dataset.translations[indices], 
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
                                                 experiment_dataset.translations[indices], 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
                
            else:
                images_batch = simulate_data_batch(volumes[vol_idx],
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 translations,
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)




            # import pdb; pdb.set_trace()
            images_batch = ftu.get_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape])).real
            plotting = False
            
            if pad_before_translate:
                from recovar import padding
                padded_images = padding.pad_images_spatial_domain(images_batch,experiment_dataset.grid_size)
                if plotting:
                    import matplotlib.pyplot as plt
                    plt.imshow(padded_images[0])
                    plt.show()

                padded_images = jax.numpy.roll(padded_images, -np.round(experiment_dataset.translations[indices]).astype(int)[:,0], axis =-1 )
                padded_images = jax.numpy.roll(padded_images, -np.round(experiment_dataset.translations[indices]).astype(int)[:,1], axis =-2 )
                if plotting:
                    plt.imshow(padded_images[0])
                    plt.show()

                images_batch = padding.unpad_images_spatial_domain(padded_images, experiment_dataset.grid_size)
                if plotting:
                    plt.imshow(images_batch[0])
                    plt.show()


            key, subkey = jax.random.split(key)
            # noise_batch[0]
            # noise_batch = jax.random.normal(subkey, images_batch.shape ) / jnp.sqrt(experiment_dataset.image_size)
            # noise_batch_ft = ftu.get_dft2(noise_batch.reshape([-1, *experiment_dataset.image_shape]))
            # noise_batch_ft *= jnp.sqrt(noise_image)
            # noise_batch = ftu.get_idft2(noise_batch_ft.reshape([-1, *experiment_dataset.image_shape])).real

            # noise_batch *= jnp.sqrt(noise_image) 
            # import matplotlib.pyplot as plt
            # plt.imshow(images_batch[0])
            # plt.show()
            # For noise to be real, we need the symmetry to hold.
            # Or just be sloppy and take out imag part? Is this right?
            # noise_batch = ftu.get_idft2(noise_batch.reshape([-1, *experiment_dataset.image_shape])).real
            noise_batch = make_noise_batch(subkey, noise_image, images_batch.shape)
            noise_batch *= per_image_noise_scale[indices][...,None,None]
            images_batch *= per_image_contrast[indices][...,None,None]
            output_array[indices] = np.array(images_batch + noise_batch)


            # import pdb; pdb.set_trace()



    if mrc_file is not None:
        return mrc_file
    else:
        return output_array 


def make_noise_batch(subkey, noise_image, images_batch_shape):
    image_size = images_batch_shape[-1] * images_batch_shape[-2]
    noise_batch = jax.random.normal(subkey, images_batch_shape ) / jnp.sqrt(image_size)
    noise_batch_ft = ftu.get_dft2(noise_batch.reshape(images_batch_shape))
    noise_batch_ft *= jnp.sqrt(noise_image)
    noise_batch = ftu.get_idft2(noise_batch_ft.reshape(images_batch_shape)).real
    return noise_batch



@functools.partial(jax.jit, static_argnums = [4,5,6,7,8,9])    
def simulate_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = core.get_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF
    # import pdb; pdb.set_trace()
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    
    return translated_images


def simulate_nufft_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = core.get_nufft_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size) * CTF
    
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    return translated_images



def simulate_nufft_data_batch_from_pdb(atom_group, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)    
    plane_coords_mol =  core.batch_get_rotated_plane_coords(rotation_matrices, image_shape, voxel_size, True) 
    slices = core.compute_projections_with_nufft(atom_group, plane_coords_mol, voxel_size)

    corrected_images = slices * CTF
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    return translated_images



