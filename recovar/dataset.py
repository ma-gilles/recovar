import logging
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from recovar import plot_utils, core, mask
import recovar.padding as pad
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)
from recovar import tilt_dataset

logger = logging.getLogger(__name__)

# Maybe should take out these dependencies?
from recovar.cryodrgn_source import ImageSource


def MRCDataMod(particles_file, ind =None , datadir = None, padding = 0, uninvert_data = False, strip_prefix = None):
    return tilt_dataset.ImageDataset(particles_file, ind = ind, datadir = datadir, padding = padding, invert_data = uninvert_data, lazy =False, strip_prefix=strip_prefix)


def LazyMRCDataMod(particles_file, ind =None , datadir = None, padding = 0, uninvert_data = False, strip_prefix = None):
    return tilt_dataset.ImageDataset(particles_file, ind = ind, datadir = datadir, padding = padding, invert_data = uninvert_data, lazy =True, strip_prefix=strip_prefix)
    
    
def get_num_images_in_dataset(mrc_path):
    return ImageSource.from_file(mrc_path, lazy=True).n

def set_standard_mask(D, dtype):
    return mask.window_mask(D, 0.85, 0.99)
    # return np.ones([D,D], dtype = dtype).real
    
# Image loader functions - supposed to give quick access to images to GPU

# I don't remember why I use two different loaders here.

# # This might work. 
# def numpy_collate(batch):
#   if isinstance(batch[0], np.ndarray):
#     return jnp.stack(batch)
#   elif isinstance(batch[0], (tuple,list)):
#     transposed = zip(*batch)
#     return [numpy_collate(samples) for samples in transposed]
#   else:
#     return jnp.array(batch)

# class NumpyLoader(torch.utils.data.DataLoader):
#   def __init__(self, dataset, batch_size=1,
#                 shuffle=False, sampler=None,
#                 batch_sampler=None, num_workers=0,
#                 pin_memory=False, drop_last=False,
#                 timeout=0, worker_init_fn=None):
#     super(self.__class__, self).__init__(dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         sampler=sampler,
#         batch_sampler=batch_sampler,
#         num_workers=num_workers,
#         collate_fn=numpy_collate,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#         timeout=timeout,
#         worker_init_fn=worker_init_fn)
    

# A dataset class, that includes images and all other information
class CryoEMDataset:

    def __init__(self, image_stack, voxel_size, rotation_matrices, translations, CTF_params, CTF_fun = core.evaluate_ctf_wrapper, dtype = np.complex64, rotation_dtype = np.float32, dataset_indices = None, grid_size = None, volume_upsampling_factor = 1, tilt_series_flag = False, premultiplied_ctf = False  ):
        
        if image_stack is not None:
            grid_size = image_stack.D
        elif grid_size is None:
            raise ValueError("Must specify grid_size if image_stack is None")
        

        self.voxel_size = voxel_size
        self.grid_size = grid_size

        self.volume_upsampling_factor = volume_upsampling_factor
        self.upsampled_grid_size = self.grid_size * volume_upsampling_factor
        self.grid_size = grid_size

        self.volume_shape = tuple(3*[self.grid_size ])
        self.volume_size = np.prod(self.volume_shape)

        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

        # self.original_volume_shape = tuple(3*[image_stack.unpadded_D])
        # self.volume_shape = tuple(3*[image_stack.unpadded_D])
        # Allows for passing None as image_stack (for simulation)
        if image_stack is None:
            self.image_stack = None
            self.image_shape = tuple(2*[self.grid_size])
            self.image_size = np.prod(self.image_shape)
            self.n_images = CTF_params.shape[0]
            self.padding = 0 
        else:
            self.image_stack = image_stack
            self.image_shape = tuple(image_stack.image_shape)
            self.image_size = np.prod(image_stack.image_shape)
            self.n_images = image_stack.n_images
            self.padding = image_stack.padding

        
        # self.n_units = self.n_images # This is the number of predictions.
        self.tilt_series_flag = tilt_series_flag # Hopefully can just switch this on and off
        self.premultiplied_ctf = premultiplied_ctf

        # For SPA, it is # of images, for ET, it is # of tilt series
        # For tilt series: A "tilt" is an image. A particle is a full tilt series 
        self.n_units = self.image_stack.Np if self.tilt_series_flag else self.n_images

        # For SPA, it is # of images, for ET, it is # of tilt series 
        # self.CTF_FUNCTION_OPTION = "cryodrgn"
        self.CTF_fun_inp = CTF_fun
        self.hpad = self.padding//2
        self.volume_mask_threshold = 4 * self.grid_size / 128 # At around 128 resolution, 4 seems good, so scale up accordingly. This probably should have a less heuristic value here. This is assuming the mask is scaled between [0,1]

        self.dtype = dtype
        self.dtype_real = dtype(0).real.dtype
        self.CTF_dtype = self.dtype_real # this might changed in the future for Ewald sphere
        # Note that images are stored in float 32 but rotations are stored in float 64.
        # There seems to be a JAX-bug with float 32 when doing the nearest neighbor approximation...
        self.rotation_dtype = rotation_dtype
        self.rotation_matrices = np.array(rotation_matrices.astype(rotation_dtype))
        self.translations = np.array(translations)

        '''
            0 - dfu (float or Bx1 tensor): DefocusU (Angstrom)
            1 - dfv (float or Bx1 tensor): DefocusV (Angstrom)
            2 - dfang (float or Bx1 tensor): DefocusAngle (degrees)
            3 - volt (float or Bx1 tensor): accelerating voltage (kV)
            4 - cs (float or Bx1 tensor): spherical aberration (mm)
            5 - w (float or Bx1 tensor): amplitude contrast ratio
            6 - phase_shift (float or Bx1 tensor): degrees 
            7 - bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
            8 - per-particle scale

            For tilt series only:
            9 - tilt number
        '''
        self.CTF_params = np.array(CTF_params.astype(self.CTF_dtype))

        self.dataset_indices = dataset_indices
        self.noise = None

    def get_noise_variance(self, indices):
        if self.noise_model is None:            
            return None
        return self.noise_model.get(indices)

    def delete(self):
        del self.image_stack.particles
        del self.image_stack
        del self.rotation_matrices
        del self.CTF_params
        del self.translations
        del self.noise_model

    def update_volume_upsampling_factor(self, volume_upsampling_factor):

        self.volume_upsampling_factor = volume_upsampling_factor
        self.upsampled_grid_size = self.grid_size * volume_upsampling_factor
        
        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

        return

    # def get_tilt_dataset_generator(self, batch_size, num_workers = 0):
    #     self.current_index=0# A very stupid way to do this for now?
    #     return self.image_stack.get_dataset_generator(batch_size,num_workers = num_workers)

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        return self.image_stack.get_dataset_generator(batch_size, num_workers=num_workers, **kwargs)
    
    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        if subset_indices is None:
            return self.get_dataset_generator(batch_size, num_workers=num_workers, **kwargs)
        return self.image_stack.get_dataset_subset_generator(batch_size, subset_indices, num_workers=num_workers, **kwargs)

    # This is a generator that iterates over individual images rather than tilt groups. For SPA, this is the same as get_dataset_subset_generator.
    def get_image_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        if self.tilt_series_flag:
            return self.image_stack.get_image_subset_generator(batch_size, subset_indices, num_workers = num_workers)
        else:
            return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers = num_workers)

    # This is a generator that iterates over individual images rather than tilt groups. For SPA, this is the same as get_dataset_generator.
    def get_image_generator(self, batch_size, num_workers = 0):
        if self.tilt_series_flag:
            return self.image_stack.get_image_generator(batch_size, num_workers = num_workers)
        else:
            return self.get_dataset_generator(batch_size, num_workers = num_workers)


    def CTF_fun(self,*args):
        # Force dtype
        return self.CTF_fun_inp(*args).astype(self.CTF_dtype)

    def get_valid_frequency_indices(self,rad = None):
        rad = self.grid_size//2 -1 if rad is None else rad
        return np.array(self.get_volume_radial_mask(rad))

    def get_valid_upsampled_frequency_indices(self,rad = None):
        rad = self.upsampled_grid_size//2 -1 if rad is None else rad
        return np.array(self.get_upsampled_volume_radial_mask(rad))


    #### All functions below are only just for plotting/debugging

    def compute_CTF(self, CTF_params):
        return self.CTF_fun(CTF_params, self.image_shape, self.voxel_size)

    def get_CTF(self, indices):
        return self.compute_CTF(self.CTF_params[indices])

    def get_volume_radial_mask(self, radius = None):
        return mask.get_radial_mask(self.volume_shape, radius = radius).reshape(-1)

    def get_upsampled_volume_radial_mask(self, radius = None):
        return mask.get_radial_mask(self.upsampled_volume_shape, radius = radius).reshape(-1)


    def get_image_radial_mask(self, radius = None):        
        return mask.get_radial_mask(self.image_shape, radius = radius).reshape(-1)

    def get_proj(self, X, to_real = np.real, axis = 0, hide_padding = True):
        im = to_real(ftu.get_idft2(jnp.take(X.reshape(self.volume_shape), self.grid_size//2, axis = axis)))
        if hide_padding:
            im = im[self.hpad:self.image_stack.unpadded_D + self.hpad,self.hpad:self.image_stack.unpadded_D + self.hpad]
        return im


    def get_slice(self, X, to_real_fn = np.abs, axis = 0):
        # zero_th freq
        z_freq = self.grid_size//2 +1
        im = to_real_fn(jnp.take(X.reshape(self.volume_shape), z_freq, axis = axis))
        return im

    def get_slice_real(self, X, to_real_fn = np.real, axis = 0):
        im = to_real_fn(ftu.get_idft3(X.reshape(self.volume_shape)))
        im2 = jnp.take(im, self.grid_size//2, axis = axis)
        return to_real_fn(im2)

    def get_image(self, i , tilt_idx = None):
        if self.tilt_series_flag:
            assert ( tilt_idx is not None), "Tilt index must be specified for tilt series"

        if tilt_idx is None:
            image = self.image_stack.__getitem__(i)[0]#[None]
        else:
            image = self.image_stack.__getitem__(i)[0][tilt_idx][None]

        processed_image = self.image_stack.process_images(image)
        return processed_image.reshape(self.image_shape)

    def get_CTF_image(self, i ):
        return self.get_CTF(np.array([i])).reshape(self.image_shape)

    def get_image_real(self,i, tilt_idx = None, to_real= np.real, hide_padding = True):
        hpad= self.image_stack.padding//2
        if hide_padding:
            return to_real(ftu.get_idft2(self.get_image(i,tilt_idx))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        else:
            return to_real(ftu.get_idft2(self.get_image(i,tilt_idx)))


    def get_denoised_image(self,i, tilt_idx=None, to_real= np.real, hide_padding = True, weiner_param =1):
        batch_image_ind = np.array([i])
        if self.tilt_series_flag:
            assert ( tilt_idx is not None), "Tilt index must be specified for tilt series"
        # tilt_idx = None if tilt_idx is None else np.array([tilt_idx])

        if tilt_idx is not None:
            images, _, image_ind = self.image_stack.__getitem__(i)
            images = images[tilt_idx][None]
            CTFs = self.CTF_fun(self.CTF_params[image_ind[tilt_idx]][None], self.image_shape, self.voxel_size) # Compute CTF
        else:
            images, _, _ = self.image_stack.__getitem__(i)
            images = images#[None]
            CTFs = self.CTF_fun(self.CTF_params[i][None], self.image_shape, self.voxel_size) # Compute CTF
        images = self.image_stack.process_images(images) # Compute DFT, masking
        images = (CTFs / (CTFs**2 + weiner_param)) * images  # CTF correction
        images = images.reshape(self.image_shape)
        # if hide_padding:
        #     return to_real(ftu.get_idft2(self.get_image(i))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        # else:
        return to_real(ftu.get_idft2(images))


    def plot_FSC(self, image1 = None, image2 = None, filename = None, threshold = 0.5, curve = None, ax = None):
        score = plot_utils.plot_fsc_new(image1, image2, self.volume_shape, self.voxel_size,  curve = curve, ax = ax, threshold = threshold, filename = filename)
        return score
    
    def get_image_mask(self, indices, mask, binary = True, soften = 5):
        indices = np.asarray(indices).astype(int)
        from recovar import covariance_core # Not sure I want this depency to exist... Could make some circular imports
        mask = covariance_core.get_per_image_tight_mask(mask, self.rotation_matrices[indices], self.image_stack.mask, self.volume_mask_threshold, self.image_shape, self.volume_shape, self.grid_size, self.padding, disc_type = 'linear_interp',  binary = binary, soften = soften)
        mask_ft = ftu.get_dft2(mask).reshape(mask.shape[0], -1)
        # Usually images are translated, here we translate back.
        batch = core.translate_images(mask_ft, -self.translations[indices].astype(int) , self.image_shape)
        mask2 = ftu.get_idft2(batch.reshape(-1, *self.image_shape))
        return mask2.real


    def get_predicted_image(self, indices, volume, skip_ctf = False, spatial = True):
        """Get predicted images for given indices using forward model.
        
        Args:
            indices: Array of indices to predict images for
            volume: Volume to use for prediction
            skip_ctf: Whether to skip CTF application
            spatial: Whether to return images in real space (True) or Fourier space (False)
            
        Returns:
            Predicted images in real space if spatial=True, otherwise in Fourier space
        """
        predicted_images = core.forward_model_from_map(
            volume,
            self.CTF_params[indices],
            self.rotation_matrices[indices],
            self.image_shape,
            self.volume_shape,
            self.voxel_size,
            self.CTF_fun,
            'linear_interp',  # Using linear interpolation for better quality
            skip_ctf = skip_ctf
        )
        if spatial:
            predicted_images = ftu.get_idft2(predicted_images.reshape(-1, *self.image_shape)).real#.reshape(predicted_images.shape[0], -1)
        return predicted_images


    def set_radial_noise_model(self, noise_variance):
        from recovar import noise
        self.noise = noise.RadialNoiseModel(noise_variance, image_shape = self.image_shape)

    def set_variable_radial_noise_model(self, noise_variance_radials):
        from recovar import noise
        _, dose_indices = jnp.unique(self.CTF_params[:,core.dose_ind], return_inverse=True)
        self.noise = noise.VariableRadialNoiseModel(noise_variance_radials, dose_indices, image_shape = self.image_shape)


# TODO: This is not used anywhere. Delete?
# def subsample_cryoem_dataset(dataset, indices):

#     import copy
#     image_stack = copy.copy(dataset.image_stack)

#     if type(image_stack.particles) is list:
#         image_stack.particles = [dataset.image_stack.particles[i] for i in indices]
#         image_stack.n_images = len(image_stack.particles)#.shape[0]
#     else:
#         image_stack.particles = dataset.image_stack.particles[indices]
#         image_stack.n_images = image_stack.particles.shape[0]

#     return CryoEMDataset( image_stack, dataset.voxel_size, dataset.rotation_matrices[indices], dataset.translations[indices], dataset.CTF_params[indices], CTF_fun = dataset.CTF_fun_inp, dtype = dataset.dtype, rotation_dtype = dataset.rotation_dtype, dataset_indices = dataset.dataset_indices[indices] , volume_upsampling_factor= dataset.volume_upsampling_factor)



# Loads dataset that are stored in the cryoDRGN format
def load_cryodrgn_dataset(particles_file, poses_file, ctf_file, datadir = None, n_images = None, ind = None, lazy = True, padding = 0, uninvert_data = False, tilt_series = False, tilt_series_ctf = None, dose_per_tilt = 2.9, angle_per_tilt = 3, premultiplied_ctf = False, strip_prefix = None):
    
    # For backward compatibility... Delete at some point?
    if tilt_series_ctf is None and tilt_series is False:
        tilt_series_ctf = 'cryoem'
    elif tilt_series_ctf is None and tilt_series is True:
        tilt_series_ctf = 'relion5'
    elif tilt_series_ctf == 'warp':
        tilt_series_ctf = 'v2_scale_from_star'

        
    if tilt_series:
            from recovar import tilt_dataset
            # particles_to_tilts, tilts_to_particles = tilt_dataset.TiltSeriesData.parse_particle_tilt(particles_file)
            tilt_file_option = 'relion5' if tilt_series_ctf == 'relion5' else 'warp'
            dataset = tilt_dataset.TiltSeriesData(particles_file, ind = ind, datadir = datadir, invert_data = uninvert_data, tilt_file_option=tilt_file_option, strip_prefix=strip_prefix)
            # # Use TF-based implementation for tilt series
            # from recovar import tf_tilt_dataset
            # tilt_file_option = 'relion5' if tilt_series_ctf == 'relion5' else 'warp'
            # dataset = tf_tilt_dataset.TFTiltSeriesData(
            #     particles_file, 
            #     ind=ind, 
            #     datadir=datadir, 
            #     invert_data=uninvert_data, 
            #     tilt_file_option=tilt_file_option,
            #     cache_size=1000,  # Adjust based on available memory
            #     prefetch_size=100  # Adjust based on batch size
            # )

    else:
        if lazy:
            dataset = LazyMRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data, strip_prefix=strip_prefix)
        else:
            dataset = MRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data, strip_prefix=strip_prefix)


    from recovar import cryodrgn_load
    ctf_params = np.array(cryodrgn_load.load_ctf_for_training(dataset.D, ctf_file))
        
    ctf_params = ctf_params if ind is None else ctf_params[ind]
    
    # Initialize bfactor == 0
    ctf_params = np.concatenate( [ctf_params, np.zeros_like(ctf_params[:,0][...,None])], axis =-1)
    
    # Initialize constrast == 1
    ctf_params = np.concatenate( [ctf_params, np.ones_like(ctf_params[:,0][...,None])], axis =-1)
    
    CTF_fun = core.evaluate_ctf_wrapper

    # This is an option used to treat a cryo-ET dataset as a cryo-EM dataset, but still use the right CTF.
    # It means, that it will use the cryo-EM pipeline but the cryoET CTF.
    if (tilt_series is False) and (tilt_series_ctf != 'cryoem'):
        from recovar import tilt_dataset
        tilt_dataset_this = tilt_dataset.TiltSeriesData(particles_file, ind = ind, datadir = datadir, invert_data = uninvert_data, sort_with_Bfac = sort_with_Bfac)
    else:
        tilt_dataset_this = dataset

    if tilt_series_ctf != 'cryoem':

        if tilt_series_ctf == 'relion5':
            ctf_params[:,core.contrast_ind+1] = tilt_dataset_this.ctfscalefactor
            dose = tilt_dataset_this.dose
            angles = np.zeros_like(dose) # Set angles to 0 - the np.cos factor is included already?
            ctf_params = np.concatenate( [ctf_params, dose[...,None], angles[...,None]], axis =-1)
            CTF_fun = core.evaluate_ctf_wrapper_tilt_series_v2


        # The angles are used to compute a scale factor cos(angles). If scale from star, then the scale factor is already in the star file, so set angle to 0
        if "scale_from_star" in tilt_series_ctf:
            angle_per_tilt = 0

        if tilt_series_ctf == "from_star":
            ctf_params[:,core.contrast_ind+1] = tilt_dataset_this.ctfscalefactor
            ctf_params[:,core.bfactor_ind+1] = -tilt_dataset_this.ctfBfactor # should be POSITIVE (negative in star file)
            logger.info('CTF from star')

        elif (tilt_series_ctf == "scale_from_star") or (tilt_series_ctf == "from_dose"):
            # Sort of a hacky way to do this.

            # + 1 because voxel_size in included.... gross
            if "scale_from_star" in tilt_series_ctf:
                ctf_params[:,core.contrast_ind+1] = tilt_dataset_this.ctfscalefactor

            tilt_numbers = tilt_dataset_this.tilt_numbers
            # tilt_angles = dataset.tilt_angles[dataset.tilt_indices]
            # tilt_angles = angle_per_tilt * torch.ceil(tilt_numbers / 2)
            ctf_params = np.concatenate( [ctf_params, tilt_numbers[...,None]], axis =-1)#, tilt_angles[...,None]], axis =-1)

            assert (np.isclose(ctf_params[0,4], 200) or np.isclose(ctf_params[0,4], 300)) , "Critical exposure calculation requires 200kV or 300kV imaging" 
            # angle_per_tilt = 3 
            # dose_per_tilt = 2.9
            CTF_fun = core.get_cryo_ET_CTF_fun(dose_per_tilt = dose_per_tilt, angle_per_tilt = angle_per_tilt)
            logger.info('CTF from dose weighting')
        elif "v2" in tilt_series_ctf:# == "tilt_ctf_v2": 
            # Sort of a hacky way to do this.
            tilt_numbers = tilt_dataset_this.tilt_numbers
            dose = - (tilt_dataset_this.ctfBfactor / 4) # WARP uses a ctfBfactor == -4 * dose

            # The angles are used to compute a scale factor cos(angles). If scale from star, then the scale factor is already in the star file
            angles = jnp.ceil(tilt_numbers/2) * angle_per_tilt 
            if 'scale_from_star' in tilt_series_ctf:
                # + 1 because voxel_size in included.... gross
                ctf_params[:,core.contrast_ind+1] = tilt_dataset_this.ctfscalefactor
                # angles *=0 
                logger.warning("Using scale from star")

            if dose_per_tilt is None:
                dose = - (tilt_dataset_this.ctfBfactor / 4) # WARP uses a ctfBfactor == -4 * dose
                logger.warning("Using dose from star file (- Bfactor/4)")


            CTF_fun = core.evaluate_ctf_wrapper_tilt_series_v2
            ctf_params = np.concatenate( [ctf_params, dose[...,None], angles[...,None]], axis =-1)
            assert (np.isclose(ctf_params[0,4], 200) or np.isclose(ctf_params[0,4], 300)) , "Critical exposure calculation requires 200kV or 300kV imaging" 
            logger.info('CTF from dose weighting - V2')
            

    # posetracker = PoseTracker.load( poses_file, dataset.n_images, dataset.unpadded_D, ind = ind) #,   None, ind, device=device)
    # from recovar import cryodrgn_load
    rots, trans, _ = cryodrgn_load.load_poses( poses_file, dataset.n_images, dataset.unpadded_D, ind = ind) #,   None, ind, device=device)


    voxel_sizes = ctf_params[:,0]
    assert np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0))
    voxel_size = np.float32(voxel_sizes[0])

    # Make sure everything is in correct dtype:
    ctf_params = ctf_params.astype(np.float32)
    translations = np.array(trans).astype(np.float32)
    rots = np.array(rots).astype(np.float32)

    if ind is not None:
        ind = ind.astype(int)

    return CryoEMDataset( dataset, voxel_size,
                              rots, translations, ctf_params[:,1:], CTF_fun = CTF_fun, dataset_indices = ind, tilt_series_flag = tilt_series, premultiplied_ctf = premultiplied_ctf)




def get_split_datasets_from_dict(dataset_loader_dict, ind_split, lazy = False):
    return get_split_datasets(**dataset_loader_dict, ind_split=ind_split, lazy =lazy)

def get_split_datasets(particles_file, poses_file, ctf_file, datadir,
                                  uninvert_data = False, ind_file = None,
                                  padding = 0, n_images = None, tilt_series = False,
                                 tilt_series_ctf = None,
                                    angle_per_tilt = 3, dose_per_tilt = 2.9,
                                   ind_split = None, lazy = False, premultiplied_ctf = False, strip_prefix = None):
    
    cryos = []
    for ind in ind_split:
        cryos.append(load_cryodrgn_dataset(particles_file, poses_file, ctf_file , datadir = datadir, n_images = n_images, ind = ind, lazy = lazy, padding = padding, uninvert_data = uninvert_data, tilt_series = tilt_series, tilt_series_ctf = tilt_series_ctf, angle_per_tilt = angle_per_tilt, dose_per_tilt = dose_per_tilt, premultiplied_ctf = premultiplied_ctf, strip_prefix = strip_prefix))
    
    return cryos


def get_split_indices(particles_file, ind_file = None):

    if ind_file is None:
        n_images = get_num_images_in_dataset(particles_file)
        indices = np.arange(n_images)
    else:
        if isinstance(ind_file, np.ndarray):
            indices = ind_file
        else:
            # Get indf
            with open( ind_file,'rb') as f:
                indices = np.asarray(pickle.load(f))
  
    split_indices = split_index_list(indices)
    return split_indices


def get_split_tilt_indices(particles_file, ind_file = None, tilt_ind_file =None, ntilts = None, datadir = None):
    # from cryodrgn import starfile
    from recovar import tilt_dataset

    # dataset = tilt_dataset.parse_particle_tilt(args.particles)
    particles_to_tilts, tilts_to_particles = tilt_dataset.TiltSeriesData.parse_particle_tilt(particles_file)

    if ntilts is not None:
        # A lot of extra compute.
        # TODO rewrite
        dataset_tmp = tilt_dataset.TiltSeriesData(particles_file, datadir = datadir)
        tilt_numbers = dataset_tmp.tilt_numbers

    n_tilt_series = len(particles_to_tilts)
    if tilt_ind_file is None:
        particle_ind = np.arange(n_tilt_series)
    else:
        particle_ind = pickle.load(open(tilt_ind_file, "rb"))
        logger.warning("Using tilt-ind file to pick PARTICLES (i.e. tilt series), not images (individual tilts). Use --ind to pick images.")
        # raise NotImplementedError

    if ind_file is not None:
        ind_images = pickle.load(open(ind_file, "rb"))

    split_tilt_series_indices = split_index_list(particle_ind)
    split_image_indices = [None,None]
    for i in range(2):
        split_image_indices[i] = np.concatenate([ particles_to_tilts[ind] for ind in split_tilt_series_indices[i]])
        if ntilts is not None:
            good_indices = np.where(tilt_numbers[split_image_indices[i]] < ntilts)[0]
            split_image_indices[i] = split_image_indices[i][good_indices]
        # intersection = set(split_image_indices[i]) & set(ind_images)
        # tmp=np.intersect1d(split_image_indices[i], ind_images)
        if ind_file is not None:
            split_image_indices[i] = np.intersect1d(split_image_indices[i], ind_images)

    return split_image_indices



def split_index_list( all_valid_image_indices, split_random_seed = 0 ):
    np.random.seed(split_random_seed)

    half_ind_size = all_valid_image_indices.size //2
    shuffled_ind = np.arange(all_valid_image_indices.size)

    np.random.shuffle(shuffled_ind)
    ind_split = [
                np.sort(all_valid_image_indices[shuffled_ind[:half_ind_size]]), 
                np.sort(all_valid_image_indices[shuffled_ind[half_ind_size:]]),
                ]
    return ind_split
        

def make_dataset_loader_dict(args):
    dataset_loader_dict = { 'particles_file' : args.particles,
                            'ctf_file': args.ctf ,
                            'poses_file' : args.poses,
                            'datadir': args.datadir,
                            'n_images' : args.n_images,
                            'ind_file': args.ind,
                            'padding' : args.padding,
                            'tilt_series' : False,
                            'tilt_series_ctf' : 'cryoem',
                            'angle_per_tilt' : None,
                            'dose_per_tilt' : None,
                            'premultiplied_ctf' : False,
                            'strip_prefix': getattr(args, 'strip_prefix', None),
                            }
    
    # For backward compatibility... Delete at some point?
    if hasattr(args,'tilt_series'):
        dataset_loader_dict['tilt_series'] = args.tilt_series
        dataset_loader_dict['tilt_series_ctf'] = args.tilt_series_ctf
        dataset_loader_dict['angle_per_tilt'] = args.angle_per_tilt
        dataset_loader_dict['dose_per_tilt'] = args.dose_per_tilt

    if hasattr(args, 'premultiplied_ctf'):
        dataset_loader_dict['premultiplied_ctf'] = args.premultiplied_ctf

    # if hasattr(args,'tilt_ind'):
    #     dataset_loader_dict['tilt_ind'] = args.tilt_ind


    if args.uninvert_data == "automatic" or  args.uninvert_data == "false":
        dataset_loader_dict['uninvert_data'] = False
    elif args.uninvert_data == "true":
        dataset_loader_dict['uninvert_data'] = True
    else:
        raise Exception("input uninvert-data option is wrong. Should be automatic, true or false ")
    
    return dataset_loader_dict

def figure_out_halfsets(args):

    if args.halfsets == None:
        logger.info("Randomly splitting dataset into halfsets")
        # ind_split = dataset.get_split_indices(args.particles_file, ind_file = args.ind)
        # # pickle.dump(ind_split, open(args.out))
        if args.tilt_series or args.tilt_series_ctf != 'cryoem':
            halfsets = get_split_tilt_indices(args.particles, ind_file = args.ind, tilt_ind_file = args.tilt_ind, ntilts = args.ntilts, datadir = args.datadir)
        else:
            halfsets = get_split_indices(args.particles, ind_file = args.ind)
    # else:
    #     logger.info("Loading halfset from file")
    #     halfsets = pickle.load(open(args.halfsets, 'rb'))
    else:
        if args.tilt_series or args.tilt_series_ctf!= 'cryoem':
            halfsets = get_split_tilt_indices(args.particles, ind_file = args.ind, tilt_ind_file = args.tilt_ind, ntilts = args.ntilts, datadir = args.datadir)
            logger.warning("Ignoring halfsets file for tilt series! Using randomly partition instead.")
            return halfsets
        
        logger.info("Loading halfset from file")
        halfsets = pickle.load(open(args.halfsets, 'rb'))

        # Ensure only the indices in args.ind are used
        if args.ind is not None:
            # Load indices from args.ind
            if isinstance(args.ind, np.ndarray):
                ind = args.ind
            else:
                with open(args.ind, 'rb') as f:
                    ind = np.asarray(pickle.load(f))
            # Intersect the loaded halfsets with ind
            halfsets = [np.intersect1d(halfset, ind) for halfset in halfsets]

    if args.n_images > 0:
        halfsets = [ halfset[:args.n_images//2] for halfset in halfsets]
        logger.info(f"using only {args.n_images} images")
        if args.tilt_series:
            raise NotImplementedError
    return halfsets


def load_dataset_from_args(args, lazy = False):
    ind_split = figure_out_halfsets(args)
    dataset_loader_dict = make_dataset_loader_dict(args)
    return get_split_datasets_from_dict(dataset_loader_dict, ind_split, lazy = lazy)



# Only used in recovar_coding_example.ipynb  
def get_default_dataset_option():
    dataset_loader_dict = { 'particles_file' : None,
                            'ctf_file': None ,
                            'poses_file' : None,
                            'datadir': None,
                            'n_images' : -1,
                            'ind': None,
                            # 'tilt_ind': None,
                            'padding' : 0,
                            # 'lazy': False,
                            'tilt_series' : False,
                            'tilt_series_ctf' : 'cryoem',
                            'angle_per_tilt' : 3,
                            'dose_per_tilt' : 2.9,
                            'uninvert_data' : False,
                            'premultiplied_ctf' : False,}
    return dataset_loader_dict

def load_dataset_from_dict(dataset_loader_dict, lazy = True):
    # if dataset_loader_dict['lazy']:
    #     return load_cryodrgn_dataset(**dataset_loader_dict, lazy = lazy)
    return load_cryodrgn_dataset(**dataset_loader_dict, lazy = lazy)


def reorder_to_original_indexing(arr, cryos, use_tilt_indices = False):
    # if cryos[0].tilt_series_flag:
    if use_tilt_indices:
        dataset_indices = [ cryo.image_stack.dataset_tilt_indices for cryo in cryos]
    else:
        dataset_indices = [ cryo.dataset_indices for cryo in cryos]
    return reorder_to_original_indexing_from_halfsets(arr, dataset_indices)

    # inv_argsort = np.argsort(dataset_indices)
    # return arr[inv_argsort]

def reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images = None ):
    if type(arr) is list:
        arr = np.concatenate(arr)
    
    dataset_indices = np.concatenate(halfsets)
    
    num_images = (np.max(dataset_indices)+1) if num_images is None else num_images 
    arr_reorder_shape = (num_images, *arr.shape[1:])
    arr_reorder = np.ones(arr_reorder_shape) * np.nan # nan things which are not in halfsets. They have been filtered out.
    arr_reorder[dataset_indices] = arr
    return arr_reorder

