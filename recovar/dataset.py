import logging

import jax.numpy as jnp
import numpy as np
import pickle 

from recovar import plot_utils, core, mask
import recovar.fourier_transform_utils as fourier_transform_utils
from recovar import tilt_dataset

logger = logging.getLogger(__name__)

# Maybe should take out these dependencies?
from recovar.image_loader import ImageSource


def MRCDataMod(particles_file, ind =None , datadir = None, padding = 0, uninvert_data = False, strip_prefix = None):
    return tilt_dataset.ImageDataset(particles_file, ind = ind, datadir = datadir, padding = padding, invert_data = uninvert_data, lazy =False, strip_prefix=strip_prefix)


def LazyMRCDataMod(particles_file, ind =None , datadir = None, padding = 0, uninvert_data = False, strip_prefix = None):
    return tilt_dataset.ImageDataset(particles_file, ind = ind, datadir = datadir, padding = padding, invert_data = uninvert_data, lazy =True, strip_prefix=strip_prefix)
    
    
def get_num_images_in_dataset(mrc_path, datadir = None, strip_prefix = None):
    return ImageSource.from_file(mrc_path, lazy=True, datadir = datadir, strip_prefix = strip_prefix).n

def set_standard_mask(D, dtype):
    return mask.window_mask(D, 0.85, 0.99)
    

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

        self.volume_shape = tuple(3*[self.grid_size ])
        self.volume_size = np.prod(self.volume_shape)

        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

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

        
        self.tilt_series_flag = tilt_series_flag # Hopefully can just switch this on and off
        self.premultiplied_ctf = premultiplied_ctf

        # For SPA, it is # of images, for ET, it is # of tilt series
        # For tilt series: A "tilt" is an image. A particle is a full tilt series 
        self.n_units = self.image_stack.Np if self.tilt_series_flag else self.n_images

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
        if self.noise is None:
            return None
        return self.noise.get(indices)

    def delete(self):
        del self.image_stack.particles
        del self.image_stack
        del self.rotation_matrices
        del self.CTF_params
        del self.translations
        del self.noise

    def update_volume_upsampling_factor(self, volume_upsampling_factor):

        self.volume_upsampling_factor = volume_upsampling_factor
        self.upsampled_grid_size = self.grid_size * volume_upsampling_factor
        
        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

        return

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
        im = to_real(fourier_transform_utils.get_idft2(jnp.take(X.reshape(self.volume_shape), self.grid_size//2, axis = axis)))
        if hide_padding:
            im = im[self.hpad:self.image_stack.unpadded_D + self.hpad,self.hpad:self.image_stack.unpadded_D + self.hpad]
        return im


    def get_slice(self, X, to_real_fn = np.abs, axis = 0):
        # zero_th freq
        z_freq = self.grid_size//2 +1
        im = to_real_fn(jnp.take(X.reshape(self.volume_shape), z_freq, axis = axis))
        return im

    def get_slice_real(self, X, to_real_fn = np.real, axis = 0):
        im = to_real_fn(fourier_transform_utils.get_idft3(X.reshape(self.volume_shape)))
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
            return to_real(fourier_transform_utils.get_idft2(self.get_image(i,tilt_idx))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        else:
            return to_real(fourier_transform_utils.get_idft2(self.get_image(i,tilt_idx)))


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
            images = images
            CTFs = self.CTF_fun(self.CTF_params[i][None], self.image_shape, self.voxel_size) # Compute CTF
        images = self.image_stack.process_images(images) # Compute DFT, masking
        images = (CTFs / (CTFs**2 + weiner_param)) * images  # CTF correction
        images = images.reshape(self.image_shape)
        return to_real(fourier_transform_utils.get_idft2(images))


    def plot_FSC(self, image1 = None, image2 = None, filename = None, threshold = 0.5, curve = None, ax = None):
        score = plot_utils.plot_fsc_new(image1, image2, self.volume_shape, self.voxel_size,  curve = curve, ax = ax, threshold = threshold, filename = filename)
        return score
    
    def get_image_mask(self, indices, mask, binary = True, soften = 5):
        indices = np.asarray(indices).astype(int)
        from recovar import covariance_core # Not sure I want this depency to exist... Could make some circular imports
        mask = covariance_core.get_per_image_tight_mask(mask, self.rotation_matrices[indices], self.image_stack.mask, self.volume_mask_threshold, self.image_shape, self.volume_shape, self.grid_size, self.padding, disc_type = 'linear_interp',  binary = binary, soften = soften)
        mask_ft = fourier_transform_utils.get_dft2(mask).reshape(mask.shape[0], -1)
        # Usually images are translated, here we translate back.
        batch = core.translate_images(mask_ft, -self.translations[indices].astype(int) , self.image_shape)
        mask2 = fourier_transform_utils.get_idft2(batch.reshape(-1, *self.image_shape))
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
            predicted_images = fourier_transform_utils.get_idft2(predicted_images.reshape(-1, *self.image_shape)).real
        return predicted_images


    def set_radial_noise_model(self, noise_variance):
        from recovar import noise
        self.noise = noise.RadialNoiseModel(noise_variance, image_shape = self.image_shape)

    def set_variable_radial_noise_model(self, noise_variance_radials):
        from recovar import noise
        _, dose_indices = jnp.unique(self.CTF_params[:,core.CTFParamIndex.DOSE], return_inverse=True)
        self.noise = noise.VariableRadialNoiseModel(noise_variance_radials, dose_indices, image_shape = self.image_shape)

# Loads dataset that are stored in the cryoDRGN format
def load_cryodrgn_dataset(
    particles_file,
    poses_file,
    ctf_file,
    datadir=None,
    n_images=None,
    ind=None,
    lazy=True,
    padding=0,
    uninvert_data=False,
    tilt_series=False,
    tilt_series_ctf=None,
    dose_per_tilt=2.9,
    angle_per_tilt=3,
    premultiplied_ctf=False,
    strip_prefix=None,
    sort_with_Bfac=False,
):
    
    # For backward compatibility... Delete at some point?
    if tilt_series_ctf is None and tilt_series is False:
        tilt_series_ctf = 'cryoem'
    elif tilt_series_ctf is None and tilt_series is True:
        tilt_series_ctf = 'relion5'
    elif tilt_series_ctf == 'warp':
        tilt_series_ctf = 'v2_scale_from_star'

        
    if tilt_series:
            from recovar import tilt_dataset
            tilt_file_option = 'relion5' if tilt_series_ctf == 'relion5' else 'warp'
            dataset = tilt_dataset.TiltSeriesData(particles_file, ind = ind, datadir = datadir, invert_data = uninvert_data, tilt_file_option=tilt_file_option, strip_prefix=strip_prefix)

    else:
        if lazy:
            dataset = LazyMRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data, strip_prefix=strip_prefix)
        else:
            dataset = MRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data, strip_prefix=strip_prefix)


    from recovar import load_utils
    ctf_params = np.array(load_utils.load_ctf_params(dataset.D, ctf_file))
        
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
        tilt_dataset_this = tilt_dataset.TiltSeriesData(
            particles_file,
            ind=ind,
            datadir=datadir,
            invert_data=uninvert_data,
            sort_with_Bfac=sort_with_Bfac,
        )
    else:
        tilt_dataset_this = dataset

    if tilt_series_ctf != 'cryoem':

        if tilt_series_ctf == 'relion5':
            ctf_params[:,core.CTFParamIndex.CONTRAST+1] = tilt_dataset_this.ctfscalefactor
            dose = tilt_dataset_this.dose
            angles = np.zeros_like(dose) # Set angles to 0 - the np.cos factor is included already?
            ctf_params = np.concatenate( [ctf_params, dose[...,None], angles[...,None]], axis =-1)
            CTF_fun = core.evaluate_ctf_wrapper_tilt_series_v2


        # The angles are used to compute a scale factor cos(angles). If scale from star, then the scale factor is already in the star file, so set angle to 0
        if "scale_from_star" in tilt_series_ctf:
            angle_per_tilt = 0

        if tilt_series_ctf == "from_star":
            ctf_params[:,core.CTFParamIndex.CONTRAST+1] = tilt_dataset_this.ctfscalefactor
            ctf_params[:,core.CTFParamIndex.BFACTOR+1] = -tilt_dataset_this.ctfBfactor # should be POSITIVE (negative in star file)
            logger.info('CTF from star')

        elif (tilt_series_ctf == "scale_from_star") or (tilt_series_ctf == "from_dose"):
            # Sort of a hacky way to do this.

            # + 1 because voxel_size in included.... gross
            if "scale_from_star" in tilt_series_ctf:
                ctf_params[:,core.CTFParamIndex.CONTRAST+1] = tilt_dataset_this.ctfscalefactor

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
                ctf_params[:,core.CTFParamIndex.CONTRAST+1] = tilt_dataset_this.ctfscalefactor
                # angles *=0 
                logger.warning("Using scale from star")

            if dose_per_tilt is None:
                dose = - (tilt_dataset_this.ctfBfactor / 4) # WARP uses a ctfBfactor == -4 * dose
                logger.warning("Using dose from star file (- Bfactor/4)")


            CTF_fun = core.evaluate_ctf_wrapper_tilt_series_v2
            ctf_params = np.concatenate( [ctf_params, dose[...,None], angles[...,None]], axis =-1)
            assert (np.isclose(ctf_params[0,4], 200) or np.isclose(ctf_params[0,4], 300)) , "Critical exposure calculation requires 200kV or 300kV imaging" 
            logger.info('CTF from dose weighting - V2')
            

    rots, trans, _ = load_utils.load_poses(poses_file, dataset.n_images, dataset.unpadded_D, ind=ind)


    voxel_sizes = ctf_params[:,0]
    assert np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0))
    voxel_size = np.float32(voxel_sizes[0])

    # Make sure everything is in correct dtype:
    ctf_params = ctf_params.astype(np.float32)
    translations = np.array(trans).astype(np.float32)
    rots = np.array(rots).astype(np.float32)

    if ind is not None:
        ind = np.asarray(ind).astype(int)

    return CryoEMDataset( dataset, voxel_size,
                              rots, translations, ctf_params[:,1:], 
                              CTF_fun = CTF_fun, 
                              dataset_indices = ind, 
                              tilt_series_flag = tilt_series, 
                              premultiplied_ctf = premultiplied_ctf)




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


def get_split_indices(particles_file, datadir=None, strip_prefix=None, ind_file=None, split_random_seed=0, validate_split=True):
    """
    Get indices for splitting dataset into halfsets.
    
    Args:
        particles_file: Path to particles STAR file
        datadir: Data directory (optional)
        strip_prefix: Prefix to strip from file paths (optional)
        ind_file: File containing specific indices to use (optional)
        split_random_seed: Random seed for reproducible splits
        validate_split: Whether to validate the split is balanced
        
    Returns:
        List of two numpy arrays containing indices for each halfset
    """
    if ind_file is None:
        n_images = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = np.arange(n_images)
    else:
        if isinstance(ind_file, np.ndarray):
            indices = ind_file
        else:
            # Get indices from file
            with open(ind_file, 'rb') as f:
                indices = np.asarray(pickle.load(f))
    
    if len(indices) == 0:
        raise ValueError("No valid indices found for dataset splitting")
    
    split_indices = split_index_list(indices, split_random_seed=split_random_seed)
    
    if validate_split:
        # Validate split is reasonably balanced
        n1, n2 = len(split_indices[0]), len(split_indices[1])
        total = n1 + n2
        if abs(n1 - n2) > max(1, total * 0.01):  # Allow 1% imbalance
            logger.warning(f"Split is imbalanced: {n1} vs {n2} images ({abs(n1-n2)/total*100:.1f}% difference)")
        
        # Check for overlap
        overlap = np.intersect1d(split_indices[0], split_indices[1])
        if len(overlap) > 0:
            raise ValueError(f"Split contains {len(overlap)} overlapping indices")
    
    logger.info(f"Split dataset into halfsets: {len(split_indices[0])} and {len(split_indices[1])} images")
    return split_indices


def get_split_tilt_indices(
    particles_file, ind_file=None, tilt_ind_file=None, ntilts=None, datadir=None, particle_halfset_indices_file=None
):
    """
    Split a tilt-series dataset into two halfsets (image indices), supporting optional filtering by image/particle indices and precomputed splits.
    """
    from recovar import tilt_dataset
    import pickle
    import numpy as np

    # Step 1: Parse STAR file for mapping
    particles_to_tilts, tilts_to_particles = tilt_dataset.TiltSeriesData.parse_particle_tilt(particles_file)

    # Step 2: Optionally get tilt numbers for ntilts filtering
    tilt_numbers = None
    if ntilts is not None:
        dataset_tmp = tilt_dataset.TiltSeriesData(particles_file, datadir=datadir)
        tilt_numbers = dataset_tmp.tilt_numbers

    # Step 3: Determine which particles to use
    if tilt_ind_file is not None:
        particle_ind = pickle.load(open(tilt_ind_file, "rb"))
    else:
        particle_ind = np.arange(len(particles_to_tilts))

    # Map selected particles to image indices
    allowed_image_indices = tilt_dataset.tilt_series_indices_to_image_indices(particle_ind, particles_file)

    # Step 4: Optionally filter by image indices
    if ind_file is not None:
        ind_images = pickle.load(open(ind_file, "rb"))
        allowed_image_indices = np.intersect1d(allowed_image_indices, ind_images)

    # Step 5: Keep only particles with at least one allowed image
    image_to_particle = np.array([tilts_to_particles[i] for i in allowed_image_indices])
    valid_particles = np.unique(image_to_particle)

    # Step 6: Determine halfset split (by particles)
    if particle_halfset_indices_file is not None:
        split_particles = pickle.load(open(particle_halfset_indices_file, "rb"))
        # If tilt_ind_file is set, intersect with valid_particles
        if tilt_ind_file is not None:
            split_particles = [np.intersect1d(split_particles[0], valid_particles),
                              np.intersect1d(split_particles[1], valid_particles)]
    else:
        split_particles = split_index_list(valid_particles)

    # Step 7: For each halfset, get all image indices for those particles, filter by ntilts if needed, and intersect with allowed images
    split_image_indices = []
    for half in split_particles:
        imgs = np.concatenate([particles_to_tilts[ind] for ind in half])
        if tilt_numbers is not None:
            imgs = imgs[tilt_numbers[imgs] < ntilts]
        imgs = np.intersect1d(imgs, allowed_image_indices)
        split_image_indices.append(imgs)

    return split_image_indices



def split_index_list(all_valid_image_indices, split_random_seed=0):
    """
    Split a list of indices into two balanced halves with reproducible randomization.
    
    Args:
        all_valid_image_indices: Array of indices to split
        split_random_seed: Random seed for reproducible splits
        
    Returns:
        List of two numpy arrays containing the split indices
    """
    all_valid_image_indices = np.asarray(all_valid_image_indices)
    if len(all_valid_image_indices) == 0:
        raise ValueError("Cannot split empty index list")
    
    n_indices = len(all_valid_image_indices)
    half_ind_size = n_indices // 2
    
    # Create shuffled indices
    shuffled_ind = np.arange(n_indices)
    rng = np.random.default_rng(split_random_seed)
    rng.shuffle(shuffled_ind)
    
    # Split into two halves
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

    if args.uninvert_data == "automatic" or  args.uninvert_data == "false":
        dataset_loader_dict['uninvert_data'] = False
    elif args.uninvert_data == "true":
        dataset_loader_dict['uninvert_data'] = True
    else:
        raise ValueError("input uninvert-data option is wrong. Should be automatic, true or false ")
    
    return dataset_loader_dict

def figure_out_halfsets(args):

    if args.halfsets is None:
        logger.info("Randomly splitting dataset into halfsets")
        if args.tilt_series or args.tilt_series_ctf != 'cryoem':
            halfsets = get_split_tilt_indices(args.particles, ind_file = args.ind, tilt_ind_file = args.tilt_ind, ntilts = args.ntilts, datadir = args.datadir)
        else:
            halfsets = get_split_indices(args.particles, datadir = args.datadir, strip_prefix = args.strip_prefix, ind_file = args.ind)
    else:
        logger.info("Loading halfsets from file")

        if args.tilt_series or args.tilt_series_ctf!= 'cryoem':
            halfsets = get_split_tilt_indices(args.particles, ind_file = args.ind, tilt_ind_file = args.tilt_ind, ntilts = args.ntilts, datadir = args.datadir, particle_halfset_indices_file = args.halfsets)
        else:
            halfsets = pickle.load(open(args.halfsets, 'rb'))
            logger.info("Loaded halfsets from file")

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
        logger.info(f"using only {args.n_images} particles")
    return halfsets


def load_dataset_from_args(args, lazy = False, ind_split = None):
    if ind_split is None:
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
    return load_cryodrgn_dataset(**dataset_loader_dict, lazy = lazy)


def reorder_to_original_indexing(arr, cryos, use_tilt_indices = False):
    if use_tilt_indices:
        dataset_indices = [ cryo.image_stack.dataset_tilt_indices for cryo in cryos]
    else:
        dataset_indices = [ cryo.dataset_indices for cryo in cryos]
    return reorder_to_original_indexing_from_halfsets(arr, dataset_indices)

def reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images = None ):
    if isinstance(arr, list):
        arr = np.concatenate(arr)
    
    dataset_indices = np.concatenate(halfsets)
    
    num_images = (np.max(dataset_indices)+1) if num_images is None else num_images 
    arr_reorder_shape = (num_images, *arr.shape[1:])
    arr_reorder = np.ones(arr_reorder_shape) * np.nan # nan things which are not in halfsets. They have been filtered out.
    arr_reorder[dataset_indices] = arr
    return arr_reorder
