import logging
from collections import defaultdict, deque

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


def _load_index_like(value):
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    with open(value, "rb") as f:
        return pickle.load(f)


def _normalize_image_indices(values, n_images_total=None, name="indices"):
    arr = np.asarray(values)
    if arr.dtype == bool:
        if n_images_total is None:
            raise ValueError(f"{name} boolean mask requires known dataset size for validation")
        if arr.ndim != 1:
            raise ValueError(f"{name} boolean mask must be 1D")
        if arr.size != int(n_images_total):
            raise ValueError(
                f"{name} boolean mask length {arr.size} must match number of images {int(n_images_total)}"
            )
        return np.flatnonzero(arr).astype(np.int32, copy=False)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} indices must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} indices must be integer or boolean mask")

    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)

    if np.any(arr < 0):
        raise ValueError(f"{name} indices must be non-negative")
    if n_images_total is not None and np.any(arr >= int(n_images_total)):
        raise ValueError(f"{name} indices contain values >= number of images ({int(n_images_total)})")

    return arr.astype(np.int32, copy=False)


def _deduplicate_preserve_order(values, name):
    values = np.asarray(values)
    if values.size == 0:
        return values
    _, first_idx = np.unique(values, return_index=True)
    if first_idx.size != values.size:
        dropped = int(values.size - first_idx.size)
        logger.warning("Dropping %d duplicate entries from %s.", dropped, name)
    return values[np.sort(first_idx)]
    

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
        self.rotation_matrices = np.asarray(rotation_matrices, dtype=rotation_dtype)
        self.translations = np.asarray(translations, dtype=self.dtype_real)

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
        self.CTF_params = np.asarray(CTF_params, dtype=self.CTF_dtype)

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
        return self.CTF_fun_inp(*args).astype(self.CTF_dtype, copy=False)

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
        # If noise_variance_radials is 1D (single radial profile), broadcast
        # to 2D (one row per dose level) so VariableRadialNoiseModel can index
        # by dose_indices.
        if noise_variance_radials is not None and np.ndim(noise_variance_radials) == 1:
            n_doses = int(jnp.max(dose_indices)) + 1
            noise_variance_radials = np.tile(noise_variance_radials, (n_doses, 1))
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
    def _normalize_dataset_indices(ind_value, n_total):
        if ind_value is None:
            return None
        arr = np.asarray(ind_value)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError("ind boolean mask must be 1D")
            if arr.size != int(n_total):
                raise ValueError(
                    f"ind boolean mask length {arr.size} must match number of images {int(n_total)}"
                )
            return np.flatnonzero(arr).astype(np.int32, copy=False)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("ind must be 1D")
        if arr.dtype.kind not in ("i", "u"):
            raise TypeError("ind must be integer indices or boolean mask")

        arr = arr.astype(np.int64, copy=False).reshape(-1)
        if arr.size > 0:
            if np.any(arr < 0):
                raise IndexError("ind contains negative indices")
            if np.any(arr >= int(n_total)):
                raise IndexError(f"ind contains values >= number of images ({int(n_total)})")
        return arr.astype(np.int32, copy=False)
    
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
    ctf_params_all = np.array(load_utils.load_ctf_params(dataset.D, ctf_file))
    dataset_indices = _normalize_dataset_indices(ind, n_total=ctf_params_all.shape[0])
    if dataset_indices is None and ctf_params_all.shape[0] != dataset.n_images:
        raise ValueError(
            f"CTF parameter count ({ctf_params_all.shape[0]}) must match loaded image count ({dataset.n_images}) "
            "when ind is not provided"
        )
    ctf_params = ctf_params_all if dataset_indices is None else ctf_params_all[dataset_indices]
    
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
            

    rots, trans, _ = load_utils.load_poses(poses_file, dataset.n_images, dataset.unpadded_D, ind=dataset_indices)


    voxel_sizes = ctf_params[:,0]
    assert np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0))
    voxel_size = np.float32(voxel_sizes[0])

    # Make sure everything is in correct dtype:
    ctf_params = ctf_params.astype(np.float32)
    rots = np.asarray(rots, dtype=np.float32)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3):
        raise ValueError(f"Rotation array must have shape (N, 3, 3), got {rots.shape}")

    if trans is None:
        # Support rotation-only pose files by assuming zero in-plane shifts.
        translations = np.zeros((rots.shape[0], 2), dtype=np.float32)
    else:
        translations = np.asarray(trans, dtype=np.float32)
        expected_t_shape = (rots.shape[0], 2)
        if translations.shape != expected_t_shape:
            raise ValueError(
                f"Translation array must have shape {expected_t_shape}, got {translations.shape}"
            )

    return CryoEMDataset( dataset, voxel_size,
                              rots, translations, ctf_params[:,1:], 
                              CTF_fun = CTF_fun, 
                              dataset_indices = dataset_indices, 
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
        indices = np.arange(n_images, dtype=np.int32)
    else:
        raw_indices = _load_index_like(ind_file)
        n_images_total = None
        if np.asarray(raw_indices).dtype == bool:
            n_images_total = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = _normalize_image_indices(raw_indices, n_images_total=n_images_total, name="ind_file")
        indices = _deduplicate_preserve_order(indices, name="ind_file").astype(np.int32, copy=False)
    
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

    def _load_index_like(value):
        if value is None:
            return None
        if isinstance(value, (np.ndarray, list, tuple)):
            return value
        with open(value, "rb") as f:
            return pickle.load(f)

    def _filter_preserve_order(values, allowed):
        values = np.asarray(values)
        allowed = np.asarray(allowed)
        if values.size == 0:
            return values.astype(np.int32, copy=False)
        return values[np.isin(values, allowed)]

    def _normalize_particle_ids(values, n_particles_total):
        arr = np.asarray(values)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError("tilt_ind_file/particle halfset boolean mask must be 1D")
            if arr.size != int(n_particles_total):
                raise ValueError(
                    f"tilt_ind_file/particle halfset boolean mask length {arr.size} "
                    f"must match number of particles {int(n_particles_total)}"
                )
            return np.flatnonzero(arr).astype(np.int64, copy=False)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("tilt_ind_file/particle halfset ids must be 1D")
        if arr.dtype.kind not in ("i", "u"):
            raise TypeError("tilt_ind_file/particle halfset ids must be integer or boolean mask")
        return arr.astype(np.int64, copy=False).reshape(-1)

    def _normalize_image_ids(values, n_images_total):
        arr = np.asarray(values)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError("ind_file boolean mask must be 1D")
            if arr.size != int(n_images_total):
                raise ValueError(
                    f"ind_file boolean mask length {arr.size} must match number of images {int(n_images_total)}"
                )
            return np.flatnonzero(arr).astype(np.int32, copy=False)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("ind_file image ids must be 1D")
        if arr.dtype.kind not in ("i", "u"):
            raise TypeError("ind_file image ids must be integer or boolean mask")
        return arr.astype(np.int32, copy=False).reshape(-1)

    def _sanitize_particle_ids(values, n_particles_total, allowed_particles):
        values = _normalize_particle_ids(values, n_particles_total=n_particles_total)
        if values.size == 0:
            return values.astype(np.int32, copy=False)
        in_bounds = (values >= 0) & (values < int(n_particles_total))
        if not np.all(in_bounds):
            dropped = int(np.sum(~in_bounds))
            logger.warning("Dropping %d out-of-range particle ids from precomputed halfset.", dropped)
        values = values[in_bounds]
        values = _filter_preserve_order(values, allowed_particles)
        if values.size > 0:
            # Remove duplicate particle ids while preserving first-seen order.
            _, first_idx = np.unique(values, return_index=True)
            if first_idx.size != values.size:
                dropped = int(values.size - first_idx.size)
                logger.warning("Dropping %d duplicate particle ids from precomputed halfset.", dropped)
            values = values[np.sort(first_idx)]
        return values.astype(np.int32, copy=False)

    # Step 1: Parse STAR file for mapping
    particles_to_tilts, tilts_to_particles = tilt_dataset.TiltSeriesData.parse_particle_tilt(particles_file)

    # Step 2: Optionally get tilt numbers for ntilts filtering
    tilt_numbers = None
    if ntilts is not None and ntilts > 0:
        dataset_tmp = tilt_dataset.TiltSeriesData(particles_file, datadir=datadir)
        tilt_numbers = dataset_tmp.tilt_numbers

    n_particles_total = len(particles_to_tilts)

    # Step 3: Determine which particles to use
    if tilt_ind_file is not None:
        particle_ind = _sanitize_particle_ids(
            _load_index_like(tilt_ind_file),
            n_particles_total=n_particles_total,
            allowed_particles=np.arange(n_particles_total, dtype=np.int32),
        )
    else:
        particle_ind = np.arange(n_particles_total, dtype=np.int32)

    if particle_ind.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Map selected particles to image indices
    allowed_image_indices = tilt_dataset.tilt_series_indices_to_image_indices(particle_ind, particles_file)

    # Step 4: Optionally filter by image indices
    if ind_file is not None:
        ind_images = _normalize_image_ids(
            _load_index_like(ind_file),
            n_images_total=len(tilts_to_particles),
        )
        allowed_image_indices = _filter_preserve_order(allowed_image_indices, ind_images)

    if len(allowed_image_indices) == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Step 5: Keep only particles with at least one allowed image.
    # Use fromiter to avoid materializing an intermediate Python list for large ET datasets.
    image_to_particle = np.fromiter(
        (tilts_to_particles[int(i)] for i in np.asarray(allowed_image_indices).reshape(-1)),
        dtype=np.int32,
        count=len(allowed_image_indices),
    )
    valid_particles = np.unique(image_to_particle)
    if valid_particles.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Step 6: Determine halfset split (by particles)
    if particle_halfset_indices_file is not None:
        split_particles_raw = _load_index_like(particle_halfset_indices_file)
        if len(split_particles_raw) != 2:
            raise ValueError("particle_halfset_indices_file must contain exactly two halfsets")
        split_particles = [
            _sanitize_particle_ids(split_particles_raw[0], n_particles_total=n_particles_total, allowed_particles=valid_particles),
            _sanitize_particle_ids(split_particles_raw[1], n_particles_total=n_particles_total, allowed_particles=valid_particles),
        ]
    else:
        split_particles = split_index_list(valid_particles)

    # Step 7: For each halfset, get all image indices for those particles, filter by ntilts if needed, and intersect with allowed images
    split_image_indices = []
    for half in split_particles:
        if len(half) == 0:
            split_image_indices.append(np.array([], dtype=np.int32))
            continue
        imgs = np.concatenate([particles_to_tilts[ind] for ind in half])
        if ntilts is not None:
            if ntilts <= 0:
                imgs = imgs[:0]
            else:
                imgs = imgs[tilt_numbers[imgs] < ntilts]
        imgs = _filter_preserve_order(imgs, allowed_image_indices)
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
            if len(halfsets) != 2:
                raise ValueError("halfsets file must contain exactly two halfsets")

            needs_n_images = any(np.asarray(h).dtype == bool for h in halfsets)
            n_images_total = None
            if needs_n_images:
                n_images_total = get_num_images_in_dataset(
                    args.particles,
                    datadir=args.datadir,
                    strip_prefix=args.strip_prefix,
                )
            halfsets = [
                _normalize_image_indices(halfsets[0], n_images_total=n_images_total, name="halfsets[0]"),
                _normalize_image_indices(halfsets[1], n_images_total=n_images_total, name="halfsets[1]"),
            ]

            # Ensure only the indices in args.ind are used
            if args.ind is not None:
                ind_raw = _load_index_like(args.ind)
                if n_images_total is None and np.asarray(ind_raw).dtype == bool:
                    n_images_total = get_num_images_in_dataset(
                        args.particles,
                        datadir=args.datadir,
                        strip_prefix=args.strip_prefix,
                    )
                ind = _normalize_image_indices(ind_raw, n_images_total=n_images_total, name="ind")
                # Intersect while preserving halfset order.
                halfsets = [np.asarray(halfset)[np.isin(np.asarray(halfset), ind)] for halfset in halfsets]

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
        if len(arr) == 0:
            arr = np.array([])
        else:
            arr = np.concatenate(arr)

    dataset_indices = np.concatenate(halfsets)
    dataset_indices = np.asarray(dataset_indices).reshape(-1)
    if dataset_indices.size == 0:
        if num_images is None:
            num_images = 0
    else:
        if np.any(dataset_indices < 0):
            raise ValueError("dataset indices must be non-negative")
        unique_count = np.unique(dataset_indices).size
        if unique_count != dataset_indices.size:
            raise ValueError("dataset indices contain duplicates across halfsets")
        inferred_num_images = int(np.max(dataset_indices)) + 1
        if num_images is None:
            num_images = inferred_num_images
        elif num_images < inferred_num_images:
            raise ValueError(
                f"num_images={num_images} is smaller than required size {inferred_num_images} from dataset indices"
            )

    if arr.shape[0] != dataset_indices.size:
        raise ValueError(
            f"arr first dimension ({arr.shape[0]}) must match number of dataset indices ({dataset_indices.size})"
        )

    arr_reorder_shape = (num_images, *arr.shape[1:])
    arr_reorder = np.ones(arr_reorder_shape) * np.nan # nan things which are not in halfsets. They have been filtered out.
    arr_reorder[dataset_indices] = arr
    return arr_reorder


class _SubsampledImageStack:
    """Lightweight image-stack wrapper that exposes a boolean/integer index subset."""

    def __init__(self, image_stack, subset_indices):
        self._stack = image_stack
        self._idx = np.asarray(subset_indices, dtype=np.int32)
        # Re-map to 0..n-1 so indices returned to the dataset are contiguous.
        self.n_images = len(self._idx)
        self.Np = self.n_images
        self.D = image_stack.D
        self.unpadded_D = getattr(image_stack, "unpadded_D", self.D)
        self.padding = getattr(image_stack, "padding", 0)
        self.image_shape = image_stack.image_shape
        self.mask = getattr(image_stack, "mask", None)

    def process_images(self, images, apply_image_mask=True):
        return self._stack.process_images(images, apply_image_mask=apply_image_mask)

    def _build_orig_to_local(self, local_indices, orig_indices):
        out = defaultdict(deque)
        for local, orig in zip(np.asarray(local_indices).reshape(-1), np.asarray(orig_indices).reshape(-1)):
            out[int(orig)].append(int(local))
        return out

    def _map_orig_to_local(self, indices, orig_to_local):
        idx_arr = np.asarray(indices)
        if idx_arr.size == 0:
            return idx_arr.astype(np.int32, copy=False)
        flat = idx_arr.reshape(-1)
        mapped_vals = []
        for i in flat:
            key = int(i)
            if key not in orig_to_local or len(orig_to_local[key]) == 0:
                raise KeyError(f"Original index {key} was not found in local mapping.")
            mapped_vals.append(orig_to_local[key].popleft())
        mapped = np.asarray(mapped_vals, dtype=np.int32)
        return mapped.reshape(idx_arr.shape)

    def _get_backing_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        # Prefer image-level generator for tilt-series stacks where dataset-level
        # subset APIs are particle-indexed.
        if hasattr(self._stack, "get_image_subset_generator"):
            return self._stack.get_image_subset_generator(
                batch_size, subset_indices, num_workers=num_workers
            )
        return self._stack.get_dataset_subset_generator(
            batch_size, subset_indices, num_workers=num_workers
        )

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        for start in range(0, self.n_images, batch_size):
            local_idx = np.arange(start, min(start + batch_size, self.n_images), dtype=np.int32)
            orig_idx = self._idx[local_idx]
            orig_to_local = self._build_orig_to_local(local_idx, orig_idx)
            for images, _, image_indices in self._get_backing_image_subset_generator(
                batch_size, orig_idx, num_workers=num_workers
            ):
                # Some image-stack implementations may emit one or many batches.
                # Always map yielded original indices back to contiguous local ones.
                local_image_indices = self._map_orig_to_local(image_indices, orig_to_local)
                yield images, local_image_indices, local_image_indices

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        if subset_indices is None:
            subset_indices = np.arange(self.n_images, dtype=np.int32)
        else:
            subset_indices = _normalize_image_indices(
                subset_indices, n_images_total=self.n_images, name="subset_indices"
            )
        orig_idx = self._idx[subset_indices]
        orig_to_local = self._build_orig_to_local(subset_indices, orig_idx)
        for images, _, image_indices in self._get_backing_image_subset_generator(
            batch_size, orig_idx, num_workers=num_workers
        ):
            local_image_indices = self._map_orig_to_local(image_indices, orig_to_local)
            yield images, local_image_indices, local_image_indices

    def get_image_generator(self, batch_size, num_workers=0):
        return self.get_dataset_generator(batch_size, num_workers=num_workers)

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers=num_workers)


def subsample_cryoem_dataset(cryo, good_indices):
    """Return a new CryoEMDataset containing only the images at *good_indices*.

    *good_indices* may be a boolean mask or an array of integer indices.
    The returned dataset has re-numbered per-image arrays so index ``i`` always
    refers to the i-th kept image.
    """
    good_indices = _normalize_image_indices(good_indices, n_images_total=cryo.n_images, name="good_indices")

    new_stack = _SubsampledImageStack(cryo.image_stack, good_indices) if cryo.image_stack is not None else None

    sub = CryoEMDataset(
        image_stack=new_stack,
        voxel_size=cryo.voxel_size,
        rotation_matrices=cryo.rotation_matrices[good_indices],
        translations=cryo.translations[good_indices],
        CTF_params=cryo.CTF_params[good_indices],
        CTF_fun=cryo.CTF_fun_inp,
        grid_size=cryo.grid_size,
        volume_upsampling_factor=cryo.volume_upsampling_factor,
        tilt_series_flag=cryo.tilt_series_flag,
        premultiplied_ctf=cryo.premultiplied_ctf,
        dataset_indices=good_indices,
    )
    if cryo.noise is not None:
        sub.noise = cryo.noise
    return sub
