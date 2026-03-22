"""Real-space and Fourier-space mask generation and manipulation."""

import logging

import jax.numpy as jnp
import numpy as np
import skimage
from scipy.ndimage import binary_dilation, distance_transform_edt

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils

logger = logging.getLogger(__name__)
## TODO: This should be heavily refactored, as most of this file
def masking_options(volume_mask_option, means, volume_shape, dtype_real = np.float32, mask_dilation_iter = 0, keep_input_mask = False, dilated_mask_dilations_iter = None):
    dilated_mask_dilations_iter = np.ceil(6 * volume_shape[0] / 128).astype(int) if dilated_mask_dilations_iter is None else dilated_mask_dilations_iter
    input_mask = volume_mask_option

    if isinstance(volume_mask_option, str):
        if volume_mask_option.endswith(".mrc"):
            input_mask = utils.load_mrc(input_mask).astype(np.float32)
            if keep_input_mask:
                volume_mask = input_mask
            else:
                logger.info('Using input mask')

                if mask_dilation_iter > 0:
                    logger.info('thresholding and dilating input mask')
                    input_mask = input_mask > 0.99
                    input_mask = binary_dilation(input_mask,iterations=mask_dilation_iter)       

                if input_mask.shape[0] != volume_shape[0]:
                    input_mask = skimage.transform.rescale( input_mask, volume_shape[0]/input_mask.shape[0])

                kernel_size = 3
                logger.info('Thresholding mask at 0.5 and softening cosine kernel of radius 3 pixels')
                input_mask = input_mask > 0.5
                volume_mask = soften_volume_mask(input_mask, kernel_size)

            dilated_volume_mask = binary_dilation(input_mask,iterations=dilated_mask_dilations_iter)
            dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
        elif volume_mask_option == 'from_halfmaps':
            volume_mask = make_mask_from_half_maps_from_means_dict(means, smax = 3 )
            kernel_size = 3
            logger.info('Softening mask')

            dilated_volume_mask = binary_dilation(volume_mask,iterations=dilated_mask_dilations_iter)
            volume_mask = soften_volume_mask(volume_mask, kernel_size)
            dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
            logger.info('using mask computed from mean')
        elif volume_mask_option == 'sphere':
            volume_mask = get_radial_mask(volume_shape)
            dilated_volume_mask = get_radial_mask(volume_shape)
            logger.info('using spherical mask')
        elif volume_mask_option == 'none':
            volume_mask = np.ones(volume_shape)
            dilated_volume_mask = volume_mask
            logger.info('using no mask')
        else:
            raise ValueError('mask option not recognized. Options are: a path ending in .mrc, from_halfmaps, sphere, none')
    else:
        raise ValueError('mask option not recognized')

    return np.array(volume_mask.astype(dtype_real)), np.array(dilated_volume_mask.astype(dtype_real))

def make_mask_from_half_maps_from_means_dict(means, smax = 3 ):
    vol_shape = utils.guess_vol_shape_from_vol_size(means.corrected0.size)
    halfmap1 = fourier_transform_utils.get_idft3(means.corrected0reg.reshape(vol_shape)).real
    halfmap2 = fourier_transform_utils.get_idft3(means.corrected1reg.reshape(vol_shape)).real
    return make_mask_from_half_maps(halfmap1, halfmap2, smax = smax )


def make_mask_from_half_maps(halfmap1, halfmap2, smax = 3 ):
    x = MaskedMaps()
    x.smax = smax
    x.arr1 = halfmap1
    x.arr2 = halfmap2
    x.iter = int(6 * halfmap1.shape[0] // 128)
    x.generate_mask()
    return x.mask


def make_mask_from_gt(gt_map_ft, smax = 3, iter = 10, from_ft = True ):
    x = MaskedMaps()
    x.smax = smax
    if iter is not None:
        x.iter = iter
    vol_shape = utils.guess_vol_shape_from_vol_size(gt_map_ft.size)
    if from_ft:
        x.arr1 = fourier_transform_utils.get_idft3(gt_map_ft.reshape(vol_shape)).real
    else:
        x.arr1 = gt_map_ft.reshape(vol_shape)
    x.generate_mask_from_gt()
    return x.mask


def make_union_gt_mask(gt_volumes_real, volume_shape, smax=3, iter=1,
                       dilation_iters=None, kern_rad=3):
    """Create a union mask from multiple ground-truth real-space volumes.

    For each volume, generates a per-volume mask via ``make_mask_from_gt``,
    thresholds at 0.5, then takes the logical OR of all per-volume masks.
    The union is dilated and softened to produce the final mask.

    Args:
        gt_volumes_real: Either a list of 3-D arrays or a 2-D array of shape
            ``(n_vols, n_voxels)`` (reshaped internally to 3-D).
        volume_shape: Tuple giving the 3-D grid dimensions.
        smax: Gaussian kernel radius for ``make_mask_from_gt``.
        iter: Dilation iterations inside ``make_mask_from_gt``.
        dilation_iters: Additional dilation iterations applied to the union
            mask.  Defaults to ``ceil(6 * volume_shape[0] / 128)`` (pipeline
            convention).
        kern_rad: Kernel radius for ``soften_volume_mask``.

    Returns:
        Tuple ``(soft_mask, binary_mask)`` where *soft_mask* is a float array
        in [0, 1] and *binary_mask* is the pre-softening boolean array.
    """
    if dilation_iters is None:
        dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))

    # Normalise input to a list of 3-D arrays
    if isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 2:
        gt_volumes_real = [gt_volumes_real[i].reshape(volume_shape)
                          for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 3:
        gt_volumes_real = [gt_volumes_real]

    union_mask = np.zeros(volume_shape, dtype=bool)
    for vol in gt_volumes_real:
        vol_3d = np.asarray(vol).reshape(volume_shape)
        per_vol_mask = make_mask_from_gt(vol_3d, smax=smax, iter=iter, from_ft=False)
        union_mask |= (per_vol_mask > 0.5)

    dilated = binary_dilation(union_mask, iterations=dilation_iters)
    binary_mask = np.asarray(dilated, dtype=bool)
    soft_mask = soften_volume_mask(binary_mask, kern_rad=kern_rad)

    return np.asarray(soft_mask, dtype=np.float32), binary_mask


def create_soft_edged_kernel_pxl(r1, shape):
    # Create soft-edged-kernel. r1 is the radius of kernel in pixels
    # This implementation is adapted from EMDA - https://gitlab.com/ccpem/emda
    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1
    
    # Are these offset by 1 pixel ? or 1/2 or something
    volume_coords =  fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))]) + 1
    distances =  jnp.linalg.norm(volume_coords, axis =-1)
    half_boxsize = boxsize // 2
    r1 = half_boxsize
    r0 = r1 - 2
    
    kern_sphere_soft = jnp.where((distances < r0), jnp.ones_like(distances), jnp.zeros_like(distances))
    kern_sphere_soft = jnp.where((distances <= r1) * (distances >= r0),
                                 (1 + jnp.cos(jnp.pi * (distances - r0) / (r1 - r0))) / 2.0,
                                 kern_sphere_soft )
    return kern_sphere_soft / jnp.sum(kern_sphere_soft)

def soften_volume_mask_new(binary_volume_mask, kernel_size):

    distance_to_mask = distance_transform_edt(binary_volume_mask < 0.9)
    mask = np.zeros_like(binary_volume_mask)
    mask = np.where((distance_to_mask >= 0) * (distance_to_mask < kernel_size ),
                    0.5 + 0.5 * np.cos(np.pi * (distance_to_mask) / kernel_size ),
                     mask)

    return np.asarray(mask.astype(np.float32))



def get_radial_mask(shape, radius = None):
    radius = shape[0]//2-1 if radius is None else radius
    volume_coords =  fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))])
    zero_out_outside_sphere_small =  jnp.linalg.norm(volume_coords, axis =-1) < radius + 1e-7
    return zero_out_outside_sphere_small


## TODO: I think window mask/standard_mask is defined in many places. REfactor/delete
# Standard image masking (Other masking are used, too)
def window_mask(D, in_rad, out_rad):
    if D % 2 != 0:
        raise ValueError(f"D must be even, got {D}")
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32), 
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask.astype(np.float32)


## What is below is copy-pasted from EMDA (https://emda.readthedocs.io/en/latest/)
# It was stripped out because it caused dependency issues.

class MaskedMaps:
    def __init__(self, hfmap_list=None):
        self.hfmap_list = hfmap_list
        self.mask = None
        self.uc = None
        self.arr1 = None
        self.arr2 = None
        self.origin = None
        self.iter = 3
        self.smax = 9 # kernel radius in pixels
        self.prob = 0.99
        self.dthresh = None

    def generate_mask(self):
        kern = create_soft_edged_kernel_pxl(self.smax, self.arr1.shape)
        self.arr1 = threshold_map(arr=self.arr1, prob=self.prob, dthresh=self.dthresh)
        self.arr2 = threshold_map(arr=self.arr2, prob=self.prob, dthresh=self.dthresh)
        halfcc3d = get_3d_realspcorrelation(self.arr1, self.arr2, kern)
        
        # radial mask stuff
        halfcc3d *= get_radial_mask(self.arr1.shape)

        self.mask = self.thereshol_ccmap(ccmap=halfcc3d)

    def generate_mask_from_gt(self):
        kern = create_soft_edged_kernel_pxl(self.smax, self.arr1.shape)
        self.arr1 = threshold_map(arr=self.arr1, prob=self.prob, dthresh=self.dthresh)
        self.arr1 = self.arr1> 0
        dilate = binary_dilation(self.arr1, iterations=self.iter)
        self.mask = soften_volume_mask(dilate, kern_rad=2)


    def thereshol_ccmap(self, ccmap):
        ccmap_binary = (ccmap >= 1e-3).astype(int)
        dilate = binary_dilation(ccmap_binary, iterations=self.iter)
        mask = soften_volume_mask(dilate, kern_rad=2)
        return mask * (mask >= 1e-3)      


def threshold_map(arr, prob = 0.99, dthresh=None):
    if dthresh is None:
        X2 = np.sort(arr.flatten())
        F2 = np.arange(len(X2)) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)


def smooth_circular_mask(image_size, radius, thickness):
    """Circular mask with a raised-cosine transition band.

    Values are 1 inside radius, 0 outside radius+thickness, and follow a
    cosine taper in the band [radius, radius+thickness].
    """
    half = image_size // 2
    coords = np.arange(-half, image_size - half, dtype=float)
    gx, gy = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(gx ** 2 + gy ** 2)
    band = (r >= radius) & (r <= radius + thickness)
    mask = np.zeros((image_size, image_size))
    mask[r < radius] = 1.0
    mask[band] = 0.5 + 0.5 * np.cos(np.pi * (r[band] - radius) / thickness)
    return mask



def raised_cosine_mask( volume_shape, radius, radius_p, offset):
    grid = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size = 1, scaled = False)
    grid -= offset

    distances =  jnp.linalg.norm(grid, axis =-1)
    mask = jnp.where(distances < radius, 1, 0)
    mask = jnp.where((distances >= radius) * (distances < radius_p ), 
                    0.5 - 0.5 * jnp.cos(np.pi * (radius_p - distances) / (radius_p - radius)),
                     mask)

    return mask.reshape(volume_shape)


def soft_mask_outside_map(vol, radius=-1, cosine_width=3, Mnoise=None):
    """Soft mask outside map, adapted from RELION."""
    vol = jnp.asarray(vol)
    if radius < 0:
        radius = np.max(np.array(vol.shape) // 2)

    radius_p = radius + cosine_width
    shape = vol.shape

    volume_coords =  fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))])
    r = jnp.linalg.norm(volume_coords, axis =-1)
    mask1 = r <= radius
    mask2 = (r > radius) * (r <= radius_p)
    mask3 = r > radius_p
    raised_cos = 0.5 + 0.5 * jnp.cos(jnp.pi * (radius_p - r) / cosine_width)
    mask = jnp.zeros_like(vol).real
    mask = jnp.where(mask1, 1, mask)
    mask = jnp.where(mask2, 1 - raised_cos, mask)

    if Mnoise is None:
        sum_bg = jnp.sum((vol * mask) * (mask3 + mask2))
        mask_sum = jnp.sum((mask) * (mask3 + mask2))
        avg_bg = sum_bg / mask_sum
    else:
        avg_bg = None

    if Mnoise is None:
        vol = jnp.where(mask3, avg_bg, vol)
    else:
        vol = jnp.where(mask3, Mnoise, vol)

    add = Mnoise if Mnoise is not None else avg_bg
    vol = mask * vol + (1 - mask) * add
    return vol, mask


def soften_volume_mask(dilated_mask, kern_rad=3):
    return soften_volume_mask_new(dilated_mask, kern_rad)



def get_3d_realspcorrelation(half1, half2, kern, mask=None):
    """3D real-space local correlation, adapted from EMDA realsp_local.py."""
    import scipy.signal

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")
    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A ** 2
    var3_B = loc3_B2 - loc3_B ** 2
    reg_a = np.max(var3_A) / 1000
    reg_b = np.max(var3_B) / 1000
    var3_A = np.where(var3_A < reg_a, reg_a, var3_A)
    var3_B = np.where(var3_B < reg_b, reg_b, var3_B)
    halfmaps_cc = cov3_AB / np.sqrt(var3_A * var3_B)
    return halfmaps_cc


def make_moving_gt_mask(gt_volumes_real, volume_shape, smax=3, iter=1,
                        dilation_iters=None, kern_rad=3):
    """Create a mask for the moving region across GT volumes.

    The moving signal is defined as the RMS deviation from the mean GT volume.
    Voxels with near-zero deviation are treated as static; the existing
    GT-masking heuristic is then applied to the deviation volume.

    Returns ``(soft_mask, binary_mask)``.
    """
    if dilation_iters is None:
        dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))

    if isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 2:
        gt_volumes_real = [gt_volumes_real[i].reshape(volume_shape)
                           for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 3:
        gt_volumes_real = [gt_volumes_real]

    if not gt_volumes_real:
        raise ValueError('gt_volumes_real must contain at least one volume')

    vols = np.asarray([np.asarray(vol).reshape(volume_shape) for vol in gt_volumes_real], dtype=np.float32)
    mean_vol = np.mean(vols, axis=0)
    moving_signal = np.sqrt(np.mean((vols - mean_vol[None]) ** 2, axis=0))

    moving_mask = make_mask_from_gt(moving_signal, smax=smax, iter=iter, from_ft=False) > 0.5
    if dilation_iters > 0 and np.any(moving_mask):
        moving_mask = binary_dilation(moving_mask, iterations=dilation_iters)

    binary_mask = np.asarray(moving_mask, dtype=bool)
    if np.any(binary_mask):
        soft_mask = soften_volume_mask(binary_mask, kern_rad=kern_rad)
    else:
        soft_mask = np.zeros(volume_shape, dtype=np.float32)

    return np.asarray(soft_mask, dtype=np.float32), binary_mask

