import logging
import jax.numpy as jnp
import numpy as np
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)
from recovar import utils 
import skimage
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

def masking_options(volume_mask_option, means, volume_shape, input_mask, dtype_real = np.float32, mask_dilation_iter = 0):
    if isinstance(volume_mask_option, str):
        if volume_mask_option == 'from_halfmaps':
            volume_mask = make_mask_from_half_maps(means, smax = 3 )
            dilated_volume_mask = volume_mask
            logger.info('using mask computed from mean')
        elif volume_mask_option == 'sphere':
            volume_mask = get_radial_mask(volume_shape)
            dilated_volume_mask = get_radial_mask(volume_shape)
            logger.info('using spherical mask')
        elif volume_mask_option == 'none':
            volume_mask = np.ones(volume_shape)
            dilated_volume_mask = volume_mask
            logger.info('using no mask')
        elif volume_mask_option == 'input':
            assert(input_mask != None, 'set volume_mask_option = input, but no mask passed')
            input_mask = utils.load_mrc(input_mask) 
            logger.info('Using input mask')

            if mask_dilation_iter > 0:
                logger.info('thresholding and dilating input mask')
                input_mask = input_mask > 0.99
                input_mask = binary_dilation(input_mask,iterations=mask_dilation_iter)       

            if input_mask.shape[0] != volume_shape[0]:
                input_mask = skimage.transform.rescale( input_mask, volume_shape[0]/input_mask.shape[0])

            kernel_size = 3
            logger.info('Softening mask')
            volume_mask = soften_volume_mask(input_mask, kernel_size)
            dilated_volume_mask = binary_dilation(input_mask,iterations=6)
            dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
    else:
        assert(False, 'mask option not recognized')
    return volume_mask.astype(dtype_real), dilated_volume_mask.astype(dtype_real)

def make_mask_from_half_maps(means, smax = 3 ):
    # from emda.ext.maskmap_class import MaskedMaps
    ftu = fourier_transform_utils(np)
    x = MaskedMaps()
    x.smax = smax
    vol_shape = utils.guess_vol_shape_from_vol_size(means['corrected0'].size)
    x.arr1 = ftu.get_idft3(means['corrected0'].reshape(vol_shape)).real
    x.arr2 = ftu.get_idft3(means['corrected1'].reshape(vol_shape)).real
    x.generate_mask()
    return x.mask

def create_soft_edged_kernel_pxl(r1, shape):
    # Create soft-edged-kernel. r1 is the radius of kernel in pixels
    # This implementation is adapted from EMDA - https://gitlab.com/ccpem/emda
    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1
    
    # Are these offset by 1 pixel ? or 1/2 or something
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))]) + 1
    distances =  jnp.linalg.norm(volume_coords, axis =-1)
    half_boxsize = boxsize // 2
    r1 = half_boxsize
    r0 = r1 - 2
    
    kern_sphere_soft = jnp.where((distances < r0), jnp.ones_like(distances), jnp.zeros_like(distances))
    # 1
    kern_sphere_soft = jnp.where((distances <= r1) * (distances >= r0),
                                 (1 + jnp.cos(jnp.pi * (distances - r0) / (r1 - r0))) / 2.0,
                                 kern_sphere_soft )
    return kern_sphere_soft / jnp.sum(kern_sphere_soft)


def soften_volume_mask(volume_mask, kernel_size):

    image_shape = volume_mask.shape
    # Soft mask
    soft_edge_kernel = create_soft_edged_kernel_pxl(kernel_size, image_shape)

    # Convolve
    soft_edge_kernel_ft = ftu.get_dft3(soft_edge_kernel)
    volume_mask_ft = ftu.get_dft3(volume_mask)
    
    volume_mask_ft = volume_mask_ft * soft_edge_kernel_ft
    volume_mask = ftu.get_idft3(volume_mask_ft).real
    return np.array(volume_mask)


def get_radial_mask(shape, radius = None):
    radius = shape[0]//2-1 if radius is None else radius
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))])
    zero_out_outside_sphere_small =  jnp.linalg.norm(volume_coords, axis =-1) < radius + 1e-7
    return zero_out_outside_sphere_small


# Standard image masking (Other masking are used, too)
def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32), 
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask


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
        #self.mask = self.histogram2(halfcc3d, prob=self.prob)
        self.mask = self.thereshol_ccmap(ccmap=halfcc3d)


    def thereshol_ccmap(self, ccmap):
        from scipy.ndimage.morphology import binary_dilation

        ccmap_binary = (ccmap >= 1e-3).astype(int)
        dilate = binary_dilation(ccmap_binary, iterations=self.iter)
        mask = make_soft(dilate, kern_rad=2)
        return mask * (mask >= 1e-3)      


def threshold_map(arr, prob = 0.99, dthresh=None):
    if dthresh is None:
        X2 = np.sort(arr.flatten())
        F2 = np.array(range(len(X2))) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)



def make_soft(dilated_mask, kern_rad=3):
    # convoluting with gaussian sphere
    import scipy.signal
    kern_sphere = create_soft_edged_kernel_pxl(kern_rad, dilated_mask.shape)
    return scipy.signal.fftconvolve(dilated_mask, kern_sphere, "same")



## from realsp_local.py in EMDA
def get_3d_realspcorrelation(half1, half2, kern, mask=None):
    import scipy.signal
    from scipy.stats import mode

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")
    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A ** 2
    var3_B = loc3_B2 - loc3_B ** 2
    # regularization
    #reg_a = mode(var3_A)[0][0][0][0] / 100
    #reg_b = mode(var3_B)[0][0][0][0] / 100
    reg_a = np.max(var3_A) / 1000
    reg_b = np.max(var3_B) / 1000
    var3_A = np.where(var3_A < reg_a, reg_a, var3_A)
    var3_B = np.where(var3_B < reg_b, reg_b, var3_B)
    halfmaps_cc = cov3_AB / np.sqrt(var3_A * var3_B) 
    return halfmaps_cc


