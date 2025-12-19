import functools
import jax
import healpy as hp
import numpy as np
from recovar import utils

def get_rotation_grid(nside_level, n_in_planes = None, matrices = False):

    #  * order	Npix	Theta-sampling
    #  * 0		12		58.6
    #  * 1		48		29.3
    #  * 2		192		14.7
    #  * 3		768		7.33
    #  * 4		3072	3.66
    #  * 5		12288	1.83
    #  * 6		49152	0.55
    #  * 7		196608	0.28
    #  * 8		786432	0.14

    nside = 2**nside_level
    m = hp.nside2npix(nside)
    z = hp.pix2ang(nside, np.arange(m))

    if n_in_planes is None:
        angle_res = 360 / ( 6 * 2**nside_level)
        n_in_planes = np.round(360 / angle_res).astype(int)

    in_angle_angles = np.linspace(0, 2 * np.pi, n_in_planes, endpoint=False)
    angles = np.meshgrid( np.arange(m), in_angle_angles )
    theta = z[0][angles[0]]
    phi = z[1][angles[0]]
    angles = np.stack( [theta, phi, angles[1] ], axis=-1)
    angles = angles.reshape( -1, 3)
    angles = angles / (2 * np.pi) * 360
    if matrices:
        angles = utils.R_from_relion(angles)
    return angles

def get_angle_resolution(nside_level):
    nside = 2**nside_level
    return hp.nside2resol(nside, arcmin=True) / 60

def get_translation_grid(max_pixel, pixel_offset):
    gridded_max_pixel = (max_pixel // pixel_offset ) * pixel_offset 
    # xrange_one_sided = np.arange(0, max_pixel//2 + 1, pixel_offset)
    xrange = np.arange(-gridded_max_pixel, gridded_max_pixel + 1, pixel_offset)
    x, y = np.meshgrid(xrange, xrange)
    grid = np.stack([x.flatten(), y.flatten()], axis = 1)
    norm_res = np.linalg.norm(grid, axis = 1) <= max_pixel + 0.001
    grid = grid[norm_res] 
    return grid




@functools.partial(jax.jit, static_argnums=[1])
def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    # Does this not work?
    indices = translations + image_shape[0]//2
    vec_indices = indices[...,1] * image_shape[1] + indices[...,0]
    # logger.warning("not sure that this is working as intended")
    return vec_indices
