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
    xrange = np.arange(-gridded_max_pixel, gridded_max_pixel + 1, pixel_offset)
    x, y = np.meshgrid(xrange, xrange)
    grid = np.stack([x.flatten(), y.flatten()], axis = 1)
    norm_res = np.linalg.norm(grid, axis = 1) <= max_pixel + 0.001
    grid = grid[norm_res] 
    return grid




def get_healpix_children(parent_pixels, parent_nside_level):
    """Return the 4 child HEALPix pixel indices for each parent pixel.

    Uses the NESTED pixel ordering property that each pixel p at nside N
    has exactly 4 children at nside 2N: 4p, 4p+1, 4p+2, 4p+3 (in NESTED).

    Args:
        parent_pixels: array-like of RING-ordered pixel indices at nside level
            ``parent_nside_level``.
        parent_nside_level: int, HEALPix level of the parent pixels
            (nside = 2**parent_nside_level).

    Returns:
        children: int array of length 4 * len(parent_pixels) with RING-ordered
            child pixel indices at level ``parent_nside_level + 1``.
            Children of ``parent_pixels[i]`` are at positions ``4*i:4*(i+1)``.
    """
    parent_pixels = np.asarray(parent_pixels)
    nside_parent = 2 ** parent_nside_level
    nside_child = 2 * nside_parent
    parent_nested = hp.ring2nest(nside_parent, parent_pixels)
    children_nested = 4 * np.repeat(parent_nested, 4) + np.tile(np.arange(4), len(parent_pixels))
    return hp.nest2ring(nside_child, children_nested)


def get_oversampled_rotation_grid(parent_pixels, parent_nside_level, oversampling_order=1):
    """Generate rotation matrices for HEALPix children of the given parent pixels.

    Subdivides each parent pixel ``oversampling_order`` times and returns
    rotation matrices for all resulting child pixels at all in-plane angles.

    Args:
        parent_pixels: array-like of RING-ordered pixel indices at level
            ``parent_nside_level``.
        parent_nside_level: int, HEALPix level of ``parent_pixels``.
        oversampling_order: int, number of subdivision levels (default 1).

    Returns:
        matrices: float64 (N, 3, 3) rotation matrices.
        parent_map: int (N,) index into ``parent_pixels`` for each rotation.
    """
    parent_pixels = np.asarray(parent_pixels)
    current_pixels = parent_pixels.copy()
    parent_map = np.arange(len(parent_pixels))

    for level in range(oversampling_order):
        children = get_healpix_children(current_pixels, parent_nside_level + level)
        parent_map = np.repeat(parent_map, 4)
        current_pixels = children

    fine_nside_level = parent_nside_level + oversampling_order
    fine_nside = 2 ** fine_nside_level
    theta, phi = hp.pix2ang(fine_nside, current_pixels)

    angle_res = 360 / (6 * 2 ** fine_nside_level)
    n_in_planes = int(np.round(360 / angle_res))
    in_plane_angles = np.linspace(0, 2 * np.pi, n_in_planes, endpoint=False)

    pix_idx, ip_idx = np.meshgrid(np.arange(len(current_pixels)), np.arange(n_in_planes))
    pix_idx_flat = pix_idx.ravel()

    euler_angles = np.stack(
        [theta[pix_idx_flat], phi[pix_idx_flat], in_plane_angles[ip_idx.ravel()]],
        axis=-1,
    )
    euler_angles = euler_angles / (2 * np.pi) * 360  # radians → degrees
    matrices = utils.R_from_relion(euler_angles)
    return matrices, parent_map[pix_idx_flat]


def get_oversampled_translation_grid(parent_translations, pixel_offset, oversampling_order=1):
    """Generate a finer translation grid by subdividing each parent cell.

    Each parent translation cell is subdivided into ``4**oversampling_order``
    child cells, with centers evenly spaced within the parent cell.

    Args:
        parent_translations: float (N, 2) parent translation grid points.
        pixel_offset: float, step size between parent grid points (pixels).
        oversampling_order: int, number of subdivision levels (default 1).

    Returns:
        fine_translations: float (N * 4**oversampling_order, 2).
        parent_map: int (N * 4**oversampling_order,) index into parent_translations.
    """
    parent_translations = np.asarray(parent_translations)
    n_parents = len(parent_translations)
    n_subdiv = 2 ** oversampling_order  # per dimension
    fine_offset = pixel_offset / n_subdiv

    half_offsets = np.linspace(
        -pixel_offset / 2 + fine_offset / 2,
        pixel_offset / 2 - fine_offset / 2,
        n_subdiv,
    )
    dx, dy = np.meshgrid(half_offsets, half_offsets)
    child_offsets = np.stack([dx.ravel(), dy.ravel()], axis=-1)  # (4**os, 2)
    n_children = n_subdiv ** 2

    fine_translations = (
        parent_translations[:, None, :] + child_offsets[None, :, :]
    ).reshape(-1, 2)
    parent_map = np.repeat(np.arange(n_parents), n_children)
    return fine_translations, parent_map


def subdivide_healpix_pixels(pixels, nside_level):
    """Subdivide HEALPix pixels and return child pixel angles and indices.

    Args:
        pixels: array-like of RING-ordered pixel indices at level
            ``nside_level``.
        nside_level: int, HEALPix level of ``pixels``.

    Returns:
        angles: float (n_child_pixels * n_in_planes, 3) Euler angles in
            degrees (theta, phi, psi) for each child orientation.
        child_pixels: int (n_child_pixels,) RING-ordered child pixel indices
            at level ``nside_level + 1``.
    """
    pixels = np.asarray(pixels)
    child_pixels = get_healpix_children(pixels, nside_level)
    child_nside_level = nside_level + 1
    child_nside = 2 ** child_nside_level
    theta, phi = hp.pix2ang(child_nside, child_pixels)

    angle_res = 360 / (6 * 2 ** child_nside_level)
    n_in_planes = int(np.round(360 / angle_res))
    in_plane_angles = np.linspace(0, 2 * np.pi, n_in_planes, endpoint=False)

    pix_idx, ip_idx = np.meshgrid(np.arange(len(child_pixels)), np.arange(n_in_planes))
    pix_idx_flat = pix_idx.ravel()

    angles = np.stack(
        [theta[pix_idx_flat], phi[pix_idx_flat], in_plane_angles[ip_idx.ravel()]],
        axis=-1,
    )
    angles = angles / (2 * np.pi) * 360  # radians → degrees
    return angles, child_pixels


@functools.partial(jax.jit, static_argnums=[1])
def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    indices = translations + image_shape[0]//2
    vec_indices = indices[...,1] * image_shape[1] + indices[...,0]
    return vec_indices
