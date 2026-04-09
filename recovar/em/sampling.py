import functools
import jax
import healpy as hp
import numpy as np
from recovar import utils


def rotation_grid_n_in_planes(order: int) -> int:
    """Number of in-plane angles used by the RELION-style HEALPix grid."""
    angle_res = 360.0 / (6.0 * 2 ** order)
    return int(np.round(360.0 / angle_res))


def rotation_grid_size(order: int) -> int:
    """Total number of rotations in the full HEALPix x psi grid."""
    nside = 2 ** order
    return hp.nside2npix(nside) * rotation_grid_n_in_planes(order)


def _split_rotation_indices(indices, healpix_order):
    """Split full-grid rotation indices into HEALPix pixel and psi components."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    n_pixels = hp.nside2npix(2 ** healpix_order)
    pixel_idx = indices % n_pixels
    psi_idx = indices // n_pixels
    return pixel_idx, psi_idx


def _combine_rotation_indices(pixel_idx, psi_idx, healpix_order):
    """Combine HEALPix pixel and psi components into full-grid indices."""
    pixel_idx = np.asarray(pixel_idx, dtype=np.int64).reshape(-1)
    psi_idx = np.asarray(psi_idx, dtype=np.int64).reshape(-1)
    n_pixels = hp.nside2npix(2 ** healpix_order)
    return psi_idx * n_pixels + pixel_idx


def get_rotation_grid(nside_level, n_in_planes=None, matrices=False):

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
        angle_res = 360 / (6 * 2**nside_level)
        n_in_planes = np.round(360 / angle_res).astype(int)

    in_angle_angles = np.linspace(0, 2 * np.pi, n_in_planes, endpoint=False)
    angles = np.meshgrid(np.arange(m), in_angle_angles)
    theta = z[0][angles[0]]
    phi = z[1][angles[0]]
    angles = np.stack([theta, phi, angles[1]], axis=-1)
    angles = angles.reshape(-1, 3)
    angles = angles / (2 * np.pi) * 360
    if matrices:
        angles = utils.R_from_relion(angles)
    return angles


def get_angle_resolution(nside_level):
    nside = 2**nside_level
    return hp.nside2resol(nside, arcmin=True) / 60


def get_translation_grid(max_pixel, pixel_offset):
    gridded_max_pixel = (max_pixel // pixel_offset) * pixel_offset
    xrange = np.arange(-gridded_max_pixel, gridded_max_pixel + 1, pixel_offset)
    x, y = np.meshgrid(xrange, xrange)
    grid = np.stack([x.flatten(), y.flatten()], axis=1)
    norm_res = np.linalg.norm(grid, axis=1) <= max_pixel + 0.001
    grid = grid[norm_res]
    return grid


def rotation_indices_to_matrices(indices, healpix_order):
    """Convert full-grid rotation indices to rotation matrices.

    The indexing convention matches :func:`get_rotation_grid`: the grid is
    flattened with psi as the slow axis and HEALPix pixel as the fast axis,
    so ``index = psi_idx * n_pixels + pixel_idx``.
    """
    nside = 2 ** healpix_order
    n_psi = rotation_grid_n_in_planes(healpix_order)
    pixel_idx, psi_idx = _split_rotation_indices(indices, healpix_order)

    theta, phi = hp.pix2ang(nside, pixel_idx)
    psi = (2.0 * np.pi / n_psi) * psi_idx
    angles = np.stack(
        [np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(psi)],
        axis=-1,
    )
    return utils.R_from_relion(angles).astype(np.float32)


def remap_rotation_indices_to_order(indices, src_order, dst_order):
    """Map full-grid rotation indices between HEALPix orders.

    The mapped indices correspond to the nearest direction/pixel center on the
    destination grid together with the nearest in-plane sample.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if src_order == dst_order:
        return indices.copy()

    src_n_psi = rotation_grid_n_in_planes(src_order)
    dst_nside = 2 ** dst_order
    dst_n_psi = rotation_grid_n_in_planes(dst_order)

    src_pixel_idx, src_psi_idx = _split_rotation_indices(indices, src_order)

    theta, phi = hp.pix2ang(2 ** src_order, src_pixel_idx)
    dst_pixel_idx = hp.ang2pix(dst_nside, theta, phi)

    src_psi_deg = (360.0 / src_n_psi) * src_psi_idx
    dst_psi_idx = np.round(src_psi_deg / (360.0 / dst_n_psi)).astype(np.int64) % dst_n_psi

    return _combine_rotation_indices(dst_pixel_idx, dst_psi_idx, dst_order)


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
    nside_parent = 2**parent_nside_level
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
    fine_nside = 2**fine_nside_level
    theta, phi = hp.pix2ang(fine_nside, current_pixels)

    angle_res = 360 / (6 * 2**fine_nside_level)
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


def get_oversampled_rotation_grid_from_samples(
    parent_rotation_indices,
    parent_nside_level,
    oversampling_order=1,
    *,
    return_rotation_indices=False,
):
    """Generate oversampled child orientations from coarse sample indices.

    RELION oversamples each coarse orientation sample, not just the HEALPix
    direction. For a single oversampling level in 3D, each coarse sample
    expands into 4 child directions and 2 child in-plane angles, yielding
    ``8`` child orientations per parent sample.

    Parameters
    ----------
    parent_rotation_indices : array-like of int
        Indices into the coarse rotation grid. Each index corresponds to a
        specific ``(healpix_pixel, psi_index)`` sample.
    parent_nside_level : int
        HEALPix level of the coarse grid.
    oversampling_order : int
        Number of oversampling levels.

    Returns
    -------
    matrices : np.ndarray, shape (n_children, 3, 3)
        Oversampled child rotation matrices.
    parent_map : np.ndarray, shape (n_children,)
        Index into ``parent_rotation_indices`` for each child orientation.
    child_rotation_indices : np.ndarray, shape (n_children,), optional
        Nearest full-grid indices of the child orientations on the fine grid.
        RELION's oversampled psi children are midpoints inside the parent bin,
        so for 3D they are generally not exact rows of the global fine grid.
        Only returned when ``return_rotation_indices=True``.
    """
    parent_rotation_indices = np.asarray(parent_rotation_indices, dtype=np.int64)
    if parent_rotation_indices.size == 0:
        empty_rot = np.empty((0, 3, 3), dtype=np.float32)
        empty_map = np.empty((0,), dtype=np.int64)
        if return_rotation_indices:
            return empty_rot, empty_map, empty_map.copy()
        return empty_rot, empty_map

    coarse_nside = 2 ** parent_nside_level
    coarse_n_pixels = hp.nside2npix(coarse_nside)
    parent_pixels = parent_rotation_indices % coarse_n_pixels
    parent_psi = parent_rotation_indices // coarse_n_pixels

    current_pixels = parent_pixels.copy()
    parent_map = np.arange(len(parent_rotation_indices), dtype=np.int64)
    for level in range(oversampling_order):
        current_pixels = get_healpix_children(current_pixels, parent_nside_level + level)
        parent_map = np.repeat(parent_map, 4)

    psi_factor = 2**oversampling_order
    coarse_n_in_planes = rotation_grid_n_in_planes(parent_nside_level)
    coarse_psi_step = 2.0 * np.pi / coarse_n_in_planes
    fine_nside_level = parent_nside_level + oversampling_order
    fine_nside = 2**fine_nside_level
    fine_n_pixels = hp.nside2npix(fine_nside)
    fine_n_in_planes = rotation_grid_n_in_planes(fine_nside_level)
    fine_psi_step = 2.0 * np.pi / fine_n_in_planes

    theta, phi = hp.pix2ang(fine_nside, current_pixels)
    current_parent_psi = parent_psi[parent_map]
    # Match RELION's pushbackOversampledPsiAngles(): oversampled psi samples
    # are midpoints inside the parent psi bin, not rows of the fine global grid.
    psi_child_angles = (
        current_parent_psi[:, None] * coarse_psi_step
        - 0.5 * coarse_psi_step
        + (0.5 + np.arange(psi_factor, dtype=np.float64)[None, :])
        * (coarse_psi_step / psi_factor)
    )
    nearest_child_psi = np.floor(
        np.mod(psi_child_angles, 2.0 * np.pi) / fine_psi_step + 0.5
    ).astype(np.int64) % fine_n_in_planes

    child_pixels = np.repeat(current_pixels, psi_factor)
    child_rotation_indices = (
        nearest_child_psi.reshape(-1) * fine_n_pixels + child_pixels
    )

    euler_angles = np.stack(
        [
            np.repeat(theta, psi_factor),
            np.repeat(phi, psi_factor),
            psi_child_angles.reshape(-1),
        ],
        axis=-1,
    )
    euler_angles = euler_angles / (2 * np.pi) * 360
    matrices = utils.R_from_relion(euler_angles)
    parent_map = np.repeat(parent_map, psi_factor)

    if return_rotation_indices:
        return (
            matrices.astype(np.float32),
            parent_map,
            child_rotation_indices.astype(np.int64),
        )

    return matrices.astype(np.float32), parent_map


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
    n_subdiv = 2**oversampling_order  # per dimension
    fine_offset = pixel_offset / n_subdiv

    half_offsets = np.linspace(
        -pixel_offset / 2 + fine_offset / 2,
        pixel_offset / 2 - fine_offset / 2,
        n_subdiv,
    )
    dx, dy = np.meshgrid(half_offsets, half_offsets, indexing="ij")
    child_offsets = np.stack([dx.ravel(), dy.ravel()], axis=-1)  # (4**os, 2)
    n_children = n_subdiv**2

    fine_translations = (parent_translations[:, None, :] + child_offsets[None, :, :]).reshape(-1, 2)
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
    child_nside = 2**child_nside_level
    theta, phi = hp.pix2ang(child_nside, child_pixels)

    angle_res = 360 / (6 * 2**child_nside_level)
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


# ---------------------------------------------------------------------------
# Variable-order rotation grid
# ---------------------------------------------------------------------------


def get_rotation_grid_at_order(order, n_in_planes=None, matrices=True):
    """Generate HEALPix rotation grid at the specified order.

    Thin wrapper around :func:`get_rotation_grid` that makes the order
    parameter explicit and defaults to returning rotation matrices.

    Parameters
    ----------
    order : int
        HEALPix order (nside = 2^order).
    n_in_planes : int or None
        Number of in-plane rotation angles.  If None, derived from the
        HEALPix angular step (matching RELION convention).
    matrices : bool
        If True, return (N, 3, 3) rotation matrices.
        If False, return (N, 3) Euler angles in degrees.

    Returns
    -------
    np.ndarray
        Rotation matrices (N, 3, 3) or Euler angles (N, 3).
    """
    return get_rotation_grid(order, n_in_planes=n_in_planes, matrices=matrices)


# ---------------------------------------------------------------------------
# Local angular search
# ---------------------------------------------------------------------------


def _angular_distance_matrices(R1, R2):
    """Geodesic distance in radians between two sets of rotation matrices.

    Parameters
    ----------
    R1 : np.ndarray, shape (..., 3, 3)
    R2 : np.ndarray, shape (..., 3, 3)

    Returns
    -------
    np.ndarray, shape (...)
        Geodesic angle in radians, in [0, pi].
    """
    # R_rel = R1^T @ R2; angle = arccos((trace(R_rel) - 1) / 2)
    R_rel = np.einsum("...ij,...ik->...jk", R1, R2)
    # Trace along last two dims
    trace_val = np.trace(R_rel, axis1=-2, axis2=-1)
    # Clamp to [-1, 3] to handle numerical noise
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


def get_local_rotation_grid(
    prior_rotations,
    sigma_rot,
    grid_order=None,
    grid_rotations=None,
    sigma_cutoff=3.0,
):
    """Generate local rotation grids around prior orientations.

    For each prior rotation, selects neighbors from the full HEALPix grid
    that lie within ``sigma_cutoff * sigma_rot`` geodesic distance.
    Returns the UNION of all selected rotations (Approach A: batched union),
    plus a mapping from each grid rotation back to the prior indices
    it is a neighbor of, and the Gaussian prior weights.

    Parameters
    ----------
    prior_rotations : np.ndarray, shape (n_images, 3, 3)
        Per-image best rotation matrices from the previous iteration.
    sigma_rot : float
        Gaussian prior sigma in radians.
    grid_order : int or None
        HEALPix order for the search grid.  Exactly one of ``grid_order``
        or ``grid_rotations`` must be provided.
    grid_rotations : np.ndarray, shape (n_grid, 3, 3), or None
        Pre-computed rotation grid.  If provided, ``grid_order`` is ignored.
    sigma_cutoff : float
        Only include grid points within ``sigma_cutoff * sigma_rot``
        of at least one prior (default 3.0).

    Returns
    -------
    selected_rotations : np.ndarray, shape (n_selected, 3, 3)
        Union of local rotation grid points across all priors.
    selected_indices : np.ndarray, shape (n_selected,), dtype int
        Indices into the full grid for each selected rotation.
    prior_weights : np.ndarray, shape (n_images, n_selected)
        Gaussian prior weight for each (image, selected_rotation) pair.
        ``prior_weights[i, j] = exp(-d^2 / (2 * sigma_rot^2))``
        where d is the geodesic distance between prior_rotations[i] and
        selected_rotations[j].  Zero if beyond sigma_cutoff.

    Raises
    ------
    ValueError
        If neither ``grid_order`` nor ``grid_rotations`` is provided.
    """
    if grid_rotations is None and grid_order is None:
        raise ValueError("Exactly one of grid_order or grid_rotations must be provided")

    if grid_rotations is None:
        grid_rotations = get_rotation_grid(grid_order, matrices=True)
    grid_rotations = np.asarray(grid_rotations, dtype=np.float64)

    prior_rotations = np.asarray(prior_rotations, dtype=np.float64)
    if prior_rotations.ndim == 2:
        prior_rotations = prior_rotations[np.newaxis]

    n_images = prior_rotations.shape[0]
    n_grid = grid_rotations.shape[0]
    cutoff_rad = sigma_cutoff * sigma_rot

    # Compute pairwise geodesic distances: (n_images, n_grid)
    # For memory efficiency with large grids, process in chunks
    CHUNK = 5000
    selected_mask = np.zeros(n_grid, dtype=bool)

    # First pass: find the union of all neighbors
    for i_start in range(0, n_images, CHUNK):
        i_end = min(i_start + CHUNK, n_images)
        priors_chunk = prior_rotations[i_start:i_end]  # (chunk, 3, 3)

        # (chunk, n_grid): geodesic distance
        dists = _angular_distance_matrices(
            priors_chunk[:, np.newaxis, :, :],
            grid_rotations[np.newaxis, :, :, :],
        )
        within = np.any(dists <= cutoff_rad, axis=0)
        selected_mask |= within

    selected_indices = np.where(selected_mask)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        # Fallback: if sigma is too tight, return the full grid
        selected_indices = np.arange(n_grid)
        n_selected = n_grid

    selected_rotations = grid_rotations[selected_indices]  # (n_selected, 3, 3)

    # Second pass: compute prior weights for the selected subset
    prior_weights = np.zeros((n_images, n_selected), dtype=np.float64)

    for i_start in range(0, n_images, CHUNK):
        i_end = min(i_start + CHUNK, n_images)
        priors_chunk = prior_rotations[i_start:i_end]  # (chunk, 3, 3)

        dists = _angular_distance_matrices(
            priors_chunk[:, np.newaxis, :, :],
            selected_rotations[np.newaxis, :, :, :],
        )  # (chunk, n_selected)

        weights = np.exp(-dists**2 / (2.0 * sigma_rot**2))
        weights[dists > cutoff_rad] = 0.0
        prior_weights[i_start:i_end] = weights

    return (
        selected_rotations.astype(np.float32),
        selected_indices,
        prior_weights.astype(np.float32),
    )


def get_local_rotation_grid_fast(
    prior_rotation_indices,
    sigma_rot,
    sigma_psi,
    healpix_order,
    sigma_cutoff=3.0,
    *,
    per_image=False,
):
    """Fast local rotation grid selection using HEALPix pixel lookup.

    Accepts either discrete full-grid rotation indices or exact prior rotation
    matrices. This matters for RELION-parity local search: adaptive pass-2
    orientations use midpoint psi children inside the parent bin, so they do
    not generally coincide with rows of the global fine HEALPix x psi grid.

    The prior is factored into independent direction and psi terms, matching
    RELION's ``P(idir) * P(ipsi)`` structure.

    Parameters
    ----------
    prior_rotation_indices : np.ndarray
        Either full-grid rotation indices of shape ``(n_priors,)`` or exact
        prior rotation matrices of shape ``(n_priors, 3, 3)``.
    sigma_rot : float
        Gaussian prior sigma for direction (radians).
    sigma_psi : float
        Gaussian prior sigma for in-plane angle (radians).
    healpix_order : int
        HEALPix order (nside = 2^order) of the rotation grid.
    sigma_cutoff : float
        Include grid points within ``sigma_cutoff * sigma`` of at least
        one prior (default 3.0).

    Returns
    -------
    selected_indices : np.ndarray, shape (n_selected,), dtype int
        Sorted indices into the full rotation grid.
    rotation_log_prior : np.ndarray
        If ``per_image=False`` (default), shape ``(n_selected,)`` and each
        rotation uses the max over all priors of
        ``-d_dir^2/(2*sigma_rot^2) - d_psi^2/(2*sigma_psi^2)``.
        If ``per_image=True``, shape ``(n_priors, n_selected)`` with an exact
        per-image log-prior over the union grid.
    """
    prior_rotation_indices = np.asarray(prior_rotation_indices)

    # --- Reconstruct grid geometry ---
    nside = 2 ** healpix_order
    n_pixels = hp.nside2npix(nside)
    n_psi = rotation_grid_n_in_planes(healpix_order)
    n_total = n_psi * n_pixels

    # Grid layout: index k -> psi_index = k // n_pixels, pixel = k % n_pixels
    # (from meshgrid(arange(n_pixels), psi_angles) then reshape(-1, 3))
    psi_angles = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)  # radians

    if prior_rotation_indices.ndim == 0:
        prior_rotation_indices = prior_rotation_indices.reshape(1)

    if prior_rotation_indices.ndim == 1:
        prior_rotations = rotation_indices_to_matrices(
            prior_rotation_indices.astype(np.int64),
            healpix_order,
        )
    else:
        prior_rotations = np.asarray(
            prior_rotation_indices,
            dtype=np.float64,
        ).reshape(-1, 3, 3)

    prior_eulers_deg = utils.R_to_relion(prior_rotations, degrees=True)
    # CORRECT view direction for each prior matrix.
    #
    # Recovar's grid construction (rotation_indices_to_matrices) passes
    # angles to R_from_relion in [theta_healpy, phi_healpy, psi] order. The
    # matrix at HEALPix pixel p has its R_to_relion[0] (= "rot") equal to
    # theta_healpy (the polar angle from pix2ang) and R_to_relion[1] (= "tilt")
    # equal to phi_healpy (azimuthal). This is OPPOSITE to the standard
    # RELION convention (rot=azim, tilt=polar). Internally consistent but
    # makes the relationship between HEALPix pixels and matrix view
    # directions non-standard.
    #
    # The actual VIEW DIRECTION of a matrix M with R_to_relion(M) = (a, b, c),
    # using the standard formula `(sin(tilt)*cos(rot), sin(tilt)*sin(rot),
    # cos(tilt))`, is `(sin(b)*cos(a), sin(b)*sin(a), cos(b))`. This vector
    # corresponds to the actual physical view of the projection.
    #
    # For local search, we want to find grid matrices whose view directions
    # are near the prior view direction (in sphere distance). Because of the
    # SWAP in the grid, the HEALPix pixel index is NOT a good proxy for view
    # direction, so HEALPix `query_disc` cannot be used directly. Instead, we
    # precompute the view directions of ALL grid matrices (one-time per
    # healpix_order) and find pixels within the cone via direct dot product.
    # This was a parity bug fix for Task #100 on 2026-04-09: the previous
    # implementation used HEALPix `query_disc` with the prior view as the
    # query vector, but this returns pixels at HEALPix coordinates near the
    # view direction, NOT pixels whose matrices have view directions near
    # the prior. The result was that the local cone was systematically
    # centered ~70-80 degrees AWAY from the prior pose for typical priors,
    # causing recovar's iter-6 noise update to spike (resolution regression
    # at the global→local transition).
    prior_rot_deg = prior_eulers_deg[:, 0]
    prior_tilt_deg = prior_eulers_deg[:, 1]
    prior_psi = np.deg2rad(prior_eulers_deg[:, 2])

    # Standard view direction for each prior (RELION's Euler_angles2direction)
    prior_dir_vecs = np.column_stack(
        [
            np.sin(np.deg2rad(prior_tilt_deg)) * np.cos(np.deg2rad(prior_rot_deg)),
            np.sin(np.deg2rad(prior_tilt_deg)) * np.sin(np.deg2rad(prior_rot_deg)),
            np.cos(np.deg2rad(prior_tilt_deg)),
        ]
    )

    # --- Direction neighbors via direct dot-product ---
    # Precompute view directions for ALL grid matrices at this order. This
    # is the correct way to find pixels with matrix view directions near
    # the prior, given recovar's non-standard rot/tilt convention. O(n_pix)
    # per call, fast enough for chunks of <100 priors at order ≤4 (3072
    # directions × 96 psi = 295k matrices, but we only need 3072 view
    # directions since psi doesn't affect the view).
    all_pixel_indices = np.arange(n_pixels, dtype=np.int64)
    grid_eulers_deg = np.empty((n_pixels, 3), dtype=np.float64)
    grid_theta, grid_phi = hp.pix2ang(nside, all_pixel_indices)
    grid_eulers_deg[:, 0] = np.rad2deg(grid_theta)  # stored as "rot" (recovar convention)
    grid_eulers_deg[:, 1] = np.rad2deg(grid_phi)    # stored as "tilt" (recovar convention)
    grid_eulers_deg[:, 2] = 0.0
    # Standard view direction formula on the recovar grid eulers
    grid_view_dirs = np.column_stack(
        [
            np.sin(np.deg2rad(grid_eulers_deg[:, 1])) * np.cos(np.deg2rad(grid_eulers_deg[:, 0])),
            np.sin(np.deg2rad(grid_eulers_deg[:, 1])) * np.sin(np.deg2rad(grid_eulers_deg[:, 0])),
            np.cos(np.deg2rad(grid_eulers_deg[:, 1])),
        ]
    )

    dir_cutoff_rad = sigma_cutoff * sigma_rot
    cos_cutoff = np.cos(dir_cutoff_rad)

    # Collect selected direction pixels: those whose view direction is
    # within `dir_cutoff_rad` of any prior view direction (sphere distance).
    selected_pixel_set = set()
    if sigma_rot > 0:
        # cos(angle) >= cos(cutoff) iff angle <= cutoff (since cos decreases on [0, pi])
        for prior_vec in prior_dir_vecs:
            dots = grid_view_dirs @ prior_vec  # (n_pixels,)
            in_cone = dots >= cos_cutoff
            in_cone_pix = np.nonzero(in_cone)[0]
            selected_pixel_set.update(in_cone_pix.tolist())
    else:
        selected_pixel_set.update(range(n_pixels))
    selected_pixels = np.array(sorted(selected_pixel_set), dtype=np.int64)
    if selected_pixels.size == 0:
        # Fallback: pick the closest pixel for each prior
        for prior_vec in prior_dir_vecs:
            dots = grid_view_dirs @ prior_vec
            best = int(np.argmax(dots))
            selected_pixel_set.add(best)
        selected_pixels = np.array(sorted(selected_pixel_set), dtype=np.int64)

    # --- Psi neighbors via circular distance ---
    psi_cutoff_rad = sigma_cutoff * sigma_psi

    # For each psi in the grid, check circular distance to nearest prior psi
    # Circular distance: min(|a-b|, 2*pi - |a-b|)
    selected_psi_set = set()
    if sigma_psi > 0:
        for psi_val in prior_psi:
            diffs = np.abs(psi_angles - psi_val)
            circ_dists = np.minimum(diffs, 2 * np.pi - diffs)
            within = np.where(circ_dists <= psi_cutoff_rad)[0]
            selected_psi_set.update(within.tolist())
    else:
        selected_psi_set.update(range(n_psi))
    selected_psi_idx = np.array(sorted(selected_psi_set), dtype=np.int64)
    if selected_psi_idx.size == 0:
        diffs = np.abs(psi_angles[:, None] - prior_psi[None, :])
        circ_dists = np.minimum(diffs, 2 * np.pi - diffs)
        selected_psi_idx = np.unique(np.argmin(circ_dists, axis=0).astype(np.int64))

    # --- Combine: selected_rotations = selected_psi x selected_pixels ---
    # Grid index = psi_index * n_pixels + pixel_index
    psi_grid, pix_grid = np.meshgrid(selected_psi_idx, selected_pixels, indexing='ij')
    selected_indices = (psi_grid * n_pixels + pix_grid).ravel()
    selected_indices.sort()

    if len(selected_indices) == 0:
        # Fallback: return all
        selected_indices = np.arange(n_total, dtype=np.int64)

    # --- Compute Gaussian log-prior (decomposed) ---
    # Direction and psi priors are independent (RELION convention):
    #   log_prior(r) = log_prior_dir(pixel_r) + log_prior_psi(psi_r)
    # where each component takes the max over all priors independently.
    # This avoids the O(n_priors * n_selected) joint computation.

    # 1. Direction component.
    unique_sel_pixels = np.unique(selected_pixels)  # already computed above

    # IMPORTANT: use grid_view_dirs (matrix view directions), NOT
    # hp.pix2vec(p) (HEALPix pixel center). Recovar's grid construction
    # has rot/tilt swapped relative to standard, so the matrix's view
    # direction is NOT at the HEALPix pixel center. See the comment at
    # the top of this function for details. (Task #100 fix.)
    sel_pix_vecs = grid_view_dirs[unique_sel_pixels]  # (n_usp, 3)

    n_usp = len(unique_sel_pixels)
    CHUNK_PIX = 5000
    if per_image:
        if sigma_rot > 0:
            log_prior_dir_u = np.empty(
                (prior_dir_vecs.shape[0], n_usp), dtype=np.float64,
            )
            for s in range(0, n_usp, CHUNK_PIX):
                e = min(s + CHUNK_PIX, n_usp)
                dots = prior_dir_vecs @ sel_pix_vecs[s:e].T  # (n_priors, chunk)
                np.clip(dots, -1.0, 1.0, out=dots)
                d = np.arccos(dots)
                log_vals = -d ** 2 / (2.0 * sigma_rot ** 2)
                log_vals[d > dir_cutoff_rad] = -1e30
                log_prior_dir_u[:, s:e] = log_vals
        else:
            log_prior_dir_u = np.zeros(
                (prior_dir_vecs.shape[0], n_usp), dtype=np.float64,
            )
    else:
        min_d_dir_sq = np.full(n_usp, np.inf, dtype=np.float64)
        for s in range(0, n_usp, CHUNK_PIX):
            e = min(s + CHUNK_PIX, n_usp)
            dots = sel_pix_vecs[s:e] @ prior_dir_vecs.T  # (chunk, n_priors)
            np.clip(dots, -1.0, 1.0, out=dots)
            d = np.arccos(dots)  # (chunk, n_priors)
            np.minimum(min_d_dir_sq[s:e], np.min(d ** 2, axis=1), out=min_d_dir_sq[s:e])

    # Map unique pixel distances back to all selected rotations
    # Build pixel -> unique index map
    pix_to_uidx = np.empty(n_pixels, dtype=np.int64)
    pix_to_uidx[unique_sel_pixels] = np.arange(n_usp)

    sel_pixels = selected_indices % n_pixels
    sel_psi_idx = selected_indices // n_pixels
    if per_image:
        log_prior_dir = log_prior_dir_u[:, pix_to_uidx[sel_pixels]]
    else:
        log_prior_dir = -min_d_dir_sq[pix_to_uidx[sel_pixels]] / (2.0 * sigma_rot ** 2)

    # 2. Psi component.
    unique_sel_psi = np.unique(selected_psi_idx)  # already computed above
    sel_psi_vals = psi_angles[unique_sel_psi]

    d_psi_raw = np.abs(sel_psi_vals[:, None] - prior_psi[None, :])
    d_psi = np.minimum(d_psi_raw, 2 * np.pi - d_psi_raw)

    # Map back to all selected rotations
    psi_to_uidx = np.empty(n_psi, dtype=np.int64)
    psi_to_uidx[unique_sel_psi] = np.arange(len(unique_sel_psi))
    if per_image:
        if sigma_psi > 0:
            log_prior_psi_u = -d_psi.T ** 2 / (2.0 * sigma_psi ** 2)
            log_prior_psi_u[d_psi.T > psi_cutoff_rad] = -1e30
        else:
            log_prior_psi_u = np.zeros(
                (prior_rotations.shape[0], len(unique_sel_psi)), dtype=np.float64,
            )
        log_prior_psi = log_prior_psi_u[:, psi_to_uidx[sel_psi_idx]]
    else:
        min_d_psi_sq = np.min(d_psi ** 2, axis=1)  # (n_unique_sel_psi,)
        log_prior_psi = -min_d_psi_sq[psi_to_uidx[sel_psi_idx]] / (2.0 * sigma_psi ** 2)

    # Total log-prior
    log_prior = log_prior_dir + log_prior_psi

    # Apply cutoff: keep only rotations where both direction and psi
    # are within sigma_cutoff of some prior
    min_log_prior = -(sigma_cutoff ** 2)  # sum of two (sigma_cutoff^2/2) terms
    keep = np.any(log_prior > min_log_prior, axis=0) if per_image else (log_prior > min_log_prior)
    if np.any(keep):
        selected_indices = selected_indices[keep]
        log_prior = log_prior[:, keep] if per_image else log_prior[keep]

    if len(selected_indices) == 0:
        selected_indices = np.arange(n_total, dtype=np.int64)
        if per_image:
            log_prior = np.zeros((prior_rotations.shape[0], n_total), dtype=np.float32)
        else:
            log_prior = np.zeros(n_total, dtype=np.float32)

    return selected_indices, log_prior.astype(np.float32)


def get_healpix_neighbors(pixel_idx, nside_level, n_neighbors=8):
    """Return the neighbor HEALPix pixel indices for a given pixel.

    Uses healpy's ``get_all_neighbours`` which returns 8 neighboring pixels
    (or fewer at poles, with -1 for missing neighbors).

    Parameters
    ----------
    pixel_idx : int or array-like
        RING-ordered HEALPix pixel index (or array of indices).
    nside_level : int
        HEALPix level (nside = 2^nside_level).

    Returns
    -------
    np.ndarray
        Neighbor pixel indices.  Shape (8,) for scalar input or
        (8, n_pixels) for array input.  Missing neighbors are -1.
    """
    nside = 2 ** nside_level
    return hp.get_all_neighbours(nside, np.atleast_1d(pixel_idx))


@functools.partial(jax.jit, static_argnums=[1])
def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    indices = translations + image_shape[0] // 2
    vec_indices = indices[..., 1] * image_shape[1] + indices[..., 0]
    return vec_indices
