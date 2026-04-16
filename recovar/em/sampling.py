import functools
import jax
import healpy as hp
import numpy as np
from recovar import utils

# Cache of full-grid rotation matrices keyed by healpix_order. Each entry has
# shape (n_pixels * n_psi, 3, 3) and is reused across calls to
# get_local_rotation_grid_fast to avoid rebuilding the grid every chunk.
# Cleared automatically when Python exits; we don't bother with manual eviction
# because there are at most a handful of healpix orders in active use.
_GRID_MATRIX_CACHE: "dict[int, np.ndarray]" = {}


def _get_full_grid_matrices(healpix_order: int) -> np.ndarray:
    """Return all (n_pixels * n_psi, 3, 3) rotation matrices for one order.

    Cached because the only inputs are integers (healpix_order) and the
    output is deterministic. Used by ``get_local_rotation_grid_fast`` to
    do axis-angle cone selection without rebuilding the grid each call.
    """
    cached = _GRID_MATRIX_CACHE.get(int(healpix_order))
    if cached is not None:
        return cached
    n_total = rotation_grid_size(int(healpix_order))
    mats = rotation_indices_to_matrices(
        np.arange(n_total, dtype=np.int64), int(healpix_order),
    )
    _GRID_MATRIX_CACHE[int(healpix_order)] = mats
    return mats


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


def relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0):
    """RELION's getAngularSampling() for 3D: 360 / (6 * 2^(order+adaptive_oversampling)).

    Ref: healpix_sampling.cpp:1589-1598.
    """
    order = int(healpix_order) + int(adaptive_oversampling)
    return 360.0 / (6 * 2**order)


def advance_relion_perturbation(prev_random_perturbation, perturbation_factor, rng):
    """Update the RELION per-iteration perturbation state.

    Ports `HealpixSampling::resetRandomlyPerturbedSampling` (healpix_sampling.cpp:167-174):
        random_perturbation += rnd_unif(0.5 * perturbation_factor, perturbation_factor)
        random_perturbation = realWRAP(random_perturbation, -pf, +pf)

    Parameters
    ----------
    prev_random_perturbation : float
        Previous iteration's random_perturbation (initialize to 0.0 before iter 1).
    perturbation_factor : float
        Typically 0.5 (RELION default `--perturb 0.5`).
    rng : np.random.Generator

    Returns
    -------
    float : new random_perturbation, wrapped to [-pf, +pf].
    """
    pf = float(perturbation_factor)
    new = prev_random_perturbation + rng.uniform(0.5 * pf, pf)
    # realWRAP — map to [-pf, +pf]
    while new > pf:
        new -= 2 * pf
    while new < -pf:
        new += 2 * pf
    return float(new)


def apply_relion_rotation_perturbation(rotations, random_perturbation, angular_sampling_deg):
    """Port of RELION's 3D grid perturbation (healpix_sampling.cpp:1909-1934).

    For each rotation matrix A in `rotations`, replace it with ``A @ R_perturb``
    where ``R_perturb = R_from_relion([myperturb, myperturb, myperturb])`` and
    ``myperturb = random_perturbation * angular_sampling_deg``. Applied AFTER
    oversampling, before scoring.

    Parameters
    ----------
    rotations : np.ndarray, shape (N, 3, 3)
    random_perturbation : float
        Current iteration's value in [-pf, +pf].
    angular_sampling_deg : float
        The nominal step at the coarse healpix order (pre-oversampling) in degrees.
        Use ``relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0)``.

    Returns
    -------
    np.ndarray, shape (N, 3, 3)
    """
    if abs(random_perturbation) < 1e-12:
        return rotations
    myperturb = float(random_perturbation) * float(angular_sampling_deg)
    R_perturb = utils.R_from_relion(np.array([[myperturb, myperturb, myperturb]], dtype=np.float64))[0]
    R_perturb = R_perturb.astype(rotations.dtype)
    # RELION: A = A * R  (right multiply, each matrix independently)
    return np.einsum("nij,jk->nik", rotations, R_perturb)


def apply_relion_translation_perturbation(translations, random_perturbation, offset_step_pixels):
    """Port of RELION's translation perturbation (healpix_sampling.cpp:1810-1820).

    Adds ``myperturb = random_perturbation * offset_step_pixels`` to both axes
    of each translation vector.
    """
    if abs(random_perturbation) < 1e-12:
        return translations
    myperturb = float(random_perturbation) * float(offset_step_pixels)
    return translations + np.asarray(myperturb, dtype=translations.dtype)


def read_relion_perturbation_from_sampling_star(sampling_star_path):
    """Read _rlnSamplingPerturbInstance and _rlnSamplingPerturbFactor from a RELION sampling.star.

    Used for exact parity replay: feed recovar the same perturbation RELION used at iter N.

    Returns
    -------
    (random_perturbation, perturbation_factor) : tuple of float
    """
    import re

    text = open(sampling_star_path).read()
    m_inst = re.search(r"_rlnSamplingPerturbInstance\s+(\S+)", text)
    m_fac = re.search(r"_rlnSamplingPerturbFactor\s+(\S+)", text)
    if not m_inst or not m_fac:
        raise ValueError(f"Missing perturb fields in {sampling_star_path}")
    return float(m_inst.group(1)), float(m_fac.group(1))


def read_relion_sampling_metadata(sampling_star_path):
    """Read the full set of RELION sampling metadata needed for replay:
    ``(random_perturbation, perturbation_factor, healpix_order, offset_range, offset_step)``.

    ``offset_range`` and ``offset_step`` are in the same units RELION writes
    (Angstroms at scale-0, or as configured). ``healpix_order`` is the order
    RELION actually used at that iter.
    """
    import re

    text = open(sampling_star_path).read()

    def _grab(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            raise ValueError(f"Missing {name} in {sampling_star_path}")
        return cast(m.group(1))

    return dict(
        random_perturbation=_grab("rlnSamplingPerturbInstance"),
        perturbation_factor=_grab("rlnSamplingPerturbFactor"),
        healpix_order=_grab("rlnHealpixOrder", int),
        offset_range=_grab("rlnOffsetRange"),
        offset_step=_grab("rlnOffsetStep"),
    )


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
    """Local rotation grid selection via SO(3) axis-angle distance.

    For each prior rotation matrix, find every grid rotation within
    ``sigma_cutoff * max(sigma_rot, sigma_psi)`` axis-angle distance.

    **Implementation note (Task #101 fix, 2026-04-09):** earlier versions
    of this function tried to factor the cone selection into independent
    direction (HEALPix pixel) and psi (in-plane angle) components, then
    score each candidate via ``log_prior_dir + log_prior_psi``. That
    approach made two distinct mistakes:

    1. The "direction" of a recovar grid matrix is **not** the standard
       ZYZ view formula `(sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt))`.
       Recovar's `R_from_relion` uses an extrinsic ZXZ convention with
       offsets `[rot+90, tilt, psi-90]` and a `[[1,-1,1],[-1,1,-1],[1,-1,1]]`
       frame-adjust multiply (`recovar/utils/helpers.py:717`). The actual
       view direction is::

           view = (-cos(psi)*sin(tilt), sin(psi)*sin(tilt), cos(tilt))

       so it depends on **psi**, not just `(rot, tilt)`. The Task #100
       fix used the standard formula and got x and y wrong; that left
       ~49% of priors with <10% cone overlap, including some priors with
       0 overlap (the cone landed on the antipodal set).

    2. RELION factors the prior into `P(direction) * P(psi)`, but it
       still requires both diffang and diffpsi to be small individually.
       Trying to mimic this with a recovar-specific direction definition
       compounds the bug above.

    Both problems disappear when we work directly in SO(3): the axis-angle
    distance between two rotation matrices is `arccos((trace(R1^T R2) - 1)/2)`,
    invariant under the Euler convention. We compute it for every (prior,
    grid_matrix) pair using a single batched einsum and select the
    in-cone subset. The full grid of matrices is cached per healpix_order
    so the cost is one matmul per call (~64 priors × 295k matrices at
    order 4, well under a millisecond on GPU/CPU).

    The log-prior is computed as a single Gaussian in axis-angle distance
    with sigma `max(sigma_rot, sigma_psi)`, which matches RELION's
    "biggest_sigma" convention in `selectOrientationsWithNonZeroPriorProbability`
    (`healpix_sampling.cpp:769`). This is NOT a perfect match to RELION's
    factored direction × psi prior, but the SELECTION is now correct
    regardless of convention, which is the dominant source of error in
    practice.

    Parameters
    ----------
    prior_rotation_indices : np.ndarray
        Either full-grid rotation indices of shape ``(n_priors,)`` or
        exact prior rotation matrices of shape ``(n_priors, 3, 3)``.
    sigma_rot : float
        Gaussian prior sigma for rotation, **radians**. Used as the
        cone radius scale and the log-prior denominator.
    sigma_psi : float
        Gaussian prior sigma for in-plane angle, **radians**. Combined
        with ``sigma_rot`` via ``max(sigma_rot, sigma_psi)`` to match
        RELION's `biggest_sigma`.
    healpix_order : int
        HEALPix order (nside = 2^order) of the rotation grid.
    sigma_cutoff : float
        Include grid points within ``sigma_cutoff * max(sigma_rot, sigma_psi)``
        SO(3) distance of at least one prior (default 3.0).
    per_image : bool
        When True, return a per-image log-prior of shape
        ``(n_priors, n_selected)``. When False, collapse to the max over
        priors per grid index, shape ``(n_selected,)``.

    Returns
    -------
    selected_indices : np.ndarray, shape (n_selected,), dtype int
        Sorted indices into the full rotation grid.
    rotation_log_prior : np.ndarray
        Per-image (or aggregated) Gaussian log-prior over the selected
        union, with out-of-cone entries set to ``-1e30``.
    """
    prior_rotation_indices = np.asarray(prior_rotation_indices)

    # --- Reconstruct grid geometry (still needed for fallbacks) ---
    n_total = rotation_grid_size(healpix_order)

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

    n_priors = prior_rotations.shape[0]
    prior_rotations = prior_rotations.astype(np.float64)

    # --- Axis-angle cone selection ---
    # Combined sigma matches RELION's `biggest_sigma = max(sigma_rot, sigma_tilt)`
    # at healpix_sampling.cpp:769. The cone is in SO(3) axis-angle distance.
    biggest_sigma = float(max(sigma_rot, sigma_psi))
    if biggest_sigma <= 0:
        # No cone -> include all rotations with zero log-prior.
        selected_indices = np.arange(n_total, dtype=np.int64)
        if per_image:
            log_prior = np.zeros((n_priors, n_total), dtype=np.float32)
        else:
            log_prior = np.zeros(n_total, dtype=np.float32)
        return selected_indices, log_prior

    cone_rad = float(sigma_cutoff) * biggest_sigma

    # Cached full-grid matrices for this healpix order.
    all_grid_mats = _get_full_grid_matrices(healpix_order)  # (n_total, 3, 3)
    n_total_grid = all_grid_mats.shape[0]

    # For each prior P, compute axis-angle distance to every grid matrix G:
    #   d(P, G) = arccos((trace(P^T @ G) - 1) / 2)
    #
    # Key trick: trace(P^T @ G) = sum_{i,j} P[i,j] * G[i,j], i.e. the
    # Frobenius inner product. So all (n_priors, n_total) traces collapse
    # to a single (n_priors, 9) @ (9, n_total) matmul over the flattened
    # 3x3 matrix entries — no need to form (n_priors, n_total, 3, 3).
    inv_two_sigma_sq = 1.0 / (2.0 * biggest_sigma * biggest_sigma)
    cos_cutoff = float(np.cos(cone_rad))  # cos is monotonically decreasing

    priors_flat = prior_rotations.reshape(n_priors, 9)            # (n_priors, 9)
    grid_flat = all_grid_mats.reshape(n_total_grid, 9)            # (n_total, 9)
    traces_all = priors_flat @ grid_flat.T                        # (n_priors, n_total)
    cos_arg_all = (traces_all - 1.0) * 0.5
    # Mask without arccos: angle <= cone_rad <=> cos(angle) >= cos(cone_rad).
    in_cone_all = cos_arg_all >= cos_cutoff                       # (n_priors, n_total)
    union_in_cone = np.any(in_cone_all, axis=0)                   # (n_total,)
    selected_indices = np.flatnonzero(union_in_cone).astype(np.int64)

    if selected_indices.size == 0:
        # Fallback: pick the single closest grid matrix for each prior so
        # we never return an empty selection.
        best_per_prior = np.argmax(cos_arg_all, axis=1)            # (n_priors,)
        selected_indices = np.unique(best_per_prior).astype(np.int64)

    # Restrict cos_arg to the selected union and compute the log-prior only
    # there. arccos is the costly trig op; do it once on the small slice.
    cos_arg_sel = cos_arg_all[:, selected_indices]                 # (n_priors, n_selected)
    np.clip(cos_arg_sel, -1.0, 1.0, out=cos_arg_sel)
    angles_sel = np.arccos(cos_arg_sel)
    log_prior = np.where(
        cos_arg_sel >= cos_cutoff,
        -(angles_sel ** 2) * inv_two_sigma_sq,
        -1e30,
    )

    if per_image:
        out_log_prior = log_prior.astype(np.float32)
    else:
        # Per-grid max log-prior across priors. Out-of-cone entries are
        # -1e30 in every row, so the max over n_priors stays -1e30 for
        # rotations no prior has selected. After the union restriction
        # above, every selected index has at least one prior with a
        # finite value, so the max is well-defined.
        out_log_prior = np.max(log_prior, axis=0).astype(np.float32)

    return selected_indices, out_log_prior


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
