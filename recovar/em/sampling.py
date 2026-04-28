import functools

import healpy as hp
import jax
import numpy as np

from recovar import utils

# Cached per-order geometry used by the exact RELION local-search selector.
# For the RELION-parity grid, the flattened index is ``psi_idx * n_pixels +
# pixel_idx`` with ``pixel_idx`` following RELION's NEST-ordered HEALPix
# enumeration.
_GRID_METADATA_CACHE: "dict[int, dict[str, np.ndarray]]" = {}


def _get_relion_grid_metadata(healpix_order: int) -> dict[str, np.ndarray]:
    """Return cached ring-order HEALPix geometry for one rotation-grid order."""
    healpix_order = int(healpix_order)
    cached = _GRID_METADATA_CACHE.get(healpix_order)
    if cached is not None:
        return cached

    from recovar.relion_bind._relion_bind_core import get_healpix_directions

    directions = np.asarray(get_healpix_directions(healpix_order), dtype=np.float32)
    n_pixels = int(directions.shape[0])
    n_psi = rotation_grid_n_in_planes(healpix_order)
    # Use the actual matrix view directions of the RELION grid rather than a
    # closed-form HEALPix angle formula. This keeps the local-search selector
    # aligned with the trial rotations that are actually scored.
    rot_deg = np.asarray(directions[:, 0], dtype=np.float32)
    tilt_deg = np.asarray(directions[:, 1], dtype=np.float32)
    psi_step = 360.0 / float(max(1, n_psi))
    psi_deg = (np.arange(n_psi, dtype=np.float32) * np.float32(psi_step)).astype(np.float32, copy=False)
    # RELION's viewing direction is the third ROW of the rotation matrix,
    # not the third column.
    dir_eulers = np.column_stack([rot_deg, tilt_deg, np.zeros(n_pixels, dtype=np.float32)])
    dir_rotations = utils.R_from_relion(dir_eulers, degrees=True)
    dir_vecs = np.asarray(dir_rotations[:, 2, :], dtype=np.float32)
    dir_norm = np.linalg.norm(dir_vecs, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm > 0.0, dir_norm, 1.0)
    dir_vecs = dir_vecs / dir_norm

    cached = {
        "rot_deg": rot_deg,
        "tilt_deg": tilt_deg,
        "dir_vecs": dir_vecs,
        "psi_deg": psi_deg,
        "n_pixels": np.asarray(n_pixels, dtype=np.int64),
        "n_psi": np.asarray(n_psi, dtype=np.int64),
    }
    _GRID_METADATA_CACHE[healpix_order] = cached
    return cached


def _wrapped_abs_diff_deg(values_deg: np.ndarray, ref_deg: np.ndarray | float) -> np.ndarray:
    """Circular absolute difference in degrees, wrapped to [0, 180]."""
    diff = np.abs(np.asarray(values_deg, dtype=np.float64) - np.asarray(ref_deg, dtype=np.float64))
    return np.where(diff > 180.0, np.abs(diff - 360.0), diff)


def _normalized_log_weights(diff_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Return log Gaussian weights normalized to sum to one."""
    if sigma_deg <= 0.0:
        out = np.full(diff_deg.shape, -np.log(max(diff_deg.size, 1)), dtype=np.float64)
        return out.astype(np.float32)

    weights = np.exp(-0.5 * (np.asarray(diff_deg, dtype=np.float64) / float(sigma_deg)) ** 2)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights.fill(1.0 / max(weights.size, 1))
    else:
        weights /= total
    return np.log(np.clip(weights, np.finfo(np.float32).tiny, None)).astype(np.float32)


def relion_psi_from_rotation_matrices(rotations: np.ndarray) -> np.ndarray:
    """Extract RELION psi angles from rotation matrices.

    This avoids a full SciPy ZXZ decomposition for the common non-singular
    case. Rows near the ZXZ singularity fall back to ``utils.R_to_relion``.
    """
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    frame_adjust = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64)
    adjusted = rotations * frame_adjust

    psi = np.empty(rotations.shape[0], dtype=np.float64)
    xy_norm = np.hypot(adjusted[:, 0, 2], adjusted[:, 1, 2])
    nonsingular = xy_norm > 1e-12

    if np.any(nonsingular):
        psi[nonsingular] = np.rad2deg(np.arctan2(adjusted[nonsingular, 0, 2], -adjusted[nonsingular, 1, 2])) + 90.0
    if np.any(~nonsingular):
        psi[~nonsingular] = utils.R_to_relion(rotations[~nonsingular], degrees=True)[:, 2]

    psi = (psi + 180.0) % 360.0 - 180.0
    return psi.astype(np.float32)


def build_local_search_grid_metadata(
    healpix_order: int,
    grid_eulers: np.ndarray | None = None,
    *,
    grid_rotations: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Prepare local-search metadata for either the canonical or a custom grid.

    When ``grid_eulers`` is provided, it must be a full-grid table in the same
    flattened index order as ``get_relion_rotation_grid_eulers`` / the live
    trial grid. If the grid still factorizes into independent direction and psi
    axes, return factorized metadata. Otherwise, fall back to full per-rotation
    metadata.
    """
    healpix_order = int(healpix_order)
    if grid_eulers is None:
        meta = _get_relion_grid_metadata(healpix_order)
        return {
            "mode": "factorized",
            "rot_deg": np.asarray(meta["rot_deg"], dtype=np.float32),
            "tilt_deg": np.asarray(meta["tilt_deg"], dtype=np.float32),
            "dir_vecs": np.asarray(meta["dir_vecs"], dtype=np.float32),
            "psi_deg": np.asarray(meta["psi_deg"], dtype=np.float32),
            "n_pixels": np.asarray(meta["n_pixels"], dtype=np.int64),
            "n_psi": np.asarray(meta["n_psi"], dtype=np.int64),
        }

    n_pixels = hp.nside2npix(2**healpix_order)
    n_psi = rotation_grid_n_in_planes(healpix_order)
    expected = n_pixels * n_psi
    if grid_rotations is not None:
        grid_rotations_full = np.asarray(grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
        if grid_rotations_full.shape[0] != expected:
            raise ValueError(
                f"grid_rotations must have shape ({expected}, 3, 3) for healpix_order={healpix_order}, "
                f"got {grid_rotations_full.shape}",
            )
    else:
        grid_rotations_full = None

    if grid_eulers is None:
        if grid_rotations_full is None:
            raise ValueError("grid_eulers or grid_rotations must be provided for custom local-search metadata")
        dir_vecs_full = np.asarray(grid_rotations_full[:, 2, :], dtype=np.float32)
        dir_norm = np.linalg.norm(dir_vecs_full, axis=1, keepdims=True)
        dir_norm = np.where(dir_norm > 0.0, dir_norm, 1.0)
        dir_vecs_full = dir_vecs_full / dir_norm
        return {
            "mode": "full",
            "dir_vecs_full": np.asarray(dir_vecs_full, dtype=np.float32),
            "psi_deg_full": relion_psi_from_rotation_matrices(grid_rotations_full),
            "n_pixels": np.asarray(n_pixels, dtype=np.int64),
            "n_psi": np.asarray(n_psi, dtype=np.int64),
        }

    grid_eulers = np.asarray(grid_eulers, dtype=np.float32).reshape(-1, 3)
    if grid_eulers.shape[0] != expected:
        raise ValueError(
            f"grid_eulers must have shape ({expected}, 3) for healpix_order={healpix_order}, got {grid_eulers.shape}",
        )

    grid_3d = grid_eulers.reshape(n_psi, n_pixels, 3)
    psi_deg_full = np.mod(grid_eulers[:, 2].astype(np.float64), 360.0)
    if grid_rotations_full is None:
        grid_rotations_full = utils.R_from_relion(grid_eulers, degrees=True).astype(np.float32)
    dir_vecs_full = np.asarray(grid_rotations_full[:, 2, :], dtype=np.float32)
    dir_norm = np.linalg.norm(dir_vecs_full, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm > 0.0, dir_norm, 1.0)
    dir_vecs_full = dir_vecs_full / dir_norm

    dir_vecs_3d = dir_vecs_full.reshape(n_psi, n_pixels, 3)
    psi_3d = psi_deg_full.reshape(n_psi, n_pixels)

    factorized_dirs = np.allclose(dir_vecs_3d, dir_vecs_3d[0:1], rtol=1e-6, atol=1e-6)
    psi_ref = psi_3d[:, :1]
    factorized_psi = float(np.max(_wrapped_abs_diff_deg(psi_3d, psi_ref))) < 1e-4

    if factorized_dirs and factorized_psi:
        return {
            "mode": "factorized",
            "rot_deg": np.asarray(grid_3d[0, :, 0], dtype=np.float32),
            "tilt_deg": np.asarray(grid_3d[0, :, 1], dtype=np.float32),
            "dir_vecs": np.asarray(dir_vecs_3d[0], dtype=np.float32),
            "psi_deg": np.asarray(psi_3d[:, 0], dtype=np.float32),
            "n_pixels": np.asarray(n_pixels, dtype=np.int64),
            "n_psi": np.asarray(n_psi, dtype=np.int64),
            "eulers_full": np.asarray(grid_eulers, dtype=np.float32),
        }

    return {
        "mode": "full",
        "dir_vecs_full": np.asarray(dir_vecs_full, dtype=np.float32),
        "psi_deg_full": np.asarray(psi_deg_full, dtype=np.float32),
        "n_pixels": np.asarray(n_pixels, dtype=np.int64),
        "n_psi": np.asarray(n_psi, dtype=np.int64),
        "eulers_full": np.asarray(grid_eulers, dtype=np.float32),
    }


def rotation_grid_n_in_planes(order: int) -> int:
    """Number of in-plane angles used by the RELION-style HEALPix grid."""
    angle_res = 360.0 / (6.0 * 2**order)
    return int(np.round(360.0 / angle_res))


def rotation_grid_size(order: int) -> int:
    """Total number of rotations in the full HEALPix x psi grid."""
    nside = 2**order
    return hp.nside2npix(nside) * rotation_grid_n_in_planes(order)


def _split_rotation_indices(indices, healpix_order):
    """Split full-grid rotation indices into HEALPix pixel and psi components."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    n_pixels = hp.nside2npix(2**healpix_order)
    pixel_idx = indices % n_pixels
    psi_idx = indices // n_pixels
    return pixel_idx, psi_idx


def _combine_rotation_indices(pixel_idx, psi_idx, healpix_order):
    """Combine HEALPix pixel and psi components into full-grid indices."""
    pixel_idx = np.asarray(pixel_idx, dtype=np.int64).reshape(-1)
    psi_idx = np.asarray(psi_idx, dtype=np.int64).reshape(-1)
    n_pixels = hp.nside2npix(2**healpix_order)
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
    # RELION convention: rot=phi (azimuth), tilt=theta (polar), psi=in-plane.
    angles = np.stack([phi, theta, angles[1]], axis=-1)
    angles = angles.reshape(-1, 3)
    angles = angles / (2 * np.pi) * 360
    if matrices:
        angles = utils.R_from_relion(angles)
    return angles


def get_translation_grid(max_pixel, pixel_offset):
    gridded_max_pixel = (max_pixel // pixel_offset) * pixel_offset
    xrange = np.arange(-gridded_max_pixel, gridded_max_pixel + 1, pixel_offset)
    # Match RELION's HealpixSampling::setTranslations loop order exactly:
    # x is the outer loop, y is the inner loop.  The ordering is scientifically
    # irrelevant but important for source-level parity of itrans diagnostics.
    x, y = np.meshgrid(xrange, xrange, indexing="ij")
    grid = np.stack([x.flatten(), y.flatten()], axis=1)
    norm_res = np.linalg.norm(grid, axis=1) <= max_pixel + 0.001
    grid = grid[norm_res]
    return grid


def rotation_indices_to_relion_eulers(indices, healpix_order):
    """Convert ring-order full-grid indices to RELION Euler angles."""
    meta = _get_relion_grid_metadata(int(healpix_order))
    pixel_idx, psi_idx = _split_rotation_indices(indices, healpix_order)
    return np.stack(
        [
            np.asarray(meta["rot_deg"], dtype=np.float32)[pixel_idx],
            np.asarray(meta["tilt_deg"], dtype=np.float32)[pixel_idx],
            np.asarray(meta["psi_deg"], dtype=np.float32)[psi_idx],
        ],
        axis=-1,
    ).astype(np.float32)


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


def _relion_euler_angles_to_matrix(eulers_deg: np.ndarray) -> np.ndarray:
    """Vectorized port of RELION ``Euler_angles2matrix``.

    This returns RELION's projector matrix ``A``. RECOVAR's rotation matrices
    are the transpose of this representation; use ``utils.R_from_relion`` when
    a RECOVAR-frame matrix is needed.
    """
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    alpha = np.deg2rad(eulers[:, 0])
    beta = np.deg2rad(eulers[:, 1])
    gamma = np.deg2rad(eulers[:, 2])

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.empty((eulers.shape[0], 3, 3), dtype=np.float64)
    A[:, 0, 0] = cg * cc - sg * sa
    A[:, 0, 1] = cg * cs + sg * ca
    A[:, 0, 2] = -cg * sb
    A[:, 1, 0] = -sg * cc - cg * sa
    A[:, 1, 1] = -sg * cs + cg * ca
    A[:, 1, 2] = sg * sb
    A[:, 2, 0] = sc
    A[:, 2, 1] = ss
    A[:, 2, 2] = cb
    return A


def _relion_matrix_to_euler_angles(A: np.ndarray) -> np.ndarray:
    """Vectorized port of RELION ``Euler_matrix2angles``."""
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3, 3)
    out = np.empty((A.shape[0], 3), dtype=np.float64)
    abs_sb = np.sqrt(A[:, 0, 2] * A[:, 0, 2] + A[:, 1, 2] * A[:, 1, 2])
    nonsingular = abs_sb > (16.0 * np.finfo(np.float32).eps)

    def relion_sgn(x):
        # RELION's SGN macro returns +1 for zero.
        return np.where(x >= 0.0, 1.0, -1.0)

    if np.any(nonsingular):
        An = A[nonsingular]
        gamma = np.arctan2(An[:, 1, 2], -An[:, 0, 2])
        alpha = np.arctan2(An[:, 2, 1], An[:, 2, 0])
        sign_sb = np.empty_like(gamma)
        small_sin_gamma = np.abs(np.sin(gamma)) < np.finfo(np.float32).eps
        if np.any(small_sin_gamma):
            sign_sb[small_sin_gamma] = relion_sgn(
                -An[small_sin_gamma, 0, 2] / np.cos(gamma[small_sin_gamma])
            )
        if np.any(~small_sin_gamma):
            sign_sb[~small_sin_gamma] = np.where(
                np.sin(gamma[~small_sin_gamma]) > 0.0,
                relion_sgn(An[~small_sin_gamma, 1, 2]),
                -relion_sgn(An[~small_sin_gamma, 1, 2]),
            )
        beta = np.arctan2(sign_sb * abs_sb[nonsingular], An[:, 2, 2])
        out[nonsingular, 0] = np.rad2deg(alpha)
        out[nonsingular, 1] = np.rad2deg(beta)
        out[nonsingular, 2] = np.rad2deg(gamma)

    if np.any(~nonsingular):
        As = A[~nonsingular]
        positive = As[:, 2, 2] >= 0.0
        alpha = np.zeros(As.shape[0], dtype=np.float64)
        beta = np.where(positive, 0.0, np.pi)
        gamma = np.empty(As.shape[0], dtype=np.float64)
        gamma[positive] = np.arctan2(-As[positive, 1, 0], As[positive, 0, 0])
        gamma[~positive] = np.arctan2(As[~positive, 1, 0], -As[~positive, 0, 0])
        out[~nonsingular, 0] = np.rad2deg(alpha)
        out[~nonsingular, 1] = np.rad2deg(beta)
        out[~nonsingular, 2] = np.rad2deg(gamma)

    return out


def apply_relion_rotation_perturbation_to_eulers(eulers_deg, random_perturbation, angular_sampling_deg):
    """Apply RELION's SamplingPerturbation and return eulers plus matrices.

    RELION does not score ``A @ R_perturb`` directly. It converts that product
    back to Euler angles with ``Euler_matrix2angles`` and later regenerates
    projector matrices with ``generateEulerMatrices``. The round trip changes
    some float32 matrix entries by one ulp; CUDA texture interpolation can
    amplify that into measurable Pmax differences for borderline particles.
    """
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    if abs(float(random_perturbation)) < 1e-12:
        return _relion_euler_angles_to_matrix(eulers).astype(np.float32), eulers.astype(np.float32)

    myperturb = float(random_perturbation) * float(angular_sampling_deg)
    A = _relion_euler_angles_to_matrix(eulers)
    R_perturb = _relion_euler_angles_to_matrix(
        np.array([[myperturb, myperturb, myperturb]], dtype=np.float64)
    )[0]
    perturbed_A = np.einsum("nij,jk->nik", A, R_perturb)
    perturbed_eulers = _relion_matrix_to_euler_angles(perturbed_A)
    perturbed_rotations = _relion_euler_angles_to_matrix(perturbed_eulers).astype(np.float32)
    return perturbed_rotations, perturbed_eulers.astype(np.float32)


def apply_relion_rotation_perturbation(rotations, random_perturbation, angular_sampling_deg):
    """Port of RELION's 3D grid perturbation (healpix_sampling.cpp:1909-1934).

    RELION computes ``A_perturbed = A @ Euler_angles2matrix(p, p, p)`` where
    ``p = random_perturbation * angular_sampling``. Since recovar's grid uses
    ``R_from_relion(theta, phi, psi) = Euler_angles2matrix(phi, theta, psi)``
    and ``R_from_relion(p, p, p) = Euler_angles2matrix(p, p, p)`` (symmetric),
    the correct implementation is right-multiply by ``R_from_relion(p, p, p)``.

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
        psi_step=_grab("rlnPsiStep"),
        offset_range=_grab("rlnOffsetRange"),
        offset_step=_grab("rlnOffsetStep"),
    )


def read_relion_model_metadata(model_star_path):
    """Read RELION model star fields needed for replay.

    Returns ``current_image_size`` and ``current_resolution`` from the
    model star file.  These are written by ``updateCurrentResolution`` +
    ``updateImageSizeAndResolutionPointers`` at the start of each RELION
    iteration and stored in the ``data_model_general`` table.
    """
    import re

    text = open(model_star_path).read()

    def _grab(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            raise ValueError(f"Missing {name} in {model_star_path}")
        return cast(m.group(1))

    return dict(
        current_image_size=_grab("rlnCurrentImageSize", int),
        current_resolution=_grab("rlnCurrentResolution"),
    )


def read_relion_optimiser_metadata(optimiser_star_path):
    """Read RELION optimiser fields needed for exact replay control flow."""
    import re

    text = open(optimiser_star_path).read()

    def _grab(name, cast=float, default=None):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            return default
        return cast(m.group(1))

    return dict(
        overall_accuracy_rotations=_grab("rlnOverallAccuracyRotations"),
        overall_accuracy_translations_angst=_grab("rlnOverallAccuracyTranslationsAngst"),
        has_converged=_grab("rlnHasConverged", int),
        number_iter_without_resolution_gain=_grab("rlnNumberOfIterWithoutResolutionGain", int),
        number_iter_without_changing_assignments=_grab("rlnNumberOfIterWithoutChangingAssignments", int),
        changes_optimal_orientations=_grab("rlnChangesOptimalOrientations"),
        changes_optimal_offsets=_grab("rlnChangesOptimalOffsets"),
        changes_optimal_classes=_grab("rlnChangesOptimalClasses"),
        smallest_changes_orientations=_grab("rlnSmallestChangesOrientations"),
        smallest_changes_offsets=_grab("rlnSmallestChangesOffsets"),
        smallest_changes_classes=_grab("rlnSmallestChangesClasses"),
    )


def read_relion_direction_prior(model_star_path):
    """Read RELION's saved orientation distribution from ``model.star``."""
    import numpy as np
    import starfile

    data = starfile.read(str(model_star_path))
    if not isinstance(data, dict) or "model_pdf_orient_class_1" not in data:
        raise ValueError(f"Missing model_pdf_orient_class_1 in {model_star_path}")
    df = data["model_pdf_orient_class_1"]
    if "rlnOrientationDistribution" not in df.columns:
        raise ValueError(f"Missing rlnOrientationDistribution in {model_star_path}")
    return np.asarray(df["rlnOrientationDistribution"], dtype=np.float32)


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
        [phi[pix_idx_flat], theta[pix_idx_flat], in_plane_angles[ip_idx.ravel()]],
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
    random_perturbation=0.0,
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
        Indices into the coarse RELION-parity rotation grid. Each index
        corresponds to a specific ``(healpix_pixel, psi_index)`` sample with
        ``healpix_pixel`` interpreted in RELION's NEST ordering.
    parent_nside_level : int
        HEALPix level of the coarse grid.
    oversampling_order : int
        Number of oversampling levels.
    random_perturbation : float
        RELION's per-iteration perturbation instance. When nonzero, the child
        orientations are right-multiplied by the same perturbation rotation as
        RELION's ``getOrientations``.

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

    coarse_n_pixels = hp.nside2npix(2**parent_nside_level)
    parent_pixels = parent_rotation_indices % coarse_n_pixels
    parent_psi = parent_rotation_indices // coarse_n_pixels

    current_pixels = parent_pixels.copy()
    parent_map = np.arange(len(parent_rotation_indices), dtype=np.int64)
    for level in range(oversampling_order):
        del level
        current_pixels = 4 * np.repeat(current_pixels.astype(np.int64, copy=False), 4) + np.tile(
            np.arange(4, dtype=np.int64), len(current_pixels)
        )
        parent_map = np.repeat(parent_map, 4)

    psi_factor = 2**oversampling_order
    coarse_n_in_planes = rotation_grid_n_in_planes(parent_nside_level)
    coarse_psi_step = 2.0 * np.pi / coarse_n_in_planes
    fine_nside_level = parent_nside_level + oversampling_order
    fine_nside = 2**fine_nside_level
    fine_n_pixels = hp.nside2npix(fine_nside)
    fine_n_in_planes = rotation_grid_n_in_planes(fine_nside_level)
    fine_psi_step = 2.0 * np.pi / fine_n_in_planes

    theta, phi = hp.pix2ang(fine_nside, current_pixels, nest=True)
    current_parent_psi = parent_psi[parent_map]
    # Match RELION's pushbackOversampledPsiAngles(): oversampled psi samples
    # are midpoints inside the parent psi bin, not rows of the fine global grid.
    psi_child_angles = (
        current_parent_psi[:, None] * coarse_psi_step
        - 0.5 * coarse_psi_step
        + (0.5 + np.arange(psi_factor, dtype=np.float64)[None, :]) * (coarse_psi_step / psi_factor)
    )
    nearest_child_psi = (
        np.floor(np.mod(psi_child_angles, 2.0 * np.pi) / fine_psi_step + 0.5).astype(np.int64) % fine_n_in_planes
    )

    child_pixels = np.repeat(current_pixels, psi_factor)
    child_rotation_indices = nearest_child_psi.reshape(-1) * fine_n_pixels + child_pixels

    euler_angles = np.stack(
        [
            np.repeat(phi, psi_factor),
            np.repeat(theta, psi_factor),
            psi_child_angles.reshape(-1),
        ],
        axis=-1,
    )
    euler_angles = euler_angles / (2 * np.pi) * 360
    if abs(float(random_perturbation)) > 1e-12:
        matrices, _ = apply_relion_rotation_perturbation_to_eulers(
            euler_angles,
            random_perturbation,
            relion_angular_sampling_deg(parent_nside_level, adaptive_oversampling=0),
        )
    else:
        matrices = _relion_euler_angles_to_matrix(euler_angles).astype(np.float32)
    parent_map = np.repeat(parent_map, psi_factor)

    if return_rotation_indices:
        return (
            matrices,
            parent_map,
            child_rotation_indices.astype(np.int64),
        )

    return matrices, parent_map


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
            degrees (rot, tilt, psi) for each child orientation.
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
        [phi[pix_idx_flat], theta[pix_idx_flat], in_plane_angles[ip_idx.ravel()]],
        axis=-1,
    )
    angles = angles / (2 * np.pi) * 360  # radians → degrees
    return angles, child_pixels


# ---------------------------------------------------------------------------
# Variable-order rotation grid
# ---------------------------------------------------------------------------


def get_relion_rotation_grid(order):
    """Generate the exact RELION HEALPix rotation grid via the C++ binding.

    Returns rotation matrices in recovar's frame that correspond to exactly
    the same set of orientations RELION uses at the given healpix_order.

    RELION orders its grid as (direction-slow, psi-fast). This function
    reorders to recovar's convention (psi-slow, direction-fast) so that
    ``index % n_pixels`` gives the HEALPix pixel index.
    """
    from recovar.relion_bind._relion_bind_core import get_coarse_orientations

    relion_euler = get_coarse_orientations(order)
    R = utils.R_from_relion(relion_euler, degrees=True)
    n_dir = hp.nside2npix(2**order)
    n_psi = R.shape[0] // n_dir
    return R.reshape(n_dir, n_psi, 3, 3).transpose(1, 0, 2, 3).reshape(-1, 3, 3)


def get_relion_rotation_grid_eulers(order):
    """Return RELION Euler angles in the same index order as get_relion_rotation_grid."""
    from recovar.relion_bind._relion_bind_core import get_coarse_orientations

    relion_euler = get_coarse_orientations(order)
    n_dir = hp.nside2npix(2**order)
    n_psi = relion_euler.shape[0] // n_dir
    return relion_euler.reshape(n_dir, n_psi, 3).transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)


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


def get_local_rotation_grid_fast(
    prior_rotation_indices,
    sigma_rot,
    sigma_psi,
    healpix_order,
    sigma_cutoff=3.0,
    *,
    per_image=False,
    grid_metadata=None,
):
    """RELION-style local rotation selection for the C1 HEALPix x psi grid.

    This mirrors the non-helical SPA path in
    ``HealpixSampling::selectOrientationsWithNonZeroPriorProbability`` for
    the common auto-refine case used on this branch:

    - C1 symmetry only
    - factored direction and psi priors
    - ``sigma_tilt = sigma_rot``
    - no bimodal psi search
    - no multi-body extra priors

    The selected set is the Cartesian product of:

    - directions with ``diffang < sigma_cutoff * max(sigma_rot, sigma_tilt)``
    - psi angles with ``diffpsi < sigma_cutoff * sigma_psi``

    and the per-rotation log-prior is
    ``log(direction_prior) + log(psi_prior)`` with both factors normalized
    exactly as RELION does before the product.

    Parameters
    ----------
    prior_rotation_indices : np.ndarray
        Either full-grid rotation indices of shape ``(n_priors,)``,
        explicit RELION Euler angles of shape ``(n_priors, 3)``, or exact
        prior rotation matrices of shape ``(n_priors, 3, 3)``.
    sigma_rot : float
        Gaussian prior sigma for rotation, **radians**. Used as the
        cone radius scale and the log-prior denominator.
    sigma_psi : float
        Gaussian prior sigma for in-plane angle, **radians**. Combined
        Independent in-plane prior sigma; it must not widen the direction
        cone. RELION's direction `biggest_sigma` is `max(sigma_rot,
        sigma_tilt)`, and this SPA path uses `sigma_tilt == sigma_rot`.
    healpix_order : int
        HEALPix order (nside = 2^order) of the rotation grid.
    sigma_cutoff : float
        Include grid points within ``sigma_cutoff * sigma_rot``
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
    healpix_order = int(healpix_order)
    grid_metadata = build_local_search_grid_metadata(healpix_order) if grid_metadata is None else grid_metadata
    mode = str(grid_metadata["mode"])
    n_pixels = int(grid_metadata["n_pixels"])
    n_psi = int(grid_metadata["n_psi"])
    n_total = int(n_pixels * n_psi)

    if prior_rotation_indices.ndim == 0:
        prior_rotation_indices = prior_rotation_indices.reshape(1)

    if prior_rotation_indices.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float32)[
                prior_rotation_indices.astype(np.int64)
            ]
        else:
            prior_eulers = rotation_indices_to_relion_eulers(prior_rotation_indices.astype(np.int64), healpix_order)
        prior_rot_deg = prior_eulers[:, 0]
        prior_tilt_deg = prior_eulers[:, 1]
        prior_psi_deg = prior_eulers[:, 2]
        prior_rotations = utils.R_from_relion(prior_eulers, degrees=True)
    elif prior_rotation_indices.ndim == 2 and prior_rotation_indices.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotation_indices, dtype=np.float64).reshape(-1, 3)
        prior_rot_deg = prior_eulers[:, 0]
        prior_tilt_deg = prior_eulers[:, 1]
        prior_psi_deg = prior_eulers[:, 2]
        prior_rotations = utils.R_from_relion(prior_eulers, degrees=True)
    else:
        prior_rotations = np.asarray(prior_rotation_indices, dtype=np.float64).reshape(-1, 3, 3)
        prior_eulers = utils.R_to_relion(prior_rotations, degrees=True)
        prior_rot_deg = prior_eulers[:, 0]
        prior_tilt_deg = prior_eulers[:, 1]
        prior_psi_deg = prior_eulers[:, 2]

    prior_dir_vecs = np.asarray(prior_rotations[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm

    n_priors = int(np.asarray(prior_rot_deg).reshape(-1).shape[0])
    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    biggest_sigma_deg = sigma_rot_deg

    selected_union = set()
    prior_entries: list[tuple[np.ndarray, np.ndarray]] = []

    if mode == "factorized":
        dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
        psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)

        for i in range(n_priors):
            if sigma_rot_deg > 0.0:
                dots = np.clip(dir_vecs @ prior_dir_vecs[i], -1.0, 1.0)
                diffang = np.rad2deg(np.arccos(dots))
                dir_mask = diffang < float(sigma_cutoff) * biggest_sigma_deg
                dir_indices = np.flatnonzero(dir_mask).astype(np.int64)
                if dir_indices.size == 0:
                    dir_indices = np.array([int(np.argmin(diffang))], dtype=np.int64)
                    dir_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    dir_log_prior = _normalized_log_weights(diffang[dir_indices], biggest_sigma_deg)
            else:
                dir_indices = np.arange(n_pixels, dtype=np.int64)
                dir_log_prior = np.full(
                    n_pixels,
                    -np.log(max(n_pixels, 1)),
                    dtype=np.float32,
                )

            if sigma_psi_deg > 0.0:
                wrapped_prior_psi = float(np.mod(prior_psi_deg[i], 360.0))
                diffpsi = _wrapped_abs_diff_deg(psi_deg_grid, wrapped_prior_psi)
                psi_mask = diffpsi < float(sigma_cutoff) * sigma_psi_deg
                psi_indices = np.flatnonzero(psi_mask).astype(np.int64)
                if psi_indices.size == 0:
                    psi_indices = np.array([int(np.argmin(diffpsi))], dtype=np.int64)
                    psi_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    psi_log_prior = _normalized_log_weights(diffpsi[psi_indices], sigma_psi_deg)
            else:
                psi_indices = np.arange(n_psi, dtype=np.int64)
                psi_log_prior = np.full(
                    n_psi,
                    -np.log(max(n_psi, 1)),
                    dtype=np.float32,
                )

            flat_indices = (psi_indices[:, None] * n_pixels + dir_indices[None, :]).reshape(-1)
            flat_log_prior = (psi_log_prior[:, None] + dir_log_prior[None, :]).reshape(-1).astype(np.float32)
            prior_entries.append((flat_indices.astype(np.int64), flat_log_prior))
            selected_union.update(flat_indices.tolist())
    else:
        dir_vecs_full = np.asarray(grid_metadata["dir_vecs_full"], dtype=np.float64)
        psi_deg_full = np.asarray(grid_metadata["psi_deg_full"], dtype=np.float64)
        sigma_rot_scale = max(biggest_sigma_deg, np.finfo(np.float64).tiny)
        sigma_psi_scale = max(sigma_psi_deg, np.finfo(np.float64).tiny)
        log_prior_full = np.full((n_priors, n_total), -1e30, dtype=np.float32)
        wrapped_prior_psi_deg = np.mod(np.asarray(prior_psi_deg, dtype=np.float64), 360.0)
        cutoff_dir_deg = float(sigma_cutoff) * biggest_sigma_deg
        cutoff_psi_deg = float(sigma_cutoff) * sigma_psi_deg
        block_size = 8

        for start in range(0, n_priors, block_size):
            stop = min(start + block_size, n_priors)
            block = slice(start, stop)

            if sigma_rot_deg > 0.0:
                dots = np.clip(np.asarray(prior_dir_vecs[block], dtype=np.float64) @ dir_vecs_full.T, -1.0, 1.0)
                diffang = np.rad2deg(np.arccos(dots))
                joint_mask = diffang < cutoff_dir_deg
            else:
                diffang = np.zeros((stop - start, n_total), dtype=np.float64)
                joint_mask = np.ones((stop - start, n_total), dtype=bool)

            if sigma_psi_deg > 0.0:
                diffpsi = _wrapped_abs_diff_deg(psi_deg_full[None, :], wrapped_prior_psi_deg[block, None])
                joint_mask &= diffpsi < cutoff_psi_deg
            else:
                diffpsi = np.zeros((stop - start, n_total), dtype=np.float64)

            row_has_support = np.any(joint_mask, axis=1)
            if np.any(row_has_support):
                joint_logw = np.zeros((stop - start, n_total), dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_logw += -0.5 * (diffang / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_logw += -0.5 * (diffpsi / sigma_psi_scale) ** 2

                support_mask = joint_mask[row_has_support]
                support_logw = joint_logw[row_has_support]
                support_logw = np.where(support_mask, support_logw, -np.inf)
                max_logw = np.max(support_logw, axis=1, keepdims=True)
                weights = np.exp(support_logw - max_logw)
                sums = np.sum(weights, axis=1, keepdims=True)
                normalized = np.where(
                    support_mask,
                    support_logw - (max_logw + np.log(sums)),
                    -1e30,
                ).astype(np.float32)
                block_prior = log_prior_full[block]
                block_prior[row_has_support] = normalized
                log_prior_full[block] = block_prior

            if np.any(~row_has_support):
                fallback_rows = np.flatnonzero(~row_has_support)
                joint_cost = np.zeros((fallback_rows.shape[0], n_total), dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_cost += (diffang[fallback_rows] / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_cost += (diffpsi[fallback_rows] / sigma_psi_scale) ** 2
                fallback_indices = np.argmin(joint_cost, axis=1)
                block_prior = log_prior_full[block]
                block_prior[fallback_rows, fallback_indices] = 0.0
                log_prior_full[block] = block_prior

        selected_mask = np.any(log_prior_full > -1e29, axis=0)
        selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
        if selected_indices.size == 0:
            selected_indices = np.arange(n_total, dtype=np.int64)
            log_prior = np.full((n_priors, n_total), -np.log(max(n_total, 1)), dtype=np.float32)
        else:
            log_prior = log_prior_full[:, selected_indices]
        if per_image:
            return selected_indices, log_prior
        return selected_indices, np.max(log_prior, axis=0).astype(np.float32)

    selected_indices = np.array(sorted(selected_union), dtype=np.int64)
    if selected_indices.size == 0:
        selected_indices = np.arange(n_total, dtype=np.int64)

    index_to_pos = {int(idx): pos for pos, idx in enumerate(selected_indices.tolist())}
    log_prior = np.full((n_priors, selected_indices.shape[0]), -1e30, dtype=np.float32)

    for i, (flat_indices, flat_log_prior) in enumerate(prior_entries):
        positions = np.array([index_to_pos[int(idx)] for idx in flat_indices], dtype=np.int64)
        log_prior[i, positions] = flat_log_prior

    if per_image:
        return selected_indices, log_prior
    return selected_indices, np.max(log_prior, axis=0).astype(np.float32)


@functools.partial(jax.jit, static_argnums=[1])
def translations_to_indices(translations, image_shape):
    # Assumes that translations are integers
    indices = translations + image_shape[0] // 2
    vec_indices = indices[..., 1] * image_shape[1] + indices[..., 0]
    return vec_indices
