"""RELION-parity translation and direction prior construction.

These functions build the Gaussian log-prior arrays that RELION uses to
bias the E-step towards the previous best orientations and offsets.
Called by ``_run_relion_iteration_loop`` and ``_run_local_search_iteration``
in ``refine.py``.
"""

import healpy as hp
import numpy as np
import scipy.special

from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size


def make_relion_translation_log_prior(
    translations,
    voxel_size,
    sigma_offset_angstrom,
    prior_centers=None,
    *,
    offset_range_pixels=None,
):
    """Return RELION-style normalized log-priors over a translation grid."""
    translations = np.asarray(translations, dtype=np.float32)
    if translations.ndim != 2:
        raise ValueError(
            f"translations must have shape (n_trans, dim), got {translations.shape}",
        )
    sigma_offset_angstrom = float(sigma_offset_angstrom)
    voxel_size = float(voxel_size if voxel_size > 0 else 1.0)
    if offset_range_pixels is not None and float(offset_range_pixels) > 0.0:
        # RELION's score path uses sigma = offset_range / 3 while an explicit
        # translational search range is active.
        sigma_offset_angstrom = float(offset_range_pixels) * voxel_size / 3.0
    n_trans = translations.shape[0]

    if prior_centers is None:
        centers = np.zeros((1, translations.shape[1]), dtype=np.float32)
        shared = True
    else:
        centers = np.asarray(prior_centers, dtype=np.float32).reshape(-1, translations.shape[1])
        shared = False

    if sigma_offset_angstrom <= 0.0:
        zeros = np.zeros((centers.shape[0], n_trans), dtype=np.float32)
        return zeros[0] if shared else zeros

    diffs_ang = (translations[None, :, :] - centers[:, None, :]) * voxel_size
    sqdist_ang = np.sum(diffs_ang**2, axis=-1)
    log_prior = -0.5 * sqdist_ang / (sigma_offset_angstrom**2)
    log_prior -= scipy.special.logsumexp(log_prior, axis=1, keepdims=True)
    log_prior += np.log(float(n_trans))
    log_prior = log_prior.astype(np.float32)
    return log_prior[0] if shared else log_prior


def relion_translation_search_base(previous_best_translations):
    """Return the stored absolute offsets used to pre-center local search."""
    if previous_best_translations is None:
        return None
    return np.asarray(previous_best_translations, dtype=np.float32)


def collapse_rotation_posterior_to_direction_prior(rotation_posterior_sums, healpix_order):
    """Collapse per-rotation posterior mass onto RELION's HEALPix directions."""
    rotation_posterior_sums = np.asarray(rotation_posterior_sums, dtype=np.float64).reshape(-1)
    n_rot = rotation_grid_size(healpix_order)
    if rotation_posterior_sums.shape[0] != n_rot:
        raise ValueError(
            f"rotation_posterior_sums must have shape ({n_rot},), got {rotation_posterior_sums.shape}",
        )

    n_pixels = n_rot // rotation_grid_n_in_planes(healpix_order)
    direction_weights = np.zeros(n_pixels, dtype=np.float64)
    np.add.at(direction_weights, np.arange(n_rot, dtype=np.int64) % n_pixels, rotation_posterior_sums)
    total = float(direction_weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        direction_weights.fill(1.0 / max(n_pixels, 1))
    else:
        direction_weights /= total
    return direction_weights.astype(np.float32)


def infer_direction_prior_healpix_order(direction_prior):
    """Infer HEALPix order from a RELION direction-prior vector length."""
    n_pixels = int(np.asarray(direction_prior).reshape(-1).shape[0])
    order = 0
    while hp.nside2npix(2**order) < n_pixels:
        order += 1
    if hp.nside2npix(2**order) != n_pixels:
        raise ValueError(f"Cannot infer healpix order from direction prior of length {n_pixels}")
    return order


def remap_direction_prior_to_healpix_order(direction_prior, src_order, dst_order):
    """Remap a RELION direction prior between HEALPix orders."""
    direction_prior = np.asarray(direction_prior, dtype=np.float64).reshape(-1)
    if src_order == dst_order:
        out = direction_prior.copy()
    elif src_order > dst_order:
        theta, phi = hp.pix2ang(2**src_order, np.arange(direction_prior.shape[0], dtype=np.int64))
        dst_idx = hp.ang2pix(2**dst_order, theta, phi)
        out = np.zeros(hp.nside2npix(2**dst_order), dtype=np.float64)
        np.add.at(out, dst_idx, direction_prior)
    else:
        theta, phi = hp.pix2ang(2**dst_order, np.arange(hp.nside2npix(2**dst_order), dtype=np.int64))
        src_idx = hp.ang2pix(2**src_order, theta, phi)
        out = direction_prior[src_idx]
    total = float(out.sum())
    if total <= 0.0 or not np.isfinite(total):
        out.fill(1.0 / max(out.shape[0], 1))
    else:
        out /= total
    return out.astype(np.float32)


def make_relion_direction_log_prior(direction_prior, healpix_order, rotations=None):
    """Expand RELION's learned ``pdf_direction`` onto a rotation grid.

    When ``rotations`` is omitted, the prior is expanded onto RELION's
    canonical sample-index ordering. This matches RELION's global
    ``pdf_direction`` handling: the prior is looked up by coarse direction
    index and only then are perturbation / oversampling applied to generate the
    actual trial orientations. The optional ``rotations`` mode is therefore a
    geometry-based expansion helper for diagnostics only, not the RELION-parity
    path used in refinement.
    """
    direction_prior = np.asarray(direction_prior, dtype=np.float32).reshape(-1)
    n_rot = rotation_grid_size(healpix_order)
    n_pixels = n_rot // rotation_grid_n_in_planes(healpix_order)
    if direction_prior.shape[0] != n_pixels:
        raise ValueError(
            f"direction_prior must have shape ({n_pixels},), got {direction_prior.shape}",
        )

    safe_prior = np.clip(direction_prior, np.finfo(np.float32).tiny, None)
    if rotations is None:
        pixel_idx = np.arange(n_rot, dtype=np.int64) % n_pixels
    else:
        rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
        if rotations.shape[0] != n_rot:
            raise ValueError(
                f"rotations must have shape ({n_rot}, 3, 3), got {rotations.shape}",
            )
        view_dirs = rotations[:, 2, :].astype(np.float64)
        norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        view_dirs = view_dirs / norms
        pixel_idx = hp.vec2pix(
            2**healpix_order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
        )
    return np.log(safe_prior[pixel_idx]).astype(np.float32)
