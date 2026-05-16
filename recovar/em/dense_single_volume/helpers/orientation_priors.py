"""RELION-parity translation and direction prior construction.

These functions build the Gaussian log-prior arrays that RELION uses to
bias the E-step towards the previous best orientations and offsets.
Called by ``_run_relion_iteration_loop`` and ``_run_local_search_iteration``
in ``refine.py``.
"""

import healpy as hp
import numpy as np

from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size


def make_relion_translation_log_prior(
    translations,
    voxel_size,
    sigma_offset_angstrom,
    prior_centers=None,
    *,
    offset_range_pixels=None,
):
    """Return RELION's offset prior scores over a translation grid.

    RELION stores ``sampling.translations_{x,y}`` in Angstrom, but uses
    translation shifts in pixels for projection. In the E-step prior path
    (``acc_ml_optimiser_impl.h`` around ``pdf_offset`` construction), it
    computes the Gaussian exponent from the Angstrom-valued sampling grid and
    then multiplies by ``my_pixel_size**2``. This intentionally mirrors that
    source behavior rather than a dimensionally simplified Gaussian.
    """
    translations = np.asarray(translations, dtype=np.float32)
    if translations.ndim != 2:
        raise ValueError(
            f"translations must have shape (n_trans, dim), got {translations.shape}",
        )
    sigma_offset_angstrom = float(sigma_offset_angstrom)
    voxel_size = float(voxel_size if voxel_size > 0 else 1.0)
    sigma2_offset = sigma_offset_angstrom**2
    if offset_range_pixels is not None and float(offset_range_pixels) > 0.0:
        # RELION's score path uses sigma = offset_range / 3 while an explicit
        # translational search range is active.
        sigma_offset_angstrom = float(offset_range_pixels) * voxel_size / 3.0
        sigma2_offset = sigma_offset_angstrom**2
    n_trans = translations.shape[0]

    if prior_centers is None:
        centers = np.zeros((1, translations.shape[1]), dtype=np.float32)
        shared = True
    else:
        centers = np.asarray(prior_centers, dtype=np.float32).reshape(-1, translations.shape[1])
        shared = False

    if sigma2_offset <= 0.0:
        zeros = np.zeros((centers.shape[0], n_trans), dtype=np.float32)
        return zeros[0] if shared else zeros

    diffs_ang = (translations[None, :, :] - centers[:, None, :]) * voxel_size
    sqdist_ang = np.sum(diffs_ang**2, axis=-1)
    log_prior = -0.5 * sqdist_ang / sigma2_offset
    # RELION applies this after accumulating per-axis terms. It is not a
    # normalization constant; it materially sharpens the translation prior.
    log_prior *= voxel_size**2
    log_prior = log_prior.astype(np.float32)
    return log_prior[0] if shared else log_prior


def relion_translation_search_base(previous_best_translations):
    """Return RELION's integer-pixel pre-shift for stored absolute offsets."""
    if previous_best_translations is None:
        return None
    previous_best_translations = np.asarray(previous_best_translations, dtype=np.float32)
    if previous_best_translations.size == 0:
        return previous_best_translations.reshape(0, 2)
    return np.rint(previous_best_translations).astype(np.float32)


def relion_translation_prior_center(previous_best_translations, voxel_size, prior_offsets=None):
    """Return RELION's offset-prior center in RECOVAR search-grid pixels.

    RELION's accelerated path builds ``pdf_offset`` from
    ``old_offset + sampling.translations - prior`` where the rounded
    ``old_offset`` term is in pixel-like metadata units while
    ``sampling.translations`` is the Angstrom-space grid.  RECOVAR scores
    shifts after ``getTranslationsInPixel``-style conversion, so the prior
    center must be divided by the pixel size.  The image pre-shift itself
    still uses :func:`relion_translation_search_base` unchanged.
    """
    old_offset = relion_translation_search_base(previous_best_translations)
    if old_offset is None:
        return None
    voxel_size = float(voxel_size if voxel_size > 0 else 1.0)
    if prior_offsets is None:
        prior = np.zeros_like(old_offset, dtype=np.float32)
    else:
        prior = np.asarray(prior_offsets, dtype=np.float32).reshape(old_offset.shape)
    return ((prior - old_offset) / voxel_size).astype(np.float32)


def relion_sigma_offset_prior_center(previous_best_translations, prior_offsets=None):
    """Return RELION's sigma-offset sufficient-statistic center in pixels.

    RELION's ``pdf_offset`` scoring path evaluates the coarse Angstrom
    sampling grid directly, but ``storeWeightedSums`` accumulates
    ``wsum_sigma2_offset`` from ``getTranslationsInPixel`` shifts:
    ``prior - rounded_old_offset - sampled_translation_pixels``.  The EM
    engines use pixel-space translation grids and convert squared distances to
    Angstroms themselves, so this center intentionally does not divide by
    pixel size.
    """
    old_offset = relion_translation_search_base(previous_best_translations)
    if old_offset is None:
        return None
    if prior_offsets is None:
        prior = np.zeros_like(old_offset, dtype=np.float32)
    else:
        prior = np.asarray(prior_offsets, dtype=np.float32).reshape(old_offset.shape)
    return (prior - old_offset).astype(np.float32)


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


def normalize_direction_prior_per_half(direction_prior):
    """Return a two-element list of RELION ``pdf_direction`` arrays.

    RELION auto-refine stores separate learned orientation distributions for
    the two half-models.  Replay callers should pass ``[half1, half2]``.  A
    single 1D vector remains accepted for older unit tests and non-auto-refine
    callers, and is shared across both halves.
    """
    if direction_prior is None:
        return [None, None]

    if isinstance(direction_prior, (list, tuple)) and len(direction_prior) == 2:
        return [
            None if direction_prior[0] is None else np.asarray(direction_prior[0], dtype=np.float32).reshape(-1),
            None if direction_prior[1] is None else np.asarray(direction_prior[1], dtype=np.float32).reshape(-1),
        ]

    arr = np.asarray(direction_prior, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return [arr[0].reshape(-1), arr[1].reshape(-1)]
    arr = arr.reshape(-1)
    return [arr.copy(), arr.copy()]


def normalize_class_direction_prior(direction_prior, n_classes):
    """Return per-class conditional RELION direction priors.

    RELION's ``pdf_direction[class]`` rows are joint class-direction masses in
    the no-orientation-prior branch, so row sums may equal ``pdf_class`` rather
    than one.  RECOVAR keeps ``class_log_priors`` separate, so K-class callers
    use row-normalized conditionals here and pass the class prior explicitly.
    """

    arr = np.asarray(direction_prior, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr[None, :], (int(n_classes), arr.shape[0])).copy()
    elif arr.ndim == 2 and arr.shape[0] == int(n_classes):
        arr = arr.copy()
    else:
        raise ValueError(
            "class direction prior must have shape (n_dirs,) or "
            f"({int(n_classes)}, n_dirs), got {arr.shape}",
        )

    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("class direction prior entries must be finite and non-negative")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each class direction prior row must have positive mass")
    return (arr / row_sums).astype(np.float32)


def normalize_class_direction_prior_per_half(direction_prior, n_classes):
    """Return two per-half arrays with shape ``(n_classes, n_dirs)``."""

    n_classes = int(n_classes)
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if direction_prior is None:
        return [None, None]

    if isinstance(direction_prior, (list, tuple)) and len(direction_prior) == 2:
        return [
            None
            if direction_prior[0] is None
            else normalize_class_direction_prior(direction_prior[0], n_classes),
            None
            if direction_prior[1] is None
            else normalize_class_direction_prior(direction_prior[1], n_classes),
        ]

    arr = np.asarray(direction_prior, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[1] == n_classes:
        return [
            normalize_class_direction_prior(arr[0], n_classes),
            normalize_class_direction_prior(arr[1], n_classes),
        ]
    if arr.ndim == 2 and n_classes == 1 and arr.shape[0] == 2:
        return [
            normalize_class_direction_prior(arr[0], n_classes),
            normalize_class_direction_prior(arr[1], n_classes),
        ]

    shared = normalize_class_direction_prior(arr, n_classes)
    return [shared.copy(), shared.copy()]


def class_weights_from_direction_prior(direction_prior, n_classes):
    """Infer RELION class weights from raw per-class ``pdf_direction`` rows."""

    priors = normalize_class_direction_prior_per_half(direction_prior, n_classes)
    for original, normalized in zip(
        direction_prior if isinstance(direction_prior, (list, tuple)) and len(direction_prior) == 2 else [direction_prior],
        priors,
    ):
        if normalized is None:
            continue
        arr = np.asarray(original, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 2:
            arr = arr[0]
        if arr.ndim == 1:
            continue
        if arr.ndim == 2 and arr.shape[0] == int(n_classes):
            weights = arr.sum(axis=1)
            total = float(weights.sum())
            if total > 0.0 and np.all(np.isfinite(weights)):
                return (weights / total).astype(np.float64)
    return None


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

    prior_for_rotations = direction_prior[pixel_idx]
    log_prior = np.full(prior_for_rotations.shape, -np.inf, dtype=np.float32)
    positive = prior_for_rotations > 0.0
    log_prior[positive] = np.log(prior_for_rotations[positive]).astype(np.float32)
    return log_prior
