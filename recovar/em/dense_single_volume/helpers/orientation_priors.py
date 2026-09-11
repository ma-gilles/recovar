"""RELION-parity translation and direction prior construction.

These functions build the Gaussian log-prior arrays that RELION uses to
bias the E-step towards the previous best orientations and offsets.
Called by ``_run_relion_iteration_loop`` and ``_run_local_search_iteration``
in ``refine.py``.
"""

from dataclasses import dataclass

import healpy as hp
import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import healpix_angular_step
from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size


def relion_round_away_from_zero(values, *, dtype: np.dtype = np.float32):
    """Vectorized RELION ``ROUND`` macro: nearest integer, ties away from zero.

    ``dtype`` controls only the returned cast; the round itself is always
    computed at float64. RELION's own ``ROUND`` macro operates on RFLOAT
    (double under ``DoublePrec_CPU``/``DoublePrec_ACC``) throughout -- pass
    ``np.float64`` to match a genuine double-precision comparison.
    """
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), -np.floor(-arr + 0.5))
    return rounded.astype(dtype, copy=False)


def make_relion_translation_log_prior(
    translations,
    voxel_size,
    sigma_offset_angstrom,
    prior_centers=None,
    *,
    offset_range_pixels=None,
    dtype: np.dtype = np.float32,
):
    """Return RELION's offset prior scores over a translation grid.

    RELION's accelerated E-step prior path (``acc_ml_optimiser_impl.h``
    around ``pdf_offset`` construction) builds the offset difference from the
    Angstrom-valued sampling grid and then applies an extra
    ``my_pixel_size**2`` factor.  RECOVAR translation grids are in projection
    pixels, so explicit prior centers intentionally use the source-equivalent
    ``pixel_size**4 / sigma_offset_A**2`` scale.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running a genuine double-precision comparison should pass
    ``np.float64`` explicitly.
    """
    translations = np.asarray(translations, dtype=dtype)
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
        return np.zeros(n_trans, dtype=dtype)

    prior_centers = np.asarray(prior_centers, dtype=dtype)
    shared = prior_centers.ndim == 1
    centers = prior_centers.reshape(-1, translations.shape[1])

    if sigma2_offset <= 0.0:
        zeros = np.zeros((centers.shape[0], n_trans), dtype=dtype)
        return zeros[0] if shared else zeros

    diffs_px = translations[None, :, :] - centers[:, None, :]
    sqdist_px = np.sum(diffs_px**2, axis=-1)
    log_prior = -0.5 * sqdist_px * (voxel_size**4) / sigma2_offset
    log_prior = log_prior.astype(dtype)
    return log_prior[0] if shared else log_prior


def relion_translation_search_base(previous_best_translations, *, dtype: np.dtype = np.float32):
    """Return RELION's integer-pixel pre-shift for stored absolute offsets."""
    if previous_best_translations is None:
        return None
    previous_best_translations = np.asarray(previous_best_translations, dtype=np.float64)
    if previous_best_translations.size == 0:
        return previous_best_translations.reshape(0, 2).astype(dtype, copy=False)
    return relion_round_away_from_zero(previous_best_translations, dtype=dtype)


def relion_translation_prior_center(
    previous_best_translations, voxel_size, prior_offsets=None, *, dtype: np.dtype = np.float32
):
    """Return dense/local score-prior centers in RECOVAR search-grid pixels.

    RELION's accelerated path builds ``pdf_offset`` from
    ``old_offset + sampling.translations - prior``.  The sampling grid is in
    Angstroms, while RECOVAR scores projection-pixel shifts, so after the
    rounded ``old_offset`` image pre-shift the score-grid prior center is
    ``(prior - rounded_old_offset) / pixel_size``.  In ordinary AutoRefine
    there is no separate origin prior, so ``prior`` defaults to zero.

    Dense and local scoring use this same center calculation. Sigma-offset
    sufficient statistics use ``relion_sigma_offset_prior_center`` instead;
    that center remains in pixels without the division by pixel size.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running a genuine double-precision comparison should pass
    ``np.float64`` explicitly. RELION itself never narrows this computation
    (RFLOAT/XFLOAT are both double under double-precision builds).
    """
    old_offset = relion_translation_search_base(previous_best_translations, dtype=dtype)
    if old_offset is None:
        return None
    voxel_size = float(voxel_size if voxel_size > 0 else 1.0)
    if prior_offsets is None:
        prior = np.zeros_like(old_offset, dtype=dtype)
    else:
        prior = np.asarray(prior_offsets, dtype=dtype).reshape(old_offset.shape)
    return ((prior - old_offset) / voxel_size).astype(dtype)


def relion_sigma_offset_prior_center(previous_best_translations, prior_offsets=None, *, dtype: np.dtype = np.float32):
    """Return RELION's sigma-offset sufficient-statistic center in pixels.

    RELION's ``pdf_offset`` scoring path evaluates the coarse Angstrom
    sampling grid directly, but ``storeWeightedSums`` accumulates
    ``wsum_sigma2_offset`` from ``getTranslationsInPixel`` shifts:
    ``prior - rounded_old_offset - sampled_translation_pixels``.  The EM
    engines use pixel-space translation grids and convert squared distances to
    Angstroms themselves, so this center intentionally does not divide by
    pixel size.

    ``dtype`` -- see :func:`relion_translation_prior_center`.
    """
    old_offset = relion_translation_search_base(previous_best_translations, dtype=dtype)
    if old_offset is None:
        return None
    if prior_offsets is None:
        prior = np.zeros_like(old_offset, dtype=dtype)
    else:
        prior = np.asarray(prior_offsets, dtype=dtype).reshape(old_offset.shape)
    return (prior - old_offset).astype(dtype)


@dataclass(frozen=True)
class HalfTranslationPriorInputs:
    """Per-half translation prior inputs for one scoring pass.

    ``prior_center`` and ``local_prior_center`` are independent arrays with the
    same ``pdf_offset`` center in search-grid pixels (``None`` on cold start);
    the dense score log-prior uses the first and the local adapter the second.
    ``sigma_center`` is the ``wsum_sigma2_offset`` accumulation center in pixels
    (``None`` on cold start) and ``engine_prior_center`` its zero-centered
    cold-start substitute. ``prior_translations`` are the translations the
    score log-prior is evaluated on.
    """

    prior_center: np.ndarray | None
    local_prior_center: np.ndarray | None
    sigma_center: np.ndarray | None
    engine_prior_center: np.ndarray
    prior_translations: np.ndarray


def relion_half_translation_prior_inputs(
    previous_best_translations,
    *,
    voxel_size,
    base_translations,
    current_translations,
    dtype: np.dtype = np.float32,
):
    """Build one half-set's RELION translation prior inputs.

    RELION's translation prior sigma follows ``ml_optimiser.cpp:7737-7746``:
    ``offset_range_x > 0`` overrides the per-axis sigma, otherwise the learned
    model ``sigma2_offset`` is used, kept per half-model in split auto-refine.
    The score prior (``pdf_offset``) and the sigma-offset sufficient statistic
    use their separate RELION center formulas, see
    :func:`relion_translation_prior_center` and
    :func:`relion_sigma_offset_prior_center`.

    On the iteration-1 cold start ``previous_best_translations`` is ``None``,
    so the sigma center is ``None`` and the engine's ``wsum_sigma2_offset``
    accumulator would stay off. RELION still accumulates
    ``sum_i E[||t_i||^2]`` around the implicit zero prior, which seeds the
    iteration-2 sigma offset, so the engine receives a zero center instead.
    The score log-prior path stays separate: ``None`` centers mean RELION's
    flat cold-start offset prior, an explicit zero center a real Gaussian.

    The score log-prior is evaluated on the base translation grid; when the
    current grid has a different size, a single current translation selects
    the central base translation and any other mismatch uses the current
    grid itself.
    """

    prior_center = relion_translation_prior_center(previous_best_translations, voxel_size, dtype=dtype)
    local_prior_center = relion_translation_prior_center(previous_best_translations, voxel_size, dtype=dtype)
    sigma_center = relion_sigma_offset_prior_center(previous_best_translations, dtype=dtype)
    engine_prior_center = np.zeros(2, dtype=dtype) if sigma_center is None else sigma_center
    prior_translations = np.asarray(base_translations, dtype=dtype)
    if current_translations.shape[0] != base_translations.shape[0]:
        if current_translations.shape[0] == 1 and base_translations.shape[0] > 1:
            center_idx = int(base_translations.shape[0] // 2)
            prior_translations = np.asarray(base_translations[center_idx : center_idx + 1], dtype=dtype)
        else:
            prior_translations = np.asarray(current_translations, dtype=dtype)
    return HalfTranslationPriorInputs(
        prior_center=prior_center,
        local_prior_center=local_prior_center,
        sigma_center=sigma_center,
        engine_prior_center=engine_prior_center,
        prior_translations=prior_translations,
    )


def initial_direction_priors_from_snapshot(init_direction_prior, *, k_class_enabled: bool, n_classes: int, dtype: np.dtype, log):
    """Initialize per-half direction priors from a RELION snapshot.

    A restart from a RELION model carries the previous iteration's
    ``pdf_direction`` as a non-uniform prior that RELION applies in its next
    E-step. K-class snapshots yield one ``(n_classes, n_pixels)`` prior per
    half, K=1 snapshots one vector per half; both keep RELION's per-half
    models. Returns the four caller-owned lists
    ``(global_prior, global_order, class_prior, class_order)``, each with one
    entry per half, all ``None`` without a snapshot prior. Orders are inferred
    from the prior length.
    """

    global_prior_per_half = [None, None]
    global_order_per_half = [None, None]
    class_prior_per_half = [None, None]
    class_order_per_half = [None, None]
    if init_direction_prior is None:
        return global_prior_per_half, global_order_per_half, class_prior_per_half, class_order_per_half
    if k_class_enabled:
        class_prior_per_half = normalize_class_direction_prior_per_half(init_direction_prior, n_classes, dtype=dtype)
        for k in range(2):
            if class_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(class_prior_per_half[k], dtype=dtype)
            class_prior_per_half[k] = prior_k
            class_order_per_half[k] = infer_direction_prior_healpix_order(prior_k[0])
            log.info(
                "RELION mode: loaded init class direction priors half-%d: %d classes, %d directions",
                k + 1,
                prior_k.shape[0],
                prior_k.shape[1],
            )
        return global_prior_per_half, global_order_per_half, class_prior_per_half, class_order_per_half
    global_prior_per_half = normalize_direction_prior_per_half(init_direction_prior, dtype=dtype)
    for k in range(2):
        if global_prior_per_half[k] is None:
            continue
        prior_k = np.asarray(global_prior_per_half[k], dtype=dtype)
        global_prior_per_half[k] = prior_k
        global_order_per_half[k] = infer_direction_prior_healpix_order(prior_k)
        log.info(
            "RELION mode: loaded init direction prior half-%d: %d directions, range=[%.6f, %.6f], %d zero-probability",
            k + 1,
            len(prior_k),
            prior_k.min(),
            prior_k.max(),
            int(np.sum(prior_k == 0)),
        )
    return global_prior_per_half, global_order_per_half, class_prior_per_half, class_order_per_half


def _sealed_direction_log_prior(direction_prior, sealed_sampling_state, *, dtype: np.dtype = np.float32):
    """Expand a full direction prior onto the exact captured direction rows."""

    prior = np.asarray(direction_prior, dtype=dtype).reshape(-1)
    direction_ids = np.asarray(sealed_sampling_state["directions_ipix"], dtype=np.int64)
    n_psi = int(np.asarray(sealed_sampling_state["psi_angles_deg"]).size)
    selected = np.tile(prior[direction_ids], n_psi)
    result = np.full(selected.shape, -np.inf, dtype=dtype)
    positive = selected > 0.0
    result[positive] = np.log(selected[positive]).astype(dtype)
    return result


@dataclass(frozen=True)
class HalfDirectionLogPriors:
    """Direction log priors handed to one half-set's global scorer.

    Exactly one of the fields is set when a prior applies: ``rotation_log_prior``
    for K=1, ``class_rotation_log_prior`` with shape ``(n_classes, n_rot)`` for
    K-class. Both are ``None`` for local searches or a uniform prior.
    """

    rotation_log_prior: np.ndarray | None
    class_rotation_log_prior: np.ndarray | None


def relion_direction_log_priors_for_half(
    *,
    use_local: bool,
    scoring_healpix_order,
    k_class_enabled: bool,
    n_classes: int,
    class_direction_prior,
    class_direction_prior_order,
    global_direction_prior,
    global_direction_prior_order,
    sealed_sampling_state,
    dtype: np.dtype,
    log,
    half_index: int,
) -> HalfDirectionLogPriors:
    """Build one half's ``pdf_direction`` log priors the way RELION scores them.

    RELION (``ml_optimiser.cpp``, ``getAllSquaredDifferences`` /
    ``convertAllSquaredDifferencesToWeights``) multiplies the orientation
    weight by ``mymodel.pdf_direction[iclass](idir)`` only in ``NOPRIOR`` mode;
    local searches (``PRIOR_ROTTILT_PSI``) use the explicit direction/psi
    priors instead, so ``use_local`` yields no direction prior here. A prior
    learned at another HEALPix order is not used: RELION calls
    ``initialisePdfDirection`` on every sampling change, which resets every
    class to an even distribution. RELION always holds one ``pdf_direction``
    per class and copies class 0 to all classes when seeding K references, so
    a K-class run that only has a shared prior applies it to every class. Each
    half scores with its own model, including RELION's joined final iteration.
    Sealed captured sampling expands the prior onto the captured direction rows
    the scorer actually uses; otherwise the canonical sample ordering is used.
    """

    if use_local:
        return HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)

    def expand(prior):
        if sealed_sampling_state is not None:
            return _sealed_direction_log_prior(prior, sealed_sampling_state, dtype=dtype)
        return make_relion_direction_log_prior(prior, scoring_healpix_order, dtype=dtype)

    if k_class_enabled:
        prior = class_direction_prior
        prior_order = class_direction_prior_order
        source = "learned per-class"
        if (prior is None or prior_order != scoring_healpix_order) and global_direction_prior is not None:
            shared = np.asarray(global_direction_prior, dtype=dtype)
            prior = np.broadcast_to(shared[None, :], (n_classes, shared.size)).copy()
            prior_order = global_direction_prior_order
            source = "shared"
        if prior is None or prior_order != scoring_healpix_order:
            return HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
        class_log_prior = np.stack([expand(prior[class_idx]) for class_idx in range(n_classes)], axis=0)
        log.info(
            "Using %s global direction prior half-%d: %d classes, %d directions at healpix_order=%d",
            source,
            half_index + 1,
            n_classes,
            prior.shape[1],
            scoring_healpix_order,
        )
        return HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=class_log_prior)

    if global_direction_prior is None or global_direction_prior_order != scoring_healpix_order:
        return HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    log.info(
        "Using learned global direction prior half-%d: %d directions at healpix_order=%d",
        half_index + 1,
        np.asarray(global_direction_prior).shape[0],
        scoring_healpix_order,
    )
    return HalfDirectionLogPriors(rotation_log_prior=expand(global_direction_prior), class_rotation_log_prior=None)


def collapse_rotation_posterior_to_direction_prior(
    rotation_posterior_sums, healpix_order, *, dtype: np.dtype = np.float32
):
    """Collapse per-rotation posterior mass onto RELION's HEALPix directions.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running double-precision scoring should pass ``np.float64``.
    RELION's ``pdf_direction`` is ``std::vector<MultidimArray<RFLOAT>>``
    (RELION ``src/ml_model.h``), never narrowed to float.
    """
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
    return direction_weights.astype(dtype)


def infer_direction_prior_healpix_order(direction_prior):
    """Infer HEALPix order from a RELION direction-prior vector length."""
    n_pixels = int(np.asarray(direction_prior).reshape(-1).shape[0])
    order = 0
    while hp.nside2npix(2**order) < n_pixels:
        order += 1
    if hp.nside2npix(2**order) != n_pixels:
        raise ValueError(f"Cannot infer healpix order from direction prior of length {n_pixels}")
    return order


def normalize_direction_prior_per_half(direction_prior, *, dtype: np.dtype = np.float32):
    """Return a two-element list of RELION ``pdf_direction`` arrays.

    RELION auto-refine stores separate learned orientation distributions for
    the two half-models.  Replay callers should pass ``[half1, half2]``.  A
    single 1D vector remains accepted for older unit tests and non-auto-refine
    callers, and is shared across both halves.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running double-precision scoring should pass ``np.float64`` --
    RELION's ``pdf_direction`` is never narrowed (see
    ``collapse_rotation_posterior_to_direction_prior``).
    """
    if direction_prior is None:
        return [None, None]

    if isinstance(direction_prior, (list, tuple)) and len(direction_prior) == 2:
        return [
            None if direction_prior[0] is None else np.asarray(direction_prior[0], dtype=dtype).reshape(-1),
            None if direction_prior[1] is None else np.asarray(direction_prior[1], dtype=dtype).reshape(-1),
        ]

    arr = np.asarray(direction_prior, dtype=dtype)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return [arr[0].reshape(-1), arr[1].reshape(-1)]
    arr = arr.reshape(-1)
    return [arr.copy(), arr.copy()]


def normalize_class_direction_prior(direction_prior, n_classes, *, dtype: np.dtype = np.float32):
    """Return per-class conditional RELION direction priors.

    RELION's ``pdf_direction[class]`` rows are joint class-direction masses in
    the no-orientation-prior branch, so row sums may equal ``pdf_class`` rather
    than one.  RECOVAR keeps ``class_log_priors`` separate, so K-class callers
    use row-normalized conditionals here and pass the class prior explicitly.

    ``dtype`` defaults to float32; double-precision callers should pass
    ``np.float64`` -- see ``normalize_direction_prior_per_half``.
    """

    arr = np.asarray(direction_prior, dtype=dtype)
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
    return (arr / row_sums).astype(dtype)


def normalize_class_direction_prior_per_half(direction_prior, n_classes, *, dtype: np.dtype = np.float32):
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
            else normalize_class_direction_prior(direction_prior[0], n_classes, dtype=dtype),
            None
            if direction_prior[1] is None
            else normalize_class_direction_prior(direction_prior[1], n_classes, dtype=dtype),
        ]

    arr = np.asarray(direction_prior, dtype=dtype)
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[1] == n_classes:
        return [
            normalize_class_direction_prior(arr[0], n_classes, dtype=dtype),
            normalize_class_direction_prior(arr[1], n_classes, dtype=dtype),
        ]
    if arr.ndim == 2 and n_classes == 1 and arr.shape[0] == 2:
        return [
            normalize_class_direction_prior(arr[0], n_classes, dtype=dtype),
            normalize_class_direction_prior(arr[1], n_classes, dtype=dtype),
        ]

    shared = normalize_class_direction_prior(arr, n_classes, dtype=dtype)
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


def remap_direction_prior_to_healpix_order(direction_prior, src_order, dst_order, *, dtype: np.dtype = np.float32):
    """Remap a RELION direction prior between HEALPix orders.

    ``dtype`` defaults to float32; double-precision callers should pass
    ``np.float64`` -- see ``normalize_direction_prior_per_half``.
    """
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
    return out.astype(dtype)


def remap_half_direction_prior_to_healpix_order(
    direction_prior, src_order, dst_order, *, n_classes=None, dtype: np.dtype = np.float32
):
    """Remap one half's global vector or class rows without mixing class mass.

    ``n_classes=None`` selects the global-prior path. Otherwise remap each class
    in its original order and stack the results along the existing class axis.
    Preserve the scalar remapper's dtype default: file replay and explicit
    runtime-dtype replay deliberately remain distinct at their call sites.
    """
    if n_classes is None:
        return remap_direction_prior_to_healpix_order(direction_prior, src_order, dst_order, dtype=dtype)
    return np.stack(
        [
            remap_direction_prior_to_healpix_order(direction_prior[class_idx], src_order, dst_order, dtype=dtype)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )


def make_relion_direction_log_prior(direction_prior, healpix_order, rotations=None, *, dtype: np.dtype = np.float32):
    """Expand RELION's learned ``pdf_direction`` onto a rotation grid.

    When ``rotations`` is omitted, the prior is expanded onto RELION's
    canonical sample-index ordering. This matches RELION's global
    ``pdf_direction`` handling: the prior is looked up by coarse direction
    index and only then are perturbation / oversampling applied to generate the
    actual trial orientations. The optional ``rotations`` mode is therefore a
    geometry-based expansion helper for diagnostics only, not the RELION-parity
    path used in refinement.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running a genuine double-precision comparison should pass
    ``np.float64`` explicitly. RELION's own ``pdf_direction``/log-prior
    arithmetic never narrows below RFLOAT/XFLOAT (both double under
    double-precision builds).
    """
    direction_prior = np.asarray(direction_prior, dtype=dtype).reshape(-1)
    n_rot = rotation_grid_size(healpix_order)
    n_pixels = n_rot // rotation_grid_n_in_planes(healpix_order)
    if direction_prior.shape[0] != n_pixels:
        raise ValueError(
            f"direction_prior must have shape ({n_pixels},), got {direction_prior.shape}",
        )

    if rotations is None:
        pixel_idx = np.arange(n_rot, dtype=np.int64) % n_pixels
    else:
        rotations = np.asarray(rotations, dtype=dtype).reshape(-1, 3, 3)
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
    log_prior = np.full(prior_for_rotations.shape, -np.inf, dtype=dtype)
    positive = prior_for_rotations > 0.0
    log_prior[positive] = np.log(prior_for_rotations[positive]).astype(dtype)
    return log_prior


def relion_local_search_sigmas(sigma_rot, sigma_psi, *, use_local, healpix_order, adaptive_oversampling):
    """Orientational prior widths (radians) of a RELION local angular search.

    A configured ``sigma_rot`` is kept and ``sigma_psi`` falls back to it when
    unset. Without a configured width, a local search uses twice the
    oversampled angular step, RELION's ``sigma2_rot = sigma2_tilt = sigma2_psi
    = 2 * 2 * step * step`` from ``ml_optimiser.cpp`` ``updateAngularSampling``;
    global searches keep the configured values. The regular iterations and the
    final all-data pass share this rule.
    """

    sigma_psi = sigma_psi if sigma_psi > 0 else sigma_rot
    if use_local and sigma_rot <= 0:
        step_rad = np.deg2rad(healpix_angular_step(healpix_order) / (2**adaptive_oversampling))
        sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
        sigma_psi = sigma_rot
    return sigma_rot, sigma_psi
