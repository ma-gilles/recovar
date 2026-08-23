"""Batched significance pruning for adaptive two-pass oversampling.

Runs a coarse E-step and identifies significant (rotation, translation)
pairs per image without materializing the full weight matrix.
Called by ``refine_single_volume`` and ``_run_relion_iteration_loop`` in ``refine.py``.
"""

import logging
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block
from recovar.em.dense_single_volume.helpers.scoring import (
    _e_step_block_scores,
    _e_step_block_scores_windowed,
    _update_logsumexp,
)
from recovar.utils.nvtx_shim import nvtx

_SIGNIFICANCE_SCORE_CACHE_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE"
_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE_MAX_GB"
_SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB = 2.0
_SIGNIFICANCE_FUSED_PASS1_ENV = "RECOVAR_PASS1_FUSED"
_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV = "RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"
_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV = (
    "RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN"
)
_K1_COARSE_GAUSSIAN_FFI_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_FFI"
_K1_COARSE_GAUSSIAN_SINCOSF_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF"
_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV = (
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"
)
_K1_RELION_EXACT_COARSE_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS"
_K1_RELION_F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"
_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"
)
_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE"
)
NVTX_DOMAIN_EM = "recovar_em"
logger = logging.getLogger(__name__)


def _compact_projection_window_positions(compact_indices, window_indices) -> np.ndarray:
    """Map full-image Fourier indices to positions in a compact projection."""

    compact = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    window = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    if np.unique(compact).size != compact.size:
        raise ValueError("compact projection indices must be unique")
    position_by_index = {int(index): position for position, index in enumerate(compact)}
    missing = [int(index) for index in window if int(index) not in position_by_index]
    if missing:
        raise ValueError(
            "projection window contains indices absent from the compact projection: "
            f"{missing[:8]}"
        )
    return np.asarray([position_by_index[int(index)] for index in window], dtype=np.int32)


class SignificanceDumpComplete(RuntimeError):
    """Raised after an explicitly targeted coarse-significance dump is durable."""

    def __init__(self, *, dump_path: str):
        self.dump_path = str(dump_path)
        super().__init__(
            "requested RECOVAR coarse-significance target was written "
            f"(dump_path={self.dump_path})"
        )


def _maybe_stop_after_significance_dump(
    dump_path: str,
    *,
    dump_dir: str,
    target_original_indices: set[int],
    current_size: int | None,
    debug_iteration: int | None,
) -> None:
    """Stop an explicit diagnostic only after its complete target set exists."""

    if os.environ.get(_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV) != "1":
        return
    if not os.path.isfile(dump_path):
        raise RuntimeError(
            "RECOVAR significance stop target is missing its dump file: "
            f"{dump_path}"
        )
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")
    iteration_suffix = (
        ""
        if not target_iteration
        else f"_it{int(debug_iteration):03d}"
    )
    current_size_label = -1 if current_size is None else int(current_size)
    expected_paths = [
        os.path.join(
            dump_dir,
            f"significance_orig{int(original_index):06d}{iteration_suffix}_cs"
            f"{current_size_label:03d}.npz",
        )
        for original_index in sorted(target_original_indices)
    ]
    missing_paths = [path for path in expected_paths if not os.path.isfile(path)]
    if missing_paths:
        logger.info(
            "RECOVAR coarse-significance stop target progress: %d/%d files written",
            len(expected_paths) - len(missing_paths),
            len(expected_paths),
        )
        return
    raise SignificanceDumpComplete(dump_path=dump_path)


def _k1_coarse_gaussian_ffi_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION coarse Gaussian FFI is active."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_FFI_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {_K1_COARSE_GAUSSIAN_FFI_ENV}={token!r}")


def _k1_coarse_gaussian_sincosf_enabled(*, default: bool = False) -> bool:
    """Return whether exact RELION coarse score translation is active."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_SINCOSF_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_GAUSSIAN_SINCOSF_ENV}={token!r}",
    )


def _k1_coarse_gaussian_native_texture_enabled(*, default: bool = False) -> bool:
    """Return whether projection and coarse scoring run in one RELION kernel."""

    token = os.environ.get(
        _K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV}={token!r}",
    )


def _k1_relion_exact_coarse_operands_enabled(*, default: bool = False) -> bool:
    """Return whether coarse Gaussian scoring uses native RFLOAT CTF operands."""

    token = os.environ.get(
        _K1_RELION_EXACT_COARSE_OPERANDS_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}={token!r}",
    )


def _k1_relion_f32_coarse_support_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION CUDA float32 coarse support is active."""

    token = os.environ.get(
        _K1_RELION_F32_COARSE_SUPPORT_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_RELION_F32_COARSE_SUPPORT_ENV}={token!r}",
    )


def _relion_coarse_gaussian_square_operands(
    shifted_half,
    score_weight_half,
    half_weights,
    score_indices,
    score_active_mask,
    *,
    batch_size: int,
    n_trans: int,
):
    """Derive RELION square-difference operands from accepted score inputs."""

    square_score_weight = score_weight_half[:, score_indices]
    square_score_weight = jnp.where(
        score_active_mask[None, :],
        square_score_weight,
        jnp.zeros((), dtype=square_score_weight.dtype),
    )
    square_shifted_weighted = shifted_half.reshape(
        batch_size,
        n_trans,
        -1,
    )[:, :, score_indices]
    nonzero_weight = square_score_weight != 0.0
    safe_weight = jnp.where(nonzero_weight, square_score_weight, 1.0)
    shifted_corrected = square_shifted_weighted / safe_weight[:, None, :]
    shifted_corrected = jnp.where(
        nonzero_weight[:, None, :],
        shifted_corrected,
        jnp.zeros((), dtype=shifted_corrected.dtype),
    )
    pixel_weight = square_score_weight * half_weights[score_indices][None, :]
    return (
        jnp.asarray(shifted_corrected, dtype=jnp.complex64),
        jnp.asarray(pixel_weight, dtype=jnp.float32),
    )


def _relion_coarse_gaussian_square_operands_sincosf(
    unshifted_score_weighted,
    score_weight_half,
    half_weights,
    score_indices,
    score_active_mask,
    translations,
    image_shape,
    *,
    translation_phase_source=None,
    return_unshifted=False,
):
    """Build corrected coarse images with RELION's CUDA ``sincosf`` path."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    if translation_phase_source is None:
        translation_phase_source = translations
    square_score_weight = score_weight_half[:, score_indices]
    square_score_weight = jnp.where(
        score_active_mask[None, :],
        square_score_weight,
        jnp.zeros((), dtype=square_score_weight.dtype),
    )
    square_unshifted_weighted = unshifted_score_weighted[:, score_indices]
    nonzero_weight = square_score_weight != 0.0
    safe_weight = jnp.where(nonzero_weight, square_score_weight, 1.0)
    unshifted_corrected = square_unshifted_weighted / safe_weight
    unshifted_corrected = jnp.where(
        nonzero_weight,
        unshifted_corrected,
        jnp.zeros((), dtype=unshifted_corrected.dtype),
    )
    shifted_corrected = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(unshifted_corrected, dtype=jnp.complex64),
        jnp.asarray(
            _relion_translation_angles_f32(translation_phase_source, image_shape),
            dtype=jnp.float32,
        ),
        score_indices,
        image_shape,
    )
    pixel_weight = square_score_weight * half_weights[score_indices][None, :]
    result = (
        shifted_corrected.reshape(
            unshifted_corrected.shape[0],
            len(translations),
            unshifted_corrected.shape[1],
        ),
        jnp.asarray(pixel_weight, dtype=jnp.float32),
    )
    if return_unshifted:
        return (*result, jnp.asarray(unshifted_corrected, dtype=jnp.complex64))
    return result


def _relion_cc_inverse_power_from_processed(processed_half, score_indices=None):
    """Return RELION firstiter-CC ``1/sum(norm(Fimg))`` in binary64.

    The strict tree rescore uses a per-image FFT to reproduce RELION's
    ``windowFourierTransform``. Its normalization must come from that same
    Fourier array; reusing the batched-FFT norm leaves a one-ULP ``corr_img``
    mismatch at marginal translation ties.
    """

    processed_half = jnp.asarray(processed_half, dtype=jnp.complex128)
    if score_indices is not None:
        processed_half = processed_half[:, jnp.asarray(score_indices, dtype=jnp.int32)]
    power_terms = (
        processed_half.real * processed_half.real
        + processed_half.imag * processed_half.imag
    )
    image_power = jnp.sum(power_terms, axis=-1, keepdims=True)
    return jnp.reciprocal(
        jnp.maximum(image_power, jnp.asarray(1e-30, dtype=jnp.float64))
    )


def _capture_offset_free_and_absolute_float32_scores(scores, log_score_offset):
    """Capture native score margins before adding a large common offset."""

    offset_free = np.asarray(scores, dtype=np.float32)
    absolute = (
        np.asarray(scores, dtype=np.float64) + np.asarray(log_score_offset, dtype=np.float64)
    ).astype(np.float32)
    return offset_free, absolute


def _global_pass1_relion_projector_texture_enabled() -> bool:
    """Whether dense/global pass-1 significance uses texture arithmetic.

    Coarse significance defaults to RELION's texture projector.  Set the
    environment flag to false to force the manual/JAX diagnostic fallback.
    """
    token = os.environ.get(_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV, "1").strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV}={token!r}")


def _firstiter_cc_tree_top2_rescore_max_margin() -> float | None:
    """Return the near-tie margin for RELION coarse-tree replay."""

    token = os.environ.get(_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV, "").strip()
    if not token:
        return None
    if token.lower() in {"off", "none", "disable", "disabled"}:
        return None
    margin = float(token)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} must be a finite "
            f"non-negative float, got {token!r}",
        )
    return margin


def _infer_relion_coarse_healpix_order(n_rotations: int) -> int | None:
    """Infer a complete RELION coarse-grid order, or return ``None``."""

    from recovar.em.sampling import rotation_grid_size

    for order in range(9):
        if int(rotation_grid_size(order)) == int(n_rotations):
            return order
    return None


def _relion_coarse_pose_tie_break_keys(
    candidate_pose_ids,
    *,
    n_trans: int,
    healpix_order: int,
    coarse_rotation_ids=None,
):
    """Map RECOVAR pose ids to RELION's direction-major coarse order."""

    from recovar.em.sampling import rotation_grid_n_in_planes

    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int64)
    if candidate_pose_ids.ndim != 2:
        raise ValueError(
            f"candidate_pose_ids must have shape (n_rows, n_candidates), got {candidate_pose_ids.shape}",
        )
    n_trans = int(n_trans)
    if n_trans <= 0:
        raise ValueError(f"n_trans must be positive, got {n_trans}")
    local_rotation_ids = candidate_pose_ids // n_trans
    if coarse_rotation_ids is None:
        canonical_rotation_ids = local_rotation_ids
    else:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if np.any(local_rotation_ids < 0) or np.any(local_rotation_ids >= coarse_rotation_ids.size):
            raise ValueError("candidate pose references a rotation outside coarse_rotation_ids")
        canonical_rotation_ids = coarse_rotation_ids[local_rotation_ids]

    healpix_order = int(healpix_order)
    n_directions = 12 * (4**healpix_order)
    n_psi = int(rotation_grid_n_in_planes(healpix_order))
    n_rotations = n_directions * n_psi
    if np.any(canonical_rotation_ids < 0) or np.any(canonical_rotation_ids >= n_rotations):
        raise ValueError(
            "canonical coarse rotation ids must index the complete "
            f"RELION order-{healpix_order} grid of size {n_rotations}",
        )
    psi_ids = canonical_rotation_ids // n_directions
    direction_ids = canonical_rotation_ids % n_directions
    relion_rotation_ids = direction_ids * n_psi + psi_ids
    return relion_rotation_ids * n_trans + candidate_pose_ids % n_trans


def _select_relion_coarse_rescore_winner_slots(
    scores,
    candidate_pose_ids,
    *,
    n_trans: int,
    healpix_order: int | None,
    coarse_rotation_ids=None,
):
    """Select maxima, resolving exact score ties in RELION's flat order."""

    scores = np.asarray(scores, dtype=np.float32)
    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int64)
    if scores.shape != candidate_pose_ids.shape or scores.ndim != 2:
        raise ValueError(
            "scores and candidate_pose_ids must have the same "
            f"(n_rows, n_candidates) shape, got {scores.shape} and {candidate_pose_ids.shape}",
        )
    maxima = np.max(scores, axis=1, keepdims=True)
    tied = scores == maxima
    exact_ties = np.count_nonzero(np.sum(tied, axis=1) > 1)
    if healpix_order is None:
        # Compatibility for synthetic/non-HEALPix callers. Production
        # RELION-parity dispatch always supplies or infers the coarse order.
        tie_break_keys = candidate_pose_ids
    else:
        tie_break_keys = _relion_coarse_pose_tie_break_keys(
            candidate_pose_ids,
            n_trans=n_trans,
            healpix_order=healpix_order,
            coarse_rotation_ids=coarse_rotation_ids,
        )
    masked_keys = np.where(tied, tie_break_keys, np.iinfo(np.int64).max)
    return np.argmin(masked_keys, axis=1).astype(np.int32), int(exact_ties)


def _dense_projection_scale(image_shape) -> float:
    """Match the dense E-step projection scaling used by the shared helper."""

    token = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
    n = int(image_shape[0])
    scale = {"-N2": -(n**2), "N2": float(n**2)}.get(token)
    if scale is None:
        raise ValueError(f"Unsupported RECOVAR_DENSE_MEANS_SCALE={token!r}")
    return scale


class ComplementSignificantSampleIndices(NamedTuple):
    """Exact dense significance mask stored as a sparse complement.

    ``None`` remains the representation for an all-True support mask.  This
    object is used when most, but not all, coarse samples are significant:
    storing the included indices would be O(n_samples) per image, while storing
    the excluded tail preserves the exact RELION adaptive mask with bounded
    host memory.
    """

    excluded_indices: np.ndarray
    total_size: int

    @property
    def size(self) -> int:
        return int(self.total_size) - int(np.asarray(self.excluded_indices).size)


def significant_sample_count(samples, total_size: int) -> int:
    """Return the number of included coarse samples for any support encoding."""

    if samples is None:
        return int(total_size)
    if isinstance(samples, ComplementSignificantSampleIndices):
        return int(samples.size)
    return int(np.asarray(samples).size)


def significant_sample_ids(samples, total_size: int) -> np.ndarray:
    """Materialize included ids for diagnostics or dense fallbacks."""

    if samples is None:
        return np.arange(int(total_size), dtype=np.int64)
    if isinstance(samples, ComplementSignificantSampleIndices):
        excluded = np.asarray(samples.excluded_indices, dtype=np.int64).reshape(-1)
        if excluded.size == 0:
            return np.arange(int(total_size), dtype=np.int64)
        keep = np.ones(int(total_size), dtype=bool)
        keep[excluded] = False
        return np.flatnonzero(keep).astype(np.int64, copy=False)
    return np.asarray(samples, dtype=np.int64).reshape(-1)


def compact_significant_sample_indices_from_mask(mask) -> object:
    """Encode one boolean significance mask without materializing dense keeps."""

    mask_np = np.asarray(mask, dtype=bool).reshape(-1)
    if bool(np.all(mask_np)):
        return None
    included = int(np.count_nonzero(mask_np))
    excluded = int(mask_np.size - included)
    if included > excluded:
        return ComplementSignificantSampleIndices(
            excluded_indices=np.flatnonzero(~mask_np).astype(np.int32),
            total_size=int(mask_np.size),
        )
    return np.flatnonzero(mask_np).astype(np.int32)


def _pass1_fused_enabled() -> bool:
    """Whether the fused-pass1 fast path is enabled.

    Off by default while the path is being validated. Set
    ``RECOVAR_PASS1_FUSED=1`` to opt in. Bit-identical to the unfused path
    when active (same ops, same order, same dtypes).
    """
    mode = os.environ.get(_SIGNIFICANCE_FUSED_PASS1_ENV, "0").strip().lower()
    return mode in {"1", "true", "yes", "on"}


@partial(
    jax.jit,
    static_argnames=(
        "image_shape",
        "proj_volume_shape",
        "volume_shape",
        "disc_type",
        "use_window",
        "use_float64_scoring",
        "rotation_block_size",
        "batch_size",
        "n_trans",
        "n_windowed",
        "max_r_static",
    ),
)
def _fused_score_priors_logsumexp_block(
    mean_for_proj,
    rots_b,
    shifted_data,
    batch_norm,
    ctf2_data,
    half_weights_for_score,
    window_indices,
    rotation_log_prior_block,
    translation_log_prior_per_image,
    class_log_prior_scalar,
    valid_count,
    class_max,
    class_sum,
    global_max,
    global_sum,
    *,
    image_shape: tuple,
    proj_volume_shape: tuple,
    volume_shape: tuple,
    disc_type: str,
    use_window: bool,
    use_float64_scoring: bool,
    rotation_block_size: int,
    batch_size: int,
    n_trans: int,
    n_windowed: int,
    max_r_static,
):
    """Fused pass-1 inner block: project + score + pad-mask + priors + 2× logsumexp.

    Replaces 4-5 separate JIT dispatches per (image_batch, class, rotation_block)
    with a single compiled boundary. JAX inlines the @jit-d leaf functions
    (compute_projections_block, _e_step_block_scores_windowed, _update_logsumexp)
    when traced from inside another @jit, so this is bit-identical to the
    unfused path while saving ~150ms of per-batch host-side dispatch at
    50k/256 K=1 (~16s/iter → ~2-4s/iter for pass1).
    """
    proj_kwargs = {}
    if use_window and max_r_static is not None:
        proj_kwargs["max_r"] = max_r_static
    proj_half_b, proj_abs2_half_b = compute_projections_block(
        mean_for_proj,
        rots_b,
        image_shape,
        proj_volume_shape,
        disc_type,
        **proj_kwargs,
    )

    if use_window:
        proj_w = proj_half_b[:, window_indices]
        proj_abs2_w = proj_abs2_half_b[:, window_indices]
        if not use_float64_scoring:
            proj_w = proj_w.astype(jnp.complex64)
            proj_abs2_w = proj_abs2_w.astype(jnp.float32)
        scores = _e_step_block_scores_windowed(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_w * half_weights_for_score,
            proj_abs2_w * half_weights_for_score,
            half_weights_for_score,
            batch_size,
            n_trans,
            n_windowed,
            image_shape,
            volume_shape,
        )
    else:
        if not use_float64_scoring:
            proj_half_b = proj_half_b.astype(jnp.complex64)
            proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
        scores = _e_step_block_scores(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_half_b * half_weights_for_score,
            proj_abs2_half_b * half_weights_for_score,
            half_weights_for_score,
            batch_size,
            n_trans,
            image_shape,
            volume_shape,
        )

    # Padding mask: -inf for rotations beyond valid_count (= n_rot - r0).
    pad_mask = jnp.arange(rotation_block_size)[None, :, None] < valid_count
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    scores = jnp.where(pad_mask, scores, neg_inf)

    # Priors: class scalar + rotation block + per-image translation.
    scores = scores + jnp.asarray(class_log_prior_scalar, dtype=scores.real.dtype)
    scores = scores + rotation_log_prior_block[None, :, None]
    scores = scores + translation_log_prior_per_image[:, None, :]

    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
    return scores, class_max, class_sum, global_max, global_sum


def _significance_score_cache_enabled(n_images, n_classes, n_rot, n_trans, *, use_float64_scoring: bool) -> bool:
    """Whether to keep pass-1 score blocks for reuse in pass 2.

    The cache is exact: it stores the already-prior-adjusted score tensors
    computed for the streaming logsumexp pass and reuses them when forming
    posterior weights/significance masks.  If the estimated tensor footprint is
    too large, callers fall back to the previous recompute path.
    """

    mode = os.environ.get(_SIGNIFICANCE_SCORE_CACHE_ENV, "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    force = mode in {"1", "true", "yes", "on", "force", "always"}
    itemsize = 8 if use_float64_scoring else 4
    estimated_bytes = int(n_images) * int(n_classes) * int(n_rot) * int(n_trans) * itemsize
    max_gb = float(os.environ.get(_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV, _SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB))
    return force or estimated_bytes <= int(max_gb * (1024**3))


def _significance_debug_dump_enabled() -> bool:
    return bool(os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR"))


def _significance_debug_dump_matches(*, current_size, debug_iteration) -> bool:
    """Return whether significance capture applies at this scoring boundary."""

    if not _significance_debug_dump_enabled():
        return False
    if not parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"):
        return False
    target_current_size = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE")
    if target_current_size and (
        current_size is None or int(current_size) != int(target_current_size)
    ):
        return False
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")
    if target_iteration and (
        debug_iteration is None or int(debug_iteration) != int(target_iteration)
    ):
        return False
    return True


def _original_indices_for_local(experiment_dataset, local_indices) -> np.ndarray:
    """Map local batch image indices to original image ids for debug dumps."""
    local_indices = np.asarray(local_indices, dtype=np.int64)
    mapper = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if mapper is not None:
        return np.asarray(mapper(local_indices), dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        return local_indices
    return np.asarray(original_indices_all, dtype=np.int64)[local_indices]


def _maybe_dump_tree_rescore_batch(
    *,
    experiment_dataset,
    indices,
    ambiguous_rows,
    candidate_pose_ids,
    original_best_pose,
    original_best_score,
    original_second_pose,
    original_second_score,
    rescored_scores,
    rescored_winner_slot,
    shifted_candidates,
    score_weight_candidates,
    numerator_weight_candidates,
    rotation_matrices,
    translation_angles,
    n_trans,
    half_weights,
    packed_to_compact,
    projector_full,
    current_size,
    padding_factor,
    projector_max_r,
    debug_iteration,
):
    """Persist exact bounded-rescore operands for selected pass-1 particles."""

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    target_original_indices = parse_env_int_set(
        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
    )
    batch_original_indices = _original_indices_for_local(experiment_dataset, indices)
    ambiguous_original_indices = batch_original_indices[
        np.asarray(ambiguous_rows, dtype=np.int64)
    ]
    dump_dir = os.environ["RECOVAR_SIGNIFICANCE_DUMP_DIR"]
    os.makedirs(dump_dir, exist_ok=True)
    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int32)
    original_best_pose = np.asarray(original_best_pose, dtype=np.int32)
    original_best_score = np.asarray(original_best_score, dtype=np.float32)
    original_second_pose = np.asarray(original_second_pose, dtype=np.int32)
    original_second_score = np.asarray(original_second_score, dtype=np.float32)
    rescored_scores = np.asarray(rescored_scores, dtype=np.float32)
    rescored_winner_slot = np.asarray(rescored_winner_slot, dtype=np.int32)
    for row, original_index in enumerate(ambiguous_original_indices):
        if int(original_index) not in target_original_indices:
            continue
        original_scores_by_candidate = np.where(
            candidate_pose_ids[row] == original_best_pose[row],
            original_best_score[row],
            original_second_score[row],
        ).astype(np.float32, copy=False)
        out_path = os.path.join(
            dump_dir,
            f"tree_rescore_orig{int(original_index):06d}_it"
            f"{int(debug_iteration):03d}_cs{int(current_size):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_index),
            candidate_pose_ids=candidate_pose_ids[row],
            candidate_rotation_ids=(candidate_pose_ids[row] // int(n_trans)),
            candidate_translation_ids=(candidate_pose_ids[row] % int(n_trans)),
            original_best_pose=original_best_pose[row],
            original_second_pose=original_second_pose[row],
            original_scores_by_candidate=original_scores_by_candidate,
            direct_texture_scores=rescored_scores[row],
            direct_texture_winner_slot=rescored_winner_slot[row],
            image_candidates=np.asarray(shifted_candidates[row], dtype=np.complex64),
            image_candidates_are_unshifted=np.asarray(True, dtype=np.bool_),
            translation_angles=np.asarray(translation_angles[row], dtype=np.float32),
            score_weight_candidates=np.asarray(
                score_weight_candidates[row], dtype=np.float32
            ),
            numerator_weight_candidates=np.asarray(
                numerator_weight_candidates[row], dtype=np.float32
            ),
            rotation_matrices=np.asarray(rotation_matrices[row], dtype=np.float32),
            half_weights=np.asarray(half_weights, dtype=np.float32),
            packed_to_compact=np.asarray(packed_to_compact, dtype=np.int32),
            projector_full=np.asarray(projector_full, dtype=np.complex64),
            current_size=np.int64(current_size),
            padding_factor=np.int64(padding_factor),
            projector_max_r=np.int64(projector_max_r),
        )


def _maybe_dump_significance_batch(
    *,
    experiment_dataset,
    indices,
    batch_weights,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    log_z,
    best_score,
    max_posterior,
    rotations,
    translations,
    rotation_log_prior,
    batch_translation_log_prior,
    current_size,
    adaptive_fraction,
    max_significants,
    scores_pre_prior_full=None,
    scores_with_prior_full=None,
    dump_target_positions=None,
    shifted_data=None,
    ctf2_data=None,
    batch_norm=None,
    window_indices=None,
    half_weights_used=None,
    debug_iteration=None,
):
    """Env-gated debug dump for RELION pass-1 significance parity."""
    import os

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    dump_dir = os.environ["RECOVAR_SIGNIFICANCE_DUMP_DIR"]
    target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")

    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)

    os.makedirs(dump_dir, exist_ok=True)
    n_trans = int(translations.shape[0])
    n_candidates = int(batch_weights.shape[1])
    flat_indices = np.arange(n_candidates, dtype=np.int32)
    rot_indices = (flat_indices // n_trans).astype(np.int32)
    trans_indices = (flat_indices % n_trans).astype(np.int32)

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights = np.asarray(batch_weights[local_pos], dtype=np.float64)
        sig_mask = np.asarray(batch_sig_mask[local_pos], dtype=bool)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        dump_row = None
        if dump_target_positions is not None:
            matches = np.flatnonzero(np.asarray(dump_target_positions, dtype=np.int64) == int(local_pos))
            if matches.size:
                dump_row = int(matches[0])
        image_rows = slice(local_pos * n_trans, (local_pos + 1) * n_trans)
        ctf2_arr = None if ctf2_data is None else np.asarray(ctf2_data)
        if ctf2_arr is not None and ctf2_arr.shape[0] == local_indices.shape[0]:
            ctf2_target = ctf2_arr[local_pos : local_pos + 1]
        elif ctf2_arr is not None:
            ctf2_target = ctf2_arr[image_rows]
        else:
            ctf2_target = None
        iteration_suffix = "" if not target_iteration else f"_it{int(debug_iteration):03d}"
        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}{iteration_suffix}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            debug_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            one_based_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_rot=np.int64(rotations.shape[0]),
            n_trans=np.int64(n_trans),
            weights_full=weights,
            significant_mask=sig_mask,
            significant_indices=np.flatnonzero(sig_mask).astype(np.int32),
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            normalization_log_z=np.float64(log_z[local_pos]),
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            rot_indices=rot_indices,
            trans_indices=trans_indices,
            rotation_log_prior=(
                np.asarray(rotation_log_prior, dtype=np.float64)
                if rotation_log_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            scores_pre_prior_full=(
                np.asarray(scores_pre_prior_full[dump_row], dtype=np.float64)
                if scores_pre_prior_full is not None and dump_row is not None
                else np.empty((0,), dtype=np.float64)
            ),
            scores_with_prior_full=(
                np.asarray(scores_with_prior_full[dump_row], dtype=np.float64)
                if scores_with_prior_full is not None and dump_row is not None
                else np.empty((0,), dtype=np.float64)
            ),
            shifted_data=(
                np.asarray(shifted_data[image_rows], dtype=np.complex128)
                if shifted_data is not None
                else np.empty((0,), dtype=np.complex128)
            ),
            ctf2_data=(
                np.asarray(ctf2_target, dtype=np.float64)
                if ctf2_target is not None
                else np.empty((0,), dtype=np.float64)
            ),
            batch_norm=(
                np.asarray(batch_norm[local_pos], dtype=np.float64)
                if batch_norm is not None
                else np.empty((0,), dtype=np.float64)
            ),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32)
                if window_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            half_weights=(
                np.asarray(half_weights_used, dtype=np.float64)
                if half_weights_used is not None
                else np.empty((0,), dtype=np.float64)
            ),
        )


def _maybe_dump_k_class_significance_batch(
    *,
    experiment_dataset,
    indices,
    n_classes: int,
    rotations,
    translations,
    class_weight_mats,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    class_assignment_batch,
    global_log_z,
    class_log_z_values,
    best_score,
    max_posterior,
    rotation_log_prior_padded,
    batch_translation_log_prior,
    class_log_priors,
    current_size,
    adaptive_fraction,
    max_significants,
    target_local_positions=None,
    target_scores_pre_prior_per_class=None,
    target_scores_with_prior_per_class=None,
    projected_reference_rotation_ids=None,
    projected_reference_per_class=None,
    projected_reference_norm_score_per_class=None,
    projected_cross_score_per_class=None,
    shifted_data=None,
    ctf2_data=None,
    window_indices=None,
    half_weights_used=None,
    coarse_gaussian_shifted_corrected=None,
    coarse_gaussian_unshifted_corrected=None,
    coarse_gaussian_pixel_weight=None,
    coarse_gaussian_initial_diff2=None,
    coarse_gaussian_score_indices=None,
    translation_phase_source=None,
    relion_f32_sum_weight=None,
    relion_f32_significant_weight=None,
    relion_f32_cutoff_count=None,
    score_capture_mode="intrusive_per_block_host_materialization",
    debug_iteration=None,
):
    """Env-gated debug dump for the K-class significance pass.

    File naming matches the single-class dump so existing diff tooling works.
    The payload extends the K=1 schema with per-class fields and an explicit
    ``n_classes`` scalar so the user can decode the joint candidate space.
    """

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    dump_dir = os.environ["RECOVAR_SIGNIFICANCE_DUMP_DIR"]
    target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")

    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices = _original_indices_for_local(experiment_dataset, local_indices)

    os.makedirs(dump_dir, exist_ok=True)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])

    weights_per_class = np.stack(
        [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats],
        axis=1,
    )
    sig_mask_full = np.asarray(batch_sig_mask, dtype=bool).reshape(
        local_indices.shape[0],
        n_classes,
        n_rot * n_trans,
    )
    class_log_z_stack = np.stack(
        [np.asarray(class_log_z, dtype=np.float64) for class_log_z in class_log_z_values],
        axis=1,
    )

    flat_indices = np.arange(n_classes * n_rot * n_trans, dtype=np.int32)
    class_indices_flat = (flat_indices // (n_rot * n_trans)).astype(np.int32)
    rot_indices_flat = ((flat_indices % (n_rot * n_trans)) // n_trans).astype(np.int32)
    trans_indices_flat = (flat_indices % n_trans).astype(np.int32)

    # Build a map from local_pos to dump-target index (row in
    # target_scores_pre_prior_per_class[c]) so we can pick the right
    # per-class raw-score slab for each saved particle.
    target_pos_to_dump_row = None
    if target_local_positions is not None:
        target_pos_to_dump_row = {int(p): row for row, p in enumerate(np.asarray(target_local_positions).tolist())}

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights_full = weights_per_class[local_pos].reshape(-1)
        sig_mask = sig_mask_full[local_pos].reshape(-1)
        sig_indices = np.flatnonzero(sig_mask).astype(np.int32)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        rot_prior_arr = (
            np.asarray(rotation_log_prior_padded, dtype=np.float64)[:, :n_rot]
            if rotation_log_prior_padded is not None
            else None
        )

        # Per-class raw scores (pre-prior and with-prior) for this image,
        # if the engine collected them. Shape per class: (n_rot, n_trans).
        scores_pre_prior_per_class = None
        scores_with_prior_per_class = None
        if target_pos_to_dump_row is not None and target_scores_pre_prior_per_class is not None:
            dump_row = target_pos_to_dump_row.get(int(local_pos))
            if dump_row is not None:
                scores_pre_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_pre_prior_per_class],
                    axis=0,
                )
                scores_with_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_with_prior_per_class],
                    axis=0,
                )

        image_rows = slice(local_pos * n_trans, (local_pos + 1) * n_trans)
        shifted_target = None
        if shifted_data is not None:
            shifted_target = np.asarray(shifted_data[image_rows], dtype=np.complex128)
        ctf2_target = None
        if ctf2_data is not None:
            ctf2_arr = np.asarray(ctf2_data)
            ctf2_target = (
                ctf2_arr[local_pos : local_pos + 1]
                if ctf2_arr.shape[0] == local_indices.shape[0]
                else ctf2_arr[image_rows]
            )

        iteration_suffix = "" if not target_iteration else f"_it{int(debug_iteration):03d}"
        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}{iteration_suffix}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        save_kwargs = dict(
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            debug_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            one_based_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_classes=np.int64(n_classes),
            n_rot=np.int64(n_rot),
            n_trans=np.int64(n_trans),
            weights_full=weights_full,
            weights_per_class=weights_per_class[local_pos],
            significant_mask=sig_mask,
            significant_indices=sig_indices,
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            class_assignment=np.int64(class_assignment_batch[local_pos]),
            normalization_log_z=np.float64(global_log_z[local_pos]),
            class_log_z=class_log_z_stack[local_pos],
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            class_indices=class_indices_flat,
            rot_indices=rot_indices_flat,
            trans_indices=trans_indices_flat,
            class_log_priors=np.asarray(class_log_priors, dtype=np.float64),
            rotation_log_prior=(rot_prior_arr if rot_prior_arr is not None else np.empty((0,), dtype=np.float64)),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            shifted_data=(
                shifted_target
                if shifted_target is not None
                else np.empty((0,), dtype=np.complex128)
            ),
            ctf2_data=(
                np.asarray(ctf2_target, dtype=np.float64)
                if ctf2_target is not None
                else np.empty((0,), dtype=np.float64)
            ),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32)
                if window_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            half_weights=(
                np.asarray(half_weights_used, dtype=np.float64)
                if half_weights_used is not None
                else np.empty((0,), dtype=np.float64)
            ),
            coarse_gaussian_unshifted_corrected=(
                np.asarray(coarse_gaussian_unshifted_corrected[local_pos], dtype=np.complex64)
                if coarse_gaussian_unshifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_shifted_corrected=(
                np.asarray(coarse_gaussian_shifted_corrected[local_pos], dtype=np.complex64)
                if coarse_gaussian_shifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_pixel_weight=(
                np.asarray(coarse_gaussian_pixel_weight[local_pos], dtype=np.float32)
                if coarse_gaussian_pixel_weight is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_initial_diff2=(
                np.asarray(coarse_gaussian_initial_diff2[local_pos], dtype=np.float32)
                if coarse_gaussian_initial_diff2 is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_score_indices=(
                np.asarray(coarse_gaussian_score_indices, dtype=np.int32)
                if coarse_gaussian_score_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            translation_phase_source=(
                np.asarray(translation_phase_source)
                if translation_phase_source is not None
                else np.empty((0, 2), dtype=np.float64)
            ),
            relion_f32_sum_weight=(
                np.float32(np.asarray(relion_f32_sum_weight)[local_pos])
                if relion_f32_sum_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_significant_weight=(
                np.float32(np.asarray(relion_f32_significant_weight)[local_pos])
                if relion_f32_significant_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_cutoff_count=(
                np.int32(np.asarray(relion_f32_cutoff_count)[local_pos])
                if relion_f32_cutoff_count is not None
                else np.int32(-1)
            ),
            score_capture_mode=np.asarray(str(score_capture_mode)),
        )
        if scores_pre_prior_per_class is not None:
            # Per-class raw recovar score (= -0.5 * residual in
            # `_e_step_block_scores`; differs from RELION's diff2 by the
            # per-image Xi2/2 constant which cancels in relative pose
            # comparisons). Shape (n_classes, n_rot, n_trans).
            save_kwargs["scores_pre_prior_per_class"] = scores_pre_prior_per_class
            save_kwargs["scores_with_prior_per_class"] = scores_with_prior_per_class
        if projected_reference_per_class is not None:
            projection_values = np.asarray(projected_reference_per_class)
            projection_ids = np.asarray(projected_reference_rotation_ids, dtype=np.int32)
            if projection_values.shape[:2] != (n_classes, projection_ids.size):
                raise ValueError(
                    "projected-reference dump must have shape "
                    f"({n_classes}, {projection_ids.size}, n_pixels), got {projection_values.shape}",
                )
            save_kwargs["projected_reference_rotation_ids"] = projection_ids
            save_kwargs["projected_reference_per_class"] = projection_values.astype(np.complex128)
            norm_scores = np.asarray(projected_reference_norm_score_per_class)
            cross_scores = np.asarray(projected_cross_score_per_class)
            expected_component_shape = (
                n_classes,
                local_indices.shape[0],
                projection_ids.size,
                n_trans,
            )
            if (
                norm_scores.shape != expected_component_shape
                or cross_scores.shape != expected_component_shape
            ):
                raise ValueError(
                    "projected score components must both have shape "
                    f"{expected_component_shape}, got "
                    f"{norm_scores.shape} and {cross_scores.shape}",
                )
            save_kwargs["projected_reference_norm_score_per_class"] = norm_scores[
                :, local_pos
            ].astype(np.float64)
            save_kwargs["projected_cross_score_per_class"] = cross_scores[
                :, local_pos
            ].astype(np.float64)
        np.savez_compressed(out_path, **save_kwargs)
        _maybe_stop_after_significance_dump(
            out_path,
            dump_dir=dump_dir,
            target_original_indices=target_original_indices,
            current_size=current_size,
            debug_iteration=debug_iteration,
        )


def _uses_relion_background_fill(experiment_dataset) -> bool:
    image_source = getattr(experiment_dataset, "image_source", None)
    while hasattr(image_source, "parent"):
        image_source = image_source.parent
    backend = getattr(image_source, "backend", image_source)
    return getattr(backend, "image_mask_mode", None) == "relion_background_fill"


@nvtx.annotate("adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _compute_significance_batched(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    score_with_masked_images=False,
    return_significant_sample_indices=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    do_gridding_correction=False,
    square_window=False,
    use_float64_scoring=False,
    projection_force_jax=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    relion_projector_texture_interp: bool | None = None,
    return_full_stats=False,
):
    """Run coarse E-step and find significant rotations in a memory-efficient way.

    Instead of materializing the full (n_images, n_rot * n_trans) weight matrix,
    this processes one image batch at a time: for each batch, it computes the
    posterior weights, finds significance, and accumulates the union of significant
    rotation indices.

    Returns
    -------
    sig_rot_any : np.ndarray, shape (n_rot,), dtype bool
        True for rotations that are significant for at least one image.
    n_sig_all : np.ndarray, shape (n_images,), dtype int32
        Per-image count of significant (rot x trans) samples.
    hard_assignments : np.ndarray, shape (n_images,), dtype int32
        Best (rot_idx * n_trans + trans_idx) per image from coarse pass.
    significant_sample_indices : list[np.ndarray], optional
        Returned only when ``return_significant_sample_indices=True``.
        ``significant_sample_indices[i]`` stores flattened
        ``rot_idx * n_trans + trans_idx`` entries kept for image ``i``.
    full_stats : dict[str, np.ndarray], optional
        Returned only when ``return_full_stats=True``.  Contains the full
        coarse-grid log normalizer and best-pose statistics before any
        significant-pose pruning.  RELION os0 uses these full-grid weights for
        Pmax / weight_norm, while ``significant_weight`` only gates
        reconstruction.
    """
    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.half_spectrum import (
        make_half_image_weights,
        make_scoring_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.image_shifts import (
        apply_relion_integer_pre_shifts,
        tiled_half_image_phase_factors,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        prepare_batch_preprocess_operands,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch as _preprocess_batch,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_projections_block as _compute_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows as _project_relion_projector_manual,
    )
    from recovar.em.dense_single_volume.helpers.scoring import (
        _e_step_block_scores,
        _e_step_block_scores_windowed,
        _update_logsumexp,
    )
    from recovar.reconstruction import noise as noise_utils

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            experiment_dataset.volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = experiment_dataset.volume_shape

    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    coarse_texture_interp = (
        _global_pass1_relion_projector_texture_enabled()
        if relion_projector_texture_interp is None
        else bool(relion_projector_texture_interp)
    )
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")

    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = window_spec.score_values(half_weights)

    use_relion_numpy_preprocess = _uses_relion_background_fill(experiment_dataset)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    norm_half_weights = make_half_image_weights(image_shape)

    def _preprocess_batch_relion_numpy(batch_data, ctf_params, batch_size):
        processed_half = experiment_dataset.process_images_half(
            np.asarray(batch_data),
            apply_image_mask=score_with_masked_images,
        )
        processed_half = jnp.asarray(processed_half)
        ctf_half = config.compute_ctf_half(ctf_params)
        ctf2_over_nv_half = ctf_half**2 / noise_variance_half
        ctf_weighted = processed_half * ctf_half / noise_variance_half
        translations_tiled = jnp.repeat(jnp.asarray(translations)[None], batch_size, axis=0).reshape(
            batch_size * n_trans,
            -1,
        )
        weighted_tiled = jnp.repeat(ctf_weighted[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans,
            -1,
        )
        shifted_half = core.translate_images(
            weighted_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )
        batch_norm = jnp.sum(
            (jnp.abs(processed_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real
        return shifted_half, batch_norm, ctf2_over_nv_half

    # Pad rotations.
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    # Accumulate results
    sig_rot_any = np.zeros(n_rot, dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [None] * n_images if return_significant_sample_indices else None
    normalization_log_z = np.empty(n_images, dtype=np.float64) if return_full_stats else None
    log_evidence = np.empty(n_images, dtype=np.float32) if return_full_stats else None
    best_log_score = np.empty(n_images, dtype=np.float32) if return_full_stats else None
    max_posterior = np.empty(n_images, dtype=np.float32) if return_full_stats else None

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}",
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got "
                    f"{translation_log_prior.shape}",
                )
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions",
            )

    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rotation_log_prior.shape != (n_rot,):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},), got {rotation_log_prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.concatenate(
                [
                    rotation_log_prior,
                    np.zeros(n_rot_padded - n_rot, dtype=np.float32),
                ]
            )
        else:
            rotation_log_prior_padded = rotation_log_prior
    else:
        rotation_log_prior_padded = None

    def _score_rotation_block_for_batch(
        *,
        rots_b,
        r0,
        r1,
        shifted_data,
        batch_norm,
        ctf2_data,
        batch_size,
        batch_translation_log_prior,
    ):
        if use_relion_projector:
            projector_kwargs = {}
            if current_size is not None:
                projector_kwargs["projector_output_size"] = int(current_size)
            if coarse_texture_interp:
                proj_half_b, proj_abs2_half_b = _compute_relion_projector_projections_block(
                    relion_projector_half,
                    jnp.asarray(rots_b),
                    image_shape,
                    r_max=int(relion_projector_r_max),
                    padding_factor=int(projection_padding_factor),
                    centered_rows=True,
                    dense_scale=True,
                    relion_texture_interp=True,
                    **projector_kwargs,
                )
            else:
                proj_half_b = _project_relion_projector_manual(
                    relion_projector_half,
                    jnp.asarray(rots_b),
                    image_shape,
                    int(relion_projector_r_max),
                    int(projection_padding_factor),
                    projector_kwargs.get("projector_output_size"),
                )
                proj_half_b = proj_half_b * _dense_projection_scale(image_shape)
                proj_abs2_half_b = jnp.abs(proj_half_b) ** 2
        else:
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )

        if use_window:
            proj_w = proj_half_b[:, window_indices]
            proj_abs2_w = proj_abs2_half_b[:, window_indices]
            if not use_float64_scoring:
                proj_w = proj_w.astype(jnp.complex64)
                proj_abs2_w = proj_abs2_w.astype(jnp.float32)
            scores = _e_step_block_scores_windowed(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_w * half_weights_windowed,
                proj_abs2_w * half_weights_windowed,
                half_weights_windowed,
                batch_size,
                n_trans,
                n_windowed,
                image_shape,
                volume_shape,
            )
        else:
            if not use_float64_scoring:
                proj_half_b = proj_half_b.astype(jnp.complex64)
                proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
            scores = _e_step_block_scores(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_half_b * half_weights,
                proj_abs2_half_b * half_weights,
                half_weights,
                batch_size,
                n_trans,
                image_shape,
                volume_shape,
            )

        if r1 > n_rot:
            valid = n_rot - r0
            pmask = jnp.arange(rotation_block_size) < valid
            scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

        scores_pre_prior = scores
        if rotation_log_prior_padded is not None:
            scores = scores + jnp.asarray(rotation_log_prior_padded[r0:r1])[None, :, None]

        if batch_translation_log_prior is not None:
            if translation_log_prior.ndim == 1:
                scores = scores + batch_translation_log_prior[None, None, :]
            else:
                scores = scores + batch_translation_log_prior[:, None, :]

        return scores, scores_pre_prior

    image_indices = np.arange(n_images)
    start_idx = 0

    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        (
            relion_cuda_preprocess,
            integer_pre_shifts,
            batch_corr_np,
            batch_scale_np,
            relion_preprocess_kwargs,
        ) = prepare_batch_preprocess_operands(
            experiment_dataset,
            batch_data,
            indices,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
        )
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied and not relion_cuda_preprocess:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
        batch_data = jnp.asarray(batch_data)
        if translation_log_prior is None:
            batch_translation_log_prior = None
        elif translation_log_prior.ndim == 1:
            batch_translation_log_prior = jnp.asarray(translation_log_prior)
        else:
            batch_translation_log_prior = jnp.asarray(
                translation_log_prior[start_idx:end_idx],
            )

        if use_relion_numpy_preprocess and not relion_cuda_preprocess:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch_relion_numpy(
                batch_data,
                ctf_params,
                batch_size,
            )
        else:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                relion_preprocess_kwargs=relion_preprocess_kwargs,
            )

        batch_scale = jnp.asarray(batch_scale_np)

        if image_corrections is not None:
            batch_corr = jnp.asarray(batch_corr_np)
            applied_corr = batch_scale if relion_cuda_preprocess else batch_corr
            corr_expanded = jnp.repeat(applied_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            # ``image_corrections`` carries ``(avg_norm/normcorr)*scale``;
            # the image-only ``|F_img|^2`` term must drop ``scale`` so it is
            # not double-counted with the reference-side ``ctf2 *= scale^2``
            # below. Matches em_engine._relion_image_correction_factors and
            # ``ml_optimiser.cpp:6240,7298,8516``.
            if not relion_cuda_preprocess:
                norm_corr = batch_corr / batch_scale
                batch_norm = batch_norm * (norm_corr**2)[:, None]

        if scale_corrections is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
            shifted_half = shifted_half * phase_expanded

        # DC exclusion (RELION parity: Minvsigma2[0] = 0)
        if half_spectrum_scoring:
            from recovar.em.dense_single_volume.helpers.half_spectrum import make_shell_indices_half as _mshi

            dc_shell = _mshi(image_shape)
            dc_mask = dc_shell == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = ctf2_over_nv_half

        if use_float64_scoring:
            shifted_half = shifted_half.astype(jnp.complex128)
            ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float64)
            if use_window:
                shifted_data = shifted_data.astype(jnp.complex128)
                ctf2_data = ctf2_data.astype(jnp.float64)
            else:
                shifted_data = shifted_half
                ctf2_data = ctf2_over_nv_half
        else:
            # Diagnostic path for RELION's accelerated kernels: XFLOAT is
            # float unless RELION is compiled with ACC_DOUBLE_PRECISION.
            shifted_half = shifted_half.astype(jnp.complex64)
            ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float32)
            if use_window:
                shifted_data = shifted_data.astype(jnp.complex64)
                ctf2_data = ctf2_data.astype(jnp.float32)
            else:
                shifted_data = shifted_half
                ctf2_data = ctf2_over_nv_half

        dump_target_positions = None
        dump_score_pre_prior_blocks = None
        dump_score_with_prior_blocks = None
        debug_dump_enabled = _significance_debug_dump_matches(
            current_size=current_size,
            debug_iteration=None,
        )
        if debug_dump_enabled:
            target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if target_original_indices:
                local_indices_for_dump = np.asarray(indices, dtype=np.int64)
                original_indices_for_dump = _original_indices_for_local(experiment_dataset, local_indices_for_dump)
                dump_target_positions = np.flatnonzero(
                    np.isin(original_indices_for_dump, np.fromiter(target_original_indices, dtype=np.int64))
                ).astype(np.int64)
                if dump_target_positions.size:
                    dump_score_pre_prior_blocks = []
                    dump_score_with_prior_blocks = []

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=jnp.float64)
        cache_score_blocks = (
            _significance_score_cache_enabled(
                batch_size,
                1,
                n_rot_padded,
                n_trans,
                use_float64_scoring=use_float64_scoring,
            )
            and not debug_dump_enabled
        )
        cached_score_blocks = [] if cache_score_blocks else None

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            scores, _ = _score_rotation_block_for_batch(
                rots_b=rots_b,
                r0=r0,
                r1=r1,
                shifted_data=shifted_data,
                batch_norm=batch_norm,
                ctf2_data=ctf2_data,
                batch_size=batch_size,
                batch_translation_log_prior=batch_translation_log_prior,
            )
            if cached_score_blocks is not None:
                cached_score_blocks.append(scores)
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: reuse pass-1 scores when memory allows, then normalize.
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            if cached_score_blocks is not None:
                scores = cached_score_blocks[b]
                scores_pre_prior = None
            else:
                scores, scores_pre_prior = _score_rotation_block_for_batch(
                    rots_b=rots_b,
                    r0=r0,
                    r1=r1,
                    shifted_data=shifted_data,
                    batch_norm=batch_norm,
                    ctf2_data=ctf2_data,
                    batch_size=batch_size,
                    batch_translation_log_prior=batch_translation_log_prior,
                )

            if dump_score_pre_prior_blocks is not None and dump_target_positions is not None:
                actual_rot = min(rotation_block_size, n_rot - r0)
                dump_score_pre_prior_blocks.append(
                    np.asarray(scores_pre_prior[dump_target_positions, :actual_rot, :], dtype=np.float64).reshape(
                        dump_target_positions.size,
                        -1,
                    )
                )
                dump_score_with_prior_blocks.append(
                    np.asarray(scores[dump_target_positions, :actual_rot, :], dtype=np.float64).reshape(
                        dump_target_positions.size,
                        -1,
                    )
                )

            probs = jnp.exp(scores - log_Z[:, None, None])

            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

            actual_rot = min(rotation_block_size, n_rot - r0)
            block_probs = probs[:, :actual_rot, :]
            batch_weights_blocks.append(block_probs.reshape(batch_size, -1))

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        if return_full_stats:
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_z_np = np.asarray(log_Z, dtype=np.float64)
            best_score_np = np.asarray(best_score, dtype=np.float64)
            normalization_log_z[start_idx:end_idx] = log_z_np
            log_evidence[start_idx:end_idx] = (log_z_np + log_score_offset).astype(np.float32)
            best_log_score[start_idx:end_idx] = (best_score_np + log_score_offset).astype(np.float32)
            max_posterior[start_idx:end_idx] = np.exp(best_score_np - log_z_np).astype(np.float32)

        # Concatenate this batch's weights -> (batch_size, n_rot * n_trans).
        batch_weights = jnp.concatenate(batch_weights_blocks, axis=1)
        dump_scores_pre_prior = (
            np.concatenate(dump_score_pre_prior_blocks, axis=1) if dump_score_pre_prior_blocks is not None else None
        )
        dump_scores_with_prior = (
            np.concatenate(dump_score_with_prior_blocks, axis=1) if dump_score_with_prior_blocks is not None else None
        )

        # Find significance for this batch
        batch_sig_mask, batch_sig_rot_mask, batch_n_sig = _find_sig(
            batch_weights,
            n_rot,
            n_trans,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

        # Accumulate global union of significant rotations
        batch_sig_rot_any = np.asarray(jnp.any(batch_sig_rot_mask, axis=0))
        sig_rot_any |= batch_sig_rot_any

        n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig)
        if debug_dump_enabled:
            batch_weights_np = np.asarray(batch_weights)
            best_score_np_for_dump = np.asarray(best_score, dtype=np.float64)
            log_z_np_for_dump = np.asarray(log_Z, dtype=np.float64)
            _maybe_dump_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                batch_weights=batch_weights_np,
                batch_sig_mask=np.asarray(batch_sig_mask, dtype=bool),
                batch_n_sig=np.asarray(batch_n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(best_argmax, dtype=np.int64),
                log_z=log_z_np_for_dump,
                best_score=best_score_np_for_dump,
                max_posterior=np.exp(best_score_np_for_dump - log_z_np_for_dump),
                rotations=rotations,
                translations=translations,
                rotation_log_prior=rotation_log_prior,
                batch_translation_log_prior=batch_translation_log_prior,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                scores_pre_prior_full=dump_scores_pre_prior,
                scores_with_prior_full=dump_scores_with_prior,
                dump_target_positions=dump_target_positions,
                shifted_data=shifted_data,
                ctf2_data=ctf2_data,
                batch_norm=batch_norm,
                window_indices=window_indices,
                half_weights_used=half_weights_windowed if use_window else half_weights,
            )
        if return_significant_sample_indices:
            batch_sig_mask_np = np.asarray(batch_sig_mask, dtype=bool)
            for local_idx, global_idx in enumerate(indices):
                significant_sample_indices[int(global_idx)] = compact_significant_sample_indices_from_mask(
                    batch_sig_mask_np[local_idx],
                )
        start_idx = end_idx

    full_stats = None
    if return_full_stats:
        full_stats = {
            "normalization_log_z": normalization_log_z,
            "log_evidence_per_image": log_evidence,
            "best_log_score_per_image": best_log_score,
            "max_posterior_per_image": max_posterior,
        }

    if return_significant_sample_indices:
        if return_full_stats:
            return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices, full_stats
        return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices
    if return_full_stats:
        return sig_rot_any, n_sig_all, hard_assignment, full_stats
    return sig_rot_any, n_sig_all, hard_assignment


@nvtx.annotate("kclass.adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _compute_k_class_significance_batched(
    experiment_dataset,
    means,
    noise_variance,
    rotations,
    translations,
    disc_type,
    *,
    class_log_priors,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    score_with_masked_images=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    do_gridding_correction=False,
    square_window=False,
    use_float64_scoring=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    relion_projector_texture_interp: bool | None = None,
    score_mode: str = "gaussian",
    collect_significance: bool = True,
    return_class_best: bool = False,
    return_class_second: bool = False,
    debug_iteration: int | None = None,
    coarse_healpix_order: int | None = None,
    coarse_rotation_ids=None,
    translation_phase_source=None,
    relion_coarse_gaussian_default: bool = False,
):
    """Find significant samples from one posterior over ``class x rotation x translation``."""

    if return_class_second and not return_class_best:
        raise ValueError("return_class_second requires return_class_best")

    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
        make_fourier_window_spec,
        relion_fftw_order_for_square_score_window,
    )
    from recovar.em.dense_single_volume.helpers.half_spectrum import (
        make_half_image_weights,
        make_scoring_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.image_shifts import (
        apply_relion_integer_pre_shifts,
        tiled_half_image_phase_factors,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        prepare_batch_preprocess_operands,
        process_half_image,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch as _preprocess_batch,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch_firstiter_cc as _preprocess_batch_firstiter_cc,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_projections_block as _compute_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows as _project_relion_projector_manual,
    )
    from recovar.em.dense_single_volume.helpers.scoring import (
        _e_step_block_scores,
        _e_step_block_scores_normalized_cc,
        _e_step_block_scores_windowed,
        _e_step_block_scores_windowed_normalized_cc,
        _relion_coarse_normalized_cc_rescore,
        _update_logsumexp,
    )
    from recovar.reconstruction import noise as noise_utils

    score_mode = str(score_mode)
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    n_classes = int(means_array.shape[0])
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    if class_log_priors_np.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {class_log_priors_np.shape}")

    rotations = np.asarray(rotations, dtype=np.float32)
    translations_source = np.asarray(
        translations if translation_phase_source is None else translation_phase_source,
    )
    translations = np.asarray(translations, dtype=np.float32)
    if translations_source.shape != translations.shape:
        raise ValueError(
            "translation_phase_source must match translations: "
            f"{translations_source.shape} != {translations.shape}",
        )
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    n_images = int(experiment_dataset.n_units)
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (n_rot,):
            raise ValueError(
                f"coarse_rotation_ids must have shape ({n_rot},), got {coarse_rotation_ids.shape}",
            )
    if coarse_healpix_order is None:
        coarse_healpix_order = _infer_relion_coarse_healpix_order(n_rot)
    elif int(coarse_healpix_order) < 0:
        raise ValueError(f"coarse_healpix_order must be non-negative, got {coarse_healpix_order}")

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim != 4 or int(relion_projector_half.shape[0]) != n_classes:
            raise ValueError(
                "relion_projector_half must have shape "
                f"({n_classes}, z, y, x_half), got {relion_projector_half.shape}",
            )
    if projection_padding_factor > 1 and not use_relion_projector:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        means_for_proj = []
        proj_volume_shape = None
        for class_index in range(n_classes):
            mean_for_proj, proj_volume_shape = pad_volume_for_projection(
                means_array[class_index],
                experiment_dataset.volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
            means_for_proj.append(mean_for_proj)
    else:
        means_for_proj = [means_array[class_index] for class_index in range(n_classes)]
        proj_volume_shape = experiment_dataset.volume_shape

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=score_mode != "normalized_cc",
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    coarse_texture_interp = (
        _global_pass1_relion_projector_texture_enabled()
        if relion_projector_texture_interp is None
        else bool(relion_projector_texture_interp)
    )
    tree_rescore_max_margin = _firstiter_cc_tree_top2_rescore_max_margin()
    # The environment setting spans the full process, while only iteration 1
    # uses normalized CC.  Later Gaussian iterations must remain unaffected.
    tree_rescore_enabled = (
        tree_rescore_max_margin is not None and score_mode == "normalized_cc"
    )
    # The environment flag spans the complete refinement process. Iteration 1
    # may use normalized CC, while this intervention applies only to later
    # Gaussian coarse passes. Keep the flag dormant for the CC call instead of
    # rejecting the process before it reaches the intended boundary.
    coarse_gaussian_ffi_requested = _k1_coarse_gaussian_ffi_enabled(
        default=relion_coarse_gaussian_default,
    )
    coarse_gaussian_ffi_enabled = (
        coarse_gaussian_ffi_requested and score_mode == "gaussian"
    )
    coarse_gaussian_sincosf_requested = _k1_coarse_gaussian_sincosf_enabled(
        default=relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled,
    )
    coarse_gaussian_sincosf_enabled = (
        coarse_gaussian_sincosf_requested and score_mode == "gaussian"
    )
    if coarse_gaussian_sincosf_enabled and not coarse_gaussian_ffi_enabled:
        raise ValueError(
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1",
        )
    exact_coarse_operands_requested = _k1_relion_exact_coarse_operands_enabled(
        default=(
            relion_coarse_gaussian_default
            and coarse_gaussian_ffi_enabled
            and coarse_gaussian_sincosf_enabled
        ),
    )
    exact_coarse_operands_enabled = (
        exact_coarse_operands_requested and score_mode == "gaussian"
    )
    if exact_coarse_operands_enabled and not coarse_gaussian_sincosf_enabled:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1 and "
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV}=1",
        )
    coarse_gaussian_native_texture_requested = (
        _k1_coarse_gaussian_native_texture_enabled(
            # Keep the fused texture scorer as an explicit diagnostic.  The
            # preprojected rectangular FFI consumes the same exact image/CTF
            # operands but matches RELION's coarse support boundary more
            # closely; the fused scorer can move marginal parents across the
            # adaptive-significance cutoff.
            default=False,
        )
    )
    coarse_gaussian_native_texture_enabled = (
        coarse_gaussian_native_texture_requested and score_mode == "gaussian"
    )
    if coarse_gaussian_native_texture_enabled and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV} requires "
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    relion_f32_coarse_support_requested = _k1_relion_f32_coarse_support_enabled(
        default=relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled,
    )
    relion_f32_coarse_support_enabled = (
        relion_f32_coarse_support_requested and score_mode == "gaussian"
    )
    if relion_f32_coarse_support_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_RELION_F32_COARSE_SUPPORT_ENV} is restricted to K=1",
            )
        if use_float64_scoring:
            raise ValueError(
                f"{_K1_RELION_F32_COARSE_SUPPORT_ENV} requires production float32 scoring",
            )
        logger.warning(
            "RELION CUDA float32 coarse support enabled (%s): current_size=%d",
            (
                "guarded fresh-K=1 default"
                if relion_coarse_gaussian_default
                and _K1_RELION_F32_COARSE_SUPPORT_ENV not in os.environ
                else "environment override"
            ),
            int(image_shape[0]) if current_size is None else int(current_size),
        )
    coarse_gaussian_full_to_compact = None
    coarse_gaussian_score_indices = None
    coarse_gaussian_score_active_mask = None
    coarse_gaussian_window_positions = None
    coarse_gaussian_powerclass = None
    coarse_gaussian_projector_full = None
    if coarse_gaussian_ffi_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} is restricted to K=1"
            )
        if use_float64_scoring:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires production float32 scoring"
            )
        if not use_relion_projector or not coarse_texture_interp:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires the supplied RELION "
                "texture projector"
            )
        if not half_spectrum_scoring:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires half-spectrum scoring"
            )
        if n_trans > 128:
            raise ValueError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} supports at most 128 "
                f"translations, got {n_trans}"
            )
        from recovar import cuda_backproject
        from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
            _relion_cuda_corr_img_from_rfloat_ctf,
            _relion_cuda_fine_full_to_compact_lookup,
            _relion_cuda_pixel_correction_from_rfloat_ctf,
            _relion_cuda_powerclass_highres_xi2_half,
            _relion_exact_ctf_half_from_source_star,
        )

        if jax.default_backend() != "gpu" or not cuda_backproject.cuda_available():
            raise RuntimeError(
                f"{_K1_COARSE_GAUSSIAN_FFI_ENV} requires the custom CUDA backend"
            )
        score_size = int(image_shape[0]) if current_size is None else int(current_size)
        square_score_indices_np, square_score_count = make_fourier_window_indices_np(
            image_shape,
            score_size,
            square=True,
            include_dc=True,
        )
        expected_square_count = score_size * (score_size // 2 + 1)
        if square_score_count != expected_square_count:
            raise ValueError(
                "RELION coarse Gaussian square crop has an unexpected size: "
                f"{square_score_count} != {expected_square_count}"
            )
        coarse_gaussian_score_indices = jnp.asarray(
            square_score_indices_np,
            dtype=jnp.int32,
        )
        active_score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else np.asarray(window_spec.score_indices_np, dtype=np.int32)
        )
        coarse_gaussian_score_active_mask = jnp.asarray(
            np.isin(square_score_indices_np, active_score_indices_np),
            dtype=jnp.bool_,
        )
        coarse_gaussian_window_positions = jnp.asarray(
            _compact_projection_window_positions(
                square_score_indices_np,
                active_score_indices_np,
            ),
            dtype=jnp.int32,
        )
        coarse_gaussian_full_to_compact = jnp.asarray(
            _relion_cuda_fine_full_to_compact_lookup(
                image_shape,
                score_size,
                square_score_indices_np,
            ),
            dtype=jnp.int32,
        )
        coarse_gaussian_powerclass = _relion_cuda_powerclass_highres_xi2_half
        logger.warning(
            "K=1 RELION coarse Gaussian FFI enabled (%s): "
            "current_size=%d square_pixels=%d translations=%d",
            (
                "guarded fresh-K=1 default"
                if relion_coarse_gaussian_default
                and _K1_COARSE_GAUSSIAN_FFI_ENV not in os.environ
                else "environment override"
            ),
            score_size,
            square_score_count,
            n_trans,
        )
        if coarse_gaussian_sincosf_enabled:
            logger.warning(
                "K=1 RELION coarse CUDA sincosf translation enabled (%s): "
                "current_size=%d square_pixels=%d translations=%d",
                (
                    "guarded fresh-K=1 default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_GAUSSIAN_SINCOSF_ENV not in os.environ
                    else "environment override"
                ),
                score_size,
                square_score_count,
                n_trans,
            )
        if exact_coarse_operands_enabled:
            logger.warning(
                "K=1 RELION exact coarse operands enabled (%s): per-image FFT, "
                "RFLOAT CTF division/square, and binary64-to-XFLOAT inverse noise",
                (
                    "guarded fresh-K=1 default"
                    if relion_coarse_gaussian_default
                    and _K1_RELION_EXACT_COARSE_OPERANDS_ENV not in os.environ
                    else "environment override"
                ),
            )
        if coarse_gaussian_native_texture_enabled:
            from recovar.em.dense_single_volume.helpers.projection import (
                relion_projector_half_to_texture_full,
            )

            coarse_gaussian_projector_full = jnp.asarray(
                relion_projector_half_to_texture_full(relion_projector_half[0])
                * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
                dtype=jnp.complex64,
            )
            logger.warning(
                "K=1 RELION native texture coarse scoring enabled (%s): "
                "projection, translation, and score reduction share one kernel",
                (
                    "guarded fresh-K=1 default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV not in os.environ
                    else "environment override"
                ),
            )
    tree_rescore_fftw_order = None
    tree_rescore_translation_angles = None
    if tree_rescore_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} currently "
                "supports K=1 only",
            )
        if not return_class_best:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "return_class_best=True",
            )
        if use_float64_scoring:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "production float32 scoring",
            )
        if not use_relion_projector or not coarse_texture_interp:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "the supplied RELION projector with texture interpolation",
            )
        if not half_spectrum_scoring:
            raise ValueError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "half-spectrum scoring",
            )
        from recovar import cuda_backproject
        from recovar.em.dense_single_volume.helpers.projection import (
            relion_projector_half_to_texture_full,
        )
        from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
            _relion_cuda_corr_img_from_rfloat_ctf,
            _relion_cuda_pixel_correction_from_rfloat_ctf,
            _relion_exact_ctf_half_from_source_star,
            _relion_translation_angles_f32,
        )

        if (
            jax.default_backend() != "gpu"
            or not cuda_backproject.custom_cuda_requested()
            or not cuda_backproject.cuda_available()
        ):
            raise RuntimeError(
                f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                "the custom CUDA backend",
            )
        coarse_gaussian_projector_full = jnp.asarray(
            relion_projector_half_to_texture_full(relion_projector_half[0])
            * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
            dtype=jnp.complex64,
        )
        score_size = int(image_shape[0]) if current_size is None else int(current_size)
        score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else window_spec.score_indices_np
        )
        tree_rescore_fftw_order = jnp.asarray(
            relion_fftw_order_for_square_score_window(
                image_shape,
                score_size,
                score_indices_np,
            ),
            dtype=jnp.int32,
        )
        tree_rescore_translation_angles = jnp.asarray(
            _relion_translation_angles_f32(translations_source, image_shape),
            dtype=jnp.float32,
        )
        logger.warning(
            "Opt-in RELION coarse-tree top-2 rescore enabled: max_margin=%g current_size=%d",
            tree_rescore_max_margin,
            score_size,
        )
    track_class_second = return_class_second or tree_rescore_enabled
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = window_spec.score_values(half_weights)

    if coarse_gaussian_native_texture_enabled:
        # RELION's SPA coarse scorer launches one particle over the complete
        # orientation set.  Splitting rotations changes the float32 atomic
        # accumulation schedule at adaptive-significance boundaries.
        rotation_block_size = n_rot
        logger.warning(
            "K=1 RELION native texture coarse diagnostic: one particle per "
            "full orientation grid (%d rotations)",
            n_rot,
        )

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
                )
        else:
            raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    norm_half_weights = make_half_image_weights(image_shape)
    use_relion_numpy_preprocess = _uses_relion_background_fill(experiment_dataset)

    def _preprocess_batch_relion_numpy(batch_data, ctf_params, batch_size):
        processed_half = experiment_dataset.process_images_half(
            np.asarray(batch_data),
            apply_image_mask=score_with_masked_images,
        )
        processed_half = jnp.asarray(processed_half)
        ctf_half = config.compute_ctf_half(ctf_params)
        ctf2_over_nv_half = ctf_half**2 / noise_variance_half
        ctf_weighted = processed_half * ctf_half / noise_variance_half
        translations_tiled = jnp.repeat(jnp.asarray(translations)[None], batch_size, axis=0).reshape(
            batch_size * n_trans,
            -1,
        )
        weighted_tiled = jnp.repeat(ctf_weighted[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans,
            -1,
        )
        shifted_half = core.translate_images(
            weighted_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )
        batch_norm = jnp.sum(
            (jnp.abs(processed_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real
        return shifted_half, batch_norm, ctf2_over_nv_half

    coarse_gaussian_shifted_corrected = None
    coarse_gaussian_unshifted_corrected = None
    coarse_gaussian_translation_angles = None
    coarse_gaussian_pixel_weight = None
    coarse_gaussian_initial_diff2 = None

    # The texture projector naturally produces a centered current-size crop.
    # Ask it only for the rows consumed by the scorer instead of scattering the
    # crop into a full image and immediately gathering the same rows again.
    # This is an exact index remapping and avoids a large transient scatter for
    # global rotation blocks.
    projector_compact_indices = None
    if use_relion_projector and coarse_texture_interp:
        if coarse_gaussian_ffi_enabled:
            projector_compact_indices = coarse_gaussian_score_indices
        elif use_window:
            projector_compact_indices = window_indices
    projector_returns_compact = projector_compact_indices is not None

    def _project_block(class_index, mean_for_proj, rots_b):
        if use_relion_projector:
            projector_kwargs = {}
            if current_size is not None:
                projector_kwargs["projector_output_size"] = int(current_size)
            if projector_returns_compact:
                projector_kwargs["pixel_indices"] = projector_compact_indices
            if coarse_texture_interp:
                proj_half_b, proj_abs2_half_b = _compute_relion_projector_projections_block(
                    relion_projector_half[class_index],
                    rots_b,
                    image_shape,
                    r_max=int(relion_projector_r_max),
                    padding_factor=int(projection_padding_factor),
                    centered_rows=True,
                    dense_scale=True,
                    relion_texture_interp=True,
                    **projector_kwargs,
                )
            else:
                proj_half_b = _project_relion_projector_manual(
                    relion_projector_half[class_index],
                    rots_b,
                    image_shape,
                    int(relion_projector_r_max),
                    int(projection_padding_factor),
                    projector_kwargs.get("projector_output_size"),
                )
                proj_half_b = proj_half_b * _dense_projection_scale(image_shape)
                proj_abs2_half_b = jnp.abs(proj_half_b) ** 2
        else:
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
        return proj_half_b, proj_abs2_half_b

    def _score_block(class_index, mean_for_proj, rots_b, shifted_data, batch_norm, ctf2_data, batch_size):
        if coarse_gaussian_native_texture_enabled:
            from recovar import cuda_backproject

            diff2 = cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
                coarse_gaussian_projector_full,
                jnp.asarray(rots_b, dtype=jnp.float32),
                coarse_gaussian_unshifted_corrected,
                coarse_gaussian_translation_angles,
                coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2,
                coarse_gaussian_full_to_compact,
                int(image_shape[0]) if current_size is None else int(current_size),
                int(projection_padding_factor),
                int(relion_projector_r_max),
            )
            return -diff2
        proj_half_b, proj_abs2_half_b = _project_block(class_index, mean_for_proj, rots_b)
        if coarse_gaussian_ffi_enabled:
            from recovar import cuda_backproject

            proj_score = (
                proj_half_b
                if projector_returns_compact
                else proj_half_b[:, coarse_gaussian_score_indices]
            )
            proj_score = jnp.asarray(proj_score, dtype=jnp.complex64)
            diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
                proj_score,
                coarse_gaussian_shifted_corrected,
                coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2,
                coarse_gaussian_full_to_compact,
            )
            return -diff2
        if use_window:
            if projector_returns_compact:
                proj_w = proj_half_b
                proj_abs2_w = proj_abs2_half_b
            else:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
            if not use_float64_scoring:
                proj_w = proj_w.astype(jnp.complex64)
                proj_abs2_w = proj_abs2_w.astype(jnp.float32)
            if score_mode == "normalized_cc":
                return _e_step_block_scores_windowed_normalized_cc(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            return _e_step_block_scores_windowed(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_w * half_weights_windowed,
                proj_abs2_w * half_weights_windowed,
                half_weights_windowed,
                batch_size,
                n_trans,
                n_windowed,
                image_shape,
                volume_shape,
            )
        if not use_float64_scoring:
            proj_half_b = proj_half_b.astype(jnp.complex64)
            proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
        if score_mode == "normalized_cc":
            return _e_step_block_scores_normalized_cc(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_half_b * half_weights,
                proj_abs2_half_b * half_weights,
                batch_size,
                n_trans,
                image_shape,
                volume_shape,
            )
        return _e_step_block_scores(
            shifted_data,
            batch_norm,
            ctf2_data,
            proj_half_b * half_weights,
            proj_abs2_half_b * half_weights,
            half_weights,
            batch_size,
            n_trans,
            image_shape,
            volume_shape,
        )

    def _add_priors(scores, class_index, r0, r1, batch_translation_log_prior):
        if score_mode == "normalized_cc":
            return scores
        scores = scores + jnp.asarray(class_log_priors_np[class_index], dtype=scores.real.dtype)
        if rotation_log_prior_padded is not None:
            scores = scores + jnp.asarray(rotation_log_prior_padded[class_index, r0:r1])[None, :, None]
        if batch_translation_log_prior is not None:
            if translation_log_prior.ndim == 1:
                scores = scores + batch_translation_log_prior[None, None, :]
            else:
                scores = scores + batch_translation_log_prior[:, None, :]
        return scores

    sig_rot_any = np.zeros((n_classes, n_rot), dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    cutoff_count_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    class_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [[None] * n_images for _ in range(n_classes)] if collect_significance else None
    normalization_log_z = np.empty(n_images, dtype=np.float64)
    normalization_log_evidence = np.empty(n_images, dtype=np.float64)
    log_evidence = np.empty(n_images, dtype=np.float32)
    best_log_score = np.empty(n_images, dtype=np.float32)
    max_posterior = np.empty(n_images, dtype=np.float32)
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_best_log_score = (
        np.empty((n_classes, n_images), dtype=np.float32) if return_class_best else None
    )
    class_second_best_log_score = (
        np.empty((n_classes, n_images), dtype=np.float32) if return_class_second else None
    )
    # Diagnostic-only native scores before the large, class-common image
    # normalization offset.  The offset is useful for absolute log evidence,
    # but adding it before a float32 cast can erase class and pose margins.
    class_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=np.float32) if return_class_best else None
    )
    class_second_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=np.float32) if return_class_second else None
    )
    class_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_best else None
    )
    class_second_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_second else None
    )
    tree_rescore_examined = 0
    tree_rescore_ambiguous = 0
    tree_rescore_winner_changes = 0
    tree_rescore_exact_ties = 0

    start_idx = 0
    image_indices = np.arange(n_images)
    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        (
            relion_cuda_preprocess,
            integer_pre_shifts,
            batch_corr_np,
            batch_scale_np,
            relion_preprocess_kwargs,
        ) = prepare_batch_preprocess_operands(
            experiment_dataset,
            batch_data,
            indices,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
        )
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied and not relion_cuda_preprocess:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
        batch_data = jnp.asarray(batch_data)
        if translation_log_prior is None:
            batch_translation_log_prior = None
        elif translation_log_prior.ndim == 1:
            batch_translation_log_prior = jnp.asarray(translation_log_prior)
        else:
            batch_translation_log_prior = jnp.asarray(translation_log_prior[start_idx:end_idx])

        if score_mode == "normalized_cc":
            cc_window_indices = window_indices if use_window else None
            score_complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
            score_real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
            cc_preprocess_result = _preprocess_batch_firstiter_cc(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                window_indices=cc_window_indices,
                score_complex_dtype=score_complex_dtype,
                score_real_dtype=score_real_dtype,
                norm_real_dtype=jnp.float64,
                relion_preprocess_kwargs=relion_preprocess_kwargs,
                return_unshifted_score_weighted=tree_rescore_enabled,
            )
            if tree_rescore_enabled:
                (
                    shifted_half,
                    batch_norm,
                    ctf2_half_score,
                    ctf2_over_nv_half,
                    tree_rescore_unshifted_half,
                ) = cc_preprocess_result
            else:
                shifted_half, batch_norm, ctf2_half_score, ctf2_over_nv_half = (
                    cc_preprocess_result
                )
        elif use_relion_numpy_preprocess and not relion_cuda_preprocess:
            if coarse_gaussian_sincosf_enabled:
                raise ValueError(
                    f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV} requires the "
                    "production half-image preprocessing path",
                )
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch_relion_numpy(
                batch_data,
                ctf_params,
                batch_size,
            )
        else:
            preprocess_result = _preprocess_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                relion_preprocess_kwargs=relion_preprocess_kwargs,
                return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled,
            )
            if coarse_gaussian_sincosf_enabled:
                (
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    coarse_gaussian_unshifted_score_weighted,
                ) = preprocess_result
            else:
                shifted_half, batch_norm, ctf2_over_nv_half = preprocess_result
        batch_scale = jnp.asarray(batch_scale_np)
        if image_corrections is not None:
            batch_corr = jnp.asarray(batch_corr_np)
            applied_corr = batch_scale if relion_cuda_preprocess else batch_corr
            corr_expanded = jnp.repeat(applied_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half * applied_corr[:, None]
                )
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = (
                    coarse_gaussian_unshifted_score_weighted
                    * applied_corr[:, None]
                )
            # ``image_corrections`` carries ``(avg_norm/normcorr) * scale``;
            # ``scale_corrections`` carries ``scale``. The image-only
            # ``|F_img|^2`` term must be weighted by ``(avg_norm/normcorr)^2``
            # alone — divide ``batch_corr`` by ``batch_scale`` to isolate it.
            # Otherwise ``batch_norm`` picks up an extra ``scale^2`` that is
            # already accounted for on the reference side via
            # ``ctf2_over_nv_half *= batch_scale^2`` below, double-counting
            # ``scale^2`` in the Wiener score offset. See
            # ``em_engine._relion_image_correction_factors`` and
            # ``ml_optimiser.cpp:6240,7298,8516``.
            if not relion_cuda_preprocess:
                norm_corr = batch_corr / batch_scale
                batch_norm = batch_norm * (norm_corr**2)[:, None]
        if scale_corrections is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if score_mode == "normalized_cc":
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]
        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            shifted_half = shifted_half * tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half
                    * tiled_half_image_phase_factors(image_shape, batch_shifts, 1)
                )
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = (
                    coarse_gaussian_unshifted_score_weighted
                    * tiled_half_image_phase_factors(
                        image_shape,
                        batch_shifts,
                        1,
                    )
                )
        if score_mode == "normalized_cc":
            inv_xi2 = 1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))
            score_weight_half = ctf2_half_score * inv_xi2
            shifted_half = shifted_half * jnp.repeat(inv_xi2, n_trans, axis=0)
            if tree_rescore_enabled:
                tree_rescore_unshifted_half = (
                    tree_rescore_unshifted_half * inv_xi2
                )
        else:
            score_weight_half = ctf2_over_nv_half
        if half_spectrum_scoring and score_mode != "normalized_cc":
            from recovar.em.dense_single_volume.helpers.half_spectrum import make_shell_indices_half as _mshi

            dc_mask = _mshi(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            score_weight_half = jnp.where(dc_mask[None, :], 0.0, score_weight_half)
            if coarse_gaussian_sincosf_enabled:
                coarse_gaussian_unshifted_score_weighted = jnp.where(
                    dc_mask[None, :],
                    0.0,
                    coarse_gaussian_unshifted_score_weighted,
                )
        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = score_weight_half[:, window_indices]
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_half[
                    :, window_indices
                ]
        else:
            shifted_data = shifted_half
            ctf2_data = score_weight_half
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_half
        if use_float64_scoring:
            shifted_data = shifted_data.astype(jnp.complex128)
            ctf2_data = ctf2_data.astype(jnp.float64)
        else:
            shifted_data = shifted_data.astype(jnp.complex64)
            ctf2_data = ctf2_data.astype(jnp.float32)
            if score_mode == "normalized_cc" and tree_rescore_enabled:
                tree_rescore_unshifted_data = tree_rescore_unshifted_data.astype(
                    jnp.complex64
                )

        if score_mode == "normalized_cc" and tree_rescore_enabled:
            if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
                raise ValueError(
                    f"{_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN_ENV} requires "
                    "RELION CUDA image preprocessing",
                )
            exact_cc_preprocess_kwargs = dict(relion_preprocess_kwargs)
            exact_cc_preprocess_kwargs["relion_fft_per_image"] = True
            exact_cc_processed = process_half_image(
                experiment_dataset,
                batch_data,
                score_with_masked_images,
                relion_preprocess_kwargs=exact_cc_preprocess_kwargs,
            )
            exact_cc_inv_xi2 = _relion_cc_inverse_power_from_processed(
                exact_cc_processed,
                window_indices if use_window else None,
            )
            exact_cc_ctf_rfloat = _relion_exact_ctf_half_from_source_star(
                experiment_dataset,
                indices,
                image_shape,
            )
            batch_scale_f32 = jnp.asarray(batch_scale_np, dtype=jnp.float32)
            exact_cc_pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                batch_scale_f32[:, None],
                exact_cc_ctf_rfloat,
            )
            exact_cc_unshifted_corrected = jnp.asarray(
                exact_cc_processed * exact_cc_pixel_correction,
                dtype=jnp.complex64,
            )
            if image_pre_shifts is not None and not real_space_pre_shift_applied:
                exact_cc_unshifted_corrected = (
                    exact_cc_unshifted_corrected
                    * tiled_half_image_phase_factors(image_shape, batch_shifts, 1)
                )
            exact_cc_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
                exact_cc_inv_xi2,
                exact_cc_ctf_rfloat,
                batch_scale_f32[:, None] if scale_corrections is not None else None,
            )
            if use_window:
                tree_rescore_unshifted_data = exact_cc_unshifted_corrected[
                    :, window_indices
                ]
                tree_rescore_corr_img_data = exact_cc_corr_img[:, window_indices]
            else:
                tree_rescore_unshifted_data = exact_cc_unshifted_corrected
                tree_rescore_corr_img_data = exact_cc_corr_img

        if coarse_gaussian_ffi_enabled:
            coarse_preprocess_kwargs = relion_preprocess_kwargs
            if exact_coarse_operands_enabled:
                if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
                    raise ValueError(
                        f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires "
                        "RELION CUDA image preprocessing",
                    )
                coarse_preprocess_kwargs = dict(relion_preprocess_kwargs)
                coarse_preprocess_kwargs["relion_fft_per_image"] = True
            processed_direct = process_half_image(
                experiment_dataset,
                batch_data,
                score_with_masked_images,
                relion_preprocess_kwargs=coarse_preprocess_kwargs,
            )
            processed_for_powerclass = processed_direct
            if image_corrections is not None and not relion_cuda_preprocess:
                image_only_correction = batch_corr / batch_scale
                processed_for_powerclass = (
                    processed_for_powerclass * image_only_correction[:, None]
                )
            # Reuse the exact operands of RECOVAR's accepted Gaussian score
            # boundary. Reprocessing and translating a second image copy here
            # changed the operands as well as the reduction. Algebraically,
            # ``shifted_half / score_weight_half`` is the shifted image divided
            # by CTF, and RELION's square-difference weight is
            # ``score_weight_half * half_weights``.
            if coarse_gaussian_sincosf_enabled:
                (
                    coarse_gaussian_shifted_corrected,
                    coarse_gaussian_pixel_weight,
                    coarse_gaussian_unshifted_corrected,
                ) = _relion_coarse_gaussian_square_operands_sincosf(
                    coarse_gaussian_unshifted_score_weighted,
                    score_weight_half,
                    half_weights,
                    coarse_gaussian_score_indices,
                    coarse_gaussian_score_active_mask,
                    translations,
                    image_shape,
                    translation_phase_source=translations_source,
                    return_unshifted=True,
                )
            else:
                (
                    coarse_gaussian_shifted_corrected,
                    coarse_gaussian_pixel_weight,
                ) = _relion_coarse_gaussian_square_operands(
                    shifted_half,
                    score_weight_half,
                    half_weights,
                    coarse_gaussian_score_indices,
                    coarse_gaussian_score_active_mask,
                    batch_size=batch_size,
                    n_trans=n_trans,
                )
            if exact_coarse_operands_enabled:
                ctf_half_rfloat = _relion_exact_ctf_half_from_source_star(
                    experiment_dataset,
                    indices,
                    image_shape,
                )
                batch_scale_f32 = jnp.asarray(batch_scale_np, dtype=jnp.float32)
                pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                    batch_scale_f32[:, None],
                    ctf_half_rfloat,
                )
                exact_unshifted_corrected = processed_direct * pixel_correction
                exact_unshifted_corrected = exact_unshifted_corrected[
                    :, coarse_gaussian_score_indices
                ]
                exact_unshifted_corrected = jnp.where(
                    coarse_gaussian_score_active_mask[None, :],
                    exact_unshifted_corrected,
                    jnp.zeros((), dtype=exact_unshifted_corrected.dtype),
                ).astype(jnp.complex64)
                from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
                    _relion_translation_angles_f32,
                )

                coarse_gaussian_translation_angles = jnp.asarray(
                    _relion_translation_angles_f32(translations_source, image_shape),
                    dtype=jnp.float32,
                )
                coarse_gaussian_shifted_corrected = cuda_backproject.relion_translate_score_f32(
                    exact_unshifted_corrected,
                    coarse_gaussian_translation_angles,
                    coarse_gaussian_score_indices,
                    image_shape,
                ).reshape(batch_size, n_trans, -1)
                inverse_noise_half = jnp.reciprocal(
                    jnp.asarray(noise_variance_half, dtype=jnp.float64),
                ).astype(jnp.float32)
                exact_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
                    inverse_noise_half[None, :],
                    ctf_half_rfloat,
                    batch_scale_f32[:, None] if scale_corrections is not None else None,
                )
                exact_square_corr_img = exact_corr_img[:, coarse_gaussian_score_indices]
                exact_square_corr_img = jnp.where(
                    coarse_gaussian_score_active_mask[None, :],
                    exact_square_corr_img,
                    jnp.zeros((), dtype=exact_square_corr_img.dtype),
                )
                coarse_gaussian_pixel_weight = jnp.asarray(
                    exact_square_corr_img
                    * jnp.asarray(half_weights[coarse_gaussian_score_indices], dtype=jnp.float32)[None, :],
                    dtype=jnp.float32,
                )
                coarse_gaussian_unshifted_corrected = exact_unshifted_corrected
            coarse_gaussian_initial_diff2 = coarse_gaussian_powerclass(
                processed_for_powerclass,
                image_shape=image_shape,
                current_size=current_size,
            )

        # Identify per-batch dump target rows so we can record raw scores
        # (pre-prior) for each target image inside the per-class block loop.
        # This enables direct diff against RELION's exp_Mweight_diff2
        # without needing the full (batch, n_classes, n_rot*n_trans) cache.
        debug_dump_enabled = collect_significance and _significance_debug_dump_matches(
            current_size=current_size,
            debug_iteration=debug_iteration,
        )
        dump_target_local_positions = None
        if debug_dump_enabled:
            _dump_targets = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if _dump_targets:
                _local_for_dump = np.asarray(indices, dtype=np.int64)
                _orig = _original_indices_for_local(experiment_dataset, _local_for_dump)
                _positions = np.flatnonzero(np.isin(_orig, np.fromiter(_dump_targets, dtype=np.int64)))
                if _positions.size:
                    dump_target_local_positions = _positions.astype(np.int64)
        # Per-class collectors for raw (pre-prior) score blocks at target rows.
        # Shape after concat per class: (n_targets, n_rot, n_trans)
        dump_target_pre_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )
        dump_target_with_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )
        passive_score_dump = bool(
            dump_target_local_positions is not None
            and os.environ.get(_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV) == "1"
        )
        passive_raw_score_blocks_per_class = (
            [[] for _ in range(n_classes)] if passive_score_dump else None
        )
        if passive_score_dump:
            # Do not materialize score blocks while production support is
            # being computed. Near an atomic cutoff that observation can
            # perturb the execution under diagnosis. Retain device buffers
            # and write them only after cached-score support selection.
            dump_target_pre_prior_blocks_per_class = None
            dump_target_with_prior_blocks_per_class = None

        global_max = jnp.full(batch_size, -jnp.inf)
        global_sum = jnp.zeros(batch_size, dtype=jnp.float64)
        class_max_values = []
        class_sum_values = []
        best_score_batch = jnp.full(batch_size, -jnp.inf)
        best_argmax_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        best_class_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        class_best_scores = [jnp.full(batch_size, -jnp.inf) for _ in range(n_classes)] if return_class_best else None
        class_best_argmaxes = [jnp.zeros(batch_size, dtype=jnp.int32) for _ in range(n_classes)] if return_class_best else None
        class_second_best_scores = (
            [jnp.full(batch_size, -jnp.inf) for _ in range(n_classes)] if track_class_second else None
        )
        class_second_best_argmaxes = (
            [jnp.zeros(batch_size, dtype=jnp.int32) for _ in range(n_classes)] if track_class_second else None
        )
        cache_score_blocks = collect_significance and _significance_score_cache_enabled(
            batch_size,
            n_classes,
            n_rot_padded,
            n_trans,
            use_float64_scoring=use_float64_scoring,
        )
        cached_class_score_blocks = [] if cache_score_blocks else None
        if passive_score_dump and cached_class_score_blocks is None:
            raise RuntimeError(
                f"{_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV}=1 requires the "
                "production significance score cache"
            )

        # ``RECOVAR_PASS1_FUSED=1`` swaps the per-block 4-5 separate JIT
        # dispatches (project/score, padding-mask, add-priors, 2× logsumexp)
        # for one fused @jit call. Bit-identical when active; disabled if any
        # debug-dump path is on so per-block pre/post-prior captures still
        # have access to intermediate scores.
        use_fused_pass1 = (
            _pass1_fused_enabled()
            and score_mode == "gaussian"
            and not use_relion_projector
            and dump_target_pre_prior_blocks_per_class is None
            and dump_target_with_prior_blocks_per_class is None
        )
        if passive_score_dump and use_fused_pass1:
            raise RuntimeError(
                f"{_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV}=1 does not support "
                "the fused pass-1 diagnostic path"
            )

        # Precompute fused-path inputs once per batch (constant across class/block).
        if use_fused_pass1:
            _fused_half_weights = half_weights_windowed if use_window else half_weights
            _fused_max_r_static = projection_kwargs.get("max_r", None) if use_window else None
            _fused_window_indices = window_indices if use_window else jnp.zeros(0, dtype=jnp.int32)
            if translation_log_prior is None:
                _fused_trans_lp_per_image = jnp.zeros((batch_size, n_trans), dtype=jnp.float32)
            elif translation_log_prior.ndim == 1:
                _fused_trans_lp_per_image = jnp.broadcast_to(
                    jnp.asarray(batch_translation_log_prior, dtype=jnp.float32),
                    (batch_size, n_trans),
                )
            else:
                _fused_trans_lp_per_image = jnp.asarray(batch_translation_log_prior, dtype=jnp.float32)

        for class_index, mean_for_proj in enumerate(means_for_proj):
            class_max = jnp.full(batch_size, -jnp.inf)
            class_sum = jnp.zeros(batch_size, dtype=jnp.float64)
            cached_score_blocks = [] if cached_class_score_blocks is not None else None
            for block_index in range(n_blocks):
                r0 = block_index * rotation_block_size
                r1 = r0 + rotation_block_size
                if use_fused_pass1:
                    valid_count = jnp.asarray(min(rotation_block_size, n_rot - r0), dtype=jnp.int32)
                    if rotation_log_prior_padded is None:
                        rot_lp_block = jnp.zeros(rotation_block_size, dtype=jnp.float32)
                    else:
                        rot_lp_block = jnp.asarray(rotation_log_prior_padded[class_index, r0:r1], dtype=jnp.float32)
                    scores, class_max, class_sum, global_max, global_sum = _fused_score_priors_logsumexp_block(
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        _fused_half_weights,
                        _fused_window_indices,
                        rot_lp_block,
                        _fused_trans_lp_per_image,
                        float(class_log_priors_np[class_index]),
                        valid_count,
                        class_max,
                        class_sum,
                        global_max,
                        global_sum,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        volume_shape=volume_shape,
                        disc_type=disc_type,
                        use_window=use_window,
                        use_float64_scoring=use_float64_scoring,
                        rotation_block_size=int(rotation_block_size),
                        batch_size=int(batch_size),
                        n_trans=int(n_trans),
                        n_windowed=int(n_windowed) if use_window else 0,
                        max_r_static=_fused_max_r_static,
                    )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                else:
                    scores = _score_block(
                        class_index,
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        batch_size,
                    )
                    if r1 > n_rot:
                        valid = n_rot - r0
                        scores = jnp.where(jnp.arange(rotation_block_size)[None, :, None] < valid, scores, -jnp.inf)
                    if passive_raw_score_blocks_per_class is not None:
                        passive_raw_score_blocks_per_class[class_index].append(scores)
                    # Capture pre-prior raw scores for dump targets BEFORE _add_priors.
                    # scores shape: (batch_size, rotation_block_size, n_trans).
                    # For comparison vs RELION exp_Mweight_diff2, recovar's score is
                    # -0.5 * residual where residual = sum_pixel((proj*ctf - shifted_img)² - |img|²)
                    # / sigma² × half_weights. RELION's diff2 has the same core term
                    # plus the per-image Xi2/2 constant. Per-pose RELATIVE differences
                    # cancel the constant, so direct diff is meaningful.
                    if dump_target_pre_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_pre_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    if dump_target_with_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_with_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
                    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
                flat_scores = scores.reshape(batch_size, -1)
                block_best = jnp.max(flat_scores, axis=1)
                block_argmax = jnp.argmax(flat_scores, axis=1)
                improved = block_best > best_score_batch
                best_score_batch = jnp.where(improved, block_best, best_score_batch)
                best_argmax_batch = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax_batch)
                best_class_batch = jnp.where(improved, class_index, best_class_batch)
                if return_class_best:
                    previous_best = class_best_scores[class_index]
                    previous_best_argmax = class_best_argmaxes[class_index]
                    class_improved = block_best > previous_best
                    if track_class_second:
                        if flat_scores.shape[1] < 2:
                            raise RuntimeError("class runner-up diagnostic requires at least two poses per block")
                        rows = jnp.arange(batch_size)
                        block_without_best = flat_scores.at[rows, block_argmax].set(-jnp.inf)
                        block_second = jnp.max(block_without_best, axis=1)
                        block_second_argmax = jnp.argmax(block_without_best, axis=1)
                        previous_second = class_second_best_scores[class_index]
                        previous_second_argmax = class_second_best_argmaxes[class_index]

                        improved_second_from_previous = previous_best >= block_second
                        improved_second = jnp.where(
                            improved_second_from_previous,
                            previous_best,
                            block_second,
                        )
                        improved_second_argmax = jnp.where(
                            improved_second_from_previous,
                            previous_best_argmax,
                            block_second_argmax + r0 * n_trans,
                        )
                        retained_second_from_previous = previous_second >= block_best
                        retained_second = jnp.where(
                            retained_second_from_previous,
                            previous_second,
                            block_best,
                        )
                        retained_second_argmax = jnp.where(
                            retained_second_from_previous,
                            previous_second_argmax,
                            block_argmax + r0 * n_trans,
                        )
                        class_second_best_scores[class_index] = jnp.where(
                            class_improved,
                            improved_second,
                            retained_second,
                        )
                        class_second_best_argmaxes[class_index] = jnp.where(
                            class_improved,
                            improved_second_argmax,
                            retained_second_argmax,
                        )
                    class_best_scores[class_index] = jnp.where(
                        class_improved,
                        block_best,
                        previous_best,
                    )
                    class_best_argmaxes[class_index] = jnp.where(
                        class_improved,
                        block_argmax + r0 * n_trans,
                        previous_best_argmax,
                    )
            if cached_class_score_blocks is not None:
                cached_class_score_blocks.append(cached_score_blocks)
            class_max_values.append(class_max)
            class_sum_values.append(class_sum)

        if tree_rescore_enabled:
            best_scores_np = np.asarray(class_best_scores[0], dtype=np.float32)
            second_scores_np = np.asarray(class_second_best_scores[0], dtype=np.float32)
            score_margins = best_scores_np - second_scores_np
            ambiguous_rows = np.flatnonzero(
                np.isfinite(score_margins) & (score_margins <= tree_rescore_max_margin)
            ).astype(np.int32)
            tree_rescore_examined += int(batch_size)
            tree_rescore_ambiguous += int(ambiguous_rows.size)
            if ambiguous_rows.size:
                best_pose_np = np.asarray(class_best_argmaxes[0], dtype=np.int32)[ambiguous_rows]
                second_pose_np = np.asarray(class_second_best_argmaxes[0], dtype=np.int32)[
                    ambiguous_rows
                ]
                candidate_pose_ids = np.sort(
                    np.stack([best_pose_np, second_pose_np], axis=1),
                    axis=1,
                )
                candidate_rotation_ids = candidate_pose_ids // n_trans
                candidate_translation_ids = candidate_pose_ids % n_trans
                candidate_rotations = jnp.asarray(
                    rotations[candidate_rotation_ids.reshape(-1)],
                    dtype=jnp.float32,
                ).reshape(
                    ambiguous_rows.size,
                    2,
                    3,
                    3,
                )
                unshifted_candidates = jnp.broadcast_to(
                    tree_rescore_unshifted_data[
                        jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                    ],
                    (
                        ambiguous_rows.size,
                        2,
                        tree_rescore_unshifted_data.shape[-1],
                    ),
                )
                candidate_translation_angles = tree_rescore_translation_angles[
                    jnp.asarray(candidate_translation_ids, dtype=jnp.int32)
                ]
                score_weight_candidates = jnp.broadcast_to(
                    tree_rescore_corr_img_data[
                        jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                    ],
                    unshifted_candidates.shape,
                )
                rescored_candidates = _relion_coarse_normalized_cc_rescore(
                    unshifted_candidates,
                    score_weight_candidates,
                    None,
                    half_weights_windowed if use_window else half_weights,
                    tree_rescore_fftw_order,
                    projector_full=coarse_gaussian_projector_full,
                    rotation_matrices=candidate_rotations,
                    translation_angles=candidate_translation_angles,
                    current_size=score_size,
                    padding_factor=projection_padding_factor,
                    projector_max_r=relion_projector_r_max,
                    numerator_weight_candidates=score_weight_candidates,
                )
                rescored_scores_np = np.asarray(rescored_candidates, dtype=np.float32)
                rescored_winner_slot, exact_ties = _select_relion_coarse_rescore_winner_slots(
                    rescored_scores_np,
                    candidate_pose_ids,
                    n_trans=n_trans,
                    healpix_order=coarse_healpix_order,
                    coarse_rotation_ids=coarse_rotation_ids,
                )
                _maybe_dump_tree_rescore_batch(
                    experiment_dataset=experiment_dataset,
                    indices=indices,
                    ambiguous_rows=ambiguous_rows,
                    candidate_pose_ids=candidate_pose_ids,
                    original_best_pose=best_pose_np,
                    original_best_score=best_scores_np[ambiguous_rows],
                    original_second_pose=second_pose_np,
                    original_second_score=second_scores_np[ambiguous_rows],
                    rescored_scores=rescored_scores_np,
                    rescored_winner_slot=rescored_winner_slot,
                    shifted_candidates=unshifted_candidates,
                    score_weight_candidates=score_weight_candidates,
                    numerator_weight_candidates=score_weight_candidates,
                    rotation_matrices=candidate_rotations,
                    translation_angles=candidate_translation_angles,
                    n_trans=n_trans,
                    half_weights=(
                        half_weights_windowed if use_window else half_weights
                    ),
                    packed_to_compact=tree_rescore_fftw_order,
                    projector_full=coarse_gaussian_projector_full,
                    current_size=score_size,
                    padding_factor=projection_padding_factor,
                    projector_max_r=relion_projector_r_max,
                    debug_iteration=debug_iteration,
                )
                tree_rescore_exact_ties += exact_ties
                row_ids = np.arange(ambiguous_rows.size, dtype=np.int32)
                rescored_runner_slot = 1 - rescored_winner_slot
                rescored_winner_pose = candidate_pose_ids[row_ids, rescored_winner_slot]
                rescored_runner_pose = candidate_pose_ids[row_ids, rescored_runner_slot]
                rescored_winner_score = rescored_scores_np[row_ids, rescored_winner_slot]
                rescored_runner_score = rescored_scores_np[row_ids, rescored_runner_slot]
                tree_rescore_winner_changes += int(
                    np.count_nonzero(rescored_winner_pose != best_pose_np)
                )
                applied_rows = np.arange(ambiguous_rows.size, dtype=np.int32)
                if applied_rows.size:
                    rows_jax = jnp.asarray(ambiguous_rows[applied_rows], dtype=jnp.int32)
                    best_argmax_batch = best_argmax_batch.at[rows_jax].set(
                        rescored_winner_pose[applied_rows]
                    )
                    best_score_batch = best_score_batch.at[rows_jax].set(
                        rescored_winner_score[applied_rows]
                    )
                    class_best_argmaxes[0] = class_best_argmaxes[0].at[rows_jax].set(
                        rescored_winner_pose[applied_rows]
                    )
                    class_best_scores[0] = class_best_scores[0].at[rows_jax].set(
                        rescored_winner_score[applied_rows]
                    )
                    class_second_best_argmaxes[0] = class_second_best_argmaxes[0].at[
                        rows_jax
                    ].set(rescored_runner_pose[applied_rows])
                    class_second_best_scores[0] = class_second_best_scores[0].at[
                        rows_jax
                    ].set(rescored_runner_score[applied_rows])

        global_log_z = global_max + jnp.log(global_sum)
        class_log_z_values = [
            class_max + jnp.log(class_sum) for class_max, class_sum in zip(class_max_values, class_sum_values)
        ]

        class_weight_mats = []
        if collect_significance:
            for class_index, mean_for_proj in enumerate(means_for_proj):
                class_weight_blocks = []
                for block_index in range(n_blocks):
                    r0 = block_index * rotation_block_size
                    r1 = r0 + rotation_block_size
                    if cached_class_score_blocks is None:
                        scores = _score_block(
                            class_index,
                            mean_for_proj,
                            rotations_padded[r0:r1],
                            shifted_data,
                            batch_norm,
                            ctf2_data,
                            batch_size,
                        )
                        if r1 > n_rot:
                            valid = n_rot - r0
                            scores = jnp.where(jnp.arange(rotation_block_size)[None, :, None] < valid, scores, -jnp.inf)
                        scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    else:
                        scores = cached_class_score_blocks[class_index][block_index]
                    actual_rot = min(rotation_block_size, n_rot - r0)
                    if relion_f32_coarse_support_enabled:
                        class_weight_blocks.append(
                            scores[:, :actual_rot, :].reshape(batch_size, -1),
                        )
                    else:
                        probs = jnp.exp(scores - global_log_z[:, None, None])
                        class_weight_blocks.append(
                            probs[:, :actual_rot, :].reshape(batch_size, -1),
                        )
                class_weight_mats.append(jnp.concatenate(class_weight_blocks, axis=1))

            batch_values = jnp.concatenate(class_weight_mats, axis=1)
            if relion_f32_coarse_support_enabled:
                from recovar.em.dense_single_volume.helpers.oversampling import (
                    relion_cuda_f32_coarse_posterior,
                )

                (
                    batch_weights,
                    batch_sig_mask,
                    batch_n_sig,
                    batch_cutoff_count,
                    _batch_sum_weight,
                    _batch_significant_weight,
                ) = relion_cuda_f32_coarse_posterior(
                    batch_values,
                    adaptive_fraction=float(adaptive_fraction),
                    max_significants=max_significants,
                )
                batch_sig_rot_mask = jnp.any(
                    batch_sig_mask.reshape(batch_size, n_classes * n_rot, n_trans),
                    axis=2,
                )
            else:
                batch_weights = batch_values
                (
                    batch_sig_mask,
                    batch_sig_rot_mask,
                    batch_n_sig,
                    batch_cutoff_count,
                ) = _find_sig(
                    batch_weights,
                    n_classes * n_rot,
                    n_trans,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    return_cutoff_count=True,
                )
            batch_sig_mask_np = np.asarray(batch_sig_mask, dtype=bool)
            sig_rot_any |= np.asarray(jnp.any(batch_sig_rot_mask, axis=0), dtype=bool).reshape(n_classes, n_rot)
            n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig, dtype=np.int32)
            cutoff_count_all[start_idx:end_idx] = np.asarray(batch_cutoff_count, dtype=np.int32)
        else:
            batch_sig_mask_np = None
            n_sig_all[start_idx:end_idx] = 0
            cutoff_count_all[start_idx:end_idx] = 0

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax_batch, dtype=np.int32)
        class_assignment[start_idx:end_idx] = np.asarray(best_class_batch, dtype=np.int32)

        log_score_offset = (
            np.zeros(batch_size, dtype=np.float64)
            if coarse_gaussian_ffi_enabled
            else -0.5
            * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        )
        global_log_z_np = np.asarray(global_log_z, dtype=np.float64)
        best_score_np = np.asarray(best_score_batch, dtype=np.float64)
        normalization_log_z[start_idx:end_idx] = global_log_z_np
        normalization_log_evidence[start_idx:end_idx] = global_log_z_np + log_score_offset
        log_evidence[start_idx:end_idx] = normalization_log_evidence[start_idx:end_idx].astype(np.float32)
        best_log_score[start_idx:end_idx] = (best_score_np + log_score_offset).astype(np.float32)
        max_posterior[start_idx:end_idx] = np.exp(best_score_np - global_log_z_np).astype(np.float32)
        for class_index, class_log_z in enumerate(class_log_z_values):
            class_log_evidence[class_index, start_idx:end_idx] = (
                np.asarray(class_log_z, dtype=np.float64) + log_score_offset
            )
        if return_class_best:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_best_scores[class_index],
                    log_score_offset,
                )
                class_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free
                class_best_log_score[class_index, start_idx:end_idx] = absolute
                class_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_best_argmaxes[class_index],
                    dtype=np.int32,
                )
        if return_class_second:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_second_best_scores[class_index],
                    log_score_offset,
                )
                class_second_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free
                class_second_best_log_score[class_index, start_idx:end_idx] = absolute
                class_second_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_second_best_argmaxes[class_index],
                    dtype=np.int32,
                )

        if debug_dump_enabled:
            # Concatenate per-class per-block raw scores for the dump targets
            # into per-class arrays of shape (n_targets, n_rot, n_trans).
            target_scores_pre_prior_per_class = None
            target_scores_with_prior_per_class = None
            target_local_positions_for_dump = None
            score_capture_mode = "intrusive_per_block_host_materialization"
            if passive_raw_score_blocks_per_class is not None:
                if cached_class_score_blocks is None:
                    raise RuntimeError("passive significance dump lost cached scores")
                target_scores_pre_prior_per_class = []
                target_scores_with_prior_per_class = []
                for class_index in range(n_classes):
                    raw_blocks = []
                    with_prior_blocks = []
                    for block_index in range(n_blocks):
                        r0 = block_index * rotation_block_size
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        raw_blocks.append(
                            np.asarray(
                                passive_raw_score_blocks_per_class[class_index][block_index][
                                    dump_target_local_positions, :actual_rot, :
                                ],
                                dtype=np.float64,
                            )
                        )
                        with_prior_blocks.append(
                            np.asarray(
                                cached_class_score_blocks[class_index][block_index][
                                    dump_target_local_positions, :actual_rot, :
                                ],
                                dtype=np.float64,
                            )
                        )
                    target_scores_pre_prior_per_class.append(
                        np.concatenate(raw_blocks, axis=1)
                    )
                    target_scores_with_prior_per_class.append(
                        np.concatenate(with_prior_blocks, axis=1)
                    )
                target_local_positions_for_dump = dump_target_local_positions
                score_capture_mode = "passive_cached_after_support"
            elif dump_target_pre_prior_blocks_per_class is not None:
                target_scores_pre_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_pre_prior_blocks_per_class
                ]
                target_scores_with_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_with_prior_blocks_per_class
                ]
                target_local_positions_for_dump = dump_target_local_positions
            projected_reference_rotation_ids = None
            projected_reference_per_class = None
            projected_reference_norm_score_per_class = None
            projected_cross_score_per_class = None
            requested_projection_rotations = sorted(
                parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS") or (),
            )
            if requested_projection_rotations and target_local_positions_for_dump is not None:
                projected_reference_rotation_ids = np.asarray(requested_projection_rotations, dtype=np.int32)
                if (
                    int(projected_reference_rotation_ids[0]) < 0
                    or int(projected_reference_rotation_ids[-1]) >= n_rot
                ):
                    raise ValueError(
                        "RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS contains an out-of-range rotation",
                    )
                projection_rotations = jnp.asarray(rotations[projected_reference_rotation_ids])
                projection_values = []
                projection_norm_scores = []
                projection_cross_scores = []
                for class_index, mean_for_proj in enumerate(means_for_proj):
                    projected_half, projected_abs2 = _project_block(
                        class_index,
                        mean_for_proj,
                        projection_rotations,
                    )
                    if use_window:
                        if projector_returns_compact:
                            if coarse_gaussian_window_positions is not None:
                                projected_half = projected_half[:, coarse_gaussian_window_positions]
                                projected_abs2 = projected_abs2[:, coarse_gaussian_window_positions]
                        else:
                            projected_half = projected_half[:, window_indices]
                            projected_abs2 = projected_abs2[:, window_indices]
                    if not use_float64_scoring:
                        projected_half = projected_half.astype(jnp.complex64)
                        projected_abs2 = projected_abs2.astype(jnp.float32)
                    score_weights = half_weights_windowed if use_window else half_weights
                    weighted_projected = projected_half * score_weights
                    weighted_projected_abs2 = projected_abs2 * score_weights
                    component_cross = (
                        -2.0
                        * jnp.matmul(
                            jnp.conj(shifted_data),
                            weighted_projected.T,
                            precision=jax.lax.Precision.HIGHEST,
                        ).real
                    )
                    component_cross = component_cross.reshape(
                        batch_size,
                        n_trans,
                        projected_reference_rotation_ids.size,
                    ).swapaxes(1, 2)
                    component_norm = jnp.matmul(
                        ctf2_data,
                        weighted_projected_abs2.T,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    component_norm = jnp.broadcast_to(
                        component_norm[..., None],
                        component_cross.shape,
                    )
                    projection_values.append(np.asarray(projected_half, dtype=np.complex128))
                    projection_norm_scores.append(
                        np.asarray(-0.5 * component_norm, dtype=np.float64)
                    )
                    projection_cross_scores.append(
                        np.asarray(-0.5 * component_cross, dtype=np.float64)
                    )
                projected_reference_per_class = np.stack(projection_values, axis=0)
                projected_reference_norm_score_per_class = np.stack(
                    projection_norm_scores,
                    axis=0,
                )
                projected_cross_score_per_class = np.stack(
                    projection_cross_scores,
                    axis=0,
                )
            _maybe_dump_k_class_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                n_classes=n_classes,
                rotations=rotations,
                translations=translations,
                class_weight_mats=(
                    [
                        np.asarray(
                            batch_weights.reshape(
                                batch_size,
                                n_classes,
                                n_rot * n_trans,
                            )[:, class_index, :],
                            dtype=np.float64,
                        )
                        for class_index in range(n_classes)
                    ]
                    if relion_f32_coarse_support_enabled
                    else [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats]
                ),
                batch_sig_mask=batch_sig_mask_np,
                batch_n_sig=np.asarray(batch_n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(best_argmax_batch, dtype=np.int64),
                class_assignment_batch=np.asarray(best_class_batch, dtype=np.int64),
                global_log_z=global_log_z_np,
                class_log_z_values=class_log_z_values,
                best_score=best_score_np,
                max_posterior=max_posterior[start_idx:end_idx],
                rotation_log_prior_padded=rotation_log_prior_padded,
                batch_translation_log_prior=batch_translation_log_prior,
                class_log_priors=class_log_priors_np,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                target_local_positions=target_local_positions_for_dump,
                target_scores_pre_prior_per_class=target_scores_pre_prior_per_class,
                target_scores_with_prior_per_class=target_scores_with_prior_per_class,
                projected_reference_rotation_ids=projected_reference_rotation_ids,
                projected_reference_per_class=projected_reference_per_class,
                projected_reference_norm_score_per_class=(
                    projected_reference_norm_score_per_class
                ),
                projected_cross_score_per_class=projected_cross_score_per_class,
                shifted_data=shifted_data,
                ctf2_data=ctf2_data,
                window_indices=window_indices,
                half_weights_used=half_weights_windowed if use_window else half_weights,
                coarse_gaussian_shifted_corrected=coarse_gaussian_shifted_corrected,
                coarse_gaussian_unshifted_corrected=coarse_gaussian_unshifted_corrected,
                coarse_gaussian_pixel_weight=coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2=coarse_gaussian_initial_diff2,
                coarse_gaussian_score_indices=coarse_gaussian_score_indices,
                translation_phase_source=translations_source,
                relion_f32_sum_weight=(
                    _batch_sum_weight if relion_f32_coarse_support_enabled else None
                ),
                relion_f32_significant_weight=(
                    _batch_significant_weight
                    if relion_f32_coarse_support_enabled
                    else None
                ),
                relion_f32_cutoff_count=(
                    batch_cutoff_count if relion_f32_coarse_support_enabled else None
                ),
                score_capture_mode=score_capture_mode,
                debug_iteration=debug_iteration,
            )

        if collect_significance:
            samples_per_class = n_rot * n_trans
            for local_idx, global_idx in enumerate(indices):
                global_idx = int(global_idx)
                for class_index in range(n_classes):
                    c0 = class_index * samples_per_class
                    c1 = c0 + samples_per_class
                    mask = batch_sig_mask_np[local_idx, c0:c1]
                    significant_sample_indices[class_index][global_idx] = compact_significant_sample_indices_from_mask(
                        mask,
                    )
        start_idx = end_idx

    full_stats = {
        "normalization_log_z": normalization_log_z,
        "normalization_log_evidence": normalization_log_evidence,
        "log_evidence_per_image": log_evidence,
        "best_log_score_per_image": best_log_score,
        "max_posterior_per_image": max_posterior,
        "class_log_evidence_per_image": class_log_evidence,
        "class_assignments": class_assignment,
        # RELION serializes the cutoff rank before inclusive threshold ties
        # expand the pass-2/M-step support represented by ``n_sig_all``.
        "significant_cutoff_counts": cutoff_count_all,
    }
    if return_class_best:
        full_stats["class_best_log_score_per_image"] = class_best_log_score
        full_stats["class_best_offset_free_log_score_per_image"] = class_best_offset_free_log_score
        full_stats["class_hard_assignments"] = class_hard_assignment
    if return_class_second:
        full_stats["class_second_best_log_score_per_image"] = class_second_best_log_score
        full_stats["class_second_hard_assignments"] = class_second_hard_assignment
        full_stats["class_second_best_offset_free_log_score_per_image"] = class_second_best_offset_free_log_score
    if tree_rescore_enabled:
        full_stats["firstiter_cc_tree_top2_rescore"] = {
            "max_margin": float(tree_rescore_max_margin),
            "examined_images": int(tree_rescore_examined),
            "ambiguous_images": int(tree_rescore_ambiguous),
            "exact_score_ties": int(tree_rescore_exact_ties),
            "winner_changes": int(tree_rescore_winner_changes),
        }
        logger.warning(
            "RELION coarse-tree top-2 rescore complete: "
            "examined=%d ambiguous=%d exact_ties=%d winner_changes=%d",
            tree_rescore_examined,
            tree_rescore_ambiguous,
            tree_rescore_exact_ties,
            tree_rescore_winner_changes,
        )
    return sig_rot_any, n_sig_all, hard_assignment, class_assignment, significant_sample_indices, full_stats
