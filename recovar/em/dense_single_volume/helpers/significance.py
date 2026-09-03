"""Batched significance pruning for adaptive two-pass oversampling.

Runs a coarse E-step and identifies significant (rotation, translation)
pairs per image without materializing the full weight matrix.
Called by ``refine_single_volume`` and ``_run_relion_iteration_loop`` in ``refine.py``.
"""

import hashlib
import json
import logging
import operator
import os
from enum import Enum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import projection_cache as projection_cache_helpers
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    SOURCE_ROTATION_BLOCK_SIZE,
    CoarseGemmHybridBlockSelection,
    CoarseGemmHybridCompactScores,
    assemble_coarse_gemm_hybrid_compact_scores_f32,
    assemble_coarse_gemm_hybrid_dense_scores_f32,
    initialize_coarse_gemm_hybrid_interval_state,
    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids,
    plan_coarse_gemm_certificate_topology,
    select_coarse_gemm_hybrid_rotation_blocks,
)
from recovar.em.dense_single_volume.helpers.coarse_gemm_streaming import (
    COARSE_GEMM_STREAMING_SCHEMA,
    aggregate_coarse_gemm_streaming_summaries,
    coarse_gemm_streaming_dual_state_bytes,
    coarse_gemm_streaming_state_bytes,
    initialize_coarse_gemm_streaming_state,
    update_coarse_gemm_streaming_state,
    write_coarse_gemm_streaming_summary,
)
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block
from recovar.em.dense_single_volume.helpers.scoring import (
    _coarse_gaussian_direct_macro_diagnostics,
    _coarse_gaussian_qualification_decision,
    _e_step_block_scores,
    _e_step_block_scores_windowed,
    _prepare_relion_coarse_gaussian_gemm_f64_image_batch,
    _relion_coarse_diff2_rotation_blocks_from_topology_f32,
    _relion_coarse_gaussian_gemm_scores,
    _relion_coarse_gaussian_gemm_update_certificate_state,
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
_K1_COARSE_FUSED_PROJECTOR_ENV = "RECOVAR_K1_COARSE_FUSED_PROJECTOR"
_RELION_COARSE_CANONICAL_REDUCTION_ENV = (
    "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION"
)
_K1_COARSE_SINGLE_LANE_CANONICAL_ENV = (
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"
)
_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV = (
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"
)
_K1_COARSE_PREHALF_WEIGHT_ENV = "RECOVAR_K1_COARSE_PREHALF_WEIGHT"
_K1_COARSE_MULTISTREAM_WORKERS_ENV = "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS"
_COARSE_SELECTOR_WRAPPER_TARGETS = {
    "relion_coarse_diff2_projector_f32": "cuda_relion_coarse_diff2_projector_f32",
    "relion_coarse_diff2_projector_multistream_f32": (
        "cuda_relion_coarse_diff2_projector_multistream_f32"
    ),
}
_COARSE_GAUSSIAN_GEMM_MACRO_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"
_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB"
)
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
)
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
)
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB = 4.0
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS = 4_608
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT = 16
_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB = 13_332_001
_COARSE_GAUSSIAN_GEMM_HYBRID_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID"
_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
)
_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"
)
_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE"
)
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
)
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"
)
_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR"
)
_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES"
)
_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR"
)
_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK"
)
_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV = (
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"
)
_K1_RELION_EXACT_COARSE_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS"
_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV = (
    "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS"
)
_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE_ENV = (
    "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE"
)
_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV = (
    "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS"
)
_K1_RELION_F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"
_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"
)
_SIGNIFICANCE_DUMP_PASSIVE_CACHE_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE"
)
NVTX_DOMAIN_EM = "recovar_em"
logger = logging.getLogger(__name__)


def _repeat_pad_batch_axis(value, target_size: int):
    """Pad a non-empty image batch by repeating row zero.

    Repeating a real row keeps normalized-CC and exact CUDA preprocessing
    finite. Callers must discard the repeated rows from all science outputs.
    """

    array = np.asarray(value)
    target_size = int(target_size)
    if array.shape[0] >= target_size:
        return array
    if array.shape[0] == 0:
        raise ValueError("cannot repeat-pad an empty image batch")
    return np.concatenate(
        [array, np.repeat(array[:1], target_size - array.shape[0], axis=0)],
        axis=0,
    )


def _pad_significance_preprocess_inputs(
    batch_data,
    ctf_params,
    integer_pre_shifts,
    batch_corr,
    batch_scale,
    relion_preprocess_kwargs,
    *,
    target_size: int,
):
    """Give a tail significance batch the same compiled image shape as full batches."""

    actual_size = int(np.asarray(batch_data).shape[0])
    target_size = max(actual_size, int(target_size))
    if actual_size == target_size:
        return (
            batch_data,
            ctf_params,
            integer_pre_shifts,
            batch_corr,
            batch_scale,
            relion_preprocess_kwargs,
        )
    padded_kwargs = None
    if relion_preprocess_kwargs is not None:
        padded_kwargs = {
            key: jnp.asarray(_repeat_pad_batch_axis(value, target_size))
            for key, value in relion_preprocess_kwargs.items()
        }
    return (
        _repeat_pad_batch_axis(batch_data, target_size),
        _repeat_pad_batch_axis(ctf_params, target_size),
        (
            None
            if integer_pre_shifts is None
            else _repeat_pad_batch_axis(integer_pre_shifts, target_size)
        ),
        None if batch_corr is None else _repeat_pad_batch_axis(batch_corr, target_size),
        _repeat_pad_batch_axis(batch_scale, target_size),
        padded_kwargs,
    )


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


def _k1_coarse_fused_projector_enabled(*, default: bool = False) -> bool:
    """Return whether coarse projection and diff2 use RELION's fused topology."""

    token = os.environ.get(
        _K1_COARSE_FUSED_PROJECTOR_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_FUSED_PROJECTOR_ENV}={token!r}",
    )


def _relion_coarse_canonical_reduction_enabled(*, default: bool = False) -> bool:
    """Whether the shared fused coarse scorer reduces lanes in index order."""

    token = os.environ.get(
        _RELION_COARSE_CANONICAL_REDUCTION_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_RELION_COARSE_CANONICAL_REDUCTION_ENV}={token!r}",
    )


def _k1_coarse_single_lane_canonical_enabled(*, default: bool = False) -> bool:
    """Whether 65--128 translations use the single-lane CUDA specialization."""

    token = os.environ.get(
        _K1_COARSE_SINGLE_LANE_CANONICAL_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}={token!r}",
    )


def _k1_coarse_single_lane_canonical_selected(
    *,
    requested: bool,
    score_mode: str,
    translation_count: int,
) -> bool:
    """Select the specialization only where one thread owns a translation."""

    return bool(
        requested
        and score_mode == "gaussian"
        and 65 <= int(translation_count) <= 128
    )


def _k1_coarse_native_atomic_reduction_enabled(*, default: bool = False) -> bool:
    """Whether the fused coarse scorer uses RELION's native atomic lane adds."""

    token = os.environ.get(
        _K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}={token!r}",
    )


def _k1_coarse_native_atomic_reduction_selected(
    *,
    requested: bool,
    score_mode: str,
    translation_count: int,
) -> bool:
    """Select only the T=29 Gaussian topology covered by the live H100 gate."""

    return bool(
        requested
        and score_mode == "gaussian"
        and int(translation_count) == 29
    )


def _k1_coarse_prehalf_weight_enabled(*, default: bool = False) -> bool:
    """Whether coarse pixel weights are halved once before rotation FMAs.

    RELION stores ``corr_img / 2`` in shared memory once per pixel.  The
    historical RECOVAR kernel instead multiplied every rotation contribution
    by ``0.5``.  The specialization is mathematically equivalent, but changes
    float32 operation order and is therefore opt-in until its trajectory gate
    is complete.
    """

    token = os.environ.get(
        _K1_COARSE_PREHALF_WEIGHT_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_COARSE_PREHALF_WEIGHT_ENV}={token!r}",
    )


def _k1_coarse_multistream_worker_count(*, default: int = 0) -> int:
    """Return the default-off RELION coarse particle-stream count.

    The first performance gate is deliberately restricted to the proven
    GUI-default ``--j 8`` ownership topology.  Zero keeps the accepted
    single-stream batch grid unchanged.
    """

    token = os.environ.get(
        _K1_COARSE_MULTISTREAM_WORKERS_ENV,
        str(int(default)),
    ).strip()
    try:
        count = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV} must be 0 or 8, got {token!r}",
        ) from error
    if count not in {0, 8}:
        raise ValueError(
            f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV} must be 0 or 8, got {token!r}",
        )
    return count


def _validate_coarse_selector_audit(audit: dict) -> dict:
    """Validate and normalize one host-observed coarse selector audit.

    A configured fused selector is not evidence that its wrapper ran.  The
    wrapper name, XLA target, and positive call/row counters are therefore
    required whenever the fused path is effective.  An inactive control is
    represented explicitly by ``None`` wrapper/target values and zero counts.
    """

    if not isinstance(audit, dict):
        raise TypeError("coarse selector audit must be a dict")
    required = {
        "score_mode",
        "translation_count",
        "requested_fused",
        "effective_fused",
        "requested_workers",
        "effective_workers",
        "requested_atomic",
        "effective_atomic",
        "wrapper",
        "target",
        "counts",
    }
    missing = sorted(required.difference(audit))
    if missing:
        raise ValueError(
            "coarse selector audit is missing fields: " + ", ".join(missing)
        )

    score_mode = audit["score_mode"]
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            f"coarse selector audit has unsupported score_mode={score_mode!r}"
        )

    integer_fields = {
        "translation_count": audit["translation_count"],
        "requested_workers": audit["requested_workers"],
        "effective_workers": audit["effective_workers"],
    }
    normalized_integers = {}
    for name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"coarse selector audit {name} must be an integer")
        normalized_integers[name] = int(value)
    translation_count = normalized_integers["translation_count"]
    requested_workers = normalized_integers["requested_workers"]
    effective_workers = normalized_integers["effective_workers"]
    if translation_count <= 0:
        raise ValueError("coarse selector audit translation_count must be positive")
    if requested_workers not in {0, 8} or effective_workers not in {0, 8}:
        raise ValueError(
            "coarse selector audit requested/effective workers must be 0 or 8"
        )

    prehalf_fields = {"requested_prehalf", "effective_prehalf"}
    present_prehalf_fields = prehalf_fields.intersection(audit)
    if present_prehalf_fields and present_prehalf_fields != prehalf_fields:
        raise ValueError(
            "coarse selector audit must provide requested/effective prehalf together"
        )

    normalized_booleans = {}
    for name in (
        "requested_fused",
        "effective_fused",
        "requested_atomic",
        "effective_atomic",
        *(sorted(prehalf_fields) if present_prehalf_fields else ()),
    ):
        value = audit[name]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"coarse selector audit {name} must be boolean")
        normalized_booleans[name] = bool(value)
    requested_fused = normalized_booleans["requested_fused"]
    effective_fused = normalized_booleans["effective_fused"]
    requested_atomic = normalized_booleans["requested_atomic"]
    effective_atomic = normalized_booleans["effective_atomic"]
    requested_prehalf = normalized_booleans.get("requested_prehalf", False)
    effective_prehalf = normalized_booleans.get("effective_prehalf", False)
    if effective_fused and not requested_fused:
        raise ValueError("effective fused coarse selector was not requested")
    if effective_workers and requested_workers != effective_workers:
        raise ValueError("effective coarse workers do not match the request")
    if effective_atomic and not requested_atomic:
        raise ValueError("effective native-atomic coarse reduction was not requested")
    if effective_prehalf and not requested_prehalf:
        raise ValueError("effective coarse prehalf weight was not requested")
    if effective_workers and not effective_fused:
        raise ValueError("effective coarse workers require the fused selector")
    if effective_atomic and not effective_fused:
        raise ValueError("effective native-atomic reduction requires the fused selector")
    if effective_prehalf and not effective_atomic:
        raise ValueError("effective coarse prehalf weight requires native-atomic reduction")
    if effective_fused and score_mode != "gaussian":
        raise ValueError("the fused coarse selector is Gaussian-only")
    if effective_workers and score_mode != "gaussian":
        raise ValueError("coarse worker streams are Gaussian-only")
    if effective_atomic and (
        score_mode != "gaussian" or translation_count != 29
    ):
        raise ValueError(
            "effective native-atomic reduction requires the Gaussian T=29 gate"
        )

    counts = audit["counts"]
    if not isinstance(counts, dict):
        raise TypeError("coarse selector audit counts must be a dict")
    required_counts = {
        "fused_calls",
        "actual_rows",
        "multistream_calls",
        "native_atomic_selected_calls",
    }
    if present_prehalf_fields:
        required_counts.add("prehalf_selected_calls")
    missing_counts = sorted(required_counts.difference(counts))
    if missing_counts:
        raise ValueError(
            "coarse selector audit counts are missing fields: "
            + ", ".join(missing_counts)
        )
    normalized_counts = {}
    for name in sorted(required_counts):
        value = counts[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"coarse selector audit count {name} must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"coarse selector audit count {name} must be non-negative")
        normalized_counts[name] = value

    wrapper = audit["wrapper"]
    target = audit["target"]
    if wrapper is not None and not isinstance(wrapper, str):
        raise TypeError("coarse selector audit wrapper must be a string or None")
    if target is not None and not isinstance(target, str):
        raise TypeError("coarse selector audit target must be a string or None")
    fused_calls = normalized_counts["fused_calls"]
    actual_rows = normalized_counts["actual_rows"]
    multistream_calls = normalized_counts["multistream_calls"]
    native_atomic_calls = normalized_counts["native_atomic_selected_calls"]
    prehalf_calls = normalized_counts.get("prehalf_selected_calls", 0)
    if effective_fused:
        expected_wrapper = (
            "relion_coarse_diff2_projector_multistream_f32"
            if effective_workers
            else "relion_coarse_diff2_projector_f32"
        )
        expected_target = _COARSE_SELECTOR_WRAPPER_TARGETS[expected_wrapper]
        if wrapper != expected_wrapper or target != expected_target:
            raise ValueError(
                "coarse selector audit observed the wrong wrapper/target: "
                f"{wrapper!r}/{target!r} != {expected_wrapper!r}/{expected_target!r}"
            )
        if fused_calls <= 0:
            raise ValueError("effective fused coarse selector recorded zero calls")
        if actual_rows <= 0:
            raise ValueError("effective fused coarse selector recorded zero actual rows")
        if actual_rows < fused_calls:
            raise ValueError("coarse selector actual rows cannot be smaller than calls")
        expected_multistream_calls = fused_calls if effective_workers else 0
        if multistream_calls != expected_multistream_calls:
            raise ValueError(
                "coarse selector multistream call count does not match the effective wrapper"
            )
        expected_atomic_calls = fused_calls if effective_atomic else 0
        if native_atomic_calls != expected_atomic_calls:
            raise ValueError(
                "coarse selector native-atomic call count does not match the effective reduction"
            )
        expected_prehalf_calls = fused_calls if effective_prehalf else 0
        if prehalf_calls != expected_prehalf_calls:
            raise ValueError(
                "coarse selector prehalf call count does not match the effective specialization"
            )
    else:
        if wrapper is not None or target is not None:
            raise ValueError("inactive coarse selector must not report a wrapper/target")
        if effective_workers or effective_atomic or effective_prehalf:
            raise ValueError(
                "inactive coarse selector cannot report effective workers/atomic/prehalf"
            )
        if any(normalized_counts.values()):
            raise ValueError("inactive coarse selector must report zero execution counts")

    normalized = dict(audit)
    normalized.update(normalized_integers)
    normalized.update(normalized_booleans)
    normalized["counts"] = normalized_counts
    return normalized


def _coarse_gaussian_gemm_macro_enabled(*, default: bool = False) -> bool:
    """Whether one coarse projection feeds the shared multi-image GEMMs.

    The exact-arithmetic objective is unchanged, but expanding the direct
    square changes operation order and can amplify cancellation.  Keep it
    default-off until same-H100 repeats establish stable numerical noise,
    unchanged discrete choices/support and quality, and a material end-to-end
    speedup.
    """

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_MACRO_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_projection_cache_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the explicit call-scoped coarse-projection cache toggle."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_hybrid_enabled(*, default: bool = False) -> bool:
    """Resolve the default-off certified GEMM/exact-source16 hybrid."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_HYBRID_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_compact_posterior_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the experimental fixed-capacity selected-score posterior."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_hybrid_image_batch_size_request() -> int | None:
    """Return the explicit compact-hybrid image-batch override, if any."""

    token = os.environ.get(_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV)
    if token is None:
        return None
    token = token.strip()
    try:
        batch_size = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {token!r}",
        ) from error
    if batch_size <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {token!r}",
        )
    return batch_size


def _resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
    input_batch_size: int,
    *,
    requested_batch_size: int | None,
    n_images: int,
    certificate_chunk_rows: int,
    n_translations: int,
    hybrid_enabled: bool,
    compact_posterior_enabled: bool,
    score_float_budget: int | None = None,
) -> int:
    """Resolve a guarded shared image batch for the compact score hybrid.

    InitialModel currently caps pass-1 batches against the full
    ``B*R*T`` score cube.  The compact hybrid never materializes that cube: its
    largest score-dependent tile is streamed over ``certificate_chunk_rows``.
    An explicit override can therefore coalesce logical image batches while
    retaining the mature shared matrix-matrix scorer and every per-image
    arithmetic/reduction order.
    """

    input_size = operator.index(input_batch_size)
    image_count = operator.index(n_images)
    chunk_rows = operator.index(certificate_chunk_rows)
    translation_count = operator.index(n_translations)
    if min(input_size, image_count, chunk_rows, translation_count) <= 0:
        raise ValueError("compact-hybrid image-batch dimensions must be positive")
    if requested_batch_size is None:
        return input_size
    requested = operator.index(requested_batch_size)
    prefix = f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} requires"
    if not hybrid_enabled:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if not compact_posterior_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
        )
    if requested <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} must be a "
            f"positive integer, got {requested!r}",
        )
    if score_float_budget is None:
        from recovar.em.dense_single_volume.batch_planning import (
            RELION_SCORE_TENSOR_FLOAT_BUDGET,
        )

        score_float_budget = RELION_SCORE_TENSOR_FLOAT_BUDGET
    score_budget = operator.index(score_float_budget)
    if score_budget <= 0:
        raise ValueError("compact-hybrid score-float budget must be positive")

    effective = min(requested, image_count)
    streamed_candidate_count = effective * chunk_rows * translation_count
    if streamed_candidate_count > score_budget:
        raise MemoryError(
            "compact-hybrid image-batch override exceeds the mature EM score "
            f"tile budget: {streamed_candidate_count} > {score_budget} floats",
        )
    return effective


def _validate_coarse_gaussian_gemm_compact_posterior_request(
    *,
    hybrid_enabled: bool,
    return_class_best: bool,
    return_class_second: bool,
) -> None:
    """Keep the first compact-posterior boundary narrow and fail closed."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1 requires"
    if not hybrid_enabled:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if return_class_best or return_class_second:
        raise ValueError(
            f"{prefix} return_class_best=False and return_class_second=False",
        )


def _coarse_significance_support_audit_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve exact, diagnostic-only coarse-support hashing."""

    token = os.environ.get(
        _COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}={token!r}",
    )


def _coarse_significance_support_audit_ids_enabled() -> bool:
    """Whether a support audit also retains its exact selected IDs."""

    token = os.environ.get(
        _COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV,
        "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV}={token!r}",
    )


def _coarse_gaussian_gemm_hybrid_block_capacity(
    *,
    default: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
) -> int:
    """Return the fixed selected-source16 capacity for one image row."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV,
        str(default),
    ).strip()
    try:
        capacity = int(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV} must be a "
            f"positive integer, got {token!r}",
        ) from error
    if capacity <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY_ENV} must be a "
            f"positive integer, got {token!r}",
        )
    return capacity


class _CoarseGaussianScoreBackend(str, Enum):
    """One resolved coarse Gaussian score/reduction implementation."""

    RECTANGULAR = "rectangular"
    FUSED = "fused"
    FUSED_CANONICAL = "fused_canonical"
    FUSED_NATIVE_ATOMIC = "fused_native_atomic"
    FUSED_SINGLE_LANE = "fused_single_lane"
    FUSED_MULTISTREAM = "fused_multistream"
    NATIVE_TEXTURE = "native_texture"
    GEMM_MACRO = "gemm_macro"


def _resolve_coarse_gaussian_score_backend(
    *,
    gemm_macro_requested: bool,
    score_mode: str,
    fused_projector_requested: bool,
    fused_projector_enabled: bool,
    canonical_reduction_requested: bool,
    canonical_reduction_enabled: bool,
    native_atomic_reduction_requested: bool,
    native_atomic_reduction_enabled: bool,
    single_lane_canonical_requested: bool,
    single_lane_canonical_enabled: bool,
    multistream_requested: bool,
    multistream_enabled: bool,
    native_texture_requested: bool,
    native_texture_enabled: bool,
) -> _CoarseGaussianScoreBackend:
    """Resolve one score backend and reject an explicit GEMM conflict.

    The exact-operand, CUDA-sincosf, and texture-projection flags are operands
    or prerequisites rather than competing score/reduction selectors.  All
    selectors that the expanded GEMM branch would otherwise silently override
    are represented here.
    """

    if gemm_macro_requested:
        if score_mode != "gaussian":
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 requires "
                "score_mode='gaussian'",
            )
        conflicts = [
            f"{name}={requested_value}"
            for name, requested_value, selected in (
                (_K1_COARSE_FUSED_PROJECTOR_ENV, "1", fused_projector_requested),
                (_RELION_COARSE_CANONICAL_REDUCTION_ENV, "1", canonical_reduction_requested),
                (_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV, "1", native_atomic_reduction_requested),
                (_K1_COARSE_SINGLE_LANE_CANONICAL_ENV, "1", single_lane_canonical_requested),
                (_K1_COARSE_MULTISTREAM_WORKERS_ENV, "8", multistream_requested),
                (_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE_ENV, "1", native_texture_requested),
            )
            if selected
        ]
        if conflicts:
            rendered = ", ".join(conflicts)
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 conflicts with {rendered}",
            )
        return _CoarseGaussianScoreBackend.GEMM_MACRO
    if native_texture_enabled:
        return _CoarseGaussianScoreBackend.NATIVE_TEXTURE
    if multistream_enabled:
        return _CoarseGaussianScoreBackend.FUSED_MULTISTREAM
    if single_lane_canonical_enabled:
        return _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE
    if native_atomic_reduction_enabled:
        return _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC
    if canonical_reduction_enabled:
        return _CoarseGaussianScoreBackend.FUSED_CANONICAL
    if fused_projector_enabled:
        return _CoarseGaussianScoreBackend.FUSED
    return _CoarseGaussianScoreBackend.RECTANGULAR


class CoarseGaussianGemmResources(NamedTuple):
    """Conservative device-transient accounting for one GEMM rotation block."""

    full_centered_projection_bytes: int
    compact_projection_bytes: int
    compact_projection_abs2_bytes: int
    predicted_peak_projection_bytes: int
    projected_transient_budget_bytes: int
    pixel_index_device_to_host_materializations: int


class CoarseGaussianGemmDiagnosticScope(NamedTuple):
    """Deterministic identity and completion contract for one diagnostic call.

    InitialModel invokes the shared significance engine once per non-empty
    pseudo-halfset/group.  ``expected_call_ids`` names that complete run, and
    exactly one (the final call) sets ``finalize=True`` so an aggregate
    manifest can prove that every globally requested particle was captured
    exactly once across the disjoint call scopes.
    """

    run_id: str
    call_id: str
    expected_call_ids: tuple[str, ...]
    finalize: bool


def _validate_coarse_gaussian_gemm_diagnostic_identifier(
    value: str,
    *,
    field: str,
) -> str:
    """Return one filename-safe deterministic diagnostic identifier."""

    token = str(value)
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if not token or len(token) > 160 or any(character not in allowed for character in token):
        raise ValueError(
            f"coarse GEMM diagnostic {field} must be 1--160 filename-safe "
            f"characters, got {token!r}",
        )
    return token


def _resolve_coarse_gaussian_gemm_diagnostic_scope(
    scope: CoarseGaussianGemmDiagnosticScope | None,
    *,
    debug_iteration: int | None,
    current_size: int | None,
) -> tuple[CoarseGaussianGemmDiagnosticScope, bool]:
    """Resolve an explicit multi-call scope or a strict single-call default."""

    if scope is None:
        iteration_label = -1 if debug_iteration is None else int(debug_iteration)
        size_label = -1 if current_size is None else int(current_size)
        iteration_token = (
            f"m{-iteration_label:04d}"
            if iteration_label < 0
            else f"{iteration_label:04d}"
        )
        size_token = f"m{-size_label:04d}" if size_label < 0 else f"{size_label:04d}"
        run_id = f"shared_it{iteration_token}_cs{size_token}"
        return (
            CoarseGaussianGemmDiagnosticScope(
                run_id=run_id,
                call_id="call0000_global",
                expected_call_ids=("call0000_global",),
                finalize=True,
            ),
            False,
        )
    run_id = _validate_coarse_gaussian_gemm_diagnostic_identifier(
        scope.run_id,
        field="run_id",
    )
    call_id = _validate_coarse_gaussian_gemm_diagnostic_identifier(
        scope.call_id,
        field="call_id",
    )
    expected_call_ids = tuple(
        _validate_coarse_gaussian_gemm_diagnostic_identifier(
            value,
            field="expected_call_id",
        )
        for value in scope.expected_call_ids
    )
    if not expected_call_ids or len(set(expected_call_ids)) != len(expected_call_ids):
        raise ValueError(
            "coarse GEMM diagnostic expected_call_ids must be non-empty and unique",
        )
    if call_id not in expected_call_ids:
        raise ValueError(
            "coarse GEMM diagnostic call_id must occur in expected_call_ids",
        )
    if bool(scope.finalize) != (call_id == expected_call_ids[-1]):
        raise ValueError(
            "coarse GEMM diagnostic finalize must be true exactly for the last "
            "expected call",
        )
    return (
        CoarseGaussianGemmDiagnosticScope(
            run_id=run_id,
            call_id=call_id,
            expected_call_ids=expected_call_ids,
            finalize=bool(scope.finalize),
        ),
        True,
    )


def _coarse_gaussian_gemm_projected_transient_budget_bytes(
    *,
    default_gb: float = 2.0,
) -> int:
    """Return the explicit conservative projection-transient budget."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


def _coarse_gaussian_gemm_projection_cache_budget_bytes(
    *,
    default_gb: float = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB,
) -> int:
    """Return the explicit conservative call-scoped cache budget."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


def _coarse_gaussian_gemm_resources(
    *,
    rotation_block_size: int,
    image_shape,
    compact_pixel_count: int,
    budget_bytes: int,
) -> CoarseGaussianGemmResources:
    """Estimate and gate projection buffers before the first GPU launch.

    The mature texture projector currently creates a full physical centered
    half-image before gathering the current-size compact pixels.  Count that
    buffer plus the compact complex projection and its real abs2 companion.
    The pixel-index table is deliberately kept as a NumPy host array, so this
    path performs no device-to-host index materialization.
    """

    block_size = int(rotation_block_size)
    image_height, image_width = (int(value) for value in image_shape)
    compact_count = int(compact_pixel_count)
    if block_size <= 0 or image_height <= 0 or image_width <= 0 or compact_count <= 0:
        raise ValueError("coarse GEMM resource dimensions must all be positive")
    full_count = image_height * (image_width // 2 + 1)
    full_bytes = block_size * full_count * np.dtype(np.complex64).itemsize
    compact_bytes = block_size * compact_count * np.dtype(np.complex64).itemsize
    abs2_bytes = block_size * compact_count * np.dtype(np.float32).itemsize
    predicted_peak = full_bytes + compact_bytes + abs2_bytes
    resources = CoarseGaussianGemmResources(
        full_centered_projection_bytes=int(full_bytes),
        compact_projection_bytes=int(compact_bytes),
        compact_projection_abs2_bytes=int(abs2_bytes),
        predicted_peak_projection_bytes=int(predicted_peak),
        projected_transient_budget_bytes=int(budget_bytes),
        pixel_index_device_to_host_materializations=0,
    )
    if resources.predicted_peak_projection_bytes > resources.projected_transient_budget_bytes:
        raise MemoryError(
            "coarse GEMM predicted projection transient exceeds "
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV}: "
            f"{resources.predicted_peak_projection_bytes} > "
            f"{resources.projected_transient_budget_bytes} bytes",
        )
    return resources


def _validate_coarse_gaussian_gemm_projection_cache_request(
    *,
    macro_enabled: bool,
    n_classes: int,
    n_rotations: int,
    coarse_gaussian_ffi_enabled: bool,
    exact_coarse_operands_enabled: bool,
    use_relion_projector: bool,
    relion_texture_interp_enabled: bool,
    half_spectrum_scoring: bool,
    use_float64_scoring: bool,
    relion_projector_dtype,
) -> None:
    """Fail closed unless the first cache seam's exact K=1 contract holds."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1 requires"
    if not macro_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
        )
    if int(n_classes) != 1:
        raise ValueError(f"{prefix} K=1, got K={int(n_classes)}")
    if int(n_rotations) <= 0 or int(n_rotations) % int(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT
    ):
        raise ValueError(
            f"{prefix} a positive rotation count divisible by "
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT}, "
            f"got {int(n_rotations)}",
        )
    if not coarse_gaussian_ffi_enabled:
        raise ValueError(
            f"{prefix} the exact RELION coarse Gaussian FFI path",
        )
    if not exact_coarse_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if not use_relion_projector or not relion_texture_interp_enabled:
        raise ValueError(
            f"{prefix} the supplied RELION texture projector",
        )
    if not half_spectrum_scoring:
        raise ValueError(f"{prefix} half-spectrum scoring")
    if use_float64_scoring:
        raise ValueError(f"{prefix} production float32/complex64 scoring")
    if relion_projector_dtype is None or np.dtype(relion_projector_dtype) != np.dtype(
        np.complex64
    ):
        raise TypeError(
            f"{prefix} a complex64 RELION projector, got "
            f"{relion_projector_dtype}",
        )


def _plan_coarse_gaussian_gemm_projection_cache(
    *,
    n_rotations: int,
    compact_pixel_count: int,
    image_shape,
    budget_bytes: int,
) -> projection_cache_helpers.ProjectionCachePlan:
    """Plan one conservative C64 K=1 cache in qualified 4,608-row chunks."""

    image_height, image_width = (int(value) for value in image_shape)
    return projection_cache_helpers.plan_projection_cache(
        table_count=1,
        row_count=int(n_rotations),
        pixel_count=int(compact_pixel_count),
        cache_dtype=np.complex64,
        requested_max_chunk_rows=(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        row_alignment=_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT,
        transient_specs=(
            projection_cache_helpers.ProjectionCacheTransientSpec(
                name="full_centered_projection",
                elements_per_row=image_height * (image_width // 2 + 1),
                dtype=np.complex64,
            ),
        ),
        budget_bytes=int(budget_bytes),
        # Job 13332001 observed donation aliasing on one H100 lowering.  The
        # production admission remains conservative and correct if that
        # informational hardware-specific observation does not generalize.
        destination_alias_proven=False,
    )


def _build_coarse_gaussian_gemm_projection_cache(plan, project_block):
    """Build one private call-scoped cache through the shared exact builder."""

    return projection_cache_helpers.build_projection_cache(plan, project_block)


def _coarse_gaussian_gemm_projection_cache_stats(plan, *, enabled: bool):
    """Describe conservative admission and narrowly scoped alias evidence."""

    h100_alias_evidence_applies = bool(
        plan.cache_shape == (1, 36_864, 5_100)
        and plan.chunk_rows
        == _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
    )
    return {
        "enabled": bool(enabled),
        "scope": "one significance call",
        "cache_shape": tuple(int(value) for value in plan.cache_shape),
        "cache_dtype": plan.cache_dtype.name,
        "stores_projection_abs2": False,
        "chunk_rows": int(plan.chunk_rows),
        "chunk_count": int(plan.chunk_count_per_table),
        "retained_bytes": int(plan.retained_bytes),
        "conservative_predicted_peak_bytes": int(plan.predicted_peak_bytes),
        "budget_bytes": int(plan.budget_bytes),
        "admission_destination_alias_proven": bool(
            plan.destination_alias_proven
        ),
        # Informational only: deterministic job 13332001 observed donated
        # insert aliasing for exactly (1, 36864, 5100) with 4608-row chunks
        # on one H100. Admission always reserves a non-aliased copy.
        "h100_alias_evidence_applies_to_plan": h100_alias_evidence_applies,
        "h100_alias_evidence_cache_shape": (1, 36_864, 5_100),
        "h100_alias_evidence_chunk_rows": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        "h100_observed_donated_insert_alias": (
            True if h100_alias_evidence_applies else None
        ),
        "h100_alias_evidence_job_id": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB
        ),
        "h100_observed_alias_peak_bytes": (
            int(plan.predicted_peak_bytes - plan.destination_copy_bytes)
            if h100_alias_evidence_applies
            else None
        ),
        "h100_alias_evidence_used_for_admission": False,
    }


def _project_coarse_gaussian_gemm_projection_cache_block_once(
    cache,
    class_index,
    mean_for_proj,
    rotations_block,
    *,
    rotation_start: int,
):
    """Serve one existing macro block from C64 cache without reprojection."""

    del mean_for_proj
    cache = jnp.asarray(cache)
    if cache.ndim != 3 or np.dtype(cache.dtype) != np.dtype(np.complex64):
        raise TypeError(
            "coarse GEMM projection cache must have shape "
            "(table, rotation, pixel) and dtype complex64",
        )
    table_index = int(class_index)
    if table_index < 0 or table_index >= int(cache.shape[0]):
        raise IndexError(
            f"coarse GEMM projection-cache table {table_index} is out of range",
        )
    start = int(rotation_start)
    requested_rows = int(rotations_block.shape[0])
    if start < 0 or requested_rows <= 0 or start >= int(cache.shape[1]):
        raise IndexError(
            "coarse GEMM projection-cache block must start inside the cache "
            "and contain at least one row",
        )
    stop = min(start + requested_rows, int(cache.shape[1]))
    projected_reference = cache[table_index, start:stop]
    padding_rows = requested_rows - int(projected_reference.shape[0])
    if padding_rows:
        # The shared significance loop masks these physical tail rows to -inf.
        # Zero padding preserves its fixed score-block shape without projecting
        # synthetic identity rotations or changing any valid cached row.
        projected_reference = jnp.pad(
            projected_reference,
            ((0, padding_rows), (0, 0)),
        )
    # This is the same C64 expression used by the mature projector callback.
    # The promoted certificate deliberately ignores this companion and forms
    # its two component squares explicitly after conversion to FP64.
    projected_reference_abs2 = jnp.abs(projected_reference) ** 2
    return projected_reference, projected_reference_abs2


def _validate_coarse_gaussian_gemm_hybrid_request(
    *,
    macro_enabled: bool,
    projection_cache_enabled: bool,
    n_classes: int,
    n_rotations: int,
    score_mode: str,
    coarse_gaussian_ffi_enabled: bool,
    exact_coarse_operands_enabled: bool,
    relion_f32_coarse_support_enabled: bool,
    collect_significance: bool,
    any_diagnostic_requested: bool,
) -> None:
    """Fail closed unless the first production hybrid contract is complete."""

    prefix = f"{_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1 requires"
    if not macro_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
        )
    if not projection_cache_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1",
        )
    if int(n_classes) != 1:
        raise ValueError(f"{prefix} K=1, got K={int(n_classes)}")
    if int(n_rotations) <= 0 or int(n_rotations) % SOURCE_ROTATION_BLOCK_SIZE:
        raise ValueError(
            f"{prefix} a rotation count divisible by "
            f"{SOURCE_ROTATION_BLOCK_SIZE}, got {int(n_rotations)}",
        )
    if score_mode != "gaussian":
        raise ValueError(f"{prefix} score_mode='gaussian'")
    if not coarse_gaussian_ffi_enabled:
        raise ValueError(f"{prefix} the exact RELION coarse Gaussian FFI path")
    if not exact_coarse_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if not relion_f32_coarse_support_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_F32_COARSE_SUPPORT_ENV}=1",
        )
    if not collect_significance:
        raise ValueError(f"{prefix} collect_significance=True")
    if any_diagnostic_requested:
        raise ValueError(
            f"{prefix} paired coarse GEMM diagnostics to be disabled",
        )


class CoarseGaussianGemmHybridBatchResult(NamedTuple):
    """One batch's exact selected score layout or full-direct fallback."""

    scores: jax.Array | None
    raw_score_max: jax.Array
    scores_include_priors: bool
    used_selected_rescore: bool
    fallback_reason: str | None
    selection: CoarseGemmHybridBlockSelection
    compact_scores: CoarseGemmHybridCompactScores | None = None
    score_representation: str = "compact_selected_exact"


def _select_coarse_gaussian_gemm_score_representation(
    *,
    n_rotations: int,
    block_capacity: int,
    compact_posterior: bool,
) -> str:
    """Choose the smaller physical exact-score layout before certification."""

    n_rotations = operator.index(n_rotations)
    block_capacity = operator.index(block_capacity)
    if n_rotations <= 0 or block_capacity <= 0:
        raise ValueError("hybrid score-layout dimensions must be positive")
    if (
        compact_posterior
        and block_capacity * SOURCE_ROTATION_BLOCK_SIZE >= n_rotations
    ):
        return "dense_full_direct_static_capacity"
    if compact_posterior:
        return "compact_selected_exact"
    return "dense_selected_exact"


def _compute_coarse_gaussian_gemm_hybrid_batch(
    projection_cache,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    *,
    topology,
    actual_image_count: int,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    certificate_chunk_rows: int = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS,
    block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
    compact_posterior: bool = False,
    force_static_dense_after_overflow: bool = False,
) -> CoarseGaussianGemmHybridBatchResult:
    """Certify, exactly rescore, and restore one K=1 coarse score table.

    The expanded FP64 GEMMs only choose complete source-16 rotation blocks.
    Published values always come from the mature direct CUDA arithmetic.  Any
    incomplete certificate, capacity overflow, or invalid selected output
    routes the whole padded image batch through one full rectangular direct
    call, preserving a simple fail-closed boundary.
    """

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
        validate_coarse_gemm_certificate_topology,
    )

    validate_coarse_gemm_certificate_topology(topology)
    cache = jnp.asarray(projection_cache)
    shifted = jnp.asarray(shifted_corrected)
    weight = jnp.asarray(pixel_weight)
    initial = jnp.asarray(initial_diff2)
    if cache.ndim != 3 or tuple(cache.shape[:1]) != (1,):
        raise ValueError(
            "certified coarse GEMM hybrid requires one [1,R,F] projection table",
        )
    if cache.dtype != jnp.complex64:
        raise TypeError("certified coarse GEMM hybrid projection cache must be complex64")
    if shifted.ndim != 3:
        raise ValueError(
            "certified coarse GEMM hybrid shifted images must have shape [B,T,F]",
        )
    batch_size, n_translations, n_pixels = map(int, shifted.shape)
    n_rotations = int(cache.shape[1])
    if (
        shifted.dtype != jnp.complex64
        or weight.dtype != jnp.float32
        or initial.dtype != jnp.float32
        or tuple(weight.shape) != (batch_size, n_pixels)
        or tuple(initial.shape) != (batch_size,)
        or int(cache.shape[2]) != n_pixels
        or n_rotations <= 0
        or n_rotations % SOURCE_ROTATION_BLOCK_SIZE
        or topology.compact_pixel_count != n_pixels
        or topology.translation_count != n_translations
    ):
        raise ValueError(
            "certified coarse GEMM hybrid operands have inconsistent shapes, "
            "dtypes, or topology",
        )
    try:
        chunk_rows = operator.index(certificate_chunk_rows)
        capacity = operator.index(block_capacity)
        actual_count = operator.index(actual_image_count)
    except TypeError as error:
        raise ValueError("certified coarse GEMM hybrid counts must be integers") from error
    if (
        chunk_rows <= 0
        or chunk_rows % SOURCE_ROTATION_BLOCK_SIZE
        or capacity <= 0
        or actual_count <= 0
        or actual_count > batch_size
    ):
        raise ValueError(
            "certified coarse GEMM hybrid requires positive aligned chunk, "
            "capacity, and image counts",
        )
    if not isinstance(force_static_dense_after_overflow, (bool, np.bool_)):
        raise ValueError("force_static_dense_after_overflow must be boolean")

    rotation_prior = None
    if rotation_log_prior is not None:
        rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
        if tuple(rotation_prior.shape) != (n_rotations,):
            raise ValueError(
                "hybrid rotation_log_prior must have one value per source rotation",
            )
    score_representation = (
        "dense_full_direct_static_capacity"
        if force_static_dense_after_overflow
        else _select_coarse_gaussian_gemm_score_representation(
            n_rotations=n_rotations,
            block_capacity=capacity,
            compact_posterior=compact_posterior,
        )
    )
    if score_representation == "dense_full_direct_static_capacity":
        empty_counts = np.zeros(batch_size, dtype=np.int32)
        selection = CoarseGemmHybridBlockSelection(
            eligible=False,
            fallback_reason=(
                "prior_batch_block_capacity_overflow"
                if force_static_dense_after_overflow
                else "compact_physical_capacity_not_smaller_than_dense"
            ),
            block_ids=np.full((batch_size, capacity), -1, dtype=np.int32),
            block_count=empty_counts,
            posterior_block_count=empty_counts.copy(),
            raw_max_block_count=empty_counts.copy(),
        )
    else:
        image_batch = _prepare_relion_coarse_gaussian_gemm_f64_image_batch(
            shifted,
            weight,
            initial,
            actual_count,
        )
        state = initialize_coarse_gemm_hybrid_interval_state(
            batch_size,
            n_rotations,
        )
        for rotation_start in range(0, n_rotations, chunk_rows):
            rotation_stop = min(rotation_start + chunk_rows, n_rotations)
            state = _relion_coarse_gaussian_gemm_update_certificate_state(
                state,
                cache[0, rotation_start:rotation_stop],
                image_batch,
                topology=topology,
                rotation_offset=rotation_start,
                class_log_prior=class_log_prior,
                rotation_log_prior=(
                    None
                    if rotation_prior is None
                    else rotation_prior[rotation_start:rotation_stop]
                ),
                translation_log_prior=translation_log_prior,
            )
        selection = select_coarse_gemm_hybrid_rotation_blocks(
            state,
            actual_image_count=actual_count,
            n_rotations=n_rotations,
            n_translations=n_translations,
            certificate_valid=True,
            block_capacity=capacity,
        )
    fallback_reason = selection.fallback_reason
    if selection.eligible:
        selected_diff2 = _relion_coarse_diff2_rotation_blocks_from_topology_f32(
            cache[0],
            shifted,
            weight,
            initial,
            jnp.asarray(selection.block_ids, dtype=jnp.int32),
            topology=topology,
        )
        assemble = (
            assemble_coarse_gemm_hybrid_compact_scores_f32
            if compact_posterior
            else assemble_coarse_gemm_hybrid_dense_scores_f32
        )
        assembled = assemble(
            selected_diff2,
            selection,
            actual_image_count=actual_count,
            n_rotations=n_rotations,
            class_log_prior=class_log_prior,
            rotation_log_prior=rotation_prior,
            translation_log_prior=translation_log_prior,
        )
        if np.all(np.asarray(assembled.selected_output_valid, dtype=bool)):
            compact_scores = assembled if compact_posterior else None
            return CoarseGaussianGemmHybridBatchResult(
                scores=(
                    None
                    if compact_posterior
                    else assembled.posterior_scores_flat.reshape(
                        batch_size,
                        n_rotations,
                        n_translations,
                    )
                ),
                raw_score_max=jnp.where(
                    jnp.arange(batch_size, dtype=jnp.int32) < actual_count,
                    assembled.raw_score_max,
                    jnp.float32(0.0),
                ),
                scores_include_priors=True,
                used_selected_rescore=True,
                fallback_reason=None,
                selection=selection,
                compact_scores=compact_scores,
                score_representation=score_representation,
            )
        fallback_reason = "invalid_selected_exact_output"
        score_representation = "dense_full_direct_dynamic_fallback"
    elif score_representation != "dense_full_direct_static_capacity":
        score_representation = "dense_full_direct_dynamic_fallback"

    full_diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
        cache[0],
        shifted,
        weight,
        initial,
        jnp.asarray(topology.full_to_compact),
    )
    raw_scores = -full_diff2
    return CoarseGaussianGemmHybridBatchResult(
        scores=raw_scores,
        raw_score_max=jnp.max(raw_scores.reshape(batch_size, -1), axis=1),
        scores_include_priors=False,
        used_selected_rescore=False,
        fallback_reason=fallback_reason or "unspecified_fail_closed_fallback",
        selection=selection,
        score_representation=score_representation,
    )


def _coarse_gaussian_gemm_diagnostic_request() -> tuple[str | None, set[int] | None]:
    """Resolve the default-off paired production-score diagnostic.

    Capture deliberately executes both the direct-square and expanded-square
    scorers on the selected production operands.  It therefore doubles score
    work for those blocks and is never a valid timed-runtime arm.  Runtime
    qualification must use separate diagnostic-off runs; capture artifacts and
    returned statistics report that exclusion explicitly.
    """

    directory = os.environ.get(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV, "").strip()
    target_token_present = bool(
        os.environ.get(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV, "").strip()
    )
    if not directory:
        if target_token_present:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} requires "
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV}",
            )
        return None, None
    targets = parse_env_int_set(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV)
    if not targets:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV} requires a non-empty "
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV}",
        )
    if any(int(target) < 0 for target in targets):
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} must contain "
            "non-negative original image indices",
        )
    return os.path.abspath(os.path.expanduser(directory)), {int(target) for target in targets}


def _coarse_gaussian_gemm_streaming_diagnostic_request(
    *,
    max_significants: int,
) -> tuple[str | None, int | None]:
    """Resolve the all-particle bounded exact-rescore diagnostic.

    The diagnostic retains ``topk + 1`` paired candidates per image (the extra
    row certifies cutoff-tie and band coverage) and reduces error statistics
    across every candidate block on device.  It never serializes a score cube.
    """

    directory = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV,
        "",
    ).strip()
    topk_token = os.environ.get(_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV, "").strip()
    if not directory:
        if topk_token:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} requires "
                f"{_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV}",
            )
        return None, None
    default_topk = max(
        2048,
        int(max_significants) + 64 if int(max_significants) > 0 else 2048,
    )
    try:
        retained_topk = int(topk_token) if topk_token else default_topk
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} must be a positive integer, "
            f"got {topk_token!r}",
        ) from error
    if retained_topk <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} must be a positive integer, "
            f"got {retained_topk}",
        )
    return os.path.abspath(os.path.expanduser(directory)), retained_topk


def _coarse_gaussian_gemm_scope_manifest_path(
    directory: str,
    scope: CoarseGaussianGemmDiagnosticScope,
) -> str:
    return os.path.join(
        directory,
        f"coarse_gemm_scope_{scope.run_id}_{scope.call_id}.json",
    )


def _write_json_exclusive(path: str, payload: dict) -> None:
    """Write immutable diagnostic metadata without hiding path collisions."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite a coarse GEMM diagnostic manifest: {path}",
        ) from error


def _seal_coarse_gaussian_gemm_diagnostic_scope(
    directory: str,
    *,
    scope: CoarseGaussianGemmDiagnosticScope,
    selection_policy: str,
    requested_targets: set[int],
    targets_in_scope: set[int],
    captured_target_counts: dict[int, int],
    artifact_paths: list[str],
) -> tuple[str, str | None]:
    """Seal one call record and, on the final call, its aggregate manifest."""

    requested = sorted(int(value) for value in requested_targets)
    in_scope = sorted(int(value) for value in targets_in_scope)
    counts = {
        str(int(target)): int(captured_target_counts.get(int(target), 0))
        for target in in_scope
    }
    bad_scope_counts = {
        target: count
        for target, count in counts.items()
        if count != 1
    }
    if bad_scope_counts:
        raise RuntimeError(
            "coarse GEMM diagnostic targets must be captured exactly once "
            f"within call {scope.call_id}: {bad_scope_counts}",
        )
    scope_record = {
        "schema_version": 1,
        "run_id": scope.run_id,
        "call_id": scope.call_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "selection_policy": str(selection_policy),
        "requested_original_indices": requested,
        "targets_in_scope": in_scope,
        "targets_explicitly_out_of_scope": sorted(
            set(requested) - set(in_scope),
        ),
        "captured_target_counts": counts,
        "artifact_paths": [os.path.basename(path) for path in artifact_paths],
    }
    scope_path = _coarse_gaussian_gemm_scope_manifest_path(directory, scope)
    _write_json_exclusive(scope_path, scope_record)
    if not scope.finalize:
        return scope_path, None

    scope_records = []
    missing_call_ids = []
    for expected_call_id in scope.expected_call_ids:
        expected_scope = scope._replace(
            call_id=expected_call_id,
            finalize=expected_call_id == scope.expected_call_ids[-1],
        )
        expected_path = _coarse_gaussian_gemm_scope_manifest_path(
            directory,
            expected_scope,
        )
        if not os.path.isfile(expected_path):
            missing_call_ids.append(expected_call_id)
            continue
        with open(expected_path, encoding="utf-8") as stream:
            record = json.load(stream)
        if (
            record.get("run_id") != scope.run_id
            or record.get("call_id") != expected_call_id
            or record.get("expected_call_ids") != list(scope.expected_call_ids)
            or record.get("requested_original_indices") != requested
        ):
            raise RuntimeError(
                "coarse GEMM diagnostic scope manifest does not match the "
                f"aggregate contract: {expected_path}",
            )
        scope_records.append(record)
    if missing_call_ids:
        raise RuntimeError(
            "coarse GEMM diagnostic aggregate is missing expected calls: "
            f"{missing_call_ids}",
        )

    aggregate_counts = {str(target): 0 for target in requested}
    for record in scope_records:
        for target, count in record["captured_target_counts"].items():
            if target not in aggregate_counts:
                raise RuntimeError(
                    "coarse GEMM scope captured an unrequested target: "
                    f"{target}",
                )
            aggregate_counts[target] += int(count)
    missing_targets = [
        int(target)
        for target, count in aggregate_counts.items()
        if count == 0
    ]
    duplicate_targets = [
        int(target)
        for target, count in aggregate_counts.items()
        if count > 1
    ]
    if missing_targets or duplicate_targets:
        raise RuntimeError(
            "coarse GEMM diagnostic aggregate requires every requested target "
            "exactly once; "
            f"missing={missing_targets}, duplicate={duplicate_targets}",
        )
    aggregate_record = {
        "schema_version": 1,
        "run_id": scope.run_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "requested_original_indices": requested,
        "captured_target_counts": aggregate_counts,
        "all_requested_captured_exactly_once": True,
        "scope_records": scope_records,
    }
    aggregate_path = os.path.join(
        directory,
        f"coarse_gemm_manifest_{scope.run_id}.json",
    )
    _write_json_exclusive(aggregate_path, aggregate_record)
    return scope_path, aggregate_path


def _coarse_gaussian_gemm_stream_scope_manifest_path(
    directory: str,
    scope: CoarseGaussianGemmDiagnosticScope,
) -> str:
    return os.path.join(
        directory,
        f"coarse_gemm_rescore_scope_{scope.run_id}_{scope.call_id}.json",
    )


def _seal_coarse_gaussian_gemm_streaming_scope(
    directory: str,
    *,
    scope: CoarseGaussianGemmDiagnosticScope,
    retained_topk: int,
    artifact_paths: list[str],
    original_indices: list[int],
) -> tuple[str, str | None]:
    """Seal compact all-particle summaries across explicit call scopes."""

    particle_ids = [int(value) for value in original_indices]
    if len(particle_ids) != len(set(particle_ids)):
        raise RuntimeError(
            "coarse GEMM streaming diagnostic captured a particle more than once "
            f"inside call {scope.call_id}",
        )
    artifact_particle_ids = []
    for artifact_path in artifact_paths:
        if not os.path.isfile(artifact_path):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic artifact is missing: "
                f"{artifact_path}",
            )
        with np.load(artifact_path, allow_pickle=False) as artifact:
            if (
                artifact.get("schema", np.asarray("")).item()
                != COARSE_GEMM_STREAMING_SCHEMA
                or artifact.get("diagnostic_run_id", np.asarray("")).item()
                != scope.run_id
                or artifact.get("diagnostic_call_id", np.asarray("")).item()
                != scope.call_id
                or bool(artifact.get("stores_score_cube", np.asarray(True)).item())
            ):
                raise RuntimeError(
                    "coarse GEMM streaming diagnostic artifact differs from its "
                    f"scope contract: {artifact_path}",
                )
            artifact_particle_ids.extend(
                int(value) for value in np.asarray(artifact["original_indices"])
            )
    if artifact_particle_ids != particle_ids:
        raise RuntimeError(
            "coarse GEMM streaming diagnostic artifacts do not cover the call's "
            "particle stream in order",
        )
    record = {
        "schema_version": 2,
        "run_id": scope.run_id,
        "call_id": scope.call_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "retained_topk": int(retained_topk),
        "particle_count": len(particle_ids),
        "original_indices": particle_ids,
        "artifact_paths": [os.path.basename(path) for path in artifact_paths],
        "stores_score_cube": False,
    }
    scope_path = _coarse_gaussian_gemm_stream_scope_manifest_path(
        directory,
        scope,
    )
    _write_json_exclusive(scope_path, record)
    if not scope.finalize:
        return scope_path, None

    scope_records = []
    aggregate_particle_ids = []
    for expected_call_id in scope.expected_call_ids:
        expected_scope = scope._replace(
            call_id=expected_call_id,
            finalize=expected_call_id == scope.expected_call_ids[-1],
        )
        expected_path = _coarse_gaussian_gemm_stream_scope_manifest_path(
            directory,
            expected_scope,
        )
        if not os.path.isfile(expected_path):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic aggregate is missing call "
                f"{expected_call_id}: {expected_path}",
            )
        with open(expected_path, encoding="utf-8") as stream:
            expected_record = json.load(stream)
        if (
            expected_record.get("schema_version") != 2
            or expected_record.get("run_id") != scope.run_id
            or expected_record.get("call_id") != expected_call_id
            or expected_record.get("expected_call_ids") != list(scope.expected_call_ids)
            or expected_record.get("retained_topk") != int(retained_topk)
            or expected_record.get("stores_score_cube") is not False
        ):
            raise RuntimeError(
                "coarse GEMM streaming scope manifest does not match the "
                f"aggregate contract: {expected_path}",
            )
        scope_records.append(expected_record)
        aggregate_particle_ids.extend(
            int(value) for value in expected_record["original_indices"]
        )
    if len(aggregate_particle_ids) != len(set(aggregate_particle_ids)):
        raise RuntimeError(
            "coarse GEMM streaming diagnostic captured duplicate particles "
            "across call scopes",
        )
    aggregate_artifact_paths = [
        os.path.join(directory, artifact_name)
        for record in scope_records
        for artifact_name in record["artifact_paths"]
    ]
    aggregate = {
        "schema_version": 2,
        "run_id": scope.run_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "retained_topk": int(retained_topk),
        "particle_count": len(aggregate_particle_ids),
        "all_particles_captured_exactly_once": True,
        "stores_score_cube": False,
        "scope_records": scope_records,
        "summary": aggregate_coarse_gemm_streaming_summaries(
            aggregate_artifact_paths,
        ),
    }
    aggregate_path = os.path.join(
        directory,
        f"coarse_gemm_rescore_manifest_{scope.run_id}.json",
    )
    _write_json_exclusive(aggregate_path, aggregate)
    return scope_path, aggregate_path


def _write_coarse_gaussian_gemm_diagnostic(
    output_path: str,
    *,
    direct_scores_pre_prior,
    macro_scores_pre_prior,
    direct_scores_with_prior,
    macro_scores_with_prior,
    direct_support,
    macro_support,
    original_indices,
    local_indices,
    actual_batch_size: int,
    padded_batch_size: int,
    adaptive_fraction: float,
    max_significants: int,
    resource_estimate: CoarseGaussianGemmResources,
    diagnostic_scope: CoarseGaussianGemmDiagnosticScope,
    diagnostic_selection_policy: str,
    debug_iteration: int | None,
    current_size: int | None,
) -> None:
    """Write one immutable paired score surface for repeat-envelope analysis."""

    if os.path.exists(output_path):
        raise FileExistsError(
            "refusing to overwrite a coarse GEMM diagnostic artifact: "
            f"{output_path}",
        )
    diagnostics = _coarse_gaussian_direct_macro_diagnostics(
        direct_scores_with_prior,
        macro_scores_with_prior,
        direct_support=direct_support,
        macro_support=macro_support,
    )
    direct_pre_prior = np.asarray(direct_scores_pre_prior)
    macro_pre_prior = np.asarray(macro_scores_pre_prior)
    macro_negative_implied_diff2 = macro_pre_prior > 0.0
    direct_negative_implied_diff2 = direct_pre_prior > 0.0
    qualification = _coarse_gaussian_qualification_decision(
        exact_arithmetic_equivalent=True,
        repeat_stable=None,
        unbiased_non_directional=None,
        bounded_non_growing=None,
        discrete_choices_equal=bool(
            np.all(diagnostics["argmax_equal"])
            and np.all(diagnostics["support_equal"])
        ),
        final_basin_quality_equal=None,
        material_runtime_win=None,
        scale_amplified=None,
        negative_implied_diff2=bool(
            np.any(macro_negative_implied_diff2 & ~direct_negative_implied_diff2)
        ),
        nonfinite_scores=bool(np.any(~np.isfinite(macro_pre_prior))),
        exact_zero_cancellation_drift=bool(
            np.any(diagnostics["exact_zero_direct_nonzero_macro_per_image"])
        ),
    )
    payload = {
        "layout": np.asarray("image,class,rotation,translation"),
        "numerical_policy": np.asarray(
            "exact-arithmetic-equivalent_expanded-square_cancellation-sensitive_qualification-only"
        ),
        "qualification_status": np.asarray(qualification["status"]),
        "qualification_policy": np.asarray(
            "allow_repeat-stable_unbiased_bounded_non-growing_noise_never_lone-epsilon_promotion"
        ),
        "automatic_no_go_reasons": np.asarray(
            qualification["failure_reasons"],
            dtype=np.str_,
        ),
        "pending_qualification_gates": np.asarray(
            qualification["pending_gates"],
            dtype=np.str_,
        ),
        "requires_bitwise_score_identity": np.asarray(
            qualification["requires_bitwise_score_identity"],
        ),
        "requires_exact_discrete_identity": np.asarray(
            qualification["requires_exact_discrete_identity"],
        ),
        "repeat_spread_assessment": np.asarray(
            "NO_GO_until_same-hardware_repeat_artifacts_establish_native-relative_spread"
        ),
        "scale_growth_assessment": np.asarray(
            "NO_GO_until_multiscale_artifacts_exclude_scale-amplified_drift"
        ),
        "paired_capture_active": np.asarray(True),
        "clean_timing_eligible": np.asarray(False),
        "timing_policy": np.asarray(
            "paired_capture_executes_both_scorers_use_separate_diagnostic-off_timing_arm"
        ),
        "diagnostic_run_id": np.asarray(diagnostic_scope.run_id),
        "diagnostic_call_id": np.asarray(diagnostic_scope.call_id),
        "diagnostic_selection_policy": np.asarray(diagnostic_selection_policy),
        "debug_iteration": np.asarray(
            -1 if debug_iteration is None else int(debug_iteration),
            dtype=np.int64,
        ),
        "current_size": np.asarray(
            -1 if current_size is None else int(current_size),
            dtype=np.int64,
        ),
        "direct_scores_pre_prior": direct_pre_prior,
        "macro_scores_pre_prior": macro_pre_prior,
        "direct_scores_with_prior": np.asarray(direct_scores_with_prior),
        "macro_scores_with_prior": np.asarray(macro_scores_with_prior),
        "original_indices": np.asarray(original_indices, dtype=np.int64),
        "local_indices": np.asarray(local_indices, dtype=np.int64),
        "actual_batch_size": np.asarray(actual_batch_size, dtype=np.int64),
        "padded_batch_size": np.asarray(padded_batch_size, dtype=np.int64),
        "adaptive_fraction": np.asarray(adaptive_fraction, dtype=np.float64),
        "max_significants": np.asarray(max_significants, dtype=np.int64),
        "direct_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(direct_negative_implied_diff2),
            dtype=np.int64,
        ),
        "macro_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(macro_negative_implied_diff2),
            dtype=np.int64,
        ),
        "macro_only_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(
                macro_negative_implied_diff2 & ~direct_negative_implied_diff2,
            ),
            dtype=np.int64,
        ),
    }
    payload.update(diagnostics)
    payload.update(
        {
            f"resource_{field}": np.asarray(value, dtype=np.int64)
            for field, value in resource_estimate._asdict().items()
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _k1_coarse_fused_projector_supports_padding(padding_factor: int) -> bool:
    """Whether the fused CUDA projector implements this RELION padding."""

    # The fused kernel stages the compact pad-1 Projector texture and samples
    # unscaled Fourier coordinates.  The general texture-projector path below
    # infers and applies larger padding factors from the projector shape.
    return int(padding_factor) == 1


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


def _k1_relion_exact_coarse_skip_generic_operands_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact coarse operands bypass overwritten generic operands."""

    token = os.environ.get(
        _K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "Unsupported "
        f"{_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV}={token!r}",
    )


def _resolve_k1_relion_exact_coarse_skip_generic_operands(
    *,
    requested: bool,
    exact_coarse_operands_enabled: bool,
) -> bool:
    """Resolve the exact-source-only operand path, failing closed."""

    if requested and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV}=1 requires "
            f"effective {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1 Gaussian scoring",
        )
    return bool(requested and exact_coarse_operands_enabled)


def _k1_relion_exact_coarse_assembly_profile_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact-coarse call-count diagnostics are published."""

    token = os.environ.get(
        _K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE_ENV}={token!r}",
    )


def _k1_relion_exact_compact_preprocess_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact compact scoring skips unused generic preprocessing."""

    token = os.environ.get(
        _K1_RELION_EXACT_COMPACT_PREPROCESS_ENV,
        "1" if default else "0",
    ).strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Unsupported {_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV}={token!r}",
    )


def _resolve_k1_relion_exact_compact_preprocess(
    *,
    requested: bool,
    exact_coarse_skip_generic_operands_enabled: bool,
    exact_coarse_operands_enabled: bool,
    coarse_gaussian_gemm_hybrid_requested: bool,
    coarse_gaussian_gemm_compact_posterior_requested: bool,
    score_mode: str,
    any_diagnostic_requested: bool,
) -> bool:
    """Resolve the narrow exact+compact preprocessing specialization."""

    if not requested:
        return False
    prefix = f"{_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV}=1 requires"
    if not exact_coarse_skip_generic_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV}=1",
        )
    if not exact_coarse_operands_enabled:
        raise ValueError(f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1")
    if not coarse_gaussian_gemm_hybrid_requested:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if not coarse_gaussian_gemm_compact_posterior_requested:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
        )
    if score_mode != "gaussian":
        raise ValueError(f"{prefix} score_mode='gaussian'")
    if any_diagnostic_requested:
        raise ValueError(f"{prefix} paired coarse diagnostics to be disabled")
    return True


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


class RelionExactCoarseGaussianOperands(NamedTuple):
    """Exact-source operands consumed by both EM and InitialModel scoring."""

    shifted_corrected: jax.Array
    pixel_weight: jax.Array
    unshifted_corrected: jax.Array
    initial_diff2: jax.Array
    translation_angles: jax.Array


def _process_relion_exact_coarse_half_image(
    experiment_dataset,
    batch_data,
    score_with_masked_images: bool,
    *,
    relion_preprocess_kwargs,
):
    """Run the one canonical per-image RELION FFT used by exact coarse scoring."""

    if relion_preprocess_kwargs is None:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires RELION CUDA "
            "image preprocessing",
        )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        process_half_image,
    )

    exact_preprocess_kwargs = dict(relion_preprocess_kwargs)
    exact_preprocess_kwargs["relion_fft_per_image"] = True
    return process_half_image(
        experiment_dataset,
        batch_data,
        score_with_masked_images,
        relion_preprocess_kwargs=exact_preprocess_kwargs,
    )


def _assemble_relion_exact_coarse_gaussian_operands(
    experiment_dataset,
    processed_direct,
    indices,
    *,
    batch_scale_np,
    actual_batch_size: int,
    batch_size: int,
    score_indices,
    score_active_mask,
    translations_source,
    image_shape,
    noise_variance_half,
    scale_corrections_enabled: bool,
    half_weights,
    powerclass,
    current_size,
) -> RelionExactCoarseGaussianOperands:
    """Assemble the single exact-source operand set without generic formulas."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_corr_img_from_native_noise_variance,
        _relion_cuda_pixel_correction_from_rfloat_ctf,
        _relion_exact_ctf_half_from_source_star,
        _relion_translation_angles_f32,
    )

    ctf_half_rfloat_np = np.asarray(
        _relion_exact_ctf_half_from_source_star(
            experiment_dataset,
            indices,
            image_shape,
        ),
        dtype=np.float64,
    )
    if batch_size > actual_batch_size:
        ctf_half_rfloat_np = _repeat_pad_batch_axis(
            ctf_half_rfloat_np,
            batch_size,
        )
    ctf_half_rfloat = jnp.asarray(ctf_half_rfloat_np, dtype=jnp.float64)
    batch_scale_f32 = jnp.asarray(batch_scale_np, dtype=jnp.float32)
    pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
        batch_scale_f32[:, None],
        ctf_half_rfloat,
    )
    exact_unshifted_corrected = processed_direct * pixel_correction
    exact_unshifted_corrected = exact_unshifted_corrected[:, score_indices]
    exact_unshifted_corrected = jnp.where(
        score_active_mask[None, :],
        exact_unshifted_corrected,
        jnp.zeros((), dtype=exact_unshifted_corrected.dtype),
    ).astype(jnp.complex64)
    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(translations_source, image_shape),
        dtype=jnp.float32,
    )
    shifted_corrected = cuda_backproject.relion_translate_score_f32(
        exact_unshifted_corrected,
        translation_angles,
        score_indices,
        image_shape,
    ).reshape(batch_size, int(translation_angles.shape[0]), -1)
    exact_corr_img = _relion_cuda_corr_img_from_native_noise_variance(
        noise_variance_half[None, :],
        ctf_half_rfloat,
        image_shape,
        batch_scale_f32[:, None] if scale_corrections_enabled else None,
    )
    exact_square_corr_img = exact_corr_img[:, score_indices]
    exact_square_corr_img = jnp.where(
        score_active_mask[None, :],
        exact_square_corr_img,
        jnp.zeros((), dtype=exact_square_corr_img.dtype),
    )
    pixel_weight = jnp.asarray(
        exact_square_corr_img
        * jnp.asarray(half_weights[score_indices], dtype=jnp.float32)[None, :],
        dtype=jnp.float32,
    )
    return RelionExactCoarseGaussianOperands(
        shifted_corrected=jnp.asarray(shifted_corrected, dtype=jnp.complex64),
        pixel_weight=pixel_weight,
        unshifted_corrected=exact_unshifted_corrected,
        initial_diff2=powerclass(
            processed_direct,
            image_shape=image_shape,
            current_size=current_size,
        ),
        translation_angles=translation_angles,
    )


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


def _score_relion_coarse_gaussian_gemm_macro(
    project_block_once,
    class_index,
    mean_for_proj,
    rotations_block,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    image_shape,
    volume_shape,
    return_projected: bool = False,
):
    """Project once, then score every physical image lane through shared GEMMs."""

    projected_reference, projected_reference_abs2 = project_block_once(
        class_index,
        mean_for_proj,
        rotations_block,
    )
    scores = _relion_coarse_gaussian_gemm_scores(
        projected_reference,
        projected_reference_abs2,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )
    if return_projected:
        return scores, projected_reference, projected_reference_abs2
    return scores


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


def _build_coarse_significance_support_audit(
    significant_sample_indices,
    *,
    samples_per_class: int,
    include_ids: bool = False,
) -> dict:
    """Hash every ordered class/image coarse support without changing it.

    The canonical byte stream for one row is four little-endian int64 header
    values ``(class, image, samples_per_class, selected_count)`` followed by
    the strictly increasing selected sample IDs as little-endian int64. Each
    length-prefixed row enters the aggregate digest in class-major/image-major
    order. Per-row digests and counts make any mismatch localizable while the
    aggregate digest provides a compact direct/hybrid equality gate.
    """

    try:
        total_size = operator.index(samples_per_class)
    except TypeError as error:
        raise ValueError("samples_per_class must be an integer") from error
    if total_size <= 0:
        raise ValueError("samples_per_class must be positive")
    if not isinstance(significant_sample_indices, (tuple, list)) or not significant_sample_indices:
        raise ValueError("support audit requires at least one class")
    n_images = None
    aggregate = hashlib.sha256()
    per_class_image_sha256: list[list[str]] = []
    per_class_counts: list[list[int]] = []
    per_class_ids: list[list[list[int]]] = []
    for class_index, rows in enumerate(significant_sample_indices):
        if not isinstance(rows, (tuple, list)):
            raise TypeError("support audit class rows must be a sequence")
        if n_images is None:
            n_images = len(rows)
            if n_images <= 0:
                raise ValueError("support audit requires at least one image")
        elif len(rows) != n_images:
            raise ValueError("support audit classes must cover the same images")
        row_digests = []
        row_counts = []
        row_ids: list[list[int]] = []
        for image_index, samples in enumerate(rows):
            ids = np.asarray(
                significant_sample_ids(samples, total_size),
                dtype=np.int64,
            ).reshape(-1)
            if (
                np.any(ids < 0)
                or np.any(ids >= total_size)
                or (ids.size > 1 and np.any(np.diff(ids) <= 0))
            ):
                raise ValueError(
                    "support audit requires unique, strictly increasing in-range IDs",
                )
            header = np.asarray(
                (class_index, image_index, total_size, ids.size),
                dtype="<i8",
            )
            ids_le = np.ascontiguousarray(ids.astype("<i8", copy=False))
            row_bytes = header.tobytes(order="C") + ids_le.tobytes(order="C")
            row_digests.append(hashlib.sha256(row_bytes).hexdigest())
            row_counts.append(int(ids.size))
            if include_ids:
                row_ids.append([int(value) for value in ids])
            aggregate.update(
                np.asarray((len(row_bytes),), dtype="<u8").tobytes(order="C"),
            )
            aggregate.update(row_bytes)
        per_class_image_sha256.append(row_digests)
        per_class_counts.append(row_counts)
        if include_ids:
            per_class_ids.append(row_ids)

    counts = np.ascontiguousarray(np.asarray(per_class_counts, dtype="<i8"))
    result = {
        "schema": "recovar.coarse_significance_support_audit.v2",
        "classification": "diagnostic_only",
        "canonical_encoding": (
            "class-major/image-major; uint64 row-byte-length; "
            "int64-le header(class,image,total,count); int64-le sorted IDs"
        ),
        "n_classes": len(per_class_counts),
        "n_images": int(n_images),
        "samples_per_class": total_size,
        "selected_count_sum": int(np.sum(counts, dtype=np.int64)),
        "selected_count_min": int(np.min(counts)),
        "selected_count_max": int(np.max(counts)),
        "per_class_image_selected_counts": per_class_counts,
        "per_class_image_selected_counts_sha256": hashlib.sha256(
            counts.tobytes(order="C"),
        ).hexdigest(),
        "support_ids_included": bool(include_ids),
        "per_class_image_support_sha256": per_class_image_sha256,
        "aggregate_support_sha256": aggregate.hexdigest(),
    }
    if include_ids:
        # Explicit opt-in avoids retaining a potentially enormous full-support
        # diagnostic in other geometries. GF46 iteration 181 has only ~5,900
        # selected IDs across all 1,000 images.
        result["per_class_image_support_ids"] = per_class_ids
    return result


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
    relion_projector_half=None,
    relion_projector_r_max=None,
    projection_padding_factor=None,
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
    sig_mask_array = np.asarray(batch_sig_mask, dtype=bool)
    sig_mask_full = sig_mask_array.reshape(
        sig_mask_array.shape[0],
        n_classes,
        n_rot * n_trans,
    )[: local_indices.shape[0]]
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

    projector_half_per_class = None
    if relion_projector_half is not None:
        projector_half_per_class = np.stack(
            [np.asarray(value, dtype=np.complex64) for value in relion_projector_half],
            axis=0,
        )
        if projector_half_per_class.shape[0] != n_classes:
            raise ValueError(
                "RELION projector dump class count differs from significance class count: "
                f"{projector_half_per_class.shape[0]} != {n_classes}",
            )

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
            relion_projector_half_per_class=(
                projector_half_per_class
                if projector_half_per_class is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            relion_projector_r_max=np.int64(
                -1 if relion_projector_r_max is None else int(relion_projector_r_max)
            ),
            projection_padding_factor=np.int64(
                -1 if projection_padding_factor is None else int(projection_padding_factor)
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
    relion_f32_coarse_tie_ulps: int = 0,
    pad_final_image_batch: bool = False,
    coarse_gemm_diagnostic_scope: CoarseGaussianGemmDiagnosticScope | None = None,
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
    input_image_batch_size = operator.index(image_batch_size)
    if input_image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    image_batch_size = input_image_batch_size
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
    coarse_fused_projector_requested = _k1_coarse_fused_projector_enabled(
        default=(
            relion_coarse_gaussian_default
            and coarse_gaussian_ffi_enabled
            and coarse_gaussian_sincosf_enabled
        ),
    )
    coarse_fused_projector_enabled = (
        coarse_fused_projector_requested
        and score_mode == "gaussian"
        and _k1_coarse_fused_projector_supports_padding(projection_padding_factor)
    )
    coarse_canonical_reduction_requested = (
        _relion_coarse_canonical_reduction_enabled(
            default=(
                relion_coarse_gaussian_default
                and coarse_fused_projector_enabled
            ),
        )
    )
    coarse_native_atomic_reduction_requested = (
        _k1_coarse_native_atomic_reduction_enabled()
    )
    coarse_native_atomic_reduction_enabled = (
        _k1_coarse_native_atomic_reduction_selected(
            requested=coarse_native_atomic_reduction_requested,
            score_mode=score_mode,
            translation_count=n_trans,
        )
    )
    if (
        coarse_native_atomic_reduction_enabled
        and _RELION_COARSE_CANONICAL_REDUCTION_ENV in os.environ
        and coarse_canonical_reduction_requested
    ):
        raise ValueError(
            f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 conflicts with "
            f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV}=1",
        )
    coarse_canonical_reduction_enabled = bool(
        coarse_canonical_reduction_requested
        and score_mode == "gaussian"
        and not coarse_native_atomic_reduction_enabled
    )
    if coarse_canonical_reduction_enabled and not coarse_fused_projector_enabled:
        raise ValueError(
            f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV} requires "
            f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
        )
    if coarse_native_atomic_reduction_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 currently "
                "supports K=1 only",
            )
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 requires "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
            )
    elif coarse_native_atomic_reduction_requested and score_mode == "gaussian":
        logger.debug(
            "RELION native atomic coarse reduction unavailable for %d "
            "translations; retaining the configured canonical reduction",
            n_trans,
        )
    coarse_prehalf_weight_requested = _k1_coarse_prehalf_weight_enabled()
    coarse_prehalf_weight_enabled = bool(
        coarse_prehalf_weight_requested and coarse_native_atomic_reduction_enabled
    )
    if coarse_prehalf_weight_requested and not coarse_prehalf_weight_enabled:
        raise ValueError(
            f"{_K1_COARSE_PREHALF_WEIGHT_ENV}=1 requires the effective "
            f"{_K1_COARSE_NATIVE_ATOMIC_REDUCTION_ENV}=1 K=1 Gaussian T=29 path",
        )
    coarse_single_lane_canonical_requested = (
        _k1_coarse_single_lane_canonical_enabled()
    )
    coarse_single_lane_canonical_enabled = (
        _k1_coarse_single_lane_canonical_selected(
            requested=coarse_single_lane_canonical_requested,
            score_mode=score_mode,
            translation_count=n_trans,
        )
    )
    if coarse_single_lane_canonical_enabled:
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}=1 requires "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1",
            )
        if not coarse_canonical_reduction_enabled:
            raise ValueError(
                f"{_K1_COARSE_SINGLE_LANE_CANONICAL_ENV}=1 requires "
                f"{_RELION_COARSE_CANONICAL_REDUCTION_ENV}=1",
            )
    elif coarse_single_lane_canonical_requested and score_mode == "gaussian":
        logger.debug(
            "RELION coarse single-lane specialization unavailable for %d "
            "translations; retaining generic canonical reduction",
            n_trans,
        )
    coarse_multistream_worker_count = _k1_coarse_multistream_worker_count()
    coarse_multistream_enabled = (
        coarse_multistream_worker_count > 0 and score_mode == "gaussian"
    )
    if coarse_multistream_enabled:
        if n_classes != 1:
            raise ValueError(
                f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV}=8 currently supports K=1 only",
            )
        if not coarse_fused_projector_enabled:
            raise ValueError(
                f"{_K1_COARSE_MULTISTREAM_WORKERS_ENV}=8 requires the accepted "
                f"{_K1_COARSE_FUSED_PROJECTOR_ENV}=1 path",
            )
    if (
        coarse_fused_projector_requested
        and score_mode == "gaussian"
        and not _k1_coarse_fused_projector_supports_padding(projection_padding_factor)
    ):
        logger.warning(
            "K=1 RELION fused coarse projector disabled for padding_factor=%d; "
            "using the padding-aware projector and exact diff2 path",
            int(projection_padding_factor),
        )
    if coarse_fused_projector_enabled and not (
        coarse_gaussian_ffi_enabled and coarse_gaussian_sincosf_enabled
    ):
        raise ValueError(
            f"{_K1_COARSE_FUSED_PROJECTOR_ENV} requires "
            f"{_K1_COARSE_GAUSSIAN_FFI_ENV}=1 and "
            f"{_K1_COARSE_GAUSSIAN_SINCOSF_ENV}=1",
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
    exact_coarse_skip_generic_operands_requested = (
        _k1_relion_exact_coarse_skip_generic_operands_enabled()
    )
    exact_coarse_skip_generic_operands_enabled = (
        _resolve_k1_relion_exact_coarse_skip_generic_operands(
            requested=exact_coarse_skip_generic_operands_requested,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
        )
    )
    exact_coarse_assembly_profile_enabled = (
        _k1_relion_exact_coarse_assembly_profile_enabled()
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
    coarse_gaussian_gemm_macro_requested = _coarse_gaussian_gemm_macro_enabled()
    coarse_gaussian_gemm_projection_cache_requested = (
        _coarse_gaussian_gemm_projection_cache_enabled()
    )
    coarse_gaussian_gemm_hybrid_requested = (
        _coarse_gaussian_gemm_hybrid_enabled()
    )
    coarse_gaussian_gemm_compact_posterior_requested = (
        _coarse_gaussian_gemm_compact_posterior_enabled()
    )
    coarse_gaussian_gemm_hybrid_image_batch_size_request = (
        _coarse_gaussian_gemm_hybrid_image_batch_size_request()
    )
    if coarse_gaussian_gemm_compact_posterior_requested:
        _validate_coarse_gaussian_gemm_compact_posterior_request(
            hybrid_enabled=coarse_gaussian_gemm_hybrid_requested,
            return_class_best=return_class_best,
            return_class_second=return_class_second,
        )
    coarse_gaussian_gemm_hybrid_capacity = (
        _coarse_gaussian_gemm_hybrid_block_capacity()
        if coarse_gaussian_gemm_hybrid_requested
        else DEFAULT_ROTATION_BLOCK_CAPACITY
    )
    coarse_gaussian_score_backend = _resolve_coarse_gaussian_score_backend(
        gemm_macro_requested=coarse_gaussian_gemm_macro_requested,
        score_mode=score_mode,
        fused_projector_requested=coarse_fused_projector_requested,
        fused_projector_enabled=coarse_fused_projector_enabled,
        canonical_reduction_requested=coarse_canonical_reduction_requested,
        canonical_reduction_enabled=coarse_canonical_reduction_enabled,
        native_atomic_reduction_requested=coarse_native_atomic_reduction_requested,
        native_atomic_reduction_enabled=coarse_native_atomic_reduction_enabled,
        single_lane_canonical_requested=coarse_single_lane_canonical_requested,
        single_lane_canonical_enabled=coarse_single_lane_canonical_enabled,
        multistream_requested=coarse_multistream_worker_count > 0,
        multistream_enabled=coarse_multistream_enabled,
        native_texture_requested=coarse_gaussian_native_texture_requested,
        native_texture_enabled=coarse_gaussian_native_texture_enabled,
    )
    coarse_gaussian_gemm_macro_enabled = (
        coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.GEMM_MACRO
    )
    (
        coarse_gaussian_gemm_diagnostic_dir,
        coarse_gaussian_gemm_diagnostic_targets,
    ) = _coarse_gaussian_gemm_diagnostic_request()
    (
        coarse_gaussian_gemm_stream_diagnostic_dir,
        coarse_gaussian_gemm_stream_topk,
    ) = _coarse_gaussian_gemm_streaming_diagnostic_request(
        max_significants=max_significants,
    )
    coarse_gaussian_gemm_requested_targets = coarse_gaussian_gemm_diagnostic_targets
    coarse_gaussian_gemm_diagnostic_scope = None
    coarse_gaussian_gemm_diagnostic_selection_policy = None
    coarse_gaussian_gemm_any_diagnostic = bool(
        coarse_gaussian_gemm_diagnostic_dir is not None
        or coarse_gaussian_gemm_stream_diagnostic_dir is not None
    )
    if coarse_gaussian_gemm_any_diagnostic:
        if not coarse_gaussian_gemm_macro_enabled:
            raise ValueError(
                "coarse GEMM diagnostics require "
                f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
            )
        if not collect_significance:
            raise ValueError(
                "coarse GEMM diagnostics require "
                "collect_significance=True so exact support can be compared",
            )
        (
            coarse_gaussian_gemm_diagnostic_scope,
            explicitly_scoped_diagnostic,
        ) = _resolve_coarse_gaussian_gemm_diagnostic_scope(
            coarse_gemm_diagnostic_scope,
            debug_iteration=debug_iteration,
            current_size=current_size,
        )
    if coarse_gaussian_gemm_diagnostic_dir is not None:
        logger.warning(
            "coarse GEMM paired capture is requested; calls containing target "
            "particles execute both direct and GEMM scorers, and the configured "
            "run is excluded from runtime qualification"
        )
        if explicitly_scoped_diagnostic:
            available_original_indices = set(
                int(value)
                for value in _original_indices_for_local(
                    experiment_dataset,
                    np.arange(n_images, dtype=np.int64),
                )
            )
            coarse_gaussian_gemm_diagnostic_targets = (
                coarse_gaussian_gemm_requested_targets & available_original_indices
            )
            coarse_gaussian_gemm_diagnostic_selection_policy = (
                "explicit_call_scope_intersection"
            )
        else:
            coarse_gaussian_gemm_diagnostic_selection_policy = "strict_single_call"
    if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
        logger.warning(
            "coarse GEMM all-particle streaming exact-rescore diagnostic is "
            "requested (topk=%d); every score block executes both scorers, only "
            "bounded summaries are retained, and the run is excluded from "
            "runtime qualification",
            coarse_gaussian_gemm_stream_topk,
        )
    if coarse_gaussian_gemm_macro_enabled and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1 requires "
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if coarse_gaussian_gemm_projection_cache_requested:
        _validate_coarse_gaussian_gemm_projection_cache_request(
            macro_enabled=coarse_gaussian_gemm_macro_enabled,
            n_classes=n_classes,
            n_rotations=n_rot,
            coarse_gaussian_ffi_enabled=coarse_gaussian_ffi_enabled,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            use_relion_projector=use_relion_projector,
            relion_texture_interp_enabled=coarse_texture_interp,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_scoring=use_float64_scoring,
            relion_projector_dtype=(
                relion_projector_half[0].dtype if use_relion_projector else None
            ),
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
    if coarse_gaussian_gemm_hybrid_requested:
        _validate_coarse_gaussian_gemm_hybrid_request(
            macro_enabled=coarse_gaussian_gemm_macro_enabled,
            projection_cache_enabled=coarse_gaussian_gemm_projection_cache_requested,
            n_classes=n_classes,
            n_rotations=n_rot,
            score_mode=score_mode,
            coarse_gaussian_ffi_enabled=coarse_gaussian_ffi_enabled,
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            relion_f32_coarse_support_enabled=relion_f32_coarse_support_enabled,
            collect_significance=collect_significance,
            any_diagnostic_requested=coarse_gaussian_gemm_any_diagnostic,
        )
    exact_compact_preprocess_requested = (
        _k1_relion_exact_compact_preprocess_enabled()
    )
    exact_compact_preprocess_enabled = (
        _resolve_k1_relion_exact_compact_preprocess(
            requested=exact_compact_preprocess_requested,
            exact_coarse_skip_generic_operands_enabled=(
                exact_coarse_skip_generic_operands_enabled
            ),
            exact_coarse_operands_enabled=exact_coarse_operands_enabled,
            coarse_gaussian_gemm_hybrid_requested=(
                coarse_gaussian_gemm_hybrid_requested
            ),
            coarse_gaussian_gemm_compact_posterior_requested=(
                coarse_gaussian_gemm_compact_posterior_requested
            ),
            score_mode=score_mode,
            any_diagnostic_requested=coarse_gaussian_gemm_any_diagnostic,
        )
    )
    if exact_compact_preprocess_enabled and (
        collect_significance
        and _significance_debug_dump_matches(
            current_size=current_size,
            debug_iteration=debug_iteration,
        )
    ):
        raise ValueError(
            f"{_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV}=1 does not support "
            "raw significance score dumps",
        )
    if coarse_gaussian_gemm_hybrid_image_batch_size_request is not None:
        prefix = (
            f"{_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE_ENV} requires"
        )
        if not coarse_gaussian_gemm_hybrid_requested:
            raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
        if not coarse_gaussian_gemm_compact_posterior_requested:
            raise ValueError(
                f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
            )
    if relion_f32_coarse_support_enabled:
        if use_float64_scoring:
            raise ValueError(
                f"{_K1_RELION_F32_COARSE_SUPPORT_ENV} requires production float32 scoring",
            )
        logger.warning(
            "RELION CUDA float32 coarse support enabled (%s): current_size=%d",
            (
                "guarded fresh InitialModel default"
                if relion_coarse_gaussian_default
                and _K1_RELION_F32_COARSE_SUPPORT_ENV not in os.environ
                else "environment override"
            ),
            int(image_shape[0]) if current_size is None else int(current_size),
        )
    coarse_gaussian_full_to_compact = None
    coarse_gaussian_full_to_compact_np = None
    coarse_gaussian_score_indices = None
    coarse_gaussian_score_indices_np = None
    coarse_gaussian_score_active_mask = None
    coarse_gaussian_window_positions = None
    coarse_gaussian_powerclass = None
    coarse_gaussian_projector_full = None
    coarse_gaussian_gemm_resource_estimate = None
    coarse_gaussian_gemm_projection_cache_plan = None
    coarse_gaussian_gemm_certificate_topology = None
    if coarse_gaussian_ffi_enabled:
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
        from recovar.em.dense_single_volume.helpers.projection import (
            relion_projector_half_to_texture_full,
        )
        from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
            _relion_cuda_corr_img_from_rfloat_ctf,
            _relion_cuda_fine_full_to_compact_lookup,
            _relion_cuda_pixel_correction_from_rfloat_ctf,
            _relion_cuda_powerclass_highres_xi2_half,
            _relion_exact_ctf_half_from_source_star,
            _relion_translation_angles_f32,
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
        coarse_gaussian_score_indices_np = np.asarray(
            square_score_indices_np,
            dtype=np.int32,
        )
        coarse_gaussian_score_indices = jnp.asarray(
            coarse_gaussian_score_indices_np,
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
        coarse_gaussian_full_to_compact_np = np.asarray(
            _relion_cuda_fine_full_to_compact_lookup(
                image_shape,
                score_size,
                square_score_indices_np,
            ),
            dtype=np.int32,
        )
        coarse_gaussian_full_to_compact = jnp.asarray(
            coarse_gaussian_full_to_compact_np,
            dtype=jnp.int32,
        )
        if coarse_gaussian_gemm_hybrid_requested:
            coarse_gaussian_gemm_certificate_topology = (
                plan_coarse_gemm_certificate_topology(
                    coarse_gaussian_full_to_compact_np,
                    compact_pixel_count=int(square_score_count),
                    translation_count=n_trans,
                )
            )
        coarse_gaussian_powerclass = _relion_cuda_powerclass_highres_xi2_half
        if coarse_gaussian_gemm_macro_enabled:
            coarse_gaussian_gemm_resource_estimate = _coarse_gaussian_gemm_resources(
                rotation_block_size=int(rotation_block_size),
                image_shape=image_shape,
                compact_pixel_count=int(square_score_count),
                budget_bytes=_coarse_gaussian_gemm_projected_transient_budget_bytes(),
            )
        if coarse_gaussian_gemm_projection_cache_requested:
            coarse_gaussian_gemm_projection_cache_plan = (
                _plan_coarse_gaussian_gemm_projection_cache(
                    n_rotations=n_rot,
                    compact_pixel_count=int(square_score_count),
                    image_shape=image_shape,
                    budget_bytes=(
                        _coarse_gaussian_gemm_projection_cache_budget_bytes()
                    ),
                )
            )
        if coarse_gaussian_gemm_hybrid_image_batch_size_request is not None:
            image_batch_size = _resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
                input_image_batch_size,
                requested_batch_size=(
                    coarse_gaussian_gemm_hybrid_image_batch_size_request
                ),
                n_images=n_images,
                certificate_chunk_rows=(
                    1
                    if coarse_gaussian_gemm_projection_cache_plan is None
                    else coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                ),
                n_translations=n_trans,
                hybrid_enabled=coarse_gaussian_gemm_hybrid_requested,
                compact_posterior_enabled=(
                    coarse_gaussian_gemm_compact_posterior_requested
                ),
            )
            if coarse_gaussian_gemm_projection_cache_plan is None:
                raise RuntimeError(
                    "compact-hybrid image-batch override is missing its "
                    "projection-cache plan",
                )
            logger.warning(
                "Opt-in shared compact-hybrid image batching enabled: "
                "input=%d requested=%d effective=%d images=%d; each streamed "
                "certificate tile retains at most %d candidate values",
                input_image_batch_size,
                coarse_gaussian_gemm_hybrid_image_batch_size_request,
                image_batch_size,
                n_images,
                image_batch_size
                * coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                * n_trans,
            )
        coarse_gaussian_projector_full_by_class = None
        coarse_gaussian_translation_angles = None
        if coarse_gaussian_score_backend in {
            _CoarseGaussianScoreBackend.FUSED,
            _CoarseGaussianScoreBackend.FUSED_CANONICAL,
            _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC,
            _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE,
            _CoarseGaussianScoreBackend.FUSED_MULTISTREAM,
        }:
            coarse_gaussian_projector_full_by_class = [
                relion_projector_half_to_texture_full(
                    relion_projector_half[class_index],
                )
                for class_index in range(n_classes)
            ]
            coarse_gaussian_translation_angles = jnp.asarray(
                _relion_translation_angles_f32(translations_source, image_shape),
                dtype=jnp.float32,
            )
            # One all-rotation launch preserves RELION's 128-orientation main
            # segment followed by its one-orientation tail.  It also avoids
            # projecting a memory-planner padding block (often 5000 rows).
            rotation_block_size = n_rot
        logger.warning(
            "RELION coarse Gaussian FFI enabled (%s): "
            "classes=%d current_size=%d square_pixels=%d translations=%d",
            (
                "guarded fresh InitialModel default"
                if relion_coarse_gaussian_default
                and _K1_COARSE_GAUSSIAN_FFI_ENV not in os.environ
                else "environment override"
            ),
            n_classes,
            score_size,
            square_score_count,
            n_trans,
        )
        if coarse_gaussian_sincosf_enabled:
            logger.warning(
                "RELION coarse CUDA sincosf translation enabled (%s): "
                "classes=%d current_size=%d square_pixels=%d translations=%d",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_GAUSSIAN_SINCOSF_ENV not in os.environ
                    else "environment override"
                ),
                n_classes,
                score_size,
                square_score_count,
                n_trans,
            )
        if exact_coarse_operands_enabled:
            logger.warning(
                "RELION exact coarse operands enabled (%s): per-image FFT, "
                "RFLOAT CTF division/square, and binary64-to-XFLOAT inverse noise",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_RELION_EXACT_COARSE_OPERANDS_ENV not in os.environ
                    else "environment override"
                ),
            )
            if exact_coarse_skip_generic_operands_enabled:
                logger.warning(
                    "Opt-in exact-coarse single-translation path enabled: "
                    "skipping generic square operands that the exact-source "
                    "assembly replaces before first use",
                )
            if exact_compact_preprocess_enabled:
                logger.warning(
                    "Opt-in exact+compact preprocessing enabled: skipping the "
                    "unused generic half-image FFT, CTF evaluation, and full "
                    "translated score image",
                )
        if coarse_gaussian_score_backend in {
            _CoarseGaussianScoreBackend.FUSED,
            _CoarseGaussianScoreBackend.FUSED_CANONICAL,
            _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC,
            _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE,
            _CoarseGaussianScoreBackend.FUSED_MULTISTREAM,
        }:
            logger.warning(
                "RELION fused coarse projector/diff2 enabled (%s): "
                "classes=%d rotations=%d translations=%d",
                (
                    "guarded fresh InitialModel default"
                    if relion_coarse_gaussian_default
                    and _K1_COARSE_FUSED_PROJECTOR_ENV not in os.environ
                    else "environment override"
                ),
                n_classes,
                n_rot,
                n_trans,
            )
            if coarse_canonical_reduction_enabled:
                logger.warning(
                    "RELION shared coarse canonical lane reduction enabled (%s)",
                    (
                        "guarded exact-path default"
                        if _RELION_COARSE_CANONICAL_REDUCTION_ENV not in os.environ
                        else "environment override"
                    ),
                )
            if coarse_native_atomic_reduction_enabled:
                logger.warning(
                    "Opt-in native RELION atomic coarse lane reduction enabled: "
                    "translations=%d",
                    n_trans,
                )
            if coarse_prehalf_weight_enabled:
                logger.warning(
                    "Opt-in RELION source-ordered coarse prehalf weight enabled: "
                    "translations=%d",
                    n_trans,
                )
            if coarse_single_lane_canonical_enabled:
                logger.warning(
                    "Opt-in RELION coarse single-lane canonical specialization "
                    "enabled: translations=%d",
                    n_trans,
                )
            if coarse_multistream_enabled:
                logger.warning(
                    "Opt-in shared K=1 coarse multistream dispatcher enabled: "
                    "workers=%d actual image rows only",
                    coarse_multistream_worker_count,
                )
        if coarse_gaussian_gemm_macro_enabled:
            logger.warning(
                "Opt-in shared coarse projection-once/GEMM macro enabled: "
                "classes=%d rotations=%d image_lanes=%d translations=%d",
                n_classes,
                n_rot,
                int(image_batch_size),
                n_trans,
            )
        if coarse_gaussian_gemm_hybrid_requested:
            logger.warning(
                "Opt-in certified K=1 coarse GEMM/source16 hybrid enabled: "
                "rotations=%d translations=%d selected_block_capacity=%d; "
                "all ineligible batches use full rectangular direct fallback",
                n_rot,
                n_trans,
                coarse_gaussian_gemm_hybrid_capacity,
            )
            if coarse_gaussian_gemm_compact_posterior_requested:
                logger.warning(
                    "Opt-in compact certified-hybrid posterior enabled: "
                    "selected scores remain in fixed-capacity source16 order; "
                    "the dense selected-score table remains the fail-closed oracle",
                )
        if coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.NATIVE_TEXTURE:
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
                    "guarded fresh InitialModel default"
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

    def _project_relion_compact_score_rows(
        class_index,
        rots_b,
        *,
        return_abs2: bool,
    ):
        """Project the exact compact rows shared by direct and GEMM scoring."""

        if not (
            use_relion_projector
            and coarse_texture_interp
            and projector_returns_compact
        ):
            raise RuntimeError(
                "compact RELION score-row projection requires the texture "
                "projector and an explicit score-row table",
            )
        return _compute_relion_projector_projections_block(
            relion_projector_half[class_index],
            rots_b,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=return_abs2,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(score_size),
            # Keep the already-host-resident table on the host so validation
            # cannot materialize its JAX mirror once per score block.
            pixel_indices=coarse_gaussian_score_indices_np,
            relion_texture_interp=True,
            # This is the mature rectangular/EM operand convention.  The
            # certificate and selected rescore must consume the same zeros in
            # current-image crop corners rather than silently rebuilding a
            # different projection table.
            mask_current_image_disk=True,
        )

    def _project_block(class_index, mean_for_proj, rots_b):
        if use_relion_projector:
            projector_kwargs = {}
            if current_size is not None:
                projector_kwargs["projector_output_size"] = int(current_size)
            if projector_returns_compact:
                projector_kwargs["pixel_indices"] = projector_compact_indices
            if coarse_texture_interp:
                if projector_returns_compact:
                    proj_half_b, proj_abs2_half_b = (
                        _project_relion_compact_score_rows(
                            class_index,
                            rots_b,
                            return_abs2=True,
                        )
                    )
                else:
                    proj_half_b, proj_abs2_half_b = (
                        _compute_relion_projector_projections_block(
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

    coarse_selector_execution = {
        "wrapper": None,
        "target": None,
        "fused_calls": 0,
        "actual_rows": 0,
        "multistream_calls": 0,
        "native_atomic_selected_calls": 0,
        "prehalf_selected_calls": 0,
    }


    def _project_coarse_gemm_rows(class_index, rots_b, *, return_abs2: bool):
        return _project_relion_compact_score_rows(
            class_index,
            rots_b,
            return_abs2=return_abs2,
        )

    def _project_coarse_gemm_block_once(class_index, mean_for_proj, rots_b):
        del mean_for_proj
        return _project_coarse_gemm_rows(
            class_index,
            rots_b,
            return_abs2=True,
        )

    coarse_gaussian_gemm_projection_cache = None
    if coarse_gaussian_gemm_projection_cache_plan is not None:

        def _project_coarse_gemm_cache_build_block(table_index, start, stop):
            projected_reference, projected_reference_abs2 = (
                _project_coarse_gemm_rows(
                    table_index,
                    rotations[start:stop],
                    return_abs2=False,
                )
            )
            if projected_reference_abs2 is not None:
                raise RuntimeError(
                    "coarse GEMM cache build unexpectedly materialized abs2",
                )
            return projected_reference

        coarse_gaussian_gemm_projection_cache = (
            _build_coarse_gaussian_gemm_projection_cache(
                coarse_gaussian_gemm_projection_cache_plan,
                _project_coarse_gemm_cache_build_block,
            )
        )
        logger.warning(
            "Opt-in call-scoped coarse GEMM C64 projection cache built: "
            "shape=%s chunks=%d conservative_peak_bytes=%d budget_bytes=%d",
            coarse_gaussian_gemm_projection_cache_plan.cache_shape,
            coarse_gaussian_gemm_projection_cache_plan.chunk_count_per_table,
            coarse_gaussian_gemm_projection_cache_plan.predicted_peak_bytes,
            coarse_gaussian_gemm_projection_cache_plan.budget_bytes,
        )

    coarse_gemm_diagnostic_positions = None
    coarse_gemm_direct_capture = {"scores": None}
    coarse_gemm_stream_capture_control = {"enabled": False}
    coarse_gaussian_gemm_hybrid_batch_result = None

    def _score_block(
        class_index,
        mean_for_proj,
        rots_b,
        shifted_data,
        batch_norm,
        ctf2_data,
        batch_size,
        *,
        rotation_start,
    ):
        if coarse_gaussian_gemm_hybrid_batch_result is not None:
            if int(class_index) != 0:
                raise RuntimeError("certified coarse GEMM hybrid is restricted to K=1")
            start = int(rotation_start)
            requested_rows = int(rots_b.shape[0])
            dense_scores = coarse_gaussian_gemm_hybrid_batch_result.scores
            if dense_scores is None:
                raise RuntimeError(
                    "compact hybrid scores must bypass dense rotation-block replay",
                )
            if (
                start < 0
                or start >= int(dense_scores.shape[1])
                or requested_rows <= 0
            ):
                raise IndexError("hybrid score block is outside the dense source table")
            stop = min(start + requested_rows, int(dense_scores.shape[1]))
            scores = dense_scores[:, start:stop, :]
            padding_rows = requested_rows - int(scores.shape[1])
            if padding_rows:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, padding_rows), (0, 0)),
                    constant_values=-jnp.inf,
                )
            return scores
        if coarse_gaussian_gemm_macro_enabled:
            coarse_gemm_direct_capture["scores"] = None
            project_block_once = _project_coarse_gemm_block_once
            if coarse_gaussian_gemm_projection_cache is not None:
                project_block_once = partial(
                    _project_coarse_gaussian_gemm_projection_cache_block_once,
                    coarse_gaussian_gemm_projection_cache,
                    rotation_start=int(rotation_start),
                )
            score_result = _score_relion_coarse_gaussian_gemm_macro(
                project_block_once,
                class_index,
                mean_for_proj,
                rots_b,
                jnp.asarray(
                    coarse_gaussian_shifted_corrected,
                    dtype=jnp.complex64,
                ),
                jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                actual_batch_size,
                image_shape=image_shape,
                volume_shape=volume_shape,
                return_projected=bool(
                    coarse_gemm_diagnostic_positions is not None
                    or coarse_gemm_stream_capture_control["enabled"]
                ),
            )
            if (
                coarse_gemm_diagnostic_positions is None
                and not coarse_gemm_stream_capture_control["enabled"]
            ):
                return score_result

            from recovar import cuda_backproject

            macro_scores, projected_reference, _projected_reference_abs2 = score_result
            active = jnp.arange(batch_size, dtype=jnp.int32) < jnp.asarray(
                actual_batch_size,
                dtype=jnp.int32,
            )
            direct_shifted = jnp.where(
                active[:, None, None],
                jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                jnp.zeros((), dtype=jnp.complex64),
            )
            direct_weight = jnp.where(
                active[:, None],
                jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32),
            )
            direct_initial = jnp.where(
                active,
                jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32),
            )
            direct_diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
                jnp.asarray(projected_reference, dtype=jnp.complex64),
                direct_shifted,
                direct_weight,
                direct_initial,
                coarse_gaussian_full_to_compact,
            )
            coarse_gemm_direct_capture["scores"] = jnp.where(
                active[:, None, None],
                -direct_diff2,
                jnp.zeros((), dtype=jnp.float32),
            )
            return macro_scores
        if coarse_gaussian_score_backend in {
            _CoarseGaussianScoreBackend.FUSED,
            _CoarseGaussianScoreBackend.FUSED_CANONICAL,
            _CoarseGaussianScoreBackend.FUSED_NATIVE_ATOMIC,
            _CoarseGaussianScoreBackend.FUSED_SINGLE_LANE,
            _CoarseGaussianScoreBackend.FUSED_MULTISTREAM,
        }:
            from recovar import cuda_backproject

            coarse_projector = (
                cuda_backproject.relion_coarse_diff2_projector_multistream_f32
                if coarse_multistream_enabled
                else cuda_backproject.relion_coarse_diff2_projector_f32
            )
            coarse_projector_kwargs = {}
            if coarse_multistream_enabled:
                coarse_projector_kwargs["actual_batch_size"] = jnp.asarray(
                    actual_batch_size,
                    dtype=jnp.int32,
                )
            selected_wrapper = getattr(coarse_projector, "__name__", None)
            selected_target = (
                cuda_backproject._TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32
                if coarse_multistream_enabled
                else cuda_backproject._TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32
            )
            previous_wrapper = coarse_selector_execution["wrapper"]
            previous_target = coarse_selector_execution["target"]
            if previous_wrapper is None:
                coarse_selector_execution["wrapper"] = selected_wrapper
                coarse_selector_execution["target"] = selected_target
            elif previous_wrapper != selected_wrapper or previous_target != selected_target:
                raise RuntimeError(
                    "coarse selector changed wrapper/target within one significance pass"
                )
            coarse_selector_execution["fused_calls"] += 1
            coarse_selector_execution["actual_rows"] += int(actual_batch_size)
            if coarse_multistream_enabled:
                coarse_selector_execution["multistream_calls"] += 1
            if coarse_native_atomic_reduction_enabled:
                coarse_selector_execution["native_atomic_selected_calls"] += 1
            if coarse_prehalf_weight_enabled:
                coarse_selector_execution["prehalf_selected_calls"] += 1
            coarse_projector_kwargs["prehalf_weight"] = coarse_prehalf_weight_enabled
            diff2 = coarse_projector(
                coarse_gaussian_projector_full_by_class[class_index],
                jnp.asarray(rots_b, dtype=jnp.float32),
                jnp.asarray(coarse_gaussian_unshifted_corrected, dtype=jnp.complex64),
                coarse_gaussian_translation_angles,
                jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                coarse_gaussian_full_to_compact,
                current_size=score_size,
                physical_image_size=int(image_shape[0]),
                model_max_r=int(relion_projector_r_max),
                canonical_reduction=coarse_canonical_reduction_enabled,
                single_lane_canonical=coarse_single_lane_canonical_enabled,
                **coarse_projector_kwargs,
            )
            return -diff2

        if coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.NATIVE_TEXTURE:
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
    relion_f32_sum_weight = (
        np.empty(n_images, dtype=np.float32)
        if relion_f32_coarse_support_enabled and collect_significance
        else None
    )
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
    coarse_gaussian_gemm_diagnostic_paths = []
    coarse_gaussian_gemm_diagnostic_found_targets = set()
    coarse_gaussian_gemm_diagnostic_target_counts = {}
    coarse_gaussian_gemm_stream_paths = []
    coarse_gaussian_gemm_stream_original_indices = []
    coarse_gaussian_gemm_hybrid_batch_count = 0
    coarse_gaussian_gemm_hybrid_selected_batch_count = 0
    coarse_gaussian_gemm_hybrid_static_dense_batch_count = 0
    coarse_gaussian_gemm_hybrid_fallback_batch_count = 0
    coarse_gaussian_gemm_hybrid_selected_image_count = 0
    coarse_gaussian_gemm_hybrid_static_dense_image_count = 0
    coarse_gaussian_gemm_hybrid_fallback_image_count = 0
    coarse_gaussian_gemm_hybrid_selected_block_count = 0
    coarse_gaussian_gemm_hybrid_max_blocks_per_image = 0
    coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates = 0
    coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates = 0
    coarse_gaussian_gemm_hybrid_fallback_reasons = {}
    coarse_gaussian_gemm_hybrid_score_representation_batch_counts = {}
    coarse_gaussian_gemm_hybrid_actual_image_batch_sizes = []
    coarse_gaussian_gemm_hybrid_physical_image_batch_sizes = []
    # Every batch in this significance call shares the exact projection
    # cache, topology, rotation/translation geometry, and selected-block
    # capacity.  Once one certified selection overflows, later batches can
    # safely skip that certificate and use the mature exact rectangular
    # scorer.  Keep this latch local so no posterior-width assumption crosses
    # an iteration, dataset, or independently constructed score call.
    coarse_gaussian_gemm_hybrid_overflow_latched = False
    coarse_gaussian_gemm_hybrid_overflow_latch_activation_count = 0
    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count = 0
    coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count = 0
    generic_coarse_operand_assembly_count = 0
    exact_coarse_operand_assembly_count = 0
    generic_score_preprocess_count = 0
    exact_source_preprocess_count = 0

    start_idx = 0
    image_indices = np.arange(n_images)
    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        actual_batch_size = len(indices)
        end_idx = start_idx + actual_batch_size
        coarse_gaussian_gemm_hybrid_batch_result = None
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
        if pad_final_image_batch and actual_batch_size < int(image_batch_size):
            (
                batch_data,
                ctf_params,
                integer_pre_shifts,
                batch_corr_np,
                batch_scale_np,
                relion_preprocess_kwargs,
            ) = _pad_significance_preprocess_inputs(
                batch_data,
                ctf_params,
                integer_pre_shifts,
                batch_corr_np,
                batch_scale_np,
                relion_preprocess_kwargs,
                target_size=int(image_batch_size),
            )
        batch_size = int(np.asarray(batch_data).shape[0])
        if coarse_gaussian_gemm_hybrid_requested:
            coarse_gaussian_gemm_hybrid_actual_image_batch_sizes.append(
                int(actual_batch_size),
            )
            coarse_gaussian_gemm_hybrid_physical_image_batch_sizes.append(
                int(batch_size),
            )
        local_indices_np = np.asarray(indices, dtype=np.int64)
        batch_original_indices_np = None
        if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
            batch_original_indices_np = _original_indices_for_local(
                experiment_dataset,
                local_indices_np,
            )
        coarse_gemm_diagnostic_positions = None
        coarse_gemm_diagnostic_original_indices = None
        if coarse_gaussian_gemm_diagnostic_targets is not None:
            original_indices_np = _original_indices_for_local(
                experiment_dataset,
                local_indices_np,
            )
            positions = np.flatnonzero(
                np.isin(
                    original_indices_np,
                    np.fromiter(
                        coarse_gaussian_gemm_diagnostic_targets,
                        dtype=np.int64,
                    ),
                )
            ).astype(np.int64)
            if positions.size:
                coarse_gemm_diagnostic_positions = positions
                coarse_gemm_diagnostic_original_indices = original_indices_np[positions]
        coarse_gemm_stream_capture_control["enabled"] = bool(
            coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        coarse_gemm_stream_state = (
            initialize_coarse_gemm_streaming_state(
                batch_size,
                coarse_gaussian_gemm_stream_topk,
                score_dtype=jnp.float32,
            )
            if coarse_gemm_stream_capture_control["enabled"]
            else None
        )
        coarse_gemm_pre_prior_stream_state = (
            initialize_coarse_gemm_streaming_state(
                batch_size,
                coarse_gaussian_gemm_stream_topk,
                score_dtype=jnp.float32,
            )
            if coarse_gemm_stream_capture_control["enabled"]
            else None
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
            batch_translation_log_prior_np = np.asarray(translation_log_prior[start_idx:end_idx])
            if batch_size > actual_batch_size:
                batch_translation_log_prior_np = _repeat_pad_batch_axis(
                    batch_translation_log_prior_np,
                    batch_size,
                )
            batch_translation_log_prior = jnp.asarray(batch_translation_log_prior_np)

        if exact_compact_preprocess_enabled:
            shifted_half = None
            batch_norm = None
            ctf2_over_nv_half = None
            coarse_gaussian_unshifted_score_weighted = None
        elif score_mode == "normalized_cc":
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
            if exact_coarse_assembly_profile_enabled:
                generic_score_preprocess_count += 1
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
        if image_corrections is not None and not exact_compact_preprocess_enabled:
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
        if scale_corrections is not None and not exact_compact_preprocess_enabled:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if score_mode == "normalized_cc":
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]
        if (
            image_pre_shifts is not None
            and not real_space_pre_shift_applied
            and not exact_compact_preprocess_enabled
        ):
            batch_shifts_np = np.asarray(image_pre_shifts)[np.asarray(indices)]
            if batch_size > actual_batch_size:
                batch_shifts_np = _repeat_pad_batch_axis(batch_shifts_np, batch_size)
            batch_shifts = jnp.asarray(batch_shifts_np)
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
        elif not exact_compact_preprocess_enabled:
            score_weight_half = ctf2_over_nv_half
        else:
            score_weight_half = None
        if (
            half_spectrum_scoring
            and score_mode != "normalized_cc"
            and not exact_compact_preprocess_enabled
        ):
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
        if exact_compact_preprocess_enabled:
            shifted_data = None
            ctf2_data = None
        elif use_window:
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
        if exact_compact_preprocess_enabled:
            pass
        elif use_float64_scoring:
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
                if exact_coarse_assembly_profile_enabled:
                    exact_source_preprocess_count += 1
                processed_direct = _process_relion_exact_coarse_half_image(
                    experiment_dataset,
                    batch_data,
                    score_with_masked_images,
                    relion_preprocess_kwargs=relion_preprocess_kwargs,
                )
            else:
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
            if not exact_coarse_skip_generic_operands_enabled:
                if exact_coarse_assembly_profile_enabled:
                    generic_coarse_operand_assembly_count += 1
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
                if exact_coarse_assembly_profile_enabled:
                    exact_coarse_operand_assembly_count += 1
                exact_operands = _assemble_relion_exact_coarse_gaussian_operands(
                    experiment_dataset,
                    processed_direct,
                    indices,
                    batch_scale_np=batch_scale_np,
                    actual_batch_size=actual_batch_size,
                    batch_size=batch_size,
                    score_indices=coarse_gaussian_score_indices,
                    score_active_mask=coarse_gaussian_score_active_mask,
                    translations_source=translations_source,
                    image_shape=image_shape,
                    noise_variance_half=noise_variance_half,
                    scale_corrections_enabled=scale_corrections is not None,
                    half_weights=half_weights,
                    powerclass=coarse_gaussian_powerclass,
                    current_size=current_size,
                )
                coarse_gaussian_shifted_corrected = exact_operands.shifted_corrected
                coarse_gaussian_pixel_weight = exact_operands.pixel_weight
                coarse_gaussian_unshifted_corrected = exact_operands.unshifted_corrected
                coarse_gaussian_initial_diff2 = exact_operands.initial_diff2
                coarse_gaussian_translation_angles = exact_operands.translation_angles
            else:
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
        coarse_gemm_macro_pre_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_direct_pre_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_macro_with_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        coarse_gemm_direct_with_prior_blocks = (
            [[] for _ in range(n_classes)]
            if coarse_gemm_diagnostic_positions is not None
            else None
        )
        if passive_score_dump:
            # Do not materialize score blocks while production support is
            # being computed. Near an atomic cutoff that observation can
            # perturb the execution under diagnosis. Retain device buffers
            # and write them only after cached-score support selection.
            dump_target_pre_prior_blocks_per_class = None
            dump_target_with_prior_blocks_per_class = None

        if coarse_gaussian_gemm_hybrid_requested:
            if coarse_gaussian_gemm_compact_posterior_requested and debug_dump_enabled:
                raise ValueError(
                    f"{_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1 does not "
                    "support raw significance score dumps",
                )
            if dump_target_local_positions is not None:
                raise ValueError(
                    f"{_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1 does not support "
                    "raw significance score dumps",
                )
            if (
                coarse_gaussian_gemm_projection_cache is None
                or coarse_gaussian_gemm_projection_cache_plan is None
                or coarse_gaussian_gemm_certificate_topology is None
            ):
                raise RuntimeError(
                    "certified coarse GEMM hybrid is missing its projection "
                    "cache, cache plan, or sealed topology",
                )
            force_static_dense_after_overflow = bool(
                coarse_gaussian_gemm_hybrid_overflow_latched
            )
            coarse_gaussian_gemm_hybrid_batch_result = (
                _compute_coarse_gaussian_gemm_hybrid_batch(
                    coarse_gaussian_gemm_projection_cache,
                    jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                    jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                    jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                    topology=coarse_gaussian_gemm_certificate_topology,
                    actual_image_count=actual_batch_size,
                    class_log_prior=class_log_priors_np[0],
                    rotation_log_prior=(
                        None
                        if rotation_log_prior_padded is None
                        else rotation_log_prior_padded[0, :n_rot]
                    ),
                    translation_log_prior=batch_translation_log_prior,
                    certificate_chunk_rows=(
                        coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                    ),
                    block_capacity=coarse_gaussian_gemm_hybrid_capacity,
                    compact_posterior=(
                        coarse_gaussian_gemm_compact_posterior_requested
                    ),
                    force_static_dense_after_overflow=(
                        force_static_dense_after_overflow
                    ),
                )
            )
            coarse_gaussian_gemm_hybrid_batch_count += 1
            if force_static_dense_after_overflow:
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count += 1
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count += (
                    actual_batch_size
                )
            score_representation = str(
                coarse_gaussian_gemm_hybrid_batch_result.score_representation,
            )
            coarse_gaussian_gemm_hybrid_score_representation_batch_counts[
                score_representation
            ] = (
                coarse_gaussian_gemm_hybrid_score_representation_batch_counts.get(
                    score_representation,
                    0,
                )
                + 1
            )
            if coarse_gaussian_gemm_hybrid_batch_result.used_selected_rescore:
                coarse_gaussian_gemm_hybrid_selected_batch_count += 1
                coarse_gaussian_gemm_hybrid_selected_image_count += actual_batch_size
                selected_counts = np.asarray(
                    coarse_gaussian_gemm_hybrid_batch_result.selection.block_count,
                    dtype=np.int32,
                )[:actual_batch_size]
                coarse_gaussian_gemm_hybrid_selected_block_count += int(
                    np.sum(selected_counts, dtype=np.int64),
                )
                coarse_gaussian_gemm_hybrid_max_blocks_per_image = max(
                    coarse_gaussian_gemm_hybrid_max_blocks_per_image,
                    int(np.max(selected_counts)),
                )
                coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates += (
                    batch_size
                    * coarse_gaussian_gemm_hybrid_capacity
                    * SOURCE_ROTATION_BLOCK_SIZE
                    * n_trans
                )
                coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates += (
                    batch_size * n_rot * n_trans
                )
            elif score_representation == "dense_full_direct_static_capacity":
                coarse_gaussian_gemm_hybrid_static_dense_batch_count += 1
                coarse_gaussian_gemm_hybrid_static_dense_image_count += (
                    actual_batch_size
                )
            else:
                coarse_gaussian_gemm_hybrid_fallback_batch_count += 1
                coarse_gaussian_gemm_hybrid_fallback_image_count += actual_batch_size
                fallback_reason = str(
                    coarse_gaussian_gemm_hybrid_batch_result.fallback_reason,
                )
                coarse_gaussian_gemm_hybrid_fallback_reasons[fallback_reason] = (
                    coarse_gaussian_gemm_hybrid_fallback_reasons.get(
                        fallback_reason,
                        0,
                    )
                    + 1
                )
                if (
                    fallback_reason == "block_capacity_overflow"
                    and not coarse_gaussian_gemm_hybrid_overflow_latched
                ):
                    coarse_gaussian_gemm_hybrid_overflow_latched = True
                    coarse_gaussian_gemm_hybrid_overflow_latch_activation_count += 1

        compact_hybrid_scores = (
            None
            if coarse_gaussian_gemm_hybrid_batch_result is None
            else coarse_gaussian_gemm_hybrid_batch_result.compact_scores
        )
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
        cache_score_blocks = compact_hybrid_scores is None and collect_significance and (
            coarse_gemm_diagnostic_positions is not None
            or _significance_score_cache_enabled(
                batch_size,
                n_classes,
                n_rot_padded,
                n_trans,
                use_float64_scoring=use_float64_scoring,
            )
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
        if relion_f32_coarse_support_enabled and use_fused_pass1:
            raise RuntimeError(
                "RELION float32 coarse support requires access to pre-prior "
                "scores for min_diff2 offset reconstruction"
            )

        if not relion_f32_coarse_support_enabled:
            relion_raw_score_max = None
        elif coarse_gaussian_gemm_hybrid_batch_result is not None:
            relion_raw_score_max = (
                coarse_gaussian_gemm_hybrid_batch_result.raw_score_max
            )
        else:
            relion_raw_score_max = jnp.full(
                batch_size,
                -jnp.inf,
                dtype=jnp.float32,
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
            score_block_count = n_blocks if compact_hybrid_scores is None else 0
            for block_index in range(score_block_count):
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
                        rotation_start=r0,
                    )
                    direct_scores_for_diagnostic = coarse_gemm_direct_capture["scores"]
                    if (
                        coarse_gemm_diagnostic_positions is not None
                        and direct_scores_for_diagnostic is None
                    ):
                        raise RuntimeError(
                            "coarse GEMM diagnostic did not capture the paired "
                            "direct-square score block",
                        )
                    if r1 > n_rot:
                        valid = n_rot - r0
                        valid_rotation = (
                            jnp.arange(rotation_block_size)[None, :, None] < valid
                        )
                        scores = jnp.where(valid_rotation, scores, -jnp.inf)
                        if direct_scores_for_diagnostic is not None:
                            direct_scores_for_diagnostic = jnp.where(
                                valid_rotation,
                                direct_scores_for_diagnostic,
                                -jnp.inf,
                            )
                    if passive_raw_score_blocks_per_class is not None:
                        passive_raw_score_blocks_per_class[class_index].append(scores)
                    if (
                        relion_raw_score_max is not None
                        and coarse_gaussian_gemm_hybrid_batch_result is None
                    ):
                        relion_raw_score_max = jnp.maximum(
                            relion_raw_score_max,
                            jnp.max(scores.reshape(batch_size, -1), axis=1),
                        )
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
                    if coarse_gemm_macro_pre_prior_blocks is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        target_rows = coarse_gemm_diagnostic_positions
                        coarse_gemm_macro_pre_prior_blocks[class_index].append(
                            scores[target_rows, :actual_rot, :]
                        )
                        coarse_gemm_direct_pre_prior_blocks[class_index].append(
                            direct_scores_for_diagnostic[
                                target_rows,
                                :actual_rot,
                                :,
                            ]
                        )
                    if coarse_gemm_pre_prior_stream_state is not None:
                        if direct_scores_for_diagnostic is None:
                            raise RuntimeError(
                                "coarse GEMM pre-prior streaming diagnostic "
                                "requires the paired direct-square score block",
                            )
                        coarse_gemm_pre_prior_stream_state = (
                            update_coarse_gemm_streaming_state(
                                coarse_gemm_pre_prior_stream_state,
                                direct_scores_for_diagnostic,
                                scores,
                                candidate_offset=(
                                    class_index * n_rot * n_trans
                                    + r0 * n_trans
                                ),
                                actual_image_count=actual_batch_size,
                            )
                        )
                    if not (
                        coarse_gaussian_gemm_hybrid_batch_result is not None
                        and coarse_gaussian_gemm_hybrid_batch_result.scores_include_priors
                    ):
                        scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    if direct_scores_for_diagnostic is not None:
                        direct_scores_for_diagnostic = _add_priors(
                            direct_scores_for_diagnostic,
                            class_index,
                            r0,
                            r1,
                            batch_translation_log_prior,
                        )
                        if coarse_gemm_stream_state is not None:
                            coarse_gemm_stream_state = update_coarse_gemm_streaming_state(
                                coarse_gemm_stream_state,
                                direct_scores_for_diagnostic,
                                scores,
                                candidate_offset=(
                                    class_index * n_rot * n_trans + r0 * n_trans
                                ),
                                actual_image_count=actual_batch_size,
                            )
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        target_rows = coarse_gemm_diagnostic_positions
                        if target_rows is not None:
                            coarse_gemm_macro_with_prior_blocks[class_index].append(
                                scores[target_rows, :actual_rot, :]
                            )
                            coarse_gemm_direct_with_prior_blocks[class_index].append(
                                direct_scores_for_diagnostic[
                                    target_rows,
                                    :actual_rot,
                                    :,
                                ]
                            )
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

        if compact_hybrid_scores is not None:
            # Active compact candidates are ordered by ascending source16
            # block, rotation, then translation.  One device reduction avoids
            # replaying every empty global rotation block; omitted candidates
            # are certified at least 138 score units below the maximum and
            # therefore contribute exact zero to RELION's float32 posterior.
            global_max, global_sum = _update_logsumexp(
                jnp.full(batch_size, -jnp.inf),
                jnp.zeros(batch_size, dtype=jnp.float64),
                compact_hybrid_scores.posterior_scores_flat,
            )
            class_max_values = [global_max]
            class_sum_values = [global_sum]
            best_score_batch = compact_hybrid_scores.best_score
            best_argmax_batch = compact_hybrid_scores.best_pose
            best_class_batch = jnp.zeros(batch_size, dtype=jnp.int32)

        # The second significance pass may recompute score blocks when the
        # production cache is disabled.  The all-particle diagnostic is a
        # first-pass online reduction; do not execute or count direct scores a
        # second time merely because support probabilities need replay.
        coarse_gemm_stream_capture_control["enabled"] = False

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
        compact_support_pose_ids = None
        if collect_significance:
            if compact_hybrid_scores is not None:
                batch_values = compact_hybrid_scores.posterior_scores_flat
            else:
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
                                rotation_start=r0,
                            )
                            if r1 > n_rot:
                                valid = n_rot - r0
                                scores = jnp.where(
                                    jnp.arange(rotation_block_size)[None, :, None]
                                    < valid,
                                    scores,
                                    -jnp.inf,
                                )
                            if not (
                                coarse_gaussian_gemm_hybrid_batch_result is not None
                                and coarse_gaussian_gemm_hybrid_batch_result.scores_include_priors
                            ):
                                scores = _add_priors(
                                    scores,
                                    class_index,
                                    r0,
                                    r1,
                                    batch_translation_log_prior,
                                )
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
                    class_weight_mats.append(
                        jnp.concatenate(class_weight_blocks, axis=1),
                    )

                batch_values = jnp.concatenate(class_weight_mats, axis=1)
            if relion_f32_coarse_support_enabled:
                from recovar.em.dense_single_volume.helpers.oversampling import (
                    relion_cuda_f32_coarse_posterior,
                )

                posterior_kwargs = {}
                if compact_hybrid_scores is not None:
                    # Stage 1's positive-only primitive is the exact
                    # correctness oracle, but its current sequential vmap
                    # performs one device-to-host count synchronization per
                    # image.  The intended runtime path instead scans this
                    # much smaller fixed-capacity table in one shape-stable
                    # computation; positive-count clamping below the CUB call
                    # still excludes exact-zero candidates from support.
                    posterior_kwargs["filter_positive_before_sort"] = False
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
                    tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    min_diff2_offsets=-relion_raw_score_max,
                    **posterior_kwargs,
                )
                relion_f32_sum_weight[start_idx:end_idx] = np.asarray(
                    _batch_sum_weight[:actual_batch_size],
                    dtype=np.float32,
                )
                if compact_hybrid_scores is None:
                    batch_sig_rot_mask = jnp.any(
                        batch_sig_mask.reshape(
                            batch_size,
                            n_classes * n_rot,
                            n_trans,
                        ),
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
            batch_sig_mask_np = np.array(batch_sig_mask, dtype=bool, copy=True)
            if compact_hybrid_scores is None:
                sig_rot_any |= np.asarray(
                    jnp.any(batch_sig_rot_mask[:actual_batch_size], axis=0),
                    dtype=bool,
                ).reshape(n_classes, n_rot)
            else:
                compact_support_pose_ids = (
                    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
                        compact_hybrid_scores,
                        batch_sig_mask_np,
                    )
                )
                for selected_pose_ids in compact_support_pose_ids[
                    :actual_batch_size
                ]:
                    if selected_pose_ids.size:
                        sig_rot_any[0, selected_pose_ids // n_trans] = True
            n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig[:actual_batch_size], dtype=np.int32)
            cutoff_count_all[start_idx:end_idx] = np.asarray(
                batch_cutoff_count[:actual_batch_size],
                dtype=np.int32,
            )
        else:
            batch_sig_mask_np = None
            n_sig_all[start_idx:end_idx] = 0
            cutoff_count_all[start_idx:end_idx] = 0

        if coarse_gemm_diagnostic_positions is not None:
            if batch_sig_mask_np is None or coarse_gaussian_gemm_resource_estimate is None:
                raise RuntimeError(
                    "coarse GEMM paired diagnostic requires production support "
                    "and a sealed resource estimate",
                )

            def _stack_coarse_gemm_diagnostic_blocks(blocks_by_class):
                return jnp.stack(
                    [jnp.concatenate(blocks, axis=1) for blocks in blocks_by_class],
                    axis=1,
                )

            direct_pre_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_direct_pre_prior_blocks,
            )
            macro_pre_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_macro_pre_prior_blocks,
            )
            direct_with_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_direct_with_prior_blocks,
            )
            macro_with_prior = _stack_coarse_gemm_diagnostic_blocks(
                coarse_gemm_macro_with_prior_blocks,
            )
            direct_values = direct_with_prior.reshape(
                direct_with_prior.shape[0],
                -1,
            )
            if relion_f32_coarse_support_enabled:
                from recovar.em.dense_single_volume.helpers.oversampling import (
                    relion_cuda_f32_coarse_posterior,
                )

                direct_raw_max = jnp.max(
                    direct_pre_prior.reshape(direct_pre_prior.shape[0], -1),
                    axis=1,
                )
                _, direct_sig_mask, *_ = relion_cuda_f32_coarse_posterior(
                    direct_values,
                    adaptive_fraction=float(adaptive_fraction),
                    max_significants=max_significants,
                    tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    min_diff2_offsets=-direct_raw_max,
                )
            else:
                direct_max = jnp.max(direct_values, axis=1, keepdims=True)
                direct_exp = jnp.exp(direct_values - direct_max)
                direct_weights = direct_exp / jnp.sum(
                    direct_exp,
                    axis=1,
                    keepdims=True,
                )
                direct_sig_mask, *_ = _find_sig(
                    direct_weights,
                    n_classes * n_rot,
                    n_trans,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    return_cutoff_count=True,
                )
            macro_support = batch_sig_mask_np[coarse_gemm_diagnostic_positions]
            iteration_label = -1 if debug_iteration is None else int(debug_iteration)
            size_label = -1 if current_size is None else int(current_size)
            diagnostic_path = os.path.join(
                coarse_gaussian_gemm_diagnostic_dir,
                "coarse_gemm_ab_"
                f"{coarse_gaussian_gemm_diagnostic_scope.run_id}_"
                f"{coarse_gaussian_gemm_diagnostic_scope.call_id}_"
                f"it{iteration_label:04d}_cs{size_label:04d}_"
                f"batch{start_idx:08d}_{end_idx:08d}.npz",
            )
            local_indices_np = np.asarray(indices, dtype=np.int64)
            _write_coarse_gaussian_gemm_diagnostic(
                diagnostic_path,
                direct_scores_pre_prior=np.asarray(direct_pre_prior),
                macro_scores_pre_prior=np.asarray(macro_pre_prior),
                direct_scores_with_prior=np.asarray(direct_with_prior),
                macro_scores_with_prior=np.asarray(macro_with_prior),
                direct_support=np.asarray(direct_sig_mask, dtype=bool),
                macro_support=macro_support,
                original_indices=coarse_gemm_diagnostic_original_indices,
                local_indices=local_indices_np[coarse_gemm_diagnostic_positions],
                actual_batch_size=actual_batch_size,
                padded_batch_size=batch_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                resource_estimate=coarse_gaussian_gemm_resource_estimate,
                diagnostic_scope=coarse_gaussian_gemm_diagnostic_scope,
                diagnostic_selection_policy=(
                    coarse_gaussian_gemm_diagnostic_selection_policy
                ),
                debug_iteration=debug_iteration,
                current_size=current_size,
            )
            coarse_gaussian_gemm_diagnostic_paths.append(diagnostic_path)
            for value in coarse_gemm_diagnostic_original_indices:
                target = int(value)
                coarse_gaussian_gemm_diagnostic_found_targets.add(target)
                coarse_gaussian_gemm_diagnostic_target_counts[target] = (
                    coarse_gaussian_gemm_diagnostic_target_counts.get(target, 0) + 1
                )

        if (coarse_gemm_stream_state is None) != (
            coarse_gemm_pre_prior_stream_state is None
        ):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic requires paired pre-prior "
                "and posterior states",
            )
        if coarse_gemm_stream_state is not None:
            if (
                batch_sig_mask_np is None
                or batch_original_indices_np is None
                or coarse_gaussian_gemm_diagnostic_scope is None
            ):
                raise RuntimeError(
                    "coarse GEMM streaming diagnostic requires production support, "
                    "particle identity, and a resolved call scope",
                )
            iteration_label = -1 if debug_iteration is None else int(debug_iteration)
            size_label = -1 if current_size is None else int(current_size)
            stream_path = os.path.join(
                coarse_gaussian_gemm_stream_diagnostic_dir,
                "coarse_gemm_rescore_"
                f"{coarse_gaussian_gemm_diagnostic_scope.run_id}_"
                f"{coarse_gaussian_gemm_diagnostic_scope.call_id}_"
                f"it{iteration_label:04d}_cs{size_label:04d}_"
                f"batch{start_idx:08d}_{end_idx:08d}.npz",
            )
            write_coarse_gemm_streaming_summary(
                stream_path,
                coarse_gemm_stream_state,
                coarse_gemm_pre_prior_stream_state,
                batch_sig_mask_np,
                original_indices=batch_original_indices_np,
                local_indices=local_indices_np,
                actual_image_count=actual_batch_size,
                padded_image_count=batch_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                diagnostic_run_id=coarse_gaussian_gemm_diagnostic_scope.run_id,
                diagnostic_call_id=coarse_gaussian_gemm_diagnostic_scope.call_id,
                debug_iteration=debug_iteration,
                current_size=current_size,
                n_rotations=n_rot,
                n_translations=n_trans,
            )
            coarse_gaussian_gemm_stream_paths.append(stream_path)
            coarse_gaussian_gemm_stream_original_indices.extend(
                int(value) for value in batch_original_indices_np
            )

        hard_assignment[start_idx:end_idx] = np.asarray(
            best_argmax_batch[:actual_batch_size],
            dtype=np.int32,
        )
        class_assignment[start_idx:end_idx] = np.asarray(
            best_class_batch[:actual_batch_size],
            dtype=np.int32,
        )

        log_score_offset = (
            np.zeros(batch_size, dtype=np.float64)
            if coarse_gaussian_ffi_enabled
            else -0.5
            * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        )
        global_log_z_np = np.asarray(global_log_z, dtype=np.float64)
        best_score_np = np.asarray(best_score_batch, dtype=np.float64)
        output_slice = slice(0, actual_batch_size)
        normalization_log_z[start_idx:end_idx] = global_log_z_np[output_slice]
        normalization_log_evidence[start_idx:end_idx] = (
            global_log_z_np[output_slice] + log_score_offset[output_slice]
        )
        log_evidence[start_idx:end_idx] = normalization_log_evidence[start_idx:end_idx].astype(np.float32)
        best_log_score[start_idx:end_idx] = (
            best_score_np[output_slice] + log_score_offset[output_slice]
        ).astype(np.float32)
        if relion_f32_coarse_support_enabled and collect_significance:
            max_posterior[start_idx:end_idx] = np.asarray(
                jnp.max(batch_weights[:actual_batch_size], axis=1),
                dtype=np.float32,
            )
        else:
            max_posterior[start_idx:end_idx] = np.exp(
                best_score_np[output_slice] - global_log_z_np[output_slice]
            ).astype(np.float32)
        for class_index, class_log_z in enumerate(class_log_z_values):
            class_log_evidence[class_index, start_idx:end_idx] = (
                np.asarray(class_log_z, dtype=np.float64)[output_slice]
                + log_score_offset[output_slice]
            )
        if return_class_best:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_best_scores[class_index],
                    log_score_offset,
                )
                class_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free[output_slice]
                class_best_log_score[class_index, start_idx:end_idx] = absolute[output_slice]
                class_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_best_argmaxes[class_index][output_slice],
                    dtype=np.int32,
                )
        if return_class_second:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    class_second_best_scores[class_index],
                    log_score_offset,
                )
                class_second_best_offset_free_log_score[class_index, start_idx:end_idx] = offset_free[output_slice]
                class_second_best_log_score[class_index, start_idx:end_idx] = absolute[output_slice]
                class_second_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_second_best_argmaxes[class_index][output_slice],
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
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=projection_padding_factor,
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
            if compact_hybrid_scores is not None:
                if compact_support_pose_ids is None:
                    raise RuntimeError("compact hybrid support mapping was not produced")
                for local_idx, global_idx in enumerate(indices):
                    significant_sample_indices[0][int(global_idx)] = (
                        compact_support_pose_ids[local_idx].copy()
                    )
            else:
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

    coarse_selector_audit = _validate_coarse_selector_audit(
        {
            "score_mode": score_mode,
            "translation_count": int(n_trans),
            "requested_fused": bool(coarse_fused_projector_requested),
            "effective_fused": bool(coarse_fused_projector_enabled),
            "requested_workers": int(coarse_multistream_worker_count),
            "effective_workers": (
                int(coarse_multistream_worker_count)
                if coarse_fused_projector_enabled and coarse_multistream_enabled
                else 0
            ),
            "requested_atomic": bool(coarse_native_atomic_reduction_requested),
            "effective_atomic": bool(coarse_native_atomic_reduction_enabled),
            "requested_prehalf": bool(coarse_prehalf_weight_requested),
            "effective_prehalf": bool(coarse_prehalf_weight_enabled),
            "wrapper": coarse_selector_execution["wrapper"],
            "target": coarse_selector_execution["target"],
            "counts": {
                "fused_calls": int(coarse_selector_execution["fused_calls"]),
                "actual_rows": int(coarse_selector_execution["actual_rows"]),
                "multistream_calls": int(
                    coarse_selector_execution["multistream_calls"]
                ),
                "native_atomic_selected_calls": int(
                    coarse_selector_execution["native_atomic_selected_calls"]
                ),
                "prehalf_selected_calls": int(
                    coarse_selector_execution["prehalf_selected_calls"]
                ),
            },
        }
    )
    coarse_gaussian_gemm_scope_manifest_path = None
    coarse_gaussian_gemm_aggregate_manifest_path = None
    coarse_gaussian_gemm_stream_scope_manifest_path = None
    coarse_gaussian_gemm_stream_aggregate_manifest_path = None
    if coarse_gaussian_gemm_diagnostic_targets is not None:
        missing_targets = (
            coarse_gaussian_gemm_diagnostic_targets
            - coarse_gaussian_gemm_diagnostic_found_targets
        )
        if missing_targets:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} contains "
                "targets that were not captured in this diagnostic call "
                f"scope: {sorted(missing_targets)}",
            )
        (
            coarse_gaussian_gemm_scope_manifest_path,
            coarse_gaussian_gemm_aggregate_manifest_path,
        ) = _seal_coarse_gaussian_gemm_diagnostic_scope(
            coarse_gaussian_gemm_diagnostic_dir,
            scope=coarse_gaussian_gemm_diagnostic_scope,
            selection_policy=coarse_gaussian_gemm_diagnostic_selection_policy,
            requested_targets=coarse_gaussian_gemm_requested_targets,
            targets_in_scope=coarse_gaussian_gemm_diagnostic_targets,
            captured_target_counts=coarse_gaussian_gemm_diagnostic_target_counts,
            artifact_paths=coarse_gaussian_gemm_diagnostic_paths,
        )
    if coarse_gaussian_gemm_stream_diagnostic_dir is not None:
        if len(coarse_gaussian_gemm_stream_original_indices) != n_images:
            raise RuntimeError(
                "coarse GEMM streaming diagnostic did not capture every particle "
                f"in its call scope: {len(coarse_gaussian_gemm_stream_original_indices)} "
                f"!= {n_images}",
            )
        (
            coarse_gaussian_gemm_stream_scope_manifest_path,
            coarse_gaussian_gemm_stream_aggregate_manifest_path,
        ) = _seal_coarse_gaussian_gemm_streaming_scope(
            coarse_gaussian_gemm_stream_diagnostic_dir,
            scope=coarse_gaussian_gemm_diagnostic_scope,
            retained_topk=coarse_gaussian_gemm_stream_topk,
            artifact_paths=coarse_gaussian_gemm_stream_paths,
            original_indices=coarse_gaussian_gemm_stream_original_indices,
        )
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
        "coarse_selector_audit": coarse_selector_audit,
    }
    if exact_coarse_assembly_profile_enabled:
        full_stats["exact_coarse_operand_assembly"] = {
            "skip_generic_default_enabled": False,
            "skip_generic_requested": bool(
                exact_coarse_skip_generic_operands_requested,
            ),
            "skip_generic_effective": bool(
                exact_coarse_skip_generic_operands_enabled,
            ),
            "exact_coarse_operands_effective": bool(exact_coarse_operands_enabled),
            "exact_compact_preprocess_default_enabled": False,
            "exact_compact_preprocess_requested": bool(
                exact_compact_preprocess_requested,
            ),
            "exact_compact_preprocess_effective": bool(
                exact_compact_preprocess_enabled,
            ),
            "generic_score_preprocess_count": int(generic_score_preprocess_count),
            "exact_source_preprocess_count": int(exact_source_preprocess_count),
            "generic_ctf_evaluation_count": int(generic_score_preprocess_count),
            "generic_full_translation_count": int(generic_score_preprocess_count),
            "generic_assembly_count": int(generic_coarse_operand_assembly_count),
            "exact_assembly_count": int(exact_coarse_operand_assembly_count),
            "translate_score_call_site_count": int(
                generic_coarse_operand_assembly_count
                * int(coarse_gaussian_sincosf_enabled)
                + exact_coarse_operand_assembly_count
            ),
            "translate_score_call_count": int(
                generic_coarse_operand_assembly_count
                * int(coarse_gaussian_sincosf_enabled)
                + exact_coarse_operand_assembly_count
            ),
            "downstream_operand_source": (
                "exact_source_star"
                if exact_coarse_operands_enabled
                else "generic"
            ),
            "diagnostic_operand_source": (
                "exact_source_star"
                if exact_coarse_operands_enabled
                else "generic"
            ),
            "raw_score_capture_changed": False,
            "generic_fallback_policy": (
                "exact_full_direct_scores_available"
                if exact_compact_preprocess_enabled
                else "available"
            ),
            "skipped_generic_outputs": (
                [
                    "coarse_gaussian_shifted_corrected",
                    "coarse_gaussian_pixel_weight",
                    "coarse_gaussian_unshifted_corrected",
                ]
                if exact_coarse_skip_generic_operands_enabled
                else []
            ),
        }
    if coarse_gaussian_gemm_resource_estimate is not None:
        full_stats["coarse_gaussian_gemm_resources"] = {
            field: int(value)
            for field, value in coarse_gaussian_gemm_resource_estimate._asdict().items()
        }
        paired_capture_requested = bool(
            coarse_gaussian_gemm_diagnostic_dir is not None
            or coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        paired_capture_active = bool(
            coarse_gaussian_gemm_diagnostic_targets
            or coarse_gaussian_gemm_stream_diagnostic_dir is not None
        )
        full_stats["coarse_gaussian_gemm_qualification"] = {
            "paired_capture_requested": paired_capture_requested,
            "paired_capture_active": paired_capture_active,
            "clean_timing_eligible": not paired_capture_requested,
            "numerical_qualification_status": "NO_GO_UNQUALIFIED",
            "requires_bitwise_score_identity": False,
            "requires_exact_discrete_identity": True,
            "timing_policy": (
                "paired capture executes both scorers; use a separate "
                "diagnostic-off timing arm"
            ),
            "numerical_policy": (
                "mathematically equivalent score noise may pass only when "
                "repeat-stable, unbiased/non-directional, bounded/non-growing, "
                "discrete-identical, basin/quality-neutral, and materially faster; "
                "scale amplification or negative implied diff2 is NO-GO"
            ),
        }
    if coarse_gaussian_gemm_projection_cache_plan is not None:
        full_stats["coarse_gaussian_gemm_projection_cache"] = (
            _coarse_gaussian_gemm_projection_cache_stats(
                coarse_gaussian_gemm_projection_cache_plan,
                enabled=coarse_gaussian_gemm_projection_cache is not None,
            )
        )
    if coarse_gaussian_gemm_hybrid_requested:
        selected_candidate_count = (
            coarse_gaussian_gemm_hybrid_selected_block_count
            * SOURCE_ROTATION_BLOCK_SIZE
            * n_trans
        )
        full_selected_image_candidate_count = (
            coarse_gaussian_gemm_hybrid_selected_image_count * n_rot * n_trans
        )
        compact_table_capacity = (
            coarse_gaussian_gemm_hybrid_selected_table_capacity_candidates
        )
        dense_table_capacity = (
            coarse_gaussian_gemm_hybrid_dense_table_capacity_candidates
        )
        full_stats["coarse_gaussian_gemm_hybrid"] = {
            "enabled": True,
            "default_enabled": False,
            "compact_posterior_enabled": bool(
                coarse_gaussian_gemm_compact_posterior_requested,
            ),
            "compact_posterior_default_enabled": False,
            "selected_score_layout": (
                "fixed_capacity_source16"
                if coarse_gaussian_gemm_compact_posterior_requested
                else "dense_global"
            ),
            "positive_only_scan_role": (
                "correctness_oracle_not_runtime"
                if coarse_gaussian_gemm_compact_posterior_requested
                else None
            ),
            "published_score_source": "exact_relion_source16_or_full_rectangular",
            "expanded_gemm_scores_published": False,
            "whole_batch_fail_closed_fallback": True,
            "score_representation_policy": (
                "compact_only_when_fixed_physical_capacity_is_smaller_than_dense"
            ),
            "static_preferred_score_representation": (
                _select_coarse_gaussian_gemm_score_representation(
                    n_rotations=n_rot,
                    block_capacity=coarse_gaussian_gemm_hybrid_capacity,
                    compact_posterior=(
                        coarse_gaussian_gemm_compact_posterior_requested
                    ),
                )
            ),
            "score_representation_batch_counts": dict(
                sorted(
                    coarse_gaussian_gemm_hybrid_score_representation_batch_counts.items(),
                ),
            ),
            "batch_count": int(coarse_gaussian_gemm_hybrid_batch_count),
            "actual_image_batch_sizes": list(
                coarse_gaussian_gemm_hybrid_actual_image_batch_sizes,
            ),
            "physical_image_batch_sizes": list(
                coarse_gaussian_gemm_hybrid_physical_image_batch_sizes,
            ),
            "input_image_batch_size": int(input_image_batch_size),
            "requested_hybrid_image_batch_size": (
                None
                if coarse_gaussian_gemm_hybrid_image_batch_size_request is None
                else int(coarse_gaussian_gemm_hybrid_image_batch_size_request)
            ),
            "effective_image_batch_size": int(image_batch_size),
            "streamed_certificate_candidate_count_at_effective_batch": int(
                image_batch_size
                * coarse_gaussian_gemm_projection_cache_plan.chunk_rows
                * n_trans
            ),
            "selected_rescore_batch_count": int(
                coarse_gaussian_gemm_hybrid_selected_batch_count,
            ),
            "static_dense_batch_count": int(
                coarse_gaussian_gemm_hybrid_static_dense_batch_count,
            ),
            "fallback_batch_count": int(
                coarse_gaussian_gemm_hybrid_fallback_batch_count,
            ),
            "selected_rescore_image_count": int(
                coarse_gaussian_gemm_hybrid_selected_image_count,
            ),
            "static_dense_image_count": int(
                coarse_gaussian_gemm_hybrid_static_dense_image_count,
            ),
            "fallback_image_count": int(
                coarse_gaussian_gemm_hybrid_fallback_image_count,
            ),
            "overflow_latch_scope": (
                "current_significance_call_exact_geometry_and_capacity"
            ),
            "overflow_latch_active_at_return": bool(
                coarse_gaussian_gemm_hybrid_overflow_latched,
            ),
            "overflow_latch_activation_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_activation_count,
            ),
            "overflow_latch_static_dense_batch_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_batch_count,
            ),
            "overflow_latch_static_dense_image_count": int(
                coarse_gaussian_gemm_hybrid_overflow_latch_static_dense_image_count,
            ),
            "selected_source16_block_count": int(
                coarse_gaussian_gemm_hybrid_selected_block_count,
            ),
            "selected_exact_candidate_count": int(selected_candidate_count),
            "full_candidate_count_for_selected_images": int(
                full_selected_image_candidate_count,
            ),
            "selected_exact_candidate_fraction": (
                float(selected_candidate_count / full_selected_image_candidate_count)
                if full_selected_image_candidate_count
                else None
            ),
            "selected_score_table_capacity_candidates": int(
                compact_table_capacity,
            ),
            "dense_global_score_table_capacity_candidates": int(
                dense_table_capacity,
            ),
            "selected_score_table_capacity_bytes_f32": int(
                compact_table_capacity * np.dtype(np.float32).itemsize,
            ),
            "dense_global_score_table_capacity_bytes_f32": int(
                dense_table_capacity * np.dtype(np.float32).itemsize,
            ),
            "selected_to_dense_score_table_capacity_fraction": (
                float(compact_table_capacity / dense_table_capacity)
                if dense_table_capacity
                else None
            ),
            "static_compact_to_dense_capacity_fraction": float(
                coarse_gaussian_gemm_hybrid_capacity
                * SOURCE_ROTATION_BLOCK_SIZE
                / n_rot
            ),
            "max_selected_blocks_per_image": int(
                coarse_gaussian_gemm_hybrid_max_blocks_per_image,
            ),
            "selected_block_capacity": int(
                coarse_gaussian_gemm_hybrid_capacity,
            ),
            "certificate_chunk_rows": int(
                coarse_gaussian_gemm_projection_cache_plan.chunk_rows,
            ),
            "certificate_chunk_count_per_batch": int(
                coarse_gaussian_gemm_projection_cache_plan.chunk_count_per_table,
            ),
            "topology_full_to_compact_sha256": (
                coarse_gaussian_gemm_certificate_topology.full_to_compact_sha256
            ),
            "fallback_reasons": dict(
                sorted(coarse_gaussian_gemm_hybrid_fallback_reasons.items()),
            ),
        }
    if coarse_gaussian_gemm_diagnostic_paths:
        full_stats["coarse_gaussian_gemm_diagnostic_paths"] = tuple(
            coarse_gaussian_gemm_diagnostic_paths,
        )
    if coarse_gaussian_gemm_scope_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_scope_manifest_path"] = (
            coarse_gaussian_gemm_scope_manifest_path
        )
    if coarse_gaussian_gemm_aggregate_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_aggregate_manifest_path"] = (
            coarse_gaussian_gemm_aggregate_manifest_path
        )
    if coarse_gaussian_gemm_stream_paths:
        full_stats["coarse_gaussian_gemm_stream_diagnostic"] = {
            "artifact_paths": tuple(coarse_gaussian_gemm_stream_paths),
            "retained_topk": int(coarse_gaussian_gemm_stream_topk),
            "persistent_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_dual_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "posterior_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "pre_prior_state_bytes_at_requested_batch_size": (
                coarse_gemm_streaming_state_bytes(
                    int(image_batch_size),
                    int(coarse_gaussian_gemm_stream_topk),
                )
            ),
            "stores_score_cube": False,
            "clean_timing_eligible": False,
            "production_behavior_changed": False,
        }
    if coarse_gaussian_gemm_stream_scope_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_stream_scope_manifest_path"] = (
            coarse_gaussian_gemm_stream_scope_manifest_path
        )
    if coarse_gaussian_gemm_stream_aggregate_manifest_path is not None:
        full_stats["coarse_gaussian_gemm_stream_aggregate_manifest_path"] = (
            coarse_gaussian_gemm_stream_aggregate_manifest_path
        )
    if _coarse_significance_support_audit_enabled():
        if significant_sample_indices is None:
            raise RuntimeError(
                f"{_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}=1 requires "
                "collect_significance=True",
            )
        full_stats["coarse_significance_support_audit"] = (
            _build_coarse_significance_support_audit(
                significant_sample_indices,
                samples_per_class=n_rot * n_trans,
                include_ids=_coarse_significance_support_audit_ids_enabled(),
            )
        )
    if relion_f32_sum_weight is not None:
        # RELION's oversampling-zero second pass deliberately reuses this
        # coarse, maximum-shifted float32 denominator numerically.  It is not
        # interchangeable with a log-evidence value because the fine pass
        # independently shifts its own maximum to 50 before division.
        full_stats["relion_f32_sum_weight"] = relion_f32_sum_weight
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
