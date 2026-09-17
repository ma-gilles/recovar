"""BPref contribution capture of the exact local EM engine.

The env-gated per-particle BPref contribution capture: its static kwargs and
priors, the reconstruction probabilities it records, the activity checks and
the debug-target bucket filters. None of it changes production arithmetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics import bpref_diagnostics
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.helpers.preprocessing import (
    resolve_image_mask_for_half_preprocess,
    uses_relion_cuda_image_preprocessing,
)
from recovar.em.local.local_layout import LocalBucketSpec
from recovar.em.sparse_pass2 import sparse_pass2_posterior


def _exact_local_bpref_capture_static_kwargs(
    *,
    experiment_dataset,
    score_with_masked_images,
    disc_type,
    projection_padding_factor,
    reconstruction_padding_factor,
    mstep_relion_x_half,
    mstep_adjoint_max_r,
    mstep_recon_window_indices,
    image_shape,
    recon_volume_shape,
) -> dict:
    """Capture fields the exact-local route fixes for every bucket.

    Both the fused and the big-JIT exact-local M-step reach the reusable
    contribution schema without raw batch data, CTF parameters, image masks or
    shadow comparisons, so those operands are recorded as absent
    (``None``/``False``/``"not-captured"``) rather than guessed, and the
    geometry (padding factors, x-half M-step layout, adjoint radius, window
    indices, image and volume shapes) is the run's M-step geometry.
    """

    return dict(
        high_precision_operand_bundle=False,
        raw_batch_data=None,
        ctf_params=None,
        noise_variance_half=None,
        integer_pre_shifts=None,
        batch_image_corrections=None,
        batch_scale_corrections=None,
        relion_preprocess_normalization_factors=None,
        relion_cuda_preprocess=False,
        score_with_masked_images=score_with_masked_images,
        image_mask=None,
        image_mask_mode="not-captured",
        voxel_size=experiment_dataset.voxel_size,
        ctf_mode="not-captured",
        ctf_dose_per_tilt=0.0,
        ctf_angle_per_tilt=0.0,
        disc_type=disc_type,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        use_relion_x_half_mstep=mstep_relion_x_half,
        winner_take_all=False,
        max_r=mstep_adjoint_max_r,
        window_indices=mstep_recon_window_indices,
        image_shape=image_shape,
        volume_shape=recon_volume_shape,
        shadow_only_mode=False,
        shadow_score_bitwise_equal=True,
        shadow_reduction_agreement=None,
    )


@dataclass(frozen=True)
class _BprefCapturePriors:
    """Candidate mask, priors and prior-free scores of one captured bucket."""

    candidate_mask: jnp.ndarray
    rotation_log_prior: jnp.ndarray
    translation_log_prior: jnp.ndarray
    preprior_scores: jnp.ndarray


def _bpref_capture_priors(scores, probs_shape, *, bucket, rotation_log_prior) -> _BprefCapturePriors:
    """Remove the pose priors from a bucket's scores for the contribution capture.

    The candidate mask covers the bucket's real rotation rows and, when the
    bucket carries one, its per-sample mask; prior-free scores outside the mask
    or non-finite are recorded as ``-inf``.
    """

    candidate_mask = jnp.broadcast_to(
        jnp.asarray(bucket.local_rotation_mask)[:, :, None],
        probs_shape,
    )
    if bucket.local_sample_mask is not None:
        candidate_mask = candidate_mask & jnp.asarray(bucket.local_sample_mask)
    translation_log_prior = jnp.asarray(bucket.translation_log_prior)
    preprior_scores = scores - rotation_log_prior[:, :, None] - translation_log_prior[:, None, :]
    preprior_scores = jnp.where(
        candidate_mask & jnp.isfinite(preprior_scores),
        preprior_scores,
        -jnp.inf,
    )
    return _BprefCapturePriors(candidate_mask, rotation_log_prior, translation_log_prior, preprior_scores)


_HIGH_PRECISION_OPERAND_BUNDLE_ENV = "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE"
# The preprocessing implementations a capture bucket can actually have run.
_PREPROCESS_PATHS = frozenset(
    {"big_jit_relion_cuda", "big_jit_jax", "split_exact", "split_backend"}
)


def _require_lossless_float32(name: str, values):
    """Return ``values`` as float32 only when that cast loses nothing.

    The shared writer stores raw images and corrections as float32, while the
    kernel's correction dtype follows ``precision_policy.score_real_dtype`` and
    may be float64.  A silent narrowing would let the capture be described as the
    operand the kernel received when it is not, so reject the unsupported
    precision instead of recording a rounded copy.
    """

    source = np.asarray(values)
    narrowed = source.astype(np.float32)
    if source.dtype != np.float32 and not np.array_equal(narrowed.astype(source.dtype), source):
        raise RuntimeError(
            f"BPref operand bundle cannot record {name} exactly: the shared writer stores "
            f"float32 and this capture source is {source.dtype}, which does not round-trip. "
            "Capture this operand at the writer's precision or leave it declared absent."
        )
    return narrowed


def _exact_local_bpref_operand_bundle(
    static_kwargs: dict,
    *,
    experiment_dataset,
    image_shape,
    preprocess_path: str,
    relion_preprocess_normalization=None,
    relion_cuda_preprocess_radius=None,
    relion_cuda_preprocess_cosine_width=None,
    applied_image_mask,
    applied_image_mask_mode,
    raw_batch_data,
    ctf_params,
    noise_variance_half,
    image_pre_shifts,
    integer_pre_shifts,
    real_space_pre_shift_applied: bool,
    image_corrections,
    scale_corrections,
    image_indices,
    unpadded_batch_size: int,
) -> dict:
    """Attach the engine's own image-side operands when explicitly requested.

    Exact-local search reaches the shared contribution schema through a route that
    historically declared these operands absent. The existing
    ``RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE`` request already selects them on the
    bucketed sparse-pass-2 route; honour the same request here so a replay can *check* a
    host reconstruction against the operands the kernel actually received.

    ``exact_preprocess_branch`` is the branch ``local_preprocessing._process_half``
    actually took, passed in by the caller. It is never inferred from the dataset's
    backend: ``relion_exact_bpref_operands`` selects ``_big_jit_preprocess_half`` and
    deliberately bypasses backend preprocessing, so a ``relion_cuda``-configured dataset
    can run a bucket to which no backend normalization was applied. Reading the backend
    instead would refuse that bucket wrongly, and on the ordinary branch it would let a
    real normalization pass unrecorded.

    Refusals preserved, each for an operand this route cannot observe:

    * ordinary branch on a RELION-CUDA-preprocessed dataset -- real normalization factors
      are applied inside ``prepare_batch_preprocess_operands`` and are not forwarded here;
    * a pre-shift that was applied without an available integer array (the
      ``processed_half_cache`` case), which zeros would misdescribe as no shift;
    * non-integral ``image_pre_shifts``, where the engine applies Fourier phase shifts;
    * an operand whose precision does not round-trip the writer's float32 storage.

    With the request absent this returns ``static_kwargs`` unchanged. Nothing here
    participates in scoring or M-step arithmetic, and the caller stays outside every JIT
    boundary.
    """

    if not parse_env_flag(_HIGH_PRECISION_OPERAND_BUNDLE_ENV, default=False):
        return static_kwargs
    if raw_batch_data is None:
        raise RuntimeError(
            "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE requires raw real-space image batches; "
            "this bucket ran from a preprocessed cache with no raw source rows"
        )
    rows = int(unpadded_batch_size)
    indices = np.asarray(image_indices, dtype=np.int64)[:rows]

    # The preprocessing implementation this bucket actually ran, named by its call site.
    # relion_exact_bpref_operands alone does NOT identify it: on the split site that flag
    # selects local_preprocessing._big_jit_preprocess_half, which bypasses backend
    # preprocessing, while on the big-JIT site the same flag (with a positive mask radius)
    # selects cuda_backproject.relion_preprocess_real_f32 -- real RELION CUDA
    # preprocessing. Labelling both from that one flag inverts the big-JIT case.
    if preprocess_path not in _PREPROCESS_PATHS:
        raise RuntimeError(
            f"BPref operand bundle requires an explicit preprocessing path, one of "
            f"{sorted(_PREPROCESS_PATHS)}; got {preprocess_path!r}"
        )
    backend_is_relion_cuda = bool(uses_relion_cuda_image_preprocessing(experiment_dataset))
    relion_cuda_preprocess_applied = preprocess_path == "big_jit_relion_cuda"

    if preprocess_path == "big_jit_relion_cuda":
        # local_big_jit.py:2257 normalizes with image_only_corrections and masks with a
        # parametric radius/cosine width. Both are recorded: the normalization as the
        # factor array, the mask as the two scalar fields the writer now carries.
        if relion_preprocess_normalization is None:
            raise RuntimeError(
                "BPref operand bundle on the big-JIT RELION CUDA path requires the actual "
                "image_only_corrections passed to relion_preprocess_real_f32; unit factors "
                "would claim a normalization this bucket did not use"
            )
        if relion_cuda_preprocess_radius is None or relion_cuda_preprocess_cosine_width is None:
            raise RuntimeError(
                "BPref operand bundle on the big-JIT RELION CUDA path requires the actual "
                "mask radius and cosine width passed to relion_preprocess_real_f32; the "
                "scored images were masked parametrically and cannot be described without them"
            )
        normalization_factors = _require_lossless_float32(
            "relion_preprocess_normalization",
            np.asarray(relion_preprocess_normalization)[:rows],
        )
    elif preprocess_path == "split_backend" and backend_is_relion_cuda:
        raise RuntimeError(
            "BPref operand bundle cannot describe this bucket: it took the ordinary "
            "local_preprocessing branch on a RELION-CUDA-preprocessed dataset, whose real "
            "normalization factors are applied inside prepare_batch_preprocess_operands and "
            "are not forwarded to this capture boundary. Recording unit factors would claim "
            "a preprocessing policy this bucket did not use."
        )
    else:
        # split_exact bypasses backend preprocessing; big_jit_jax has a zero mask radius
        # so relion_preprocess_real_f32 is not reached. No normalization was applied.
        normalization_factors = np.ones(rows, dtype=np.float32)

    if real_space_pre_shift_applied and integer_pre_shifts is None:
        raise RuntimeError(
            "BPref operand bundle cannot describe this bucket's pre-shift: a real-space shift "
            "was applied but the integer array is unavailable here (processed_half_cache), and "
            "zeros would claim that no shift was applied"
        )
    if image_pre_shifts is not None and integer_pre_shifts is None:
        raise RuntimeError(
            "BPref operand bundle cannot describe non-integral image_pre_shifts: the engine "
            "applies Fourier phase shifts on this path, which the integer_pre_shifts field "
            "cannot represent without claiming the wrong shift path"
        )

    # The mask the bucket actually used. local_preprocessing resolves it only on the
    # exact branch; re-resolving here would invent one for the ordinary branch.
    if relion_cuda_preprocess_applied:
        # No array mask exists on this path; the two scalars are the record.
        image_mask = np.zeros((0,), dtype=np.float32)
        image_mask_mode = "relion_cuda_parametric"
    elif applied_image_mask is None:
        if bool(static_kwargs["score_with_masked_images"]):
            raise RuntimeError(
                "BPref operand bundle requires the image mask this bucket scored with, but "
                "none was supplied while score_with_masked_images is set"
            )
        image_mask = np.zeros((0,), dtype=np.float32)
        image_mask_mode = "no-mask-applied"
    else:
        image_mask = np.asarray(applied_image_mask, dtype=np.float32)
        image_mask_mode = str(applied_image_mask_mode)

    # Absent corrections are not guessed: the engine multiplies by nothing, which is
    # exactly the unit operand the bucketed route also records for this case.
    batch_image_corrections = (
        np.ones(rows, dtype=np.float32)
        if image_corrections is None
        else _require_lossless_float32("image_corrections", np.asarray(image_corrections)[indices])
    )
    batch_scale_corrections = (
        np.ones(rows, dtype=np.float32)
        if scale_corrections is None
        else _require_lossless_float32("scale_corrections", np.asarray(scale_corrections)[indices])
    )
    return {
        **static_kwargs,
        "high_precision_operand_bundle": True,
        "raw_batch_data": _require_lossless_float32(
            "raw_batch_data", np.asarray(raw_batch_data)[:rows]
        ),
        "ctf_params": np.asarray(ctf_params)[:rows],
        "noise_variance_half": np.asarray(noise_variance_half),
        "integer_pre_shifts": (
            np.zeros((rows, 2), dtype=np.int32)
            if integer_pre_shifts is None
            else np.asarray(integer_pre_shifts, dtype=np.int32)[:rows]
        ),
        "batch_image_corrections": batch_image_corrections,
        "batch_scale_corrections": batch_scale_corrections,
        "relion_preprocess_normalization_factors": normalization_factors,
        # The parametric mask relion_preprocess_real_f32 applied, or None on the
        # paths that never reach it (the writer records NaN there).
        # The engine path is its own explicit field, never folded into a mode string:
        # existing replayers compare image_mask_mode verbatim.
        "preprocess_path": preprocess_path,
        "relion_cuda_preprocess_radius": (
            relion_cuda_preprocess_radius if relion_cuda_preprocess_applied else None
        ),
        "relion_cuda_preprocess_cosine_width": (
            relion_cuda_preprocess_cosine_width if relion_cuda_preprocess_applied
            else None
        ),
        "relion_cuda_preprocess": relion_cuda_preprocess_applied,
        "image_mask": image_mask,
        "image_mask_mode": image_mask_mode,
        # The exact branch builds the CTF from the source STAR
        # (_relion_exact_ctf_half_from_source_star_host), not from these ctf_params, so a
        # replay must know which construction produced the scored CTF.
        "ctf_mode": ("relion_exact_source_star"
                     if preprocess_path in ("split_exact", "big_jit_relion_cuda", "big_jit_jax")
                     else str(static_kwargs["ctf_mode"])),
    }


def _maybe_dump_exact_local_bpref_contribution_rows(**kwargs) -> None:
    """Write exact-local pre-scatter rows without claiming device geometry.

    The reusable contribution schema already describes the posterior-reduced
    BPref operands needed for canonical replay.  Exact-local search reaches the
    same boundary through a different engine, so forward its materialized
    bucket there when explicitly requested.  Device-produced neighbor
    signatures remain unsupported on this route and must continue to fail
    before execution rather than silently emitting an incomplete capture.
    """

    if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip():
        return
    if os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip():
        raise RuntimeError(
            "Exact-local BPref contribution capture does not yet support device signatures"
        )
    bpref_diagnostics._maybe_dump_bpref_contribution_rows(**kwargs)


def _exact_local_bpref_reconstruction_probs_for_capture(
    scores,
    generic_probs,
    reconstruction_sample_mask,
    *,
    use_relion_f32_fine_posterior: bool,
    adaptive_fraction: float,
):
    """Return the same reconstruction weights consumed by the big-JIT M-step.

    ``debug_probs`` deliberately exposes the generic normalized posterior for
    score diagnostics.  It is not the M-step tensor when the RELION float32
    exp/sort/scan/divide path is active, so a contribution capture must rebuild
    that source-faithful tensor from the returned score boundary.
    """

    if not use_relion_f32_fine_posterior:
        return jnp.where(reconstruction_sample_mask, generic_probs, 0.0)
    reconstruction_probs, exact_mask, *_diagnostics = (
        sparse_pass2_posterior._relion_f32_fine_reconstruction_probs(
            scores,
            adaptive_fraction=float(adaptive_fraction),
        )
    )
    if not np.array_equal(
        np.asarray(exact_mask, dtype=bool),
        np.asarray(reconstruction_sample_mask, dtype=bool),
    ):
        raise RuntimeError(
            "big-JIT BPref capture rebuilt a different RELION reconstruction mask"
        )
    return reconstruction_probs


def _exact_local_bpref_contribution_capture_active(
    *, current_size: int | None, debug_iteration: int | None
) -> bool:
    """Return whether this exact-local half is the explicitly targeted boundary."""

    if not os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "").strip():
        return False
    context = bpref_diagnostics._bpref_contribution_context
    context_iteration = int(context["iteration"])
    context_half = int(context["half"])
    target_iteration = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "").strip()
    target_half = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "").strip()
    target_current_size = os.environ.get(
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE", ""
    ).strip()
    if not (target_iteration and target_half and target_current_size):
        return False
    if context_iteration != int(target_iteration):
        return False
    if context_half != int(target_half):
        return False
    if current_size is None or int(current_size) != int(target_current_size):
        return False
    if debug_iteration is not None and context_iteration != int(debug_iteration):
        return False
    return context_iteration > 0 and context_half in {1, 2}


def _exact_local_bpref_contribution_capture_for_call(
    *,
    current_size: int | None,
    debug_iteration: int | None,
    score_only: bool,
    mstep_relion_x_half: bool,
) -> bool:
    """Activate capture only at a compatible fine-pass M-step boundary."""

    requested = _exact_local_bpref_contribution_capture_active(
        current_size=current_size,
        debug_iteration=debug_iteration,
    )
    if not requested or score_only:
        return False
    if not mstep_relion_x_half:
        raise RuntimeError(
            "Exact-local BPref contribution capture requires RELION x-half M-step geometry"
        )
    return True


def _bucket_contains_debug_target(experiment_dataset, image_indices, pending_targets: set[int] | None) -> bool:
    if not pending_targets:
        return False
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    return any(int(original_idx) in pending_targets for original_idx in original_indices.tolist())


def _filter_buckets_to_debug_targets(
    experiment_dataset,
    bucket_specs: list[LocalBucketSpec],
    pending_targets: set[int],
) -> list[LocalBucketSpec]:
    if not pending_targets:
        return bucket_specs
    return [
        bucket
        for bucket in bucket_specs
        if _bucket_contains_debug_target(
            experiment_dataset,
            bucket.image_indices,
            pending_targets,
        )
    ]
