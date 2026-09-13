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
from recovar.em.local.local_layout import LocalBucketSpec
from recovar.em.sparse_pass2 import sparse_pass2_bucketed


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
        sparse_pass2_bucketed._relion_f32_fine_reconstruction_probs(
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
