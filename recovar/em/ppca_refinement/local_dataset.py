"""Exact-local PPCA refinement over ``LocalHypothesisLayout`` supports."""

from __future__ import annotations

import dataclasses
import os
from functools import partial
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.preprocessing import (
    prepare_reconstruction_batch,
    preprocess_batch,
)
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout, bucket_local_hypothesis_layout
from recovar.em.ppca_refinement.config import (
    GeometryConfig,
    PoseSelectionConfig,
    ScheduleConfig,
    ScoringConfig,
    SparsePass2Config,
)
from recovar.em.ppca_refinement.dense_dataset import prepare_dense_ppca_dataset_inputs
from recovar.em.ppca_refinement.engine import (
    DensePPCAFusedBlock,
    DensePPCAFusedEMResult,
    PosteriorDiagnostics,
    _enforce_augmented_x0,
)
from recovar.em.ppca_refinement.diagnostics import build_iteration_diagnostics, resolve_image_scale_range
from recovar.em.ppca_refinement.mean_regularization import (
    MeanRegularizationConfig,
    resolve_mean_precision,
)
from recovar.em.ppca_refinement.postprocess import PostprocessConfig, postprocess_ppca_half_volumes
from recovar.em.ppca_refinement.pose_selection import (
    select_distinct_top_poses,
    top_pose_candidate_count,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.ppca import AugmentedPPCAStats, augmented_ppca_mstep_objective, solve_augmented_ppca_mstep
from recovar.ppca.pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast
from recovar.ppca.triangular import _tri_size
from recovar.reconstruction import noise as noise_utils

# Smart-sizing defaults mirror the K-class engine's
# ``EXACT_LOCAL_TARGET_ROW_PIXELS`` (190 M scoring pixels per microbatch),
# scaled down by the PPCA augmented-volume row factor ``P = 1 + q``.
PPCA_LOCAL_TARGET_ROW_PIXELS = 190_000_000
PPCA_LOCAL_TARGET_ROW_PIXELS_ENV = "RECOVAR_PPCA_LOCAL_TARGET_ROW_PIXELS"
PPCA_LOCAL_MAX_HYPO_FLOOR = 2_048
PPCA_LOCAL_MAX_HYPO_CEIL = 16_384
PPCA_LOCAL_IMAGE_BATCH_FLOOR = 1
# Conservative ceil: bigger ``image_batch_size`` amortizes JIT/data-transfer
# overhead but raises peak memory linearly (proj_aug tensor is
# ``image_batch × R × P × F × 8`` bytes plus working buffers, typically
# ~3-5× that). At 8 we comfortably fit in ~10 GiB; 16 needs ~20 GiB. Bump
# only when the GPU has plenty of free memory.
PPCA_LOCAL_IMAGE_BATCH_CEIL = int(os.environ.get("RECOVAR_PPCA_LOCAL_IMAGE_BATCH_CEIL", "8"))
PPCA_LOCAL_AUTO_DISABLE_ENV = "RECOVAR_PPCA_LOCAL_AUTO_SIZING_DISABLE"


def _ppca_local_smart_max_hypotheses_per_microbatch(default: int | None, n_windowed: int, q: int) -> int:
    """Mirror ``_exact_local_max_hypotheses_per_microbatch`` but divide the
    pixel budget by ``P = 1 + q`` so the augmented projection tensor stays
    inside the same per-microbatch memory envelope as the K-class kernel."""
    if default is not None:
        value = int(default)
        if value <= 0:
            raise ValueError("max_hypotheses_per_microbatch must be positive")
        return value
    if os.environ.get(PPCA_LOCAL_AUTO_DISABLE_ENV, "0") == "1":
        return 32_768  # legacy default
    target_row_pixels = int(os.environ.get(PPCA_LOCAL_TARGET_ROW_PIXELS_ENV, PPCA_LOCAL_TARGET_ROW_PIXELS))
    if target_row_pixels <= 0:
        raise ValueError(f"{PPCA_LOCAL_TARGET_ROW_PIXELS_ENV} must be positive")
    P = max(1, int(q) + 1)
    raw = target_row_pixels // (max(1, int(n_windowed)) * P)
    return int(max(PPCA_LOCAL_MAX_HYPO_FLOOR, min(PPCA_LOCAL_MAX_HYPO_CEIL, raw)))


def _ppca_local_smart_image_batch_size(
    default: int | None,
    *,
    max_hypotheses_per_microbatch: int,
    mean_bucket_size: int,
) -> int:
    """Pick the largest ``image_batch_size`` that still fits inside the
    microbatch hypothesis budget for the typical bucket size of the run.

    Falls back to the K-class style cap of 32 to avoid huge JIT compile
    times. Caller-supplied ``default`` (when not None and not 2) is
    respected so explicit user choices win."""
    if default is not None and int(default) not in (0, 2):
        return int(default)
    if os.environ.get(PPCA_LOCAL_AUTO_DISABLE_ENV, "0") == "1":
        return 2 if default is None else int(default)
    bucket = max(1, int(mean_bucket_size))
    cap = max(1, int(max_hypotheses_per_microbatch) // bucket)
    return int(max(PPCA_LOCAL_IMAGE_BATCH_FLOOR, min(PPCA_LOCAL_IMAGE_BATCH_CEIL, cap)))


class LocalPPCAFusedBucketBlock(NamedTuple):
    """Prepared exact-local PPCA bucket with image-specific rotations."""

    Y1: jax.Array
    proj_aug: jax.Array
    ctf2_over_noise: jax.Array
    y_norm: jax.Array
    rotations: jax.Array
    pose_log_prior: jax.Array
    Y1_recon: jax.Array
    ctf2_over_noise_recon: jax.Array
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    image_indices: np.ndarray
    recon_window_indices: jax.Array | None = None
    use_recon_window: bool = False
    backprojection_max_r: float | None = None


class LocalPPCAAccumulationResult(NamedTuple):
    """Stats-only exact-local PPCA E-step result before the augmented M-step."""

    stats: AugmentedPPCAStats
    postprocess_bandlimit_max_r: float | None = None


class LocalPPCAPoseScoringResult(NamedTuple):
    """Exact-local PPCA score-only E-step diagnostics."""

    diagnostics: dict


class LocalScoreAndMomentsStats(NamedTuple):
    """Local score pass output reused by exact or top-k M-step backprojection."""

    score: jax.Array
    alpha: jax.Array
    G_tri: jax.Array
    diagnostics: PosteriorDiagnostics


def _resolve_local_image_indices(experiment_dataset, local_layout: LocalHypothesisLayout, image_indices):
    if image_indices is None:
        n_dataset = int(getattr(experiment_dataset, "n_units", experiment_dataset.n_images))
        if int(local_layout.n_images) != n_dataset:
            raise ValueError(
                f"local_layout.n_images={local_layout.n_images} does not match dataset image count {n_dataset}; "
                "pass image_indices when using a subset layout",
            )
        return np.arange(n_dataset, dtype=np.int64)
    image_indices = np.asarray(image_indices, dtype=np.int64).reshape(-1)
    if int(local_layout.n_images) != int(image_indices.shape[0]):
        raise ValueError(
            f"local_layout.n_images={local_layout.n_images} does not match image_indices length "
            f"{image_indices.shape[0]}",
        )
    return image_indices


def _slice_local_hypothesis_layout(layout: LocalHypothesisLayout, rows) -> LocalHypothesisLayout:
    """Return a layout containing only the requested image rows."""

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    offsets = np.zeros(rows.shape[0] + 1, dtype=np.int64)
    counts = np.asarray(layout.rotation_counts[rows], dtype=np.int32)
    rotation_ids_parts = []
    rotations_parts = []
    prior_parts = []
    posterior_id_parts = [] if layout.rotation_posterior_ids_flat is not None else None
    sample_mask_parts = [] if layout.sample_mask_flat is not None else None
    for out_row, row in enumerate(rows.tolist()):
        start = int(layout.rotation_offsets[row])
        end = int(layout.rotation_offsets[row + 1])
        offsets[out_row + 1] = offsets[out_row] + (end - start)
        rotation_ids_parts.append(np.asarray(layout.rotation_ids_flat[start:end], dtype=np.int32))
        rotations_parts.append(np.asarray(layout.rotations_flat[start:end], dtype=np.float32))
        prior_parts.append(np.asarray(layout.rotation_log_priors_flat[start:end], dtype=np.float32))
        if posterior_id_parts is not None:
            posterior_id_parts.append(np.asarray(layout.rotation_posterior_ids_flat[start:end], dtype=np.int32))
        if sample_mask_parts is not None:
            sample_mask_parts.append(np.asarray(layout.sample_mask_flat[start:end], dtype=bool))

    translation_priors = np.asarray(layout.translation_log_priors, dtype=np.float32)
    if translation_priors.ndim == 1:
        translation_priors = np.broadcast_to(
            translation_priors[None, :],
            (rows.shape[0], translation_priors.shape[0]),
        ).copy()
    elif translation_priors.ndim == 2:
        translation_priors = translation_priors[rows]
    else:
        raise ValueError(f"translation_log_priors must be 1D or 2D, got {translation_priors.shape}")

    return LocalHypothesisLayout(
        n_global_rotations=int(layout.n_global_rotations),
        n_pixels=int(layout.n_pixels),
        n_psi=int(layout.n_psi),
        rotation_offsets=offsets,
        rotation_ids_flat=np.concatenate(rotation_ids_parts) if rotation_ids_parts else np.zeros(0, dtype=np.int32),
        rotations_flat=(
            np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
        ),
        rotation_log_priors_flat=np.concatenate(prior_parts) if prior_parts else np.zeros(0, dtype=np.float32),
        rotation_counts=counts,
        translation_grid=np.asarray(layout.translation_grid, dtype=np.float32),
        translation_log_priors=translation_priors,
        rotation_posterior_ids_flat=(
            None
            if posterior_id_parts is None
            else np.concatenate(posterior_id_parts) if posterior_id_parts else np.zeros(0, dtype=np.int32)
        ),
        sample_mask_flat=(
            None
            if sample_mask_parts is None
            else np.concatenate(sample_mask_parts, axis=0)
            if sample_mask_parts
            else np.zeros((0, int(layout.translation_grid.shape[0])), dtype=bool)
        ),
    )


def _local_translation_log_prior(layout: LocalHypothesisLayout, image_index: int) -> np.ndarray:
    prior = np.asarray(layout.translation_log_priors, dtype=np.float32)
    if prior.ndim == 1:
        return prior
    if prior.ndim == 2:
        return prior[int(image_index)]
    raise ValueError(f"translation_log_priors must be 1D or 2D, got {prior.shape}")


def _fetch_single_image_batch(experiment_dataset, image_index: int):
    batch_iter = experiment_dataset.iter_batches(
        1,
        indices=np.asarray([int(image_index)], dtype=np.int64),
        by_image=False,
    )
    try:
        return next(batch_iter)
    except StopIteration as exc:
        raise ValueError(f"Could not fetch image index {image_index}") from exc


from recovar.em.ppca_refinement.dense_dataset import (
    _project_augmented_half_volumes as _project_local_augmented,
)


def _per_pose_stats_local_bucket(Y1, proj_aug, ctf2_over_noise, y_norm):
    """Build PPCA sufficient stats for image-specific local rotation buckets."""

    B, T, F = Y1.shape
    proj_B, R, P, proj_F = proj_aug.shape
    if proj_B != B or proj_F != F:
        raise ValueError(f"proj_aug shape {proj_aug.shape} is incompatible with Y1 shape {Y1.shape}")
    if ctf2_over_noise.shape != (B, F):
        raise ValueError(f"ctf2_over_noise shape {ctf2_over_noise.shape} != ({B}, {F})")
    if y_norm.shape != (B,):
        raise ValueError(f"y_norm shape {y_norm.shape} != ({B},)")

    q = P - 1
    proj_mu = proj_aug[:, :, 0, :]
    proj_W = proj_aug[:, :, 1:, :]
    ctf2 = ctf2_over_noise.astype(proj_aug.dtype)

    nu_mm = jnp.einsum("bf,brf,brf->br", ctf2, jnp.conj(proj_mu), proj_mu).real
    t_mx = jnp.einsum("btf,brf->btr", jnp.conj(Y1), proj_mu).real
    if q == 0:
        g_zx = jnp.zeros((B, T, R, 0), dtype=proj_aug.dtype)
        h_zm = jnp.zeros((B, R, 0), dtype=proj_aug.dtype)
        Hzz = jnp.zeros((B, R, 0, 0), dtype=proj_aug.dtype)
    else:
        g_zx = jnp.einsum("btf,brqf->btrq", Y1, jnp.conj(proj_W)).real
        h_zm = jnp.einsum("bf,brqf,brf->brq", ctf2, jnp.conj(proj_W), proj_mu).real
        Hzz = jnp.einsum("bf,brqf,brpf->brqp", ctf2, jnp.conj(proj_W), proj_W).real
    return (
        jnp.broadcast_to(y_norm[:, None, None], (B, T, R)),
        t_mx,
        jnp.broadcast_to(nu_mm[:, None, :], (B, T, R)),
        g_zx,
        jnp.broadcast_to(h_zm[:, None, :, :], (B, T, R, q)),
        jnp.broadcast_to(Hzz[:, None, :, :, :], (B, T, R, q, q)),
    )


def _score_gamma_and_moments_local_bucket(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior,
    significance_threshold: float,
    top_pose_count: int = 1,
):
    y_stats = _per_pose_stats_local_bucket(Y1, proj_aug, ctf2_over_noise, y_norm)
    score_pre, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=True,
    )
    score = score_pre + jnp.swapaxes(jnp.asarray(pose_log_prior), -1, -2)
    B, T, R = score.shape
    score_flat = score.reshape(B, T * R)
    logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)
    gamma = jnp.exp(score - logZ[:, None, None])
    best_flat = jnp.argmax(score_flat, axis=-1)
    pmax = jnp.max(gamma.reshape(B, T * R), axis=-1)
    k = max(1, min(int(top_pose_count), int(score_flat.shape[-1])))
    top_scores, top_flat = jax.lax.top_k(score_flat, k)
    top_rot = (top_flat % R).astype(jnp.int32)
    top_trans = (top_flat // R).astype(jnp.int32)
    top_prob = jnp.exp(top_scores - logZ[:, None]).astype(jnp.float32)
    diagnostics = PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=(best_flat % R).astype(jnp.int32),
        best_translation_idx=(best_flat // R).astype(jnp.int32),
        n_significant_per_image=jnp.sum(gamma > float(significance_threshold), axis=(1, 2)).astype(jnp.int32),
        best_log_score_per_image=jnp.max(score_flat, axis=-1).astype(jnp.float32),
        rotation_posterior_sums=jnp.sum(gamma, axis=(0, 1)).astype(jnp.float32),
        max_posterior_per_image=pmax,
        top_rotation_idx=top_rot,
        top_translation_idx=top_trans,
        top_log_score_per_image=top_scores.astype(jnp.float32),
        top_posterior_per_image=top_prob,
    )
    return gamma, alpha, G_tri, diagnostics


@partial(
    jax.jit,
    static_argnames=(
        "significance_threshold",
        "top_pose_count",
    ),
)
def score_local_pose_ppca_bucket(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior,
    *,
    significance_threshold: float = 1e-3,
    top_pose_count: int = 1,
) -> PosteriorDiagnostics:
    """Compute exact-local PPCA pose posterior diagnostics without M-step moments."""

    y_stats = _per_pose_stats_local_bucket(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
    )
    score_pre, _alpha, _G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=False,
    )
    score = score_pre + jnp.swapaxes(jnp.asarray(pose_log_prior), -1, -2)
    B, T, R = score.shape
    score_flat = score.reshape(B, T * R)
    logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)
    gamma = jnp.exp(score - logZ[:, None, None])
    best_flat = jnp.argmax(score_flat, axis=-1)
    pmax = jnp.max(gamma.reshape(B, T * R), axis=-1)
    k = max(1, min(int(top_pose_count), int(score_flat.shape[-1])))
    top_scores, top_flat = jax.lax.top_k(score_flat, k)
    return PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=(best_flat % R).astype(jnp.int32),
        best_translation_idx=(best_flat // R).astype(jnp.int32),
        n_significant_per_image=jnp.sum(gamma > float(significance_threshold), axis=(1, 2)).astype(jnp.int32),
        best_log_score_per_image=jnp.max(score_flat, axis=-1).astype(jnp.float32),
        rotation_posterior_sums=jnp.sum(gamma, axis=(0, 1)).astype(jnp.float32),
        max_posterior_per_image=pmax,
        top_rotation_idx=(top_flat % R).astype(jnp.int32),
        top_translation_idx=(top_flat // R).astype(jnp.int32),
        top_log_score_per_image=top_scores.astype(jnp.float32),
        top_posterior_per_image=jnp.exp(top_scores - logZ[:, None]).astype(jnp.float32),
    )


def _score_local_pose_ppca_bucket_rotation_chunked(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior,
    *,
    significance_threshold: float,
    top_pose_count: int,
    rotation_chunk_size: int,
) -> PosteriorDiagnostics:
    """Memory-tiled wrapper around :func:`score_local_pose_ppca_bucket`.

    Splits the R (rotation) dim of ``proj_aug`` into chunks of
    ``rotation_chunk_size``, dispatches the existing JIT'd score kernel
    on each chunk, and aggregates the per-chunk diagnostics into a single
    :class:`PosteriorDiagnostics` matching the one-shot kernel. R is padded
    to a multiple of ``rotation_chunk_size`` with rotations whose
    ``pose_log_prior`` is ``-inf`` so they cannot win the top-K.

    Use this when the one-shot kernel OOMs on large per-image neighborhoods
    (e.g. HP6 top-p p=4 at 256² with ~1700-2900 rotations per image):
    peak working memory scales with ``image_batch × chunk × (q+1) × n_pixels``
    instead of ``image_batch × R × (q+1) × n_pixels``.

    The ``rotation_posterior_sums`` field is set to zeros — the streaming
    chunked path does not reconstruct per-rotation marginals (the EM
    M-step uses a separate accumulator kernel that is not affected).
    The ``n_significant_per_image`` field is a conservative over-estimate
    (sum of per-chunk counts using per-chunk local normalization) since
    each chunk-local gamma is larger than the globally-normalized gamma.
    """

    proj_aug = jnp.asarray(proj_aug)
    pose_log_prior = jnp.asarray(pose_log_prior)
    B = int(proj_aug.shape[0])
    R = int(proj_aug.shape[1])
    chunk = int(rotation_chunk_size)
    if chunk <= 0:
        raise ValueError("rotation_chunk_size must be positive")
    if chunk >= R:
        return score_local_pose_ppca_bucket(
            Y1,
            proj_aug,
            ctf2_over_noise,
            y_norm,
            pose_log_prior,
            significance_threshold=significance_threshold,
            top_pose_count=top_pose_count,
        )

    # Pad R to a multiple of chunk so every dispatch sees the same (B, chunk, P, F) shape
    pad_amount = (-R) % chunk
    if pad_amount > 0:
        proj_pad_shape = (B, pad_amount) + proj_aug.shape[2:]
        proj_aug_padded = jnp.concatenate(
            [proj_aug, jnp.zeros(proj_pad_shape, dtype=proj_aug.dtype)], axis=1
        )
        prior_pad_shape = (B, pad_amount) + pose_log_prior.shape[2:]
        pose_log_prior_padded = jnp.concatenate(
            [pose_log_prior, jnp.full(prior_pad_shape, -jnp.inf, dtype=pose_log_prior.dtype)],
            axis=1,
        )
    else:
        proj_aug_padded = proj_aug
        pose_log_prior_padded = pose_log_prior

    R_padded = R + pad_amount
    n_chunks = R_padded // chunk
    K = max(1, min(int(top_pose_count), R * int(Y1.shape[1])))

    chunk_logZs = []
    chunk_best_scores = []
    chunk_best_rot = []
    chunk_best_trans = []
    chunk_top_K_scores = []
    chunk_top_K_rot = []
    chunk_top_K_trans = []
    chunk_n_significant = []
    chunk_r_offsets = []

    for c in range(n_chunks):
        r0 = c * chunk
        r1 = r0 + chunk
        slab_proj = proj_aug_padded[:, r0:r1]
        slab_prior = pose_log_prior_padded[:, r0:r1]
        diag = score_local_pose_ppca_bucket(
            Y1,
            slab_proj,
            ctf2_over_noise,
            y_norm,
            slab_prior,
            significance_threshold=significance_threshold,
            top_pose_count=top_pose_count,
        )
        chunk_logZs.append(diag.logZ)
        chunk_best_scores.append(diag.best_log_score_per_image)
        chunk_best_rot.append(diag.best_rotation_idx)
        chunk_best_trans.append(diag.best_translation_idx)
        chunk_top_K_scores.append(diag.top_log_score_per_image)
        chunk_top_K_rot.append(diag.top_rotation_idx)
        chunk_top_K_trans.append(diag.top_translation_idx)
        chunk_n_significant.append(diag.n_significant_per_image)
        chunk_r_offsets.append(r0)

    logZ_stack = jnp.stack(chunk_logZs, axis=-1)  # (B, n_chunks)
    global_logZ = jax.scipy.special.logsumexp(logZ_stack, axis=-1)

    best_score_stack = jnp.stack(chunk_best_scores, axis=-1).astype(jnp.float32)  # (B, n_chunks)
    best_chunk_idx = jnp.argmax(best_score_stack, axis=-1)  # (B,)
    global_best_score = jnp.take_along_axis(best_score_stack, best_chunk_idx[:, None], axis=-1).squeeze(-1)

    best_rot_stack = jnp.stack(chunk_best_rot, axis=-1).astype(jnp.int32)
    best_trans_stack = jnp.stack(chunk_best_trans, axis=-1).astype(jnp.int32)
    r_offsets = jnp.asarray(chunk_r_offsets, dtype=jnp.int32)  # (n_chunks,)
    best_chunk_local_rot = jnp.take_along_axis(best_rot_stack, best_chunk_idx[:, None], axis=-1).squeeze(-1)
    best_chunk_offset = r_offsets[best_chunk_idx]
    best_rotation_idx = (best_chunk_local_rot + best_chunk_offset).astype(jnp.int32)
    best_translation_idx = jnp.take_along_axis(best_trans_stack, best_chunk_idx[:, None], axis=-1).squeeze(-1).astype(jnp.int32)

    top_K_scores_stack = jnp.stack(chunk_top_K_scores, axis=1)  # (B, n_chunks, K)
    top_K_rot_stack = jnp.stack(chunk_top_K_rot, axis=1).astype(jnp.int32)
    top_K_trans_stack = jnp.stack(chunk_top_K_trans, axis=1).astype(jnp.int32)
    chunk_offsets_b = jnp.broadcast_to(r_offsets[None, :, None], top_K_rot_stack.shape)
    top_K_rot_global = top_K_rot_stack + chunk_offsets_b

    combined_scores = top_K_scores_stack.reshape(B, n_chunks * K)
    combined_rot = top_K_rot_global.reshape(B, n_chunks * K)
    combined_trans = top_K_trans_stack.reshape(B, n_chunks * K)

    final_top_scores, final_top_indices = jax.lax.top_k(combined_scores, K)
    final_top_rot = jnp.take_along_axis(combined_rot, final_top_indices, axis=1)
    final_top_trans = jnp.take_along_axis(combined_trans, final_top_indices, axis=1)
    final_top_posterior = jnp.exp(final_top_scores - global_logZ[:, None]).astype(jnp.float32)

    pmax = jnp.exp(global_best_score - global_logZ).astype(jnp.float32)
    n_sig_total = jnp.sum(jnp.stack(chunk_n_significant, axis=-1), axis=-1).astype(jnp.int32)

    return PosteriorDiagnostics(
        logZ=global_logZ,
        pmax=pmax,
        best_rotation_idx=best_rotation_idx,
        best_translation_idx=best_translation_idx,
        n_significant_per_image=n_sig_total,
        best_log_score_per_image=global_best_score.astype(jnp.float32),
        rotation_posterior_sums=jnp.zeros((R,), dtype=jnp.float32),
        max_posterior_per_image=pmax,
        top_rotation_idx=final_top_rot,
        top_translation_idx=final_top_trans,
        top_log_score_per_image=final_top_scores.astype(jnp.float32),
        top_posterior_per_image=final_top_posterior,
    )


@partial(
    jax.jit,
    static_argnames=(
        "significance_threshold",
        "top_pose_count",
    ),
)
def score_local_pose_ppca_bucket_with_moments(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior,
    *,
    significance_threshold: float = 1e-3,
    top_pose_count: int = 1,
) -> LocalScoreAndMomentsStats:
    """Compute local PPCA scores and moments once for reusable M-step paths."""

    y_stats = _per_pose_stats_local_bucket(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
    )
    score_pre, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=True,
    )
    score = score_pre + jnp.swapaxes(jnp.asarray(pose_log_prior), -1, -2)
    B, T, R = score.shape
    score_flat = score.reshape(B, T * R)
    logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)
    gamma = jnp.exp(score - logZ[:, None, None])
    best_flat = jnp.argmax(score_flat, axis=-1)
    pmax = jnp.max(gamma.reshape(B, T * R), axis=-1)
    k = max(1, min(int(top_pose_count), int(score_flat.shape[-1])))
    top_scores, top_flat = jax.lax.top_k(score_flat, k)
    diagnostics = PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=(best_flat % R).astype(jnp.int32),
        best_translation_idx=(best_flat // R).astype(jnp.int32),
        n_significant_per_image=jnp.sum(gamma > float(significance_threshold), axis=(1, 2)).astype(jnp.int32),
        best_log_score_per_image=jnp.max(score_flat, axis=-1).astype(jnp.float32),
        rotation_posterior_sums=jnp.sum(gamma, axis=(0, 1)).astype(jnp.float32),
        max_posterior_per_image=pmax,
        top_rotation_idx=(top_flat % R).astype(jnp.int32),
        top_translation_idx=(top_flat // R).astype(jnp.int32),
        top_log_score_per_image=top_scores.astype(jnp.float32),
        top_posterior_per_image=jnp.exp(top_scores - logZ[:, None]).astype(jnp.float32),
    )
    return LocalScoreAndMomentsStats(score=score, alpha=alpha, G_tri=G_tri, diagnostics=diagnostics)


@partial(
    jax.jit,
    static_argnames=(
        "disc_type_backproject",
        "use_recon_window",
        "backprojection_max_r",
        "image_shape",
        "volume_shape",
    ),
)
def accumulate_local_pose_ppca_bucket_cached(
    score,
    alpha,
    G_tri,
    logZ,
    rotations_bucket,
    image_shape,
    volume_shape,
    rhs_volume,
    lhs_tri_volume,
    Y1_recon,
    ctf2_over_noise_recon,
    *,
    disc_type_backproject: str = "linear_interp",
    recon_window_indices=None,
    use_recon_window: bool = False,
    backprojection_max_r=None,
):
    """Exact local PPCA M-step backprojection from cached score moments."""

    from recovar.em.dense_single_volume.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed

    score = jnp.asarray(score)
    alpha = jnp.asarray(alpha)
    G_tri = jnp.asarray(G_tri)
    logZ = jnp.asarray(logZ)
    rotations_bucket = jnp.asarray(rotations_bucket)
    rhs_volume = jnp.asarray(rhs_volume)
    lhs_tri_volume = jnp.asarray(lhs_tri_volume)
    Y1_recon = jnp.asarray(Y1_recon)
    ctf2_over_noise_recon = jnp.asarray(ctf2_over_noise_recon)
    B, T, R = score.shape
    gamma = jnp.exp(score - logZ[:, None, None])
    rhs_dtype = rhs_volume.dtype
    lhs_dtype = lhs_tri_volume.dtype
    flat_rotations = rotations_bucket.reshape(B * R, 3, 3)

    rhs_images = jnp.einsum(
        "btr,btrp,btf->pbrf",
        gamma.astype(rhs_dtype),
        jnp.conj(alpha).astype(rhs_dtype),
        Y1_recon.astype(rhs_dtype),
    ).reshape(rhs_volume.shape[0], B * R, Y1_recon.shape[-1])
    rhs_volume = batch_adjoint_slice_volume_maybe_windowed(
        rhs_images,
        recon_window_indices,
        flat_rotations,
        rhs_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )

    lhs_images = (
        jnp.einsum(
            "btr,btrk,bf->kbrf",
            gamma.astype(lhs_dtype),
            G_tri,
            ctf2_over_noise_recon.astype(lhs_dtype),
        )
        .real.astype(lhs_dtype)
        .reshape(lhs_tri_volume.shape[0], B * R, Y1_recon.shape[-1])
    )
    lhs_tri_volume = batch_adjoint_slice_volume_maybe_windowed(
        lhs_images,
        recon_window_indices,
        flat_rotations,
        lhs_tri_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )
    return rhs_volume, lhs_tri_volume, jnp.ones((B,), dtype=jnp.float32)


@partial(
    jax.jit,
    static_argnames=(
        "disc_type_backproject",
        "use_recon_window",
        "backprojection_max_r",
        "image_shape",
        "volume_shape",
        "top_k_mstep",
    ),
)
def accumulate_local_pose_ppca_bucket_topk_cached(
    score,
    alpha,
    G_tri,
    logZ,
    rotations_bucket,
    image_shape,
    volume_shape,
    rhs_volume,
    lhs_tri_volume,
    Y1_recon,
    ctf2_over_noise_recon,
    *,
    disc_type_backproject: str = "linear_interp",
    recon_window_indices=None,
    use_recon_window: bool = False,
    backprojection_max_r=None,
    top_k_mstep: int = 1,
):
    """Approximate local M-step that backprojects only the top-k posterior poses."""

    from recovar.em.dense_single_volume.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed

    score = jnp.asarray(score)
    alpha = jnp.asarray(alpha)
    G_tri = jnp.asarray(G_tri)
    logZ = jnp.asarray(logZ)
    rotations_bucket = jnp.asarray(rotations_bucket)
    rhs_volume = jnp.asarray(rhs_volume)
    lhs_tri_volume = jnp.asarray(lhs_tri_volume)
    Y1_recon = jnp.asarray(Y1_recon)
    ctf2_over_noise_recon = jnp.asarray(ctf2_over_noise_recon)

    B, T, R = score.shape
    P = alpha.shape[-1]
    tri = G_tri.shape[-1]
    F_recon = Y1_recon.shape[-1]
    k = max(1, min(int(top_k_mstep), int(T * R)))
    top_scores, top_flat = jax.lax.top_k(score.reshape(B, T * R), k)
    top_gamma = jnp.exp(top_scores - logZ[:, None])
    top_rot = (top_flat % R).astype(jnp.int32)
    top_trans = (top_flat // R).astype(jnp.int32)
    alpha_top = jnp.take_along_axis(alpha.reshape(B, T * R, P), top_flat[..., None], axis=1)
    G_top = jnp.take_along_axis(G_tri.reshape(B, T * R, tri), top_flat[..., None], axis=1)
    Y1_top = jnp.take_along_axis(Y1_recon, top_trans[..., None], axis=1)
    rotations_top = jnp.take_along_axis(rotations_bucket, top_rot[..., None, None], axis=1)
    ctf2_top = jnp.broadcast_to(ctf2_over_noise_recon[:, None, :], (B, k, F_recon))

    rhs_dtype = rhs_volume.dtype
    lhs_dtype = lhs_tri_volume.dtype
    rhs_images = jnp.einsum(
        "bk,bkp,bkf->pbkf",
        top_gamma.astype(rhs_dtype),
        jnp.conj(alpha_top).astype(rhs_dtype),
        Y1_top.astype(rhs_dtype),
    ).reshape(P, B * k, F_recon)
    rhs_volume = batch_adjoint_slice_volume_maybe_windowed(
        rhs_images,
        recon_window_indices,
        rotations_top.reshape(B * k, 3, 3),
        rhs_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )

    lhs_images = (
        jnp.einsum(
            "bk,bkc,bkf->cbkf",
            top_gamma.astype(lhs_dtype),
            G_top,
            ctf2_top.astype(lhs_dtype),
        )
        .real.astype(lhs_dtype)
        .reshape(tri, B * k, F_recon)
    )
    lhs_tri_volume = batch_adjoint_slice_volume_maybe_windowed(
        lhs_images,
        recon_window_indices,
        rotations_top.reshape(B * k, 3, 3),
        lhs_tri_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )
    retained_mass = jnp.sum(top_gamma, axis=1).astype(jnp.float32)
    return rhs_volume, lhs_tri_volume, retained_mass


@partial(
    jax.jit,
    static_argnames=(
        "significance_threshold",
        "disc_type_backproject",
        "use_recon_window",
        "backprojection_max_r",
        "image_shape",
        "volume_shape",
        "top_pose_count",
    ),
)
def fused_local_pose_ppca_bucket(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    rotations_bucket,
    image_shape,
    volume_shape,
    rhs_volume,
    lhs_tri_volume,
    pose_log_prior,
    Y1_recon,
    ctf2_over_noise_recon,
    *,
    significance_threshold: float = 1e-3,
    disc_type_backproject: str = "linear_interp",
    recon_window_indices=None,
    use_recon_window: bool = False,
    backprojection_max_r=None,
    top_pose_count: int = 1,
):
    """Fuse PPCA pass 2 for a padded exact-local bucket.

    Unlike the dense block kernel, each image row has its own local rotations.
    Backprojection therefore flattens ``[image, local_rotation]`` rows after
    posterior accumulation instead of summing over images before the adjoint.
    """

    from recovar.em.dense_single_volume.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed

    Y1 = jnp.asarray(Y1)
    proj_aug = jnp.asarray(proj_aug)
    ctf2_over_noise = jnp.asarray(ctf2_over_noise)
    y_norm = jnp.asarray(y_norm)
    rotations_bucket = jnp.asarray(rotations_bucket)
    rhs_volume = jnp.asarray(rhs_volume)
    lhs_tri_volume = jnp.asarray(lhs_tri_volume)
    Y1_recon = jnp.asarray(Y1_recon)
    ctf2_over_noise_recon = jnp.asarray(ctf2_over_noise_recon)

    B, T, F = Y1.shape
    proj_B, R, P, proj_F = proj_aug.shape
    if (proj_B, proj_F) != (B, F):
        raise ValueError(f"proj_aug shape {proj_aug.shape} is incompatible with Y1 shape {Y1.shape}")
    if rotations_bucket.shape != (B, R, 3, 3):
        raise ValueError(f"rotations_bucket shape {rotations_bucket.shape} != ({B}, {R}, 3, 3)")
    tri = _tri_size(P)
    if rhs_volume.ndim != 2 or rhs_volume.shape[0] != P:
        raise ValueError(f"rhs_volume must have shape [P={P}, half_vol], got {rhs_volume.shape}")
    if lhs_tri_volume.ndim != 2 or lhs_tri_volume.shape[0] != tri:
        raise ValueError(f"lhs_tri_volume must have shape [tri(P)={tri}, half_vol], got {lhs_tri_volume.shape}")
    if jnp.asarray(pose_log_prior).shape != (B, R, T):
        raise ValueError(f"pose_log_prior shape {jnp.asarray(pose_log_prior).shape} != ({B}, {R}, {T})")
    F_recon = int(Y1_recon.shape[-1])
    if Y1_recon.shape[:2] != (B, T):
        raise ValueError(f"Y1_recon leading shape {Y1_recon.shape[:2]} != ({B}, {T})")
    if ctf2_over_noise_recon.shape != (B, F_recon):
        raise ValueError(f"ctf2_over_noise_recon shape {ctf2_over_noise_recon.shape} != ({B}, {F_recon})")

    gamma, alpha, G_tri, diagnostics = _score_gamma_and_moments_local_bucket(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        pose_log_prior,
        significance_threshold,
        top_pose_count=top_pose_count,
    )
    rhs_dtype = rhs_volume.dtype
    lhs_dtype = lhs_tri_volume.dtype
    flat_rotations = rotations_bucket.reshape(B * R, 3, 3)

    rhs_images = jnp.einsum(
        "btr,btrp,btf->pbrf",
        gamma.astype(rhs_dtype),
        jnp.conj(alpha).astype(rhs_dtype),
        Y1_recon.astype(rhs_dtype),
    ).reshape(P, B * R, F_recon)
    rhs_volume = batch_adjoint_slice_volume_maybe_windowed(
        rhs_images,
        recon_window_indices,
        flat_rotations,
        rhs_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )

    lhs_images = (
        jnp.einsum(
            "btr,btrk,bf->kbrf",
            gamma.astype(lhs_dtype),
            G_tri,
            ctf2_over_noise_recon.astype(lhs_dtype),
        )
        .real.astype(lhs_dtype)
        .reshape(tri, B * R, F_recon)
    )
    lhs_tri_volume = batch_adjoint_slice_volume_maybe_windowed(
        lhs_images,
        recon_window_indices,
        flat_rotations,
        lhs_tri_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )
    return rhs_volume, lhs_tri_volume, diagnostics


def iter_local_ppca_dataset_blocks(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    local_layout: LocalHypothesisLayout | None = None,
    *,
    disc_type: str = "linear_interp",
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    class_log_prior: float = 0.0,
    image_scale_corrections: np.ndarray | None = None,
) -> Iterable[tuple[int, np.ndarray, DensePPCAFusedBlock]]:
    """Yield one exact-local PPCA block per image.

    The support is entirely defined by ``LocalHypothesisLayout``. Pruning is
    support-only: candidate masks add ``-inf`` to the same PPCA score algebra
    used by the dense path.
    """

    if local_layout is None:
        raise ValueError("local_layout is required")
    if noise_variance is None:
        raise ValueError("noise_variance is required")
    if int(local_layout.n_images) != int(getattr(experiment_dataset, "n_units", experiment_dataset.n_images)):
        raise ValueError(
            f"local_layout.n_images={local_layout.n_images} does not match dataset image count",
        )

    resolved = prepare_dense_ppca_dataset_inputs(
        experiment_dataset,
        mu,
        W,
        q=q,
        volume_domain=volume_domain,
        current_size=current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
    )
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    n_trans = int(local_layout.translation_grid.shape[0])

    for image_index in range(int(local_layout.n_images)):
        start = int(local_layout.rotation_offsets[image_index])
        end = int(local_layout.rotation_offsets[image_index + 1])
        if end <= start:
            continue
        rotations = np.asarray(local_layout.rotations_flat[start:end], dtype=np.float32)
        rotation_ids = np.asarray(local_layout.rotation_ids_flat[start:end], dtype=np.int32)
        batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, indices = _fetch_single_image_batch(
            experiment_dataset,
            image_index,
        )
        shifted_score_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            local_layout.translation_grid,
            config,
            score_with_masked_images=score_with_masked_images,
        )
        if score_with_masked_images:
            shifted_recon_half = prepare_reconstruction_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                local_layout.translation_grid,
                config,
            )
        else:
            shifted_recon_half = shifted_score_half

        F = int(shifted_score_half.shape[-1])
        if image_scale_corrections is None:
            image_scale = jnp.asarray(1.0, dtype=shifted_score_half.real.dtype)
        else:
            scale_arr = np.asarray(image_scale_corrections, dtype=np.float32)
            original_index = int(np.asarray(indices, dtype=np.int64).reshape(-1)[0])
            if scale_arr.shape[0] > original_index:
                image_scale = jnp.asarray(scale_arr[original_index], dtype=shifted_score_half.real.dtype)
            elif scale_arr.shape[0] > image_index:
                image_scale = jnp.asarray(scale_arr[image_index], dtype=shifted_score_half.real.dtype)
            else:
                raise ValueError(
                    f"image_scale_corrections has {scale_arr.shape[0]} entries but image indices "
                    f"{original_index} and {image_index} are out of range"
                )
        image_scale_sq = image_scale**2
        Y1_score = shifted_score_half.reshape(1, n_trans, F) * image_scale * resolved.score_mask[None, None, :]
        ctf2_score = ctf2_over_nv_half * image_scale_sq * resolved.score_mask[None, :]
        Y1_recon = shifted_recon_half.reshape(1, n_trans, F) * image_scale * resolved.recon_mask[None, None, :]
        ctf2_recon = ctf2_over_nv_half * image_scale_sq * resolved.recon_mask[None, :]
        proj_aug = _project_local_augmented(
            resolved.augmented_half_volumes,
            rotations,
            resolved.image_shape,
            resolved.volume_shape,
            disc_type,
            max_r=resolved.projection_max_r,
        )

        rotation_prior = np.asarray(local_layout.rotation_log_priors_flat[start:end], dtype=np.float32)
        translation_prior = _local_translation_log_prior(local_layout, image_index)
        pose_prior = rotation_prior[:, None] + translation_prior[None, :] + float(class_log_prior)
        if local_layout.sample_mask_flat is not None:
            sample_mask = np.asarray(local_layout.sample_mask_flat[start:end], dtype=bool)
            if sample_mask.shape != (end - start, n_trans):
                raise ValueError(f"sample_mask shape {sample_mask.shape} != ({end - start}, {n_trans})")
            pose_prior = np.where(sample_mask, pose_prior, -np.inf)

        yield (
            image_index,
            rotation_ids,
            DensePPCAFusedBlock(
                Y1=Y1_score,
                proj_aug=proj_aug,
                ctf2_over_noise=ctf2_score,
                y_norm=jnp.asarray(batch_norm).reshape(1),
                rotations=jnp.asarray(rotations),
                pose_log_prior=jnp.asarray(pose_prior[None, :, :], dtype=jnp.float32),
                Y1_recon=Y1_recon,
                ctf2_over_noise_recon=ctf2_recon,
            ),
        )


def iter_local_ppca_dataset_bucket_blocks(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    local_layout: LocalHypothesisLayout | None = None,
    *,
    disc_type: str = "linear_interp",
    image_batch_size: int = 2,
    rotation_block_size: int = 512,
    max_hypotheses_per_microbatch: int = 32768,
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    image_indices: np.ndarray | None = None,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    class_log_prior: float = 0.0,
    image_scale_corrections: np.ndarray | None = None,
) -> Iterable[LocalPPCAFusedBucketBlock]:
    """Yield padded exact-local PPCA buckets using K-class local bucketization."""

    if local_layout is None:
        raise ValueError("local_layout is required")
    if noise_variance is None:
        raise ValueError("noise_variance is required")

    original_image_indices = _resolve_local_image_indices(experiment_dataset, local_layout, image_indices)
    resolved = prepare_dense_ppca_dataset_inputs(
        experiment_dataset,
        mu,
        W,
        q=q,
        volume_domain=volume_domain,
        current_size=current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
    )
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    n_trans = int(local_layout.translation_grid.shape[0])
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=int(image_batch_size),
        rotation_block_size=int(rotation_block_size),
        max_hypotheses_per_microbatch=int(max_hypotheses_per_microbatch),
    )

    for bucket in bucket_specs:
        layout_rows = np.asarray(bucket.image_indices, dtype=np.int64)
        requested_indices = np.asarray(original_image_indices[layout_rows], dtype=np.int64)
        batch_data, ctf_params, returned_indices = fetch_indexed_batch(experiment_dataset, requested_indices)
        returned_indices = np.asarray(returned_indices, dtype=np.int64).reshape(-1)
        if returned_indices.shape != requested_indices.shape:
            raise RuntimeError(
                "Dataset returned a different number of local PPCA bucket images than requested; "
                f"requested {requested_indices.shape[0]}, got {returned_indices.shape[0]}",
            )
        batch_count = int(returned_indices.shape[0])
        bucket_rot_count = int(bucket.bucket_rotation_count)
        translations = np.asarray(local_layout.translation_grid, dtype=np.float32)
        shifted_score_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
            score_with_masked_images=score_with_masked_images,
        )
        if score_with_masked_images:
            shifted_recon_half = prepare_reconstruction_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
            )
        else:
            shifted_recon_half = shifted_score_half

        F = int(shifted_score_half.shape[-1])
        if image_scale_corrections is None:
            batch_scale = jnp.ones((batch_count,), dtype=shifted_score_half.real.dtype)
        else:
            scale_arr = np.asarray(image_scale_corrections, dtype=np.float32)
            if np.max(returned_indices, initial=-1) >= scale_arr.shape[0]:
                raise ValueError(
                    f"image_scale_corrections has {scale_arr.shape[0]} entries but bucket "
                    f"contains image index {int(np.max(returned_indices))}"
                )
            batch_scale = jnp.asarray(scale_arr[returned_indices], dtype=shifted_score_half.real.dtype)
        batch_scale_sq = batch_scale**2
        Y1_score_full = (
            shifted_score_half.reshape(batch_count, n_trans, F)
            * batch_scale[:, None, None]
            * resolved.score_mask[None, None, :]
        )
        ctf2_score_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.score_mask[None, :]
        Y1_recon_full = (
            shifted_recon_half.reshape(batch_count, n_trans, F)
            * batch_scale[:, None, None]
            * resolved.recon_mask[None, None, :]
        )
        ctf2_recon_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.recon_mask[None, :]
        if resolved.score_indices is None:
            Y1_score = Y1_score_full
            ctf2_score = ctf2_score_full
        else:
            Y1_score = Y1_score_full[:, :, resolved.score_indices]
            ctf2_score = ctf2_score_full[:, resolved.score_indices]
        if resolved.recon_indices is None:
            Y1_recon = Y1_recon_full
            ctf2_recon = ctf2_recon_full
        else:
            Y1_recon = Y1_recon_full[:, :, resolved.recon_indices]
            ctf2_recon = ctf2_recon_full[:, resolved.recon_indices]

        local_rotations = np.asarray(bucket.local_rotations, dtype=np.float32)[:batch_count, :bucket_rot_count]
        flat_rotations = local_rotations.reshape(batch_count * bucket_rot_count, 3, 3)
        proj_aug = _project_local_augmented(
            resolved.augmented_half_volumes,
            flat_rotations,
            resolved.image_shape,
            resolved.volume_shape,
            disc_type,
            max_r=resolved.projection_max_r,
        ).reshape(batch_count, bucket_rot_count, int(resolved.q) + 1, -1)
        if resolved.score_indices is not None:
            proj_aug = proj_aug[:, :, :, resolved.score_indices]

        pose_prior = (
            np.asarray(bucket.local_rotation_log_prior, dtype=np.float32)[:batch_count, :bucket_rot_count, None]
            + np.asarray(bucket.translation_log_prior, dtype=np.float32)[:batch_count, None, :]
            + float(class_log_prior)
        )
        local_mask = np.asarray(bucket.local_rotation_mask, dtype=bool)[:batch_count, :bucket_rot_count]
        pose_prior = np.where(local_mask[:, :, None], pose_prior, -np.inf)
        if bucket.local_sample_mask is not None:
            sample_mask = np.asarray(bucket.local_sample_mask, dtype=bool)[:batch_count, :bucket_rot_count, :]
            pose_prior = np.where(sample_mask, pose_prior, -np.inf)

        yield LocalPPCAFusedBucketBlock(
            Y1=Y1_score,
            proj_aug=proj_aug,
            ctf2_over_noise=ctf2_score,
            y_norm=jnp.asarray(batch_norm).reshape(batch_count),
            rotations=jnp.asarray(local_rotations),
            pose_log_prior=jnp.asarray(pose_prior, dtype=jnp.float32),
            Y1_recon=Y1_recon,
            ctf2_over_noise_recon=ctf2_recon,
            local_rotation_ids=np.asarray(bucket.local_rotation_ids, dtype=np.int32)[:batch_count, :bucket_rot_count],
            local_rotations=local_rotations,
            image_indices=returned_indices.astype(np.int64, copy=False),
            recon_window_indices=resolved.recon_indices,
            use_recon_window=bool(resolved.use_window),
            backprojection_max_r=resolved.backprojection_max_r,
        )


def _accumulate_local_ppca_fused_stats(
    experiment_dataset,
    mu,
    W,
    *,
    mean_prior,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    geometry: GeometryConfig,
    schedule: ScheduleConfig,
    scoring: ScoringConfig,
    mean_reg: MeanRegularizationConfig,
    disc_type: str,
    image_indices: np.ndarray | None,
    max_hypotheses_per_microbatch: int,
    image_batch_size: int,
    rotation_block_size: int,
    pose_selection: PoseSelectionConfig,
    class_log_prior: float,
    sparse_pass2: SparsePass2Config,
    q_resolved: int | None = None,
) -> LocalPPCAAccumulationResult:
    """Accumulate exact-local PPCA sufficient stats without solving the M-step."""

    current_size = geometry.current_size
    q = geometry.q
    volume_domain = geometry.volume_domain
    score_with_masked_images = scoring.score_with_masked_images
    half_spectrum_scoring = scoring.half_spectrum_scoring
    square_window = scoring.square_window
    image_scale_corrections = scoring.image_scale_corrections

    mean_prior = jnp.asarray(mean_prior)
    if q_resolved is None:
        W_arr = None if W is None else np.asarray(W)
        q_resolved = int(W_arr.shape[1]) if W_arr is not None and W_arr.ndim == 2 else int(q or 0)
    P = q_resolved + 1
    tri = _tri_size(P)
    rhs_volume = jnp.zeros((P, mean_prior.shape[0]), dtype=jnp.complex64)
    lhs_tri_volume = jnp.zeros((tri, mean_prior.shape[0]), dtype=jnp.float32)
    log_likelihood = 0.0
    n_images = 0
    pmax_values = []
    nsig_values = []
    best_local_values = []
    best_translation_values = []
    best_global_values = []
    best_rotation_matrices = []
    best_translations = []
    top_rotation_values = []
    top_global_values = []
    top_rotation_matrix_values = []
    top_translation_index_values = []
    top_log_score_values = []
    top_posterior_values = []
    output_image_indices = []
    postprocess_bandlimit_max_r = None
    local_topk_mstep = int(sparse_pass2.local_mstep_top_k) if bool(sparse_pass2.enabled) else 0
    local_topk_min_pmax = float(sparse_pass2.local_mstep_min_pmax)
    local_mstep_topk_buckets = 0
    local_mstep_exact_buckets = 0
    local_mstep_retained_mass_values = []

    for block in iter_local_ppca_dataset_bucket_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        local_layout,
        disc_type=disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        current_size=current_size,
        q=q,
        volume_domain=volume_domain,
        image_indices=image_indices,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        class_log_prior=class_log_prior,
        image_scale_corrections=image_scale_corrections,
    ):
        if postprocess_bandlimit_max_r is None and bool(block.use_recon_window):
            postprocess_bandlimit_max_r = block.backprojection_max_r
        candidate_count = int(block.rotations.shape[1]) * int(local_layout.translation_grid.shape[0])
        raw_top_pose_count = top_pose_candidate_count(pose_selection, candidate_count)
        if local_topk_mstep > 0:
            score_result = score_local_pose_ppca_bucket_with_moments(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
                top_pose_count=raw_top_pose_count,
            )
            posterior = score_result.diagnostics
            pmax_np = np.asarray(jax.block_until_ready(posterior.pmax), dtype=np.float32)
            use_topk_mstep = bool(pmax_np.size) and float(np.min(pmax_np)) >= local_topk_min_pmax
            if use_topk_mstep:
                rhs_volume, lhs_tri_volume, retained_mass = accumulate_local_pose_ppca_bucket_topk_cached(
                    score_result.score,
                    score_result.alpha,
                    score_result.G_tri,
                    posterior.logZ,
                    block.rotations,
                    tuple(int(x) for x in experiment_dataset.image_shape),
                    tuple(int(x) for x in experiment_dataset.volume_shape),
                    rhs_volume,
                    lhs_tri_volume,
                    block.Y1_recon,
                    block.ctf2_over_noise_recon,
                    disc_type_backproject=disc_type,
                    recon_window_indices=block.recon_window_indices,
                    use_recon_window=block.use_recon_window,
                    backprojection_max_r=block.backprojection_max_r,
                    top_k_mstep=min(local_topk_mstep, candidate_count),
                )
                local_mstep_topk_buckets += 1
            else:
                rhs_volume, lhs_tri_volume, retained_mass = accumulate_local_pose_ppca_bucket_cached(
                    score_result.score,
                    score_result.alpha,
                    score_result.G_tri,
                    posterior.logZ,
                    block.rotations,
                    tuple(int(x) for x in experiment_dataset.image_shape),
                    tuple(int(x) for x in experiment_dataset.volume_shape),
                    rhs_volume,
                    lhs_tri_volume,
                    block.Y1_recon,
                    block.ctf2_over_noise_recon,
                    disc_type_backproject=disc_type,
                    recon_window_indices=block.recon_window_indices,
                    use_recon_window=block.use_recon_window,
                    backprojection_max_r=block.backprojection_max_r,
                )
                local_mstep_exact_buckets += 1
            local_mstep_retained_mass_values.append(jnp.asarray(retained_mass, dtype=jnp.float32))
        else:
            rhs_volume, lhs_tri_volume, posterior = fused_local_pose_ppca_bucket(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.rotations,
                tuple(int(x) for x in experiment_dataset.image_shape),
                tuple(int(x) for x in experiment_dataset.volume_shape),
                rhs_volume,
                lhs_tri_volume,
                block.pose_log_prior,
                block.Y1_recon,
                block.ctf2_over_noise_recon,
                disc_type_backproject=disc_type,
                recon_window_indices=block.recon_window_indices,
                use_recon_window=block.use_recon_window,
                backprojection_max_r=block.backprojection_max_r,
                top_pose_count=raw_top_pose_count,
            )
            local_mstep_exact_buckets += 1
        log_likelihood += float(jnp.sum(posterior.logZ))
        n_images += int(posterior.logZ.shape[0])
        pmax_values.append(jnp.asarray(posterior.pmax))
        nsig_values.append(jnp.asarray(posterior.n_significant_per_image))
        best_local = np.asarray(jax.block_until_ready(posterior.best_rotation_idx), dtype=np.int64)
        best_trans = np.asarray(jax.block_until_ready(posterior.best_translation_idx), dtype=np.int64)
        local_rotation_ids = np.asarray(block.local_rotation_ids, dtype=np.int32)
        local_rotations = np.asarray(block.local_rotations, dtype=np.float32)
        image_rows = np.asarray(block.image_indices, dtype=np.int64)
        best_global = local_rotation_ids[np.arange(best_local.shape[0]), best_local]
        best_rot_mats = local_rotations[np.arange(best_local.shape[0]), best_local]
        translation_grid = np.asarray(local_layout.translation_grid, dtype=np.float32)
        best_trans_vecs = translation_grid[best_trans]
        raw_top_local = np.asarray(jax.block_until_ready(posterior.top_rotation_idx), dtype=np.int64)
        raw_top_trans_idx = np.asarray(jax.block_until_ready(posterior.top_translation_idx), dtype=np.int64)
        raw_top_global = np.take_along_axis(local_rotation_ids, raw_top_local, axis=1)
        raw_top_rot_mats = local_rotations[np.arange(raw_top_local.shape[0])[:, None], raw_top_local]
        top_selection = select_distinct_top_poses(
            np.asarray(jax.block_until_ready(posterior.top_log_score_per_image), dtype=np.float32),
            raw_top_global,
            raw_top_trans_idx,
            logZ=np.asarray(jax.block_until_ready(posterior.logZ), dtype=np.float64),
            rotations=np.asarray(local_layout.rotations_flat, dtype=np.float32)
            if int(local_layout.rotations_flat.shape[0]) == int(local_layout.n_global_rotations)
            else None,
            candidate_rotation_matrices=raw_top_rot_mats,
            translations=translation_grid,
            config=pose_selection,
        )
        top_global = top_selection.rotation_idx
        top_trans_idx = top_selection.translation_idx

        best_local_values.append(jnp.asarray(best_local, dtype=jnp.int32))
        best_translation_values.append(jnp.asarray(best_trans, dtype=jnp.int32))
        best_global_values.append(jnp.asarray(best_global, dtype=jnp.int32))
        best_rotation_matrices.append(jnp.asarray(best_rot_mats, dtype=jnp.float32))
        best_translations.append(jnp.asarray(best_trans_vecs, dtype=jnp.float32))
        top_local_values = np.full_like(top_global, -1, dtype=np.int32)
        for row in range(top_global.shape[0]):
            mapping = {int(rot_id): int(col) for col, rot_id in enumerate(local_rotation_ids[row]) if rot_id >= 0}
            for col in range(top_global.shape[1]):
                top_local_values[row, col] = mapping.get(int(top_global[row, col]), -1)
        top_local_values = np.where(top_global >= 0, top_local_values, -1)
        top_rot_mats = np.zeros(top_local_values.shape + (3, 3), dtype=np.float32)
        valid_top = top_local_values >= 0
        if np.any(valid_top):
            row_idx, col_idx = np.nonzero(valid_top)
            top_rot_mats[row_idx, col_idx] = local_rotations[row_idx, top_local_values[row_idx, col_idx]]
        top_rotation_values.append(jnp.asarray(top_local_values, dtype=jnp.int32))
        top_global_values.append(jnp.asarray(top_global, dtype=jnp.int32))
        top_rotation_matrix_values.append(jnp.asarray(top_rot_mats, dtype=jnp.float32))
        top_translation_index_values.append(jnp.asarray(top_trans_idx, dtype=jnp.int32))
        top_log_score_values.append(jnp.asarray(top_selection.log_score, dtype=jnp.float32))
        top_posterior_values.append(jnp.asarray(top_selection.posterior, dtype=jnp.float32))
        output_image_indices.append(jnp.asarray(image_rows, dtype=jnp.int32))

    image_scale_min, image_scale_max = resolve_image_scale_range(image_scale_corrections, image_indices)
    diagnostics = build_iteration_diagnostics(
        pmax_values=pmax_values,
        nsig_values=nsig_values,
        best_rotations=best_local_values,
        best_translations=best_translation_values,
        top_rotations=top_rotation_values,
        top_rotation_ids=top_global_values,
        top_translations=top_translation_index_values,
        top_log_scores=top_log_score_values,
        top_posteriors=top_posterior_values,
        log_likelihood=log_likelihood,
        n_images=n_images,
        mean_reg=mean_reg,
        image_scale_min=image_scale_min,
        image_scale_max=image_scale_max,
        image_scale_corrections=image_scale_corrections,
        extras={
            "best_rotation_id": jnp.concatenate(best_global_values)
            if best_global_values
            else jnp.zeros((0,), dtype=jnp.int32),
            "best_rotation_matrix": jnp.concatenate(best_rotation_matrices)
            if best_rotation_matrices
            else jnp.zeros((0, 3, 3), dtype=jnp.float32),
            "top_rotation_matrix": jnp.concatenate(top_rotation_matrix_values)
            if top_rotation_matrix_values
            else jnp.zeros((0, int(pose_selection.top_p_poses), 3, 3), dtype=jnp.float32),
            "best_translation": jnp.concatenate(best_translations)
            if best_translations
            else jnp.zeros((0, 2), dtype=jnp.float32),
            "image_indices": jnp.concatenate(output_image_indices)
            if output_image_indices
            else jnp.zeros((0,), dtype=jnp.int32),
            "local_bucketed": True,
            "local_image_batch_size": int(image_batch_size),
            "local_rotation_block_size": int(rotation_block_size),
            "local_max_hypotheses_per_microbatch": int(max_hypotheses_per_microbatch),
            "top_p_poses": int(pose_selection.top_p_poses),
            "top_pose_max_log_score_gap": float(pose_selection.top_pose_max_log_score_gap),
            "top_pose_min_angle_deg": float(pose_selection.top_pose_min_angle_deg),
            "top_pose_min_translation_px": float(pose_selection.top_pose_min_translation_px),
            "local_mstep_top_k": int(local_topk_mstep),
            "local_mstep_topk_min_pmax": float(local_topk_min_pmax),
            "local_mstep_topk_buckets": int(local_mstep_topk_buckets),
            "local_mstep_exact_buckets": int(local_mstep_exact_buckets),
            "local_mstep_topk_bucket_fraction": float(
                local_mstep_topk_buckets / max(1, local_mstep_topk_buckets + local_mstep_exact_buckets)
            ),
            "local_mstep_retained_mass_per_image": (
                jnp.concatenate(local_mstep_retained_mass_values)
                if local_mstep_retained_mass_values
                else jnp.ones((n_images,), dtype=jnp.float32)
            ),
            "local_mstep_retained_mass_mean": (
                float(jnp.mean(jnp.concatenate(local_mstep_retained_mass_values)))
                if local_mstep_retained_mass_values
                else 1.0
            ),
            "local_mstep_omitted_mass_max": (
                float(jnp.max(1.0 - jnp.concatenate(local_mstep_retained_mass_values)))
                if local_mstep_retained_mass_values
                else 0.0
            ),
        },
    )
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
        diagnostics=diagnostics,
    )
    return LocalPPCAAccumulationResult(stats=stats, postprocess_bandlimit_max_r=postprocess_bandlimit_max_r)


def _score_local_ppca_pose_diagnostics(
    experiment_dataset,
    mu,
    W,
    *,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    geometry: GeometryConfig,
    schedule: ScheduleConfig,
    scoring: ScoringConfig,
    mean_reg: MeanRegularizationConfig,
    disc_type: str,
    image_indices: np.ndarray | None,
    max_hypotheses_per_microbatch: int,
    image_batch_size: int,
    rotation_block_size: int,
    pose_selection: PoseSelectionConfig,
    class_log_prior: float,
) -> LocalPPCAPoseScoringResult:
    """Collect exact-local PPCA pose diagnostics without forming M-step stats."""

    current_size = geometry.current_size
    q = geometry.q
    volume_domain = geometry.volume_domain
    score_with_masked_images = scoring.score_with_masked_images
    half_spectrum_scoring = scoring.half_spectrum_scoring
    square_window = scoring.square_window
    image_scale_corrections = scoring.image_scale_corrections

    log_likelihood = 0.0
    n_images = 0
    pmax_values = []
    nsig_values = []
    best_local_values = []
    best_translation_values = []
    best_global_values = []
    best_rotation_matrices = []
    best_translations = []
    top_rotation_values = []
    top_global_values = []
    top_rotation_matrix_values = []
    top_translation_index_values = []
    top_log_score_values = []
    top_posterior_values = []
    output_image_indices = []

    for block in iter_local_ppca_dataset_bucket_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        local_layout,
        disc_type=disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        current_size=current_size,
        q=q,
        volume_domain=volume_domain,
        image_indices=image_indices,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        class_log_prior=class_log_prior,
        image_scale_corrections=image_scale_corrections,
    ):
        candidate_count = int(block.rotations.shape[1]) * int(local_layout.translation_grid.shape[0])
        score_top_k = top_pose_candidate_count(pose_selection, candidate_count)
        # ``RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE`` opt-in enables R-chunked scoring
        # for the pose-only path: each bucket's R dim is tiled into chunks of
        # the requested size. Cuts per-bucket peak working memory by
        # ``R / chunk`` so users can raise ``--image-batch-size`` past 1 on
        # large per-image neighborhoods (HP6 top-p at box 256² needs
        # ~1700-2900 rotations/image; full bucket peaks 3-5x proj_aug).
        # See docs/perf/ppca_local_hp6_topp_runtime.md for the design.
        r_chunk_env = os.environ.get("RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE", "")
        r_chunk = int(r_chunk_env) if r_chunk_env.strip().lstrip("-").isdigit() else 0
        if r_chunk > 0 and r_chunk < int(block.proj_aug.shape[1]):
            posterior = _score_local_pose_ppca_bucket_rotation_chunked(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
                significance_threshold=1e-3,
                top_pose_count=score_top_k,
                rotation_chunk_size=r_chunk,
            )
        else:
            posterior = score_local_pose_ppca_bucket(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
                top_pose_count=score_top_k,
            )
        log_likelihood += float(jnp.sum(posterior.logZ))
        n_images += int(posterior.logZ.shape[0])
        pmax_values.append(jnp.asarray(posterior.pmax))
        nsig_values.append(jnp.asarray(posterior.n_significant_per_image))

        best_local = np.asarray(jax.block_until_ready(posterior.best_rotation_idx), dtype=np.int64)
        best_trans = np.asarray(jax.block_until_ready(posterior.best_translation_idx), dtype=np.int64)
        local_rotation_ids = np.asarray(block.local_rotation_ids, dtype=np.int32)
        local_rotations = np.asarray(block.local_rotations, dtype=np.float32)
        image_rows = np.asarray(block.image_indices, dtype=np.int64)
        best_global = local_rotation_ids[np.arange(best_local.shape[0]), best_local]
        best_rot_mats = local_rotations[np.arange(best_local.shape[0]), best_local]
        translation_grid = np.asarray(local_layout.translation_grid, dtype=np.float32)
        best_trans_vecs = translation_grid[best_trans]

        raw_top_local = np.asarray(jax.block_until_ready(posterior.top_rotation_idx), dtype=np.int64)
        raw_top_trans_idx = np.asarray(jax.block_until_ready(posterior.top_translation_idx), dtype=np.int64)
        raw_top_global = np.take_along_axis(local_rotation_ids, raw_top_local, axis=1)
        raw_top_rot_mats = local_rotations[np.arange(raw_top_local.shape[0])[:, None], raw_top_local]
        top_selection = select_distinct_top_poses(
            np.asarray(jax.block_until_ready(posterior.top_log_score_per_image), dtype=np.float32),
            raw_top_global,
            raw_top_trans_idx,
            logZ=np.asarray(jax.block_until_ready(posterior.logZ), dtype=np.float64),
            rotations=np.asarray(local_layout.rotations_flat, dtype=np.float32)
            if int(local_layout.rotations_flat.shape[0]) == int(local_layout.n_global_rotations)
            else None,
            candidate_rotation_matrices=raw_top_rot_mats,
            translations=translation_grid,
            config=pose_selection,
        )
        top_global = top_selection.rotation_idx
        top_trans_idx = top_selection.translation_idx
        top_local_values = np.full_like(top_global, -1, dtype=np.int32)
        for row in range(top_global.shape[0]):
            mapping = {int(rot_id): int(col) for col, rot_id in enumerate(local_rotation_ids[row]) if rot_id >= 0}
            for col in range(top_global.shape[1]):
                top_local_values[row, col] = mapping.get(int(top_global[row, col]), -1)
        top_local_values = np.where(top_global >= 0, top_local_values, -1)
        top_rot_mats = np.zeros(top_local_values.shape + (3, 3), dtype=np.float32)
        valid_top = top_local_values >= 0
        if np.any(valid_top):
            row_idx, col_idx = np.nonzero(valid_top)
            top_rot_mats[row_idx, col_idx] = local_rotations[row_idx, top_local_values[row_idx, col_idx]]

        best_local_values.append(jnp.asarray(best_local, dtype=jnp.int32))
        best_translation_values.append(jnp.asarray(best_trans, dtype=jnp.int32))
        best_global_values.append(jnp.asarray(best_global, dtype=jnp.int32))
        best_rotation_matrices.append(jnp.asarray(best_rot_mats, dtype=jnp.float32))
        best_translations.append(jnp.asarray(best_trans_vecs, dtype=jnp.float32))
        top_rotation_values.append(jnp.asarray(top_local_values, dtype=jnp.int32))
        top_global_values.append(jnp.asarray(top_global, dtype=jnp.int32))
        top_rotation_matrix_values.append(jnp.asarray(top_rot_mats, dtype=jnp.float32))
        top_translation_index_values.append(jnp.asarray(top_trans_idx, dtype=jnp.int32))
        top_log_score_values.append(jnp.asarray(top_selection.log_score, dtype=jnp.float32))
        top_posterior_values.append(jnp.asarray(top_selection.posterior, dtype=jnp.float32))
        output_image_indices.append(jnp.asarray(image_rows, dtype=jnp.int32))

    image_scale_min, image_scale_max = resolve_image_scale_range(image_scale_corrections, image_indices)
    diagnostics = build_iteration_diagnostics(
        pmax_values=pmax_values,
        nsig_values=nsig_values,
        best_rotations=best_local_values,
        best_translations=best_translation_values,
        top_rotations=top_rotation_values,
        top_rotation_ids=top_global_values,
        top_translations=top_translation_index_values,
        top_log_scores=top_log_score_values,
        top_posteriors=top_posterior_values,
        log_likelihood=log_likelihood,
        n_images=n_images,
        mean_reg=mean_reg,
        image_scale_min=image_scale_min,
        image_scale_max=image_scale_max,
        image_scale_corrections=image_scale_corrections,
        extras={
            "best_rotation_id": jnp.concatenate(best_global_values)
            if best_global_values
            else jnp.zeros((0,), dtype=jnp.int32),
            "best_rotation_matrix": jnp.concatenate(best_rotation_matrices)
            if best_rotation_matrices
            else jnp.zeros((0, 3, 3), dtype=jnp.float32),
            "top_rotation_matrix": jnp.concatenate(top_rotation_matrix_values)
            if top_rotation_matrix_values
            else jnp.zeros((0, int(pose_selection.top_p_poses), 3, 3), dtype=jnp.float32),
            "best_translation": jnp.concatenate(best_translations)
            if best_translations
            else jnp.zeros((0, 2), dtype=jnp.float32),
            "image_indices": jnp.concatenate(output_image_indices)
            if output_image_indices
            else jnp.zeros((0,), dtype=jnp.int32),
            "local_bucketed": True,
            "local_pose_score_only": True,
            "local_image_batch_size": int(image_batch_size),
            "local_rotation_block_size": int(rotation_block_size),
            "local_max_hypotheses_per_microbatch": int(max_hypotheses_per_microbatch),
            "top_p_poses": int(pose_selection.top_p_poses),
            "top_pose_max_log_score_gap": float(pose_selection.top_pose_max_log_score_gap),
            "top_pose_min_angle_deg": float(pose_selection.top_pose_min_angle_deg),
            "top_pose_min_translation_px": float(pose_selection.top_pose_min_translation_px),
            "mstep_mode": "pose_only_no_mstep",
        },
    )
    return LocalPPCAPoseScoringResult(diagnostics=diagnostics)


def _merge_local_ppca_accumulations(results: list[LocalPPCAAccumulationResult]) -> LocalPPCAAccumulationResult:
    """Sum additive PPCA stats and concatenate per-image diagnostics."""

    if not results:
        raise ValueError("Cannot merge zero PPCA accumulation shards")
    rhs = jnp.asarray(results[0].stats.rhs)
    lhs_tri = jnp.asarray(results[0].stats.lhs_tri)
    residual_num = results[0].stats.residual_num
    residual_den = results[0].stats.residual_den
    log_likelihood = float(results[0].stats.log_likelihood)
    n_images = int(results[0].stats.n_images)
    for result in results[1:]:
        rhs = rhs + jnp.asarray(result.stats.rhs)
        lhs_tri = lhs_tri + jnp.asarray(result.stats.lhs_tri)
        if residual_num is not None or result.stats.residual_num is not None:
            residual_num = (
                jnp.asarray(0.0 if residual_num is None else residual_num)
                + jnp.asarray(0.0 if result.stats.residual_num is None else result.stats.residual_num)
            )
        if residual_den is not None or result.stats.residual_den is not None:
            residual_den = (
                jnp.asarray(0.0 if residual_den is None else residual_den)
                + jnp.asarray(0.0 if result.stats.residual_den is None else result.stats.residual_den)
            )
        log_likelihood += float(result.stats.log_likelihood)
        n_images += int(result.stats.n_images)

    diagnostics = dict(results[0].stats.diagnostics)
    concat_keys = (
        "max_posterior_per_image",
        "n_significant_per_image",
        "best_rotation_idx",
        "best_translation_idx",
        "top_rotation_idx",
        "top_rotation_id",
        "top_translation_idx",
        "top_log_score",
        "top_log_score_per_image",
        "top_posterior",
        "top_posterior_per_image",
        "best_rotation_id",
        "best_rotation_matrix",
        "top_rotation_matrix",
        "best_translation",
        "image_indices",
        "local_mstep_retained_mass_per_image",
    )
    for key in concat_keys:
        parts = [jnp.asarray(result.stats.diagnostics[key]) for result in results if key in result.stats.diagnostics]
        if parts:
            diagnostics[key] = jnp.concatenate(parts, axis=0)

    pmax = diagnostics.get("max_posterior_per_image")
    nsig = diagnostics.get("n_significant_per_image")
    diagnostics["pmax_mean"] = float(jnp.mean(jnp.asarray(pmax))) if pmax is not None and pmax.size else float("nan")
    diagnostics["nsig_mean"] = float(jnp.mean(jnp.asarray(nsig))) if nsig is not None and nsig.size else float("nan")
    diagnostics["log_likelihood"] = float(log_likelihood)
    diagnostics["logZ_mean"] = float(log_likelihood / n_images) if n_images else float("nan")
    diagnostics["uses_image_scale_corrections"] = any(
        bool(result.stats.diagnostics.get("uses_image_scale_corrections", False)) for result in results
    )
    diagnostics["image_scale_min"] = float(
        np.nanmin([float(result.stats.diagnostics.get("image_scale_min", np.nan)) for result in results])
    )
    diagnostics["image_scale_max"] = float(
        np.nanmax([float(result.stats.diagnostics.get("image_scale_max", np.nan)) for result in results])
    )
    diagnostics["local_image_sharded"] = True
    diagnostics["local_image_shard_count"] = int(len(results))
    diagnostics["local_mstep_topk_buckets"] = int(
        sum(int(result.stats.diagnostics.get("local_mstep_topk_buckets", 0)) for result in results)
    )
    diagnostics["local_mstep_exact_buckets"] = int(
        sum(int(result.stats.diagnostics.get("local_mstep_exact_buckets", 0)) for result in results)
    )
    total_mstep_buckets = diagnostics["local_mstep_topk_buckets"] + diagnostics["local_mstep_exact_buckets"]
    diagnostics["local_mstep_topk_bucket_fraction"] = float(
        diagnostics["local_mstep_topk_buckets"] / total_mstep_buckets
    ) if total_mstep_buckets else 0.0
    retained_mass = diagnostics.get("local_mstep_retained_mass_per_image")
    if retained_mass is not None and retained_mass.size:
        retained_mass = jnp.asarray(retained_mass, dtype=jnp.float32)
        diagnostics["local_mstep_retained_mass_mean"] = float(jnp.mean(retained_mass))
        diagnostics["local_mstep_omitted_mass_max"] = float(jnp.max(1.0 - retained_mass))
    if "top_rotation_id" in diagnostics or "top_rotation_idx" in diagnostics:
        top_width_source = diagnostics.get("top_rotation_id", diagnostics.get("top_rotation_idx"))
        diagnostics["top_p_poses"] = int(jnp.asarray(top_width_source).shape[-1])
        diagnostics["top_pose_count"] = int(jnp.asarray(top_width_source).shape[-1])

    stats = AugmentedPPCAStats(
        rhs=rhs,
        lhs_tri=lhs_tri,
        residual_num=residual_num,
        residual_den=residual_den,
        log_likelihood=log_likelihood,
        n_images=n_images,
        diagnostics=diagnostics,
    )
    postprocess_bandlimit_max_r = next(
        (result.postprocess_bandlimit_max_r for result in results if result.postprocess_bandlimit_max_r is not None),
        None,
    )
    return LocalPPCAAccumulationResult(stats=stats, postprocess_bandlimit_max_r=postprocess_bandlimit_max_r)


def _accumulate_local_ppca_fused_stats_sharded(
    experiment_dataset,
    mu,
    W,
    *,
    mean_prior,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    geometry: GeometryConfig,
    schedule: ScheduleConfig,
    scoring: ScoringConfig,
    mean_reg: MeanRegularizationConfig,
    disc_type: str,
    image_indices: np.ndarray | None,
    max_hypotheses_per_microbatch: int,
    image_batch_size: int,
    rotation_block_size: int,
    pose_selection: PoseSelectionConfig,
    class_log_prior: float,
    shard_count: int,
    sparse_pass2: SparsePass2Config,
    q_resolved: int | None = None,
) -> LocalPPCAAccumulationResult:
    """Image-shard local PPCA accumulation and reduce stats before one M-step."""

    shard_count = int(shard_count)
    if shard_count <= 1 or int(local_layout.n_images) <= 1:
        return _accumulate_local_ppca_fused_stats(
            experiment_dataset,
            mu,
            W,
            mean_prior=mean_prior,
            noise_variance=noise_variance,
            local_layout=local_layout,
            geometry=geometry,
            schedule=schedule,
            scoring=scoring,
            mean_reg=mean_reg,
            disc_type=disc_type,
            image_indices=image_indices,
            max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            pose_selection=pose_selection,
            class_log_prior=class_log_prior,
            sparse_pass2=sparse_pass2,
            q_resolved=q_resolved,
        )
    original_image_indices = _resolve_local_image_indices(experiment_dataset, local_layout, image_indices)
    rows = np.arange(int(local_layout.n_images), dtype=np.int64)
    row_shards = [part for part in np.array_split(rows, min(shard_count, rows.size)) if part.size]
    results = []
    for row_shard in row_shards:
        shard_layout = _slice_local_hypothesis_layout(local_layout, row_shard)
        shard_image_indices = np.asarray(original_image_indices[row_shard], dtype=np.int64)
        results.append(
            _accumulate_local_ppca_fused_stats(
                experiment_dataset,
                mu,
                W,
                mean_prior=mean_prior,
                noise_variance=noise_variance,
                local_layout=shard_layout,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                mean_reg=mean_reg,
                disc_type=disc_type,
                image_indices=shard_image_indices,
                max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
                image_batch_size=image_batch_size,
                rotation_block_size=rotation_block_size,
                pose_selection=pose_selection,
                class_log_prior=class_log_prior,
                sparse_pass2=sparse_pass2,
                q_resolved=q_resolved,
            )
        )
    merged = _merge_local_ppca_accumulations(results)
    merged.stats.diagnostics["local_requested_image_shard_count"] = shard_count
    return merged


def run_local_ppca_fused_em_iteration(
    experiment_dataset,
    mu,
    W=None,
    *,
    mean_prior,
    W_prior,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    sparse_pass2: SparsePass2Config | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    disc_type: str = "linear_interp",
    enforce_x0: bool = True,
    image_indices: np.ndarray | None = None,
    max_hypotheses_per_microbatch: int = 32768,
    fixed_mean_half=None,
    top_pose_count: int = 1,
    pose_selection: PoseSelectionConfig | None = None,
):
    """Run one exact-local PPCA EM update over a ``LocalHypothesisLayout``."""
    geometry = geometry if geometry is not None else GeometryConfig()
    # Local PPCA's default batch is intentionally smaller than dense (2 vs 500).
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
    sparse_pass2 = sparse_pass2 if sparse_pass2 is not None else SparsePass2Config(enabled=False)
    # Hoist into locals so the rest of the body reads cleanly.
    current_size = geometry.current_size
    q = geometry.q
    volume_domain = geometry.volume_domain
    score_with_masked_images = scoring.score_with_masked_images
    half_spectrum_scoring = scoring.half_spectrum_scoring
    square_window = scoring.square_window
    class_log_prior = scoring.class_log_prior
    image_scale_corrections = scoring.image_scale_corrections
    mstep_chunk_size = schedule.mstep_chunk_size
    image_batch_size = schedule.image_batch_size
    rotation_block_size = schedule.rotation_block_size
    local_image_shard_count = getattr(schedule, "local_image_shard_count", 1)
    pose_selection = (
        pose_selection
        if pose_selection is not None
        else PoseSelectionConfig(top_p_poses=max(1, int(top_pose_count)))
    )
    top_pose_count = int(pose_selection.top_p_poses)
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()

    q_resolved = int(jnp.asarray(W_prior).shape[1])
    mean_prior = jnp.asarray(mean_prior)
    W_prior = jnp.asarray(W_prior)
    if W_prior.shape != (mean_prior.shape[0], q_resolved):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({mean_prior.shape[0]}, {q_resolved})")

    # Smart-size the microbatch and image batch when the caller is at the
    # legacy library defaults (max_hypotheses=32768, image_batch_size=2),
    # mirroring the K-class engine's ``EXACT_LOCAL_TARGET_ROW_PIXELS`` rule.
    # User-specified non-default values are preserved.
    # Estimate the score-window pixel count: half-spectrum disk of radius
    # current_size/2, area ≈ pi * current_size^2 / 8. Falls back to a
    # conservative box-128 default when current_size is None.
    _box_for_window = int(current_size) if current_size else 64
    n_windowed_estimate = max(64, int(np.pi * (_box_for_window**2) / 8))
    if int(max_hypotheses_per_microbatch) == 32768:
        max_hypotheses_per_microbatch = _ppca_local_smart_max_hypotheses_per_microbatch(
            None, n_windowed_estimate, q_resolved
        )
    if int(image_batch_size) == 2:
        mean_bucket = (
            int(np.median(np.asarray(local_layout.rotation_counts, dtype=np.int64))) if local_layout.n_images else 256
        )
        image_batch_size = _ppca_local_smart_image_batch_size(
            None,
            max_hypotheses_per_microbatch=int(max_hypotheses_per_microbatch),
            mean_bucket_size=int(mean_bucket),
        )

    accumulation = _accumulate_local_ppca_fused_stats_sharded(
        experiment_dataset,
        mu,
        W,
        mean_prior=mean_prior,
        noise_variance=noise_variance,
        local_layout=local_layout,
        geometry=geometry,
        schedule=schedule,
        scoring=scoring,
        mean_reg=mean_reg,
        disc_type=disc_type,
        image_indices=image_indices,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        pose_selection=pose_selection,
        class_log_prior=class_log_prior,
        shard_count=local_image_shard_count,
        sparse_pass2=sparse_pass2,
        q_resolved=q_resolved,
    )
    stats = accumulation.stats
    diagnostics = dict(stats.diagnostics)
    if enforce_x0:
        rhs_volume = jnp.swapaxes(stats.rhs, 0, 1)
        lhs_tri_volume = jnp.swapaxes(stats.lhs_tri, 0, 1)
        rhs_volume = _enforce_augmented_x0(rhs_volume, tuple(int(x) for x in experiment_dataset.volume_shape))
        lhs_tri_volume = _enforce_augmented_x0(
            lhs_tri_volume.astype(jnp.complex64),
            tuple(int(x) for x in experiment_dataset.volume_shape),
        ).real.astype(jnp.float32)
        stats = AugmentedPPCAStats(
            rhs=jnp.swapaxes(rhs_volume, 0, 1),
            lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
            log_likelihood=stats.log_likelihood,
            n_images=stats.n_images,
            diagnostics=diagnostics,
        )
    mean_precision = resolve_mean_precision(
        stats, mean_prior, tuple(int(x) for x in experiment_dataset.volume_shape), mean_reg
    )

    mu_half, W_half = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        fixed_mean=fixed_mean_half,
        chunk_size=mstep_chunk_size,
    )
    solved_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    postprocessed = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        tuple(int(x) for x in experiment_dataset.volume_shape),
        config=dataclasses.replace(postprocess, bandlimit_max_r=accumulation.postprocess_bandlimit_max_r),
    )
    diagnostics.update(postprocessed.diagnostics)
    mu_half, W_half = postprocessed.mu_half, postprocessed.W_half
    diagnostics["mean_frozen"] = fixed_mean_half is not None
    diagnostics["mstep_mode"] = "fixed_mean_conditional_W" if fixed_mean_half is not None else "joint_mu_W"
    if fixed_mean_half is not None:
        mu_half = jnp.asarray(fixed_mean_half)
    output_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    diagnostics.update(solved_objective.diagnostics("mstep_objective_solved", n_images=stats.n_images))
    diagnostics.update(output_objective.diagnostics("mstep_objective_output", n_images=stats.n_images))
    diagnostics["mstep_objective_postprocess_delta"] = float(output_objective.total - solved_objective.total)
    diagnostics["mstep_objective_postprocess_delta_per_image"] = (
        float((output_objective.total - solved_objective.total) / stats.n_images) if stats.n_images else float("nan")
    )
    diagnostics["mstep_objective_scope"] = "fixed_e_step_augmented_quadratic_without_constants"
    diagnostics["mstep_objective_postprocess_in_objective"] = False
    return DensePPCAFusedEMResult(mu_half=mu_half, W_half=W_half, stats=stats, diagnostics=diagnostics)


def run_local_ppca_pose_scoring_iteration(
    experiment_dataset,
    mu,
    W=None,
    *,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    disc_type: str = "linear_interp",
    image_indices: np.ndarray | None = None,
    max_hypotheses_per_microbatch: int = 32768,
    top_pose_count: int = 1,
    pose_selection: PoseSelectionConfig | None = None,
) -> LocalPPCAPoseScoringResult:
    """Score exact-local PPCA poses and return diagnostics without updating ``mu`` or ``W``."""

    geometry = geometry if geometry is not None else GeometryConfig()
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    pose_selection = (
        pose_selection
        if pose_selection is not None
        else PoseSelectionConfig(top_p_poses=max(1, int(top_pose_count)))
    )

    W_arr = None if W is None else np.asarray(W)
    q_resolved = int(W_arr.shape[1]) if W_arr is not None and W_arr.ndim == 2 else int(geometry.q or 0)
    image_batch_size = int(schedule.image_batch_size)
    rotation_block_size = int(schedule.rotation_block_size)
    _box_for_window = int(geometry.current_size) if geometry.current_size else 64
    n_windowed_estimate = max(64, int(np.pi * (_box_for_window**2) / 8))
    if int(max_hypotheses_per_microbatch) == 32768:
        max_hypotheses_per_microbatch = _ppca_local_smart_max_hypotheses_per_microbatch(
            None, n_windowed_estimate, q_resolved
        )
    if int(image_batch_size) == 2:
        mean_bucket = (
            int(np.median(np.asarray(local_layout.rotation_counts, dtype=np.int64))) if local_layout.n_images else 256
        )
        image_batch_size = _ppca_local_smart_image_batch_size(
            None,
            max_hypotheses_per_microbatch=int(max_hypotheses_per_microbatch),
            mean_bucket_size=int(mean_bucket),
        )

    return _score_local_ppca_pose_diagnostics(
        experiment_dataset,
        mu,
        W,
        noise_variance=noise_variance,
        local_layout=local_layout,
        geometry=geometry,
        schedule=schedule,
        scoring=scoring,
        mean_reg=mean_reg,
        disc_type=disc_type,
        image_indices=image_indices,
        max_hypotheses_per_microbatch=int(max_hypotheses_per_microbatch),
        image_batch_size=int(image_batch_size),
        rotation_block_size=int(rotation_block_size),
        pose_selection=pose_selection,
        class_log_prior=float(scoring.class_log_prior),
    )


def run_local_ppca_halfset_pose_scoring_iteration(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    halfset_local_layouts,
    *,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    pose_selection: PoseSelectionConfig | None = None,
    top_pose_count: int = 1,
    disc_type: str = "linear_interp",
) -> PoseMarginalPPCAEMState:
    """Run a two-halfset exact-local PPCA pose-only pass.

    This is the high-resolution warmup/probe stage: it updates only
    ``pose_diagnostics`` so the following EM iteration can build local supports
    around PPCA-selected candidates without contaminating the probe with an
    M-step update.
    """

    if len(halfset_datasets) != 2 or len(halfset_local_layouts) != 2:
        raise ValueError("halfset_datasets and halfset_local_layouts must each have length 2")
    geometry = geometry if geometry is not None else GeometryConfig(volume_domain="fourier_half")
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    results = []
    for half_dataset, half_layout in zip(halfset_datasets, halfset_local_layouts, strict=True):
        results.append(
            run_local_ppca_pose_scoring_iteration(
                half_dataset,
                state.mu_score,
                state.W_score,
                noise_variance=state.noise_variance,
                local_layout=half_layout,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                mean_reg=mean_reg,
                pose_selection=pose_selection,
                top_pose_count=top_pose_count,
                disc_type=disc_type,
            )
        )
    pose_diagnostics = {
        "halfset0": results[0].diagnostics,
        "halfset1": results[1].diagnostics,
        "delta_rms_mu": 0.0,
        "delta_rms_W": 0.0,
        "pose_score_only": True,
    }
    return state.replace(pose_diagnostics=pose_diagnostics)


def run_local_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    halfset_local_layouts,
    *,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    sparse_pass2: SparsePass2Config | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    pose_selection: PoseSelectionConfig | None = None,
    top_pose_count: int = 1,
    disc_type: str = "linear_interp",
) -> PoseMarginalPPCAEMState:
    """Run one exact-local PPCA iteration for two halfsets."""

    from recovar.em.ppca_refinement.dense_dataset import combine_halfset_scoring_model

    if len(halfset_datasets) != 2 or len(halfset_local_layouts) != 2:
        raise ValueError("halfset_datasets and halfset_local_layouts must each have length 2")
    geometry = geometry if geometry is not None else GeometryConfig(volume_domain="fourier_half")
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
    sparse_pass2 = sparse_pass2 if sparse_pass2 is not None else SparsePass2Config(enabled=False)
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()
    results = []
    for half_dataset, half_layout in zip(halfset_datasets, halfset_local_layouts, strict=True):
        results.append(
            run_local_ppca_fused_em_iteration(
                half_dataset,
                state.mu_score,
                state.W_score,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                noise_variance=state.noise_variance,
                local_layout=half_layout,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                sparse_pass2=sparse_pass2,
                mean_reg=mean_reg,
                postprocess=postprocess,
                pose_selection=pose_selection,
                top_pose_count=top_pose_count,
                disc_type=disc_type,
            )
        )
    mu_half = (results[0].mu_half, results[1].mu_half)
    W_half = (results[0].W_half, results[1].W_half)
    mu_score, W_score = combine_halfset_scoring_model(mu_half, W_half)
    pose_diagnostics = {
        "halfset0": results[0].diagnostics,
        "halfset1": results[1].diagnostics,
        "delta_rms_mu": float(jnp.sqrt(jnp.mean(jnp.abs(mu_score - jnp.asarray(state.mu_score)) ** 2))),
        "delta_rms_W": float(jnp.sqrt(jnp.mean(jnp.abs(W_score - jnp.asarray(state.W_score)) ** 2)))
        if jnp.asarray(state.W_score).size
        else 0.0,
    }
    return state.replace(
        mu_half=mu_half,
        W_half=W_half,
        mu_score=mu_score,
        W_score=W_score,
        pose_diagnostics=pose_diagnostics,
    )
