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
    ScheduleConfig,
    ScoringConfig,
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
    diagnostics = PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=(best_flat % R).astype(jnp.int32),
        best_translation_idx=(best_flat // R).astype(jnp.int32),
        n_significant_per_image=jnp.sum(gamma > float(significance_threshold), axis=(1, 2)).astype(jnp.int32),
        best_log_score_per_image=jnp.max(score_flat, axis=-1).astype(jnp.float32),
        rotation_posterior_sums=jnp.sum(gamma, axis=(0, 1)).astype(jnp.float32),
        max_posterior_per_image=pmax,
    )
    return gamma, alpha, G_tri, diagnostics


@partial(
    jax.jit,
    static_argnames=(
        "significance_threshold",
        "disc_type_backproject",
        "use_recon_window",
        "backprojection_max_r",
        "image_shape",
        "volume_shape",
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
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    disc_type: str = "linear_interp",
    enforce_x0: bool = True,
    image_indices: np.ndarray | None = None,
    max_hypotheses_per_microbatch: int = 32768,
    fixed_mean_half=None,
):
    """Run one exact-local PPCA EM update over a ``LocalHypothesisLayout``."""
    geometry = geometry if geometry is not None else GeometryConfig()
    # Local PPCA's default batch is intentionally smaller than dense (2 vs 500).
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
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
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()

    q_resolved = int(jnp.asarray(W_prior).shape[1])
    P = q_resolved + 1
    tri = _tri_size(P)
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
    output_image_indices = []
    postprocess_bandlimit_max_r = None

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
        rhs_volume, lhs_tri_volume, diag = fused_local_pose_ppca_bucket(
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
        )
        log_likelihood += float(jnp.sum(diag.logZ))
        n_images += int(diag.logZ.shape[0])
        pmax_values.append(jnp.asarray(diag.pmax))
        nsig_values.append(jnp.asarray(diag.n_significant_per_image))
        best_local = np.asarray(jax.block_until_ready(diag.best_rotation_idx), dtype=np.int64)
        best_trans = np.asarray(jax.block_until_ready(diag.best_translation_idx), dtype=np.int64)
        local_rotation_ids = np.asarray(block.local_rotation_ids, dtype=np.int32)
        local_rotations = np.asarray(block.local_rotations, dtype=np.float32)
        image_rows = np.asarray(block.image_indices, dtype=np.int64)
        best_global = local_rotation_ids[np.arange(best_local.shape[0]), best_local]
        best_rot_mats = local_rotations[np.arange(best_local.shape[0]), best_local]
        translation_grid = np.asarray(local_layout.translation_grid, dtype=np.float32)
        best_trans_vecs = translation_grid[best_trans]

        best_local_values.append(jnp.asarray(best_local, dtype=jnp.int32))
        best_translation_values.append(jnp.asarray(best_trans, dtype=jnp.int32))
        best_global_values.append(jnp.asarray(best_global, dtype=jnp.int32))
        best_rotation_matrices.append(jnp.asarray(best_rot_mats, dtype=jnp.float32))
        best_translations.append(jnp.asarray(best_trans_vecs, dtype=jnp.float32))
        output_image_indices.append(jnp.asarray(image_rows, dtype=jnp.int32))

    if enforce_x0:
        rhs_volume = _enforce_augmented_x0(rhs_volume, tuple(int(x) for x in experiment_dataset.volume_shape))
        lhs_tri_volume = _enforce_augmented_x0(
            lhs_tri_volume.astype(jnp.complex64),
            tuple(int(x) for x in experiment_dataset.volume_shape),
        ).real.astype(jnp.float32)

    image_scale_min, image_scale_max = resolve_image_scale_range(image_scale_corrections, image_indices)
    diagnostics = build_iteration_diagnostics(
        pmax_values=pmax_values,
        nsig_values=nsig_values,
        best_rotations=best_local_values,
        best_translations=best_translation_values,
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
        },
    )
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
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
        config=dataclasses.replace(postprocess, bandlimit_max_r=postprocess_bandlimit_max_r),
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
    diagnostics.update(solved_objective.diagnostics("mstep_objective_solved", n_images=n_images))
    diagnostics.update(output_objective.diagnostics("mstep_objective_output", n_images=n_images))
    diagnostics["mstep_objective_postprocess_delta"] = float(output_objective.total - solved_objective.total)
    diagnostics["mstep_objective_postprocess_delta_per_image"] = (
        float((output_objective.total - solved_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_scope"] = "fixed_e_step_augmented_quadratic_without_constants"
    diagnostics["mstep_objective_postprocess_in_objective"] = False
    return DensePPCAFusedEMResult(mu_half=mu_half, W_half=W_half, stats=stats, diagnostics=diagnostics)


def run_local_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    halfset_local_layouts,
    *,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    disc_type: str = "linear_interp",
) -> PoseMarginalPPCAEMState:
    """Run one exact-local PPCA iteration for two halfsets."""

    from recovar.em.ppca_refinement.dense_dataset import combine_halfset_scoring_model

    if len(halfset_datasets) != 2 or len(halfset_local_layouts) != 2:
        raise ValueError("halfset_datasets and halfset_local_layouts must each have length 2")
    geometry = geometry if geometry is not None else GeometryConfig(volume_domain="fourier_half")
    schedule = schedule if schedule is not None else ScheduleConfig(image_batch_size=2, rotation_block_size=512)
    scoring = scoring if scoring is not None else ScoringConfig()
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
                mean_reg=mean_reg,
                postprocess=postprocess,
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
