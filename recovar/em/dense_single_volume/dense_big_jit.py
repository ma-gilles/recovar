"""Dense/global single-volume EM bucket big-JIT path.

This module provides the compiled per-rotation-bucket boundary used by
``em_engine.run_em`` for eligible dense/global RELION buckets. Inputs stay in
half-spectrum layout so the hot path avoids full Fourier image tensors.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from recovar import core
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.projection import (
    DEFAULT_PROJECTION_MAX_R,
    compute_noise_block,
    project_half_spectrum,
)
from recovar.em.dense_single_volume.helpers.score_constraints import apply_dense_score_constraints


class DenseBucketResult(NamedTuple):
    """Outputs produced by one dense/global rotation bucket."""

    Ft_y: jax.Array
    Ft_ctf: jax.Array
    noise_wsum: jax.Array
    noise_a2: jax.Array
    noise_xa: jax.Array
    noise_sigma2_offset: jax.Array
    block_max: jax.Array
    block_sum_exp: jax.Array
    block_best: jax.Array
    block_argmax: jax.Array
    max_posterior: jax.Array
    probs_sum_t: jax.Array


class _DenseBucketView(NamedTuple):
    """Window/full arrays prepared for one dense big-JIT bucket."""

    shifted_score: jax.Array
    score_weight: jax.Array
    shifted_recon: jax.Array
    ctf2_over_nv_recon: jax.Array
    score_half_weights: jax.Array
    proj_score: jax.Array
    adjoint_window_indices: jax.Array


def _project_half(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    projection_max_r,
):
    max_r = DEFAULT_PROJECTION_MAX_R if projection_max_r == "auto" else projection_max_r
    return project_half_spectrum(
        mean_for_proj,
        rotations_block,
        image_shape,
        proj_volume_shape,
        disc_type,
        max_r=max_r,
    )


def _dense_bucket_view(
    *,
    shifted_score_half,
    score_weight_half,
    shifted_recon_half,
    ctf2_over_nv_recon_half,
    half_weights,
    proj_half,
    window_indices,
    recon_window_indices,
    use_window: bool,
) -> _DenseBucketView:
    if use_window:
        return _DenseBucketView(
            shifted_score=shifted_score_half[:, window_indices],
            score_weight=score_weight_half[:, window_indices],
            shifted_recon=shifted_recon_half[:, recon_window_indices],
            ctf2_over_nv_recon=ctf2_over_nv_recon_half[:, recon_window_indices],
            score_half_weights=half_weights[window_indices],
            proj_score=proj_half[:, window_indices],
            adjoint_window_indices=recon_window_indices,
        )
    return _DenseBucketView(
        shifted_score=shifted_score_half,
        score_weight=score_weight_half,
        shifted_recon=shifted_recon_half,
        ctf2_over_nv_recon=ctf2_over_nv_recon_half,
        score_half_weights=half_weights,
        proj_score=proj_half,
        adjoint_window_indices=recon_window_indices,
    )


def _cast_dense_bucket_view(view: _DenseBucketView, precision_policy: DensePrecisionPolicy) -> _DenseBucketView:
    (
        shifted_score,
        shifted_recon,
        score_weight,
        ctf2_over_nv_recon,
        score_half_weights,
        proj_score,
    ) = precision_policy.cast_dense_big_jit_inputs(
        view.shifted_score,
        view.shifted_recon,
        view.score_weight,
        view.ctf2_over_nv_recon,
        view.score_half_weights,
        view.proj_score,
    )
    return view._replace(
        shifted_score=shifted_score,
        shifted_recon=shifted_recon,
        score_weight=score_weight,
        ctf2_over_nv_recon=ctf2_over_nv_recon,
        score_half_weights=score_half_weights,
        proj_score=proj_score,
    )


def _score_block(
    shifted_score_half,
    score_weight_half,
    proj_weighted,
    proj_abs2_weighted,
    batch_norm,
    *,
    score_mode: str,
):
    batch_size = batch_norm.shape[0]
    n_trans = shifted_score_half.shape[0] // batch_size
    rot_block_size = proj_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_score_half),
            proj_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(batch_size, n_trans, rot_block_size).swapaxes(1, 2)
    norms = jnp.matmul(
        score_weight_half,
        proj_abs2_weighted.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    if score_mode == "normalized_cc":
        denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
        return (-0.5 * cross) / denom[..., None]
    return -0.5 * (cross + norms[..., None])


def _block_logsumexp_stats(scores, *, use_float64_normalization: bool):
    flat_scores = scores.reshape(scores.shape[0], -1)
    block_max = jnp.max(flat_scores, axis=1)
    # When every pose is masked out (e.g. K-class adaptive 2-pass for a class
    # whose pass-1 significance mask is empty for this image), block_max is
    # -inf and ``flat_scores - block_max`` becomes -inf - (-inf) = NaN.
    # Use a finite shift in that case; the resulting sum_exp is 0, giving
    # log_Z = -inf rather than NaN, which the K-class aggregator handles.
    safe_max = jnp.where(jnp.isfinite(block_max), block_max, jnp.zeros_like(block_max))
    if use_float64_normalization:
        exp_terms = jnp.exp((flat_scores - safe_max[:, None]).astype(jnp.float64))
    else:
        exp_terms = jnp.exp(flat_scores - safe_max[:, None])
    return block_max, jnp.sum(exp_terms, axis=1)


def _mstep_half_sums(
    shifted_recon_half,
    ctf2_over_nv_recon_half,
    scores,
    log_Z,
    valid_image_mask,
):
    batch_size = scores.shape[0]
    rot_block_size = scores.shape[1]
    n_trans = scores.shape[2]
    # Guard probs computation against -inf - (-inf) = NaN: when either the
    # per-pose score or the per-image normalizer is -inf, the contribution
    # is mathematically zero. This handles K-class adaptive 2-pass cases
    # where a class has an empty significance mask for some images.
    diff = scores - log_Z[:, None, None]
    probs = jnp.where(jnp.isfinite(diff), jnp.exp(diff), 0.0)
    probs = jnp.where(valid_image_mask[:, None, None], probs, 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, batch_size * n_trans)
    summed_half = P @ shifted_recon_half
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_recon_half
    flat_scores = scores.reshape(batch_size, -1)
    block_best = jnp.max(flat_scores, axis=1)
    block_argmax = jnp.argmax(flat_scores, axis=1)
    best_diff = block_best - log_Z
    max_posterior = jnp.where(jnp.isfinite(best_diff), jnp.exp(best_diff), 0.0)
    block_best = jnp.where(valid_image_mask, block_best, -jnp.inf)
    block_argmax = jnp.where(valid_image_mask, block_argmax, 0)
    max_posterior = jnp.where(valid_image_mask, max_posterior, 0.0)
    return probs, probs_sum_t, summed_half, ctf_probs_half, block_best, block_argmax, max_posterior


def _mstep_half_sums_wta(
    shifted_recon_half,
    ctf2_over_nv_recon_half,
    scores,
    valid_image_mask,
    *,
    wta_argmax,
    wta_best_score,
    block_r0,
):
    """Winner-take-all M-step: one-hot probs anchored on the global per-image argmax.

    ``wta_argmax`` is the global flat index ``rot * n_trans + trans`` over the
    full grid; ``block_r0`` is this block's first global rotation index (runtime
    scalar). Images whose winner falls outside ``[r0, r0 + rotation_block_size)``
    contribute zero, matching RELION's ``--firstiter_cc`` binarization
    (ml_optimiser.cpp:9181-9207). Padded rotations within a block are masked to
    -inf upstream so they never claim the global argmax.
    """
    batch_size = scores.shape[0]
    rot_block_size = scores.shape[1]
    n_trans = scores.shape[2]
    valid_image = jnp.isfinite(wta_best_score) & valid_image_mask
    winning_rot = wta_argmax // n_trans
    winning_trans = wta_argmax % n_trans
    in_block = (winning_rot >= block_r0) & (winning_rot < (block_r0 + rot_block_size))
    local_rot = jnp.clip(winning_rot - block_r0, 0, rot_block_size - 1)
    flat_local = local_rot * n_trans + winning_trans
    probs = jax.nn.one_hot(
        flat_local,
        rot_block_size * n_trans,
        dtype=scores.real.dtype,
    ).reshape(batch_size, rot_block_size, n_trans)
    probs = probs * (in_block & valid_image)[:, None, None].astype(probs.dtype)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, batch_size * n_trans)
    summed_half = P @ shifted_recon_half
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_recon_half
    flat_scores = scores.reshape(batch_size, -1)
    block_best = jnp.where(valid_image_mask, jnp.max(flat_scores, axis=1), -jnp.inf)
    block_argmax = jnp.where(valid_image_mask, jnp.argmax(flat_scores, axis=1), 0)
    # max_posterior is 1.0 for images whose winner sits in this block; the
    # caller's streaming max across blocks then yields 1.0 globally.
    max_posterior = (valid_image & in_block).astype(scores.real.dtype)
    return probs, probs_sum_t, summed_half, ctf_probs_half, block_best, block_argmax, max_posterior


def _batch_adjoint_dense_slices(
    slices,
    volumes,
    rotations_block,
    window_indices,
    image_shape,
    recon_volume_shape,
    disc_type,
    *,
    use_window: bool,
    half_volume: bool,
    backprojection_max_r,
):
    if not use_window:
        return core.batch_adjoint_slice_volume(
            slices,
            rotations_block,
            image_shape,
            recon_volume_shape,
            disc_type,
            volumes=volumes,
            half_image=True,
            half_volume=half_volume,
        )
    kwargs = {}
    if backprojection_max_r != "auto":
        kwargs["max_r"] = backprojection_max_r
    return core.batch_adjoint_slice_volume_indexed(
        slices,
        window_indices,
        rotations_block,
        image_shape,
        recon_volume_shape,
        disc_type,
        volumes=volumes,
        half_image=True,
        half_volume=half_volume,
        **kwargs,
    )


def _adjoint_dense_bucket(
    summed_half,
    ctf_probs_half,
    rotations_block,
    Ft_y,
    Ft_ctf,
    window_indices,
    image_shape,
    recon_volume_shape,
    disc_type,
    *,
    use_window: bool,
    half_volume: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    backprojection_max_r,
):
    if disable_adjoint_y and disable_adjoint_ctf:
        return Ft_y, Ft_ctf

    if not disable_adjoint_y and not disable_adjoint_ctf:
        volumes = jnp.stack([Ft_y, Ft_ctf], axis=0)
        slices = jnp.stack([summed_half, ctf_probs_half], axis=0)
        updated = _batch_adjoint_dense_slices(
            slices,
            volumes,
            rotations_block,
            window_indices,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            half_volume=half_volume,
            backprojection_max_r=backprojection_max_r,
        )
        return updated[0], updated[1]

    if not disable_adjoint_y:
        Ft_y = _batch_adjoint_dense_slices(
            summed_half[None, :, :],
            Ft_y[None, :],
            rotations_block,
            window_indices,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            half_volume=half_volume,
            backprojection_max_r=backprojection_max_r,
        )[0]
    elif not disable_adjoint_ctf:
        Ft_ctf = _batch_adjoint_dense_slices(
            ctf_probs_half[None, :, :],
            Ft_ctf[None, :],
            rotations_block,
            window_indices,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            half_volume=half_volume,
            backprojection_max_r=backprojection_max_r,
        )[0]
    return Ft_y, Ft_ctf


def _noise_half_sums(
    *,
    probs,
    shifted_noise,
    ctf2_over_nv_recon,
    proj_for_noise,
    noise_variance,
    shell_indices,
    translation_sqdist_ang,
    return_noise_split: bool,
    n_shells: int,
):
    """Compute RELION-style per-shell noise sums for one dense bucket."""

    batch_size = probs.shape[0]
    rot_block_size = probs.shape[1]
    n_trans = probs.shape[2]
    P_noise = probs.swapaxes(0, 1).reshape(rot_block_size, batch_size * n_trans)
    summed_masked_noise = P_noise @ shifted_noise
    probs_sum_t_noise = jnp.sum(probs, axis=-1)
    ctf_probs = probs_sum_t_noise.T @ ctf2_over_nv_recon
    proj_abs2_for_noise = jnp.abs(proj_for_noise) ** 2
    noise_wsum, noise_a2, noise_xa = compute_noise_block(
        proj_for_noise,
        proj_abs2_for_noise,
        summed_masked_noise,
        ctf_probs,
        noise_variance,
        shell_indices,
        n_shells,
        return_noise_split,
    )
    translation_posterior = jnp.sum(probs, axis=1)
    noise_sigma2_offset = jnp.sum(translation_posterior * translation_sqdist_ang)
    return noise_wsum, noise_a2, noise_xa, noise_sigma2_offset


@partial(
    jax.jit,
    static_argnames=(
        "score_mode",
        "zero_dc_for_scoring",
        "use_window",
        "use_float64_scoring",
        "use_float64_normalization",
        "run_mstep",
        "winner_take_all",
        "image_shape",
        "proj_volume_shape",
        "recon_volume_shape",
        "disc_type",
        "projection_max_r",
        "backprojection_max_r",
        "mstep_half_volume",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "accumulate_noise",
        "return_noise_split",
        "n_shells",
    ),
)
def run_dense_bucket_big_jit(
    shifted_score_half,
    batch_norm,
    score_weight_half,
    shifted_recon_half,
    ctf2_over_nv_recon_half,
    mean_for_proj,
    Ft_y,
    Ft_ctf,
    rotations_block,
    half_weights,
    rotation_log_prior_block,
    translation_log_prior_block,
    candidate_mask_block,
    valid_rotation_mask,
    valid_image_mask,
    log_Z,
    window_indices,
    recon_window_indices,
    shifted_noise_half=None,
    noise_variance_half=None,
    shell_indices_noise=None,
    translation_sqdist_ang=None,
    wta_argmax=None,
    wta_best_score=None,
    wta_block_r0=None,
    *,
    score_mode: str = "gaussian",
    zero_dc_for_scoring: bool = True,
    use_window: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    run_mstep: bool = True,
    winner_take_all: bool = False,
    image_shape,
    proj_volume_shape,
    recon_volume_shape,
    disc_type: str,
    projection_max_r="auto",
    backprojection_max_r="auto",
    mstep_half_volume: bool = False,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
    accumulate_noise: bool = False,
    return_noise_split: bool = False,
    n_shells: int = 0,
) -> DenseBucketResult:
    """Run one dense/global rotation bucket inside one compiled boundary.

    Inputs are half-spectrum arrays.  The caller owns batch preprocessing and
    the two-pass schedule: call with ``run_mstep=False`` to get pass-1 block
    logsumexp summaries, then call with ``run_mstep=True`` and the global
    per-image ``log_Z`` to accumulate the M-step for the same bucket.

    ``rotation_log_prior_block`` must be shaped ``(batch, rot_block)`` and
    ``translation_log_prior_block`` must be shaped ``(batch, n_trans)``.  Use
    zeros for flat priors.  ``candidate_mask_block`` and
    ``valid_rotation_mask`` should already include any padding/candidate masks.
    ``valid_image_mask`` marks real image rows when the caller pads a tail
    image batch to a stable shape class.
    """
    if score_mode not in ("gaussian", "normalized_cc"):
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")

    if zero_dc_for_scoring and score_mode != "normalized_cc":
        dc_index = (int(image_shape[0]) // 2) * (int(image_shape[1]) // 2 + 1)
        shifted_score_half = shifted_score_half.at[:, dc_index].set(0.0)
        score_weight_half = score_weight_half.at[:, dc_index].set(0.0)

    proj_half = _project_half(
        mean_for_proj,
        rotations_block,
        image_shape,
        proj_volume_shape,
        disc_type,
        projection_max_r=projection_max_r,
    )

    bucket_view = _cast_dense_bucket_view(
        _dense_bucket_view(
            shifted_score_half=shifted_score_half,
            score_weight_half=score_weight_half,
            shifted_recon_half=shifted_recon_half,
            ctf2_over_nv_recon_half=ctf2_over_nv_recon_half,
            half_weights=half_weights,
            proj_half=proj_half,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            use_window=use_window,
        ),
        DensePrecisionPolicy(use_float64_scoring=use_float64_scoring),
    )

    proj_abs2_score = jnp.abs(bucket_view.proj_score) ** 2
    proj_weighted = bucket_view.proj_score * bucket_view.score_half_weights[None, :]
    proj_abs2_weighted = proj_abs2_score * bucket_view.score_half_weights[None, :]
    scores = _score_block(
        bucket_view.shifted_score,
        bucket_view.score_weight,
        proj_weighted,
        proj_abs2_weighted,
        batch_norm,
        score_mode=score_mode,
    )
    scores = apply_dense_score_constraints(
        scores,
        rotation_log_prior_block,
        translation_log_prior_block,
        candidate_mask_block,
        valid_rotation_mask,
        valid_image_mask.astype(bool),
        score_mode=score_mode,
    )

    block_max, block_sum_exp = _block_logsumexp_stats(
        scores,
        use_float64_normalization=use_float64_normalization,
    )
    batch_size = scores.shape[0]
    rot_block_size = scores.shape[1]
    n_trans = scores.shape[2]
    block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
    block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
    max_posterior = jnp.zeros(batch_size, dtype=scores.real.dtype)
    probs_sum_t = jnp.zeros((batch_size, rot_block_size), dtype=scores.real.dtype)
    noise_wsum = jnp.zeros(n_shells, dtype=jnp.float32)
    noise_a2 = jnp.zeros(n_shells, dtype=jnp.float32)
    noise_xa = jnp.zeros(n_shells, dtype=jnp.float32)
    noise_sigma2_offset = jnp.asarray(0.0, dtype=jnp.float32)

    if run_mstep:
        if winner_take_all:
            (
                probs,
                probs_sum_t,
                summed_half,
                ctf_probs_half,
                block_best,
                block_argmax,
                max_posterior,
            ) = _mstep_half_sums_wta(
                bucket_view.shifted_recon,
                bucket_view.ctf2_over_nv_recon,
                scores,
                valid_image_mask.astype(bool),
                wta_argmax=wta_argmax,
                wta_best_score=wta_best_score,
                block_r0=wta_block_r0,
            )
        else:
            (
                probs,
                probs_sum_t,
                summed_half,
                ctf_probs_half,
                block_best,
                block_argmax,
                max_posterior,
            ) = _mstep_half_sums(
                bucket_view.shifted_recon,
                bucket_view.ctf2_over_nv_recon,
                scores,
                log_Z,
                valid_image_mask.astype(bool),
            )
        if accumulate_noise:
            if use_window:
                shifted_noise = shifted_noise_half[:, recon_window_indices]
                proj_for_noise = proj_half[:, recon_window_indices]
                noise_variance = noise_variance_half[recon_window_indices]
            else:
                shifted_noise = shifted_noise_half
                proj_for_noise = proj_half
                noise_variance = noise_variance_half
            noise_wsum, noise_a2, noise_xa, noise_sigma2_offset = _noise_half_sums(
                probs=probs,
                shifted_noise=shifted_noise,
                ctf2_over_nv_recon=bucket_view.ctf2_over_nv_recon,
                proj_for_noise=proj_for_noise,
                noise_variance=noise_variance,
                shell_indices=shell_indices_noise,
                translation_sqdist_ang=translation_sqdist_ang,
                return_noise_split=return_noise_split,
                n_shells=n_shells,
            )
        Ft_y, Ft_ctf = _adjoint_dense_bucket(
            summed_half,
            ctf_probs_half,
            rotations_block,
            Ft_y,
            Ft_ctf,
            bucket_view.adjoint_window_indices,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            half_volume=mstep_half_volume,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            backprojection_max_r=backprojection_max_r,
        )

    return DenseBucketResult(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        noise_wsum=noise_wsum,
        noise_a2=noise_a2,
        noise_xa=noise_xa,
        noise_sigma2_offset=noise_sigma2_offset,
        block_max=block_max,
        block_sum_exp=block_sum_exp,
        block_best=block_best,
        block_argmax=block_argmax,
        max_posterior=max_posterior,
        probs_sum_t=probs_sum_t,
    )


__all__ = ["DenseBucketResult", "run_dense_bucket_big_jit"]
