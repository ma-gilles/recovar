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


def _project_half(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    projection_half_volume: bool,
    projection_max_r,
):
    if projection_max_r == "auto":
        return core.slice_volume(
            mean_for_proj,
            rotations_block,
            image_shape,
            proj_volume_shape,
            disc_type,
            half_volume=projection_half_volume,
            half_image=True,
        )
    return core.slice_volume(
        mean_for_proj,
        rotations_block,
        image_shape,
        proj_volume_shape,
        disc_type,
        half_volume=projection_half_volume,
        half_image=True,
        max_r=projection_max_r,
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
    cross = -2.0 * jnp.matmul(
        jnp.conj(shifted_score_half),
        proj_weighted.T,
        precision=jax.lax.Precision.HIGHEST,
    ).real
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


def _apply_dense_score_postprocess(
    scores,
    rotation_log_prior_block,
    translation_log_prior_block,
    candidate_mask_block,
    valid_rotation_mask,
    valid_image_mask,
    *,
    score_mode: str,
):
    if score_mode == "gaussian":
        scores = scores + rotation_log_prior_block[:, :, None]
        scores = scores + translation_log_prior_block[:, None, :]
    scores = jnp.where(candidate_mask_block[None, :, :], scores, -jnp.inf)
    scores = jnp.where(valid_rotation_mask[None, :, None], scores, -jnp.inf)
    scores = jnp.where(valid_image_mask[:, None, None], scores, 0.0)
    return scores


def _block_logsumexp_stats(scores, *, use_float64_normalization: bool):
    flat_scores = scores.reshape(scores.shape[0], -1)
    block_max = jnp.max(flat_scores, axis=1)
    if use_float64_normalization:
        exp_terms = jnp.exp((flat_scores - block_max[:, None]).astype(jnp.float64))
    else:
        exp_terms = jnp.exp(flat_scores - block_max[:, None])
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
    probs = jnp.exp(scores - log_Z[:, None, None])
    probs = jnp.where(valid_image_mask[:, None, None], probs, 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, batch_size * n_trans)
    summed_half = P @ shifted_recon_half
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_recon_half
    flat_scores = scores.reshape(batch_size, -1)
    block_best = jnp.max(flat_scores, axis=1)
    block_argmax = jnp.argmax(flat_scores, axis=1)
    max_posterior = jnp.exp(block_best - log_Z)
    block_best = jnp.where(valid_image_mask, block_best, -jnp.inf)
    block_argmax = jnp.where(valid_image_mask, block_argmax, 0)
    max_posterior = jnp.where(valid_image_mask, max_posterior, 0.0)
    return probs, probs_sum_t, summed_half, ctf_probs_half, block_best, block_argmax, max_posterior


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
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    mstep_half_volume: bool,
    backprojection_max_r,
):
    if disable_adjoint_y and disable_adjoint_ctf:
        return Ft_y, Ft_ctf

    if not disable_adjoint_y and not disable_adjoint_ctf:
        volumes = jnp.stack([Ft_y, Ft_ctf], axis=0)
        slices = jnp.stack([summed_half, ctf_probs_half], axis=0)
        if use_window:
            if backprojection_max_r == "auto":
                updated = core.batch_adjoint_slice_volume_indexed(
                    slices,
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=volumes,
                    half_image=True,
                    half_volume=mstep_half_volume,
                )
            else:
                updated = core.batch_adjoint_slice_volume_indexed(
                    slices,
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=volumes,
                    half_image=True,
                    half_volume=mstep_half_volume,
                    max_r=backprojection_max_r,
                )
        else:
            updated = core.batch_adjoint_slice_volume(
                slices,
                rotations_block,
                image_shape,
                recon_volume_shape,
                disc_type,
                volumes=volumes,
                half_image=True,
                half_volume=mstep_half_volume,
            )
        return updated[0], updated[1]

    if not disable_adjoint_y:
        if use_window:
            if backprojection_max_r == "auto":
                Ft_y = core.batch_adjoint_slice_volume_indexed(
                    summed_half[None, :, :],
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=Ft_y[None, :],
                    half_image=True,
                    half_volume=mstep_half_volume,
                )[0]
            else:
                Ft_y = core.batch_adjoint_slice_volume_indexed(
                    summed_half[None, :, :],
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=Ft_y[None, :],
                    half_image=True,
                    half_volume=mstep_half_volume,
                    max_r=backprojection_max_r,
                )[0]
        else:
            Ft_y = core.batch_adjoint_slice_volume(
                summed_half[None, :, :],
                rotations_block,
                image_shape,
                recon_volume_shape,
                disc_type,
                volumes=Ft_y[None, :],
                half_image=True,
                half_volume=mstep_half_volume,
            )[0]
    else:
        if use_window:
            if backprojection_max_r == "auto":
                Ft_ctf = core.batch_adjoint_slice_volume_indexed(
                    ctf_probs_half[None, :, :],
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=Ft_ctf[None, :],
                    half_image=True,
                    half_volume=mstep_half_volume,
                )[0]
            else:
                Ft_ctf = core.batch_adjoint_slice_volume_indexed(
                    ctf_probs_half[None, :, :],
                    window_indices,
                    rotations_block,
                    image_shape,
                    recon_volume_shape,
                    disc_type,
                    volumes=Ft_ctf[None, :],
                    half_image=True,
                    half_volume=mstep_half_volume,
                    max_r=backprojection_max_r,
                )[0]
        else:
            Ft_ctf = core.batch_adjoint_slice_volume(
                ctf_probs_half[None, :, :],
                rotations_block,
                image_shape,
                recon_volume_shape,
                disc_type,
                volumes=Ft_ctf[None, :],
                half_image=True,
                half_volume=mstep_half_volume,
            )[0]
    return Ft_y, Ft_ctf


def _compute_noise_block(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    n_shells: int,
    *,
    return_noise_split: bool,
):
    ctf_probs_raw = ctf_probs * noise_variance_half
    a2 = jnp.sum(proj_abs2_half * ctf_probs_raw, axis=0)
    cross = jnp.sum(proj_half * jnp.conj(summed_masked), axis=0)
    xa = noise_variance_half * cross.real
    block_noise = a2 - 2.0 * xa

    noise_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    noise_shells = noise_shells.at[shell_indices].add(block_noise.astype(jnp.float32))
    if not return_noise_split:
        zeros = jnp.zeros(n_shells, dtype=jnp.float32)
        return noise_shells, zeros, zeros
    a2_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    a2_shells = a2_shells.at[shell_indices].add(a2.astype(jnp.float32))
    xa_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    xa_shells = xa_shells.at[shell_indices].add(xa.astype(jnp.float32))
    return noise_shells, a2_shells, xa_shells


@partial(
    jax.jit,
    static_argnames=(
        "score_mode",
        "zero_dc_for_scoring",
        "use_window",
        "use_float64_scoring",
        "use_float64_normalization",
        "run_mstep",
        "accumulate_noise",
        "return_noise_split",
        "has_translation_sqdist",
        "image_shape",
        "proj_volume_shape",
        "recon_volume_shape",
        "disc_type",
        "projection_half_volume",
        "projection_max_r",
        "mstep_half_volume",
        "backprojection_max_r",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
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
    noise_wsum,
    noise_a2,
    noise_xa,
    noise_sigma2_offset,
    shifted_masked_half,
    noise_variance_half,
    shell_indices_half,
    translation_sqdist_ang,
    window_indices,
    recon_window_indices,
    *,
    score_mode: str = "gaussian",
    zero_dc_for_scoring: bool = True,
    use_window: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    run_mstep: bool = True,
    accumulate_noise: bool = False,
    return_noise_split: bool = False,
    has_translation_sqdist: bool = False,
    image_shape,
    proj_volume_shape,
    recon_volume_shape,
    disc_type: str,
    projection_half_volume: bool = False,
    projection_max_r="auto",
    mstep_half_volume: bool = False,
    backprojection_max_r="auto",
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
    n_shells: int = 0,
) -> DenseBucketResult:
    """Run one dense/global rotation bucket inside one compiled boundary.

    Inputs are half-spectrum arrays.  The caller owns batch preprocessing and
    the two-pass schedule: call with ``run_mstep=False`` to get pass-1 block
    logsumexp summaries, then call with ``run_mstep=True`` and the global
    per-image ``log_Z`` to accumulate M-step/noise for the same bucket.

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
        dc_mask = shell_indices_half == 0
        shifted_score_half = jnp.where(dc_mask[None, :], 0.0, shifted_score_half)
        score_weight_half = jnp.where(dc_mask[None, :], 0.0, score_weight_half)

    proj_half = _project_half(
        mean_for_proj,
        rotations_block,
        image_shape,
        proj_volume_shape,
        disc_type,
        projection_half_volume=projection_half_volume,
        projection_max_r=projection_max_r,
    )

    if use_window:
        shifted_score = shifted_score_half[:, window_indices]
        score_weight = score_weight_half[:, window_indices]
        shifted_recon = shifted_recon_half[:, recon_window_indices]
        ctf2_over_nv_recon = ctf2_over_nv_recon_half[:, recon_window_indices]
        shifted_noise = shifted_masked_half[:, recon_window_indices]
        score_half_weights = half_weights[window_indices]
        proj_score = proj_half[:, window_indices]
        proj_noise = proj_half[:, recon_window_indices]
        noise_variance = noise_variance_half[recon_window_indices]
        shell_indices_noise = shell_indices_half[recon_window_indices]
        adjoint_window_indices = recon_window_indices
    else:
        shifted_score = shifted_score_half
        score_weight = score_weight_half
        shifted_recon = shifted_recon_half
        ctf2_over_nv_recon = ctf2_over_nv_recon_half
        shifted_noise = shifted_masked_half
        score_half_weights = half_weights
        proj_score = proj_half
        proj_noise = proj_half
        noise_variance = noise_variance_half
        shell_indices_noise = shell_indices_half
        adjoint_window_indices = recon_window_indices

    if use_float64_scoring:
        shifted_score = shifted_score.astype(jnp.complex128)
        shifted_recon = shifted_recon.astype(jnp.complex128)
        shifted_noise = shifted_noise.astype(jnp.complex128)
        score_weight = score_weight.astype(jnp.float64)
        ctf2_over_nv_recon = ctf2_over_nv_recon.astype(jnp.float64)
        score_half_weights = score_half_weights.astype(jnp.float64)
        proj_score = proj_score.astype(jnp.complex128)
        proj_noise = proj_noise.astype(jnp.complex128)
    else:
        shifted_score = shifted_score.astype(jnp.complex64)
        shifted_recon = shifted_recon.astype(jnp.complex64)
        shifted_noise = shifted_noise.astype(jnp.complex64)
        score_weight = score_weight.astype(jnp.float32)
        ctf2_over_nv_recon = ctf2_over_nv_recon.astype(jnp.float32)
        score_half_weights = score_half_weights.astype(jnp.float32)
        proj_score = proj_score.astype(jnp.complex64)
        proj_noise = proj_noise.astype(jnp.complex64)

    proj_abs2_score = jnp.abs(proj_score) ** 2
    proj_weighted = proj_score * score_half_weights[None, :]
    proj_abs2_weighted = proj_abs2_score * score_half_weights[None, :]
    scores = _score_block(
        shifted_score,
        score_weight,
        proj_weighted,
        proj_abs2_weighted,
        batch_norm,
        score_mode=score_mode,
    )
    scores = _apply_dense_score_postprocess(
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

    if run_mstep:
        (
            probs,
            probs_sum_t,
            summed_half,
            ctf_probs_half,
            block_best,
            block_argmax,
            max_posterior,
        ) = _mstep_half_sums(
            shifted_recon,
            ctf2_over_nv_recon,
            scores,
            log_Z,
            valid_image_mask.astype(bool),
        )
        Ft_y, Ft_ctf = _adjoint_dense_bucket(
            summed_half,
            ctf_probs_half,
            rotations_block,
            Ft_y,
            Ft_ctf,
            adjoint_window_indices,
            image_shape,
            recon_volume_shape,
            disc_type,
            use_window=use_window,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            mstep_half_volume=mstep_half_volume,
            backprojection_max_r=backprojection_max_r,
        )

        if accumulate_noise:
            P_noise = probs.swapaxes(0, 1).reshape(rot_block_size, batch_size * n_trans)
            summed_masked_noise = P_noise @ shifted_noise
            proj_abs2_noise = jnp.abs(proj_noise) ** 2
            block_noise, block_a2, block_xa = _compute_noise_block(
                proj_noise,
                proj_abs2_noise,
                summed_masked_noise,
                ctf_probs_half,
                noise_variance,
                shell_indices_noise,
                n_shells,
                return_noise_split=return_noise_split,
            )
            noise_wsum = noise_wsum + block_noise
            if return_noise_split:
                noise_a2 = noise_a2 + block_a2
                noise_xa = noise_xa + block_xa
            if has_translation_sqdist:
                translation_posterior = jnp.sum(probs, axis=1)
                noise_sigma2_offset = noise_sigma2_offset + jnp.sum(
                    translation_posterior * translation_sqdist_ang,
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
