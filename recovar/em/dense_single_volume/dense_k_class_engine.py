"""Native dense K-class EM over the joint class x pose grid."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils

from .helpers.backprojection import (
    batch_adjoint_slice_volume_half as _batch_adjoint_slice_volume_half,
    batch_adjoint_slice_volume_windowed as _batch_adjoint_slice_volume_windowed,
)
from .helpers.dtype_policy import DensePrecisionPolicy
from .helpers.fourier_window import make_fourier_window_spec
from .helpers.half_spectrum import (
    half_spectrum_dc_index,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
)
from .helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
    tiled_half_image_phase_factors,
)
from .helpers.jax_runtime import block_until_ready as _block_until_ready
from .helpers.preprocessing import (
    apply_half_translation_phases,
    backend_half_preprocess_uses_host_images,
    dense_batch_half_input_pair,
    half_translation_phase_table,
    preprocess_half_image_pair_device,
    prepare_reconstruction_batch as _prepare_reconstruction_batch,
    process_half_image,
    preprocess_batch as _preprocess_batch,
    preprocess_batch_firstiter_cc as _preprocess_batch_firstiter_cc,
    resolve_image_mask_for_half_preprocess,
)
from .helpers.projection import (
    compute_noise_block as _compute_noise_block,
    compute_projections_block as _compute_projections_block,
)
from .helpers.scoring import _score_rotation_block, _update_logsumexp
from .helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from .helpers.types import make_noise_stats, make_relion_stats

logger = logging.getLogger(__name__)


class DenseKClassNativeOutputs(NamedTuple):
    """Raw native dense K-class outputs consumed by ``k_class`` assembly."""

    class_log_evidence: np.ndarray
    new_means: list[object | None]
    Ft_y: list[object]
    Ft_ctf: list[object]
    hard_assignments: np.ndarray
    per_class_stats: tuple[object, ...]
    noise_stats: tuple[object, ...] | None
    profile_summary: dict[str, object] | None = None
    grouped_Ft_y: object | None = None
    grouped_Ft_ctf: object | None = None


@dataclass(frozen=True)
class _KClassScoreConstraints:
    """Dense K-class priors and masks normalized to padded rotation blocks."""

    rotation_log_prior: object | None
    per_image_rotation_prior: bool
    translation_log_prior: object | None
    per_image_translation_prior: bool
    candidate_mask: object | None
    n_classes: int
    n_images: int
    n_rot: int
    n_trans: int
    n_rot_padded: int

    @classmethod
    def from_inputs(
        cls,
        *,
        n_classes: int,
        n_images: int,
        n_rot: int,
        n_trans: int,
        n_rot_padded: int,
        rotation_log_prior,
        class_rotation_log_prior,
        translation_log_prior,
        rotation_translation_mask,
    ) -> "_KClassScoreConstraints":
        if rotation_log_prior is not None and class_rotation_log_prior is not None:
            raise ValueError("Provide only one of rotation_log_prior or class_rotation_log_prior")

        per_image_rotation_prior = False
        rotation_prior_jnp = None
        if class_rotation_log_prior is not None:
            rotation_np = np.asarray(class_rotation_log_prior, dtype=np.float32)
            if rotation_np.ndim == 2:
                if rotation_np.shape != (n_classes, n_rot):
                    raise ValueError(
                        "class_rotation_log_prior must have shape "
                        f"({n_classes}, {n_rot}), got {rotation_np.shape}",
                    )
                padded = np.full((n_classes, n_rot_padded), -1e30, dtype=np.float32)
                padded[:, :n_rot] = rotation_np
            elif rotation_np.ndim == 3:
                if rotation_np.shape != (n_classes, n_images, n_rot):
                    raise ValueError(
                        "class_rotation_log_prior must have shape "
                        f"({n_classes}, {n_images}, {n_rot}), got {rotation_np.shape}",
                    )
                padded = np.full((n_classes, n_images, n_rot_padded), -1e30, dtype=np.float32)
                padded[:, :, :n_rot] = rotation_np
                per_image_rotation_prior = True
            else:
                raise ValueError(
                    "class_rotation_log_prior must be 2D or 3D, "
                    f"got {rotation_np.ndim} dimensions",
                )
            rotation_prior_jnp = jnp.asarray(padded)
        elif rotation_log_prior is not None:
            rotation_np = np.asarray(rotation_log_prior, dtype=np.float32)
            if rotation_np.ndim == 1:
                if rotation_np.shape != (n_rot,):
                    raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {rotation_np.shape}")
                padded = np.full((n_classes, n_rot_padded), -1e30, dtype=np.float32)
                padded[:, :n_rot] = rotation_np[None, :]
            elif rotation_np.ndim == 2:
                if rotation_np.shape != (n_images, n_rot):
                    raise ValueError(
                        "rotation_log_prior must have shape "
                        f"({n_images}, {n_rot}) when image-specific, got {rotation_np.shape}",
                    )
                padded = np.full((n_classes, n_images, n_rot_padded), -1e30, dtype=np.float32)
                padded[:, :, :n_rot] = rotation_np[None, :, :]
                per_image_rotation_prior = True
            else:
                raise ValueError(f"rotation_log_prior must be 1D or 2D, got {rotation_np.ndim} dimensions")
            rotation_prior_jnp = jnp.asarray(padded)

        per_image_translation_prior = False
        translation_prior_jnp = None
        if translation_log_prior is not None:
            translation_np = np.asarray(translation_log_prior, dtype=np.float32)
            if translation_np.ndim == 1:
                if translation_np.shape != (n_trans,):
                    raise ValueError(f"translation_log_prior must have shape ({n_trans},), got {translation_np.shape}")
            elif translation_np.ndim == 2:
                if translation_np.shape != (n_images, n_trans):
                    raise ValueError(
                        "translation_log_prior must have shape "
                        f"({n_images}, {n_trans}) when image-specific, got {translation_np.shape}",
                    )
                per_image_translation_prior = True
            else:
                raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_np.ndim} dimensions")
            translation_prior_jnp = jnp.asarray(translation_np)

        candidate_mask_jnp = None
        if rotation_translation_mask is not None:
            candidate_mask = np.asarray(rotation_translation_mask, dtype=bool)
            if candidate_mask.shape != (n_rot, n_trans):
                raise ValueError(
                    f"rotation_translation_mask must have shape ({n_rot}, {n_trans}), got {candidate_mask.shape}",
                )
            padded_mask = np.zeros((n_rot_padded, n_trans), dtype=bool)
            padded_mask[:n_rot] = candidate_mask
            candidate_mask_jnp = jnp.asarray(padded_mask)

        return cls(
            rotation_log_prior=rotation_prior_jnp,
            per_image_rotation_prior=per_image_rotation_prior,
            translation_log_prior=translation_prior_jnp,
            per_image_translation_prior=per_image_translation_prior,
            candidate_mask=candidate_mask_jnp,
            n_classes=int(n_classes),
            n_images=int(n_images),
            n_rot=int(n_rot),
            n_trans=int(n_trans),
            n_rot_padded=int(n_rot_padded),
        )

    def block_inputs(self, *, r0: int, r1: int, start: int, end: int, batch_count: int, rotation_block_size: int):
        actual_count = int(end - start)
        if self.rotation_log_prior is None:
            rotation_prior = jnp.zeros((batch_count, self.n_classes, rotation_block_size), dtype=jnp.float32)
        elif self.per_image_rotation_prior:
            rotation_prior = self.rotation_log_prior[:, start:end, r0:r1]
            rotation_prior = jnp.swapaxes(rotation_prior, 0, 1)
            if batch_count != actual_count:
                rotation_prior = jnp.pad(
                    rotation_prior,
                    ((0, batch_count - actual_count), (0, 0), (0, 0)),
                    constant_values=0,
                )
        else:
            rotation_prior = jnp.broadcast_to(
                self.rotation_log_prior[:, r0:r1][None, :, :],
                (batch_count, self.n_classes, rotation_block_size),
            )

        if self.translation_log_prior is None:
            translation_prior = jnp.zeros((batch_count, self.n_trans), dtype=jnp.float32)
        elif self.per_image_translation_prior:
            translation_prior = self.translation_log_prior[start:end]
            if batch_count != actual_count:
                translation_prior = jnp.pad(
                    translation_prior,
                    ((0, batch_count - actual_count), (0, 0)),
                    constant_values=0,
                )
        else:
            translation_prior = jnp.broadcast_to(
                self.translation_log_prior[None, :],
                (batch_count, self.n_trans),
            )

        if self.candidate_mask is None:
            candidate_mask = jnp.ones((rotation_block_size, self.n_trans), dtype=bool)
        else:
            candidate_mask = self.candidate_mask[r0:r1]
        valid = max(0, min(rotation_block_size, self.n_rot - int(r0)))
        valid_rotation_mask = jnp.arange(rotation_block_size) < valid
        return rotation_prior, translation_prior, candidate_mask, valid_rotation_mask


@dataclass
class _DenseKClassNoiseState:
    """Host-side per-class noise accumulators for one dense K-class EM pass."""

    wsum: np.ndarray
    img_power: np.ndarray
    sumw: np.ndarray
    sigma2_offset: np.ndarray

    @classmethod
    def zeros(cls, *, n_classes: int, n_shells: int) -> "_DenseKClassNoiseState":
        return cls(
            wsum=np.zeros((n_classes, n_shells), dtype=np.float64),
            img_power=np.zeros((n_classes, n_shells), dtype=np.float64),
            sumw=np.zeros(n_classes, dtype=np.float64),
            sigma2_offset=np.zeros(n_classes, dtype=np.float64),
        )

    def add_image_power(self, batch_img_power_shells, class_mass) -> None:
        self.img_power += np.asarray(batch_img_power_shells, dtype=np.float64)
        self.sumw += np.asarray(class_mass, dtype=np.float64).sum(axis=0)

    def add_translation_offset(self, probs, translation_sqdist_ang) -> None:
        if translation_sqdist_ang is None:
            return
        translation_posterior = np.asarray(jnp.sum(probs, axis=2), dtype=np.float64)
        self.sigma2_offset += np.sum(
            translation_posterior * translation_sqdist_ang[:, None, :],
            axis=(0, 2),
            dtype=np.float64,
        )

    def add_noise_shells(self, block_noise_shells) -> None:
        self.wsum += np.asarray(jnp.stack(block_noise_shells, axis=0), dtype=np.float64)

    def stats(self):
        return tuple(
            make_noise_stats(
                wsum_sigma2_noise=self.wsum[class_index],
                wsum_img_power=self.img_power[class_index],
                wsum_sigma2_offset=self.sigma2_offset[class_index],
                sumw=self.sumw[class_index],
            )
            for class_index in range(int(self.wsum.shape[0]))
        )


@jax.jit
def _update_class_logsumexp(max_s, sum_exp, scores):
    """Streaming class-wise logsumexp update from one ``(image, class, pose)`` block."""

    accumulator_dtype = sum_exp.dtype
    scores_flat = scores.reshape(scores.shape[0], scores.shape[1], -1)
    block_max = jnp.max(scores_flat, axis=2)
    new_max = jnp.maximum(max_s, block_max)
    exp_terms = jnp.sum(
        jnp.exp((scores_flat - new_max[:, :, None]).astype(accumulator_dtype)),
        axis=2,
    )
    sum_exp = sum_exp * jnp.exp((max_s - new_max).astype(accumulator_dtype)) + exp_terms
    return new_max, sum_exp


@partial(jax.jit, static_argnames=("score_mode",))
def _apply_k_class_score_constraints(
    scores,
    rotation_prior,
    translation_prior,
    class_log_priors,
    candidate_mask,
    valid_rotation_mask,
    *,
    score_mode: str,
):
    """Apply dense score priors and masks to ``(batch, class, rotation, translation)`` scores."""

    if score_mode == "gaussian":
        scores = scores + rotation_prior[:, :, :, None]
        scores = scores + translation_prior[:, None, None, :]
        scores = scores + class_log_priors[None, :, None, None]
    scores = jnp.where(candidate_mask[None, None, :, :], scores, -jnp.inf)
    return jnp.where(valid_rotation_mask[None, None, :, None], scores, -jnp.inf)


@partial(jax.jit, static_argnums=(4,))
def _k_class_m_step_block(shifted_recon, scores, log_z, ctf2_over_nv, n_trans: int):
    """Normalize joint class x pose scores and form per-class adjoint slices."""

    batch_size, n_classes, rotation_block_size, _ = scores.shape
    probs = jnp.exp(scores - log_z[:, None, None, None])
    shifted_by_translation = shifted_recon.reshape(batch_size, n_trans, shifted_recon.shape[-1])
    summed = jnp.einsum(
        "bkrt,btn->krn",
        probs,
        shifted_by_translation,
        precision=jax.lax.Precision.HIGHEST,
    )
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs = jnp.einsum(
        "bkr,bn->krn",
        probs_sum_t,
        ctf2_over_nv,
        precision=jax.lax.Precision.HIGHEST,
    )

    class_scores_flat = scores.reshape(batch_size, n_classes, rotation_block_size * n_trans)
    block_best_class = jnp.max(class_scores_flat, axis=2)
    block_argmax_class = jnp.argmax(class_scores_flat, axis=2)
    global_scores_flat = scores.reshape(batch_size, n_classes * rotation_block_size * n_trans)
    block_best_global = jnp.max(global_scores_flat, axis=1)
    block_argmax_global = jnp.argmax(global_scores_flat, axis=1)
    rotation_sums = jnp.sum(probs, axis=(0, 3))
    return (
        probs,
        block_best_class,
        block_argmax_class,
        block_best_global,
        block_argmax_global,
        summed,
        ctf_probs,
        rotation_sums,
    )


@partial(jax.jit, static_argnums=(5, 6))
def _grouped_k_class_m_step_block(
    shifted_recon,
    scores,
    log_z,
    ctf2_over_nv,
    group_ids,
    n_trans: int,
    n_groups: int,
):
    """Normalize scores and form per-class adjoint slices split by image group."""

    batch_size, n_classes, rotation_block_size, _ = scores.shape
    probs = jnp.exp(scores - log_z[:, None, None, None])
    shifted_by_translation = shifted_recon.reshape(probs.shape[0], int(n_trans), shifted_recon.shape[-1])
    group_mask = jax.nn.one_hot(group_ids, int(n_groups), dtype=probs.real.dtype)
    grouped_probs = probs[:, None, :, :, :] * group_mask[:, :, None, None, None]
    grouped_summed = jnp.einsum(
        "bgkrt,btn->gkrn",
        grouped_probs,
        shifted_by_translation,
        precision=jax.lax.Precision.HIGHEST,
    )
    grouped_probs_sum_t = jnp.sum(grouped_probs, axis=-1)
    grouped_ctf_probs = jnp.einsum(
        "bgkr,bn->gkrn",
        grouped_probs_sum_t,
        ctf2_over_nv,
        precision=jax.lax.Precision.HIGHEST,
    )

    class_scores_flat = scores.reshape(batch_size, n_classes, rotation_block_size * n_trans)
    block_best_class = jnp.max(class_scores_flat, axis=2)
    block_argmax_class = jnp.argmax(class_scores_flat, axis=2)
    global_scores_flat = scores.reshape(batch_size, n_classes * rotation_block_size * n_trans)
    block_best_global = jnp.max(global_scores_flat, axis=1)
    block_argmax_global = jnp.argmax(global_scores_flat, axis=1)
    rotation_sums = jnp.sum(probs, axis=(0, 3))
    return (
        probs,
        block_best_class,
        block_argmax_class,
        block_best_global,
        block_argmax_global,
        grouped_summed,
        grouped_ctf_probs,
        rotation_sums,
    )


@jax.jit
def _update_k_class_wta_pass1(
    max_s,
    sum_exp,
    class_max_s,
    class_sum_exp,
    best_score_class,
    best_argmax_class,
    global_best_score,
    global_best_argmax,
    scores,
    r0,
    n_rot_padded,
    n_trans,
):
    """Update evidence and WTA winners from one class x pose score block."""

    batch_size, n_classes, rotation_block_size, _ = scores.shape
    class_scores_flat = scores.reshape(batch_size, n_classes, rotation_block_size * scores.shape[-1])
    block_best_class = jnp.max(class_scores_flat, axis=2)
    accumulator_dtype = class_sum_exp.dtype
    block_class_sum_exp = jnp.sum(
        jnp.exp((class_scores_flat - block_best_class[:, :, None]).astype(accumulator_dtype)),
        axis=2,
    )

    new_class_max = jnp.maximum(class_max_s, block_best_class)
    class_sum_exp = (
        class_sum_exp * jnp.exp((class_max_s - new_class_max).astype(accumulator_dtype))
        + block_class_sum_exp * jnp.exp((block_best_class - new_class_max).astype(accumulator_dtype))
    )
    class_max_s = new_class_max

    block_best_global = jnp.max(block_best_class, axis=1)
    block_sum_exp_global = jnp.sum(
        block_class_sum_exp * jnp.exp((block_best_class - block_best_global[:, None]).astype(accumulator_dtype)),
        axis=1,
    )
    new_max = jnp.maximum(max_s, block_best_global)
    sum_exp = (
        sum_exp * jnp.exp((max_s - new_max).astype(accumulator_dtype))
        + block_sum_exp_global * jnp.exp((block_best_global - new_max).astype(accumulator_dtype))
    )
    max_s = new_max

    block_argmax_class = jnp.argmax(class_scores_flat, axis=2)
    improved_class = block_best_class > best_score_class
    best_score_class = jnp.where(improved_class, block_best_class, best_score_class)
    best_argmax_class = jnp.where(improved_class, block_argmax_class + r0 * n_trans, best_argmax_class)

    block_class = jnp.argmax(block_best_class, axis=1)
    block_pose = block_argmax_class[jnp.arange(batch_size), block_class]
    block_global_argmax = block_class * (n_rot_padded * n_trans) + block_pose + r0 * n_trans
    improved = block_best_global > global_best_score
    global_best_score = jnp.where(improved, block_best_global, global_best_score)
    global_best_argmax = jnp.where(improved, block_global_argmax, global_best_argmax)

    return (
        max_s,
        sum_exp,
        class_max_s,
        class_sum_exp,
        best_score_class,
        best_argmax_class,
        global_best_score,
        global_best_argmax,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 7, 8))
def _k_class_wta_m_step_block(
    shifted_recon,
    global_best_argmax,
    r0,
    actual_rot: int,
    rotation_block_size: int,
    n_rot_padded: int,
    ctf2_over_nv,
    n_trans: int,
    n_classes: int,
):
    """Form hard M-step slices for global class x pose winners."""

    batch_size = int(global_best_argmax.shape[0])
    shifted_by_translation = shifted_recon.reshape(batch_size, n_trans, shifted_recon.shape[-1])
    class_stride = int(n_rot_padded) * int(n_trans)
    winning_class = global_best_argmax // class_stride
    winner_within_class = global_best_argmax % class_stride
    winning_rot = winner_within_class // n_trans
    winning_trans = winner_within_class % n_trans
    in_block = (winning_rot >= r0) & (winning_rot < (r0 + int(actual_rot)))
    local_rot = jnp.clip(winning_rot - r0, 0, max(int(rotation_block_size), 1) - 1)
    flat_local = (
        (winning_class * int(rotation_block_size) + local_rot) * n_trans
        + winning_trans
    )
    probs = jax.nn.one_hot(
        flat_local,
        n_classes * int(rotation_block_size) * n_trans,
        dtype=shifted_recon.real.dtype,
    ).reshape(batch_size, n_classes, int(rotation_block_size), n_trans)
    probs = probs * in_block[:, None, None, None]
    summed = jnp.einsum(
        "bkrt,btn->krn",
        probs,
        shifted_by_translation,
        precision=jax.lax.Precision.HIGHEST,
    )
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs = jnp.einsum(
        "bkr,bn->krn",
        probs_sum_t,
        ctf2_over_nv,
        precision=jax.lax.Precision.HIGHEST,
    )
    rotation_sums = jnp.sum(probs, axis=(0, 3))
    return probs, summed, ctf_probs, rotation_sums


@partial(jax.jit, static_argnums=(4, 5, 6))
def _k_class_wta_image_m_step(
    shifted_recon,
    global_best_argmax,
    rotations_padded,
    ctf2_over_nv,
    n_rot_padded: int,
    n_trans: int,
    n_classes: int,
):
    """Pack global WTA winners as one adjoint row per image instead of per rotation."""

    batch_size = int(global_best_argmax.shape[0])
    shifted_by_translation = shifted_recon.reshape(batch_size, int(n_trans), shifted_recon.shape[-1])
    class_stride = int(n_rot_padded) * int(n_trans)
    winning_class = global_best_argmax // class_stride
    winner_within_class = global_best_argmax % class_stride
    winning_rot = winner_within_class // int(n_trans)
    winning_trans = winner_within_class % int(n_trans)

    image_rows = shifted_by_translation[jnp.arange(batch_size), winning_trans]
    class_mask = jax.nn.one_hot(
        winning_class,
        int(n_classes),
        dtype=image_rows.real.dtype,
    ).T
    summed_by_image = class_mask[:, :, None] * image_rows[None, :, :]
    ctf_by_image = class_mask.astype(ctf2_over_nv.dtype)[:, :, None] * ctf2_over_nv[None, :, :]
    winner_rots = rotations_padded[winning_rot]

    class_rot = winning_class * int(n_rot_padded) + winning_rot
    rotation_sums = jnp.sum(
        jax.nn.one_hot(
            class_rot,
            int(n_classes) * int(n_rot_padded),
            dtype=image_rows.real.dtype,
        ),
        axis=0,
    ).reshape(int(n_classes), int(n_rot_padded))
    return summed_by_image, ctf_by_image, winner_rots, rotation_sums


def _iter_rotation_blocks(n_blocks: int, rotation_block_size: int):
    for block_index in range(n_blocks):
        r0 = block_index * rotation_block_size
        yield block_index, r0, r0 + rotation_block_size


def _batch_image_count(batch) -> int:
    shape = getattr(batch, "shape", None)
    if shape is None:
        shape = np.asarray(batch).shape
    return int(shape[0])


def _select_class_or_shared(value, class_index: int, n_classes: int):
    value_array = jnp.asarray(value)
    if value_array.ndim >= 2 and int(value_array.shape[0]) == n_classes:
        return value_array[class_index]
    return value


def _prepare_projection_volumes(
    means,
    volume_shape,
    projection_padding_factor: int,
    do_gridding_correction: bool,
    current_size,
    precision_policy: DensePrecisionPolicy,
):
    if projection_padding_factor <= 1:
        return jnp.stack([precision_policy.cast_projection_volume(mean) for mean in means], axis=0), volume_shape

    from recovar.reconstruction.relion_functions import pad_volume_for_projection

    padded = []
    proj_volume_shape = None
    for mean in means:
        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
        padded.append(precision_policy.cast_projection_volume(mean_for_proj))
    return jnp.stack(padded, axis=0), proj_volume_shape


def _score_k_class_block(
    *,
    window_spec,
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_half_by_class,
    proj_abs2_by_class,
    half_weights,
    batch_size: int,
    n_classes: int,
    n_trans: int,
    image_shape,
    volume_shape,
    score_mode: str,
    precision_policy: DensePrecisionPolicy,
):
    rotation_block_size = int(proj_half_by_class.shape[1])
    scores = _score_rotation_block(
        window_spec,
        shifted_score=shifted_windowed,
        batch_norm=batch_norm,
        score_weight=ctf2_over_nv_windowed,
        proj_half=proj_half_by_class.reshape(n_classes * rotation_block_size, proj_half_by_class.shape[-1]),
        proj_abs2_half=proj_abs2_by_class.reshape(n_classes * rotation_block_size, proj_abs2_by_class.shape[-1]),
        half_weights=half_weights,
        n_images=batch_size,
        n_trans=n_trans,
        image_shape=image_shape,
        volume_shape=volume_shape,
        score_mode=score_mode,
        precision_policy=precision_policy,
    )
    return scores.reshape(batch_size, n_classes, rotation_block_size, n_trans)


def _project_k_class_block(
    mean_for_proj_by_class,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type: str,
    projection_kwargs: dict,
):
    """Project every class volume for one rotation block in one JAX transform."""

    def _project_one_class(mean_for_proj):
        return _compute_projections_block(
            mean_for_proj,
            rotations_block,
            image_shape,
            proj_volume_shape,
            disc_type,
            **projection_kwargs,
        )

    return jax.vmap(_project_one_class)(mean_for_proj_by_class)


def _accumulate_k_class_adjoint(
    Ft_y,
    Ft_ctf,
    summed_half,
    ctf_probs_half,
    rotations_block,
    *,
    window_spec,
    current_size,
    image_shape,
    recon_volume_shape,
):
    """Backproject per-class adjoint slices through the active Fourier layout."""

    if window_spec.use_window:
        max_r = float(current_size // 2)
        Ft_y = _batch_adjoint_slice_volume_windowed(
            summed_half,
            window_spec.recon_indices,
            rotations_block,
            Ft_y,
            image_shape,
            recon_volume_shape,
            "linear_interp",
            True,
            False,
            max_r,
        )
        Ft_ctf = _batch_adjoint_slice_volume_windowed(
            ctf_probs_half,
            window_spec.recon_indices,
            rotations_block,
            Ft_ctf,
            image_shape,
            recon_volume_shape,
            "linear_interp",
            True,
            False,
            max_r,
        )
        return Ft_y, Ft_ctf

    Ft_y = _batch_adjoint_slice_volume_half(
        summed_half,
        rotations_block,
        Ft_y,
        image_shape,
        recon_volume_shape,
        "linear_interp",
        True,
    )
    Ft_ctf = _batch_adjoint_slice_volume_half(
        ctf_probs_half,
        rotations_block,
        Ft_ctf,
        image_shape,
        recon_volume_shape,
        "linear_interp",
        True,
    )
    return Ft_y, Ft_ctf


def run_dense_k_class_em_native(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    class_log_priors,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int = None,
    rotation_log_prior=None,
    class_rotation_log_prior=None,
    translation_log_prior=None,
    image_indices=None,
    rotation_translation_mask=None,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    translation_prior_centers=None,
    relion_firstiter_score_mode: str = "gaussian",
    relion_firstiter_winner_take_all: bool = False,
    use_float64_scoring: bool = False,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    sparse_pass2: bool = False,
    accumulate_noise: bool = False,
    return_profile: bool = False,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    cache_projection_blocks: bool | None = None,
) -> DenseKClassNativeOutputs:
    """Run dense K-class EM with one E-step over class and pose axes."""

    if sparse_pass2:
        raise NotImplementedError("native dense K-class does not yet support sparse pass-2 skipping")
    if reconstruction_group_ids is not None and relion_firstiter_winner_take_all:
        raise NotImplementedError("dense K-class reconstruction groups are only implemented for soft M-step")
    if relion_firstiter_score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            "relion_firstiter_score_mode must be 'gaussian' or 'normalized_cc', "
            f"got {relion_firstiter_score_mode!r}",
        )

    overall_t0 = time.time()
    sync_timers = bool(return_profile)
    profile = {
        "setup_s": 0.0,
        "batch_fetch_s": 0.0,
        "preprocess_s": 0.0,
        "projection_pass1_s": 0.0,
        "projection_pass1_enqueue_s": 0.0,
        "projection_pass1_sync_s": 0.0,
        "score_pass1_s": 0.0,
        "projection_pass2_s": 0.0,
        "projection_pass2_enqueue_s": 0.0,
        "projection_pass2_sync_s": 0.0,
        "projection_cache_hits": 0,
        "projection_cache_misses": 0,
        "score_mstep_pass2_s": 0.0,
        "wta_mstep_s": 0.0,
        "adjoint_s": 0.0,
        "noise_s": 0.0,
        "host_accumulate_s": 0.0,
        "postprocess_s": 0.0,
        "new_means_s": 0.0,
        "final_sync_s": 0.0,
        "batches": 0,
        "rotation_blocks": int((int(rotations.shape[0]) + int(rotation_block_size) - 1) // int(rotation_block_size)),
    }
    means_array = jnp.asarray(means)
    n_classes = int(means_array.shape[0])
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    image_indices = np.arange(experiment_dataset.n_units) if image_indices is None else np.asarray(image_indices)
    n_images = int(image_indices.size)
    grouped_reconstruction = reconstruction_group_ids is not None
    if grouped_reconstruction:
        reconstruction_group_ids = np.asarray(reconstruction_group_ids, dtype=np.int32)
        if reconstruction_group_ids.shape != (n_images,):
            raise ValueError(
                "reconstruction_group_ids must have one entry per selected image: "
                f"expected ({n_images},), got {reconstruction_group_ids.shape}",
            )
        if reconstruction_group_count is None:
            reconstruction_group_count = int(np.max(reconstruction_group_ids)) + 1 if n_images else 0
        reconstruction_group_count = int(reconstruction_group_count)
        if reconstruction_group_count <= 0:
            raise ValueError("reconstruction_group_count must be positive when reconstruction_group_ids is set")
        if np.any(reconstruction_group_ids < 0) or np.any(reconstruction_group_ids >= reconstruction_group_count):
            raise ValueError("reconstruction_group_ids contains entries outside reconstruction_group_count")
    else:
        reconstruction_group_ids = None
        reconstruction_group_count = 0
    noise_variance_array = jnp.asarray(noise_variance)
    if noise_variance_array.ndim >= 2 and int(noise_variance_array.shape[0]) == n_classes:
        raise NotImplementedError("native dense K-class does not yet support class-specific noise variance")
    class_log_priors = jnp.asarray(class_log_priors, dtype=jnp.float32)
    if class_log_priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {class_log_priors.shape}")

    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance_array, image_shape).squeeze()

    precision_policy = DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    mean_for_proj_by_class, proj_volume_shape = _prepare_projection_volumes(
        means_array,
        volume_shape,
        projection_padding_factor,
        do_gridding_correction,
        current_size,
        precision_policy,
    )

    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
        recon_volume_size = int(np.prod(recon_volume_shape))
    else:
        recon_volume_shape = volume_shape
        recon_volume_size = int(np.prod(volume_shape))

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    ).astype(precision_policy.score_real_dtype)
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    projection_kwargs = window_spec.projection_kwargs()
    n_recon_windowed = window_spec.n_recon
    score_dc_index = half_spectrum_dc_index(image_shape)
    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations.shape[1],
        max_image_index=int(np.max(image_indices)) if image_indices.size else None,
    )

    if cache_projection_blocks is None:
        # Windowed RELION-style runs keep projection tensors compact and reuse
        # them across image batches and pass 2.  Full-resolution K-class
        # projections can be multi-GB, so do not cache those unless requested.
        cache_projection_blocks = bool(window_spec.use_window and not relion_firstiter_winner_take_all)
    cache_projection_blocks = bool(cache_projection_blocks)

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

    score_constraints = _KClassScoreConstraints.from_inputs(
        n_classes=n_classes,
        n_images=n_images,
        n_rot=n_rot,
        n_trans=n_trans,
        n_rot_padded=n_rot_padded,
        rotation_log_prior=rotation_log_prior,
        class_rotation_log_prior=class_rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_translation_mask=rotation_translation_mask,
    )

    Ft_y = jnp.zeros((n_classes, recon_volume_size), dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros((n_classes, recon_volume_size), dtype=experiment_dataset.dtype)
    grouped_Ft_y = None
    grouped_Ft_ctf = None
    if grouped_reconstruction:
        grouped_shape = (reconstruction_group_count, n_classes, recon_volume_size)
        grouped_Ft_y = jnp.zeros(grouped_shape, dtype=experiment_dataset.dtype)
        grouped_Ft_ctf = jnp.zeros(grouped_shape, dtype=experiment_dataset.dtype)
    hard_assignments = np.empty((n_classes, n_images), dtype=np.int32)
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float32)
    class_best_log_score = np.empty((n_classes, n_images), dtype=np.float32)
    class_max_posterior = np.empty((n_classes, n_images), dtype=np.float32)
    class_rotation_posterior_sums = np.zeros((n_classes, n_rot), dtype=np.float64)
    noise_state = None
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_windowed = window_spec.recon_values(noise_variance_half)
        noise_state = _DenseKClassNoiseState.zeros(n_classes=n_classes, n_shells=n_shells)

    start_idx = 0
    batch_iter = experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    )
    projection_cache = [None] * n_blocks if cache_projection_blocks else None

    def project_block(block_index: int, rotations_block):
        nonlocal projection_cache
        if projection_cache is not None:
            cached = projection_cache[int(block_index)]
            if cached is not None:
                profile["projection_cache_hits"] += 1
                return cached
        projected = _project_k_class_block(
            mean_for_proj_by_class,
            rotations_block,
            image_shape,
            proj_volume_shape,
            disc_type,
            projection_kwargs,
        )
        if projection_cache is not None:
            projection_cache[int(block_index)] = projected
        profile["projection_cache_misses"] += 1
        return projected

    use_host_half_preprocess = backend_half_preprocess_uses_host_images(experiment_dataset)
    device_half_mask = None
    device_half_mask_mode = None
    if use_host_half_preprocess and relion_firstiter_score_mode == "gaussian" and score_with_masked_images:
        device_half_mask, device_half_mask_mode = resolve_image_mask_for_half_preprocess(
            experiment_dataset,
            image_shape,
            require_mask=True,
        )
        use_host_half_preprocess = False
    if sync_timers:
        profile["setup_s"] += time.time() - overall_t0
    while True:
        try:
            t0 = time.time()
            batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
            if sync_timers:
                profile["batch_fetch_s"] += time.time() - t0
        except StopIteration:
            break

        preprocess_t0 = time.time()
        actual_batch_size = len(indices)
        batch_indices_np = np.asarray(indices)
        end_idx = start_idx + actual_batch_size
        batch_size = _batch_image_count(batch_data)
        if grouped_reconstruction:
            batch_group_ids = jnp.asarray(reconstruction_group_ids[start_idx:end_idx], dtype=jnp.int32)
            if int(batch_group_ids.shape[0]) > int(batch_size):
                raise ValueError("batch data has fewer rows than returned image indices")
            if int(batch_group_ids.shape[0]) < int(batch_size):
                batch_group_ids = jnp.pad(
                    batch_group_ids,
                    (0, int(batch_size) - int(batch_group_ids.shape[0])),
                    constant_values=0,
                )
        else:
            batch_group_ids = None
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, indices, batch=batch_data)
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)

        preprocess_batch_data = batch_data if use_host_half_preprocess else jnp.asarray(batch_data)
        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                batch_indices_np,
                batch_size=actual_batch_size,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                translations,
                centers,
                experiment_dataset.voxel_size,
            )

        processed_score_half_for_noise = None
        if relion_firstiter_score_mode == "normalized_cc":
            shifted_half, batch_norm, ctf2_half_score, ctf2_over_nv_half = _preprocess_batch_firstiter_cc(
                experiment_dataset,
                preprocess_batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
            )
            shifted_recon_half = _prepare_reconstruction_batch(
                experiment_dataset,
                preprocess_batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
            )
        elif score_with_masked_images:
            if device_half_mask is None:
                (
                    processed_score_half,
                    processed_recon_half,
                    ctf_half,
                    noise_variance_raw_half,
                    translation_phases_half,
                ) = dense_batch_half_input_pair(
                    experiment_dataset,
                    preprocess_batch_data,
                    ctf_params,
                    noise_variance_half,
                    translations,
                    config,
                    apply_image_mask_a=True,
                    apply_image_mask_b=False,
                )
            else:
                processed_score_half, processed_recon_half = preprocess_half_image_pair_device(
                    preprocess_batch_data,
                    device_half_mask,
                    config,
                    apply_image_mask_a=True,
                    apply_image_mask_b=False,
                    mask_mode=device_half_mask_mode,
                )
                ctf_half = config.compute_ctf_half(ctf_params)
                noise_variance_raw_half = jnp.asarray(noise_variance_half)
                translation_phases_half = half_translation_phase_table(translations, config.image_shape)
                processed_score_half_for_noise = processed_score_half
            score_weighted_half = processed_score_half * ctf_half / noise_variance_raw_half
            shifted_half = apply_half_translation_phases(score_weighted_half, translation_phases_half)
            half_weights_for_norm = make_scoring_half_image_weights(
                image_shape,
                relion_half_sum=False,
            )
            batch_norm = jnp.sum(
                (jnp.abs(processed_score_half) ** 2 / noise_variance_raw_half) * half_weights_for_norm[None, :],
                axis=-1,
                keepdims=True,
            ).real
            ctf2_over_nv_half = ctf_half**2 / noise_variance_raw_half
            ctf2_half_score = None
            shifted_recon_half = apply_half_translation_phases(
                processed_recon_half * ctf_half / noise_variance_raw_half,
                translation_phases_half,
            )
        else:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
                experiment_dataset,
                preprocess_batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
            )
            ctf2_half_score = None
            shifted_recon_half = shifted_half

        if scale_corrections is not None:
            batch_scale = jnp.asarray(np.asarray(scale_corrections, dtype=np.float32)[batch_indices_np])
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)

        if image_corrections is not None:
            batch_corr = jnp.asarray(np.asarray(image_corrections, dtype=np.float32)[batch_indices_np])
            image_only_corr = batch_corr / batch_scale
            if relion_firstiter_score_mode == "normalized_cc":
                score_batch_corr = batch_corr / (batch_scale**2)
                norm_batch_corr = image_only_corr
            else:
                score_batch_corr = batch_corr
                norm_batch_corr = image_only_corr
            shifted_half = shifted_half * jnp.repeat(score_batch_corr, n_trans)[:, None]
            shifted_recon_half = shifted_recon_half * jnp.repeat(batch_corr, n_trans)[:, None]
            batch_norm = batch_norm * (norm_batch_corr**2)[:, None]

        if scale_corrections is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if ctf2_half_score is not None:
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts_np = np.asarray(image_pre_shifts, dtype=np.float32)[batch_indices_np]
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts_np, n_trans)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded

        if relion_firstiter_score_mode == "normalized_cc":
            score_weight_half = ctf2_half_score / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))
            shifted_score_half = shifted_half * jnp.repeat(score_weight_half, n_trans, axis=0)
        else:
            score_weight_half = ctf2_over_nv_half
            shifted_score_half = shifted_half

        shifted_half_with_dc = shifted_score_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half
        if half_spectrum_scoring and relion_firstiter_score_mode != "normalized_cc":
            shifted_score_half = shifted_score_half.at[:, score_dc_index].set(0.0)
            score_weight_half = score_weight_half.at[:, score_dc_index].set(0.0)

        shifted_score_half, score_weight_half, shifted_recon_half = precision_policy.cast_scoring_inputs(
            shifted_score_half,
            score_weight_half,
            shifted_recon_half,
        )
        shifted_windowed = window_spec.score_values(shifted_score_half)
        shifted_recon_windowed = window_spec.recon_values(shifted_recon_half)
        ctf2_over_nv_windowed = window_spec.score_values(score_weight_half)
        ctf2_over_nv_windowed_mstep = window_spec.recon_values(ctf2_over_nv_half_with_dc)
        shifted_masked_for_noise = None
        if accumulate_noise:
            shifted_masked_for_noise = window_spec.recon_values(shifted_half_with_dc)
            if processed_score_half_for_noise is None:
                processed_masked_half = process_half_image(
                    experiment_dataset,
                    preprocess_batch_data,
                    score_with_masked_images,
                )
            else:
                processed_masked_half = processed_score_half_for_noise
            if image_corrections is not None:
                processed_masked_half = processed_masked_half * image_only_corr[:, None]
        if sync_timers:
            _block_until_ready(
                shifted_windowed,
                shifted_recon_windowed,
                ctf2_over_nv_windowed,
                ctf2_over_nv_windowed_mstep,
            )
            profile["preprocess_s"] += time.time() - preprocess_t0

        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=precision_policy.normalization_real_dtype)
        class_max_s = jnp.full((batch_size, n_classes), -jnp.inf)
        class_sum_exp = jnp.zeros((batch_size, n_classes), dtype=precision_policy.normalization_real_dtype)
        global_best_score = jnp.full(batch_size, -jnp.inf)
        global_best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        best_score_class = jnp.full((batch_size, n_classes), -jnp.inf)
        best_argmax_class = jnp.zeros((batch_size, n_classes), dtype=jnp.int32)

        profile["batches"] += 1
        for block_index, r0, r1 in _iter_rotation_blocks(n_blocks, rotation_block_size):
            rots_b = rotations_padded[r0:r1]
            t0 = time.time()
            proj_half_by_class, proj_abs2_by_class = project_block(block_index, rots_b)
            if sync_timers:
                projection_enqueue_s = time.time() - t0
                _block_until_ready(proj_half_by_class, proj_abs2_by_class)
                projection_total_s = time.time() - t0
                profile["projection_pass1_s"] += projection_total_s
                profile["projection_pass1_enqueue_s"] += projection_enqueue_s
                profile["projection_pass1_sync_s"] += projection_total_s - projection_enqueue_s
                t0 = time.time()
            scores = _score_k_class_block(
                window_spec=window_spec,
                shifted_windowed=shifted_windowed,
                batch_norm=batch_norm,
                ctf2_over_nv_windowed=ctf2_over_nv_windowed,
                proj_half_by_class=proj_half_by_class,
                proj_abs2_by_class=proj_abs2_by_class,
                half_weights=half_weights,
                batch_size=batch_size,
                n_classes=n_classes,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode=relion_firstiter_score_mode,
                precision_policy=precision_policy,
            )
            (
                rotation_prior_block,
                translation_prior_block,
                candidate_mask_block,
                valid_rotation_mask,
            ) = score_constraints.block_inputs(
                r0=r0,
                r1=r1,
                start=start_idx,
                end=end_idx,
                batch_count=batch_size,
                rotation_block_size=rotation_block_size,
            )
            scores = _apply_k_class_score_constraints(
                scores,
                rotation_prior_block,
                translation_prior_block,
                class_log_priors,
                candidate_mask_block,
                valid_rotation_mask,
                score_mode=relion_firstiter_score_mode,
            )
            if relion_firstiter_winner_take_all:
                (
                    max_s,
                    sum_exp,
                    class_max_s,
                    class_sum_exp,
                    best_score_class,
                    best_argmax_class,
                    global_best_score,
                    global_best_argmax,
                ) = _update_k_class_wta_pass1(
                    max_s,
                    sum_exp,
                    class_max_s,
                    class_sum_exp,
                    best_score_class,
                    best_argmax_class,
                    global_best_score,
                    global_best_argmax,
                    scores,
                    r0,
                    n_rot_padded,
                    n_trans,
                )
            else:
                max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)
                class_max_s, class_sum_exp = _update_class_logsumexp(class_max_s, class_sum_exp, scores)
            if sync_timers:
                _block_until_ready(max_s, sum_exp, class_max_s, class_sum_exp)
                profile["score_pass1_s"] += time.time() - t0

        log_Z = max_s + jnp.log(sum_exp)
        class_log_Z = class_max_s + jnp.log(class_sum_exp)
        if not relion_firstiter_winner_take_all:
            best_score_class = jnp.full((batch_size, n_classes), -jnp.inf)
            best_argmax_class = jnp.zeros((batch_size, n_classes), dtype=jnp.int32)

        if relion_firstiter_winner_take_all and not accumulate_noise:
            shifted_for_mstep = shifted_recon_windowed if window_spec.use_window else shifted_recon_half
            ctf_for_mstep = ctf2_over_nv_windowed_mstep if window_spec.use_window else ctf2_over_nv_half_with_dc
            t0 = time.time()
            summed_half, ctf_probs_half, winner_rots, rotation_sums = _k_class_wta_image_m_step(
                shifted_for_mstep,
                global_best_argmax,
                jnp.asarray(rotations_padded),
                ctf_for_mstep,
                n_rot_padded,
                n_trans,
                n_classes,
            )
            if sync_timers:
                _block_until_ready(summed_half, ctf_probs_half, rotation_sums)
                profile["wta_mstep_s"] += time.time() - t0
            t0 = time.time()
            class_rotation_posterior_sums += np.asarray(
                rotation_sums[:, :n_rot],
                dtype=np.float64,
            )
            if sync_timers:
                profile["host_accumulate_s"] += time.time() - t0
            t0 = time.time()
            Ft_y, Ft_ctf = _accumulate_k_class_adjoint(
                Ft_y,
                Ft_ctf,
                summed_half,
                ctf_probs_half,
                winner_rots,
                window_spec=window_spec,
                current_size=current_size,
                image_shape=image_shape,
                recon_volume_shape=recon_volume_shape,
            )
            if sync_timers:
                _block_until_ready(Ft_y, Ft_ctf)
                profile["adjoint_s"] += time.time() - t0
        else:
            for block_index, r0, r1 in _iter_rotation_blocks(n_blocks, rotation_block_size):
                rots_b = rotations_padded[r0:r1]
                proj_half_by_class = proj_abs2_by_class = None
                if (not relion_firstiter_winner_take_all) or accumulate_noise:
                    t0 = time.time()
                    proj_half_by_class, proj_abs2_by_class = project_block(block_index, rots_b)
                    if sync_timers:
                        projection_enqueue_s = time.time() - t0
                        _block_until_ready(proj_half_by_class, proj_abs2_by_class)
                        projection_total_s = time.time() - t0
                        profile["projection_pass2_s"] += projection_total_s
                        profile["projection_pass2_enqueue_s"] += projection_enqueue_s
                        profile["projection_pass2_sync_s"] += projection_total_s - projection_enqueue_s
                if not relion_firstiter_winner_take_all:
                    t0 = time.time()
                    scores = _score_k_class_block(
                        window_spec=window_spec,
                        shifted_windowed=shifted_windowed,
                        batch_norm=batch_norm,
                        ctf2_over_nv_windowed=ctf2_over_nv_windowed,
                        proj_half_by_class=proj_half_by_class,
                        proj_abs2_by_class=proj_abs2_by_class,
                        half_weights=half_weights,
                        batch_size=batch_size,
                        n_classes=n_classes,
                        n_trans=n_trans,
                        image_shape=image_shape,
                        volume_shape=volume_shape,
                        score_mode=relion_firstiter_score_mode,
                        precision_policy=precision_policy,
                    )
                    (
                        rotation_prior_block,
                        translation_prior_block,
                        candidate_mask_block,
                        valid_rotation_mask,
                    ) = score_constraints.block_inputs(
                        r0=r0,
                        r1=r1,
                        start=start_idx,
                        end=end_idx,
                        batch_count=batch_size,
                        rotation_block_size=rotation_block_size,
                    )
                    scores = _apply_k_class_score_constraints(
                        scores,
                        rotation_prior_block,
                        translation_prior_block,
                        class_log_priors,
                        candidate_mask_block,
                        valid_rotation_mask,
                        score_mode=relion_firstiter_score_mode,
                    )
                shifted_for_mstep = shifted_recon_windowed if window_spec.use_window else shifted_recon_half
                ctf_for_mstep = ctf2_over_nv_windowed_mstep if window_spec.use_window else ctf2_over_nv_half_with_dc
                if relion_firstiter_winner_take_all:
                    actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                    probs, summed_half, ctf_probs_half, rotation_sums = _k_class_wta_m_step_block(
                        shifted_for_mstep,
                        global_best_argmax,
                        r0,
                        actual_rot,
                        rotation_block_size,
                        n_rot_padded,
                        ctf_for_mstep,
                        n_trans,
                        n_classes,
                    )
                else:
                    if grouped_reconstruction:
                        (
                            probs,
                            block_best_class,
                            block_argmax_class,
                            _block_best_global,
                            _block_argmax_global,
                            grouped_summed_half,
                            grouped_ctf_probs_half,
                            rotation_sums,
                        ) = _grouped_k_class_m_step_block(
                            shifted_for_mstep,
                            scores,
                            log_Z,
                            ctf_for_mstep,
                            batch_group_ids,
                            n_trans,
                            reconstruction_group_count,
                        )
                        grouped_ready_values = (grouped_summed_half, grouped_ctf_probs_half)
                        adjoint_ready_values = grouped_ready_values
                    else:
                        (
                            probs,
                            block_best_class,
                            block_argmax_class,
                            _block_best_global,
                            _block_argmax_global,
                            summed_half,
                            ctf_probs_half,
                            rotation_sums,
                        ) = _k_class_m_step_block(
                            shifted_for_mstep,
                            scores,
                            log_Z,
                            ctf_for_mstep,
                            n_trans,
                        )
                        grouped_ready_values = ()
                        adjoint_ready_values = (summed_half, ctf_probs_half)
                    improved = block_best_class > best_score_class
                    best_score_class = jnp.where(improved, block_best_class, best_score_class)
                    best_argmax_class = jnp.where(improved, block_argmax_class + r0 * n_trans, best_argmax_class)
                    if sync_timers:
                        _block_until_ready(
                            probs,
                            *adjoint_ready_values,
                            rotation_sums,
                            best_score_class,
                            best_argmax_class,
                        )
                        profile["score_mstep_pass2_s"] += time.time() - t0

                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                if actual_rot > 0:
                    t0 = time.time()
                    class_rotation_posterior_sums[:, r0 : r0 + actual_rot] += np.asarray(
                        rotation_sums[:, :actual_rot],
                        dtype=np.float64,
                    )
                    if sync_timers:
                        profile["host_accumulate_s"] += time.time() - t0
                if accumulate_noise:
                    t0 = time.time()
                    class_mass = jnp.sum(probs, axis=(2, 3))
                    class_img_power = jnp.einsum(
                        "bk,bn->kn",
                        class_mass,
                        jnp.abs(processed_masked_half) ** 2,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    batch_img_power_shells = jax.vmap(
                        lambda values: jnp.zeros(n_shells, dtype=jnp.float32).at[shell_indices_half].add(
                            values.astype(jnp.float32),
                        )
                    )(class_img_power)
                    noise_state.add_image_power(batch_img_power_shells, class_mass)
                    noise_state.add_translation_offset(probs, translation_sqdist_ang)

                    shifted_by_translation_noise = shifted_masked_for_noise.reshape(
                        batch_size,
                        n_trans,
                        shifted_masked_for_noise.shape[-1],
                    )
                    summed_masked_noise = jnp.einsum(
                        "bkrt,btn->krn",
                        probs,
                        shifted_by_translation_noise,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    probs_sum_t_noise = jnp.sum(probs, axis=-1)
                    if window_spec.use_window:
                        ctf2_nv_noise = window_spec.recon_values(ctf2_over_nv_half_with_dc)
                        ctf_probs_for_noise = jnp.einsum(
                            "bkr,bn->krn",
                            probs_sum_t_noise,
                            ctf2_nv_noise,
                            precision=jax.lax.Precision.HIGHEST,
                        )
                        nv_for_noise = noise_variance_windowed
                        si_for_noise = shell_indices_noise
                        proj_for_noise = window_spec.recon_values(proj_half_by_class)
                        proj_abs2_for_noise = window_spec.recon_values(proj_abs2_by_class)
                    else:
                        ctf_probs_for_noise = jnp.einsum(
                            "bkr,bn->krn",
                            probs_sum_t_noise,
                            ctf2_over_nv_half_with_dc,
                            precision=jax.lax.Precision.HIGHEST,
                        )
                        nv_for_noise = noise_variance_half
                        si_for_noise = shell_indices_noise
                        proj_for_noise = proj_half_by_class
                        proj_abs2_for_noise = proj_abs2_by_class

                    block_noise_shells = []
                    for class_index in range(n_classes):
                        class_noise_shells, _, _ = _compute_noise_block(
                            proj_for_noise[class_index],
                            proj_abs2_for_noise[class_index],
                            summed_masked_noise[class_index],
                            ctf_probs_for_noise[class_index],
                            nv_for_noise,
                            si_for_noise,
                            n_shells,
                            False,
                        )
                        block_noise_shells.append(class_noise_shells)
                    noise_state.add_noise_shells(block_noise_shells)
                    if sync_timers:
                        _block_until_ready(block_noise_shells)
                        profile["noise_s"] += time.time() - t0

                t0 = time.time()
                if grouped_reconstruction:
                    flat_grouped_Ft_y, flat_grouped_Ft_ctf = _accumulate_k_class_adjoint(
                        grouped_Ft_y.reshape(reconstruction_group_count * n_classes, recon_volume_size),
                        grouped_Ft_ctf.reshape(reconstruction_group_count * n_classes, recon_volume_size),
                        grouped_summed_half.reshape(
                            reconstruction_group_count * n_classes,
                            grouped_summed_half.shape[2],
                            grouped_summed_half.shape[3],
                        ),
                        grouped_ctf_probs_half.reshape(
                            reconstruction_group_count * n_classes,
                            grouped_ctf_probs_half.shape[2],
                            grouped_ctf_probs_half.shape[3],
                        ),
                        rots_b,
                        window_spec=window_spec,
                        current_size=current_size,
                        image_shape=image_shape,
                        recon_volume_shape=recon_volume_shape,
                    )
                    grouped_Ft_y = flat_grouped_Ft_y.reshape(reconstruction_group_count, n_classes, recon_volume_size)
                    grouped_Ft_ctf = flat_grouped_Ft_ctf.reshape(
                        reconstruction_group_count,
                        n_classes,
                        recon_volume_size,
                    )
                    if sync_timers:
                        _block_until_ready(grouped_Ft_y, grouped_Ft_ctf)
                        profile["adjoint_s"] += time.time() - t0
                else:
                    Ft_y, Ft_ctf = _accumulate_k_class_adjoint(
                        Ft_y,
                        Ft_ctf,
                        summed_half,
                        ctf_probs_half,
                        rots_b,
                        window_spec=window_spec,
                        current_size=current_size,
                        image_shape=image_shape,
                        recon_volume_shape=recon_volume_shape,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_y, Ft_ctf)
                        profile["adjoint_s"] += time.time() - t0

        t0 = time.time()
        log_score_offset = -0.5 * jnp.squeeze(batch_norm[:actual_batch_size], axis=1)
        log_Z_actual = log_Z[:actual_batch_size]
        class_log_Z_actual = class_log_Z[:actual_batch_size]
        best_score_actual = best_score_class[:actual_batch_size]
        pmax_class = jnp.exp(best_score_class - log_Z[:, None])[:actual_batch_size]
        class_log_evidence[:, start_idx:end_idx] = np.asarray(
            (class_log_Z_actual + log_score_offset[:, None]).T,
            dtype=np.float32,
        )
        class_best_log_score[:, start_idx:end_idx] = np.asarray(
            (best_score_actual + log_score_offset[:, None]).T,
            dtype=np.float32,
        )
        class_max_posterior[:, start_idx:end_idx] = np.asarray(pmax_class.T, dtype=np.float32)
        hard_assignments[:, start_idx:end_idx] = np.asarray(best_argmax_class[:actual_batch_size].T, dtype=np.int32)
        if sync_timers:
            profile["postprocess_s"] += time.time() - t0
        start_idx = end_idx

    if grouped_reconstruction:
        Ft_y = jnp.sum(grouped_Ft_y, axis=0)
        Ft_ctf = jnp.sum(grouped_Ft_ctf, axis=0)

    from recovar.reconstruction import relion_functions

    t0 = time.time()
    if reconstruction_padding_factor > 1:
        new_means = [None for _ in range(n_classes)]
    else:
        new_means = [
            relion_functions.post_process_from_filter(
                experiment_dataset,
                Ft_ctf[class_index],
                Ft_y[class_index],
                tau=_select_class_or_shared(mean_variance, class_index, n_classes),
                disc_type=disc_type,
            ).reshape(-1)
            for class_index in range(n_classes)
        ]
    if sync_timers:
        profile["new_means_s"] += time.time() - t0

    t0 = time.time()
    if grouped_reconstruction:
        _block_until_ready(Ft_y, Ft_ctf, grouped_Ft_y, grouped_Ft_ctf)
    else:
        _block_until_ready(Ft_y, Ft_ctf)
    if sync_timers:
        profile["final_sync_s"] += time.time() - t0
    elapsed_s = time.time() - overall_t0
    profile_summary = None
    if return_profile:
        accounted_s = sum(float(profile[name]) for name in (
            "setup_s",
            "batch_fetch_s",
            "preprocess_s",
            "projection_pass1_s",
            "score_pass1_s",
            "projection_pass2_s",
            "score_mstep_pass2_s",
            "wta_mstep_s",
            "adjoint_s",
            "noise_s",
            "host_accumulate_s",
            "postprocess_s",
            "new_means_s",
            "final_sync_s",
        ))
        profile_summary = {
            **profile,
            "em_time_s": float(elapsed_s),
            "accounted_em_time_s": float(accounted_s),
            "unattributed_em_time_s": float(max(elapsed_s - accounted_s, 0.0)),
            "n_images": int(n_images),
            "n_classes": int(n_classes),
            "n_rotations": int(n_rot),
            "n_translations": int(n_trans),
            "image_batch_size": int(image_batch_size),
            "rotation_block_size": int(rotation_block_size),
            "current_size": None if current_size is None else int(current_size),
            "winner_take_all_mstep": bool(relion_firstiter_winner_take_all),
            "accumulate_noise": bool(accumulate_noise),
            "reconstruction_group_count": int(reconstruction_group_count),
            "cache_projection_blocks": bool(cache_projection_blocks),
        }
    logger.info(
        "Native dense K-class EM completed: K=%d images=%d rotations=%d translations=%d elapsed=%.1fs",
        n_classes,
        n_images,
        n_rot,
        n_trans,
        elapsed_s,
    )
    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=class_log_evidence[class_index],
            best_log_score_per_image=class_best_log_score[class_index],
            max_posterior_per_image=class_max_posterior[class_index],
            rotation_posterior_sums=class_rotation_posterior_sums[class_index],
        )
        for class_index in range(n_classes)
    )
    noise_stats = noise_state.stats() if noise_state is not None else None
    return DenseKClassNativeOutputs(
        class_log_evidence=class_log_evidence,
        new_means=new_means,
        Ft_y=[Ft_y[class_index] for class_index in range(n_classes)],
        Ft_ctf=[Ft_ctf[class_index] for class_index in range(n_classes)],
        hard_assignments=hard_assignments,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        profile_summary=profile_summary,
        grouped_Ft_y=grouped_Ft_y,
        grouped_Ft_ctf=grouped_Ft_ctf,
    )
