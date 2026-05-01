"""Sparse / local pose-marginalized PPCA E-step (Milestone 6).

Replaces dense exhaustive enumeration with a flat per-hypothesis layout:
each image carries a variable number of (rotation, translation)
hypotheses, and aggregations across hypotheses are done with
``jax.ops.segment_sum`` keyed by the image index.

This is the engine for both:

  * **Mode A (coarse-to-fine):** score a coarse grid → keep significant
    parents → expand to oversampled children → score with the same
    augmented PPCA score function.
  * **Mode B (local refinement):** start from current hard poses, build
    a local angular + shift neighborhood per image, score with the same
    function, update hard poses from posterior maxima.

Both modes share this engine — they differ only in how the
:class:`SparseHypothesisLayout` is built upstream.

Non-negotiable #8 (CLAUDE.md): the sparse engine MUST call the same
score function as the dense engine
(``recovar.ppca.pose_marginal.compute_ppca_pose_scores_and_moments_no_contrast``).
The unit test ``test_unpruned_sparse_equals_dense`` enforces this gate.

Non-negotiable #9: pruning may restrict the hypothesis support, but it
must NOT alter per-hypothesis scores. Significance pruning happens at
the layout-construction layer, not inside the score function.

For production we will adapt
``recovar.em.dense_single_volume.local_layout.LocalHypothesisLayout`` →
:class:`SparseHypothesisLayout` (the conversion lands at M10 alongside
the dataset wiring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from recovar.ppca.pose_marginal import (
    compute_ppca_pose_scores_and_moments_no_contrast,
)

from .dense_engine import DenseImageStats

__all__ = [
    "SparseHypothesisLayout",
    "SparsePosteriorDiagnostics",
    "sparse_pose_ppca_E_step_flat",
]


@dataclass(frozen=True)
class SparseHypothesisLayout:
    """Flat per-hypothesis layout for the sparse engine.

    All arrays have leading axis ``Nh`` = total number of hypotheses
    (summed across all images in the block). The image each hypothesis
    belongs to is given by ``image_id_flat``.

    Attributes
    ----------
    Y1:
        ``[Nh, F]`` complex64. Per-hypothesis pre-shifted CTF-weighted
        whitened image (for hypothesis ``h``,
        ``Y1[h] = (C_{i(h)} · y_{i(h)} · phase_{t(h)}) / σ²_{i(h)}``).
    proj_aug:
        ``[Nh, P, F]`` complex64. Per-hypothesis augmented template
        ``[μ, W₁, …, W_q]`` projected by the hypothesis's rotation. Note
        each hypothesis carries its own ``proj_aug`` slice — duplicates
        are common when many hypotheses share a rotation, and the layout
        builder is responsible for any deduplication strategy upstream.
    ctf2_over_noise:
        ``[Nh, F]`` real32. Per-hypothesis ``C² / σ²`` (typically constant
        across hypotheses sharing an image).
    y_norm:
        ``[Nh]`` real32. Per-hypothesis ``y_norm`` (constant across
        hypotheses sharing an image).
    pose_log_prior:
        ``[Nh]`` real32 or None. Per-hypothesis ``log π_ia``.
    image_id:
        ``[Nh]`` int32. Image index ``i`` for each hypothesis.
    n_images:
        Number of distinct images in the block (``image_id`` values
        range over ``0 .. n_images-1``).
    """

    Y1: jax.Array
    proj_aug: jax.Array
    ctf2_over_noise: jax.Array
    y_norm: jax.Array
    pose_log_prior: jax.Array | None
    image_id: jax.Array
    n_images: int


class SparsePosteriorDiagnostics(NamedTuple):
    logZ: jax.Array  # [n_images]
    pmax: jax.Array  # [n_images]
    n_significant_per_image: jax.Array  # [n_images]
    omitted_log_mass: jax.Array  # [n_images] — log(1 - Σ γ inside support)
    best_hypothesis_idx: jax.Array  # [n_images] int32 — flat index into Nh per image's argmax


def _segmented_logsumexp(scores, image_id, n_images):
    """Compute logsumexp per image segment.

    Implementation: subtract per-image max for stability, exp, segment-sum,
    log + max.
    """
    # Per-image max (broadcast back).
    seg_max = jax.ops.segment_max(scores, image_id, num_segments=n_images)  # [n_images]
    seg_max_per_h = seg_max[image_id]  # [Nh]
    exp_shifted = jnp.exp(scores - seg_max_per_h)  # [Nh]
    seg_sum = jax.ops.segment_sum(exp_shifted, image_id, num_segments=n_images)  # [n_images]
    return jnp.log(seg_sum) + seg_max  # [n_images]


def sparse_pose_ppca_E_step_flat(
    layout: SparseHypothesisLayout,
    *,
    significance_threshold: float = 1e-3,
):
    """Two-pass sparse E-step on a flat hypothesis layout.

    Returns
    -------
    image_stats:
        :class:`DenseImageStats` (same NamedTuple — image-level moments
        accumulated over hypotheses inside support; the sparse engine
        emits the same image-level contract as the dense engine for
        downstream backprojection).
    diagnostics:
        :class:`SparsePosteriorDiagnostics` with per-image logZ, pmax,
        n_significant, omitted log mass (= 0 here for unpruned input;
        the prune-builder is responsible for setting an upper bound on
        omitted mass when constructing the layout), and the best
        hypothesis flat index per image.
    """
    Nh, F = layout.Y1.shape
    P = layout.proj_aug.shape[1]
    q = P - 1
    n_images = layout.n_images

    # K_aug per hypothesis: [Nh, P, P].
    K_aug = jnp.einsum(
        "hf, hpf, hqf -> hpq",
        layout.ctf2_over_noise.astype(layout.proj_aug.dtype),
        jnp.conj(layout.proj_aug),
        layout.proj_aug,
    )
    nu_mm = K_aug[..., 0, 0].real  # [Nh]
    h_zm = K_aug[..., 1:, 0]  # [Nh, q]
    Hzz = K_aug[..., 1:, 1:]  # [Nh, q, q]

    # First-order: D_h_p = sum_f conj(Y1)_h_f * proj_aug_h_p_f.
    D = jnp.einsum("hf, hpf -> hp", jnp.conj(layout.Y1), layout.proj_aug)
    t_mx = D[..., 0].real  # [Nh]
    g_zx = D[..., 1:]  # [Nh, q]

    # ---- Pass 1: scores + segmented logsumexp + per-image logZ + best ----
    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        layout.y_norm,
        t_mx,
        nu_mm,
        g_zx,
        h_zm,
        Hzz,
        return_moments=False,
    )  # [Nh]
    if layout.pose_log_prior is not None:
        score = score + layout.pose_log_prior

    logZ = _segmented_logsumexp(score, layout.image_id, n_images)  # [n_images]
    logZ_per_h = logZ[layout.image_id]  # [Nh]
    gamma_pass1 = jnp.exp(score - logZ_per_h)  # [Nh]

    pmax = jax.ops.segment_max(gamma_pass1, layout.image_id, num_segments=n_images)  # [n_images]

    significant_mask = (gamma_pass1 > significance_threshold).astype(jnp.int32)
    n_sig = jax.ops.segment_sum(significant_mask, layout.image_id, num_segments=n_images)

    # Argmax per image: implement via segment_argmax surrogate. We use
    # segment_max on the score, then locate the matching hypothesis per
    # image. For ties this returns the first match — same convention as
    # the dense engine's ``jnp.argmax``.
    seg_max_score = jax.ops.segment_max(score, layout.image_id, num_segments=n_images)
    seg_max_per_h = seg_max_score[layout.image_id]
    is_argmax = score >= seg_max_per_h - 1e-12
    # For each image, the FIRST hypothesis matching the max wins.
    first_matching_idx = _segment_first_true(is_argmax, layout.image_id, n_images)
    best_hyp_idx = first_matching_idx  # [n_images]

    # No omitted mass for unpruned layouts; pruners must adjust.
    omitted_log_mass = jnp.zeros((n_images,), dtype=jnp.float32)

    # ---- Pass 2: recompute moments + accumulate gamma·alpha, gamma·G_tri ----
    if q == 0:
        # Augmented moments are trivially [1] / tri([1]) = [1] regardless
        # of pose. After γ-weighted segment_sum (= 1 per image), each
        # image gets [1] in alpha_aug_acc and G_aug_tri_acc.
        ones = jnp.ones((n_images, 1), dtype=jnp.complex64)
        image_stats = DenseImageStats(
            alpha_aug_acc=ones,
            G_aug_tri_acc=ones,
            log_evidence=logZ,
        )
        return image_stats, SparsePosteriorDiagnostics(
            logZ=logZ,
            pmax=pmax,
            n_significant_per_image=n_sig,
            omitted_log_mass=omitted_log_mass,
            best_hypothesis_idx=best_hyp_idx,
        )

    score2, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        layout.y_norm,
        t_mx,
        nu_mm,
        g_zx,
        h_zm,
        Hzz,
        return_moments=True,
    )
    if layout.pose_log_prior is not None:
        score2 = score2 + layout.pose_log_prior
    gamma = jnp.exp(score2 - logZ_per_h)  # [Nh]

    # Weighted segment-sum into per-image accumulators.
    weight = gamma.astype(alpha.dtype)
    alpha_weighted = weight[:, None] * alpha  # [Nh, P]
    G_tri_weighted = weight[:, None] * G_tri  # [Nh, tri(P)]
    alpha_aug_acc = jax.ops.segment_sum(
        alpha_weighted,
        layout.image_id,
        num_segments=n_images,
    )  # [n_images, P]
    G_aug_tri_acc = jax.ops.segment_sum(
        G_tri_weighted,
        layout.image_id,
        num_segments=n_images,
    )  # [n_images, tri(P)]

    image_stats = DenseImageStats(
        alpha_aug_acc=alpha_aug_acc,
        G_aug_tri_acc=G_aug_tri_acc,
        log_evidence=logZ,
    )
    return image_stats, SparsePosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        n_significant_per_image=n_sig,
        omitted_log_mass=omitted_log_mass,
        best_hypothesis_idx=best_hyp_idx,
    )


def _segment_first_true(mask, image_id, n_images):
    """Return the first index in each segment where ``mask`` is True.

    Implementation: positions where mask is False are mapped to N (an
    unreachable sentinel); segment_min recovers the smallest position
    where mask is True. If the segment has no True (impossible here
    since we built mask from the per-image argmax), the result is the
    sentinel.
    """
    Nh = mask.shape[0]
    pos = jnp.arange(Nh, dtype=jnp.int32)
    sentinel = jnp.int32(Nh)
    masked_pos = jnp.where(mask, pos, sentinel)
    return jax.ops.segment_min(masked_pos, image_id, num_segments=n_images)
