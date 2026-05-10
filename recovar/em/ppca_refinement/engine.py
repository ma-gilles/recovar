"""Dense PPCA E-step and fused backprojection foundation.

These functions consume pre-shifted/weighted images and projected augmented
templates. The dataset-facing driver in :mod:`dense_dataset` builds these
blocks from current dense EM helpers; pass 2 recomputes posterior weights and
immediately backprojects into augmented half-volume accumulators without
materializing global pose moment tensors.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from recovar.em.ppca_refinement.diagnostics import build_iteration_diagnostics
from recovar.em.ppca_refinement.mean_regularization import (
    MeanRegularizationConfig,
    resolve_mean_precision,
)
from recovar.em.ppca_refinement.postprocess import PostprocessConfig, postprocess_ppca_half_volumes
from recovar.ppca import AugmentedPPCAStats, augmented_ppca_mstep_objective, solve_augmented_ppca_mstep
from recovar.ppca.pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast
from recovar.ppca.triangular import _tri_size


class DensePPCAFusedBlock(NamedTuple):
    """Prepared dense PPCA block for fused pass-2 accumulation."""

    Y1: jax.Array
    proj_aug: jax.Array
    ctf2_over_noise: jax.Array
    y_norm: jax.Array
    rotations: jax.Array
    pose_log_prior: jax.Array | None = None
    Y1_recon: jax.Array | None = None
    ctf2_over_noise_recon: jax.Array | None = None
    recon_window_indices: jax.Array | None = None
    use_recon_window: bool = False
    backprojection_max_r: float | None = None
    batch_start: int = 0
    rotation_start: int = 0


class DensePPCAFusedEMResult(NamedTuple):
    """One fused dense PPCA EM update over prepared blocks."""

    mu_half: jax.Array
    W_half: jax.Array
    stats: AugmentedPPCAStats
    diagnostics: dict


class DenseImageStats(NamedTuple):
    alpha_aug_acc: jax.Array
    G_aug_tri_acc: jax.Array
    log_evidence: jax.Array


class PosteriorDiagnostics(NamedTuple):
    logZ: jax.Array
    pmax: jax.Array
    best_rotation_idx: jax.Array
    best_translation_idx: jax.Array
    n_significant_per_image: jax.Array
    best_log_score_per_image: jax.Array
    rotation_posterior_sums: jax.Array
    max_posterior_per_image: jax.Array


class DenseScoreStats(NamedTuple):
    """Score-only pass-1 statistics for one PPCA pose block."""

    logZ: jax.Array
    best_log_score_per_image: jax.Array
    best_rotation_idx: jax.Array
    best_translation_idx: jax.Array


def _per_pose_stats_block(Y1, proj_aug, ctf2_over_noise, y_norm):
    r"""Per-pose sufficient statistics for the PPCA pose-marginal E-step.

    PPCA generative model (per image ``i``, latent ``z ~ N(0, I_q)``):

    .. math::

        y_i \;=\; \mathrm{CTF}_i \odot P_{R_i, t_i}\!\left(\mu + W z\right)
                  + \varepsilon_i,
        \qquad \varepsilon_i \sim \mathcal{N}(0, \mathrm{diag}(\sigma_i^2)).

    where :math:`P_{R,t}(\cdot)` is the Fourier slice of the rotated /
    translated volume. The negative log-likelihood (up to a constant
    independent of pose and ``z``) is

    .. math::

        -2 \log p(y_i \mid R, t, z) \;=\;
            \underbrace{\sum_f \frac{|y_i|^2}{\sigma_i^2}}_{=\, y\_norm_i}
            \;-\; 2\,\Re\!\left\langle Y_1^{(i,t)},\, P_R(\mu) \right\rangle
            \;+\; \nu_{mm}^{(i,r)}
            \;-\; 2\,\Re\!\left\langle z,\; g_{zx}^{(i,t,r)} - h_{zm}^{(i,r)} \right\rangle
            \;+\; z^{\top}\!\left(H_{zz}^{(i,r)} + I_q\right) z .

    Whitened data is :math:`Y_1 \;=\; y \cdot \mathrm{CTF} / \sigma^2`
    (the translation phase shift is folded into ``Y1`` upstream, indexed
    by ``t``).

    The tensors built here are the building blocks that
    :func:`compute_ppca_pose_scores_and_moments_no_contrast` plugs into
    the closed-form ``z``-marginal to produce the pose log-score
    :math:`\log p(y_i \mid R, t)` and the posterior moments
    :math:`\mathbb{E}[z_{\mathrm{aug}}], \mathbb{E}[z_{\mathrm{aug}} z_{\mathrm{aug}}^{\top}]`.

    Parameters
    ----------
    Y1 : ``(B, T, F)``
        Whitened, translation-shifted data
        :math:`Y_1^{(i,t,f)} = y_{i,f} \cdot \mathrm{CTF}_{i,f} / \sigma^2_{i,f}`
        for image ``i`` at translation ``t``, Fourier pixel ``f``.
    proj_aug : ``(R, q+1, F)``
        Augmented projection. Component 0 is :math:`P_R(\mu)`, components
        1..q are :math:`P_R(W_q)`. Same ``F`` (half-spectrum) layout as ``Y1``.
    ctf2_over_noise : ``(B, F)``
        :math:`\mathrm{CTF}_{i,f}^2 / \sigma_{i,f}^2`. Used for the
        CTF²/σ²-weighted projection-norm tensors below.
    y_norm : ``(B,)``
        :math:`\sum_f |y_i|^2 / \sigma_{i,f}^2` — image-energy constant
        that's broadcast through but otherwise pose-independent.

    Returns
    -------
    Tuple of six tensors, all broadcast to a common ``(B, T, R, ...)`` shape:

    - ``y_norm``  ``(B, T, R)``         constant in (t, r)
    - ``t_mx``    ``(B, T, R)``         :math:`\Re\langle Y_1^{(i,t)}, P_R(\mu)\rangle`
    - ``nu_mm``   ``(B, T, R)``         :math:`\sum_f (\mathrm{CTF}^2/\sigma^2)\,|P_R(\mu)|^2`, broadcast in t
    - ``g_zx``    ``(B, T, R, q)``      :math:`\Re\langle Y_1^{(i,t)}, P_R(W_q)\rangle`
    - ``h_zm``    ``(B, T, R, q)``      :math:`\Re \sum_f (\mathrm{CTF}^2/\sigma^2)\,\overline{P_R(W_q)}\,P_R(\mu)`, broadcast in t
    - ``H_zz``    ``(B, T, R, q, q)``   :math:`\Re \sum_f (\mathrm{CTF}^2/\sigma^2)\,\overline{P_R(W_q)}\,P_R(W_{q'})`, broadcast in t

    The translation dimension ``T`` only carries information through
    ``Y1`` (which encodes the translation phase shift); the projection
    tensors (``nu_mm``, ``h_zm``, ``H_zz``) are translation-invariant
    and broadcast in ``t`` purely for downstream einsum convenience.

    Notes
    -----
    Even though ``proj_W`` is complex (a Fourier slice of a real-space
    volume), the latent ``z`` is real, so all PPCA arithmetic happens on
    the real parts of the Hermitian inner products. The ``.real``
    projections are exact (not approximations) provided the half-image
    weights are baked into ``Y1``, ``ctf2_over_noise``, and ``y_norm``
    upstream. They are: ``recovar.em.dense_single_volume.helpers.preprocessing.preprocess_batch``
    computes ``y_norm = Σ_f (|y|²/σ²) · w_f``, and
    :func:`iter_dense_ppca_dataset_blocks` bakes
    ``score_mask = window_mask × half_weights`` into both ``Y1`` and
    ``ctf2_over_noise``. The weights are ``w_f = 2`` for non-DC /
    non-Nyquist Fourier pixels and ``w_f = 1`` for DC and Nyquist (see
    :func:`recovar.em.dense_single_volume.helpers.half_spectrum.make_half_image_weights`),
    so each einsum below is the **full-Fourier** inner product and its
    real part is exact under the Hermitian symmetry of real-space volumes.
    """
    B, T, F = Y1.shape
    R, P, proj_F = proj_aug.shape
    if proj_F != F:
        raise ValueError(f"proj_aug last dim {proj_F} != Y1 Fourier dim {F}")
    if ctf2_over_noise.shape != (B, F):
        raise ValueError(f"ctf2_over_noise shape {ctf2_over_noise.shape} != ({B}, {F})")
    if y_norm.shape != (B,):
        raise ValueError(f"y_norm shape {y_norm.shape} != ({B},)")

    q = P - 1
    proj_mu = proj_aug[:, 0, :]
    proj_W = proj_aug[:, 1:, :]
    ctf2 = ctf2_over_noise.astype(proj_aug.dtype)

    # Keep the mean-column arithmetic identical for q=0 and W=0. Computing
    # it as part of a wider augmented GEMM changes float32 contraction order
    # enough to obscure the homogeneous parity invariant.
    nu_mm = jnp.einsum("bf,rf,rf->br", ctf2, jnp.conj(proj_mu), proj_mu).real
    t_mx = jnp.einsum("btf,rf->btr", jnp.conj(Y1), proj_mu).real
    if q == 0:
        g_zx = jnp.zeros((B, T, R, 0), dtype=proj_aug.dtype)
        h_zm = jnp.zeros((B, R, 0), dtype=proj_aug.dtype)
        Hzz = jnp.zeros((B, R, 0, 0), dtype=proj_aug.dtype)
    else:
        # PPCA coordinates are real; the Fourier-domain loading columns carry
        # complex phases only because they represent real-space volumes.
        g_zx = jnp.einsum("btf,rqf->btrq", Y1, jnp.conj(proj_W)).real
        h_zm = jnp.einsum("bf,rqf,rf->brq", ctf2, jnp.conj(proj_W), proj_mu).real
        Hzz = jnp.einsum("bf,rqf,rpf->brqp", ctf2, jnp.conj(proj_W), proj_W).real
    return (
        jnp.broadcast_to(y_norm[:, None, None], (B, T, R)),
        t_mx,
        jnp.broadcast_to(nu_mm[:, None, :], (B, T, R)),
        g_zx,
        jnp.broadcast_to(h_zm[:, None, :, :], (B, T, R, q)),
        jnp.broadcast_to(Hzz[:, None, :, :, :], (B, T, R, q, q)),
    )


def _add_pose_log_prior(score, pose_log_prior):
    if pose_log_prior is None:
        return score
    return score + jnp.swapaxes(jnp.asarray(pose_log_prior), -1, -2)


@partial(jax.jit, static_argnames=("significance_threshold",))
def dense_pose_ppca_E_step_blocked(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior=None,
    *,
    significance_threshold: float = 1e-3,
):
    """Run a dense PPCA E-step on one static block.

    ``Y1`` has shape ``[B, T, F]``. ``proj_aug`` has shape ``[R, q+1, F]``
    with component 0 equal to the mean projection and components 1..q equal
    to loading projections. ``pose_log_prior`` is optional ``[B, R, T]``.
    """
    B, T, _F = jnp.asarray(Y1).shape
    R, P, _ = jnp.asarray(proj_aug).shape
    if pose_log_prior is not None and jnp.asarray(pose_log_prior).shape != (B, R, T):
        raise ValueError(f"pose_log_prior shape {jnp.asarray(pose_log_prior).shape} != ({B}, {R}, {T})")
    y_stats = _per_pose_stats_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
    )
    score_pre, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=True,
    )
    score = _add_pose_log_prior(score_pre, pose_log_prior)
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
    alpha_aug_acc = jnp.einsum("btr,btrp->bp", gamma.astype(alpha.dtype), alpha)
    G_aug_tri_acc = jnp.einsum("btr,btrk->bk", gamma.astype(G_tri.dtype), G_tri)
    return DenseImageStats(alpha_aug_acc=alpha_aug_acc, G_aug_tri_acc=G_aug_tri_acc, log_evidence=logZ), diagnostics


@jax.jit
def dense_pose_ppca_score_stats_blocked(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior=None,
):
    """Return block log normalizers and maxima without PPCA moments."""
    B, T, _F = jnp.asarray(Y1).shape
    R, _P, _ = jnp.asarray(proj_aug).shape
    if pose_log_prior is not None and jnp.asarray(pose_log_prior).shape != (B, R, T):
        raise ValueError(f"pose_log_prior shape {jnp.asarray(pose_log_prior).shape} != ({B}, {R}, {T})")
    y_stats = _per_pose_stats_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
    )
    score_pre, _alpha, _G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=False,
    )
    score = _add_pose_log_prior(score_pre, pose_log_prior)
    score_flat = score.reshape(B, T * R)
    best_flat = jnp.argmax(score_flat, axis=-1)
    return DenseScoreStats(
        logZ=jax.scipy.special.logsumexp(score_flat, axis=-1),
        best_log_score_per_image=jnp.max(score_flat, axis=-1).astype(jnp.float32),
        best_rotation_idx=(best_flat % R).astype(jnp.int32),
        best_translation_idx=(best_flat // R).astype(jnp.int32),
    )


def dense_pose_ppca_logZ_blocked(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior=None,
):
    """Return block log normalizers without materializing PPCA moments."""
    return dense_pose_ppca_score_stats_blocked(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        pose_log_prior,
    ).logZ


def _score_gamma_and_moments(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    pose_log_prior,
    significance_threshold: float,
    normalization_logZ=None,
):
    y_stats = _per_pose_stats_block(Y1, proj_aug, ctf2_over_noise, y_norm)
    score_pre, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        *y_stats,
        return_moments=True,
    )
    score = _add_pose_log_prior(score_pre, pose_log_prior)
    B, T, R = score.shape
    score_flat = score.reshape(B, T * R)
    logZ = (
        jax.scipy.special.logsumexp(score_flat, axis=-1)
        if normalization_logZ is None
        else jnp.asarray(normalization_logZ)
    )
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
def fused_dense_pose_ppca_block(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    rotations_block,
    image_shape,
    volume_shape,
    rhs_volume,
    lhs_tri_volume,
    pose_log_prior=None,
    Y1_recon=None,
    ctf2_over_noise_recon=None,
    normalization_logZ=None,
    *,
    significance_threshold: float = 1e-3,
    disc_type_backproject: str = "linear_interp",
    recon_window_indices=None,
    use_recon_window: bool = False,
    backprojection_max_r=None,
):
    """Fuse dense PPCA pass 2 with augmented half-volume backprojection.

    Inputs are one image/translation/rotation block:

    - ``Y1``: ``[B, T, F]`` shifted, CTF/noise-weighted half-images.
    - ``proj_aug``: ``[R, q+1, F]`` projections of ``[mu, W_1, ..., W_q]``.
    - ``ctf2_over_noise``: ``[B, F]`` CTF-square/noise weights.
    - ``rotations_block``: ``[R, 3, 3]`` rotations matching ``proj_aug``.

    Pass 2 consumes each rotation immediately:

    ``rhs[p] += A_r^* sum_t gamma_btr alpha_btrp Y1_bt``

    ``lhs[pq] += A_r^* sum_t gamma_btr G_btrpq (CTF^2/noise)_b``

    This keeps tensors at block scope and avoids a global
    ``[images, rotations, translations, q, q]`` moment tensor.
    """
    from recovar.em.dense_single_volume.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed

    Y1 = jnp.asarray(Y1)
    proj_aug = jnp.asarray(proj_aug)
    ctf2_over_noise = jnp.asarray(ctf2_over_noise)
    y_norm = jnp.asarray(y_norm)
    rotations_block = jnp.asarray(rotations_block)
    rhs_volume = jnp.asarray(rhs_volume)
    lhs_tri_volume = jnp.asarray(lhs_tri_volume)

    B, T, F = Y1.shape
    R, P, proj_F = proj_aug.shape
    if proj_F != F:
        raise ValueError(f"proj_aug Fourier dim {proj_F} != Y1 Fourier dim {F}")
    if rotations_block.shape != (R, 3, 3):
        raise ValueError(f"rotations_block shape {rotations_block.shape} != ({R}, 3, 3)")
    tri = _tri_size(P)
    if rhs_volume.ndim != 2 or rhs_volume.shape[0] != P:
        raise ValueError(f"rhs_volume must have shape [P={P}, half_vol], got {rhs_volume.shape}")
    if lhs_tri_volume.ndim != 2 or lhs_tri_volume.shape[0] != tri:
        raise ValueError(f"lhs_tri_volume must have shape [tri(P)={tri}, half_vol], got {lhs_tri_volume.shape}")
    if pose_log_prior is not None and jnp.asarray(pose_log_prior).shape != (B, R, T):
        raise ValueError(f"pose_log_prior shape {jnp.asarray(pose_log_prior).shape} != ({B}, {R}, {T})")
    Y1_recon = Y1 if Y1_recon is None else jnp.asarray(Y1_recon)
    ctf2_over_noise_recon = ctf2_over_noise if ctf2_over_noise_recon is None else jnp.asarray(ctf2_over_noise_recon)
    F_recon = int(Y1_recon.shape[-1])
    if Y1_recon.shape[:2] != (B, T):
        raise ValueError(f"Y1_recon leading shape {Y1_recon.shape[:2]} != ({B}, {T})")
    if ctf2_over_noise_recon.shape != (B, F_recon):
        raise ValueError(f"ctf2_over_noise_recon shape {ctf2_over_noise_recon.shape} != ({B}, {F_recon})")

    gamma, alpha, G_tri, diagnostics = _score_gamma_and_moments(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        pose_log_prior,
        significance_threshold,
        normalization_logZ=normalization_logZ,
    )
    rhs_dtype = rhs_volume.dtype
    lhs_dtype = lhs_tri_volume.dtype

    rhs_images = jnp.einsum(
        "btr,btrp,btf->prf",
        gamma.astype(rhs_dtype),
        jnp.conj(alpha).astype(rhs_dtype),
        Y1_recon.astype(rhs_dtype),
    ).astype(rhs_dtype)
    rhs_volume = batch_adjoint_slice_volume_maybe_windowed(
        rhs_images,
        recon_window_indices,
        rotations_block,
        rhs_volume,
        image_shape,
        volume_shape,
        disc_type_backproject,
        True,
        True,
        use_window=bool(use_recon_window),
        max_r=backprojection_max_r,
    )

    lhs_images = jnp.einsum(
        "btr,btrk,bf->krf",
        gamma.astype(lhs_dtype),
        G_tri,
        ctf2_over_noise_recon.astype(lhs_dtype),
    ).real.astype(lhs_dtype)
    lhs_tri_volume = batch_adjoint_slice_volume_maybe_windowed(
        lhs_images,
        recon_window_indices,
        rotations_block,
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


def _enforce_augmented_x0(volumes, volume_shape):
    from recovar.em.dense_single_volume.local_backprojection import enforce_relion_half_volume_x0_hermitian

    enforced = [enforce_relion_half_volume_x0_hermitian(volumes[i], volume_shape) for i in range(volumes.shape[0])]
    return jnp.stack(enforced, axis=0)


def run_dense_ppca_fused_refinement_blocks(
    blocks,
    *,
    q: int,
    image_shape,
    volume_shape,
    mean_prior,
    W_prior,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    disc_type_backproject: str = "linear_interp",
    enforce_x0: bool = True,
    mstep_chunk_size: int | None = None,
    fixed_mean_half=None,
):
    """Run one dense PPCA EM update over prepared fused blocks.

    This is the first integration layer above :func:`fused_dense_pose_ppca_block`.
    It streams the prepared blocks into augmented ``[μ, W]`` half-volume
    sufficient statistics (``rhs``, ``lhs_tri``) and then calls the joint
    augmented M-step :func:`recovar.ppca.augmented_mstep.solve_augmented_ppca_mstep`.

    The caller is responsible for building the block list from the dataset
    given the current iteration's geometry (``current_size``, ``q``,
    ``volume_domain``) and schedule (batch sizes); this function does not
    know about HEALPix orders or per-iter schedules. The dataset-facing
    entry points :func:`recovar.em.ppca_refinement.dense_dataset.iter_dense_ppca_dataset_blocks`
    / :func:`recovar.em.ppca_refinement.local_dataset.iter_local_ppca_dataset_blocks`
    are what produce the blocks.
    """
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()
    q = int(q)
    P = q + 1
    tri = _tri_size(P)
    mean_prior = jnp.asarray(mean_prior)
    W_prior = jnp.asarray(W_prior)
    if W_prior.shape != (mean_prior.shape[0], q):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({mean_prior.shape[0]}, {q})")

    rhs_volume = jnp.zeros((P, mean_prior.shape[0]), dtype=jnp.complex64)
    lhs_tri_volume = jnp.zeros((tri, mean_prior.shape[0]), dtype=jnp.float32)
    log_likelihood = 0.0
    n_images = 0
    pmax_values = []
    nsig_values = []
    best_rotations = []
    best_translations = []
    postprocess_bandlimit_max_r = None

    for block in blocks:
        if postprocess_bandlimit_max_r is None and bool(block.use_recon_window):
            postprocess_bandlimit_max_r = block.backprojection_max_r
        rhs_volume, lhs_tri_volume, diag = fused_dense_pose_ppca_block(
            block.Y1,
            block.proj_aug,
            block.ctf2_over_noise,
            block.y_norm,
            block.rotations,
            image_shape,
            volume_shape,
            rhs_volume,
            lhs_tri_volume,
            block.pose_log_prior,
            Y1_recon=block.Y1_recon,
            ctf2_over_noise_recon=block.ctf2_over_noise_recon,
            disc_type_backproject=disc_type_backproject,
            recon_window_indices=block.recon_window_indices,
            use_recon_window=block.use_recon_window,
            backprojection_max_r=block.backprojection_max_r,
        )
        log_likelihood += float(jnp.sum(diag.logZ))
        n_images += int(diag.logZ.shape[0])
        pmax_values.append(jnp.asarray(diag.pmax))
        nsig_values.append(jnp.asarray(diag.n_significant_per_image))
        best_rotations.append(jnp.asarray(diag.best_rotation_idx))
        best_translations.append(jnp.asarray(diag.best_translation_idx))

    if enforce_x0:
        rhs_volume = _enforce_augmented_x0(rhs_volume, volume_shape)
        lhs_tri_volume = _enforce_augmented_x0(lhs_tri_volume.astype(jnp.complex64), volume_shape).real.astype(
            jnp.float32
        )

    diagnostics = build_iteration_diagnostics(
        pmax_values=pmax_values,
        nsig_values=nsig_values,
        best_rotations=best_rotations,
        best_translations=best_translations,
        log_likelihood=log_likelihood,
        n_images=n_images,
        mean_reg=mean_reg,
        image_scale_min=1.0,
        image_scale_max=1.0,
        image_scale_corrections=None,
    )
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
        diagnostics=diagnostics,
    )
    mean_precision = resolve_mean_precision(stats, mean_prior, volume_shape, mean_reg)
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
        volume_shape,
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
