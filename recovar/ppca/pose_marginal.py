"""Per-pose PPCA score and moment functions for refinement EM."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .triangular import _tri_size, pack_upper_tri


def compute_ppca_pose_scores_and_moments_no_contrast(
    y_norm,
    t_mx,
    nu_mm,
    g_zx,
    h_zm,
    Hzz,
    *,
    return_moments: bool,
    debug_jitter: float = 0.0,
):
    r"""Closed-form pose-marginal PPCA score and posterior moments.

    Takes the six sufficient-statistic tensors produced by
    :func:`recovar.em.ppca_refinement.engine._per_pose_stats_block` and
    integrates the latent ``z ~ N(0, I_q)`` analytically.

    Quadratic completion. For a single image at pose (R, t):

    .. math::

        -2 \log p(y \mid R, t, z)
            \;=\; \rho \;-\; 2\,\Re\langle z, b\rangle \;+\; z^{\top} M z,
        \quad\text{where}\quad
        \rho \;=\; y\_norm - 2\,t\_mx + \nu_{mm}, \quad
        b \;=\; g_{zx} - h_{zm}, \quad
        M \;=\; I_q + H_{zz}.

    Integrating out :math:`z` analytically (Gaussian) gives the pose score

    .. math::

        \log p(y \mid R, t)
            \;=\; -\tfrac{1}{2}\left[\rho - b^{\top} M^{-1} b + \log\det M\right]
            \;+\; \text{const}.

    The posterior over ``z`` at fixed pose is

    .. math::

        z \mid y, R, t \;\sim\; \mathcal{N}\!\left(M^{-1} b,\; M^{-1}\right),

    so we return the augmented moments

    .. math::

        \alpha_{\mathrm{aug}}
            \;=\; \begin{pmatrix} 1 \\ \mathbb{E}[z] \end{pmatrix},
        \qquad
        G_{\mathrm{aug}}
            \;=\; \mathbb{E}\!\left[\begin{pmatrix} 1 \\ z \end{pmatrix}
                                   \begin{pmatrix} 1 \\ z \end{pmatrix}^{\!\top}\right]
            \;=\; \begin{pmatrix} 1 & \mathbb{E}[z]^{\!\top} \\
                                  \mathbb{E}[z] & M^{-1} + \mathbb{E}[z]\,\mathbb{E}[z]^{\!\top}
                  \end{pmatrix}.

    These are the augmented [μ, W] sufficient statistics consumed by the
    augmented M-step (:func:`recovar.ppca.augmented_mstep.solve_augmented_ppca_mstep`).

    Parameters
    ----------
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz
        Sufficient-statistic tensors from
        :func:`_per_pose_stats_block`. See that function's docstring for the
        math definitions and shapes (broadcast to ``(B, T, R, ...)``).
    return_moments : bool
        If False, return only ``score`` (saves memory when called by the
        score-only pass-1).
    debug_jitter : float
        Add ``debug_jitter * I`` to ``M`` before the Cholesky; only used
        when probing ill-conditioned ``M``.

    Returns
    -------
    score : ``(B, T, R)``
        Pose log-score :math:`\log p(y_i \mid R, t)`, up to an additive
        constant independent of pose.
    alpha_aug : ``(B, T, R, q+1)`` or ``None``
        :math:`[\,1,\; \mathbb{E}[z]\,]` per (image, pose). ``None`` when
        ``return_moments=False``.
    G_aug_tri : ``(B, T, R, (q+2)(q+1)/2)`` or ``None``
        Upper-triangular packed :math:`\mathbb{E}[z_{\mathrm{aug}}
        z_{\mathrm{aug}}^{\top}]`. ``None`` when ``return_moments=False``.

    Special case
    ------------
    ``q == 0`` reduces this to homogeneous K=1 scoring:
    ``score = -0.5 * rho`` and ``alpha_aug = G_aug = [1]``.
    """
    y_norm = jnp.asarray(y_norm)
    t_mx = jnp.asarray(t_mx)
    nu_mm = jnp.asarray(nu_mm)
    g_zx = jnp.asarray(g_zx)
    h_zm = jnp.asarray(h_zm)
    Hzz = jnp.asarray(Hzz)

    q = int(Hzz.shape[-1])
    if Hzz.shape[-2] != q:
        raise ValueError(f"Hzz must be square in its last two dimensions, got {Hzz.shape}")
    if int(g_zx.shape[-1]) != q:
        raise ValueError(f"g_zx last dim {g_zx.shape[-1]} != q={q}")
    if int(h_zm.shape[-1]) != q:
        raise ValueError(f"h_zm last dim {h_zm.shape[-1]} != q={q}")

    rho = y_norm - 2.0 * t_mx + nu_mm
    if q == 0:
        score = -0.5 * rho
        if not return_moments:
            return score, None, None
        ones = jnp.ones(rho.shape + (1,), dtype=jnp.complex64)
        return score, ones, ones

    Hzz_sym = 0.5 * (Hzz + jnp.swapaxes(jnp.conj(Hzz), -1, -2))
    eye = jnp.eye(q, dtype=Hzz_sym.dtype)
    M = Hzz_sym + eye
    if debug_jitter > 0.0:
        M = M + float(debug_jitter) * eye

    L = jnp.linalg.cholesky(M)
    diag_L = jnp.diagonal(L, axis1=-2, axis2=-1)
    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.real(diag_L)), axis=-1)

    b = g_zx - h_zm
    M_inv_b = jax.scipy.linalg.cho_solve((L, True), b[..., None])[..., 0]
    quad = jnp.sum(jnp.conj(b) * M_inv_b, axis=-1).real
    score = -0.5 * (rho - quad + logdet_M)

    if not return_moments:
        return score, None, None

    z_bar = M_inv_b
    eye_batched = jnp.broadcast_to(eye, M.shape)
    S_z = jax.scipy.linalg.cho_solve((L, True), eye_batched)
    leading = z_bar.shape[:-1]
    one = jnp.ones(leading + (1,), dtype=z_bar.dtype)
    alpha_aug = jnp.concatenate([one, z_bar], axis=-1)

    bottom_right = S_z + z_bar[..., :, None] * z_bar[..., None, :]
    top = jnp.concatenate(
        [jnp.ones(leading + (1, 1), dtype=z_bar.dtype), z_bar[..., None, :]],
        axis=-1,
    )
    bottom = jnp.concatenate([z_bar[..., :, None], bottom_right], axis=-1)
    G_aug = jnp.concatenate([top, bottom], axis=-2)
    G_aug_tri = pack_upper_tri(G_aug)
    if int(G_aug_tri.shape[-1]) != _tri_size(q + 1):
        raise AssertionError("internal triangular packing size mismatch")
    return score, alpha_aug, G_aug_tri
