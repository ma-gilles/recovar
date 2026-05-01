"""Per-pose PPCA score and moment functions (Milestone 1).

This module implements the per-pose marginalization of the latent ``z`` for
fixed contrast ``c=1``. The model and derivation live in
``docs/math/ppca_refine_plan_2026_05_01.md``; the operating contract is in
``recovar/em/ppca_refinement/CLAUDE.md`` §5.

For each (image, pose) hypothesis, given the per-pose sufficient stats

    y_norm  = <x, x>            scalar
    t_mx    = <x, m>            scalar
    nu_mm   = <m, m>            scalar
    g_zx    = B* x              [q]
    h_zm    = B* m              [q]
    Hzz     = B* B              [q, q]  (Hermitian PD)

with ``B = Σ^{-1/2} C P_R W``, ``m = Σ^{-1/2} C P_R μ``,
``x = T_{-t} Σ^{-1/2} y``, the analytic latent integral gives

    M    = I_q + Hzz                                         (q x q PD)
    b    = g_zx − h_zm                                       (q,)
    z̄    = M^{-1} b
    S_z  = M^{-1}
    ρ    = y_norm − 2 t_mx + nu_mm
    ℓ    = − ½ [ ρ − Re(b* M^{-1} b) + log det M ]

The augmented (q+1)-component moments returned to the M-step use
``r(z) = [1; z]`` and (no contrast) ``c ≡ 1``:

    alpha_aug = [1; z̄]                                        (q+1,)
    G_aug     = [[1,    z̄*],
                 [z̄,    S_z + z̄ z̄*]]                          (q+1, q+1)

We return ``G_aug`` packed as upper-triangular in row-major order to match
``recovar.ppca.ppca.unpack_tri_to_full`` (so accumulators can use the same
unpacker).

Numerical contract (CLAUDE.md §11): Cholesky everywhere, log-det from L,
``cho_solve`` for ``S_z`` and ``M_inv_b``, defensive Hermitian
symmetrization of ``Hzz``, jitter only behind an explicit debug flag, no
``pinv``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .ppca import _tri_size

__all__ = [
    "compute_ppca_pose_scores_and_moments_no_contrast",
    "compute_ppca_pose_scores_and_moments_with_contrast",
    "renormalize_contrast_into_theta",
    "_pack_upper_tri",
]


def _pack_upper_tri(M):
    """Pack ``(..., p, p)`` into ``(..., p(p+1)/2)`` row-major upper triangle.

    Convention matches ``recovar.ppca.ppca.unpack_tri_to_full`` (uses
    ``np.triu_indices`` order). Trailing ``p`` is read from the last axis.
    """
    p = M.shape[-1]
    if p == 0:
        return jnp.zeros(M.shape[:-2] + (0,), dtype=M.dtype)
    tri_i, tri_j = np.triu_indices(p)
    return M[..., tri_i, tri_j]


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
    """Pose-marginalized PPCA log-score and (optional) augmented moments.

    Vectorized over arbitrary leading batch dimensions. JIT-friendly. No
    Python-side branching on the leading shape.

    Parameters
    ----------
    y_norm, t_mx, nu_mm:
        Real, shape ``[...]``. Per-pose scalar sufficient stats.
    g_zx, h_zm:
        Complex, shape ``[..., q]``. Per-pose latent first-order stats.
    Hzz:
        Complex Hermitian, shape ``[..., q, q]``. Per-pose latent Gram.
        Symmetrized defensively as ``0.5 * (Hzz + Hzz^*)``.
    return_moments:
        If ``True`` also return ``alpha_aug`` and ``G_aug_tri``.
    debug_jitter:
        Optional ``η ≥ 0`` added to ``M`` as ``η · I_q`` before Cholesky.
        Default ``0.0`` — we *fail loudly* on Cholesky breakdown rather
        than silently regularize. Use only for explicit numerical debugging.

    Returns
    -------
    score : real, shape ``[...]``
        ``ℓ = − ½ [ ρ − Re(b* M^{-1} b) + log det M ]``.
    alpha_aug : complex, shape ``[..., q+1]`` or ``None``
        ``[1; z̄]`` when ``return_moments`` is True, else ``None``.
    G_aug_tri : complex, shape ``[..., (q+1)(q+2)/2]`` or ``None``
        Upper triangle of the (q+1)-augmented second-moment matrix.
        ``None`` if ``return_moments`` is False.
    """
    # Shape checks (cheap, JIT-traced once).
    q = Hzz.shape[-1]
    if Hzz.shape[-2] != q:
        raise ValueError(f"Hzz must be square in last two dims, got {Hzz.shape}")
    if g_zx.shape[-1] != q:
        raise ValueError(f"g_zx last dim {g_zx.shape[-1]} != q={q} from Hzz")
    if h_zm.shape[-1] != q:
        raise ValueError(f"h_zm last dim {h_zm.shape[-1]} != q={q} from Hzz")

    rho = y_norm - 2.0 * t_mx + nu_mm  # [...]

    if q == 0:
        # No latent; score reduces to homogeneous −½ρ. Augmented moments are
        # the trivial scalar [1] and [[1]].
        score = -0.5 * rho
        if not return_moments:
            return score, None, None
        leading = rho.shape
        ones = jnp.ones(leading + (1,), dtype=jnp.complex64)
        return score, ones, ones  # G_aug_tri == alpha_aug here (both [1])

    # Defensive Hermitian symmetrization. Hzz should already be Hermitian
    # PD; this guards against float roundoff in upstream GEMMs.
    Hzz_sym = 0.5 * (Hzz + jnp.swapaxes(jnp.conj(Hzz), -1, -2))
    I_q = jnp.eye(q, dtype=Hzz.dtype)
    M = I_q + Hzz_sym
    if debug_jitter > 0.0:
        M = M + debug_jitter * I_q

    # Cholesky factor; .H L L = M. log det M = 2 sum log Re(diag L).
    L = jnp.linalg.cholesky(M)
    diag_L = jnp.diagonal(L, axis1=-2, axis2=-1)  # [..., q]
    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.real(diag_L)), axis=-1)  # [...]

    b = g_zx - h_zm  # [..., q]
    # cho_solve handles batched L. shape: [..., q]
    M_inv_b = jax.scipy.linalg.cho_solve((L, True), b[..., None])[..., 0]
    quad = jnp.sum(jnp.conj(b) * M_inv_b, axis=-1).real  # [...]

    score = -0.5 * (rho - quad + logdet_M)

    if not return_moments:
        return score, None, None

    # z̄ = M^{-1} b
    z_bar = M_inv_b  # [..., q]
    # S_z = M^{-1}
    eye_batched = jnp.broadcast_to(I_q, M.shape)
    S_z = jax.scipy.linalg.cho_solve((L, True), eye_batched)  # [..., q, q]

    # alpha_aug = [1; z̄], complex64 to match downstream accumulators.
    leading = z_bar.shape[:-1]
    one_col = jnp.ones(leading + (1,), dtype=z_bar.dtype)
    alpha_aug = jnp.concatenate([one_col, z_bar], axis=-1)  # [..., q+1]

    # G_aug = [[1, z̄*], [z̄, S_z + z̄ z̄*]]
    # Build by blocks then pack upper triangle.
    zzT = z_bar[..., :, None] * jnp.conj(z_bar)[..., None, :]  # [..., q, q]
    G_bot_right = S_z + zzT
    # Top row: [1, conj(z̄)] of shape [..., 1, q+1]
    top = jnp.concatenate(
        [jnp.ones(leading + (1, 1), dtype=z_bar.dtype), jnp.conj(z_bar)[..., None, :]],
        axis=-1,
    )
    # Bottom rows: [z̄, S_z + z̄ z̄*] of shape [..., q, q+1]
    bot = jnp.concatenate([z_bar[..., :, None], G_bot_right], axis=-1)
    G_aug = jnp.concatenate([top, bot], axis=-2)  # [..., q+1, q+1]
    G_aug_tri = _pack_upper_tri(G_aug)  # [..., tri_size]

    # tri_size sanity check (cheap, traces once).
    assert G_aug_tri.shape[-1] == _tri_size(q + 1)

    return score, alpha_aug, G_aug_tri


# ===========================================================================
# Contrast-aware variant (Milestone 8)
# ===========================================================================


def _build_aug_moments_with_contrast(mean_c, mean_cz, second_c, mean_c2z, second_c2zz):
    """Build (alpha_aug, G_aug_tri) from contrast posterior moments.

    Per CLAUDE.md §5.2:
        alpha_aug[0]   = E[c]
        alpha_aug[1:]  = E[c z]
        G_aug[0, 0]    = E[c²]
        G_aug[0, 1:]   = E[c² z]      (top row, conjugate of column)
        G_aug[1:, 0]   = E[c² z]
        G_aug[1:, 1:]  = E[c² z z*]
    """
    leading = mean_c.shape
    cdtype = mean_cz.dtype
    # alpha_aug: complex64 with mean_c in slot 0 and mean_cz in slots 1..q.
    mean_c_c = mean_c.astype(cdtype)[..., None]  # [..., 1]
    alpha_aug = jnp.concatenate([mean_c_c, mean_cz], axis=-1)  # [..., q+1]

    q = mean_cz.shape[-1]
    P = q + 1
    second_c_c = second_c.astype(cdtype)
    # Build G_aug full then pack upper triangle (consistent with the
    # no-contrast variant's _pack_upper_tri convention).
    top_row_left = second_c_c[..., None, None]  # [..., 1, 1]
    top_row_right = mean_c2z[..., None, :]  # [..., 1, q]
    bot_left = mean_c2z[..., :, None]  # [..., q, 1]
    G_top = jnp.concatenate([top_row_left, top_row_right], axis=-1)  # [..., 1, P]
    G_bot = jnp.concatenate([bot_left, second_c2zz], axis=-1)  # [..., q, P]
    G_full = jnp.concatenate([G_top, G_bot], axis=-2)  # [..., P, P]
    G_aug_tri = _pack_upper_tri(G_full)  # [..., tri(P)]

    assert G_aug_tri.shape[-1] == _tri_size(P)
    return alpha_aug, G_aug_tri


def compute_ppca_pose_scores_and_moments_with_contrast(
    y_norm,
    t_mx,
    nu_mm,
    g_zx,
    h_zm,
    Hzz,
    *,
    contrast_mode: str = "marginalize",
    contrast_nodes=None,
    contrast_weights=None,
    contrast_mean: float = 1.0,
    contrast_variance: float = float("inf"),
    contrast_rule: str = "gauss_legendre",
    contrast_interval: tuple[float, float] = (0.0, 3.0),
    n_contrast_nodes: int = 32,
    return_moments: bool,
):
    """Contrast-aware per-pose score + augmented moments (Milestone 8).

    Wraps :func:`recovar.ppca.contrast_posterior.solve_latent_posterior`.
    The latent prior is identity (CLAUDE.md non-negotiable #2): we pass
    ``lambdas = ones(q)``.

    Parameters
    ----------
    Same per-pose stats as :func:`compute_ppca_pose_scores_and_moments_no_contrast`,
    plus contrast options forwarded to ``solve_latent_posterior``.
    ``contrast_mode`` is one of ``"profile"`` or ``"marginalize"``.

    Returns
    -------
    score : real, shape ``[...]``
        ``marginal_ll`` for ``contrast_mode="marginalize"``;
        the per-image profile-MAP log-score for ``contrast_mode="profile"``.
    alpha_aug : complex, shape ``[..., q+1]`` or None
        Augmented first-moment ``[E[c]; E[c z]]``.
    G_aug_tri : complex, shape ``[..., (q+1)(q+2)/2]`` or None
        Upper triangle of the augmented second-moment matrix
        with ``E[c²]``, ``E[c² z]``, ``E[c² z z*]`` blocks.
    """
    if contrast_mode not in ("profile", "marginalize"):
        raise ValueError(
            f"contrast_mode must be 'profile' or 'marginalize', got {contrast_mode!r}. "
            "Use compute_ppca_pose_scores_and_moments_no_contrast for c=1."
        )

    # Lazy import to avoid the heavy contrast solver showing up in the
    # no-contrast hot path.
    from .contrast_posterior import solve_latent_posterior

    # Flatten arbitrary leading dims to a single batch axis so the
    # contrast solver (which expects [B, …]) can run.
    leading_shape = y_norm.shape
    flat = (-1,)
    y_norm_f = y_norm.reshape(flat)
    t_mx_f = t_mx.reshape(flat)
    nu_mm_f = nu_mm.reshape(flat)
    q = Hzz.shape[-1]
    g_zx_f = g_zx.reshape(flat + (q,))
    h_zm_f = h_zm.reshape(flat + (q,))
    Hzz_f = Hzz.reshape(flat + (q, q))
    # ``solve_latent_posterior`` operates on the real-valued projection of
    # the per-pose Gram and inner products — the same convention the
    # legacy ``recovar.ppca.ppca._e_step_half_inner`` uses (H = (...).real,
    # g = (...).real). For half-spectrum images with full-spec Parseval
    # weights these are real up to roundoff.
    Hzz_f = Hzz_f.real.astype(jnp.float32)
    g_zx_f = g_zx_f.real.astype(jnp.float32)
    h_zm_f = h_zm_f.real.astype(jnp.float32)
    lambdas = jnp.ones((q,), dtype=jnp.float32)  # latent prior identity in v1

    posterior = solve_latent_posterior(
        Hzz_f,
        g_zx_f,
        h_zm_f,
        t_mx_f,
        nu_mm_f,
        y_norm_f,
        lambdas,
        contrast_mode=contrast_mode,
        contrast_nodes=contrast_nodes,
        contrast_weights=contrast_weights,
        contrast_mean=contrast_mean,
        contrast_variance=contrast_variance,
        contrast_rule=contrast_rule,
        contrast_interval=contrast_interval,
        n_contrast_nodes=n_contrast_nodes,
    )

    # Score: marginal_ll for marginalize; max profile_score per image for profile.
    if contrast_mode == "marginalize":
        score_flat = posterior.marginal_ll
    else:
        # profile: take the maximum profile score per image.
        score_flat = jnp.max(posterior.profile_scores, axis=-1)

    score = score_flat.reshape(leading_shape)

    if not return_moments:
        return score, None, None

    alpha_aug_flat, G_aug_tri_flat = _build_aug_moments_with_contrast(
        posterior.mean_c,
        posterior.mean_cz,
        posterior.second_moment_c,
        posterior.mean_c2z,
        posterior.second_moment_czz,
    )
    alpha_aug = alpha_aug_flat.reshape(leading_shape + (q + 1,))
    G_aug_tri = G_aug_tri_flat.reshape(leading_shape + (_tri_size(q + 1),))
    return score, alpha_aug, G_aug_tri


def renormalize_contrast_into_theta(mu, W, mean_contrast: float):
    """Absorb a global contrast scale into ``(μ, W)`` so ``E[c] → 1``.

    When the contrast posterior reports a per-batch mean ``<c>``, the
    natural fix is to scale BOTH ``μ`` AND ``W`` by ``<c>`` while
    rescaling ``z → z / <c>``. The latent prior remains ``z ~ N(0, I)``
    after rescaling because ``Var(z / <c>) = 1 / <c>²`` ⇒ rescaling
    ``W → <c> · W`` preserves the model identity.

    The failure mode this guards against (CLAUDE.md anti-pattern: "Contrast
    renormalization that scales only μ not W") would silently change the
    PPCA subspace amplitude. This helper enforces "scale both" by
    construction.

    Parameters
    ----------
    mu : array
        Mean volume in any representation (real or Fourier).
    W : array
        Loading bank with shape ``(q, *vs)`` matching ``mu`` representation.
    mean_contrast : float
        ``<c>`` — typically the average of ``E[c | y]`` across images.

    Returns
    -------
    (mu_new, W_new) : tuple of arrays
        ``mu_new = mean_contrast * mu``, ``W_new = mean_contrast * W``.
    """
    if mean_contrast <= 0:
        raise ValueError(f"mean_contrast must be positive, got {mean_contrast}.")
    scale = jnp.asarray(mean_contrast, dtype=jnp.float32)
    return scale * mu, scale * W
