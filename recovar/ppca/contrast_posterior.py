"""Standalone contrast-aware latent posterior solver for PPCA/EM.

This module implements exact posterior inference over latent coordinates ``z``
with three contrast handling modes:

- **none**: fixed ``c = 1`` (no contrast estimation).
- **profile**: profile-MAP over ``(z, c)`` — backward-compatible with RECOVAR's
  existing grid-search behavior.  The objective is
  ``J_prof(c) = r(c) - q(c)^T A(c)^{-1} q(c) - 2 log p(c)`` which has **no**
  ``log det A(c)`` term.
- **marginalize**: exact contrast marginalization via quadrature.  The collapsed
  score is ``J_marg(c) = r(c) - q(c)^T A(c)^{-1} q(c) + log det A(c) - 2 log p(c)``
  and the ``log det`` is essential because different contrast values yield
  different posterior volumes.

The eigendecomposition of ``Lambda^{1/2} H Lambda^{1/2}`` is an **optimization**,
not a mathematical requirement.  It avoids repeated Cholesky factorizations by
diagonalizing the entire one-parameter family ``A(c) = Lambda^{-1} + c^2 H``
at once.

The solver operates purely on precomputed sufficient statistics and does not
depend on RECOVAR dataset objects.

Scope restriction
-----------------
Exact marginalization is supported only when there is **one** contrast scalar
per solve (SPA image-level, or grouped solves with one shared contrast).
"""

from __future__ import annotations

import functools
from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


class LatentPosteriorResult(NamedTuple):
    """Posterior moments returned by the latent solver.

    All shapes assume a leading batch dimension ``B``.

    Attributes
    ----------
    mean_z : (B, K)
        Posterior mean ``E[z | y]``.
    cov_z : (B, K, K)
        Posterior covariance ``Cov[z | y]``.
    second_moment_z : (B, K, K)
        ``E[z z^T | y]``.
    mean_c : (B,)
        ``E[c | y]``.
    second_moment_c : (B,)
        ``E[c^2 | y]``.
    mean_cz : (B, K)
        ``E[c z | y]``.
    mean_c2z : (B, K)
        ``E[c^2 z | y]``.
    second_moment_czz : (B, K, K)
        ``E[c^2 z z^T | y]``.
    contrast_nodes : (C,)
        Quadrature / grid nodes used.
    contrast_weights_posterior : (B, C)
        Normalized posterior weights ``omega_j`` at each node.
    best_contrast : (B,)
        Contrast value with highest posterior weight (or profile-MAP argmin).
    profile_scores : (B, C) or None
        Raw profile / marginal scores at each contrast node.
    marginal_ll : (B,) or None
        Per-image marginalized log-likelihood ``log sum_j w_j p(y_i | c_j)``.
        Only set when contrast_mode="marginalize"; None otherwise.
    """

    mean_z: jax.Array
    cov_z: jax.Array
    second_moment_z: jax.Array
    mean_c: jax.Array
    second_moment_c: jax.Array
    mean_cz: jax.Array
    mean_c2z: jax.Array
    second_moment_czz: jax.Array
    contrast_nodes: jax.Array
    contrast_weights_posterior: jax.Array
    best_contrast: jax.Array
    profile_scores: Optional[jax.Array]
    marginal_ll: Optional[jax.Array]


class LegacyEmbeddingResult(NamedTuple):
    """Legacy RECOVAR-compatible outputs from the solver.

    Attributes
    ----------
    xs : (B, K)
        Latent coordinates at the best contrast.
    contrasts : (B,)
        Best contrast per image.
    precision : (B, K, K)
        Legacy "precision" matrix (the cov_batch from old code).
    """

    xs: jax.Array
    contrasts: jax.Array
    precision: jax.Array


# ---------------------------------------------------------------------------
# Quadrature construction
# ---------------------------------------------------------------------------


def make_contrast_quadrature(
    rule: Literal["gauss_legendre", "trapezoid", "custom"] = "gauss_legendre",
    interval: tuple[float, float] = (0.0, 3.0),
    n_nodes: int = 32,
    nodes: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> tuple[jax.Array, jax.Array]:
    """Build quadrature nodes and weights on a contrast interval.

    Parameters
    ----------
    rule : str
        ``"gauss_legendre"`` (default, 16 nodes on ``[0, 3]``),
        ``"trapezoid"`` (composite trapezoid on supplied or uniform nodes),
        ``"custom"`` (user-supplied nodes and weights).
    interval : (float, float)
        Integration interval ``[c_min, c_max]``.
    n_nodes : int
        Number of quadrature nodes (ignored for ``"custom"``).
    nodes : array, optional
        Explicit nodes (required for ``"custom"``, optional for ``"trapezoid"``).
    weights : array, optional
        Explicit weights (required for ``"custom"``).

    Returns
    -------
    (nodes, weights) : tuple of jax arrays, each shape ``(C,)``.
    """
    c_min, c_max = interval

    if rule == "gauss_legendre":
        # Gauss-Legendre on [-1, 1] mapped to [c_min, c_max]
        ref_nodes, ref_weights = np.polynomial.legendre.leggauss(n_nodes)
        # Map from [-1, 1] to [c_min, c_max]
        half_len = 0.5 * (c_max - c_min)
        mid = 0.5 * (c_min + c_max)
        out_nodes = mid + half_len * ref_nodes
        out_weights = half_len * ref_weights
        return jnp.array(out_nodes, dtype=jnp.float32), jnp.array(out_weights, dtype=jnp.float32)

    elif rule == "trapezoid":
        if nodes is not None:
            out_nodes = np.asarray(nodes, dtype=np.float32)
        else:
            out_nodes = np.linspace(c_min, c_max, n_nodes, dtype=np.float32)
        n = len(out_nodes)
        if n == 1:
            out_weights = np.ones(1, dtype=np.float32) * (c_max - c_min)
        else:
            out_weights = np.empty(n, dtype=np.float32)
            out_weights[0] = 0.5 * (out_nodes[1] - out_nodes[0])
            out_weights[-1] = 0.5 * (out_nodes[-1] - out_nodes[-2])
            for i in range(1, n - 1):
                out_weights[i] = 0.5 * (out_nodes[i + 1] - out_nodes[i - 1])
        return jnp.array(out_nodes), jnp.array(out_weights)

    elif rule == "custom":
        if nodes is None or weights is None:
            raise ValueError("custom rule requires both nodes and weights")
        return jnp.array(nodes, dtype=jnp.float32), jnp.array(weights, dtype=jnp.float32)

    else:
        raise ValueError(f"Unknown quadrature rule: {rule!r}")


# ---------------------------------------------------------------------------
# Spectral precomputation (shared across all three modes)
# ---------------------------------------------------------------------------


def _spectral_decomposition(H, lambdas, g, h):
    """One-time eigendecomposition for the contrast-parameterized family.

    Parameters
    ----------
    H : (B, K, K)  Gram matrix  B^T B
    lambdas : (K,)  Prior variances (eigenvalues of latent prior covariance)
    g : (B, K)  B^T y
    h : (B, K)  B^T m

    Returns
    -------
    d : (B, K)   eigenvalues of Lambda^{1/2} H Lambda^{1/2}
    T : (B, K, K) transform matrix Lambda^{1/2} Q
    alpha : (B, K)  Q^T Lambda^{1/2} g
    beta  : (B, K)  Q^T Lambda^{1/2} h
    """
    L = jnp.sqrt(lambdas)  # (K,)
    # G = Lambda^{1/2} H Lambda^{1/2}  -- shape (B, K, K)
    G = L[None, :, None] * H * L[None, None, :]
    # Symmetrize for numerical stability
    G = 0.5 * (G + jnp.swapaxes(G, -1, -2))

    d, Q = jnp.linalg.eigh(G)  # d: (B, K), Q: (B, K, K)
    d = jnp.clip(d, min=0.0)

    Lg = L[None, :] * g  # (B, K)
    Lh = L[None, :] * h  # (B, K)

    # alpha = Q^T (L * g),  beta = Q^T (L * h)
    alpha = jnp.einsum("bji,bj->bi", Q, Lg)  # (B, K)
    beta = jnp.einsum("bji,bj->bi", Q, Lh)  # (B, K)

    T = L[None, :, None] * Q  # (B, K, K)

    return d, T, alpha, beta


# ---------------------------------------------------------------------------
# Shared contrast-score and moment computation
# ---------------------------------------------------------------------------


class _SpectralScores(NamedTuple):
    """Intermediate spectral quantities shared by profile and marginalize."""

    d: jax.Array  # (B, K) eigenvalues
    T: jax.Array  # (B, K, K) transform matrix
    s: jax.Array  # (B, C, K) posterior precisions per node
    r_spec: jax.Array  # (B, C, K) spectral-basis posterior means per node
    quad: jax.Array  # (B, C) quadratic form
    rho: jax.Array  # (B, C) residual
    log_prior: jax.Array  # (C,) contrast prior


def _compute_contrast_scores(H, g, h, t, nu, y_norm_sq, lambdas, contrast_nodes, contrast_mean, contrast_variance):
    """Spectral decomposition + per-node scores shared by profile and marginalize.

    Returns all intermediate quantities that both solvers need.
    """
    d, T, alpha, beta = _spectral_decomposition(H, lambdas, g, h)

    c = contrast_nodes  # (C,)
    c2 = c**2  # (C,)

    # s(c) = 1 / (1 + c^2 d)  -- (B, C, K)
    s = 1.0 / (1.0 + d[:, None, :] * c2[None, :, None])
    # v(c) = c alpha - c^2 beta  -- (B, C, K)
    v = c[None, :, None] * alpha[:, None, :] - c2[None, :, None] * beta[:, None, :]
    # r_spec = s * v  -- (B, C, K)
    r_spec = s * v

    # quad = sum_k s_k v_k^2  -- (B, C)
    quad = jnp.sum(v * r_spec, axis=-1)
    # rho = ||y||^2 - 2c <y, m> + c^2 ||m||^2  -- (B, C)
    rho = y_norm_sq[:, None] - 2.0 * c[None, :] * t[:, None] + c2[None, :] * nu[:, None]

    # Contrast prior (handles both finite and infinite variance)
    is_finite_var = jnp.isfinite(contrast_variance)
    log_prior = jnp.where(
        is_finite_var,
        -0.5 * (contrast_nodes - contrast_mean) ** 2 / jnp.where(is_finite_var, contrast_variance, 1.0),
        0.0,
    )  # (C,)

    return _SpectralScores(d=d, T=T, s=s, r_spec=r_spec, quad=quad, rho=rho, log_prior=log_prior)


def _compute_weighted_moments(T, s, r_spec, contrast_nodes, omega):
    """Compute all posterior moments given node weights omega.

    Works for both profile (one-hot omega) and marginalize (softmax omega).

    Parameters
    ----------
    T : (B, K, K)  spectral transform
    s : (B, C, K)  posterior precisions per node
    r_spec : (B, C, K)  spectral-basis posterior means per node
    contrast_nodes : (C,)
    omega : (B, C)  normalized weights over nodes

    Returns
    -------
    mean_z, cov_z, second_moment_z, mean_c, second_moment_c,
    mean_cz, mean_c2z, second_moment_czz
    """
    K = T.shape[-1]
    c = contrast_nodes
    c2 = c**2

    # E[z | y] = T @ sum_j omega_j r_j
    mean_spec = jnp.sum(omega[:, :, None] * r_spec, axis=1)  # (B, K)
    mean_z = jnp.einsum("bij,bj->bi", T, mean_spec)  # (B, K)

    # E[zz^T | y] = T @ (diag(sum_j omega_j s_j) + sum_j omega_j r_j r_j^T) @ T^T
    Sigma_diag_spec = jnp.sum(omega[:, :, None] * s, axis=1)  # (B, K)
    R = jnp.einsum("bc,bci,bcj->bij", omega, r_spec, r_spec)  # (B, K, K)
    M = jnp.diag(jnp.ones(K)) * Sigma_diag_spec[:, None, :] + R  # (B, K, K)
    second_moment_z = jnp.einsum("bij,bjl,bkl->bik", T, M, T)  # (B, K, K)
    second_moment_z = 0.5 * (second_moment_z + jnp.swapaxes(second_moment_z, -1, -2))
    cov_z = second_moment_z - mean_z[:, :, None] * mean_z[:, None, :]
    cov_z = 0.5 * (cov_z + jnp.swapaxes(cov_z, -1, -2))

    # Contrast moments
    mean_c = jnp.sum(omega * c[None, :], axis=-1)  # (B,)
    second_moment_c = jnp.sum(omega * c2[None, :], axis=-1)  # (B,)

    # E[c z | y]
    mean_cz_spec = jnp.sum(omega[:, :, None] * (c[None, :, None] * r_spec), axis=1)  # (B, K)
    mean_cz = jnp.einsum("bij,bj->bi", T, mean_cz_spec)  # (B, K)

    # E[c^2 z | y]
    mean_c2z_spec = jnp.sum(omega[:, :, None] * (c2[None, :, None] * r_spec), axis=1)  # (B, K)
    mean_c2z = jnp.einsum("bij,bj->bi", T, mean_c2z_spec)  # (B, K)

    # E[c^2 z z^T | y]
    Sigma_c2_diag_spec = jnp.sum(omega[:, :, None] * (c2[None, :, None] * s), axis=1)  # (B, K)
    R_c2 = jnp.einsum("bc,bci,bcj->bij", omega * c2[None, :], r_spec, r_spec)  # (B, K, K)
    M_c2 = jnp.diag(jnp.ones(K)) * Sigma_c2_diag_spec[:, None, :] + R_c2
    second_moment_czz = jnp.einsum("bij,bjl,bkl->bik", T, M_c2, T)  # (B, K, K)
    second_moment_czz = 0.5 * (second_moment_czz + jnp.swapaxes(second_moment_czz, -1, -2))

    return mean_z, cov_z, second_moment_z, mean_c, second_moment_c, mean_cz, mean_c2z, second_moment_czz


# ---------------------------------------------------------------------------
# Core solvers
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("return_legacy",))
def solve_no_contrast(
    H,
    g,
    h,
    t,
    nu,
    y_norm_sq,
    lambdas,
    *,
    return_legacy: bool = False,
):
    """Fixed c=1 solver (no contrast estimation).

    Parameters
    ----------
    H : (B, K, K)  AU_t_AU
    g : (B, K)     AU_t_images
    h : (B, K)     AU_t_Amean
    t : (B,)       image_T_A_mean
    nu : (B,)      A_mean_norm_sq
    y_norm_sq : (B,)  image_norms_sq
    lambdas : (K,)    prior variances (eigenvalues)
    return_legacy : bool
        If True, also return LegacyEmbeddingResult.
    """
    K = lambdas.shape[0]
    B = g.shape[0]

    # A = Lambda^{-1} + H
    A = H + jnp.diag(1.0 / lambdas)  # (B, K, K)
    # q = g - h
    q = g - h  # (B, K)

    # Solve A mu = q via Cholesky
    mu = jax.scipy.linalg.solve(A, q, assume_a="pos")  # (B, K)
    # Sigma = A^{-1}
    Sigma = jnp.linalg.inv(A)  # (B, K, K)

    c_scalar = jnp.ones(B, dtype=g.dtype)
    c2_scalar = jnp.ones(B, dtype=g.dtype)

    second_moment_z = Sigma + mu[:, :, None] * mu[:, None, :]
    cov_z = Sigma

    result = LatentPosteriorResult(
        mean_z=mu,
        cov_z=cov_z,
        second_moment_z=second_moment_z,
        mean_c=c_scalar,
        second_moment_c=c2_scalar,
        mean_cz=mu,  # c=1, so E[cz] = E[z]
        mean_c2z=mu,  # c=1, so E[c^2 z] = E[z]
        second_moment_czz=second_moment_z,  # c=1
        contrast_nodes=jnp.ones(1, dtype=g.dtype),
        contrast_weights_posterior=jnp.ones((B, 1), dtype=g.dtype),
        best_contrast=c_scalar,
        profile_scores=None,
        marginal_ll=None,
    )

    if not return_legacy:
        return result

    # Legacy precision: (gram + Lambda^{-1}) @ pinv(gram) @ (gram + Lambda^{-1})
    gram = H  # c=1
    precision = A @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ A
    legacy = LegacyEmbeddingResult(xs=mu, contrasts=c_scalar, precision=precision)
    return result, legacy


@functools.partial(jax.jit, static_argnames=("return_legacy",))
def solve_profile_contrast(
    H,
    g,
    h,
    t,
    nu,
    y_norm_sq,
    lambdas,
    contrast_nodes,
    contrast_mean,
    contrast_variance,
    *,
    return_legacy: bool = False,
):
    """Profile-MAP contrast solver (backward-compatible with RECOVAR).

    Selects the contrast node that minimizes the profile objective
    ``J_prof(c) = r(c) - q(c)^T A(c)^{-1} q(c) - 2 log p(c)``
    (no log-determinant term).

    Returns full posterior moments at the selected contrast, as if that
    contrast were known with certainty.
    """
    C = contrast_nodes.shape[0]

    # Shared spectral precomputation
    ss = _compute_contrast_scores(H, g, h, t, nu, y_norm_sq, lambdas, contrast_nodes, contrast_mean, contrast_variance)

    # Profile objective (lower is better): no logdet term
    scores = ss.rho - ss.quad - 2.0 * ss.log_prior[None, :]  # (B, C)

    best_idx = jnp.argmin(scores, axis=-1)  # (B,)
    best_c = contrast_nodes[best_idx]  # (B,)

    # One-hot weights at best contrast → shared moment computation
    omega = jax.nn.one_hot(best_idx, C, dtype=g.dtype)  # (B, C)
    mean_z, cov_z, second_moment_z, mean_c, second_moment_c, mean_cz, mean_c2z, second_moment_czz = (
        _compute_weighted_moments(ss.T, ss.s, ss.r_spec, contrast_nodes, omega)
    )

    result = LatentPosteriorResult(
        mean_z=mean_z,
        cov_z=cov_z,
        second_moment_z=second_moment_z,
        mean_c=mean_c,
        second_moment_c=second_moment_c,
        mean_cz=mean_cz,
        mean_c2z=mean_c2z,
        second_moment_czz=second_moment_czz,
        contrast_nodes=contrast_nodes,
        contrast_weights_posterior=omega,
        best_contrast=best_c,
        profile_scores=scores,
        marginal_ll=None,
    )

    if not return_legacy:
        return result

    # Legacy precision
    best_c2 = best_c**2
    gram = best_c2[:, None, None] * H
    A_best = gram + jnp.diag(1.0 / lambdas)
    precision = A_best @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ A_best
    legacy = LegacyEmbeddingResult(xs=mean_z, contrasts=best_c, precision=precision)
    return result, legacy


@functools.partial(jax.jit, static_argnames=("return_legacy",))
def solve_marginalized_contrast(
    H,
    g,
    h,
    t,
    nu,
    y_norm_sq,
    lambdas,
    contrast_nodes,
    contrast_weights,
    contrast_mean,
    contrast_variance,
    *,
    return_legacy: bool = False,
):
    """Exact contrast-marginalized latent posterior via quadrature.

    Uses a single batched eigendecomposition of ``Lambda^{1/2} H Lambda^{1/2}``
    to evaluate all contrast nodes with elementwise arithmetic only.

    The collapsed marginal score is:
    ``J_marg(c) = r(c) - q(c)^T A(c)^{-1} q(c) + log det A(c) - 2 log p(c)``

    The ``log det`` term is essential here — it accounts for the volume of the
    Gaussian posterior at each contrast, which varies across nodes.

    Scope: supports one contrast scalar per solve only.
    """
    # Shared spectral precomputation
    ss = _compute_contrast_scores(H, g, h, t, nu, y_norm_sq, lambdas, contrast_nodes, contrast_mean, contrast_variance)

    c2 = contrast_nodes**2

    # Log-determinant: log det A(c) = const + sum_k log(1 + c^2 d_k)
    # The constant (sum_k log(1/lambda_k)) cancels in normalization
    logdet = jnp.sum(jnp.log1p(ss.d[:, None, :] * c2[None, :, None]), axis=-1)  # (B, C)

    # Log unnormalized weights: log w_j + log p(c_j) - 0.5 * (rho - quad + logdet)
    log_unnorm = (
        jnp.log(jnp.clip(contrast_weights, min=1e-30))[None, :]
        + ss.log_prior[None, :]
        - 0.5 * (ss.rho - ss.quad + logdet)
    )  # (B, C)

    # Stable softmax → normalized weights
    omega = jax.nn.softmax(log_unnorm, axis=-1)  # (B, C)

    # Marginal log-likelihood: log sum_j exp(log_unnorm_j)
    marginal_ll = jax.scipy.special.logsumexp(log_unnorm, axis=-1)  # (B,)

    # Shared moment computation
    mean_z, cov_z, second_moment_z, mean_c, second_moment_c, mean_cz, mean_c2z, second_moment_czz = (
        _compute_weighted_moments(ss.T, ss.s, ss.r_spec, contrast_nodes, omega)
    )

    # Best contrast (MAP of posterior weights)
    best_idx = jnp.argmax(omega, axis=-1)
    best_contrast = contrast_nodes[best_idx]

    result = LatentPosteriorResult(
        mean_z=mean_z,
        cov_z=cov_z,
        second_moment_z=second_moment_z,
        mean_c=mean_c,
        second_moment_c=second_moment_c,
        mean_cz=mean_cz,
        mean_c2z=mean_c2z,
        second_moment_czz=second_moment_czz,
        contrast_nodes=contrast_nodes,
        contrast_weights_posterior=omega,
        best_contrast=best_contrast,
        profile_scores=-log_unnorm,  # negative log for consistency
        marginal_ll=marginal_ll,
    )

    if not return_legacy:
        return result

    # Legacy: use posterior mean contrast for the precision matrix
    gram = mean_c[:, None, None] ** 2 * H
    A_mc = gram + jnp.diag(1.0 / lambdas)
    precision = A_mc @ jnp.linalg.pinv(gram, rcond=1e-6, hermitian=True) @ A_mc
    legacy = LegacyEmbeddingResult(xs=mean_z, contrasts=best_contrast, precision=precision)
    return result, legacy


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------


def solve_latent_posterior(
    H: jax.Array,
    g: jax.Array,
    h: jax.Array,
    t: jax.Array,
    nu: jax.Array,
    y_norm_sq: jax.Array,
    lambdas: jax.Array,
    *,
    contrast_mode: Literal["none", "profile", "marginalize"] = "profile",
    contrast_nodes: Optional[jax.Array] = None,
    contrast_weights: Optional[jax.Array] = None,
    contrast_mean: float = 1.0,
    contrast_variance: float = np.inf,
    contrast_rule: Literal["gauss_legendre", "trapezoid", "custom"] = "gauss_legendre",
    contrast_interval: tuple[float, float] = (0.0, 3.0),
    n_contrast_nodes: int = 32,
    return_legacy: bool = False,
) -> LatentPosteriorResult | tuple[LatentPosteriorResult, LegacyEmbeddingResult]:
    """Unified dispatch for latent posterior inference.

    Parameters
    ----------
    H : (B, K, K)
        Gram matrix ``B^T B`` (AU_t_AU).
    g : (B, K)
        ``B^T y`` (AU_t_images).
    h : (B, K)
        ``B^T m`` (AU_t_Amean).
    t : (B,)
        ``y^T m`` (image_T_A_mean).
    nu : (B,)
        ``m^T m`` (A_mean_norm_sq).
    y_norm_sq : (B,)
        ``y^T y`` (image_norms_sq).
    lambdas : (K,)
        Prior variances (eigenvalues of the latent prior covariance Lambda).
    contrast_mode : str
        ``"none"`` (c=1), ``"profile"`` (grid-search MAP), ``"marginalize"``
        (quadrature integration).
    contrast_nodes : array, optional
        Explicit contrast grid/quadrature nodes.
    contrast_weights : array, optional
        Quadrature weights (only for ``"marginalize"``).
    contrast_mean : float
        Prior mean for contrast.
    contrast_variance : float
        Prior variance for contrast (``inf`` for flat prior).
    contrast_rule : str
        Quadrature rule when nodes/weights are not supplied.
    contrast_interval : (float, float)
        Integration interval for contrast.
    n_contrast_nodes : int
        Number of quadrature nodes.
    return_legacy : bool
        Also return ``LegacyEmbeddingResult`` for backward compatibility.

    Returns
    -------
    result : LatentPosteriorResult
        Posterior moments.
    legacy : LegacyEmbeddingResult (only if return_legacy=True)
        Legacy RECOVAR-compatible outputs.
    """
    contrast_mean_jax = jnp.asarray(contrast_mean, dtype=jnp.float32)
    contrast_variance_jax = jnp.asarray(contrast_variance, dtype=jnp.float32)

    if contrast_mode == "none":
        return solve_no_contrast(
            H,
            g,
            h,
            t,
            nu,
            y_norm_sq,
            lambdas,
            return_legacy=return_legacy,
        )

    # Build or validate quadrature
    if contrast_nodes is None:
        nodes, weights = make_contrast_quadrature(
            rule=contrast_rule,
            interval=contrast_interval,
            n_nodes=n_contrast_nodes,
        )
    else:
        nodes = jnp.asarray(contrast_nodes, dtype=jnp.float32)
        if contrast_weights is not None:
            weights = jnp.asarray(contrast_weights, dtype=jnp.float32)
        else:
            # Trapezoid weights for user-supplied nodes
            _, weights = make_contrast_quadrature(
                rule="trapezoid",
                nodes=np.asarray(nodes),
            )

    if contrast_mode == "profile":
        return solve_profile_contrast(
            H,
            g,
            h,
            t,
            nu,
            y_norm_sq,
            lambdas,
            nodes,
            contrast_mean_jax,
            contrast_variance_jax,
            return_legacy=return_legacy,
        )
    elif contrast_mode == "marginalize":
        return solve_marginalized_contrast(
            H,
            g,
            h,
            t,
            nu,
            y_norm_sq,
            lambdas,
            nodes,
            weights,
            contrast_mean_jax,
            contrast_variance_jax,
            return_legacy=return_legacy,
        )
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode!r}")
