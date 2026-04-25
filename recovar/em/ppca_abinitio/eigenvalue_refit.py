"""Post-EM eigenvalue calibration for ab-initio PPCA.

Phase 2 of the ab-initio v0 plan. The EM loop runs with `s = 1` flat
because the prior is negligible vs the likelihood Gram at cryo-EM
SNR (validated 2026-04-16, see `ppca_abinitio_clean_algorithm.md`).
The eigenvalue magnitudes themselves are still useful for downstream
embedding scaling, multi-restart selection, and reporting the relative
PC variance, so we estimate them in a separate post-EM pass.

Approach: one-shot posterior-covariance refit. After EM converges,
run one E-step at the trained noise level (no annealing factor),
form the per-image expected outer product
`E[α_i α_i^T | y_i] = Σ_{r,t} γ_{i,r,t} (m_{i,r,t} m_{i,r,t}^T + H_{i,r}^{-1})`
where H is translation-independent (Section 4.6 of the v0 plan),
sample-average over images, and eigendecompose the resulting q×q
covariance to get the refit eigenvalues and a basis rotation.

This is the analog of the `pca_by_projected_covariance` step on the
sister branch `claude/ppca-refit-algorithms`, restricted to the
posterior-moment formulation that fits naturally inside the v0
half-volume forward model. A future variant that uses the data
covariance directly (matching the sister branch's full implementation)
would reduce posterior-discretization bias further; that is recorded
as Phase 6 future work.

Important caveats:
- One-shot only — does NOT propagate the refit s back into another EM
  iteration. Iterative Tipping-Bishop has been shown harmful in the
  pose-marginalized setting (see Section 9.2 of the status doc).
- Refits eigenvalues AND rotates the U basis to align with the new
  principal directions. The represented covariance `U diag(s) U^T`
  is preserved as a model object: U_new diag(s_new) U_new^T equals
  the sample-average posterior covariance projected back through U.
- Designed for evaluation/reporting, not as a feedback signal during
  EM.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .posterior import score_and_posterior_moments_eqx
from .types import PPCAInit


@dataclass
class EigenvalueRefitInfo:
    """Diagnostic bundle returned alongside the refit state."""

    s_em: np.ndarray  # eigenvalues going into the refit (typically all 1.0 with --s-init flat)
    s_refit: np.ndarray  # refit eigenvalues, descending
    rotation: np.ndarray  # (q, q) basis rotation; U_new = rotation.T @ U_old
    sigma_alpha: np.ndarray  # (q, q) sample-averaged posterior covariance


def _empirical_alpha_covariance(stats) -> jnp.ndarray:
    """Per-image E[α α^T] aggregated to the q×q dataset-level covariance.

    Uses the v0 PosteriorStats layout where post_Hinv is
    translation-independent: shape (n_img, n_rot, q, q).
    """
    log_resp = stats.log_resp  # (n_img, n_rot, n_trans), log responsibilities
    post_mean = stats.post_mean  # (n_img, n_rot, n_trans, q)
    post_Hinv = stats.post_Hinv  # (n_img, n_rot, q, q)

    gamma = jnp.exp(log_resp)  # responsibilities

    # m m^T term, summed over (r, t) and weighted by γ
    # ein: gamma_{i,r,t} · m_{i,r,t,k} · m_{i,r,t,l} → (n_img, q, q)
    mm_term = jnp.einsum("irt,irtk,irtl->ikl", gamma, post_mean, post_mean)

    # H^{-1} term: post_Hinv is per-(image, rotation); marginalize γ over translations
    gamma_per_rot = jnp.sum(gamma, axis=2)  # (n_img, n_rot)
    hinv_term = jnp.einsum("ir,irkl->ikl", gamma_per_rot, post_Hinv)

    M_per_image = mm_term + hinv_term  # (n_img, q, q)
    sigma_alpha = jnp.mean(M_per_image, axis=0)  # (q, q)
    # Symmetrize against floating-point asymmetry from the einsum.
    return 0.5 * (sigma_alpha + sigma_alpha.T)


def refit_eigenvalues_post_em(
    cur: PPCAInit,
    cfg,
    ds,
) -> tuple[PPCAInit, EigenvalueRefitInfo]:
    """One-shot posterior-covariance eigenvalue refit.

    Parameters
    ----------
    cur : PPCAInit
        Final EM state (μ, U, s).
    cfg : ForwardModelConfig
    ds : SyntheticDataset (or any object exposing the same fields used
         by `score_and_posterior_moments_eqx`).

    Returns
    -------
    refit_state : PPCAInit
        Same μ, rotated U, and refit s. The new s is in descending
        order; U has been rotated so that column k corresponds to
        the k-th principal direction in the refit basis.
    info : EigenvalueRefitInfo
        Diagnostic bundle.
    """
    stats = score_and_posterior_moments_eqx(
        cfg,
        cur.mu,
        cur.U,
        cur.s,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    sigma_alpha = _empirical_alpha_covariance(stats)
    sigma_alpha_np = np.asarray(sigma_alpha)

    # Eigendecompose; jnp.linalg.eigh returns ascending eigenvalues.
    eigvals_asc, V_asc = np.linalg.eigh(sigma_alpha_np)
    s_refit = eigvals_asc[::-1].astype(np.float64)
    V = V_asc[:, ::-1].astype(np.float64)  # (q, q)

    # Rotate U so the new k-th row aligns with the k-th principal direction.
    # Old U has shape (q, V_half); U_new[k, v] = Σ_l V[l, k] · U[l, v]
    U_new = jnp.asarray(V.T) @ cur.U
    U_new = U_new.astype(jnp.complex128)

    refit_state = PPCAInit(
        mu=cur.mu,
        U=U_new,
        s=jnp.asarray(np.maximum(s_refit, 1e-30), dtype=jnp.float64),
        volume_shape=cur.volume_shape,
    )
    info = EigenvalueRefitInfo(
        s_em=np.asarray(cur.s),
        s_refit=s_refit,
        rotation=V,
        sigma_alpha=sigma_alpha_np,
    )
    return refit_state, info
