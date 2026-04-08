"""Metrics for the PPCA-ab-initio v0 stage gates.

Per spec Section 10, every stage has exactly one **pre-registered
primary metric** plus context metrics. v0 ships the metrics needed
for Stage 0B / 1A (score stages); the mean / subspace / embedding
metrics for Stage 1B+ are added in their own commits.

Pre-registered primary metrics:

- Score stages (0B, 1A): `true_state_mass` on the validation split.
- Mean stages (1B): `fourier_relative_error_mu`.
- Factor stages (1C+): `projector_frobenius_error`.

`bootstrap_ci_mean` returns a 95% bootstrap CI on the mean of any
per-image metric, which is the form Section 10.7 / 11.2 requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

# ---------------------------------------------------------------------------
# Hidden-state / alignment metrics (Section 10.1)
# ---------------------------------------------------------------------------


def per_image_true_state_mass(log_resp, r_true_idx, t_true_idx) -> np.ndarray:
    """For each image, return `gamma_{i, g_true(i)}` — the posterior
    responsibility on the true grid hidden state.

    Parameters
    ----------
    log_resp : (n_img, n_rot, n_trans) — image-normalized log responsibilities
    r_true_idx : (n_img,) int — true rotation index per image
    t_true_idx : (n_img,) int — true translation index per image

    Returns
    -------
    mass : (n_img,) float64 in [0, 1]
    """
    log_resp_np = np.asarray(log_resp)
    n_img = log_resp_np.shape[0]
    r = np.asarray(r_true_idx).astype(np.int64)
    t = np.asarray(t_true_idx).astype(np.int64)
    return np.exp(log_resp_np[np.arange(n_img), r, t])


def true_state_mass(log_resp, r_true_idx, t_true_idx) -> float:
    """Mean of `per_image_true_state_mass`. The Section 10.1 primary
    metric for score stages.
    """
    return float(np.mean(per_image_true_state_mass(log_resp, r_true_idx, t_true_idx)))


def per_image_top1_acc(log_resp, r_true_idx, t_true_idx) -> np.ndarray:
    """For each image, 1 if the argmax of `log_resp` equals
    `(r_true(i), t_true(i))`, else 0."""
    log_resp_np = np.asarray(log_resp)
    n_img, n_rot, n_trans = log_resp_np.shape
    flat = log_resp_np.reshape(n_img, -1)
    argmax = flat.argmax(axis=-1)
    r_pred = argmax // n_trans
    t_pred = argmax % n_trans
    return ((r_pred == np.asarray(r_true_idx)) & (t_pred == np.asarray(t_true_idx))).astype(np.float64)


def top1_acc(log_resp, r_true_idx, t_true_idx) -> float:
    return float(np.mean(per_image_top1_acc(log_resp, r_true_idx, t_true_idx)))


def per_image_true_state_rank(log_resp, r_true_idx, t_true_idx) -> np.ndarray:
    """For each image, the rank of the true `(r, t)` in the per-image
    posterior, where rank 0 is the argmax."""
    log_resp_np = np.asarray(log_resp)
    n_img, n_rot, n_trans = log_resp_np.shape
    flat = log_resp_np.reshape(n_img, -1)
    r = np.asarray(r_true_idx).astype(np.int64)
    t = np.asarray(t_true_idx).astype(np.int64)
    true_flat_idx = r * n_trans + t
    # Rank of true_flat_idx when entries are sorted descending
    # rank = number of entries strictly greater than the true entry
    true_log_resp = flat[np.arange(n_img), true_flat_idx]
    return np.sum(flat > true_log_resp[:, None], axis=-1).astype(np.float64)


def true_state_rank(log_resp, r_true_idx, t_true_idx) -> float:
    return float(np.mean(per_image_true_state_rank(log_resp, r_true_idx, t_true_idx)))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (Section 10.7)
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    mean: float
    ci_low: float
    ci_high: float
    level: float


def bootstrap_ci_mean(per_image_values, *, level: float = 0.95, n_bootstrap: int = 1000, seed: int = 0) -> BootstrapCI:
    """Bootstrap CI on the mean of a per-image metric.

    Parameters
    ----------
    per_image_values : (n_img,) array
    level : float
        Two-sided coverage. Default 0.95.
    n_bootstrap : int
    seed : int

    Returns
    -------
    BootstrapCI
    """
    vals = np.asarray(per_image_values, dtype=np.float64).reshape(-1)
    n = vals.size
    if n == 0:
        raise ValueError("per_image_values is empty")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(vals[idx]))
    alpha = (1.0 - level) / 2.0
    return BootstrapCI(
        mean=float(np.mean(vals)),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        level=level,
    )


# ---------------------------------------------------------------------------
# Stage 0B exit-criterion helpers (Section 11.2)
# ---------------------------------------------------------------------------


@dataclass
class ScoreDiagnostic:
    """One Stage 0B / 1A score-diagnostic result for one (family, seed)."""

    family: str
    seed: int
    n_val: int
    homog_true_state_mass: BootstrapCI
    ppca_true_state_mass: BootstrapCI
    delta_true_state_mass: BootstrapCI


def score_diagnostic_one_seed(
    log_resp_homog,
    log_resp_ppca,
    r_true_idx,
    t_true_idx,
    val_idx,
    *,
    family: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> ScoreDiagnostic:
    """Compute the Stage 0B per-(family, seed) score diagnostic.

    The Stage 0B exit criterion (Section 11.2) requires the
    PPCA-over-homogeneous gain on validation `true_state_mass` to
    have a 95% bootstrap CI excluding zero on family B, and to be
    `|Δ| ≤ 0.01` on family A.
    """
    val = np.asarray(val_idx, dtype=np.int64)
    r = np.asarray(r_true_idx)[val]
    t = np.asarray(t_true_idx)[val]
    log_resp_h = np.asarray(log_resp_homog)[val]
    log_resp_p = np.asarray(log_resp_ppca)[val]

    per_image_h = per_image_true_state_mass(log_resp_h, r, t)
    per_image_p = per_image_true_state_mass(log_resp_p, r, t)
    per_image_delta = per_image_p - per_image_h

    return ScoreDiagnostic(
        family=family,
        seed=seed,
        n_val=int(len(val)),
        homog_true_state_mass=bootstrap_ci_mean(per_image_h, level=0.95, n_bootstrap=n_bootstrap, seed=seed),
        ppca_true_state_mass=bootstrap_ci_mean(per_image_p, level=0.95, n_bootstrap=n_bootstrap, seed=seed + 1),
        delta_true_state_mass=bootstrap_ci_mean(per_image_delta, level=0.95, n_bootstrap=n_bootstrap, seed=seed + 2),
    )


# ---------------------------------------------------------------------------
# Mean metrics (Section 10.2)
# ---------------------------------------------------------------------------


def fourier_relative_error_mu(mu_est_half, mu_true_half, *, weights_half=None) -> float:
    """Relative Fourier-norm error of `mu_est` against `mu_true`,
    in half-volume layout.

    Returns
    -------
    err : float
        `||mu_est - mu_true|| / ||mu_true||` under the rfft-weighted
        Hermitian inner product (so that the result equals the
        full-spectrum relative error).
    """
    a = np.asarray(mu_est_half, dtype=np.complex128)
    b = np.asarray(mu_true_half, dtype=np.complex128)
    diff = a - b
    if weights_half is None:
        # Unweighted half-spectrum norm — coarse but layout-agnostic.
        num = float(np.sum(np.abs(diff) ** 2))
        den = float(np.sum(np.abs(b) ** 2))
    else:
        w = np.asarray(weights_half, dtype=np.float64)
        num = float(np.sum(w * np.abs(diff) ** 2))
        den = float(np.sum(w * np.abs(b) ** 2))
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return float(np.sqrt(num / den))


def oracle_fsc_gt(mu_est_half, mu_true_half, volume_shape: Sequence[int]) -> np.ndarray:
    """Shell-averaged Fourier shell correlation between `mu_est` and
    `mu_true`, both in half-volume layout. Returns the per-shell FSC
    array.

    **Important**: this is an oracle FSC against ground truth, NOT a
    split-map FSC. Per spec Section 10.2, do not apply half-bit /
    0.143 thresholds to this curve. Use `fourier_relative_error_mu`
    as the primary mean metric.
    """
    a = jnp.asarray(mu_est_half).reshape((volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1))
    b = jnp.asarray(mu_true_half).reshape((volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1))

    # Build a per-voxel radial shell index (matches the 3D rfft layout)
    N0, N1, N2 = (int(s) for s in volume_shape)
    kz = np.fft.fftshift(np.fft.fftfreq(N0)) * N0
    ky = np.fft.fftshift(np.fft.fftfreq(N1)) * N1
    kx = np.arange(N2 // 2 + 1, dtype=np.float64)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    R = np.round(np.sqrt(KZ**2 + KY**2 + KX**2)).astype(np.int32)
    n_shells = int(R.max()) + 1

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    fsc = np.zeros(n_shells, dtype=np.float64)
    for sh in range(n_shells):
        mask = R == sh
        if not np.any(mask):
            fsc[sh] = 0.0
            continue
        num = np.sum((a_np[mask].conj() * b_np[mask]).real)
        den = np.sqrt(np.sum(np.abs(a_np[mask]) ** 2) * np.sum(np.abs(b_np[mask]) ** 2))
        fsc[sh] = float(num / den) if den > 0 else 0.0
    return fsc


# ---------------------------------------------------------------------------
# Subspace metrics (Section 10.3)
# ---------------------------------------------------------------------------


def _decoded_real_basis(U_half, volume_shape):
    """Decode a `(q, half_volume_size)` complex matrix into a
    `(q, N_full)` real matrix via `get_idft3_real`. Used by every
    subspace metric below."""
    q = U_half.shape[0]
    N0, N1, N2 = (int(s) for s in volume_shape)
    half_shape = (N0, N1, N2 // 2 + 1)
    out = np.empty((q, N0 * N1 * N2), dtype=np.float64)
    for k in range(q):
        grid = jnp.asarray(U_half[k]).reshape(half_shape)
        real_vol = ftu.get_idft3_real(grid, volume_shape=tuple(volume_shape))
        out[k] = np.asarray(real_vol).reshape(-1)
    return out


def _orthonormalize_real(U_real):
    """QR-orthonormalize the rows of a real `(q, N)` matrix in the
    standard real-space inner product. Used so that we can compare
    spans even if the input is not pre-orthonormalized."""
    q, _N = U_real.shape
    Q, _R = np.linalg.qr(U_real.T)
    return Q.T[:q]  # orthonormal rows


def projector_frobenius_error(U_est_half, U_true_half, volume_shape: Sequence[int]) -> float:
    """Frobenius norm of the difference between projectors onto the
    row spans of `U_est` and `U_true`, both decoded to real space.
    The metric is gauge-invariant under real `O(q)` transformations
    of either basis.
    """
    Ue = _orthonormalize_real(_decoded_real_basis(U_est_half, volume_shape))
    Ut = _orthonormalize_real(_decoded_real_basis(U_true_half, volume_shape))
    Pe = Ue.T @ Ue
    Pt = Ut.T @ Ut
    return float(np.linalg.norm(Pe - Pt, ord="fro"))


def principal_angles_deg(U_est_half, U_true_half, volume_shape: Sequence[int]) -> np.ndarray:
    """Principal angles (in degrees) between the row spans of
    `U_est` and `U_true` after decoding both to real space and
    orthonormalizing.
    """
    Ue = _orthonormalize_real(_decoded_real_basis(U_est_half, volume_shape))
    Ut = _orthonormalize_real(_decoded_real_basis(U_true_half, volume_shape))
    M = Ue @ Ut.T  # (q_e, q_t)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.rad2deg(np.arccos(s))


# ---------------------------------------------------------------------------
# Embedding metrics (Section 10.5)
# ---------------------------------------------------------------------------


def _orthogonal_procrustes(A, B):
    """Find the real orthogonal matrix R minimizing ||A R - B||_F.

    A, B : (n, q) real
    Returns
    -------
    R : (q, q) real orthogonal
    """
    M = A.T @ B
    U, _S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R


def embedding_error_oracle(post_mean, alpha_true, r_true_idx, t_true_idx) -> float:
    """Oracle embedding error: extract `m_{i, g_true(i)}` from the
    posterior, align to `alpha_true` via real orthogonal Procrustes,
    return the residual L2 error per image.

    post_mean : (n_img, n_rot, n_trans, q) float64
    alpha_true : (n_img, q) float64
    """
    pm = np.asarray(post_mean)
    n_img, _n_rot, _n_trans, q = pm.shape
    r = np.asarray(r_true_idx).astype(np.int64)
    t = np.asarray(t_true_idx).astype(np.int64)
    m_oracle = pm[np.arange(n_img), r, t, :]  # (n_img, q)
    alpha = np.asarray(alpha_true, dtype=np.float64)
    R = _orthogonal_procrustes(m_oracle, alpha)
    aligned = m_oracle @ R
    diff = aligned - alpha
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))


def embedding_error_marginal(post_mean, log_resp, alpha_true) -> float:
    """Marginal embedding error: compute the marginal latent
    `α̂_i = Σ_g γ_{i,g} m_{i,g}`, align via real orthogonal
    Procrustes, return the residual L2 error per image.
    """
    pm = np.asarray(post_mean)  # (n_img, n_rot, n_trans, q)
    lr = np.asarray(log_resp)  # (n_img, n_rot, n_trans)
    gamma = np.exp(lr)
    alpha_hat = np.sum(gamma[..., None] * pm, axis=(1, 2))  # (n_img, q)
    alpha = np.asarray(alpha_true, dtype=np.float64)
    R = _orthogonal_procrustes(alpha_hat, alpha)
    aligned = alpha_hat @ R
    diff = aligned - alpha
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
