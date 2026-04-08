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

import numpy as np

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
