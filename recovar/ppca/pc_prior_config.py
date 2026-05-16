"""Configuration for PPCA loading priors used by refinement EM."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PCPriorConfig:
    """JSON-serializable policy for the PPCA loading-volume prior.

    The latent prior is identity in this refinement-first implementation.
    Eigenvalue scale lives in ``W``. ``W_prior`` is variance-like: larger
    values mean weaker regularization, and solvers invert it explicitly.
    """

    latent_prior_mode: str = "identity"
    pc_prior_mode: str = "hybrid_shell"
    prior_scale: float = 1.0
    variance_floor: float = 1e-8
    use_q_total_for_division: bool = True
    smooth_shell_prior: bool = True
    prior_freeze_iters: int = 3
    recompute_once_after_iter: int | None = 5
    allow_every_iter_prior_update: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def make_shell_w_prior(shell_variance, shell_indices, n_components: int, *, q_total: int, config: PCPriorConfig):
    """Broadcast the documented shell formula to ``[n_frequency, q]``.

    ``W_prior(xi, k) = max(tau_floor, alpha_prior * d_ppca(shell(xi)) / q_total)``.
    ``q_total`` is the requested model dimension, not the number currently
    active in an annealed run.
    """
    if config.latent_prior_mode != "identity":
        raise ValueError("Only latent_prior_mode='identity' is supported in the refinement foundation")
    n_components = int(n_components)
    q_total = int(q_total)
    if n_components < 0:
        raise ValueError("n_components must be nonnegative")
    if q_total <= 0 and n_components > 0:
        raise ValueError("q_total must be positive when n_components > 0")
    shell_variance = np.asarray(shell_variance, dtype=np.float64)
    shell_indices = np.asarray(shell_indices, dtype=np.int64)
    if shell_indices.ndim != 1:
        raise ValueError(f"shell_indices must be 1D, got {shell_indices.shape}")
    if shell_indices.size and (shell_indices.min() < 0 or shell_indices.max() >= shell_variance.size):
        raise ValueError("shell_indices contain values outside shell_variance")
    if n_components == 0:
        return np.zeros((shell_indices.size, 0), dtype=np.float32)
    divisor = q_total if config.use_q_total_for_division else n_components
    per_frequency = config.prior_scale * shell_variance[shell_indices] / float(divisor)
    per_frequency = np.maximum(float(config.variance_floor), per_frequency)
    return np.broadcast_to(per_frequency[:, None], (shell_indices.size, n_components)).astype(np.float32)
