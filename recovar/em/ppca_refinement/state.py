"""Driver state for ``recovar ppca-refine`` (Milestones 3+).

Two state containers grow across milestones:

* :class:`FixedPosePPCAState` (M3) — the minimum needed to call the legacy
  fixed-pose ``recovar.ppca.ppca.EM(...)``: mean, loadings, prior, mask,
  contrast, masks-for-multimask, loose hyperparameters. No pose tables.
* :class:`PoseMarginalPPCAEMState` (M5+) — the full augmented EM state
  with halfsets, pose tables, pose priors, refinement schedule, and
  diagnostics.

Both are frozen dataclasses (PyTree-safe) with ``replace(...)`` semantics
inherited from :mod:`dataclasses`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax

from recovar.ppca import PCPriorConfig

__all__ = [
    "FixedPosePPCAState",
    "PoseMarginalPPCAEMState",
]


@dataclass(frozen=True)
class FixedPosePPCAState:
    """Minimal state for ``recovar ppca-refine --pose-mode fixed`` (M3).

    Holds inputs to a single call into ``recovar.ppca.ppca.EM(...)``: mean
    estimate, initial loadings, loading prior, masks, optional contrast
    parameters. Pose tables are not part of this state — the dataset
    already carries fixed poses for the fixed-pose mode.
    """

    mean_estimate: jax.Array  # Fourier volume (full, complex64)
    W_initial: jax.Array  # [half_vol, q] complex64
    W_prior: jax.Array  # [half_vol, q] real, variance
    volume_mask: jax.Array | None = None
    dilated_volume_mask: jax.Array | None = None
    masks: jax.Array | None = None  # multimask: (M, D, D, D) real
    pc_mask_assignment: jax.Array | None = None  # (q,) int
    contrast_mode: str = "none"
    contrast_grid: jax.Array | None = None
    contrast_weights: jax.Array | None = None
    contrast_mean: float = 1.0
    contrast_variance: float = float("inf")
    pc_prior_config: PCPriorConfig | None = None

    def replace(self, **changes) -> "FixedPosePPCAState":
        return replace(self, **changes)


@dataclass(frozen=True)
class PoseMarginalPPCAEMState:
    """Full augmented EM state for M5+ (dense / sparse pose-marginalized).

    ``mu_score`` and ``W_score`` are the filtered halfset averages used by
    the E-step (see ``recovar.em.dense_single_volume.helpers.convergence``).
    Other fields mirror the contract in
    ``recovar/em/ppca_refinement/CLAUDE.md`` §10.
    """

    mu_half: tuple[jax.Array, jax.Array]
    W_half: tuple[jax.Array, jax.Array]
    mu_score: jax.Array
    W_score: jax.Array

    W_prior: jax.Array  # [half_vol, q] variance
    mean_prior: Any
    z_prior_precision_diag: jax.Array  # ones(q) in v1

    noise_variance: jax.Array
    contrast_params: Any  # ContrastParams (mode='none' until M8)
    masks: Any  # MaskSpec (single or multi)

    pose_estimates: Any
    pose_priors: Any
    refinement_schedule_state: Any
    hyperparams: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> "PoseMarginalPPCAEMState":
        return replace(self, **changes)
