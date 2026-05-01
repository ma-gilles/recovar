"""Augmented PPCA sufficient-statistics container (Milestone 2).

The augmented M-step solves jointly for ``θ_aug = [μ, W₁, …, W_q]`` (i.e.
``q+1`` components, with the mean ``μ`` in slot 0). The per-pose moments
``α`` and ``G`` from the per-pose marginalization (see
``recovar/ppca/pose_marginal.py``) are accumulated against per-image
backprojection operators ``A_ia`` to produce:

* ``rhs[ξ, r]`` — complex-half-Fourier coefficient at voxel ξ for component
  ``r ∈ {0, …, q}``. Sums over images and pose hypotheses of
  ``γ · α_r · A* x``.
* ``lhs_tri[ξ, idx]`` — the upper-triangle of the (q+1, q+1) Hermitian
  Gram per voxel, in row-major order matching
  ``recovar.ppca.ppca.unpack_tri_to_full``. Sums over images and pose
  hypotheses of ``γ · G_rs · A* A``.

This dataclass is frozen and pickleable so it can be cached / written into
diagnostics without aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax

__all__ = ["AugmentedPPCAStats"]


@dataclass(frozen=True)
class AugmentedPPCAStats:
    """Sufficient stats for the augmented [μ, W₁..W_q] M-step.

    Attributes
    ----------
    rhs:
        ``[half_vol, q+1]`` complex64. Component ``r=0`` is the mean
        backprojection; ``r ≥ 1`` are the loading components.
    lhs_tri:
        ``[half_vol, (q+1)(q+2)/2]``. Real if no contrast (Hermitian
        symmetric reduces to symmetric for the no-contrast case where
        diagonal is real and off-diagonals carry the conjugate-pair
        structure absorbed into the upper-triangle). Complex64 for
        contrast paths.
    residual_num, residual_den:
        Per-Fourier-shell residual numerator and denominator for noise /
        FSC updates. ``None`` until residual accounting is wired in by the
        engines.
    log_likelihood:
        Sum of per-image log-evidences for diagnostics.
    n_images:
        Number of images that contributed to this accumulator.
    diagnostics:
        Free-form dict for engine-specific diagnostics (e.g. omitted-mass
        estimates, posterior pmax statistics). Not deeply structured — the
        EM driver picks out keys it cares about.
    """

    rhs: jax.Array
    lhs_tri: jax.Array
    residual_num: jax.Array | None = None
    residual_den: jax.Array | None = None
    log_likelihood: float = 0.0
    n_images: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
