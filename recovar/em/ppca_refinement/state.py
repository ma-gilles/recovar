"""State containers for PPCA angle-refinement EM."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax


@dataclass(frozen=True)
class PoseMarginalPPCAEMState:
    """Halfset-aware PPCA refinement state.

    ``mu_score`` and ``W_score`` are the combined scoring model. ``mu_half``
    and ``W_half`` keep gold-standard halfset estimates.
    """

    mu_half: tuple[jax.Array, jax.Array]
    W_half: tuple[jax.Array, jax.Array]
    mu_score: jax.Array
    W_score: jax.Array
    W_prior: jax.Array
    mean_prior: Any
    noise_variance: jax.Array
    schedule_state: Any
    pose_diagnostics: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> "PoseMarginalPPCAEMState":
        return replace(self, **changes)
