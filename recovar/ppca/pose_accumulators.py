"""Augmented PPCA sufficient-statistics containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax


@dataclass(frozen=True)
class AugmentedPPCAStats:
    """Sufficient stats for the joint augmented ``[mu, W_1, ..., W_q]`` M-step.

    ``rhs`` has shape ``[n_frequency, q+1]``. ``lhs_tri`` has shape
    ``[n_frequency, (q+1)(q+2)//2]`` and stores the upper triangle of the
    augmented Gram for each frequency. Component 0 is the mean. Components
    1..q are PPCA loadings.
    """

    rhs: jax.Array
    lhs_tri: jax.Array
    residual_num: jax.Array | None = None
    residual_den: jax.Array | None = None
    log_likelihood: float = 0.0
    n_images: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
