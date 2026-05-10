"""Centralized configuration dataclasses for PPCA refinement EM iterations.

Every tunable knob that an EM iteration accepts lives here. A reader who wants
to know what dials exist should be able to read this file end-to-end in two
minutes.

Concerns are split orthogonally:

* :class:`GeometryConfig` — volume/image shape, latent dim, domain.
* :class:`ScheduleConfig` — batch sizes, M-step solve chunk size.
* :class:`ScoringConfig` — image-side scoring options.
* :class:`SparsePass2Config` — pass-2 backprojection sparsity culling.
* :class:`MeanRegularizationConfig` — re-exported from :mod:`mean_regularization`.
* :class:`PostprocessConfig` — re-exported from :mod:`postprocess`.

Every config is ``frozen=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovar.em.ppca_refinement.mean_regularization import MeanRegularizationConfig
from recovar.em.ppca_refinement.postprocess import PostprocessConfig

__all__ = [
    "GeometryConfig",
    "ScheduleConfig",
    "ScoringConfig",
    "SparsePass2Config",
    "MeanRegularizationConfig",
    "PostprocessConfig",
]


@dataclass(frozen=True)
class GeometryConfig:
    """Shape / domain knobs."""

    current_size: int | None = None
    q: int | None = None
    volume_domain: str = "auto"


@dataclass(frozen=True)
class ScheduleConfig:
    """Batching / chunking knobs."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000
    mstep_chunk_size: int | None = None


@dataclass(frozen=True)
class ScoringConfig:
    """Per-image scoring options."""

    score_with_masked_images: bool = False
    half_spectrum_scoring: bool = False
    square_window: bool = False
    relion_texture_interp: bool = True
    class_log_prior: float = 0.0
    image_scale_corrections: np.ndarray | None = None


@dataclass(frozen=True)
class SparsePass2Config:
    """Pass-2 backprojection culling."""

    enabled: bool = True
    log_threshold: float = float(np.log(1.0e-6))
