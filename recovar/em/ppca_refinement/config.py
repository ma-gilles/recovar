"""Centralized configuration dataclasses for PPCA refinement EM iterations.

Every tunable knob that an EM iteration accepts lives here. A reader who wants
to know what dials exist should be able to read this file end-to-end in two
minutes.

Concerns are split orthogonally:

* :class:`GeometryConfig` — volume/image shape, latent dim, domain.
* :class:`ScheduleConfig` — batch sizes, M-step solve chunk size.
* :class:`ScoringConfig` — image-side scoring options.
* :class:`PoseSelectionConfig` — best/top-p pose diagnostic selection.
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
    "PoseSelectionConfig",
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
    local_image_shard_count: int = 1


@dataclass(frozen=True)
class ScoringConfig:
    """Per-image scoring options.

    Attributes
    ----------
    half_spectrum_scoring : bool
        Misleading legacy name. ``False`` (default) **enables Hermitian
        half-image weights** (``w_f = 2`` for non-DC/non-Nyquist pixels,
        ``w_f = 1`` else) so every Fourier inner product in the E-step
        equals the full-Fourier inner product exactly, the ``.real()``
        projections in :func:`engine._per_pose_stats_block` are exact,
        and ``z`` stays exactly real. ``True`` switches to RELION-style
        unit weights everywhere, which biases pose scores by ~½, makes
        ``Re(half_IP)`` an approximation of the full IP, and shifts the
        effective ``z`` prior — only use for RELION-parity tests.
    """

    score_with_masked_images: bool = False
    half_spectrum_scoring: bool = False
    square_window: bool = False
    relion_texture_interp: bool = True
    class_log_prior: float = 0.0
    image_scale_corrections: np.ndarray | None = None


@dataclass(frozen=True)
class PoseSelectionConfig:
    """Top-p pose diagnostic selection.

    The E-step/M-step math never depends on this config. It only controls which
    high-scoring poses are retained for diagnostics and for seeding later
    adaptive/local supports.
    """

    top_p_poses: int = 1
    top_pose_max_log_score_gap: float = 3.0
    top_pose_min_angle_deg: float = 0.5
    top_pose_min_translation_px: float = 0.5
    candidate_pool_factor: int = 8
    min_candidate_pool: int = 16

    def __post_init__(self) -> None:
        if int(self.top_p_poses) < 1:
            raise ValueError("top_p_poses must be >= 1")
        if int(self.candidate_pool_factor) < 1:
            raise ValueError("candidate_pool_factor must be >= 1")
        if int(self.min_candidate_pool) < 1:
            raise ValueError("min_candidate_pool must be >= 1")


@dataclass(frozen=True)
class SparsePass2Config:
    """Pass-2 backprojection culling."""

    enabled: bool = True
    log_threshold: float = float(np.log(1.0e-6))
    local_mstep_top_k: int = 0
    local_mstep_min_pmax: float = 0.999

    def __post_init__(self) -> None:
        if int(self.local_mstep_top_k) < 0:
            raise ValueError("local_mstep_top_k must be >= 0")
        if not 0.0 <= float(self.local_mstep_min_pmax) <= 1.0:
            raise ValueError("local_mstep_min_pmax must be in [0, 1]")
