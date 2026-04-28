"""Shared low-level EM primitives reused by dense and local engines.

This file intentionally re-exports the current dense-engine helpers so the
local exact engine can reuse geometry-agnostic kernels without routing through
the dense `run_em(...)` orchestration.
"""

from .em_engine import (
    _adjoint_slice_volume_half,
    _adjoint_slice_volume_windowed,
    _batch_adjoint_slice_volume_half,
    _batch_adjoint_slice_volume_windowed,
    _block_until_ready,
    _compute_noise_block,
    _compute_projections_block,
    _prepare_reconstruction_batch,
    _preprocess_batch,
)
from .helpers.half_spectrum import make_half_image_weights, make_relion_noise_shell_indices_half, make_shell_indices_half

__all__ = [
    "_adjoint_slice_volume_half",
    "_adjoint_slice_volume_windowed",
    "_batch_adjoint_slice_volume_half",
    "_batch_adjoint_slice_volume_windowed",
    "_block_until_ready",
    "_compute_noise_block",
    "_compute_projections_block",
    "_prepare_reconstruction_batch",
    "_preprocess_batch",
    "make_half_image_weights",
    "make_relion_noise_shell_indices_half",
    "make_shell_indices_half",
]
