"""Shared low-level EM primitives reused by dense and local engines.

This file intentionally re-exports the current dense-engine helpers so the
local exact engine can reuse geometry-agnostic kernels without routing through
the dense `run_em(...)` orchestration.
"""

from .helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
    batch_adjoint_slice_volume_half as _batch_adjoint_slice_volume_half,
    batch_adjoint_slice_volume_windowed as _batch_adjoint_slice_volume_windowed,
)
from .helpers.half_spectrum import make_half_image_weights, make_relion_noise_shell_indices_half, make_shell_indices_half
from .helpers.jax_runtime import block_until_ready as _block_until_ready
from .helpers.preprocessing import (
    prepare_reconstruction_batch as _prepare_reconstruction_batch,
    preprocess_batch as _preprocess_batch,
)
from .helpers.projection import (
    compute_noise_block as _compute_noise_block,
    compute_projections_block as _compute_projections_block,
)

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
