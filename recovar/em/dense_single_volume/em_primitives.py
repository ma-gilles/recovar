"""Bridge for dense-owned kernels reused by local EM engines.

The remaining exports are JIT kernels still implemented in ``em_engine``.
Higher-level helpers should live in purpose-specific helper modules instead of
being re-exported here.
"""

from .em_engine import (
    _adjoint_slice_volume_half,
    _adjoint_slice_volume_windowed,
    _batch_adjoint_slice_volume_half,
    _batch_adjoint_slice_volume_windowed,
    _block_until_ready,
    _compute_noise_block,
    _compute_projections_block,
)

__all__ = [
    "_adjoint_slice_volume_half",
    "_adjoint_slice_volume_windowed",
    "_batch_adjoint_slice_volume_half",
    "_batch_adjoint_slice_volume_windowed",
    "_block_until_ready",
    "_compute_noise_block",
    "_compute_projections_block",
]
