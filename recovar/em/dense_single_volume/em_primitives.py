"""Bridge for shared dense/local EM kernels.

Higher-level helpers should live in purpose-specific helper modules instead of
being re-exported here.
"""

from .helpers.backprojection import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
    batch_adjoint_slice_volume_half as _batch_adjoint_slice_volume_half,
    batch_adjoint_slice_volume_windowed as _batch_adjoint_slice_volume_windowed,
)
from .helpers.jax_runtime import block_until_ready as _block_until_ready
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
]
