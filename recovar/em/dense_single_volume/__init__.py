"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.
"""

from .engine_v2 import (
    compute_e_step_weights as compute_e_step_weights,
)
from .refine import (
    fsc_to_current_size as fsc_to_current_size,
)
from .refine import (
    refine_single_volume as refine_single_volume,
)
from .refine_dev_helpers.adaptive import (
    find_significant_mask as find_significant_mask,
)
from .refine_dev_helpers.adaptive import (
    find_significant_rotations as find_significant_rotations,
)
from .refine_dev_helpers.fourier_window import (
    ALLOWED_CURRENT_SIZES as ALLOWED_CURRENT_SIZES,
)
from .refine_dev_helpers.fourier_window import (
    make_fourier_window_indices_np as make_fourier_window_indices_np,
)
from .refine_dev_helpers.fourier_window import (
    quantize_current_size as quantize_current_size,
)
from .refine_dev_helpers.types import MeanStats as MeanStats
