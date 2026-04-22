"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.
"""

from .em_engine import (
    compute_e_step_weights as compute_e_step_weights,
)
from .iteration_loop import (
    fsc_to_current_size as fsc_to_current_size,
)
from .iteration_loop import (
    refine_single_volume as refine_single_volume,
)
from .helpers.oversampling import (
    find_significant_mask as find_significant_mask,
)
from .helpers.oversampling import (
    find_significant_rotations as find_significant_rotations,
)
from .helpers.fourier_window import (
    ALLOWED_CURRENT_SIZES as ALLOWED_CURRENT_SIZES,
)
from .helpers.fourier_window import (
    make_fourier_window_indices_np as make_fourier_window_indices_np,
)
from .helpers.fourier_window import (
    quantize_current_size as quantize_current_size,
)
from .helpers.types import MeanStats as MeanStats
