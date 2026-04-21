"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.
"""

from .types import MeanStats as MeanStats
from .fourier_window import (
    ALLOWED_CURRENT_SIZES as ALLOWED_CURRENT_SIZES,
    make_fourier_window_indices_np as make_fourier_window_indices_np,
    quantize_current_size as quantize_current_size,
)
from .refine import (
    fsc_to_current_size as fsc_to_current_size,
    refine_single_volume as refine_single_volume,
)
from .adaptive import (
    find_significant_mask as find_significant_mask,
    find_significant_rotations as find_significant_rotations,
)
from .engine_v2 import (
    compute_e_step_weights as compute_e_step_weights,
)
from .initial_model_vdam import (
    GUI_INITIALMODEL_DEFAULTS as GUI_INITIALMODEL_DEFAULTS,
    InitialModelRunPlan as InitialModelRunPlan,
    build_initial_model_run_plan as build_initial_model_run_plan,
)
