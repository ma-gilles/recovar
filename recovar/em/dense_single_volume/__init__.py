"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.
"""

from .em_engine import (
    compute_e_step_weights as compute_e_step_weights,
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
from .helpers.oversampling import (
    find_significant_mask as find_significant_mask,
)
from .helpers.oversampling import (
    find_significant_rotations as find_significant_rotations,
)
from .helpers.resolution import (
    fsc_to_current_size as fsc_to_current_size,
)
from .helpers.types import MeanStats as MeanStats
from .iteration_loop import (
    refine_single_volume as refine_single_volume,
)
from .k_class import KClassEMResult as KClassEMResult
from .k_class import run_dense_k_class_em as run_dense_k_class_em
from .k_class import run_local_k_class_em as run_local_k_class_em
from .refinement_options import AdaptiveOptions as AdaptiveOptions
from .refinement_options import EngineDebugOptions as EngineDebugOptions
from .refinement_options import KClassOptions as KClassOptions
from .refinement_options import LocalSearchOptions as LocalSearchOptions
from .refinement_options import RefinementBatching as RefinementBatching
from .refinement_options import RefinementOptions as RefinementOptions
from .refinement_options import RefinementSchedule as RefinementSchedule
from .refinement_options import RelionParityOptions as RelionParityOptions
from .refinement_options import ReplayState as ReplayState
