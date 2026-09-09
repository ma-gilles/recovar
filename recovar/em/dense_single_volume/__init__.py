"""RELION-style volume refinement with dense, adaptive and exact-local routes.

Import the refinement controller from ``.iteration_loop`` and K-class execution
from ``.k_class``. This package exposes grouped options and sampling/statistics
helpers without loading execution engines. See the algorithm and ownership map
in ``docs/math/relion_refinement_algorithm.md``.
"""

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
from .helpers.types import DenseEMResult as DenseEMResult
from .helpers.types import MeanStats as MeanStats
from .refinement_options import AdaptiveOptions as AdaptiveOptions
from .refinement_options import EngineDebugOptions as EngineDebugOptions
from .refinement_options import ExpectedAccuracyOptions as ExpectedAccuracyOptions
from .refinement_options import KClassOptions as KClassOptions
from .refinement_options import LocalSearchOptions as LocalSearchOptions
from .refinement_options import RefinementBatching as RefinementBatching
from .refinement_options import RefinementOptions as RefinementOptions
from .refinement_options import RefinementSchedule as RefinementSchedule
from .refinement_options import RelionParityOptions as RelionParityOptions
from .refinement_options import ReplayState as ReplayState
