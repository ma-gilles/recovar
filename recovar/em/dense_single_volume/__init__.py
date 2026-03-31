"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.

See docs/math/dense_single_volume_em.md for the algorithm derivation.
"""

from .types import DensePoseGrid, DenseEMPlan, MeanStats
from .plan import plan_em_iteration
from .projection_cache import precompute_projections
from .posterior import compute_posterior
from .accumulate import accumulate_sufficient_statistics
from .solver import solve_mean
