"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.

See docs/math/dense_single_volume_em.md for the algorithm derivation.
"""

from .accumulate import accumulate_sufficient_statistics as accumulate_sufficient_statistics
from .engine import run_dense_em_iteration as run_dense_em_iteration
from .plan import plan_em_iteration as plan_em_iteration
from .posterior import compute_posterior as compute_posterior
from .projection_cache import precompute_projections as precompute_projections
from .solver import solve_mean as solve_mean
from .types import DenseEMPlan as DenseEMPlan
from .types import DensePoseGrid as DensePoseGrid
from .types import MeanStats as MeanStats
