"""Public PPCA surface.

Combines:
- classical PPCA math (``ppca.py``, ``prior_estimation.py``) used by the
  recovar pipeline's covariance-vs-PPCA path;
- pose-marginal PPCA helpers (``augmented_mstep.py``, ``pose_marginal.py``,
  ``pc_prior_config.py``, ``pose_accumulators.py``, ``triangular.py``) used
  by EM refinement.
"""

from .augmented_mstep import AugmentedPPCAObjective, augmented_ppca_mstep_objective, solve_augmented_ppca_mstep
from .pc_prior_config import PCPriorConfig, make_shell_w_prior
from .pose_accumulators import AugmentedPPCAStats
from .pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast
from .ppca import (
    EM,
    E_M_step_batch_half,
    EM_step_half,
    batch_unvec,
    batch_vec,
    check_imaginary_part,
)
from .prior_estimation import (
    estimate_gaussian_shell_prior_from_data,
    estimate_hybrid_shell_prior_from_data,
    make_estimated_prior_from_combined,
    make_gt_prior_from_variance_total,
    make_radial_prior_from_shell_total,
    repair_shell_total_with_mean_sq,
    shell_average_real,
)
from .triangular import _tri_size, pack_upper_tri, unpack_tri_to_full

__all__ = [
    # Main PPCA functions
    "EM",
    "EM_step_half",
    "E_M_step_batch_half",
    "unpack_tri_to_full",
    "_tri_size",
    "pack_upper_tri",
    # Batch processing utilities
    "batch_vec",
    "batch_unvec",
    "check_imaginary_part",
    # Pose-marginal PPCA (EM refinement)
    "AugmentedPPCAStats",
    "AugmentedPPCAObjective",
    "PCPriorConfig",
    "compute_ppca_pose_scores_and_moments_no_contrast",
    "make_shell_w_prior",
    "augmented_ppca_mstep_objective",
    "solve_augmented_ppca_mstep",
    # Prior estimation helpers
    "shell_average_real",
    "make_radial_prior_from_shell_total",
    "repair_shell_total_with_mean_sq",
    "make_estimated_prior_from_combined",
    "make_gt_prior_from_variance_total",
    "estimate_gaussian_shell_prior_from_data",
    "estimate_hybrid_shell_prior_from_data",
]
