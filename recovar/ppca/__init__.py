"""Public PPCA surface."""

from .augmented_mstep import solve_augmented_ppca_mstep
from .pc_prior_config import PCPriorConfig
from .pose_accumulators import AugmentedPPCAStats
from .ppca import (
    EM,
    E_M_step_batch_half,
    EM_step_half,
    _tri_size,
    batch_unvec,
    batch_vec,
    check_imaginary_part,
    unpack_tri_to_full,
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

__all__ = [
    # Main PPCA functions
    "EM",
    "EM_step_half",
    "E_M_step_batch_half",
    "unpack_tri_to_full",
    "_tri_size",
    # Batch processing utilities
    "batch_vec",
    "batch_unvec",
    "check_imaginary_part",
    # Prior estimation helpers
    "shell_average_real",
    "make_radial_prior_from_shell_total",
    "repair_shell_total_with_mean_sq",
    "make_estimated_prior_from_combined",
    "make_gt_prior_from_variance_total",
    "estimate_gaussian_shell_prior_from_data",
    "estimate_hybrid_shell_prior_from_data",
    # ppca-refine project
    "PCPriorConfig",
    "AugmentedPPCAStats",
    "solve_augmented_ppca_mstep",
]
