"""EM subpackage public API.

This module re-exports commonly used classes and functions from the
EM implementation split across `core.py`, `e_step.py`, `m_step.py`,
`heterogeneity.py`, and `states.py`.
"""

# Classes
from .states import EMState, SGDState, HeterogeneousEMState

# Core utilities and batch orchestration
from .sampling import (
    get_rotation_grid,
    get_translation_grid,
    get_angle_resolution,
    translations_to_indices,
)
from .core import (
    crosscorr_from_ft,
    norm_squared_residuals_from_ft_one_image,
    norm_squared_residuals_from_ft,
    compute_dot_products,
    compute_CTFed_proj_norms,
    probabilities_to_hard_assignment_pose,
    probabilities_to_hard_assignment_idx,
    hard_assignment_idx_to_pose,
)

# E-step
from .e_step import (
    E_with_precompute,
    compute_probability_from_residual_normal_squared_one_image,
    compute_probability_from_residual_normal_squared,
    compute_residuals_many_poses,
    batch_take,
)

# M-step
from .m_step import (
    M_with_precompute,
    backproject_one_image,
    sum_up_translate_one_image,
    sum_up_images_fixed_rots,
)

# Heterogeneity / covariance utilities
from .heterogeneity import (
    compute_UPLambdainvPU,
    compute_little_H_b,
    compute_bHb_terms,
    compute_bLambdainvPU_terms,
    compute_H_B,
    sum_up_images_fixed_rots_covariance_precompute,
    sum_up_images_fixed_rots_covariance_with_precompute,
    compute_projected_covariance,
    compute_projected_covariance_rhs_lhs,
    solve_covariance,
    estimate_principal_components_simple,
    estimate_principal_components_halfset,
)
from .iterations import E_M_batches_2, split_E_M_v2

__all__ = [
    # Classes
    "EMState",
    "SGDState",
    "HeterogeneousEMState",
    # Core
    "translations_to_indices",
    "crosscorr_from_ft",
    "norm_squared_residuals_from_ft_one_image",
    "norm_squared_residuals_from_ft",
    "compute_dot_products",
    "compute_CTFed_proj_norms",
    "batch_take",
    "E_M_batches_2",
    "split_E_M_v2",
    "probabilities_to_hard_assignment_pose",
    "probabilities_to_hard_assignment_idx",
    "hard_assignment_idx_to_pose",
    # E-step
    "E_with_precompute",
    "compute_probability_from_residual_normal_squared_one_image",
    "compute_probability_from_residual_normal_squared",
    "compute_residuals_many_poses",
    # M-step
    "M_with_precompute",
    "backproject_one_image",
    "sum_up_translate_one_image",
    "sum_up_images_fixed_rots",
    # Heterogeneity
    "compute_UPLambdainvPU",
    "compute_little_H_b",
    "compute_bHb_terms",
    "compute_bLambdainvPU_terms",
    "compute_H_B",
    "sum_up_images_fixed_rots_covariance_precompute",
    "sum_up_images_fixed_rots_covariance_with_precompute",
    "compute_projected_covariance",
    "compute_projected_covariance_rhs_lhs",
    "solve_covariance",
    "estimate_principal_components_simple",
    "estimate_principal_components_halfset",
]


