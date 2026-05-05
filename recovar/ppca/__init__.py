"""Pose-marginal PPCA math helpers used by EM refinement."""

from .augmented_mstep import solve_augmented_ppca_mstep
from .pc_prior_config import PCPriorConfig, make_shell_w_prior
from .pose_accumulators import AugmentedPPCAStats
from .pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast
from .triangular import _tri_size, pack_upper_tri, unpack_tri_to_full

__all__ = [
    "AugmentedPPCAStats",
    "PCPriorConfig",
    "_tri_size",
    "pack_upper_tri",
    "unpack_tri_to_full",
    "compute_ppca_pose_scores_and_moments_no_contrast",
    "make_shell_w_prior",
    "solve_augmented_ppca_mstep",
]
