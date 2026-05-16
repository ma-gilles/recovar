"""Pose-marginal PPCA refinement scaffolding for EM."""

from .engine import (
    DenseImageStats,
    DensePPCAFusedBlock,
    DensePPCAFusedEMResult,
    PosteriorDiagnostics,
    dense_pose_ppca_E_step_blocked,
    fused_dense_pose_ppca_block,
    run_dense_ppca_fused_refinement_blocks,
)
from .dense_dataset import (
    DensePPCADatasetBlockInputs,
    combine_halfset_scoring_model,
    iter_dense_ppca_dataset_blocks,
    run_dense_ppca_fused_em_iteration,
    run_dense_ppca_halfset_fused_em_iteration,
)
from .initialization import (
    PPCAInitialization,
    covariance_from_loading_matrix,
    empirical_weighted_covariance,
    initialize_ppca_from_gt_volumes,
    initialize_ppca_from_kclass_volumes,
    real_volume_to_centered_fourier,
    real_volume_to_centered_fourier_half,
)
from .fixture_validation import KClassPPCAFixtureValidation, validate_kclass_to_ppca_initialization
from .local_dataset import (
    iter_local_ppca_dataset_blocks,
    run_local_ppca_fused_em_iteration,
    run_local_ppca_halfset_fused_em_iteration,
)
from .refinement_loop import (
    HalfsetMeanComparison,
    PPCARefinementIterationRecord,
    compare_halfset_means_by_fsc,
    propose_next_current_size,
    run_dense_ppca_refinement_loop,
    run_local_ppca_refinement_loop,
)
from .schedule import (
    HalfsetResolutionGateDecision,
    PPCARefinementScheduleState,
    compute_pose_change_fraction,
    evaluate_halfset_resolution_gate,
    loading_subspace_agreement,
)
from .state import PoseMarginalPPCAEMState

__all__ = [
    "DenseImageStats",
    "DensePPCADatasetBlockInputs",
    "DensePPCAFusedBlock",
    "DensePPCAFusedEMResult",
    "HalfsetResolutionGateDecision",
    "HalfsetMeanComparison",
    "KClassPPCAFixtureValidation",
    "PPCAInitialization",
    "PPCARefinementIterationRecord",
    "PPCARefinementScheduleState",
    "PoseMarginalPPCAEMState",
    "PosteriorDiagnostics",
    "combine_halfset_scoring_model",
    "compare_halfset_means_by_fsc",
    "covariance_from_loading_matrix",
    "dense_pose_ppca_E_step_blocked",
    "fused_dense_pose_ppca_block",
    "empirical_weighted_covariance",
    "evaluate_halfset_resolution_gate",
    "initialize_ppca_from_gt_volumes",
    "initialize_ppca_from_kclass_volumes",
    "iter_local_ppca_dataset_blocks",
    "loading_subspace_agreement",
    "iter_dense_ppca_dataset_blocks",
    "real_volume_to_centered_fourier",
    "real_volume_to_centered_fourier_half",
    "propose_next_current_size",
    "run_dense_ppca_fused_em_iteration",
    "run_dense_ppca_halfset_fused_em_iteration",
    "run_dense_ppca_refinement_loop",
    "run_dense_ppca_fused_refinement_blocks",
    "run_local_ppca_fused_em_iteration",
    "run_local_ppca_halfset_fused_em_iteration",
    "run_local_ppca_refinement_loop",
    "compute_pose_change_fraction",
    "validate_kclass_to_ppca_initialization",
]
