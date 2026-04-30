"""RELION InitialModel / VDAM ab-initio refinement.

Implements exact parity with the GUI-generated RELION 5 `relion_refine --grad
--denovo_3dref --pad 1 --auto_sampling` command. See
`docs/math/plan_ab_initio_relion_parity_v3.md` for scope and rollout.

Phase 1 exposes schedules, particle ordering, and denovo initialisation as
pure-Python functions so they can be validated against RELION source via
binding tests in Phase 2.
"""

from .align_symmetry import AlignSymmetrySpec, build_align_symmetry_tokens
from .avg_unaligned import compute_avg_unaligned_and_sigma2
from .dense_adapter import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    class_log_priors_from_state,
    dense_initial_model_expectation_step,
    reference_to_dense_means,
    run_dense_initial_model_estep,
    split_pseudo_halfset_particle_ids,
)
from .e_step import (
    VdamPosterior,
    build_posterior_summary,
    fourier_crop_half,
    hermitian_weights_relion,
    minvsigma2_with_dc_zero,
)
from .init import (
    INI_HIGH_DIGITAL_FREQ,
    compute_current_size_for_denovo,
    compute_ini_high_angstrom,
    compute_ini_high_shell,
    initialise_denovo_state,
    seed_noise_from_mavg,
)
from .layout import bpref_to_run_em_output, relion_bpref_frame_scales, run_em_output_to_bpref
from .schedules import (
    DEFAULT_GRAD_FIN_FRAC,
    DEFAULT_GRAD_INI_FRAC,
    GuiInitialModelDefaults,
    VdamPhaseLengths,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
    default_step_size_for_3d_initial_model,
    default_subset_sizes_for_3d_initial_model,
    default_tau2_fudge_for_3d_initial_model,
)
from .state import (
    MOM2_INIT_CONSTANT,
    InitialModelState,
    half_slot_count,
    half_slot_index,
)
from .subset import (
    assign_pseudo_halfsets,
    pseudo_halfsets_active,
    randomise_particles_order,
    select_vdam_subset,
)

__all__ = [
    "DEFAULT_GRAD_INI_FRAC",
    "DEFAULT_GRAD_FIN_FRAC",
    "GuiInitialModelDefaults",
    "VdamPhaseLengths",
    "compute_phase_lengths",
    "compute_subset_size",
    "compute_stepsize",
    "compute_tau2_fudge",
    "default_subset_sizes_for_3d_initial_model",
    "default_step_size_for_3d_initial_model",
    "default_tau2_fudge_for_3d_initial_model",
    "randomise_particles_order",
    "select_vdam_subset",
    "assign_pseudo_halfsets",
    "pseudo_halfsets_active",
    "InitialModelState",
    "MOM2_INIT_CONSTANT",
    "half_slot_count",
    "half_slot_index",
    "INI_HIGH_DIGITAL_FREQ",
    "compute_current_size_for_denovo",
    "compute_ini_high_angstrom",
    "compute_ini_high_shell",
    "initialise_denovo_state",
    "seed_noise_from_mavg",
    "VdamPosterior",
    "build_posterior_summary",
    "fourier_crop_half",
    "hermitian_weights_relion",
    "minvsigma2_with_dc_zero",
    "AlignSymmetrySpec",
    "build_align_symmetry_tokens",
    "compute_avg_unaligned_and_sigma2",
    "DenseInitialModelEstepConfig",
    "DenseInitialModelEstepResult",
    "class_log_priors_from_state",
    "dense_initial_model_expectation_step",
    "reference_to_dense_means",
    "run_dense_initial_model_estep",
    "split_pseudo_halfset_particle_ids",
    "run_em_output_to_bpref",
    "bpref_to_run_em_output",
    "relion_bpref_frame_scales",
]
