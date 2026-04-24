"""RELION InitialModel / VDAM ab-initio refinement.

Implements exact parity with the GUI-generated RELION 5 `relion_refine --grad
--denovo_3dref --pad 1 --auto_sampling` command. See
`docs/math/plan_ab_initio_relion_parity_v3.md` for scope and rollout.

Phase 1 exposes schedules, particle ordering, and denovo initialisation as
pure-Python functions so they can be validated against RELION source via
binding tests in Phase 2.
"""

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
]
