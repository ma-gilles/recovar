"""Config dataclasses for ``refine_single_volume``.

These typed structs group the ~90 kwargs that ``refine_single_volume`` and
``_run_relion_iteration_loop`` need.

Each dataclass is ``frozen=True`` so it hashes by value and can be reused
across iterations without copy-on-write surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recovar.em.dense_single_volume.helpers.convergence import LOCAL_SEARCH_HEALPIX_ORDER


@dataclass(frozen=True)
class RefinementSchedule:
    """How long the refinement runs and what grid it starts at."""

    max_iter: int = 10
    init_current_size: int = 32
    fsc_threshold: float = 1.0 / 7.0
    init_healpix_order: int = 2
    max_healpix_order: int = 7
    init_translation_range: float = 10.0
    init_translation_step: float = 2.0
    init_translation_sigma_angstrom: float = 10.0
    particle_diameter_ang: float | None = None
    init_relion_iteration: int = 0
    init_fsc: Any | None = None
    init_ave_Pmax: float | None = None
    init_has_high_fsc_at_limit: bool | None = None
    force_max_iter_after_convergence: bool = False
    skip_final_iteration: bool = False
    init_relion_incr_size: int = 10


@dataclass(frozen=True)
class AdaptiveOptions:
    """Pose-search resolution + adaptive-oversampling knobs."""

    adaptive_oversampling: int = 0
    max_significants: int = 500
    nside_level: int | None = None
    translation_pixel_offset: float | None = None
    relion_current_sizes: tuple[int, ...] | None = None
    relion_healpix_orders: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RelionParityOptions:
    """Knobs that pin RELION numerical behavior."""

    low_resol_join_halves_angstrom: float = 40.0
    tau2_fudge: float = 1.0
    perturb_factor: float = 0.0
    perturb_seed: int | None = None
    perturb_replay_relion_dir: str | None = None
    perturb_replay_relion_prefix: str = "run"
    perturb_replay_precision: Literal["auto", "seed_exact", "star"] = "auto"
    perturb_replay_restart_state_iterations: tuple[int, ...] = ()
    final_sampling_replay_relion_dir: str | None = None
    emulate_relion_firstiter_cc: bool = False
    relion_firstiter_ini_high_angstrom: float | None = None
    do_solvent_fsc_correction: bool = False
    first_iteration_score_mode: str = "gaussian"
    first_iteration_reconstruction_mode: str = "soft"
    image_fourier_backend: Literal["host_numpy", "jax_gpu", "relion_cuda"] = "host_numpy"
    optimizer_random_seed: int | None = None
    use_per_half_mean_variance: bool = False
    preserve_bpref_particle_order: bool = False


@dataclass(frozen=True)
class LocalSearchOptions:
    """Local angular-search controls."""

    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER
    local_search_profile_mode: str = "auto"
    local_search_translation_prior_mode: str = "coarse"


@dataclass(frozen=True)
class ExpectedAccuracyOptions:
    """Half1 oracle inputs for RELION's expected-accuracy calculation."""

    half1_base_order_local: Any | None = None
    half1_trial_order_local: Any | None = None
    half1_optics_group_ids: Any | None = None
    half1_particle_ids: Any | None = None
    half1_ctf_params: Any | None = None
    do_ctf_correction: bool | None = None


@dataclass(frozen=True)
class EngineDebugOptions:
    """Adjoint ablation, intermediate-dump, and test-harness controls."""

    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    save_intermediates_dir: str | None = None
    save_intermediates_skip_unregularized: bool = False
    state_swap_probe: str | None = None
    assert_initial_scoring_state_immutable: bool = False
    stop_after_local_search_profile: bool = False
    stop_after_local_search: bool = False
    stop_after_local_search_score_only: bool = False
    sealed_sampling_state: Any | None = None
    sealed_scoring_context: Any | None = None
    expected_accuracy: ExpectedAccuracyOptions = field(default_factory=ExpectedAccuracyOptions)


@dataclass(frozen=True)
class KClassOptions:
    """K-class refinement controls."""

    n_classes: int = 1
    init_class_log_priors: Any | None = None


@dataclass(frozen=True)
class ReplayState:
    """Per-iteration RELION-replay seed state.

    Mirrors what ``refine_single_volume`` takes as ``init_*`` and
    ``replay_iteration_overrides`` so a downstream replay harness can build
    one struct instead of passing many kwargs.
    """

    init_image_corrections: Any | None = None
    init_scale_corrections: Any | None = None
    init_group_ids: Any | None = None
    init_group_count: Any | None = None
    init_direction_prior: Any | None = None
    init_previous_best_translations: Any | None = None
    init_previous_best_rotation_eulers: Any | None = None
    preserve_initial_direction_prior: bool = False
    replay_iteration_overrides: Any | None = None
    final_replay_override: Any | None = None
    final_replay_reference_maps: Any | None = None
    final_replay_source_iteration: int | None = None
    init_reference_real: Any | None = None
    init_refinement_state_fields: Any | None = None
    init_relion_particle_ids: Any | None = None
    init_relion_optics_group_ids: Any | None = None
    init_relion_optics_group_count: Any | None = None
    relion_scale_follower_count: int = 0
    relion_scale_follower_owners_by_iteration: Any | None = None
    relion_follower_scale_replay: Any | None = None


@dataclass(frozen=True)
class RefinementBatching:
    """Batch sizes the iteration loop hands down to the engines."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000


@dataclass(frozen=True)
class RefinementOptions:
    """Top-level container for all configuration groups.

    Passed as the ``options`` argument of ``refine_single_volume`` and
    ``_run_relion_iteration_loop``.
    """

    schedule: RefinementSchedule = field(default_factory=RefinementSchedule)
    adaptive: AdaptiveOptions = field(default_factory=AdaptiveOptions)
    parity: RelionParityOptions = field(default_factory=RelionParityOptions)
    local_search: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    k_class: KClassOptions = field(default_factory=KClassOptions)
    replay: ReplayState = field(default_factory=ReplayState)
    debug: EngineDebugOptions = field(default_factory=EngineDebugOptions)
    batching: RefinementBatching = field(default_factory=RefinementBatching)
    disc_type: str = "linear_interp"


__all__ = [
    "RefinementSchedule",
    "AdaptiveOptions",
    "RelionParityOptions",
    "LocalSearchOptions",
    "ExpectedAccuracyOptions",
    "EngineDebugOptions",
    "KClassOptions",
    "ReplayState",
    "RefinementBatching",
    "RefinementOptions",
]
