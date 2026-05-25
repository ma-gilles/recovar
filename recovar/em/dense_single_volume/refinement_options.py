"""Config dataclasses for ``refine_single_volume``.

These typed structs group the ~60 kwargs of ``refine_single_volume`` by
concern. Today they sit alongside the existing kwargs surface as
documentation + reusable holders; future callers can populate one of these
instead of memorising the kwarg name set. A follow-up migration step can
switch ``refine_single_volume`` to accept these directly and drop the long
kwarg list.

Each dataclass is ``frozen=True`` so it hashes by value and can be reused
across iterations without copy-on-write surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recovar.em.dense_single_volume.bnb.options import BranchBoundOptions
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


@dataclass(frozen=True)
class AdaptiveOptions:
    """Pose-search resolution + adaptive-oversampling knobs."""

    adaptive_oversampling: int = 0
    max_significants: int = 500
    nside_level: int | None = None
    translation_pixel_offset: float | None = None
    relion_current_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RelionParityOptions:
    """Knobs that pin RELION numerical behavior."""

    low_resol_join_halves_angstrom: float = 40.0
    tau2_fudge: float = 1.0
    perturb_factor: float = 0.0
    perturb_seed: int | None = None
    perturb_replay_relion_dir: str | None = None
    emulate_relion_firstiter_cc: bool = False
    relion_firstiter_ini_high_angstrom: float | None = None
    first_iteration_score_mode: str = "gaussian"
    first_iteration_reconstruction_mode: str = "soft"


@dataclass(frozen=True)
class LocalSearchOptions:
    """Local angular-search controls."""

    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER
    local_search_profile_mode: str = "auto"
    local_search_translation_prior_mode: str = "coarse"


@dataclass(frozen=True)
class EngineDebugOptions:
    """Adjoint ablation + intermediate-dump controls."""

    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    save_intermediates_dir: str | None = None


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
    init_direction_prior: Any | None = None
    init_previous_best_translations: Any | None = None
    init_previous_best_rotation_eulers: Any | None = None
    replay_iteration_overrides: Any | None = None


@dataclass(frozen=True)
class RefinementBatching:
    """Batch sizes the iteration loop hands down to the engines."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000


@dataclass(frozen=True)
class RefinementOptions:
    """Top-level container for all configuration groups.

    Future shape of ``refine_single_volume(experiment_datasets, init_volume,
    init_noise_variance, init_mean_variance, rotations, translations,
    *, schedule, adaptive, parity, local_search, k_class, replay, debug,
    batching, disc_type="linear_interp")``. Today this exists only as a
    typed grouping; the public signature still uses individual kwargs.
    """

    schedule: RefinementSchedule = field(default_factory=RefinementSchedule)
    adaptive: AdaptiveOptions = field(default_factory=AdaptiveOptions)
    parity: RelionParityOptions = field(default_factory=RelionParityOptions)
    local_search: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    k_class: KClassOptions = field(default_factory=KClassOptions)
    replay: ReplayState = field(default_factory=ReplayState)
    debug: EngineDebugOptions = field(default_factory=EngineDebugOptions)
    batching: RefinementBatching = field(default_factory=RefinementBatching)
    bnb: BranchBoundOptions = field(default_factory=BranchBoundOptions)
    refinement_strategy: Literal["relion_dense", "relion_local", "cryosparc_bnb"] = "relion_dense"
    disc_type: str = "linear_interp"


__all__ = [
    "RefinementSchedule",
    "AdaptiveOptions",
    "RelionParityOptions",
    "LocalSearchOptions",
    "EngineDebugOptions",
    "KClassOptions",
    "ReplayState",
    "RefinementBatching",
    "RefinementOptions",
    "BranchBoundOptions",
]
