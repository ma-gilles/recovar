"""Per-stage diagnostics for cryoSPARC-style branch-and-bound."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BnBStageDiagnostics:
    """Summary statistics for one BnB stage."""

    stage: int
    L: int
    angular_spacing_deg: float
    shift_spacing_px: float
    n_active_rotations: int
    n_active_shifts: int
    n_active_joint: int
    n_survivors_mean: float
    n_survivors_max: int
    pmax_high_mean: float
    high_correction_mean: float
    cap_applied_count: int
    omitted_mass_upper_mean: float
    omitted_mass_upper_max: float


@dataclass
class BnBDiagnostics:
    """Aggregated diagnostics for one full BnB pass over an image batch."""

    L_schedule: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    angle_spacing_schedule_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
    )
    shift_spacing_schedule_px: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
    )
    stages: list[BnBStageDiagnostics] = field(default_factory=list)

    candidates_initial_mean: float = 0.0
    candidates_final_mean: float = 0.0
    candidates_final_max: int = 0

    fallback_image_count: int = 0
    fallback_reason_counts: dict[str, int] = field(default_factory=dict)

    timing: dict[str, float] = field(default_factory=dict)

    def append_stage(self, summary: BnBStageDiagnostics) -> None:
        self.stages.append(summary)
