"""cryoSPARC-style branch-and-bound E-step support selection for K=1 EM.

Phase 1 contents:
- ``options.BranchBoundOptions`` — configuration dataclass.
- ``frequency.make_bnb_frequency_schedule`` / ``make_bnb_low_window_spec`` /
  ``make_bnb_high_indices_np`` — L schedule and low/high score-support split.
- ``bounds.compute_high_model_pmax_per_image`` — Suppl Eq 19 (max-CTF-power slice).
- ``bounds.cryosparc_score_upper_correction`` — Suppl Eq 22 score-space upper bound.
- ``bounds.cauchy_score_upper_correction`` — deterministic Cauchy-Schwarz fallback.
- ``diagnostics.BnBDiagnostics`` — per-stage instrumentation.

Subsequent phases will add the pose-tree subdivision, support selector,
``LocalHypothesisLayout`` bridge, and the ``run_bnb_em_k1`` driver.
"""

from .bounds import (
    cauchy_score_upper_correction,
    compute_high_model_pmax_per_image,
    compute_image_high_power_per_image,
    cryosparc_score_upper_correction,
)
from .diagnostics import BnBDiagnostics, BnBStageDiagnostics
from .frequency import (
    fourier_window_spec_from_indices,
    make_bnb_frequency_schedule,
    make_bnb_high_indices_np,
    make_bnb_low_window_spec,
)
from .options import BranchBoundOptions

__all__ = [
    "BnBDiagnostics",
    "BnBStageDiagnostics",
    "BranchBoundOptions",
    "cauchy_score_upper_correction",
    "compute_high_model_pmax_per_image",
    "compute_image_high_power_per_image",
    "cryosparc_score_upper_correction",
    "fourier_window_spec_from_indices",
    "make_bnb_frequency_schedule",
    "make_bnb_high_indices_np",
    "make_bnb_low_window_spec",
]
