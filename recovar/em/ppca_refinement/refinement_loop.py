"""Dense PPCA refinement loop and schedule handoff helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Callable

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_refinement.dense_dataset import run_dense_ppca_halfset_fused_em_iteration
from recovar.em.ppca_refinement.local_dataset import run_local_ppca_halfset_fused_em_iteration
from recovar.em.ppca_refinement.schedule import (
    HalfsetResolutionGateDecision,
    PPCARefinementScheduleState,
    evaluate_halfset_resolution_gate,
    loading_subspace_agreement,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.reconstruction import regularization


@dataclass(frozen=True)
class HalfsetMeanComparison:
    """Gold-standard halfset mean comparison for a proposed PPCA resolution."""

    means_aligned: bool
    resolution_supports: bool
    no_halfset_drift: bool
    fsc: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PPCARefinementIterationRecord:
    """Compact per-iteration PPCA refinement loop record."""

    iteration: int
    current_size: int
    proposed_current_size: int
    resolution_decision: HalfsetResolutionGateDecision
    diagnostics: dict = field(default_factory=dict)


def propose_next_current_size(current_size: int, *, max_current_size: int, growth_factor: float = 2.0) -> int:
    """Conservative even current-size proposal used behind the PPCA gate."""

    current = int(current_size)
    maximum = int(max_current_size)
    if current >= maximum:
        return maximum
    proposed = max(current + 2, int(np.ceil(current * float(growth_factor))))
    if proposed % 2:
        proposed += 1
    return min(proposed, maximum)


def _half_volume_to_full_flat(volume_half, volume_shape):
    return ftu.half_volume_to_full_volume(jnp.asarray(volume_half), tuple(volume_shape)).reshape(-1)


def compare_halfset_means_by_fsc(
    mu_half,
    *,
    volume_shape,
    proposed_current_size: int,
    fsc_threshold: float = 0.143,
    means_aligned: bool = True,
) -> HalfsetMeanComparison:
    """Compare halfset PPCA means by FSC at the proposed current-size shell."""

    full0 = _half_volume_to_full_flat(mu_half[0], volume_shape)
    full1 = _half_volume_to_full_flat(mu_half[1], volume_shape)
    fsc = np.asarray(regularization.get_fsc(full0, full1, tuple(volume_shape)))
    if fsc.size == 0:
        return HalfsetMeanComparison(
            means_aligned=bool(means_aligned),
            resolution_supports=False,
            no_halfset_drift=False,
            fsc=fsc,
            diagnostics={"reason": "empty_fsc"},
        )
    shell = min(max(int(proposed_current_size) // 2 - 1, 0), fsc.size - 1)
    shell_value = float(fsc[shell])
    finite = bool(np.isfinite(shell_value))
    return HalfsetMeanComparison(
        means_aligned=bool(means_aligned),
        resolution_supports=finite and shell_value >= float(fsc_threshold),
        no_halfset_drift=finite,
        fsc=fsc,
        diagnostics={
            "proposed_shell": int(shell),
            "fsc_at_proposed_shell": shell_value,
            "fsc_threshold": float(fsc_threshold),
        },
    )


def _combined_best_pose_ids(pose_diagnostics, n_trans: int) -> np.ndarray:
    best = []
    for key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(key, {})
        if "best_rotation_idx" not in diag or "best_translation_idx" not in diag:
            continue
        rot = np.asarray(diag["best_rotation_idx"], dtype=np.int64)
        trans = np.asarray(diag["best_translation_idx"], dtype=np.int64)
        best.append(rot * int(n_trans) + trans)
    if not best:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(best).astype(np.int32)


def _mean_halfset_diag(pose_diagnostics, name: str) -> float:
    vals = []
    for key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(key, {})
        if name in diag:
            vals.append(float(diag[name]))
    return float(np.mean(vals)) if vals else float("nan")


def _resolve_kclass_allows(
    kclass_schedule_allows,
    iteration: int,
    state: PoseMarginalPPCAEMState,
    *,
    current_size: int,
    proposed_current_size: int,
    halfset_comparison: HalfsetMeanComparison,
) -> bool:
    if callable(kclass_schedule_allows):
        signature = inspect.signature(kclass_schedule_allows)
        parameters = signature.parameters
        supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if supports_kwargs or "current_size" in parameters or "proposed_current_size" in parameters:
            return bool(
                kclass_schedule_allows(
                    iteration,
                    state,
                    current_size=current_size,
                    proposed_current_size=proposed_current_size,
                    halfset_comparison=halfset_comparison,
                )
            )
        return bool(kclass_schedule_allows(iteration, state))
    return bool(kclass_schedule_allows)


def _initial_schedule_state(state: PoseMarginalPPCAEMState, experiment_dataset, init_current_size: int | None):
    if state.schedule_state is not None:
        return state.schedule_state
    current_size = int(init_current_size if init_current_size is not None else experiment_dataset.image_shape[0])
    q = int(jnp.asarray(state.W_score).shape[1]) if jnp.asarray(state.W_score).ndim == 2 else 0
    return PPCARefinementScheduleState(
        current_size=current_size,
        healpix_order=0,
        q=q,
    )


def run_dense_ppca_refinement_loop(
    state: PoseMarginalPPCAEMState,
    experiment_dataset,
    *,
    rotations,
    translations,
    n_iterations: int,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    init_current_size: int | None = None,
    max_current_size: int | None = None,
    kclass_schedule_allows=True,
    halfset_comparator: Callable[[PoseMarginalPPCAEMState, int], HalfsetMeanComparison] | None = None,
    pose_stability_threshold: float = 0.0,
    fsc_threshold: float = 0.143,
    current_size_growth_factor: float = 2.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    mstep_chunk_size: int | None = None,
) -> tuple[PoseMarginalPPCAEMState, list[PPCARefinementIterationRecord]]:
    """Run dense halfset PPCA refinement iterations with gold-standard gating."""

    if int(n_iterations) < 0:
        raise ValueError("n_iterations must be nonnegative")
    max_current_size = int(max_current_size if max_current_size is not None else experiment_dataset.image_shape[0])
    schedule_state = _initial_schedule_state(state, experiment_dataset, init_current_size)
    state = state.replace(schedule_state=schedule_state)
    records: list[PPCARefinementIterationRecord] = []
    n_trans = int(np.asarray(translations).shape[0])

    for iteration in range(int(n_iterations)):
        schedule_state = state.schedule_state
        current_size = int(schedule_state.current_size)
        proposed_size = propose_next_current_size(
            current_size,
            max_current_size=max_current_size,
            growth_factor=current_size_growth_factor,
        )
        updated = run_dense_ppca_halfset_fused_em_iteration(
            state,
            experiment_dataset,
            rotations=rotations,
            translations=translations,
            disc_type=disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            square_window=square_window,
            mstep_chunk_size=mstep_chunk_size,
        )
        best_pose_indices = _combined_best_pose_ids(updated.pose_diagnostics, n_trans)
        comparison = (
            halfset_comparator(updated, proposed_size)
            if halfset_comparator is not None
            else compare_halfset_means_by_fsc(
                updated.mu_half,
                volume_shape=experiment_dataset.volume_shape,
                proposed_current_size=proposed_size,
                fsc_threshold=fsc_threshold,
                means_aligned=True,
            )
        )
        W_agreement = loading_subspace_agreement(
            np.asarray(updated.W_half[0]).T,
            np.asarray(updated.W_half[1]).T,
        )
        kclass_allows = _resolve_kclass_allows(
            kclass_schedule_allows,
            iteration,
            updated,
            current_size=current_size,
            proposed_current_size=proposed_size,
            halfset_comparison=comparison,
        )
        candidate_state = schedule_state.replace(
            previous_best_pose_indices=schedule_state.best_pose_indices,
            best_pose_indices=best_pose_indices,
            halfset_mean_fsc=comparison.fsc,
            halfset_means_aligned=comparison.means_aligned,
            halfset_resolution_supports=comparison.resolution_supports,
            no_halfset_drift=comparison.no_halfset_drift,
            kclass_schedule_allows=kclass_allows and proposed_size > current_size,
            pmax_mean=_mean_halfset_diag(updated.pose_diagnostics, "pmax_mean"),
            logZ_mean=_mean_halfset_diag(updated.pose_diagnostics, "logZ_mean"),
            nsig_mean=_mean_halfset_diag(updated.pose_diagnostics, "nsig_mean"),
            W_subspace_agreement=W_agreement,
        )
        decision = evaluate_halfset_resolution_gate(
            candidate_state,
            pose_stability_threshold=pose_stability_threshold,
        )
        next_size = proposed_size if decision.allow_increase else current_size
        next_schedule = candidate_state.replace(
            current_size=next_size,
            pose_change_fraction=decision.pose_change_fraction,
            diagnostics={
                **candidate_state.diagnostics,
                "iteration": int(iteration),
                "proposed_current_size": int(proposed_size),
                "resolution_increased": bool(decision.allow_increase),
                "halfset_comparison": comparison.diagnostics,
                "gate_reasons": decision.reasons,
            },
        )
        updated = updated.replace(schedule_state=next_schedule)
        records.append(
            PPCARefinementIterationRecord(
                iteration=iteration,
                current_size=next_size,
                proposed_current_size=proposed_size,
                resolution_decision=decision,
                diagnostics=next_schedule.diagnostics,
            )
        )
        state = updated
    return state, records


def run_local_ppca_refinement_loop(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    halfset_local_layouts,
    *,
    n_iterations: int,
    disc_type: str = "linear_interp",
    init_current_size: int | None = None,
    max_current_size: int | None = None,
    kclass_schedule_allows=True,
    halfset_comparator: Callable[[PoseMarginalPPCAEMState, int], HalfsetMeanComparison] | None = None,
    pose_stability_threshold: float = 0.0,
    fsc_threshold: float = 0.143,
    current_size_growth_factor: float = 2.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    mstep_chunk_size: int | None = None,
) -> tuple[PoseMarginalPPCAEMState, list[PPCARefinementIterationRecord]]:
    """Run exact-local halfset PPCA refinement iterations behind the same gate."""

    if len(halfset_datasets) != 2 or len(halfset_local_layouts) != 2:
        raise ValueError("halfset_datasets and halfset_local_layouts must each have length 2")
    reference_dataset = halfset_datasets[0]
    if int(n_iterations) < 0:
        raise ValueError("n_iterations must be nonnegative")
    max_current_size = int(max_current_size if max_current_size is not None else reference_dataset.image_shape[0])
    schedule_state = _initial_schedule_state(state, reference_dataset, init_current_size)
    state = state.replace(schedule_state=schedule_state)
    records: list[PPCARefinementIterationRecord] = []
    n_trans = int(np.asarray(halfset_local_layouts[0].translation_grid).shape[0])

    for iteration in range(int(n_iterations)):
        schedule_state = state.schedule_state
        current_size = int(schedule_state.current_size)
        proposed_size = propose_next_current_size(
            current_size,
            max_current_size=max_current_size,
            growth_factor=current_size_growth_factor,
        )
        updated = run_local_ppca_halfset_fused_em_iteration(
            state,
            halfset_datasets,
            halfset_local_layouts,
            disc_type=disc_type,
            current_size=current_size,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            square_window=square_window,
            mstep_chunk_size=mstep_chunk_size,
        )
        best_pose_indices = _combined_best_pose_ids(updated.pose_diagnostics, n_trans)
        comparison = (
            halfset_comparator(updated, proposed_size)
            if halfset_comparator is not None
            else compare_halfset_means_by_fsc(
                updated.mu_half,
                volume_shape=reference_dataset.volume_shape,
                proposed_current_size=proposed_size,
                fsc_threshold=fsc_threshold,
                means_aligned=True,
            )
        )
        W_agreement = loading_subspace_agreement(
            np.asarray(updated.W_half[0]).T,
            np.asarray(updated.W_half[1]).T,
        )
        kclass_allows = _resolve_kclass_allows(
            kclass_schedule_allows,
            iteration,
            updated,
            current_size=current_size,
            proposed_current_size=proposed_size,
            halfset_comparison=comparison,
        )
        candidate_state = schedule_state.replace(
            previous_best_pose_indices=schedule_state.best_pose_indices,
            best_pose_indices=best_pose_indices,
            halfset_mean_fsc=comparison.fsc,
            halfset_means_aligned=comparison.means_aligned,
            halfset_resolution_supports=comparison.resolution_supports,
            no_halfset_drift=comparison.no_halfset_drift,
            kclass_schedule_allows=kclass_allows and proposed_size > current_size,
            pmax_mean=_mean_halfset_diag(updated.pose_diagnostics, "pmax_mean"),
            logZ_mean=_mean_halfset_diag(updated.pose_diagnostics, "logZ_mean"),
            nsig_mean=_mean_halfset_diag(updated.pose_diagnostics, "nsig_mean"),
            W_subspace_agreement=W_agreement,
        )
        decision = evaluate_halfset_resolution_gate(
            candidate_state,
            pose_stability_threshold=pose_stability_threshold,
        )
        next_size = proposed_size if decision.allow_increase else current_size
        next_schedule = candidate_state.replace(
            current_size=next_size,
            pose_change_fraction=decision.pose_change_fraction,
            diagnostics={
                **candidate_state.diagnostics,
                "iteration": int(iteration),
                "proposed_current_size": int(proposed_size),
                "resolution_increased": bool(decision.allow_increase),
                "halfset_comparison": comparison.diagnostics,
                "gate_reasons": decision.reasons,
                "path": "exact_local",
            },
        )
        updated = updated.replace(schedule_state=next_schedule)
        records.append(
            PPCARefinementIterationRecord(
                iteration=iteration,
                current_size=next_size,
                proposed_current_size=proposed_size,
                resolution_decision=decision,
                diagnostics=next_schedule.diagnostics,
            )
        )
        state = updated
    return state, records
