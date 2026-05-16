"""High-resolution PPCA pose refinement over the K-class pose hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.local_layout import (
    LocalHypothesisLayout,
    build_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
)
from recovar.em.dense_single_volume.ppca_bridge import (
    PPCAKClassScheduleBridge,
    make_ppca_kclass_schedule_bridge,
)
from recovar.em.ppca_refinement.config import (
    GeometryConfig,
    PoseSelectionConfig,
    ScheduleConfig,
    ScoringConfig,
    SparsePass2Config,
)
from recovar.em.ppca_refinement.dense_dataset import (
    coerce_augmented_half_volumes,
    combine_halfset_scoring_model,
    compute_dense_ppca_adaptive_significance,
    run_dense_ppca_halfset_fused_em_iteration,
)
from recovar.em.ppca_refinement.local_dataset import (
    run_local_ppca_halfset_fused_em_iteration,
    run_local_ppca_halfset_pose_scoring_iteration,
)
from recovar.em.ppca_refinement.mean_regularization import MeanRegularizationConfig
from recovar.em.ppca_refinement.postprocess import PostprocessConfig
from recovar.em.ppca_refinement.refinement_loop import (
    HalfsetMeanComparison,
    PPCARefinementIterationRecord,
    _combined_best_pose_ids,
    _mean_halfset_diagnostic,
    _resolve_kclass_allows,
    compare_halfset_means_by_fsc,
    propose_next_current_size,
)
from recovar.em.ppca_refinement.schedule import PPCARefinementScheduleState, evaluate_halfset_resolution_gate
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.em.sampling import (
    build_local_search_grid_metadata,
    get_oversampled_translation_grid,
    get_relion_rotation_grid,
)
from recovar.ppca import PCPriorConfig


@dataclass(frozen=True)
class HighresPPCARefinementResult:
    final_state: PoseMarginalPPCAEMState
    iteration_records: list[PPCARefinementIterationRecord]
    bridge: PPCAKClassScheduleBridge
    diagnostics: dict


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(max_value), max_value, 0.0)
    summed = np.sum(np.exp(np.where(np.isfinite(values), values - safe_max, -np.inf)), axis=axis)
    with np.errstate(divide="ignore"):
        return np.squeeze(safe_max, axis=axis) + np.log(summed)


def _coerce_pipeline_W(W=None, *, eigenvectors=None, eigenvalues=None):
    if W is not None:
        return W
    if eigenvectors is None:
        return None
    eigvec = np.asarray(eigenvectors)
    if eigenvalues is None:
        return eigvec
    eig = np.asarray(eigenvalues, dtype=np.float32).reshape(-1)
    if eigvec.shape[0] == eig.shape[0]:
        return eigvec * np.sqrt(eig)[:, None, None, None]
    if eigvec.ndim == 2 and eigvec.shape[1] == eig.shape[0]:
        return eigvec * np.sqrt(eig)[None, :]
    raise ValueError(f"Cannot align eigenvalues {eig.shape} with eigenvectors {eigvec.shape}")


def initialize_state_from_pipeline_ppca(
    mean,
    *,
    W=None,
    eigenvectors=None,
    eigenvalues=None,
    mean_prior,
    W_prior,
    noise_variance,
    volume_shape,
    q: int | None = None,
    volume_domain: str = "auto",
    schedule_state: PPCARefinementScheduleState | None = None,
    pc_prior_config: PCPriorConfig | None = None,
) -> PoseMarginalPPCAEMState:
    """Create a halfset-aware PPCA refinement state from pipeline PPCA arrays."""

    W_input = _coerce_pipeline_W(W, eigenvectors=eigenvectors, eigenvalues=eigenvalues)
    augmented_half, q_resolved = coerce_augmented_half_volumes(
        mean,
        W_input,
        volume_shape=volume_shape,
        q=q,
        volume_domain=volume_domain,
    )
    mu = augmented_half[0]
    W_half = (
        jnp.swapaxes(augmented_half[1:], 0, 1)
        if q_resolved
        else jnp.zeros((mu.shape[0], 0), dtype=mu.dtype)
    )
    W_prior_arr = jnp.asarray(W_prior)
    if W_prior_arr.shape != (mu.shape[0], q_resolved):
        raise ValueError(f"W_prior shape {W_prior_arr.shape} != ({mu.shape[0]}, {q_resolved})")
    mean_prior_arr = jnp.asarray(mean_prior)
    if mean_prior_arr.shape != mu.shape:
        raise ValueError(f"mean_prior shape {mean_prior_arr.shape} != {mu.shape}")
    return PoseMarginalPPCAEMState(
        mu_half=(mu, mu),
        W_half=(W_half, W_half),
        mu_score=mu,
        W_score=W_half,
        W_prior=W_prior_arr,
        mean_prior=mean_prior_arr,
        noise_variance=jnp.asarray(noise_variance),
        z_prior_precision_diag=jnp.ones((q_resolved,), dtype=jnp.float32),
        schedule_state=schedule_state,
        pc_prior_config=pc_prior_config if pc_prior_config is not None else PCPriorConfig(),
    )


def _initial_schedule_state(state: PoseMarginalPPCAEMState, dataset, init_current_size: int | None):
    if state.schedule_state is not None:
        return state.schedule_state
    current_size = int(init_current_size if init_current_size is not None else dataset.image_shape[0])
    q = int(jnp.asarray(state.W_score).shape[1]) if jnp.asarray(state.W_score).ndim == 2 else 0
    return PPCARefinementScheduleState(current_size=current_size, healpix_order=0, q=q)


def _gate_ppca_iteration(
    state: PoseMarginalPPCAEMState,
    *,
    reference_dataset,
    bridge: PPCAKClassScheduleBridge,
    iteration: int,
    current_size: int,
    max_current_size: int,
    translations,
    halfset_comparator: Callable[[PoseMarginalPPCAEMState, int], HalfsetMeanComparison] | None,
    fsc_threshold: float,
    current_size_growth_factor: float,
    pose_stability_threshold: float,
) -> tuple[PoseMarginalPPCAEMState, PPCARefinementIterationRecord]:
    proposed_size = propose_next_current_size(
        current_size,
        max_current_size=max_current_size,
        growth_factor=current_size_growth_factor,
    )
    comparison = (
        halfset_comparator(state, proposed_size)
        if halfset_comparator is not None
        else compare_halfset_means_by_fsc(
            state.mu_half,
            volume_shape=reference_dataset.volume_shape,
            proposed_current_size=proposed_size,
            fsc_threshold=fsc_threshold,
            means_aligned=True,
        )
    )
    schedule_state = state.schedule_state
    best_pose_indices = _combined_best_pose_ids(state.pose_diagnostics, int(np.asarray(translations).shape[0]))
    kclass_allows = _resolve_kclass_allows(
        bridge,
        iteration,
        state,
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
        pmax_mean=_mean_halfset_diagnostic(state.pose_diagnostics, "pmax_mean"),
        logZ_mean=_mean_halfset_diagnostic(state.pose_diagnostics, "logZ_mean"),
        nsig_mean=_mean_halfset_diagnostic(state.pose_diagnostics, "nsig_mean"),
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
            "bridge_do_local_search": bool(bridge.state.do_local_search),
            "bridge_healpix_order": int(bridge.state.healpix_order),
        },
    )
    state = state.replace(schedule_state=next_schedule)
    return state, PPCARefinementIterationRecord(
        iteration=int(iteration),
        current_size=int(next_size),
        proposed_current_size=int(proposed_size),
        resolution_decision=decision,
        diagnostics=next_schedule.diagnostics,
    )


def _diagnostic_top_centers(diag: dict) -> tuple[np.ndarray, np.ndarray]:
    rot_key = "top_rotation_id" if "top_rotation_id" in diag else "top_rotation_idx"
    if rot_key not in diag or "top_translation_idx" not in diag:
        if "best_rotation_id" in diag:
            rot = np.asarray(diag["best_rotation_id"], dtype=np.int32)[:, None]
        else:
            rot = np.asarray(diag["best_rotation_idx"], dtype=np.int32)[:, None]
        trans = np.asarray(diag["best_translation_idx"], dtype=np.int32)[:, None]
    else:
        rot = np.asarray(diag[rot_key], dtype=np.int32)
        trans = np.asarray(diag["top_translation_idx"], dtype=np.int32)
    image_indices = diag.get("image_indices")
    if image_indices is not None:
        image_indices = np.asarray(image_indices, dtype=np.int64).reshape(-1)
        if image_indices.shape[0] == rot.shape[0]:
            order = np.argsort(image_indices)
            rot = rot[order]
            trans = trans[order]
    return rot, trans


_POSE_DIAGNOSTIC_ROW_KEYS = frozenset(
    {
        "max_posterior_per_image",
        "n_significant_per_image",
        "best_rotation_idx",
        "best_rotation_id",
        "best_translation_idx",
        "best_rotation_matrix",
        "best_translation",
        "top_rotation_idx",
        "top_rotation_id",
        "top_translation_idx",
        "top_log_score",
        "top_log_score_per_image",
        "top_posterior",
        "top_posterior_per_image",
        "best_log_score_per_image",
        "local_mstep_retained_mass_per_image",
        "image_indices",
    }
)


def _sort_halfset_diagnostic_by_image_index(diag: dict) -> dict:
    if not diag or "image_indices" not in diag:
        return dict(diag)
    image_indices = np.asarray(diag["image_indices"], dtype=np.int64).reshape(-1)
    if image_indices.size == 0:
        return dict(diag)
    order = np.argsort(image_indices, kind="stable")
    out = dict(diag)
    for key in _POSE_DIAGNOSTIC_ROW_KEYS:
        if key not in out:
            continue
        value = out[key]
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == image_indices.shape[0]:
            out[key] = arr[order]
    return out


def _canonicalize_pose_diagnostics_by_image_index(pose_diagnostics: dict) -> dict:
    if not pose_diagnostics:
        return {}
    out = dict(pose_diagnostics)
    for half_key in ("halfset0", "halfset1"):
        if half_key in out:
            out[half_key] = _sort_halfset_diagnostic_by_image_index(dict(out[half_key]))
    return out


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    cos_angle = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _pose_probe_delta_diagnostics(
    reference_diagnostics: dict,
    candidate_diagnostics: dict,
    *,
    reference_rotation_grid,
    candidate_rotation_grid,
    reference_translations,
    candidate_translations,
    prefix: str,
) -> dict:
    """Compare top-1 pose diagnostics before/after a score-only probe."""

    ref_rots_grid = np.asarray(reference_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    cand_rots_grid = np.asarray(candidate_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    ref_trans_grid = np.asarray(reference_translations, dtype=np.float32).reshape(-1, 2)
    cand_trans_grid = np.asarray(candidate_translations, dtype=np.float32).reshape(-1, 2)
    all_changed = []
    all_angles = []
    all_offsets = []
    out: dict = {}
    for half_key in ("halfset0", "halfset1"):
        if half_key not in reference_diagnostics or half_key not in candidate_diagnostics:
            continue
        ref_diag = reference_diagnostics.get(half_key, {})
        cand_diag = candidate_diagnostics.get(half_key, {})
        ref_rot, ref_trans = _diagnostic_top_centers(ref_diag)
        cand_rot, cand_trans = _diagnostic_top_centers(cand_diag)
        if ref_rot.shape[0] == 0 or cand_rot.shape[0] == 0:
            continue
        ref_labels = np.asarray(ref_diag.get("image_indices", np.arange(ref_rot.shape[0])), dtype=np.int64)
        cand_labels = np.asarray(cand_diag.get("image_indices", np.arange(cand_rot.shape[0])), dtype=np.int64)
        common, ref_order, cand_order = np.intersect1d(ref_labels, cand_labels, return_indices=True)
        if common.size == 0:
            continue
        ref_rot = np.asarray(ref_rot[ref_order, 0], dtype=np.int64)
        cand_rot = np.asarray(cand_rot[cand_order, 0], dtype=np.int64)
        ref_trans = np.asarray(ref_trans[ref_order, 0], dtype=np.int64)
        cand_trans = np.asarray(cand_trans[cand_order, 0], dtype=np.int64)
        valid = (
            (ref_rot >= 0)
            & (cand_rot >= 0)
            & (ref_rot < ref_rots_grid.shape[0])
            & (cand_rot < cand_rots_grid.shape[0])
            & (ref_trans >= 0)
            & (cand_trans >= 0)
            & (ref_trans < ref_trans_grid.shape[0])
            & (cand_trans < cand_trans_grid.shape[0])
        )
        if not np.any(valid):
            continue
        changed = ((ref_rot != cand_rot) | (ref_trans != cand_trans))[valid]
        ref_mats = ref_diag.get("best_rotation_matrix")
        cand_mats = cand_diag.get("best_rotation_matrix")
        if ref_mats is not None and cand_mats is not None:
            ref_mats = np.asarray(ref_mats, dtype=np.float32)[ref_order][valid]
            cand_mats = np.asarray(cand_mats, dtype=np.float32)[cand_order][valid]
            angles = np.asarray(
                [_rotation_angle_deg(r0, r1) for r0, r1 in zip(ref_mats, cand_mats, strict=True)],
                dtype=np.float32,
            )
        else:
            angles = np.asarray(
                [
                    _rotation_angle_deg(ref_rots_grid[int(r0)], cand_rots_grid[int(r1)])
                    for r0, r1 in zip(ref_rot[valid], cand_rot[valid], strict=True)
                ],
                dtype=np.float32,
            )
        ref_translation_values = ref_diag.get("best_translation")
        cand_translation_values = cand_diag.get("best_translation")
        if ref_translation_values is not None and cand_translation_values is not None:
            ref_t = np.asarray(ref_translation_values, dtype=np.float32)[ref_order][valid]
            cand_t = np.asarray(cand_translation_values, dtype=np.float32)[cand_order][valid]
            offsets = np.linalg.norm(cand_t - ref_t, axis=1).astype(np.float32)
        else:
            offsets = np.linalg.norm(
                cand_trans_grid[cand_trans[valid]] - ref_trans_grid[ref_trans[valid]],
                axis=1,
            ).astype(np.float32)
        all_changed.append(changed.astype(np.float32))
        all_angles.append(angles)
        all_offsets.append(offsets)
        out[f"{prefix}_{half_key}_top1_changed_fraction"] = float(np.mean(changed))
        out[f"{prefix}_{half_key}_top1_angle_deg_mean"] = float(np.mean(angles))
        out[f"{prefix}_{half_key}_top1_angle_deg_max"] = float(np.max(angles))
        out[f"{prefix}_{half_key}_top1_translation_px_mean"] = float(np.mean(offsets))
        out[f"{prefix}_{half_key}_top1_translation_px_max"] = float(np.max(offsets))
        out[f"{prefix}_{half_key}_top1_angle_deg"] = angles
        out[f"{prefix}_{half_key}_top1_translation_px"] = offsets
    if all_changed:
        changed_cat = np.concatenate(all_changed)
        angles_cat = np.concatenate(all_angles)
        offsets_cat = np.concatenate(all_offsets)
        out[f"{prefix}_top1_changed_fraction"] = float(np.mean(changed_cat))
        out[f"{prefix}_top1_angle_deg_mean"] = float(np.mean(angles_cat))
        out[f"{prefix}_top1_angle_deg_p95"] = float(np.percentile(angles_cat, 95))
        out[f"{prefix}_top1_angle_deg_max"] = float(np.max(angles_cat))
        out[f"{prefix}_top1_translation_px_mean"] = float(np.mean(offsets_cat))
        out[f"{prefix}_top1_translation_px_p95"] = float(np.percentile(offsets_cat, 95))
        out[f"{prefix}_top1_translation_px_max"] = float(np.max(offsets_cat))
    return out


def _pose_diagnostic_summary(pose_diagnostics: dict) -> dict:
    summary = {}
    for half_key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(half_key, {})
        for key in ("pmax_mean", "logZ_mean", "nsig_mean"):
            if key in diag:
                summary[f"{half_key}_{key}"] = float(diag[key])
    return summary


def build_top_p_local_hypothesis_layout(
    top_rotation_ids,
    top_translation_idx,
    *,
    center_rotation_grid,
    top_rotation_matrices=None,
    center_translation_grid=None,
    target_rotation_grid,
    healpix_order: int,
    translations,
    sigma_rot: float,
    sigma_psi: float,
    sigma_offset_angstrom: float,
    voxel_size: float,
    grid_metadata=None,
) -> LocalHypothesisLayout:
    """Build exact-local support as the union of top-p pose neighborhoods."""

    top_rotation_ids = np.asarray(top_rotation_ids, dtype=np.int64)
    top_translation_idx = np.asarray(top_translation_idx, dtype=np.int64)
    if top_rotation_ids.shape != top_translation_idx.shape:
        raise ValueError(f"top pose shapes differ: {top_rotation_ids.shape} vs {top_translation_idx.shape}")
    if top_rotation_ids.ndim == 1:
        top_rotation_ids = top_rotation_ids[:, None]
        top_translation_idx = top_translation_idx[:, None]
    top_mats = None
    if top_rotation_matrices is not None:
        top_mats = np.asarray(top_rotation_matrices, dtype=np.float32)
        if top_mats.ndim == 3:
            top_mats = top_mats[:, None, :, :]
        if top_mats.shape[:2] != top_rotation_ids.shape or top_mats.shape[-2:] != (3, 3):
            raise ValueError(
                "top_rotation_matrices must have shape "
                f"{top_rotation_ids.shape} + (3, 3), got {top_mats.shape}"
            )
    translations_np = np.asarray(translations, dtype=np.float32)
    center_translations_np = (
        translations_np
        if center_translation_grid is None
        else np.asarray(center_translation_grid, dtype=np.float32).reshape(-1, translations_np.shape[1])
    )
    center_grid = np.asarray(center_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    target_grid = np.asarray(target_rotation_grid, dtype=np.float32).reshape(-1, 3, 3)
    metadata = build_local_search_grid_metadata(int(healpix_order)) if grid_metadata is None else grid_metadata

    offsets = np.zeros(top_rotation_ids.shape[0] + 1, dtype=np.int64)
    counts = np.zeros(top_rotation_ids.shape[0], dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    rotation_prior_parts: list[np.ndarray] = []
    translation_prior_rows: list[np.ndarray] = []

    for image_idx in range(top_rotation_ids.shape[0]):
        valid = (top_rotation_ids[image_idx] >= 0) & (top_translation_idx[image_idx] >= 0)
        if not np.any(valid):
            valid = np.zeros_like(top_rotation_ids[image_idx], dtype=bool)
            valid[0] = True
            top_rotation_ids[image_idx, 0] = 0
            top_translation_idx[image_idx, 0] = 0
        center_ids = top_rotation_ids[image_idx, valid]
        center_trans = top_translation_idx[image_idx, valid]
        if top_mats is None:
            if int(np.max(center_ids, initial=0)) >= center_grid.shape[0]:
                raise ValueError("top_rotation_ids exceed center_rotation_grid size")
            center_mats = center_grid[center_ids]
        else:
            center_mats = top_mats[image_idx, valid]
        if int(np.max(center_trans, initial=0)) >= center_translations_np.shape[0]:
            raise ValueError("top_translation_idx exceed center_translation_grid size")
        center_translations = center_translations_np[center_trans]
        local = build_local_hypothesis_layout(
            center_mats,
            target_grid,
            float(sigma_rot),
            float(sigma_psi),
            int(healpix_order),
            translations_np,
            center_translations,
            float(sigma_offset_angstrom),
            None,
            float(voxel_size),
            grid_metadata=metadata,
        )
        unique_ids, first_idx, inverse = np.unique(
            np.asarray(local.rotation_ids_flat, dtype=np.int32),
            return_index=True,
            return_inverse=True,
        )
        priors = np.full(unique_ids.shape[0], -np.inf, dtype=np.float64)
        for local_idx, inverse_idx in enumerate(inverse):
            priors[inverse_idx] = np.logaddexp(priors[inverse_idx], float(local.rotation_log_priors_flat[local_idx]))
        priors = (priors - np.log(max(1, center_ids.size))).astype(np.float32)
        rotations_selected = np.asarray(local.rotations_flat, dtype=np.float32)[first_idx]
        trans_prior = _logsumexp_np(np.asarray(local.translation_log_priors, dtype=np.float32), axis=0)
        trans_prior = (trans_prior - np.log(max(1, center_ids.size))).astype(np.float32)

        counts[image_idx] = int(unique_ids.shape[0])
        offsets[image_idx + 1] = offsets[image_idx] + int(unique_ids.shape[0])
        rotation_ids_parts.append(unique_ids.astype(np.int32, copy=False))
        rotations_parts.append(rotations_selected.astype(np.float32, copy=False))
        rotation_prior_parts.append(priors)
        translation_prior_rows.append(trans_prior)

    return LocalHypothesisLayout(
        n_global_rotations=int(target_grid.shape[0]),
        n_pixels=int(metadata["n_pixels"]),
        n_psi=int(metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=np.concatenate(rotation_ids_parts) if rotation_ids_parts else np.zeros(0, dtype=np.int32),
        rotations_flat=(
            np.concatenate(rotations_parts, axis=0) if rotations_parts else np.zeros((0, 3, 3), dtype=np.float32)
        ),
        rotation_log_priors_flat=(
            np.concatenate(rotation_prior_parts) if rotation_prior_parts else np.zeros(0, dtype=np.float32)
        ),
        rotation_counts=counts,
        translation_grid=translations_np,
        translation_log_priors=(
            np.stack(translation_prior_rows, axis=0)
            if translation_prior_rows
            else np.zeros((0, translations_np.shape[0]), dtype=np.float32)
        ),
    )


def build_top_p_local_hypothesis_layouts_from_diagnostics(
    pose_diagnostics: dict,
    *,
    center_rotation_grid,
    target_rotation_grid,
    healpix_order: int,
    translations,
    sigma_rot: float,
    sigma_psi: float,
    sigma_offset_angstrom: float,
    voxel_size: float,
) -> tuple[LocalHypothesisLayout, LocalHypothesisLayout]:
    layouts = []
    metadata = build_local_search_grid_metadata(int(healpix_order))
    for half_key in ("halfset0", "halfset1"):
        half_diag = pose_diagnostics.get(half_key, {})
        rot_ids, trans_ids = _diagnostic_top_centers(half_diag)
        top_mats = half_diag.get("top_rotation_matrix")
        layouts.append(
            build_top_p_local_hypothesis_layout(
                rot_ids,
                trans_ids,
                center_rotation_grid=center_rotation_grid,
                top_rotation_matrices=top_mats,
                center_translation_grid=translations,
                target_rotation_grid=target_rotation_grid,
                healpix_order=healpix_order,
                translations=translations,
                sigma_rot=sigma_rot,
                sigma_psi=sigma_psi,
                sigma_offset_angstrom=sigma_offset_angstrom,
                voxel_size=voxel_size,
                grid_metadata=metadata,
            )
        )
    return tuple(layouts)


def run_adaptive_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    *,
    coarse_rotations,
    coarse_translations,
    nside_level: int,
    adaptive_oversampling: int,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    pose_selection: PoseSelectionConfig | None = None,
    sparse_pass2: SparsePass2Config | None = None,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    translation_step: float | None = None,
    disc_type: str = "linear_interp",
) -> PoseMarginalPPCAEMState:
    """Run PPCA adaptive pass-1 significance followed by exact fine support."""

    if len(halfset_datasets) != 2:
        raise ValueError("halfset_datasets must have length 2")
    geometry = geometry if geometry is not None else GeometryConfig(volume_domain="fourier_half")
    coarse_geometry = GeometryConfig(
        current_size=geometry.current_size,
        q=geometry.q,
        volume_domain=geometry.volume_domain,
    )
    schedule = schedule if schedule is not None else ScheduleConfig()
    scoring = scoring if scoring is not None else ScoringConfig()
    sparse_pass2 = sparse_pass2 if sparse_pass2 is not None else SparsePass2Config(enabled=False)
    pose_selection = pose_selection if pose_selection is not None else PoseSelectionConfig()
    significance = []
    layouts = []
    for half_dataset in halfset_datasets:
        sig = compute_dense_ppca_adaptive_significance(
            half_dataset,
            state.mu_score,
            state.W_score,
            noise_variance=state.noise_variance,
            rotations=coarse_rotations,
            translations=coarse_translations,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            geometry=coarse_geometry,
            schedule=schedule,
            scoring=scoring,
            pose_selection=pose_selection,
            disc_type=disc_type,
        )
        significance.append(sig)
        layouts.append(
            build_pass2_hypothesis_layout(
                sig.significant_sample_indices,
                int(np.asarray(coarse_rotations).shape[0]),
                int(np.asarray(coarse_translations).shape[0]),
                int(nside_level),
                np.asarray(coarse_translations, dtype=np.float32),
                oversampling_order=int(adaptive_oversampling),
                translation_step=translation_step,
                allow_empty=False,
            )
        )
    updated = run_local_ppca_halfset_fused_em_iteration(
        state,
        halfset_datasets,
        tuple(layouts),
        geometry=geometry,
        schedule=ScheduleConfig(
            image_batch_size=min(2, int(schedule.image_batch_size)),
            rotation_block_size=min(512, int(schedule.rotation_block_size)),
            mstep_chunk_size=schedule.mstep_chunk_size,
            local_image_shard_count=int(schedule.local_image_shard_count),
        ),
        scoring=scoring,
        sparse_pass2=sparse_pass2,
        mean_reg=mean_reg,
        postprocess=postprocess,
        pose_selection=pose_selection,
        disc_type=disc_type,
    )
    diagnostics = _canonicalize_pose_diagnostics_by_image_index(dict(updated.pose_diagnostics))
    for half_idx, half_key in enumerate(("halfset0", "halfset1")):
        half_diag = dict(diagnostics.get(half_key, {}))
        half_diag.update(
            {
                "path": "adaptive",
                "adaptive_oversampling": int(adaptive_oversampling),
                "coarse_healpix_order": int(nside_level),
                "adaptive_fraction": float(adaptive_fraction),
                "adaptive_max_significants": int(max_significants),
                "adaptive_nsig_mean": float(significance[half_idx].diagnostics.get("nsig_mean", float("nan"))),
                "adaptive_pmax_mean": float(significance[half_idx].diagnostics.get("pmax_mean", float("nan"))),
                "adaptive_fine_rotation_count_mean": float(np.mean(layouts[half_idx].rotation_counts)),
                "adaptive_fine_translation_count": int(layouts[half_idx].translation_grid.shape[0]),
            }
        )
        diagnostics[half_key] = half_diag
    return updated.replace(pose_diagnostics=diagnostics)


def run_highres_ppca_refinement_with_kclass_pose_hierarchy(
    state: PoseMarginalPPCAEMState,
    experiment_dataset,
    *,
    rotations=None,
    translations,
    n_pose_warmup_iterations: int = 0,
    n_dense_iterations: int = 1,
    n_adaptive_iterations: int = 0,
    n_local_iterations: int = 0,
    init_current_size: int | None = None,
    max_current_size: int | None = None,
    init_healpix_order: int = 2,
    max_healpix_order: int = 7,
    auto_local_healpix_order: int = 5,
    adaptive_oversampling: int = 0,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    max_dense_pose_candidates: int = 2_000_000,
    top_p_poses: int = 1,
    pose_selection: PoseSelectionConfig | None = None,
    pose_warmup_dense_current_size: int | None = None,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    mstep_chunk_size: int | None = None,
    local_image_shard_count: int = 1,
    local_mstep_top_k: int = 1,
    local_mstep_min_pmax: float = 0.999,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    image_scale_corrections: np.ndarray | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    halfset_comparator: Callable[[PoseMarginalPPCAEMState, int], HalfsetMeanComparison] | None = None,
    pose_stability_threshold: float = 0.0,
    fsc_threshold: float = 0.143,
    current_size_growth_factor: float = 2.0,
    bridge: PPCAKClassScheduleBridge | None = None,
) -> HighresPPCARefinementResult:
    """Run high-resolution PPCA pose probing and dense/adaptive/local refinement.

    ``n_pose_warmup_iterations`` runs score-only dense -> exact-local pose
        selection first. That stage updates only ``pose_diagnostics`` so the first
        updating local EM iteration can be bounded around PPCA-selected top-p poses.
    """

    if min(
        int(n_pose_warmup_iterations),
        int(n_dense_iterations),
        int(n_adaptive_iterations),
        int(n_local_iterations),
    ) < 0:
        raise ValueError("iteration counts must be nonnegative")
    pose_selection = pose_selection if pose_selection is not None else PoseSelectionConfig(top_p_poses=top_p_poses)
    translations_np = np.asarray(translations, dtype=np.float32)
    reference_dataset = experiment_dataset.get_halfset(0) if hasattr(experiment_dataset, "get_halfset") else experiment_dataset
    max_current_size = int(max_current_size if max_current_size is not None else reference_dataset.image_shape[0])
    state = state.replace(schedule_state=_initial_schedule_state(state, reference_dataset, init_current_size))
    if bridge is None:
        bridge = make_ppca_kclass_schedule_bridge(
            n_rotations=int(np.asarray(rotations).shape[0]) if rotations is not None else int(get_relion_rotation_grid(init_healpix_order).shape[0]),
            translations=translations_np,
            grid_size=int(reference_dataset.image_shape[0]),
            voxel_size_angstrom=float(getattr(reference_dataset, "voxel_size", 1.0)),
            init_healpix_order=int(init_healpix_order),
            max_healpix_order=int(max_healpix_order),
            auto_local_healpix_order=int(auto_local_healpix_order),
            adaptive_oversampling=int(adaptive_oversampling),
        )
    schedule = ScheduleConfig(
        image_batch_size=int(image_batch_size),
        rotation_block_size=int(rotation_block_size),
        mstep_chunk_size=mstep_chunk_size,
        local_image_shard_count=int(local_image_shard_count),
    )
    scoring = ScoringConfig(
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        image_scale_corrections=image_scale_corrections,
    )
    local_sparse_pass2 = SparsePass2Config(
        enabled=int(local_mstep_top_k) > 0,
        local_mstep_top_k=int(local_mstep_top_k),
        local_mstep_min_pmax=float(local_mstep_min_pmax),
    )
    halfsets = (experiment_dataset.get_halfset(0), experiment_dataset.get_halfset(1))
    records: list[PPCARefinementIterationRecord] = []
    stage_history: list[dict] = []
    center_rotation_grid = np.asarray(rotations if rotations is not None else get_relion_rotation_grid(init_healpix_order), dtype=np.float32)
    center_translation_grid = translations_np

    def _rotation_grid_for_bridge() -> np.ndarray:
        if rotations is not None and int(bridge.state.healpix_order) <= int(init_healpix_order):
            return np.asarray(rotations, dtype=np.float32)
        return np.asarray(get_relion_rotation_grid(int(bridge.state.healpix_order)), dtype=np.float32)

    def _gate(stage: str, iteration: int) -> None:
        nonlocal state
        bridge.n_translations = int(center_translation_grid.shape[0])
        bridge.translations = np.asarray(center_translation_grid, dtype=np.float32)
        current = int(state.schedule_state.current_size)
        state, record = _gate_ppca_iteration(
            state,
            reference_dataset=reference_dataset,
            bridge=bridge,
            iteration=iteration,
            current_size=current,
            max_current_size=max_current_size,
            translations=center_translation_grid,
            halfset_comparator=halfset_comparator,
            fsc_threshold=fsc_threshold,
            current_size_growth_factor=current_size_growth_factor,
            pose_stability_threshold=pose_stability_threshold,
        )
        records.append(record)
        stage_history.append(
            {
                "iteration": int(iteration),
                "stage": stage,
                "current_size": int(state.schedule_state.current_size),
                "healpix_order": int(bridge.state.healpix_order),
                "do_local_search": bool(bridge.state.do_local_search),
                "top_p_poses": int(pose_selection.top_p_poses),
            }
        )

    global_iter = 0
    for _ in range(int(n_pose_warmup_iterations)):
        dense_rotations = _rotation_grid_for_bridge()
        dense_pose_count = int(dense_rotations.shape[0]) * int(translations_np.shape[0])
        if dense_pose_count > int(max_dense_pose_candidates):
            stage_history.append(
                {
                    "iteration": int(global_iter),
                    "stage": "pose_warmup_dense_skipped_grid_cap",
                    "dense_pose_count": int(dense_pose_count),
                    "max_dense_pose_candidates": int(max_dense_pose_candidates),
                }
            )
            break

        previous_pose_diagnostics = dict(state.pose_diagnostics or {})
        warmup_dense_diags = {}
        dense_probe_current_size = (
            int(pose_warmup_dense_current_size)
            if pose_warmup_dense_current_size is not None
            else int(state.schedule_state.current_size)
        )
        for half_idx, half_dataset in enumerate(halfsets):
            sig = compute_dense_ppca_adaptive_significance(
                half_dataset,
                state.mu_score,
                state.W_score,
                noise_variance=state.noise_variance,
                rotations=dense_rotations,
                translations=translations_np,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                geometry=GeometryConfig(current_size=dense_probe_current_size, volume_domain="fourier_half"),
                schedule=schedule,
                scoring=scoring,
                pose_selection=pose_selection,
                disc_type=disc_type,
            )
            half_diag = dict(sig.diagnostics)
            dense_best_rot = np.asarray(half_diag.get("best_rotation_id", half_diag["best_rotation_idx"]), dtype=np.int64)
            dense_best_trans = np.asarray(half_diag["best_translation_idx"], dtype=np.int64)
            half_diag.update(
                {
                    "path": "pose_warmup_dense",
                    "pose_score_only": True,
                    "dense_pose_count": int(dense_pose_count),
                    "dense_probe_current_size": int(dense_probe_current_size),
                    "image_indices": np.arange(dense_best_rot.shape[0], dtype=np.int32),
                    "best_rotation_matrix": dense_rotations[dense_best_rot],
                    "best_translation": translations_np[dense_best_trans],
                }
            )
            warmup_dense_diags[f"halfset{half_idx}"] = half_diag
        dense_probe_delta = (
            _pose_probe_delta_diagnostics(
                previous_pose_diagnostics,
                warmup_dense_diags,
                reference_rotation_grid=center_rotation_grid,
                candidate_rotation_grid=dense_rotations,
                reference_translations=center_translation_grid,
                candidate_translations=translations_np,
                prefix="pose_probe_dense_vs_input",
            )
            if previous_pose_diagnostics
            else {}
        )
        state = state.replace(
            pose_diagnostics=_canonicalize_pose_diagnostics_by_image_index(
                {
                    **warmup_dense_diags,
                    "delta_rms_mu": 0.0,
                    "delta_rms_W": 0.0,
                    "pose_score_only": True,
                    **dense_probe_delta,
                }
            )
        )
        center_rotation_grid = dense_rotations
        center_translation_grid = translations_np
        bridge.n_rotations = int(center_rotation_grid.shape[0])
        bridge.n_translations = int(center_translation_grid.shape[0])
        bridge.translations = np.asarray(center_translation_grid, dtype=np.float32)
        stage_history.append(
            {
                "iteration": int(global_iter),
                "stage": "pose_warmup_dense",
                "current_size": int(state.schedule_state.current_size),
                "dense_probe_current_size": int(dense_probe_current_size),
                "healpix_order": int(bridge.state.healpix_order),
                "top_p_poses": int(pose_selection.top_p_poses),
                "dense_pose_count": int(dense_pose_count),
                **_pose_diagnostic_summary(state.pose_diagnostics),
                **{
                    key: value
                    for key, value in dense_probe_delta.items()
                    if key.endswith("_changed_fraction")
                    or key.endswith("_angle_deg_mean")
                    or key.endswith("_translation_px_mean")
                },
            }
        )

        local_order = int(bridge.state.healpix_order) + int(bridge.state.adaptive_oversampling)
        target_rotation_grid = np.asarray(get_relion_rotation_grid(local_order), dtype=np.float32)
        sigma_rot = float(bridge.state.sigma_rot or np.deg2rad(max(bridge.state.effective_step, 1e-6)))
        sigma_psi = float(bridge.state.sigma_psi or sigma_rot)
        sigma_offset = float(max(bridge.state.translation_step, 1e-6) * getattr(reference_dataset, "voxel_size", 1.0))
        layouts = build_top_p_local_hypothesis_layouts_from_diagnostics(
            state.pose_diagnostics,
            center_rotation_grid=center_rotation_grid,
            target_rotation_grid=target_rotation_grid,
            healpix_order=local_order,
            translations=center_translation_grid,
            sigma_rot=sigma_rot,
            sigma_psi=sigma_psi,
            sigma_offset_angstrom=sigma_offset,
            voxel_size=float(getattr(reference_dataset, "voxel_size", 1.0)),
        )
        dense_pose_diagnostics = dict(state.pose_diagnostics or {})
        state = run_local_ppca_halfset_pose_scoring_iteration(
            state,
            halfsets,
            layouts,
            geometry=GeometryConfig(current_size=int(state.schedule_state.current_size), volume_domain="fourier_half"),
            schedule=ScheduleConfig(
                image_batch_size=2,
                rotation_block_size=512,
                mstep_chunk_size=mstep_chunk_size,
                local_image_shard_count=int(local_image_shard_count),
            ),
            scoring=scoring,
            mean_reg=mean_reg,
            pose_selection=pose_selection,
            disc_type=disc_type,
        )
        state = state.replace(
            pose_diagnostics=_canonicalize_pose_diagnostics_by_image_index(state.pose_diagnostics)
        )
        local_probe_delta = _pose_probe_delta_diagnostics(
            dense_pose_diagnostics,
            state.pose_diagnostics,
            reference_rotation_grid=center_rotation_grid,
            candidate_rotation_grid=target_rotation_grid,
            reference_translations=center_translation_grid,
            candidate_translations=center_translation_grid,
            prefix="pose_probe_local_vs_dense",
        )
        local_pose_diags = dict(state.pose_diagnostics)
        for half_idx, half_key in enumerate(("halfset0", "halfset1")):
            half_layout = layouts[half_idx]
            rotation_counts = getattr(half_layout, "rotation_counts", None)
            translation_grid = getattr(half_layout, "translation_grid", None)
            half_diag = dict(local_pose_diags.get(half_key, {}))
            half_diag.update(
                {
                    "path": "pose_warmup_local",
                    "pose_score_only": True,
                    "warmup_local_healpix_order": int(local_order),
                    "warmup_local_rotation_count_mean": (
                        float(np.mean(rotation_counts)) if rotation_counts is not None else 0.0
                    ),
                    "warmup_local_translation_count": (
                        int(np.asarray(translation_grid).shape[0]) if translation_grid is not None else 0
                    ),
                }
            )
            local_pose_diags[half_key] = half_diag
        state = state.replace(
            pose_diagnostics=_canonicalize_pose_diagnostics_by_image_index(
                {**local_pose_diags, **dense_probe_delta, **local_probe_delta}
            )
        )
        center_rotation_grid = target_rotation_grid
        bridge.n_rotations = int(center_rotation_grid.shape[0])
        stage_history.append(
            {
                "iteration": int(global_iter),
                "stage": "pose_warmup_local",
                "current_size": int(state.schedule_state.current_size),
                "healpix_order": int(local_order),
                "top_p_poses": int(pose_selection.top_p_poses),
                "do_local_search": bool(bridge.state.do_local_search),
                **_pose_diagnostic_summary(state.pose_diagnostics),
                **{
                    key: value
                    for key, value in local_probe_delta.items()
                    if key.endswith("_changed_fraction")
                    or key.endswith("_angle_deg_mean")
                    or key.endswith("_translation_px_mean")
                },
            }
        )
        global_iter += 1

    for _ in range(int(n_dense_iterations)):
        dense_rotations = _rotation_grid_for_bridge()
        if int(dense_rotations.shape[0]) * int(translations_np.shape[0]) > int(max_dense_pose_candidates):
            stage_history.append(
                {
                    "iteration": int(global_iter),
                    "stage": "dense_skipped_grid_cap",
                    "dense_pose_count": int(dense_rotations.shape[0] * translations_np.shape[0]),
                    "max_dense_pose_candidates": int(max_dense_pose_candidates),
                }
            )
            break
        state = run_dense_ppca_halfset_fused_em_iteration(
            state,
            experiment_dataset,
            rotations=dense_rotations,
            translations=translations_np,
            geometry=GeometryConfig(current_size=int(state.schedule_state.current_size), volume_domain="fourier_half"),
            schedule=schedule,
            scoring=scoring,
            mean_reg=mean_reg,
            postprocess=postprocess,
            pose_selection=pose_selection,
            disc_type=disc_type,
        )
        center_rotation_grid = dense_rotations
        center_translation_grid = translations_np
        bridge.n_rotations = int(center_rotation_grid.shape[0])
        _gate("dense", global_iter)
        global_iter += 1
        if bridge.state.do_local_search:
            break

    for _ in range(int(n_adaptive_iterations)):
        if int(adaptive_oversampling) <= 0 or bridge.state.do_local_search:
            break
        coarse_rotations = _rotation_grid_for_bridge()
        state = run_adaptive_ppca_halfset_fused_em_iteration(
            state,
            halfsets,
            coarse_rotations=coarse_rotations,
            coarse_translations=translations_np,
            nside_level=int(bridge.state.healpix_order),
            adaptive_oversampling=int(adaptive_oversampling),
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            geometry=GeometryConfig(current_size=int(state.schedule_state.current_size), volume_domain="fourier_half"),
            schedule=schedule,
            scoring=scoring,
            mean_reg=mean_reg,
            postprocess=postprocess,
            pose_selection=pose_selection,
            sparse_pass2=local_sparse_pass2,
            translation_step=float(bridge.state.translation_step),
            disc_type=disc_type,
        )
        center_rotation_grid = get_relion_rotation_grid(int(bridge.state.healpix_order) + int(adaptive_oversampling))
        center_translation_grid = np.asarray(
            get_oversampled_translation_grid(
                translations_np,
                float(bridge.state.translation_step),
                oversampling_order=int(adaptive_oversampling),
            )[0],
            dtype=np.float32,
        )
        bridge.n_rotations = int(center_rotation_grid.shape[0])
        _gate("adaptive", global_iter)
        global_iter += 1

    for _ in range(int(n_local_iterations)):
        local_order = int(bridge.state.healpix_order) + int(bridge.state.adaptive_oversampling)
        target_rotation_grid = np.asarray(get_relion_rotation_grid(local_order), dtype=np.float32)
        sigma_rot = float(bridge.state.sigma_rot or np.deg2rad(max(bridge.state.effective_step, 1e-6)))
        sigma_psi = float(bridge.state.sigma_psi or sigma_rot)
        sigma_offset = float(max(bridge.state.translation_step, 1e-6) * getattr(reference_dataset, "voxel_size", 1.0))
        state = state.replace(
            pose_diagnostics=_canonicalize_pose_diagnostics_by_image_index(state.pose_diagnostics)
        )
        layouts = build_top_p_local_hypothesis_layouts_from_diagnostics(
            state.pose_diagnostics,
            center_rotation_grid=center_rotation_grid,
            target_rotation_grid=target_rotation_grid,
            healpix_order=local_order,
            translations=center_translation_grid,
            sigma_rot=sigma_rot,
            sigma_psi=sigma_psi,
            sigma_offset_angstrom=sigma_offset,
            voxel_size=float(getattr(reference_dataset, "voxel_size", 1.0)),
        )
        state = run_local_ppca_halfset_fused_em_iteration(
            state,
            halfsets,
            layouts,
            geometry=GeometryConfig(current_size=int(state.schedule_state.current_size), volume_domain="fourier_half"),
            schedule=ScheduleConfig(
                image_batch_size=2,
                rotation_block_size=512,
                mstep_chunk_size=mstep_chunk_size,
                local_image_shard_count=int(local_image_shard_count),
            ),
            scoring=scoring,
            sparse_pass2=local_sparse_pass2,
            mean_reg=mean_reg,
            postprocess=postprocess,
            pose_selection=pose_selection,
            disc_type=disc_type,
        )
        state = state.replace(
            pose_diagnostics=_canonicalize_pose_diagnostics_by_image_index(state.pose_diagnostics)
        )
        center_rotation_grid = target_rotation_grid
        bridge.n_rotations = int(center_rotation_grid.shape[0])
        _gate("exact_local", global_iter)
        global_iter += 1

    return HighresPPCARefinementResult(
        final_state=state,
        iteration_records=records,
        bridge=bridge,
        diagnostics={
            "stage_history": stage_history,
            "n_pose_warmup_iterations": int(n_pose_warmup_iterations),
            "pose_warmup_dense_current_size": (
                None if pose_warmup_dense_current_size is None else int(pose_warmup_dense_current_size)
            ),
            "top_p_poses": int(pose_selection.top_p_poses),
            "max_dense_pose_candidates": int(max_dense_pose_candidates),
            "bridge_history": bridge.history,
        },
    )


def run_highres_ppca_refinement_from_pipeline_ppca(
    experiment_dataset,
    *,
    mean,
    mean_prior,
    W_prior,
    noise_variance,
    W=None,
    eigenvectors=None,
    eigenvalues=None,
    volume_shape=None,
    q: int | None = None,
    volume_domain: str = "auto",
    **kwargs,
) -> HighresPPCARefinementResult:
    """Initialize from pipeline PPCA arrays, then run high-resolution refinement."""

    volume_shape = tuple(
        int(x)
        for x in (
            experiment_dataset.volume_shape
            if volume_shape is None
            else volume_shape
        )
    )
    state = initialize_state_from_pipeline_ppca(
        mean,
        W=W,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        mean_prior=mean_prior,
        W_prior=W_prior,
        noise_variance=noise_variance,
        volume_shape=volume_shape,
        q=q,
        volume_domain=volume_domain,
    )
    return run_highres_ppca_refinement_with_kclass_pose_hierarchy(
        state,
        experiment_dataset,
        **kwargs,
    )
