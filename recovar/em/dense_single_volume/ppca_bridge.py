"""PPCA ↔ K-class refinement-state bridge.

The PPCA refinement loops (``recovar.em.ppca_refinement``) own the PPCA
halfset/FSC gate. This module owns the K-class side of the agreement:
``PPCAKClassScheduleBridge`` feeds PPCA best-pose / Pmax diagnostics through
``update_refinement_state`` so HEALPix / local-search / convergence state
evolves with the same controller used by ``refine_single_volume``.

Extracted from ``iteration_loop.py`` so the master iteration loop stays
focused on the EM dispatch.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import (
    LOCAL_SEARCH_HEALPIX_ORDER,
    RefinementState,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.resolution import (
    shell_index_to_resolution_angstrom,
)


def _ppca_best_pose_ids_from_diagnostics(pose_diagnostics: dict, n_translations: int) -> np.ndarray:
    """Return PPCA best pose ids in the same packed convention as K-class EM."""

    best = []
    for key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(key, {})
        if "best_rotation_id" in diag:
            rot_key = "best_rotation_id"
        elif "best_rotation_idx" in diag:
            rot_key = "best_rotation_idx"
        else:
            continue
        if "best_translation_idx" not in diag:
            continue
        rotations = np.asarray(diag[rot_key], dtype=np.int64)
        translations = np.asarray(diag["best_translation_idx"], dtype=np.int64)
        if rotations.shape != translations.shape:
            raise ValueError(
                f"PPCA diagnostic shapes differ for {key}: {rotations.shape} vs {translations.shape}",
            )
        best.append(rotations * int(n_translations) + translations)
    if not best:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(best).astype(np.int32)


def _ppca_pmax_array_from_diagnostics(pose_diagnostics: dict, n_images: int) -> np.ndarray | None:
    """Build a per-image Pmax array for ``update_refinement_state`` when possible."""

    pmax_values = []
    for key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(key, {})
        if "max_posterior_per_image" in diag:
            pmax_values.append(np.asarray(diag["max_posterior_per_image"], dtype=np.float32).reshape(-1))
        elif "pmax" in diag:
            pmax_values.append(np.asarray(diag["pmax"], dtype=np.float32).reshape(-1))
    if pmax_values:
        return np.concatenate(pmax_values).astype(np.float32)
    means = []
    for key in ("halfset0", "halfset1"):
        diag = pose_diagnostics.get(key, {})
        if "pmax_mean" in diag:
            means.append(float(diag["pmax_mean"]))
    if not means:
        return None
    return np.full(int(n_images), float(np.mean(means)), dtype=np.float32)


class PPCAKClassScheduleBridge:
    """Callable PPCA schedule hook backed by production K-class RefinementState.

    The PPCA refinement loops own the PPCA halfset/FSC gate. This bridge owns
    the K-class side of the agreement: it feeds PPCA best-pose/Pmax diagnostics
    through ``update_refinement_state`` so healpix/local-search/convergence
    state evolves with the same controller used by ``refine_single_volume``.
    """

    def __init__(
        self,
        *,
        n_rotations: int,
        translations,
        grid_size: int,
        voxel_size_angstrom: float = 1.0,
        init_healpix_order: int = 2,
        max_healpix_order: int = 7,
        auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER,
        adaptive_oversampling: int = 0,
        init_translation_range: float = 10.0,
        init_translation_step: float = 2.0,
        particle_diameter_angstrom: float = 0.0,
        refinement_state: RefinementState | None = None,
    ):
        self.n_rotations = int(n_rotations)
        self.translations = np.asarray(translations, dtype=np.float32)
        self.n_translations = int(self.translations.shape[0])
        self.grid_size = int(grid_size)
        self.voxel_size_angstrom = float(voxel_size_angstrom if voxel_size_angstrom > 0 else 1.0)
        self.previous_assignments: np.ndarray | None = None
        self.history: list[dict] = []
        if refinement_state is None:
            refinement_state = RefinementState(
                iteration=0,
                healpix_order=int(init_healpix_order),
                adaptive_oversampling=int(adaptive_oversampling),
                translation_range=float(init_translation_range),
                translation_step=float(init_translation_step),
                max_healpix_order=int(max_healpix_order),
                auto_local_healpix_order=int(auto_local_healpix_order),
                voxel_size_angstrom=self.voxel_size_angstrom,
                particle_diameter_angstrom=float(particle_diameter_angstrom),
            )
        self.state = refinement_state

    def _resolution_angstrom_from_current_size(self, proposed_current_size: int) -> float:
        shell = max(1, min(int(proposed_current_size) // 2, self.grid_size // 2))
        return float(shell_index_to_resolution_angstrom(shell, self.grid_size, self.voxel_size_angstrom))

    def __call__(
        self,
        iteration: int,
        ppca_state,
        *,
        current_size: int,
        proposed_current_size: int,
        halfset_comparison=None,
    ) -> bool:
        best = _ppca_best_pose_ids_from_diagnostics(ppca_state.pose_diagnostics, self.n_translations)
        if best.size == 0:
            self.history.append(
                {
                    "iteration": int(iteration),
                    "current_size": int(current_size),
                    "proposed_current_size": int(proposed_current_size),
                    "allowed": False,
                    "reason": "missing_ppca_best_pose_diagnostics",
                    "refinement_state": self.state,
                }
            )
            return False
        pmax = _ppca_pmax_array_from_diagnostics(ppca_state.pose_diagnostics, best.size)
        new_resolution = self._resolution_angstrom_from_current_size(proposed_current_size)
        self.state = update_refinement_state(
            self.state,
            current_assignments=best,
            previous_assignments=self.previous_assignments,
            n_rotations=self.n_rotations,
            n_translations=self.n_translations,
            translations=self.translations,
            new_resolution=new_resolution,
            max_posterior_per_image=pmax,
            voxel_size_angstrom=self.voxel_size_angstrom,
        )
        self.previous_assignments = best
        allowed = bool((not self.state.has_converged) and int(proposed_current_size) > int(current_size))
        if halfset_comparison is not None and not bool(halfset_comparison.resolution_supports):
            allowed = False
        self.history.append(
            {
                "iteration": int(iteration),
                "current_size": int(current_size),
                "proposed_current_size": int(proposed_current_size),
                "allowed": allowed,
                "refinement_state": self.state,
                "do_local_search": bool(self.state.do_local_search),
                "healpix_order": int(self.state.healpix_order),
            }
        )
        return allowed


def make_ppca_kclass_schedule_bridge(
    *,
    n_rotations: int,
    translations,
    grid_size: int,
    voxel_size_angstrom: float = 1.0,
    init_healpix_order: int = 2,
    max_healpix_order: int = 7,
    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER,
    adaptive_oversampling: int = 0,
    init_translation_range: float = 10.0,
    init_translation_step: float = 2.0,
    particle_diameter_angstrom: float = 0.0,
    refinement_state: RefinementState | None = None,
) -> PPCAKClassScheduleBridge:
    """Create the K-class schedule hook used by PPCA refinement loops."""

    return PPCAKClassScheduleBridge(
        n_rotations=n_rotations,
        translations=translations,
        grid_size=grid_size,
        voxel_size_angstrom=voxel_size_angstrom,
        init_healpix_order=init_healpix_order,
        max_healpix_order=max_healpix_order,
        auto_local_healpix_order=auto_local_healpix_order,
        adaptive_oversampling=adaptive_oversampling,
        init_translation_range=init_translation_range,
        init_translation_step=init_translation_step,
        particle_diameter_angstrom=particle_diameter_angstrom,
        refinement_state=refinement_state,
    )


def run_dense_ppca_refinement_with_kclass_schedule(
    ppca_state,
    experiment_dataset,
    *,
    rotations,
    translations,
    n_iterations: int,
    **kwargs,
):
    """Run dense PPCA refinement using the production K-class schedule bridge."""

    from recovar.em.ppca_refinement import run_dense_ppca_refinement_loop

    bridge = make_ppca_kclass_schedule_bridge(
        n_rotations=int(np.asarray(rotations).shape[0]),
        translations=translations,
        grid_size=int(experiment_dataset.image_shape[0]),
        voxel_size_angstrom=float(getattr(experiment_dataset, "voxel_size", 1.0)),
        init_healpix_order=int(kwargs.pop("init_healpix_order", 2)),
        max_healpix_order=int(kwargs.pop("max_healpix_order", 7)),
        auto_local_healpix_order=int(kwargs.pop("auto_local_healpix_order", LOCAL_SEARCH_HEALPIX_ORDER)),
        adaptive_oversampling=int(kwargs.pop("adaptive_oversampling", 0)),
        init_translation_range=float(kwargs.pop("init_translation_range", 10.0)),
        init_translation_step=float(kwargs.pop("init_translation_step", 2.0)),
        particle_diameter_angstrom=float(kwargs.pop("particle_diameter_angstrom", 0.0)),
    )
    final_state, records = run_dense_ppca_refinement_loop(
        ppca_state,
        experiment_dataset,
        rotations=rotations,
        translations=translations,
        n_iterations=n_iterations,
        kclass_schedule_allows=bridge,
        **kwargs,
    )
    return final_state, records, bridge


def run_local_ppca_refinement_with_kclass_schedule(
    ppca_state,
    halfset_datasets,
    halfset_local_layouts,
    *,
    n_iterations: int,
    **kwargs,
):
    """Run exact-local PPCA refinement using the production K-class schedule bridge."""

    from recovar.em.ppca_refinement import run_local_ppca_refinement_loop

    first_layout = halfset_local_layouts[0]
    first_dataset = halfset_datasets[0]
    bridge = make_ppca_kclass_schedule_bridge(
        n_rotations=int(first_layout.n_global_rotations),
        translations=first_layout.translation_grid,
        grid_size=int(first_dataset.image_shape[0]),
        voxel_size_angstrom=float(getattr(first_dataset, "voxel_size", 1.0)),
        init_healpix_order=int(kwargs.pop("init_healpix_order", 2)),
        max_healpix_order=int(kwargs.pop("max_healpix_order", 7)),
        auto_local_healpix_order=int(kwargs.pop("auto_local_healpix_order", LOCAL_SEARCH_HEALPIX_ORDER)),
        adaptive_oversampling=int(kwargs.pop("adaptive_oversampling", 0)),
        init_translation_range=float(kwargs.pop("init_translation_range", 10.0)),
        init_translation_step=float(kwargs.pop("init_translation_step", 2.0)),
        particle_diameter_angstrom=float(kwargs.pop("particle_diameter_angstrom", 0.0)),
    )
    final_state, records = run_local_ppca_refinement_loop(
        ppca_state,
        halfset_datasets,
        halfset_local_layouts,
        n_iterations=n_iterations,
        kclass_schedule_allows=bridge,
        **kwargs,
    )
    return final_state, records, bridge
