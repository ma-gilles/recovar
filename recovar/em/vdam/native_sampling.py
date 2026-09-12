"""Native InitialModel sampling state, plan and accuracy updates.

RELION's ``updateAngularSampling`` counterpart for the native driver: the
sampling state and plan records, the per-iteration random perturbation, the
expected-accuracy estimate and the assignment-change trackers that decide when
the angular sampling is refined.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.em import sampling
from recovar.em.helpers.convergence import (
    compute_relion_offset_changes_angstrom,
    compute_relion_orientation_changes,
    relion_mpi_hidden_variable_change_is_small,
)
from recovar.em.helpers.expected_accuracy import (
    estimate_relion_expected_accuracy_from_prepared_inputs,
    estimate_relion_expected_accuracy_in_spawned_process_from_prepared_inputs,
)
from recovar.em.vdam.native_options import (
    DEFAULT_HEALPIX_ORDER,
    DEFAULT_OFFSET_RANGE_PX,
    DEFAULT_OFFSET_STEP_PX,
    DEFAULT_OVERSAMPLING,
    NativeInitialModelOptions,
)
from recovar.em.vdam.star_io import NativeParticleState
from recovar.em.vdam.state import InitialModelState
from recovar.em.vdam.subset import RndUnifFn
from recovar.utils.helpers import R_to_relion, recovar_volume_to_relion

RELION_INITIALMODEL_LOCAL_SEARCH_HEALPIX_ORDER = 4


RELION_ORIENTATIONAL_PRIOR_NOPRIOR = 0


RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI = 1


RELION_INITIALMODEL_MIN_TRANSLATION_STEP_ANGSTROM = 1.5


RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN = 1


RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS = 999.0


RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS = 999.0


RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES = 9999999.0


INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV = "RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS"


@dataclass(frozen=True)
class NativeSamplingPlan:
    """Trial geometry; sparse execution can defer the unused dense fine grid."""

    rotations: np.ndarray | None
    translations: np.ndarray
    random_perturbation: float
    healpix_order: int = DEFAULT_HEALPIX_ORDER
    oversampling: int = DEFAULT_OVERSAMPLING
    offset_range_px: float = DEFAULT_OFFSET_RANGE_PX
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX
    offset_range_angstrom: float = DEFAULT_OFFSET_RANGE_PX
    offset_step_angstrom: float = DEFAULT_OFFSET_STEP_PX
    coarse_translations: np.ndarray | None = None
    coarse_prior_translations: np.ndarray | None = None
    translation_parent: np.ndarray | None = None
    metadata_translations: np.ndarray | None = None

    @property
    def n_rotations(self) -> int:
        if self.rotations is not None:
            return int(self.rotations.shape[0])
        return int(sampling.rotation_grid_size(self.healpix_order)) * 8 ** int(self.oversampling)


@dataclass
class NativeSamplingState:
    """RELION InitialModel autosampling state (Angstroms internally, like RELION)."""

    healpix_order: int
    adaptive_oversampling: int
    offset_range_angstrom: float
    offset_step_angstrom: float
    offset_range_ori_angstrom: float
    offset_step_ori_angstrom: float
    pixel_size: float
    auto_local_healpix_order: int = RELION_INITIALMODEL_LOCAL_SEARCH_HEALPIX_ORDER
    # acc_rot=0 means "not fine enough yet" pending calculateExpectedAngularErrors port.
    acc_rot: float = 0.0
    acc_trans_angstrom: float = 999.0
    current_changes_optimal_offsets_angstrom: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    current_changes_optimal_orientations: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    current_changes_optimal_classes: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES
    smallest_changes_optimal_offsets_angstrom: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    smallest_changes_optimal_orientations: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    smallest_changes_optimal_classes: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES
    nr_iter_wo_resol_gain: int = 0
    nr_iter_wo_large_hidden_variable_changes: int = 0
    has_fine_enough_angular_sampling: bool = False
    last_current_resolution: float = 0.0
    orientational_prior_mode: int = RELION_ORIENTATIONAL_PRIOR_NOPRIOR
    uniform_local_orientation_prior: bool = False

    @property
    def offset_range_px(self) -> float:
        return float(self.offset_range_angstrom) / float(self.pixel_size)

    @property
    def offset_step_px(self) -> float:
        return float(self.offset_step_angstrom) / float(self.pixel_size)

    @property
    def effective_offset_step_angstrom(self) -> float:
        return float(self.offset_step_angstrom) / (2 ** int(self.adaptive_oversampling))


@dataclass(frozen=True)
class NativeOpticsState:
    """Scalar optics plus per-particle CTF parameters for the SPA InitialModel path."""

    voltage: float
    Cs: float
    Q0: float
    pixel_size: float
    defU: np.ndarray
    defV: np.ndarray
    defAngle: np.ndarray
    phase_shift: np.ndarray


def _relion_rnd_unif_factory(seed: int) -> RndUnifFn:
    """Return a RELION ``rnd_unif`` source using the local C++ binding."""

    from recovar.relion_bind import _relion_bind_core as bind

    cache = np.asarray(bind.vdam_rnd_unif_sequence(int(seed), 1024), dtype=np.float64)

    def _rnd(call_idx: int) -> float:
        nonlocal cache
        if call_idx >= cache.size:
            new_size = max(call_idx + 1, cache.size * 2)
            cache = np.asarray(bind.vdam_rnd_unif_sequence(int(seed), int(new_size)), dtype=np.float64)
        return float(cache[call_idx])

    return _rnd


def _initial_sampling_state(opts: NativeInitialModelOptions, *, pixel_size: float) -> NativeSamplingState:
    pixel_size = float(pixel_size)
    if pixel_size <= 0.0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    return NativeSamplingState(
        healpix_order=int(opts.healpix_order),
        adaptive_oversampling=int(opts.oversampling),
        offset_range_angstrom=float(opts.offset_range_px) * pixel_size,
        offset_step_angstrom=float(opts.offset_step_px) * pixel_size,
        offset_range_ori_angstrom=float(opts.offset_range_px) * pixel_size,
        offset_step_ori_angstrom=float(opts.offset_step_px) * pixel_size,
        pixel_size=pixel_size,
    )


def _should_update_native_sampling(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """Mirror the InitialModel ``updateAngularSampling`` call cadence."""
    iteration = int(iteration)
    if iteration <= 1 or (bool(do_grad) and iteration % 10 != 0):
        return False
    return iteration <= int(nr_iter)


def _record_resolution_stall_for_sampling(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    iteration: int,
) -> None:
    """Track RELION's post-maximization resolution-stall counter."""
    current_resolution = float(state.current_resolution)
    if current_resolution <= float(sampling_state.last_current_resolution) + 0.0001:
        sampling_state.nr_iter_wo_resol_gain += 1
    else:
        sampling_state.nr_iter_wo_resol_gain = 0
    sampling_state.last_current_resolution = current_resolution


def _record_native_sampling_post_iteration(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    iteration: int,
    meta: dict,
) -> None:
    """Record sampling-controller state after the completed M-step.

    RELION calls ``updateAngularSampling`` near the start of expectation, then
    calls ``updateCurrentResolution`` after maximization.  Keep that ordering:
    the sampling decision for iteration ``N`` must only see the stall counter
    written by iteration ``N - 1``.
    """
    _record_resolution_stall_for_sampling(sampling_state, state, iteration=iteration)
    meta["sampling_nr_iter_wo_resol_gain"] = int(sampling_state.nr_iter_wo_resol_gain)
    meta["sampling_nr_iter_wo_large_hidden_variable_changes"] = int(
        sampling_state.nr_iter_wo_large_hidden_variable_changes
    )
    meta["sampling_last_current_resolution"] = float(sampling_state.last_current_resolution)


def _reset_native_sampling_change_trackers(sampling_state: NativeSamplingState) -> None:
    sampling_state.nr_iter_wo_resol_gain = 0
    sampling_state.nr_iter_wo_large_hidden_variable_changes = 0
    sampling_state.smallest_changes_optimal_offsets_angstrom = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    sampling_state.smallest_changes_optimal_orientations = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    sampling_state.smallest_changes_optimal_classes = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES


def _relion_update_native_sampling_state(
    sampling_state: NativeSamplingState,
    *,
    do_grad: bool,
    do_auto_refine: bool = False,
) -> bool:
    """RELION InitialModel autosampling update (``--auto_sampling --grad`` mode).

    HEALPix growth stops before the local-search order; translation range/step
    follow the RELION formulas.
    """
    old_angular_step = sampling.relion_angular_sampling_deg(
        sampling_state.healpix_order,
        sampling_state.adaptive_oversampling,
    )
    if old_angular_step < 0.75 * float(sampling_state.acc_rot):
        sampling_state.has_fine_enough_angular_sampling = True
        return False
    sampling_state.has_fine_enough_angular_sampling = False

    oversampling_factor = 2 ** int(sampling_state.adaptive_oversampling)
    new_step = (
        min(
            RELION_INITIALMODEL_MIN_TRANSLATION_STEP_ANGSTROM,
            0.75 * float(sampling_state.acc_trans_angstrom),
        )
        * oversampling_factor
    )
    new_range = 5.0 * float(sampling_state.current_changes_optimal_offsets_angstrom)
    new_range = min(1.3 * float(sampling_state.offset_range_angstrom), new_range)
    new_range = max(new_range, 1.5 * new_step)
    if new_range > 4.0 * new_step:
        new_range /= 2.0
    if new_range > 4.0 * new_step:
        new_step = new_range / 4.0

    new_healpix_order = int(sampling_state.healpix_order)
    requested_healpix_order = new_healpix_order + 1
    gradient_ceiling_reached = (
        bool(do_grad)
        and not bool(do_auto_refine)
        and requested_healpix_order >= int(sampling_state.auto_local_healpix_order)
    )
    if not gradient_ceiling_reached:
        new_healpix_order = requested_healpix_order

    if new_step > float(sampling_state.offset_step_angstrom):
        new_step = float(sampling_state.offset_step_angstrom)
        new_range = float(sampling_state.offset_range_angstrom)

    old_orientational_prior_mode = int(sampling_state.orientational_prior_mode)
    old_uniform_local_orientation_prior = bool(sampling_state.uniform_local_orientation_prior)
    if requested_healpix_order >= int(sampling_state.auto_local_healpix_order):
        sampling_state.orientational_prior_mode = RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI
        # RELION's gradient InitialModel stops at the exhaustive HEALPix-3
        # grid, but its pinned updateAngularSampling path still switches to
        # PRIOR_ROTTILT_PSI. The stored zero angular-prior widths then produce
        # uniform direction and psi priors, as observed at the live GPU score
        # boundary. Keep this transition explicit instead of carrying the
        # learned pdf_direction into iteration 90 and later.
        sampling_state.uniform_local_orientation_prior = bool(gradient_ceiling_reached)

    changed = (
        new_healpix_order != int(sampling_state.healpix_order)
        or abs(new_step - float(sampling_state.offset_step_angstrom)) > 1e-12
        or abs(new_range - float(sampling_state.offset_range_angstrom)) > 1e-12
        or int(sampling_state.orientational_prior_mode) != old_orientational_prior_mode
        or bool(sampling_state.uniform_local_orientation_prior) != old_uniform_local_orientation_prior
    )
    sampling_state.healpix_order = int(new_healpix_order)
    sampling_state.offset_step_angstrom = float(new_step)
    sampling_state.offset_range_angstrom = float(new_range)
    _reset_native_sampling_change_trackers(sampling_state)
    return changed


def _prepare_native_sampling_for_iteration(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    iteration: int,
    do_grad: bool,
) -> bool:
    # MlOptimiser::iterate resets both convergence counters before expectation
    # during the initial gradient burn-in.  The completed M-step may populate
    # them again for the checkpoint written by this iteration.
    if bool(do_grad) and int(iteration) < 10:
        sampling_state.nr_iter_wo_resol_gain = 0
        sampling_state.nr_iter_wo_large_hidden_variable_changes = 0
    if not _should_update_native_sampling(iteration=iteration, nr_iter=int(state.nr_iter), do_grad=do_grad):
        return False
    if sampling_state.nr_iter_wo_resol_gain < RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN:
        return False
    # RELION initialiseGeneral sets auto_ignore_angle_changes for the entire
    # gradient_refine run, including its final EM phase. InitialModel therefore
    # updates sampling on resolution stalls even when assignments still change.
    return _relion_update_native_sampling_state(sampling_state, do_grad=do_grad)


def _isolate_native_sampling_accuracy_diagnostic() -> bool:
    """Return whether expected accuracy runs in a fresh spawned process."""
    value = os.environ.get(INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV, "").strip()
    if value not in {"", "0", "1"}:
        raise ValueError(f"{INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV} must be 0 or 1")
    return value == "1"


def _best_eulers_from_particle_state(
    particle_state: NativeParticleState,
    particle_ids: np.ndarray,
    *,
    rotation_grid_order: int,
) -> np.ndarray | None:
    """Resolve exact source rows first, with explicit legacy fallback per row."""
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    n_particles = len(particle_state.translation_offsets)
    source = particle_state.best_pose_eulers_deg
    valid = particle_state.best_pose_eulers_valid
    if valid is not None:
        valid = np.asarray(valid)
        if valid.dtype != bool or valid.shape != (n_particles,):
            raise ValueError("source Euler validity must match the particle table")
    if source is not None:
        source = np.asarray(source)
        if (
            source.dtype != np.float64
            or source.shape != (n_particles, 3)
            or valid is None
            or not np.all(np.isfinite(source[valid]))
        ):
            raise ValueError("source Euler metadata must be finite float64 triples on valid particle rows")
    elif valid is not None and np.any(valid):
        raise ValueError("valid source Euler metadata requires an Euler array")
    resolved = np.zeros(ids.size, dtype=bool) if valid is None else valid[ids].copy()
    result = np.empty((ids.size, 3), dtype=np.float64)
    if source is not None:
        result[resolved] = source[ids[resolved]]
    rotations = particle_state.best_pose_rotations
    if rotations is not None and not np.all(resolved):
        rotations = np.asarray(rotations)
        if rotations.shape != (n_particles, 3, 3):
            raise ValueError("pose matrices must match the particle table")
        selected = rotations[ids]
        matrix_rows = ~resolved & np.any(np.abs(selected.reshape(ids.size, 9)) > 0, axis=1)
        if np.any(matrix_rows):
            result[matrix_rows] = np.asarray(R_to_relion(selected[matrix_rows].astype(np.float64), degrees=True))
            resolved[matrix_rows] = True
    if not np.all(resolved) and particle_state.best_pose_rotation_ids is not None:
        best_ids = np.asarray(particle_state.best_pose_rotation_ids, dtype=np.int64)[ids]
        eulers = sampling.get_relion_rotation_grid_eulers(int(rotation_grid_order), rotation_index_order="relion")
        grid_rows = ~resolved & (best_ids >= 0) & (best_ids < len(eulers))
        result[grid_rows] = eulers[best_ids[grid_rows]]
        resolved[grid_rows] = True
    return result if np.all(resolved) else None


def _estimate_native_sampling_accuracy(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    particle_state: NativeParticleState,
    optics_state: NativeOpticsState,
    *,
    particle_order: np.ndarray,
    random_seed: int,
    padding_factor: int,
    sigma2_fudge: float,
) -> dict[str, object] | None:
    n_trials = min(100, int(particle_order.size))
    if n_trials <= 0:
        return None
    trial_particle_ids = np.asarray(particle_order[:n_trials], dtype=np.int64)
    eulers = _best_eulers_from_particle_state(
        particle_state,
        trial_particle_ids,
        rotation_grid_order=int(sampling_state.healpix_order) + int(sampling_state.adaptive_oversampling),
    )
    if eulers is None:
        return None
    class_ids = np.asarray(particle_state.class_assignments, dtype=np.int32)[trial_particle_ids]
    if np.any(class_ids < 0) or np.any(class_ids >= int(state.K)):
        return None

    random_seed_particle_ids = np.arange(n_trials, dtype=np.int64)
    if state.sorted_particle_part_ids is not None:
        sorted_particle_ids = np.asarray(state.sorted_particle_ids, dtype=np.int64)
        sorted_part_ids = np.asarray(state.sorted_particle_part_ids, dtype=np.int64)
        if sorted_particle_ids.shape != sorted_part_ids.shape:
            raise ValueError("stored RELION particle ids and part ids must have matching shapes")
        if sorted_particle_ids.size < particle_order.size or not np.array_equal(
            sorted_particle_ids[: particle_order.size],
            np.asarray(particle_order, dtype=np.int64),
        ):
            raise ValueError("sampling-accuracy particle order is not the stored RELION subset prefix")
        random_seed_particle_ids = sorted_part_ids[:n_trials].copy()

    refs_relion = np.stack(
        [np.asarray(recovar_volume_to_relion(ref), dtype=np.float64) for ref in np.asarray(state.Iref)],
        axis=0,
    )
    current_image_size = int(state.current_size if state.current_size > 0 else state.ori_size)
    accuracy_estimator = (
        estimate_relion_expected_accuracy_in_spawned_process_from_prepared_inputs
        if _isolate_native_sampling_accuracy_diagnostic()
        else estimate_relion_expected_accuracy_from_prepared_inputs
    )
    accuracy = accuracy_estimator(
        references_relion=refs_relion,
        trial_eulers_deg=eulers,
        trial_local_indices=trial_particle_ids,
        trial_class_ids=class_ids,
        class_weights=np.asarray(state.pdf_class, dtype=np.float64),
        sigma2_noise_relion=np.asarray(state.sigma2_noise[0], dtype=np.float64),
        defocus_u=np.asarray(optics_state.defU, dtype=np.float64),
        defocus_v=np.asarray(optics_state.defV, dtype=np.float64),
        defocus_angle=np.asarray(optics_state.defAngle, dtype=np.float64),
        phase_shift=np.asarray(optics_state.phase_shift, dtype=np.float64),
        voltage=float(optics_state.voltage),
        spherical_aberration=float(optics_state.Cs),
        amplitude_contrast=float(optics_state.Q0),
        pixel_size=float(optics_state.pixel_size),
        ori_size=int(state.ori_size),
        current_image_size=current_image_size,
        padding_factor=int(padding_factor),
        sigma2_fudge=float(sigma2_fudge),
        random_seed=int(random_seed),
        do_ctf_correction=True,
        # RELION seeds these trials with Experiment's internal ``part_id``,
        # not the original input-table row ids carried by RECOVAR's dataset.
        random_seed_particle_ids=random_seed_particle_ids,
    )
    dump_dir = os.environ.get("RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR", "").strip()
    dump_iterations = os.environ.get(
        "RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_ITERATIONS",
        "",
    ).strip()
    selected_dump_iterations = {
        int(value.strip()) for value in dump_iterations.split(",") if value.strip()
    }
    if dump_dir and (
        not selected_dump_iterations or int(state.iter) in selected_dump_iterations
    ):
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path / f"iter{int(state.iter):03d}_expected_accuracy_inputs.npz",
            refs_relion=refs_relion,
            eulers=eulers,
            source_eulers_valid=(
                np.zeros(n_trials, dtype=bool)
                if particle_state.best_pose_eulers_valid is None
                else np.asarray(particle_state.best_pose_eulers_valid)[trial_particle_ids]
            ),
            source_eulers_deg=(
                np.zeros((n_trials, 3), dtype=np.float64)
                if particle_state.best_pose_eulers_deg is None
                else np.asarray(particle_state.best_pose_eulers_deg)[trial_particle_ids]
            ),
            trial_particle_ids=trial_particle_ids,
            class_ids=class_ids,
            pdf_class=np.asarray(state.pdf_class, dtype=np.float64),
            sigma2_noise=np.asarray(state.sigma2_noise[0], dtype=np.float64),
            defU=np.asarray(optics_state.defU, dtype=np.float64),
            defV=np.asarray(optics_state.defV, dtype=np.float64),
            defAngle=np.asarray(optics_state.defAngle, dtype=np.float64),
            phase_shift=np.asarray(optics_state.phase_shift, dtype=np.float64),
            voltage=np.asarray(float(optics_state.voltage), dtype=np.float64),
            Cs=np.asarray(float(optics_state.Cs), dtype=np.float64),
            Q0=np.asarray(float(optics_state.Q0), dtype=np.float64),
            pixel_size=np.asarray(float(optics_state.pixel_size), dtype=np.float64),
            ori_size=np.asarray(int(state.ori_size), dtype=np.int64),
            current_image_size=np.asarray(current_image_size, dtype=np.int64),
            padding_factor=np.asarray(int(padding_factor), dtype=np.int64),
            sigma2_fudge=np.asarray(float(sigma2_fudge), dtype=np.float64),
            random_seed=np.asarray(int(random_seed), dtype=np.int64),
            random_seed_particle_ids=random_seed_particle_ids,
            acc_rot=np.asarray(accuracy.acc_rot, dtype=np.float64),
            acc_trans=np.asarray(accuracy.acc_trans_angstrom, dtype=np.float64),
        )
    sampling_state.acc_rot = accuracy.acc_rot
    sampling_state.acc_trans_angstrom = accuracy.acc_trans_angstrom
    return {
        "estimated_acc_rot": accuracy.acc_rot,
        "estimated_acc_trans_angstrom": accuracy.acc_trans_angstrom,
        "estimated_acc_rot_class": accuracy.acc_rot_per_class,
        "estimated_acc_trans_class": accuracy.acc_trans_per_class_angstrom,
        "estimated_acc_class_counts": accuracy.class_counts,
        "estimated_acc_n_trials": int(n_trials),
        "estimated_acc_sigma2_fudge": float(sigma2_fudge),
        "estimated_acc_seed_part_ids": random_seed_particle_ids,
    }


def _record_native_sampling_assignment_changes(
    sampling_state: NativeSamplingState,
    *,
    particle_ids: np.ndarray | None,
    previous_translations: np.ndarray,
    current_translations: np.ndarray,
    previous_rotations: np.ndarray | None,
    current_rotations: np.ndarray | None,
    previous_classes: np.ndarray,
    current_classes: np.ndarray,
) -> None:
    if particle_ids is None:
        return
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return

    prev_t = np.asarray(previous_translations, dtype=np.float64)
    curr_t = np.asarray(current_translations, dtype=np.float64)
    current_offsets = compute_relion_offset_changes_angstrom(
        curr_t[ids, :2],
        prev_t[ids, :2],
        float(sampling_state.pixel_size),
    )
    sampling_state.current_changes_optimal_offsets_angstrom = current_offsets

    current_orientations = compute_relion_orientation_changes(
        None if current_rotations is None else np.asarray(current_rotations)[ids],
        None if previous_rotations is None else np.asarray(previous_rotations)[ids],
    )
    sampling_state.current_changes_optimal_orientations = current_orientations

    prev_c = np.asarray(previous_classes, dtype=np.int32)
    curr_c = np.asarray(current_classes, dtype=np.int32)
    class_changes = float(np.count_nonzero(curr_c[ids] != prev_c[ids])) / float(ids.size)
    sampling_state.current_changes_optimal_classes = class_changes

    if np.isfinite(current_offsets) and np.isfinite(current_orientations):
        # The shared predicate is named for the MPI controller because that
        # path supplies leader-held sampling steps.  InitialModel is the same
        # RELION predicate with this process's current effective steps.
        changes_are_small = relion_mpi_hidden_variable_change_is_small(
            current_classes=class_changes,
            current_offsets_angstrom=current_offsets,
            current_orientations_deg=current_orientations,
            smallest_classes=sampling_state.smallest_changes_optimal_classes,
            smallest_offsets_angstrom=(
                sampling_state.smallest_changes_optimal_offsets_angstrom
            ),
            smallest_orientations_deg=(
                sampling_state.smallest_changes_optimal_orientations
            ),
            mpi_leader_angular_step_deg=sampling.relion_angular_sampling_deg(
                sampling_state.healpix_order,
                sampling_state.adaptive_oversampling,
            ),
            mpi_leader_translation_step_angstrom=(
                sampling_state.effective_offset_step_angstrom
            ),
        )
        if changes_are_small:
            sampling_state.nr_iter_wo_large_hidden_variable_changes += 1
        else:
            sampling_state.nr_iter_wo_large_hidden_variable_changes = 0

        # RELION updates the sticky minima after evaluating the counter.
        if current_offsets < sampling_state.smallest_changes_optimal_offsets_angstrom:
            sampling_state.smallest_changes_optimal_offsets_angstrom = current_offsets
        if current_orientations < sampling_state.smallest_changes_optimal_orientations:
            sampling_state.smallest_changes_optimal_orientations = current_orientations
    else:
        sampling_state.nr_iter_wo_large_hidden_variable_changes = 0

    if class_changes < sampling_state.smallest_changes_optimal_classes:
        # RELION's ROUND macro is floor(x + 0.5), not Python's bankers round.
        sampling_state.smallest_changes_optimal_classes = float(
            np.floor(class_changes + 0.5)
        )


def _build_sampling_plan(
    opts: NativeInitialModelOptions,
    *,
    iteration: int = 1,
    sampling_state: NativeSamplingState | None = None,
    defer_fine_rotations: bool = False,
) -> NativeSamplingPlan:
    if sampling_state is None:
        healpix_order = int(opts.healpix_order)
        oversampling = int(opts.oversampling)
        offset_range_px = offset_range_angstrom = float(opts.offset_range_px)
        offset_step_px = offset_step_angstrom = float(opts.offset_step_px)
    else:
        healpix_order = int(sampling_state.healpix_order)
        oversampling = int(sampling_state.adaptive_oversampling)
        offset_range_px = float(sampling_state.offset_range_px)
        offset_step_px = float(sampling_state.offset_step_px)
        offset_range_angstrom = float(sampling_state.offset_range_angstrom)
        offset_step_angstrom = float(sampling_state.offset_step_angstrom)
    if oversampling < 0:
        raise ValueError("oversampling must be >= 0")

    random_perturbation = _random_perturbation_for_iteration(opts, iteration)
    perturbed = abs(random_perturbation) > 1e-12

    source_units_per_pixel = (
        float(sampling_state.pixel_size) if sampling_state is not None else 1.0
    )
    metadata_coarse_translations = sampling.get_relion_translation_grid(
        max_pixel=offset_range_px,
        pixel_offset=offset_step_px,
        source_units_per_pixel=source_units_per_pixel,
    )
    coarse_translations = metadata_coarse_translations.astype(np.float32)
    coarse_pass1_translations = (
        sampling.apply_relion_translation_perturbation(coarse_translations, random_perturbation, offset_step_px).astype(
            np.float32
        )
        if perturbed
        else coarse_translations
    )

    if oversampling == 0:
        rotations = sampling.get_relion_hidden_rotation_grid(healpix_order, matrices=True).astype(np.float32)
        translations = coarse_translations
        metadata_translations = metadata_coarse_translations
        if perturbed:
            rotations = sampling.apply_relion_rotation_perturbation(
                rotations, random_perturbation, sampling.relion_angular_sampling_deg(healpix_order)
            ).astype(np.float32)
            translations = sampling.apply_relion_translation_perturbation(
                translations, random_perturbation, offset_step_px
            ).astype(np.float32)
            metadata_translations = sampling.apply_relion_translation_perturbation(
                metadata_translations, random_perturbation, offset_step_px
            )
        translation_parent = None
    else:
        rotations = None
        if not defer_fine_rotations:
            rotations, _ = sampling.get_oversampled_relion_hidden_rotation_grid_from_samples(
                np.arange(sampling.rotation_grid_size(healpix_order), dtype=np.int64),
                parent_nside_level=healpix_order,
                oversampling_order=oversampling,
                random_perturbation=random_perturbation,
            )
        oversampled_trans, _translation_parent = sampling.get_oversampled_translation_grid(
            coarse_translations, pixel_offset=offset_step_px, oversampling_order=oversampling
        )
        metadata_translations, _metadata_translation_parent = sampling.get_oversampled_translation_grid(
            metadata_coarse_translations,
            pixel_offset=offset_step_px,
            oversampling_order=oversampling,
        )
        if not np.array_equal(_translation_parent, _metadata_translation_parent):
            raise RuntimeError("GPU and metadata translation parent maps differ")
        translations = sampling.apply_relion_translation_perturbation(
            oversampled_trans.astype(np.float32, copy=False), random_perturbation, offset_step_pixels=offset_step_px
        )
        metadata_translations = sampling.apply_relion_translation_perturbation(
            metadata_translations,
            random_perturbation,
            offset_step_pixels=offset_step_px,
        )
        translation_parent = np.asarray(_translation_parent, dtype=np.int64)

    return NativeSamplingPlan(
        rotations=None if rotations is None else np.asarray(rotations, dtype=np.float32),
        translations=np.asarray(translations, dtype=np.float32),
        random_perturbation=random_perturbation,
        healpix_order=healpix_order,
        oversampling=oversampling,
        offset_range_px=offset_range_px,
        offset_step_px=offset_step_px,
        offset_range_angstrom=offset_range_angstrom,
        offset_step_angstrom=offset_step_angstrom,
        coarse_translations=coarse_pass1_translations,
        coarse_prior_translations=coarse_translations,
        translation_parent=translation_parent,
        metadata_translations=np.asarray(metadata_translations, dtype=np.float64),
    )


def _random_perturbation_for_iteration(opts: NativeInitialModelOptions, iteration: int) -> float:
    env_override = os.environ.get("RECOVAR_RANDOM_PERTURBATION")
    if env_override is not None:
        return float(env_override)
    if opts.random_perturbation is not None:
        return float(opts.random_perturbation)
    return _random_perturbation_sequence(
        int(opts.random_seed),
        float(opts.perturbation_factor),
        max(1, int(iteration)),
    )


def _random_perturbation_sequence(random_seed: int, perturbation_factor: float, n_steps: int) -> float:
    """Replay RELION's per-iter perturbation sequence with source float arithmetic."""
    if perturbation_factor <= 0.0:
        return 0.0
    # rnd_unif(low, high) performs its range scaling inside RELION's float
    # function. Scaling a separately rounded unit draw changes the result by
    # one float32 ulp for seed 0 / iteration 1, which is enough to flip the
    # integer-truncated fine-projector radius predicate on the rounded rim.
    return sampling.relion_sampling_perturbation_for_iteration(
        float(perturbation_factor),
        int(random_seed),
        max(1, int(n_steps)),
    )
