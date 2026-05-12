"""Native InitialModel / ab-initio K-class driver.

This module owns the executable recovar path behind ``scripts/run_ab_initio``.
The script remains a thin argparse and RELION-command-snapshot layer; all
data loading, denovo seeding, dense K-class E-step wiring, VDAM iteration, and
artifact writing lives here so InitialModel does not grow a second EM stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from recovar.core import mask as core_mask
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star, write_star
from recovar.em import sampling
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    make_relion_translation_log_prior,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
)
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import R_to_relion, recovar_volume_to_relion, write_relion_mrc

from .avg_unaligned import compute_avg_unaligned_and_sigma2
from .bootstrap_iref import compute_bootstrap_iref_via_cpp, postprocess_bootstrap_iref_via_cpp
from .dense_adapter import DenseInitialModelEstepConfig, run_dense_initial_model_estep
from .init import initialise_data_vs_prior_from_references, initialise_denovo_state, seed_noise_from_mavg
from .iteration_loop import relion_solvent_flatten_state, relion_solvent_mask, run_vdam_iterations
from .schedules import DEFAULT_GRAD_EM_ITERS, DEFAULT_GRAD_MU, default_subset_sizes_for_3d_initial_model
from .state import InitialModelState
from .subset import RndUnifFn

DEFAULT_WIDTH_MASK_EDGE_PX = 5.0
DEFAULT_HEALPIX_ORDER = 1
DEFAULT_OFFSET_RANGE_PX = 6.0
DEFAULT_OFFSET_STEP_PX = 2.0
DEFAULT_RANDOM_SEED = 0
DEFAULT_OVERSAMPLING = 1
DEFAULT_PERTURBATION_FACTOR = 0.5
RELION_INITIALMODEL_LOCAL_SEARCH_HEALPIX_ORDER = 4
RELION_INITIALMODEL_MIN_TRANSLATION_STEP_ANGSTROM = 1.5
RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN = 1
RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS = 999.0
RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS = 999.0
RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES = 9999999.0


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions:
    """Options for a native InitialModel run; defaults mirror the GUI command."""

    fn_img: str
    outputname: str = "ab_initio/run"
    nr_iter: int = 200
    nr_classes: int = 1
    tau2_fudge: float = 4.0
    sym_name: str = "C1"
    do_run_C1: bool = True
    particle_diameter: float = 200.0
    do_solvent: bool = True
    do_zero_mask: bool = True
    do_ctf_correction: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED
    width_mask_edge_px: float = DEFAULT_WIDTH_MASK_EDGE_PX
    healpix_order: int = DEFAULT_HEALPIX_ORDER
    oversampling: int = DEFAULT_OVERSAMPLING
    perturbation_factor: float = DEFAULT_PERTURBATION_FACTOR
    random_perturbation: float | None = None
    offset_range_px: float = DEFAULT_OFFSET_RANGE_PX
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    bootstrap_min_particles: int = 1000
    sigma2_min_particles: int = 1000
    padding_factor: int = 1
    lazy: bool = True
    datadir: str | None = None
    strip_prefix: str | None = None
    translation_sigma_angstrom: float | None = None
    write_iter_artifacts: bool = True
    run_relion_align_symmetry: bool = False


@dataclass(frozen=True)
class NativeInitialModelResult:
    """Summary returned by ``run_native_initial_model``."""

    state: InitialModelState
    output_prefix: str
    final_model_star: str
    final_mrc: str
    class_mrcs: tuple[str, ...]


@dataclass(frozen=True)
class NativeSamplingPlan:
    """Dense trial grid used by one native InitialModel E-step."""

    rotations: np.ndarray
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

    @property
    def offset_range_px(self) -> float:
        return float(self.offset_range_angstrom) / float(self.pixel_size)

    @property
    def offset_step_px(self) -> float:
        return float(self.offset_step_angstrom) / float(self.pixel_size)

    @property
    def effective_offset_step_angstrom(self) -> float:
        return float(self.offset_step_angstrom) / (2 ** int(self.adaptive_oversampling))


@dataclass
class NativeParticleState:
    """Per-particle metadata carried between native InitialModel iterations."""

    translation_offsets: np.ndarray
    class_assignments: np.ndarray
    max_posterior: np.ndarray
    pose_assignments: np.ndarray | None = None
    best_pose_rotations: np.ndarray | None = None
    best_pose_translations: np.ndarray | None = None
    best_pose_rotation_ids: np.ndarray | None = None
    best_pose_rotation_orders: np.ndarray | None = None
    visited: np.ndarray | None = None


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


def _output_dir_from_prefix(outputname: str) -> Path:
    parent = Path(outputname).parent
    return Path(".") if str(parent) == "" else parent


def _initial_model_mrc_from_prefix(outputname: str) -> str:
    """Mirror RELION's GUI ``outputname.rstrip("run") + initial_model.mrc``."""

    return outputname.rstrip("run") + "initial_model.mrc"


def _micrograph_sort_order(main_star) -> np.ndarray:
    """RELION reads InitialModel particles in stable micrograph-name order."""

    n = len(main_star)
    if "_rlnMicrographName" not in main_star.columns:
        return np.arange(n, dtype=np.int64)
    names = np.asarray(main_star["_rlnMicrographName"].astype(str).to_numpy())
    return np.argsort(names, kind="stable").astype(np.int64, copy=False)


def _optics_group_indices(main_star) -> np.ndarray:
    if "_rlnOpticsGroup" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.int64)
    raw = main_star["_rlnOpticsGroup"].to_numpy()
    try:
        numeric = np.asarray(raw, dtype=np.int64)
        unique = {value: i for i, value in enumerate(sorted(np.unique(numeric).tolist()))}
        return np.asarray([unique[int(value)] for value in numeric], dtype=np.int64)
    except (TypeError, ValueError):
        labels = np.asarray(raw, dtype=str)
        unique = {value: i for i, value in enumerate(sorted(np.unique(labels).tolist()))}
        return np.asarray([unique[str(value)] for value in labels], dtype=np.int64)


def _single_optics_scalars(main_star, optics_star, ds) -> tuple[float, float, float, float]:
    """Return voltage, Cs, amplitude contrast, and pixel size.

    The current C++ bootstrap binding takes scalar optics parameters. To avoid
    wrong native output, reject genuinely multi-optics inputs until the binding
    grows per-particle voltage/Cs/Q0 support.
    """

    pixel_size = float(ds.voxel_size)
    if optics_star is None:
        required = ("_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast")
        missing = [name for name in required if name not in main_star.columns]
        if missing:
            raise ValueError(
                "native InitialModel needs voltage/Cs/amplitude contrast in the STAR file; "
                f"missing {', '.join(missing)}"
            )
        values = tuple(float(main_star[name].astype(float).iloc[0]) for name in required)
        return values[0], values[1], values[2], pixel_size

    groups = _optics_group_indices(main_star)
    if np.unique(groups).size != 1 or len(optics_star) != 1:
        raise NotImplementedError(
            "native InitialModel bootstrap currently supports one optics group; "
            "multi-optics support needs per-particle optics in the RELION bootstrap binding"
        )
    row = optics_star.iloc[0]
    return (
        float(row["_rlnVoltage"]),
        float(row["_rlnSphericalAberration"]),
        float(row["_rlnAmplitudeContrast"]),
        pixel_size,
    )


def _phase_shift(main_star) -> np.ndarray:
    if "_rlnPhaseShift" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.float64)
    return np.asarray(main_star["_rlnPhaseShift"].astype(float).to_numpy(), dtype=np.float64)


def _native_optics_state(main_star, optics_star, dataset) -> NativeOpticsState:
    voltage, Cs, Q0, pixel_size = _single_optics_scalars(main_star, optics_star, dataset)
    required = ("_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle")
    missing = [name for name in required if name not in main_star.columns]
    if missing:
        raise ValueError(f"native InitialModel needs per-particle CTF columns: {', '.join(missing)}")
    return NativeOpticsState(
        voltage=float(voltage),
        Cs=float(Cs),
        Q0=float(Q0),
        pixel_size=float(pixel_size),
        defU=np.asarray(main_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(main_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(main_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=_phase_shift(main_star),
    )


def _star_column(main_star, name: str):
    if name in main_star.columns:
        return main_star[name]
    no_prefix = name[1:] if name.startswith("_") else name
    if no_prefix in main_star.columns:
        return main_star[no_prefix]
    return None


def _stack_star_pair(main_star, x_name: str, y_name: str) -> np.ndarray | None:
    x = _star_column(main_star, x_name)
    y = _star_column(main_star, y_name)
    if (x is None) != (y is None):
        raise ValueError(f"STAR file must provide both {x_name} and {y_name}")
    if x is None:
        return None
    return np.stack(
        [
            np.asarray(x.astype(float).to_numpy(), dtype=np.float64),
            np.asarray(y.astype(float).to_numpy(), dtype=np.float64),
        ],
        axis=1,
    )


def _image_origin_offsets_pixels_from_star(main_star, dataset) -> np.ndarray:
    n_images = int(len(main_star))
    angst = _stack_star_pair(main_star, "_rlnOriginXAngst", "_rlnOriginYAngst")
    if angst is not None:
        pixel_size = float(dataset.voxel_size)
        if pixel_size <= 0.0:
            raise ValueError("dataset voxel_size must be positive to convert STAR origins from Angstroms")
        shifts = angst / pixel_size
    else:
        pixels = _stack_star_pair(main_star, "_rlnOriginX", "_rlnOriginY")
        if pixels is None:
            return np.zeros((n_images, 2), dtype=np.float32)
        shifts = pixels
    if not np.all(np.isfinite(shifts)):
        raise ValueError("STAR origin shifts must be finite")
    return shifts.astype(np.float32)


def _image_pre_shifts_from_star(main_star, dataset) -> np.ndarray:
    """RELION rounded old-offset image pre-shifts in pixel units (accelerated path)."""
    return np.rint(_image_origin_offsets_pixels_from_star(main_star, dataset)).astype(np.float32)


def _particle_state_from_star(main_star, dataset) -> NativeParticleState:
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")
    class_col = _star_column(main_star, "_rlnClassNumber")
    if class_col is None:
        class_assignments = np.zeros(n_images, dtype=np.int32)
    else:
        class_assignments = np.asarray(class_col.astype(int).to_numpy(), dtype=np.int32) - 1
        if np.any(class_assignments < 0):
            raise ValueError("_rlnClassNumber values must be one-indexed positive class ids")

    pmax_col = _star_column(main_star, "_rlnMaxValueProbDistribution")
    if pmax_col is None:
        max_posterior = np.zeros(n_images, dtype=np.float32)
    else:
        max_posterior = np.asarray(pmax_col.astype(float).to_numpy(), dtype=np.float32)
        if not np.all(np.isfinite(max_posterior)):
            raise ValueError("_rlnMaxValueProbDistribution values must be finite")
    return NativeParticleState(
        translation_offsets=_image_origin_offsets_pixels_from_star(main_star, dataset),
        class_assignments=class_assignments,
        max_posterior=max_posterior,
        pose_assignments=np.full(n_images, -1, dtype=np.int32),
        visited=max_posterior > 0.0,
    )


def _load_raw_images(dataset, image_indices: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Load raw real-space particle images through ``CryoEMDataset`` I/O."""

    images: list[np.ndarray] = []
    for batch_images, _particle_indices, _local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        images.append(np.asarray(batch_images))
    if not images:
        return np.empty((0, dataset.grid_size, dataset.grid_size), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(images, axis=0))


def _image_sigma2_iter(
    dataset,
    image_indices: np.ndarray,
    optics_group_by_particle: np.ndarray,
    *,
    batch_size: int,
) -> Iterable[tuple[int, np.ndarray]]:
    for batch_images, _particle_indices, local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        batch_images = np.asarray(batch_images)
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        for image, local_idx in zip(batch_images, local_indices):
            yield int(optics_group_by_particle[int(local_idx)]), image


def _configure_relion_image_mask(dataset, opts: NativeInitialModelOptions) -> None:
    """Configure dataset preprocessing to match InitialModel scoring masks."""

    source = dataset.image_source
    backend = getattr(source, "backend", source)
    if backend is None:
        return
    image_mask = core_mask.relion_soft_image_mask(
        int(dataset.grid_size),
        float(dataset.voxel_size),
        float(opts.particle_diameter),
        float(opts.width_mask_edge_px),
    )
    if hasattr(backend, "image_mask"):
        backend.image_mask = image_mask
    if hasattr(backend, "mask"):
        backend.mask = image_mask
    if hasattr(backend, "image_mask_mode"):
        backend.image_mask_mode = "relion_background_fill"


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


def _native_initialmodel_do_grad(state: InitialModelState, iteration: int) -> bool:
    return ((int(state.nr_iter) - int(iteration)) >= DEFAULT_GRAD_EM_ITERS) and not bool(state.has_converged)


def _should_update_native_sampling(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """Mirror the InitialModel ``updateAngularSampling`` call cadence."""
    iteration = int(iteration)
    if iteration <= 1 or (bool(do_grad) and iteration % 10 != 0):
        return False
    return iteration <= int(nr_iter)


def _should_record_native_sampling_changes(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """Per-iteration ``monitorHiddenVariableChanges`` cadence (must NOT defer to autosampling)."""
    return int(iteration) <= int(nr_iter)


def _record_resolution_stall_for_sampling(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    iteration: int,
) -> None:
    """Track RELION's resolution-stall counter (audit only; not used for autosampling here)."""
    current_resolution = float(state.current_resolution)
    if int(iteration) < 10:
        sampling_state.nr_iter_wo_resol_gain = 0
        sampling_state.nr_iter_wo_large_hidden_variable_changes = 0
    elif current_resolution <= float(sampling_state.last_current_resolution) + 0.0001:
        sampling_state.nr_iter_wo_resol_gain += 1
    else:
        sampling_state.nr_iter_wo_resol_gain = 0
    sampling_state.last_current_resolution = current_resolution


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
    if not (
        bool(do_grad)
        and not bool(do_auto_refine)
        and (int(sampling_state.healpix_order) + 1) >= int(sampling_state.auto_local_healpix_order)
    ):
        new_healpix_order += 1

    if new_step > float(sampling_state.offset_step_angstrom):
        new_step = float(sampling_state.offset_step_angstrom)
        new_range = float(sampling_state.offset_range_angstrom)

    changed = (
        new_healpix_order != int(sampling_state.healpix_order)
        or abs(new_step - float(sampling_state.offset_step_angstrom)) > 1e-12
        or abs(new_range - float(sampling_state.offset_range_angstrom)) > 1e-12
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
    _record_resolution_stall_for_sampling(sampling_state, state, iteration=iteration)
    if not _should_update_native_sampling(iteration=iteration, nr_iter=int(state.nr_iter), do_grad=do_grad):
        return False
    if sampling_state.nr_iter_wo_resol_gain < RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN:
        return False
    return _relion_update_native_sampling_state(sampling_state, do_grad=do_grad)


def _should_estimate_native_sampling_accuracy(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """RELION's ``calculateExpectedAngularErrors`` cadence."""
    iteration = int(iteration)
    if iteration <= 1:
        return True
    if bool(do_grad) and iteration % 10 != 0:
        return False
    return iteration <= int(nr_iter)


def _best_eulers_from_particle_state(
    particle_state: NativeParticleState,
    particle_ids: np.ndarray,
    *,
    rotation_grid_order: int,
) -> np.ndarray | None:
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    rotations = particle_state.best_pose_rotations
    if rotations is not None:
        rotations_arr = np.asarray(rotations, dtype=np.float64)
        expected_ndim = 3
        if rotations_arr.ndim == expected_ndim and rotations_arr.shape[1:] == (3, 3):
            selected = rotations_arr[ids]
            if np.all(np.abs(selected.reshape(selected.shape[0], -1)).sum(axis=1) > 0.0):
                return np.asarray(R_to_relion(selected, degrees=True), dtype=np.float64)

    rotation_ids = particle_state.best_pose_rotation_ids
    if rotation_ids is None:
        return None
    best_ids = np.asarray(rotation_ids, dtype=np.int64)[ids]
    if np.any(best_ids < 0):
        return None
    eulers = sampling.get_relion_rotation_grid_eulers(int(rotation_grid_order), rotation_index_order="relion")
    if np.max(best_ids) >= eulers.shape[0]:
        return None
    return np.asarray(eulers[best_ids], dtype=np.float64)


def _estimate_native_sampling_accuracy(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    particle_state: NativeParticleState,
    optics_state: NativeOpticsState,
    *,
    particle_order: np.ndarray,
    random_seed: int,
    padding_factor: int,
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

    from recovar.relion_bind import _relion_bind_core as bind

    refs_relion = np.stack(
        [np.asarray(recovar_volume_to_relion(ref), dtype=np.float64) for ref in np.asarray(state.Iref)],
        axis=0,
    )
    current_image_size = int(state.current_size if state.current_size > 0 else state.ori_size)
    out = bind.vdam_expected_angular_errors(
        refs_relion,
        eulers,
        trial_particle_ids.astype(np.int64, copy=False),
        class_ids.astype(np.int32, copy=False),
        np.asarray(state.pdf_class, dtype=np.float64),
        np.asarray(state.sigma2_noise[0], dtype=np.float64),
        np.asarray(optics_state.defU, dtype=np.float64),
        np.asarray(optics_state.defV, dtype=np.float64),
        np.asarray(optics_state.defAngle, dtype=np.float64),
        np.asarray(optics_state.phase_shift, dtype=np.float64),
        float(optics_state.voltage),
        float(optics_state.Cs),
        float(optics_state.Q0),
        float(optics_state.pixel_size),
        int(state.ori_size),
        current_image_size,
        int(padding_factor),
        1,
        float(state.tau2_fudge_factor),
        int(random_seed),
        True,
        False,
    )
    sampling_state.acc_rot = float(out["acc_rot"])
    sampling_state.acc_trans_angstrom = float(out["acc_trans"])
    return {
        "estimated_acc_rot": float(out["acc_rot"]),
        "estimated_acc_trans_angstrom": float(out["acc_trans"]),
        "estimated_acc_rot_class": np.asarray(out["acc_rot_class"], dtype=np.float64),
        "estimated_acc_trans_class": np.asarray(out["acc_trans_class"], dtype=np.float64),
        "estimated_acc_class_counts": np.asarray(out["class_counts"], dtype=np.int64),
        "estimated_acc_n_trials": int(n_trials),
    }


def _record_native_sampling_assignment_changes(
    sampling_state: NativeSamplingState,
    *,
    particle_ids: np.ndarray | None,
    previous_translations: np.ndarray,
    current_translations: np.ndarray,
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
    delta = curr_t[ids, :2] - prev_t[ids, :2]
    if delta.size:
        rms_pixels = float(np.sqrt(np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2) / (2.0 * float(ids.size))))
        sampling_state.current_changes_optimal_offsets_angstrom = rms_pixels * float(sampling_state.pixel_size)
        if (
            sampling_state.current_changes_optimal_offsets_angstrom
            < sampling_state.smallest_changes_optimal_offsets_angstrom
        ):
            sampling_state.smallest_changes_optimal_offsets_angstrom = (
                sampling_state.current_changes_optimal_offsets_angstrom
            )

    prev_c = np.asarray(previous_classes, dtype=np.int32)
    curr_c = np.asarray(current_classes, dtype=np.int32)
    class_changes = float(np.count_nonzero(curr_c[ids] != prev_c[ids]))
    sampling_state.current_changes_optimal_classes = class_changes
    if class_changes < sampling_state.smallest_changes_optimal_classes:
        sampling_state.smallest_changes_optimal_classes = class_changes


def _build_sampling_plan(
    opts: NativeInitialModelOptions,
    *,
    iteration: int = 1,
    sampling_state: NativeSamplingState | None = None,
) -> NativeSamplingPlan:
    if sampling_state is None:
        healpix_order = int(opts.healpix_order)
        oversampling = int(opts.oversampling)
        offset_range_px = float(opts.offset_range_px)
        offset_step_px = float(opts.offset_step_px)
        offset_range_angstrom = offset_range_px
        offset_step_angstrom = offset_step_px
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

    coarse_translations = sampling.get_translation_grid(
        max_pixel=offset_range_px,
        pixel_offset=offset_step_px,
    ).astype(np.float32)
    coarse_pass1_translations = (
        sampling.apply_relion_translation_perturbation(
            coarse_translations,
            random_perturbation,
            offset_step_px,
        ).astype(np.float32)
        if perturbed
        else coarse_translations
    )

    translation_parent: np.ndarray | None
    if oversampling == 0:
        rotations = sampling.get_relion_hidden_rotation_grid(healpix_order, matrices=True).astype(np.float32)
        translations = coarse_translations
        if perturbed:
            rotations = sampling.apply_relion_rotation_perturbation(
                rotations,
                random_perturbation,
                sampling.relion_angular_sampling_deg(healpix_order),
            ).astype(np.float32)
            translations = sampling.apply_relion_translation_perturbation(
                translations,
                random_perturbation,
                offset_step_px,
            ).astype(np.float32)
        translation_parent = None
    else:
        coarse_indices = np.arange(sampling.rotation_grid_size(healpix_order), dtype=np.int64)
        rotations, _ = sampling.get_oversampled_relion_hidden_rotation_grid_from_samples(
            coarse_indices,
            parent_nside_level=healpix_order,
            oversampling_order=oversampling,
            random_perturbation=random_perturbation,
        )
        oversampled_trans, _translation_parent = sampling.get_oversampled_translation_grid(
            coarse_translations,
            pixel_offset=offset_step_px,
            oversampling_order=oversampling,
        )
        translations = sampling.apply_relion_translation_perturbation(
            oversampled_trans.astype(np.float32, copy=False),
            random_perturbation,
            offset_step_pixels=offset_step_px,
        )
        translation_parent = np.asarray(_translation_parent, dtype=np.int64)

    return NativeSamplingPlan(
        rotations=np.asarray(rotations, dtype=np.float32),
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
    )


def _translation_log_prior(
    translations: np.ndarray,
    *,
    voxel_size: float,
    sigma_angstrom: float | None,
    centers: np.ndarray | None = None,
) -> np.ndarray | None:
    if sigma_angstrom is None:
        return None
    sigma_angstrom = float(sigma_angstrom)
    if sigma_angstrom <= 0.0:
        raise ValueError("translation_sigma_angstrom must be positive when provided")
    translations = np.asarray(translations, dtype=np.float32)
    centers_arr = None
    if centers is not None:
        centers_arr = np.asarray(centers, dtype=np.float32)
        if centers_arr.ndim != 2 or centers_arr.shape[1] != 2:
            raise ValueError(f"translation prior centers must have shape (N, 2), got {centers_arr.shape}")
    return make_relion_translation_log_prior(
        translations[:, :2],
        voxel_size=float(voxel_size),
        sigma_offset_angstrom=sigma_angstrom,
        prior_centers=centers_arr,
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
    """Replay RELION's per-iter perturbation sequence (seed=1 first, then random_seed+step)."""
    if perturbation_factor <= 0.0:
        return 0.0
    pf = float(perturbation_factor)
    value = 0.0
    for step in range(max(1, int(n_steps)) + 1):
        seed = 1 if step == 0 else int(random_seed) + step
        value += 0.5 * pf + (pf - 0.5 * pf) * _relion_rnd_unif_factory(seed)(0)
        while value > pf:
            value -= 2.0 * pf
        while value < -pf:
            value += 2.0 * pf
    return float(value)


def _noise_variance_from_sigma2(sigma2_noise: np.ndarray, ori_size: int) -> np.ndarray:
    """Convert RELION normalized shell power to engine-frame radial noise (unnormalised FFT)."""
    n4 = int(ori_size) ** 4
    return (
        np.asarray(make_radial_noise(np.asarray(sigma2_noise)[0] * n4, (ori_size, ori_size)))
        .astype(np.float32, copy=False)
        .reshape(-1)
    )


def _n_directions_for_healpix_order(healpix_order: int) -> int:
    return int(
        sampling.rotation_grid_size(int(healpix_order)) // sampling.rotation_grid_n_in_planes(int(healpix_order))
    )


def _class_direction_rotation_log_prior(state: InitialModelState, healpix_order: int) -> np.ndarray:
    """Return RELION's class-specific direction prior over coarse rotations."""

    n_psi = int(sampling.rotation_grid_n_in_planes(int(healpix_order)))
    n_dir = _n_directions_for_healpix_order(int(healpix_order))
    n_rot = int(n_dir * n_psi)
    pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
    if pdf_direction.shape != (int(state.K), n_dir):
        pdf_direction = np.full((int(state.K), n_dir), 1.0 / float(int(state.K) * n_dir), dtype=np.float64)
    mean_pdf = float(np.mean(pdf_direction))
    if mean_pdf <= 0.0 or not np.isfinite(mean_pdf):
        return np.zeros((int(state.K), n_rot), dtype=np.float32)
    direction_ids = np.arange(n_rot, dtype=np.int64) // n_psi
    values = pdf_direction[:, direction_ids]
    out = np.full(values.shape, -1.0e30, dtype=np.float64)
    positive = values > 0.0
    out[positive] = np.log(values[positive] / mean_pdf)
    return out.astype(np.float32)


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    translation_offsets: np.ndarray,
    sigma_offset_angstrom: float | None = None,
    class_log_priors: np.ndarray | None = None,
) -> DenseInitialModelEstepConfig:
    image_pre_shifts = np.rint(np.asarray(translation_offsets, dtype=np.float32)).astype(np.float32)
    coarse_translations = np.asarray(
        sampling_plan.coarse_translations
        if sampling_plan.coarse_translations is not None
        else sampling_plan.translations,
        dtype=np.float32,
    )
    coarse_prior_translations = np.asarray(
        sampling_plan.coarse_prior_translations
        if sampling_plan.coarse_prior_translations is not None
        else coarse_translations,
        dtype=np.float32,
    )
    # Default σ_offset = 10 Å matches RELION's _rlnSigmaOffsetsAngst at iter000.
    if sigma_offset_angstrom is None:
        sigma_angstrom = opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    else:
        sigma_angstrom = float(sigma_offset_angstrom)
    # relion_translation_prior_center cancels voxel_size scaling in make_relion_translation_log_prior
    # (RELION's mixed-unit pdf_offset; K=2 c2 parity fix 2026-05-08).
    _prior_kwargs = dict(
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=sigma_angstrom,
        centers=relion_translation_prior_center(translation_offsets, float(dataset.voxel_size)),
    )
    coarse_translation_log_prior = _translation_log_prior(coarse_prior_translations, **_prior_kwargs)
    translation_log_prior = _translation_log_prior(sampling_plan.translations, **_prior_kwargs)

    engine_kwargs: dict = {
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": False,
        # VDAM --grad subtracts Frefctf (ml_optimiser.cpp:10092-10105); lifts BPref CC +0.91→+0.996.
        "reconstruction_subtract_projected_reference": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": image_pre_shifts,
        "translation_prior_centers": relion_sigma_offset_prior_center(translation_offsets),
        # RECOVAR_DISABLE_SPARSE_PASS2=1 forces dense path (cuFFT plan OOM at 256²+).
        "sparse_pass2": (
            int(sampling_plan.oversampling) > 0
            and os.environ.get("RECOVAR_DISABLE_SPARSE_PASS2", "") not in ("1", "true", "TRUE")
        ),
    }
    if int(sampling_plan.oversampling) > 0:
        engine_kwargs.update(
            healpix_order=int(sampling_plan.healpix_order),
            oversampling_order=int(sampling_plan.oversampling),
            translation_step=float(sampling_plan.offset_step_px),
            random_perturbation=float(sampling_plan.random_perturbation),
            coarse_translations=coarse_translations,
            particle_diameter_ang=float(opts.particle_diameter),
            return_profile=bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE")),
        )
        if _af := os.environ.get("RECOVAR_ADAPTIVE_FRACTION"):
            engine_kwargs["adaptive_fraction"] = float(_af)
    for env_var, kwarg in (
        ("RECOVAR_USE_FLOAT64_SCORING", "use_float64_scoring"),
        ("RECOVAR_HALF_SPECTRUM_SCORING", "half_spectrum_scoring"),
        ("RECOVAR_SQUARE_WINDOW", "square_window"),
    ):
        if os.environ.get(env_var):
            engine_kwargs[kwarg] = True
    if (_recon_sq := os.environ.get("RECOVAR_RECON_SQUARE_WINDOW")) is not None:
        engine_kwargs["recon_square_window"] = bool(int(_recon_sq))
    if os.environ.get("RECOVAR_DISABLE_SUBTRACT_PROJECTED_REFERENCE"):
        engine_kwargs["reconstruction_subtract_projected_reference"] = False
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = translation_log_prior
    if coarse_translation_log_prior is not None:
        engine_kwargs["coarse_translation_log_prior"] = coarse_translation_log_prior

    return DenseInitialModelEstepConfig(
        noise_variance=noise_variance,
        rotations=sampling_plan.rotations,
        translations=sampling_plan.translations,
        image_batch_size=int(opts.image_batch_size),
        rotation_block_size=int(opts.rotation_block_size),
        padding_factor=int(opts.padding_factor),
        relion_bpref_frame=True,
        relion_projector_frame=True,
        class_log_priors=class_log_priors,
        engine_kwargs=engine_kwargs,
    )


def _native_expectation_step(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    particle_state: NativeParticleState | np.ndarray,
    sampling_state: NativeSamplingState | None = None,
    optics_state: NativeOpticsState | None = None,
):
    if not isinstance(particle_state, NativeParticleState):
        particle_state = NativeParticleState(
            translation_offsets=np.asarray(particle_state, dtype=np.float32).copy(),
            class_assignments=np.zeros(int(dataset.n_images), dtype=np.int32),
            max_posterior=np.zeros(int(dataset.n_images), dtype=np.float32),
            pose_assignments=np.full(int(dataset.n_images), -1, dtype=np.int32),
        )

    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        iteration = max(1, int(state.iter))
        do_grad = _native_initialmodel_do_grad(state, iteration)
        sampling_updated = False
        accuracy_meta = None
        if sampling_state is None:
            sampling_plan = _build_sampling_plan(opts, iteration=iteration)
        else:
            if optics_state is not None and _should_estimate_native_sampling_accuracy(
                iteration=iteration,
                nr_iter=int(state.nr_iter),
                do_grad=do_grad,
            ):
                accuracy_meta = _estimate_native_sampling_accuracy(
                    sampling_state,
                    state,
                    particle_state,
                    optics_state,
                    particle_order=np.asarray(particle_ids, dtype=np.int64),
                    random_seed=int(opts.random_seed),
                    padding_factor=int(opts.padding_factor),
                )
            sampling_updated = _prepare_native_sampling_for_iteration(
                sampling_state,
                state,
                iteration=iteration,
                do_grad=do_grad,
            )
            sampling_plan = _build_sampling_plan(
                opts,
                iteration=iteration,
                sampling_state=sampling_state,
            )
        sigma_offset_angstrom = float(np.sqrt(max(float(state.sigma2_offset), 0.0)))
        current_noise_variance = _noise_variance_from_sigma2(state.sigma2_noise, int(state.ori_size))
        config = _dense_estep_config(
            dataset,
            opts,
            current_noise_variance,
            sampling_plan,
            particle_state.translation_offsets,
            sigma_offset_angstrom=sigma_offset_angstrom,
            class_log_priors=np.zeros(int(state.K), dtype=np.float64),
        )
        config.engine_kwargs["class_rotation_log_prior"] = _class_direction_rotation_log_prior(
            state, int(sampling_plan.healpix_order)
        )
        config.engine_kwargs["debug_iteration"] = iteration
        previous_translations = np.asarray(particle_state.translation_offsets, dtype=np.float32).copy()
        previous_classes = np.asarray(particle_state.class_assignments, dtype=np.int32).copy()
        result = run_dense_initial_model_estep(
            dataset, state, config, particle_ids=particle_ids, halfset_ids=halfset_ids
        )
        result.meta.update(
            random_perturbation=float(sampling_plan.random_perturbation),
            n_rotations=int(sampling_plan.rotations.shape[0]),
            n_translations=int(sampling_plan.translations.shape[0]),
            healpix_order=int(sampling_plan.healpix_order),
            oversampling=int(sampling_plan.oversampling),
            offset_range_px=float(sampling_plan.offset_range_px),
            offset_step_px=float(sampling_plan.offset_step_px),
            offset_range_angstrom=float(sampling_plan.offset_range_angstrom),
            offset_step_angstrom=float(sampling_plan.offset_step_angstrom),
            sigma_offset_angstrom=sigma_offset_angstrom,
            sigma2_offset_before=float(state.sigma2_offset),
        )
        if sampling_state is not None:
            result.meta["sampling_accuracy_estimated"] = accuracy_meta is not None
            if accuracy_meta is not None:
                result.meta.update(accuracy_meta)
            result.meta.update(
                sampling_updated=bool(sampling_updated),
                effective_offset_step_angstrom=float(sampling_state.effective_offset_step_angstrom),
                sampling_acc_rot=float(sampling_state.acc_rot),
                sampling_acc_trans_angstrom=float(sampling_state.acc_trans_angstrom),
                sampling_nr_iter_wo_resol_gain=int(sampling_state.nr_iter_wo_resol_gain),
                sampling_has_fine_enough_angular_sampling=bool(sampling_state.has_fine_enough_angular_sampling),
            )
        _update_particle_state_from_estep_meta(
            particle_state,
            result.meta,
            sampling_plan.translations,
        )
        if sampling_state is not None and _should_record_native_sampling_changes(
            iteration=iteration,
            nr_iter=int(state.nr_iter),
            do_grad=do_grad,
        ):
            _record_native_sampling_assignment_changes(
                sampling_state,
                particle_ids=result.meta.get("selected_particle_ids"),
                previous_translations=previous_translations,
                current_translations=particle_state.translation_offsets,
                previous_classes=previous_classes,
                current_classes=particle_state.class_assignments,
            )
        if sampling_state is not None:
            result.meta["current_changes_optimal_offsets_angstrom"] = float(
                sampling_state.current_changes_optimal_offsets_angstrom
            )
            result.meta["current_changes_optimal_classes"] = float(sampling_state.current_changes_optimal_classes)
        return result.accumulators, result.meta

    return _expectation_step


def _ensure_field(arr: np.ndarray | None, shape: tuple, dtype, fill=0) -> np.ndarray:
    if arr is None or arr.shape != shape:
        return np.full(shape, fill, dtype=dtype) if fill != 0 else np.zeros(shape, dtype=dtype)
    return arr


def _update_particle_state_from_estep_meta(
    particle_state: NativeParticleState,
    meta: dict,
    translations: np.ndarray,
) -> None:
    selected = meta.get("selected_particle_ids")
    if selected is None:
        return
    ids = np.asarray(selected, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return
    N = particle_state.translation_offsets.shape[0]
    if np.any(ids < 0) or np.any(ids >= N):
        raise ValueError("selected_particle_ids contains entries outside the particle state table")
    particle_state.visited = np.zeros(N, dtype=bool)
    particle_state.visited[ids] = True

    if (pose := meta.get("pose_assignments")) is not None:
        assignments = np.asarray(pose, dtype=np.int64).reshape(-1)
        trans = np.asarray(translations, dtype=np.float32)
        translation_ids = np.mod(assignments, int(trans.shape[0]))
        base = np.rint(particle_state.translation_offsets[ids]).astype(np.float32)
        particle_state.translation_offsets[ids] = base + trans[translation_ids, :2]
        particle_state.pose_assignments = _ensure_field(particle_state.pose_assignments, (N,), np.int32, -1)
        particle_state.pose_assignments[ids] = assignments.astype(np.int32, copy=False)

    if (rot := meta.get("best_pose_rotations")) is not None:
        particle_state.best_pose_rotations = _ensure_field(particle_state.best_pose_rotations, (N, 3, 3), np.float32)
        particle_state.best_pose_rotations[ids] = np.asarray(rot, dtype=np.float32)

    if (bt := meta.get("best_pose_translations")) is not None:
        particle_state.best_pose_translations = _ensure_field(particle_state.best_pose_translations, (N, 2), np.float32)
        particle_state.best_pose_translations[ids] = np.asarray(bt, dtype=np.float32)

    if (rid := meta.get("best_pose_rotation_ids")) is not None:
        particle_state.best_pose_rotation_ids = _ensure_field(particle_state.best_pose_rotation_ids, (N,), np.int32, -1)
        particle_state.best_pose_rotation_ids[ids] = np.asarray(rid, dtype=np.int32).reshape(-1)
        particle_state.best_pose_rotation_orders = _ensure_field(
            particle_state.best_pose_rotation_orders, (N,), np.int32, -1
        )
        particle_state.best_pose_rotation_orders[ids] = int(meta.get("healpix_order", 0)) + int(
            meta.get("oversampling", 0)
        )

    if (cls := meta.get("class_assignments")) is not None:
        particle_state.class_assignments[ids] = np.asarray(cls, dtype=np.int32).reshape(-1)

    if (pmax := meta.get("max_posterior_per_image")) is not None:
        particle_state.max_posterior[ids] = np.asarray(pmax, dtype=np.float32).reshape(-1)


def _initial_state_from_particles(
    dataset,
    main_star,
    optics_star,
    opts: NativeInitialModelOptions,
    rotations: np.ndarray,
) -> tuple[InitialModelState, np.ndarray]:
    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _micrograph_sort_order(main_star)
    optics_group_by_particle = _optics_group_indices(main_star)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    if nr_optics_groups != 1:
        raise NotImplementedError("native InitialModel currently supports one optics group")

    Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        _image_sigma2_iter(
            dataset,
            order,
            optics_group_by_particle,
            batch_size=max(1, int(opts.image_batch_size)),
        ),
        ori_size=ori_size,
        pixel_size=pixel_size,
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=int(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        nr_optics_groups=nr_optics_groups,
        minimum_nr_particles=int(opts.sigma2_min_particles),
    )

    bootstrap_count = min(len(order), int(opts.bootstrap_min_particles))
    bootstrap_order = order[:bootstrap_count]
    images = _load_raw_images(dataset, bootstrap_order, batch_size=max(1, int(opts.image_batch_size)))
    sorted_star = main_star.iloc[bootstrap_order]
    voltage, Cs, Q0, pixel_size = _single_optics_scalars(sorted_star, optics_star, dataset)

    iref = compute_bootstrap_iref_via_cpp(
        images=images,
        defU=np.asarray(sorted_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(sorted_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(sorted_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=_phase_shift(sorted_star),
        voltage=voltage,
        Cs=Cs,
        Q0=Q0,
        pixel_size=pixel_size,
        ori_size=ori_size,
        nr_classes=int(opts.nr_classes),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        do_ctf_correction=bool(opts.do_ctf_correction),
        random_seed=int(opts.random_seed),
        padding_factor=int(opts.padding_factor),
        current_size=-1,
        minimum_nr_particles=int(opts.bootstrap_min_particles),
    )

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=_n_directions_for_healpix_order(int(opts.healpix_order)),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    init_sigma_offset_angstrom = (
        opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    )
    state.sigma2_offset = float(init_sigma_offset_angstrom) ** 2
    state.Mavg = Mavg
    # RECOVAR_INITIAL_IREF_OVERRIDE lets a parity caller swap in RELION's
    # iter000 ref directly when isolating E/M-step behavior from bootstrap.
    override_path = os.environ.get("RECOVAR_INITIAL_IREF_OVERRIDE")
    if override_path:
        # Parity hook: load Iref directly. Comma-separated paths for K-class,
        # single path broadcast across K, or a "{k}" template expanded k=1..K.
        from recovar.utils.helpers import load_relion_volume

        K = int(opts.nr_classes)
        paths = [p.strip() for p in override_path.split(",") if p.strip()]
        if len(paths) == 1 and "{k" in paths[0]:
            paths = [paths[0].format(k=k + 1) for k in range(K)]
        if len(paths) not in (1, K):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE expects 1 or K={K} paths, got {len(paths)}")
        vols = np.stack(
            [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
            axis=0,
        )
        if vols.shape[1:] != (ori_size, ori_size, ori_size):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE volume shape {vols.shape[1:]} != {(ori_size,) * 3}")
        state.Iref = np.broadcast_to(vols, (K, ori_size, ori_size, ori_size)).copy() if len(paths) == 1 else vols
    else:
        state.Iref = postprocess_bootstrap_iref_via_cpp(
            iref,
            pixel_size=pixel_size,
            ini_high_ang=float(state.ini_high),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
            do_init_blobs=True,
            is_helical_segment=False,
        )
    state = initialise_data_vs_prior_from_references(
        state,
        nr_particles=len(main_star),
        fix_tau=False,
    )
    return state, optics_group_by_particle


def _class_mrc_paths(output_prefix: str, iteration: int, K: int) -> tuple[str, ...]:
    return tuple(f"{output_prefix}_it{iteration:03d}_class{k + 1:03d}.mrc" for k in range(K))


def _write_model_star(path: str, state: InitialModelState, class_mrcs: tuple[str, ...]) -> None:
    current_resolution_angstrom = (
        1.0 / float(state.current_resolution) if float(state.current_resolution) > 0.0 else float("inf")
    )
    pixel_x_ori = float(state.pixel_size) * float(state.ori_size)
    n_shells = int(state.ori_size) // 2 + 1
    pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)

    lines: list[str] = [
        "# Created by recovar native InitialModel\n",
        "\ndata_model_general\n\n",
        f"_rlnCurrentResolution {current_resolution_angstrom:.12g}\n",
        f"_rlnCurrentImageSize {int(state.current_size)}\n",
        f"_rlnCurrentIteration {int(state.iter)}\n",
        f"_rlnNrClasses {int(state.K)}\n",
        f"_rlnTau2FudgeFactor {float(state.tau2_fudge_factor):.12g}\n",
        f"_rlnAveragePmax {float(state.ave_Pmax):.12g}\n",
        f"_rlnSigmaOffsetsAngst {float(np.sqrt(max(float(state.sigma2_offset), 0.0))):.12g}\n\n",
        "data_model_classes\n\nloop_\n_rlnReferenceImage #1\n_rlnClassDistribution #2\n_rlnEstimatedResolution #3\n",
    ]
    for class_mrc, probability in zip(class_mrcs, np.asarray(state.pdf_class)):
        lines.append(f"{class_mrc} {float(probability):.12g} {current_resolution_angstrom:.12g}\n")

    for k in range(int(state.K)):
        lines.append(
            f"\n\ndata_model_class_{k + 1}\n\nloop_\n"
            "_rlnSpectralIndex #1\n_rlnResolution #2\n_rlnAngstromResolution #3\n"
            "_rlnSsnrMap #4\n_rlnGoldStandardFsc #5\n_rlnFourierCompleteness #6\n"
            "_rlnReferenceSigma2 #7\n_rlnReferenceTau2 #8\n"
        )
        tau2 = np.asarray(state.tau2_class[k], dtype=np.float64)
        dvp = np.asarray(state.data_vs_prior_class[k], dtype=np.float64)
        fsc = np.asarray(state.fsc_halves_class[k], dtype=np.float64)
        sigma2_class = np.asarray(state.sigma2_class[k], dtype=np.float64)
        fourier_coverage = np.asarray(state.fourier_coverage_class[k], dtype=np.float64)
        for shell in range(n_shells):
            resolution = float(shell) / pixel_x_ori
            resolution_angstrom = pixel_x_ori / float(shell) if shell > 0 else 999.0
            lines.append(
                f"{int(shell)} {resolution:.12g} {resolution_angstrom:.12g} "
                f"{float(dvp[shell]):.12g} {float(fsc[shell]):.12g} "
                f"{float(fourier_coverage[shell]):.12g} "
                f"{float(sigma2_class[shell]):.12g} {float(tau2[shell]):.12g}\n"
            )
        if pdf_direction.ndim == 2 and k < pdf_direction.shape[0]:
            lines.append(f"\n\ndata_model_pdf_orient_class_{k + 1}\n\nloop_\n_rlnOrientationDistribution #1\n")
            lines.extend(f"{float(p):.12g}\n" for p in pdf_direction[k])

    lines.append(
        "\n\ndata_model_optics_group_1\n\nloop_\n_rlnSpectralIndex #1\n_rlnResolution #2\n_rlnSigma2Noise #3\n"
    )
    lines.extend(
        f"{int(shell)} 0 {float(sigma2):.12g}\n" for shell, sigma2 in enumerate(np.asarray(state.sigma2_noise)[0])
    )

    with open(path, "w") as f:
        f.writelines(lines)


def _set_star_column(table, column: str, values) -> None:
    target = column
    no_prefix = column[1:] if column.startswith("_") else column
    if target not in table.columns and no_prefix in table.columns:
        target = no_prefix
    table[target] = values


def _format_float_column(values: np.ndarray, precision: int = 6) -> list[str]:
    return [f"{float(value):.{precision}f}" for value in np.asarray(values).reshape(-1)]


def _write_data_star(path: str, main_star, optics_star, dataset, particle_state: NativeParticleState) -> None:
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")

    output_order = _micrograph_sort_order(main_star)
    table = main_star.copy()
    visited = particle_state.visited
    if visited is None:
        visited = (
            np.asarray(particle_state.pose_assignments, dtype=np.int32) >= 0
            if particle_state.pose_assignments is not None
            else np.ones(n_images, dtype=bool)
        )
    visited = np.asarray(visited, dtype=bool).reshape(-1)

    offsets_angstrom = np.asarray(particle_state.translation_offsets, dtype=np.float64) * float(dataset.voxel_size)
    _set_star_column(table, "_rlnOriginXAngst", _format_float_column(offsets_angstrom[:, 0]))
    _set_star_column(table, "_rlnOriginYAngst", _format_float_column(offsets_angstrom[:, 1]))
    if _star_column(table, "_rlnOriginX") is not None or _star_column(table, "_rlnOriginY") is not None:
        offsets_pixels = np.asarray(particle_state.translation_offsets, dtype=np.float64)
        _set_star_column(table, "_rlnOriginX", _format_float_column(offsets_pixels[:, 0]))
        _set_star_column(table, "_rlnOriginY", _format_float_column(offsets_pixels[:, 1]))
    class_numbers = np.zeros(n_images, dtype=np.int32)
    class_numbers[visited] = np.asarray(particle_state.class_assignments, dtype=np.int32)[visited] + 1
    _set_star_column(table, "_rlnClassNumber", class_numbers)
    _set_star_column(table, "_rlnMaxValueProbDistribution", _format_float_column(particle_state.max_posterior))

    has_rotations = particle_state.best_pose_rotation_ids is not None or particle_state.best_pose_rotations is not None
    if has_rotations:

        def _angle(col):
            return table[col].astype(float).to_numpy(copy=True) if col in table else np.zeros(n_images)

        angle_rot, angle_tilt, angle_psi = (_angle(c) for c in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"))
        remaining_rot = visited.copy()
        if particle_state.best_pose_rotations is not None:
            rotations = np.asarray(particle_state.best_pose_rotations, dtype=np.float64)
            valid_matrix = visited & np.any(np.abs(rotations.reshape(n_images, -1)) > 0.0, axis=1)
            if np.any(valid_matrix):
                eulers = np.asarray(R_to_relion(rotations[valid_matrix], degrees=True), dtype=np.float64)
                angle_rot[valid_matrix] = eulers[:, 0]
                angle_tilt[valid_matrix] = eulers[:, 1]
                angle_psi[valid_matrix] = eulers[:, 2]
                remaining_rot[valid_matrix] = False

        if particle_state.best_pose_rotation_ids is not None:
            rotation_ids = np.asarray(particle_state.best_pose_rotation_ids, dtype=np.int64).reshape(-1)
            valid_rot = remaining_rot & (rotation_ids >= 0)
            if np.any(valid_rot):
                rotation_orders = (
                    np.asarray(particle_state.best_pose_rotation_orders, dtype=np.int32).reshape(-1)
                    if particle_state.best_pose_rotation_orders is not None
                    else None
                )
                if rotation_orders is None:
                    max_rotations = int(np.max(rotation_ids[valid_rot])) + 1
                    inferred_order = next(
                        (o for o in range(16) if sampling.rotation_grid_size(o) >= max_rotations), None
                    )
                    if inferred_order is None:
                        raise ValueError(
                            f"cannot infer HEALPix order for max rotation id {int(np.max(rotation_ids[valid_rot]))}"
                        )
                    rotation_orders = np.full(n_images, inferred_order, dtype=np.int32)
                for order in np.unique(rotation_orders[valid_rot]):
                    order = int(order)
                    if order < 0:
                        continue
                    order_mask = valid_rot & (rotation_orders == order)
                    eulers = sampling.get_relion_rotation_grid_eulers(order, rotation_index_order="relion")
                    angle_rot[order_mask] = eulers[rotation_ids[order_mask], 0]
                    angle_tilt[order_mask] = eulers[rotation_ids[order_mask], 1]
                    angle_psi[order_mask] = eulers[rotation_ids[order_mask], 2]
        _set_star_column(table, "_rlnAngleRot", _format_float_column(angle_rot))
        _set_star_column(table, "_rlnAngleTilt", _format_float_column(angle_tilt))
        _set_star_column(table, "_rlnAnglePsi", _format_float_column(angle_psi))

    table = table.iloc[output_order].reset_index(drop=True)
    out_path = Path(path)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    write_star(str(out_path), table, optics_star.copy() if optics_star is not None else None)


def _write_iteration_artifacts(
    output_prefix: str,
    state: InitialModelState,
    iteration: int,
    meta: dict,
    *,
    main_star=None,
    optics_star=None,
    dataset=None,
    particle_state: NativeParticleState | None = None,
) -> None:
    out_dir = _output_dir_from_prefix(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    for k, class_mrc in enumerate(class_mrcs):
        write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    _write_model_star(model_star, state, class_mrcs)
    meta_path = f"{output_prefix}_it{iteration:03d}_recovar_meta.json"
    with open(meta_path, "w") as f:
        json.dump(_json_ready(meta), f, indent=2, sort_keys=True)
    if main_star is not None and dataset is not None and particle_state is not None:
        _write_data_star(
            f"{output_prefix}_it{iteration:03d}_data.star",
            main_star,
            optics_star,
            dataset,
            particle_state,
        )


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_final_outputs(output_prefix: str, state: InitialModelState) -> tuple[str, tuple[str, ...]]:
    iteration = int(state.iter)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    out_dir = _output_dir_from_prefix(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, class_mrc in enumerate(class_mrcs):
        if not os.path.exists(class_mrc):
            write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    final_mrc = _initial_model_mrc_from_prefix(output_prefix)
    best_class = int(np.argmax(np.asarray(state.pdf_class)))
    write_relion_mrc(final_mrc, np.asarray(state.Iref[best_class]), voxel_size=float(state.pixel_size))
    return final_mrc, class_mrcs


def run_native_initial_model(opts: NativeInitialModelOptions) -> NativeInitialModelResult:
    """Run native recovar InitialModel refinement."""

    if opts.nr_classes < 1:
        raise ValueError("nr_classes must be >= 1")
    if opts.nr_iter < 1:
        raise ValueError("nr_iter must be >= 1")
    if opts.padding_factor not in (1, 2):
        raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 or 2 only")
    if opts.run_relion_align_symmetry:
        raise NotImplementedError("native post-run relion_align_symmetry execution is not wired yet")

    main_star, optics_star = read_star(opts.fn_img)
    particle_order = _micrograph_sort_order(main_star)
    dataset = load_dataset(
        opts.fn_img,
        lazy=bool(opts.lazy),
        datadir=opts.datadir,
        strip_prefix=opts.strip_prefix,
    )
    if getattr(dataset, "tilt_series_flag", False):
        raise NotImplementedError("native InitialModel currently supports SPA particle STAR files, not tilt-series")

    _configure_relion_image_mask(dataset, opts)
    optics_state = _native_optics_state(main_star, optics_star, dataset)
    sampling_state = _initial_sampling_state(opts, pixel_size=float(dataset.voxel_size))
    sampling_plan = _build_sampling_plan(opts, iteration=1, sampling_state=sampling_state)
    particle_state = _particle_state_from_star(main_star, dataset)
    state, optics_group_by_particle = _initial_state_from_particles(
        dataset,
        main_star,
        optics_star,
        opts,
        sampling_plan.rotations,
    )
    noise_variance = _noise_variance_from_sigma2(state.sigma2_noise, int(state.ori_size))
    expectation_step = _native_expectation_step(
        dataset,
        opts,
        noise_variance,
        particle_state,
        sampling_state,
        optics_state,
    )
    grad_ini_subset_size, grad_fin_subset_size = default_subset_sizes_for_3d_initial_model(int(dataset.n_images))

    if opts.write_iter_artifacts:
        _output_dir_from_prefix(opts.outputname).mkdir(parents=True, exist_ok=True)
        config_path = f"{opts.outputname}_native_options.json"
        with open(config_path, "w") as f:
            json.dump(asdict(opts), f, indent=2, sort_keys=True)

    if opts.write_iter_artifacts:

        def artifact_sink(current, iteration, meta):
            _write_iteration_artifacts(
                opts.outputname,
                current,
                iteration,
                meta,
                main_star=main_star,
                optics_star=optics_star,
                dataset=dataset,
                particle_state=particle_state,
            )
    else:
        artifact_sink = lambda *args, **kwargs: None

    post_mstep_update = None
    if opts.do_solvent:
        solvent_mask = relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )
        post_mstep_update = lambda current, _iteration, _meta: relion_solvent_flatten_state(current, mask=solvent_mask)

    final_state = run_vdam_iterations(
        state,
        nr_particles=int(dataset.n_images),
        optics_group_by_particle=optics_group_by_particle,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        tau2_fudge_arg=float(opts.tau2_fudge),
        grad_em_iters=DEFAULT_GRAD_EM_ITERS,
        random_seed=int(opts.random_seed),
        rnd_unif_factory=_relion_rnd_unif_factory,
        expectation_step=expectation_step,
        iter_artifact_sink=artifact_sink,
        post_mstep_update=post_mstep_update,
        particle_order=particle_order,
        mu=DEFAULT_GRAD_MU,
        projector_padding_factor=int(opts.padding_factor),
    )
    final_mrc, class_mrcs = _write_final_outputs(opts.outputname, final_state)
    final_model_star = f"{opts.outputname}_it{final_state.iter:03d}_model.star"
    if not os.path.exists(final_model_star):
        _write_model_star(final_model_star, final_state, class_mrcs)
    return NativeInitialModelResult(
        state=final_state,
        output_prefix=opts.outputname,
        final_model_star=final_model_star,
        final_mrc=final_mrc,
        class_mrcs=class_mrcs,
    )


__all__ = [
    "NativeInitialModelOptions",
    "NativeInitialModelResult",
    "run_native_initial_model",
]
