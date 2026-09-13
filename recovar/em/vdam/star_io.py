"""STAR-file and artifact I/O of the native InitialModel driver.

Optics and particle state read from the data STAR, the model and data STAR writers, the
per-iteration artifact bundle and the final outputs live here; ``driver``
orchestrates and imports what it publishes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.data_io.starfile import write_star
from recovar.em import sampling
from recovar.em.vdam.state import InitialModelState, NativeParticleState
from recovar.utils.helpers import R_from_relion, R_to_relion, write_relion_mrc


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

def _relion_star_list_value(text: str, label: str, cast=str):
    """Read one required scalar from a RELION list-style STAR block."""

    import re
    import shlex

    matches = re.findall(rf"(?m)^_{re.escape(label)}\s+(.+?)\s*$", text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one _{label} field, found {len(matches)}")
    tokens = shlex.split(matches[0], comments=False, posix=True)
    if len(tokens) != 1:
        raise ValueError(f"_{label} must contain exactly one scalar token")
    return cast(tokens[0])


def _initial_model_mrc_from_prefix(outputname: str) -> str:
    """Mirror RELION's GUI ``outputname.rstrip("run") + initial_model.mrc``."""

    return outputname.rstrip("run") + "initial_model.mrc"


def _experiment_read_order(main_star) -> np.ndarray:
    """RELION Experiment::read order for bootstrap, noise and subset scheduling."""

    mic_col = _star_column(main_star, "_rlnMicrographName")
    if mic_col is None:
        return np.arange(len(main_star), dtype=np.int64)
    mic_names = mic_col.astype(str).to_numpy()
    return np.asarray(sorted(range(len(mic_names)), key=lambda i: mic_names[i]), dtype=np.int64)


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
    return shifts.astype(np.float64, copy=False)


def _particle_state_from_star(
    main_star,
    dataset,
    *,
    allow_unvisited_class_zero: bool = False,
    nr_classes: int | None = None,
) -> NativeParticleState:
    """Load particle state, optionally accepting RELION's K=1 restart sentinel.

    Fresh production inputs retain the ordinary one-indexed positive-class
    contract.  Native InitialModel continuation STARs use class zero only for
    particles that the stochastic gradient schedule has not visited yet.
    """

    if allow_unvisited_class_zero and nr_classes != 1:
        raise ValueError(
            "unvisited _rlnClassNumber=0 is supported only for a verified K=1 "
            "diagnostic continuation",
        )
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")
    class_col = _star_column(main_star, "_rlnClassNumber")
    if class_col is None:
        if allow_unvisited_class_zero:
            raise ValueError(
                "K=1 diagnostic continuation requires _rlnClassNumber",
            )
        class_numbers = None
        class_assignments = np.zeros(n_images, dtype=np.int32)
    else:
        class_numbers = np.asarray(class_col.astype(int).to_numpy(), dtype=np.int32)
        if allow_unvisited_class_zero:
            if np.any((class_numbers < 0) | (class_numbers > 1)):
                raise ValueError(
                    "K=1 diagnostic continuation _rlnClassNumber values must be 0 or 1",
                )
            class_assignments = np.zeros(n_images, dtype=np.int32)
        else:
            class_assignments = class_numbers - 1
        if not allow_unvisited_class_zero and np.any(class_assignments < 0):
            raise ValueError("_rlnClassNumber values must be one-indexed positive class ids")

    pmax_col = _star_column(main_star, "_rlnMaxValueProbDistribution")
    if pmax_col is None:
        max_posterior = np.zeros(n_images, dtype=np.float32)
        max_posterior_values = None
    else:
        max_posterior_values = np.asarray(
            pmax_col.astype(float).to_numpy(),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(max_posterior_values)):
            raise ValueError("_rlnMaxValueProbDistribution values must be finite")
        max_posterior = max_posterior_values.astype(np.float32)

    if allow_unvisited_class_zero:
        assert class_numbers is not None
        zero_state_evidence: list[np.ndarray] = []
        if max_posterior_values is not None:
            if np.any(max_posterior_values < 0.0):
                raise ValueError(
                    "diagnostic continuation probability values must be non-negative",
                )
            zero_state_evidence.append(max_posterior_values == 0.0)
        significant_col = _star_column(main_star, "_rlnNrOfSignificantSamples")
        if significant_col is not None:
            significant_samples = np.asarray(
                significant_col.astype(float).to_numpy(),
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(significant_samples))
                or np.any(significant_samples < 0.0)
                or np.any(significant_samples != np.floor(significant_samples))
            ):
                raise ValueError(
                    "diagnostic continuation significant-sample counts must be "
                    "finite non-negative integers",
                )
            zero_state_evidence.append(significant_samples == 0.0)
        if not zero_state_evidence:
            raise ValueError(
                "K=1 diagnostic continuation cannot validate unvisited class-zero rows "
                "without posterior or significant-sample state",
            )
        state_is_unvisited = np.logical_and.reduce(zero_state_evidence)
        class_is_unvisited = class_numbers == 0
        if not np.array_equal(class_is_unvisited, state_is_unvisited):
            mismatched_rows = np.flatnonzero(class_is_unvisited != state_is_unvisited)
            raise ValueError(
                "K=1 diagnostic continuation class-zero sentinels disagree with "
                f"unvisited particle state at rows {mismatched_rows[:8].tolist()}",
            )
        visited = ~class_is_unvisited
    else:
        visited = max_posterior > 0.0

    angle_names = ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi")
    angle_columns = tuple(_star_column(main_star, name) for name in angle_names)
    if any(column is not None for column in angle_columns) and not all(column is not None for column in angle_columns):
        missing = [name for name, column in zip(angle_names, angle_columns) if column is None]
        raise ValueError(f"STAR file must provide all Euler-angle columns; missing {', '.join(missing)}")
    best_pose_rotations = None
    source_eulers = None
    if all(column is not None for column in angle_columns):
        eulers = np.stack(
            [np.asarray(column.astype(float).to_numpy(), dtype=np.float64) for column in angle_columns],
            axis=1,
        )
        if not np.all(np.isfinite(eulers)):
            raise ValueError("STAR Euler angles must be finite")
        # RELION keeps the input metadata orientations available before every
        # gradient subset has been visited. Sampling-accuracy estimation uses
        # those orientations, then replaces rows as fresh E-step poses arrive.
        best_pose_rotations = np.asarray(R_from_relion(eulers, degrees=True), dtype=np.float32)
        source_eulers = eulers.copy()
    return NativeParticleState(
        translation_offsets=_image_origin_offsets_pixels_from_star(main_star, dataset),
        class_assignments=class_assignments,
        max_posterior=max_posterior,
        pose_assignments=np.full(n_images, -1, dtype=np.int32),
        best_pose_rotations=best_pose_rotations,
        best_pose_eulers_deg=source_eulers,
        best_pose_eulers_valid=(None if source_eulers is None else np.ones(n_images, dtype=bool)),
        visited=visited,
    )


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


def _initial_model_random_subsets(main_star) -> np.ndarray:
    """Return RELION's one-based pseudo-halfset for every input-table row.

    InitialModel routes ``Experiment`` part ids by ``part_id % 2`` even when
    ordinary split-half refinement is disabled.  ``_experiment_read_order``
    maps those internal part ids to RECOVAR's input-table rows; invert that
    map here so the written data STAR records the same persistent identity.
    """

    order = np.asarray(_experiment_read_order(main_star), dtype=np.int64)
    n_images = len(main_star)
    if (
        order.shape != (n_images,)
        or np.unique(order).size != n_images
        or np.any(order < 0)
        or np.any(order >= n_images)
    ):
        raise ValueError("RELION experiment read order must be a particle-row permutation")
    part_ids = np.empty(n_images, dtype=np.int64)
    part_ids[order] = np.arange(n_images, dtype=np.int64)
    return (part_ids % 2 + 1).astype(np.int32, copy=False)


def _write_data_star(path: str, main_star, optics_star, dataset, particle_state: NativeParticleState) -> None:
    array_rows_token = os.environ.get("RECOVAR_VDAM_STAR_ARRAY_ROWS", "0").strip()
    if array_rows_token not in {"0", "1"}:
        raise ValueError("RECOVAR_VDAM_STAR_ARRAY_ROWS must be 0 or 1")
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")

    output_order = _experiment_read_order(main_star)
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
    _set_star_column(table, "_rlnRandomSubset", _initial_model_random_subsets(main_star))
    _set_star_column(table, "_rlnMaxValueProbDistribution", _format_float_column(particle_state.max_posterior))

    has_rotations = (
        particle_state.best_pose_rotation_ids is not None
        or particle_state.best_pose_rotations is not None
        or particle_state.best_pose_eulers_deg is not None
    )
    if has_rotations:

        def _angle(col):
            return table[col].astype(float).to_numpy(copy=True) if col in table else np.zeros(n_images)

        angle_rot, angle_tilt, angle_psi = (_angle(c) for c in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"))
        remaining_rot = visited.copy()
        if particle_state.best_pose_eulers_deg is not None and particle_state.best_pose_eulers_valid is not None:
            source_valid = remaining_rot & np.asarray(particle_state.best_pose_eulers_valid, dtype=bool)
            eulers = np.asarray(particle_state.best_pose_eulers_deg, dtype=np.float64)[source_valid]
            angle_rot[source_valid], angle_tilt[source_valid], angle_psi[source_valid] = eulers.T
            remaining_rot[source_valid] = False
        if particle_state.best_pose_rotations is not None:
            rotations = np.asarray(particle_state.best_pose_rotations, dtype=np.float64)
            valid_matrix = remaining_rot & np.any(np.abs(rotations.reshape(n_images, -1)) > 0.0, axis=1)
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
    writer_kwargs = {"array_rows": True} if array_rows_token == "1" else {}
    write_star(str(out_path), table, optics_star.copy() if optics_star is not None else None, **writer_kwargs)


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
    profile_artifacts = bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE"))
    artifact_started = time.perf_counter()
    stage_started = artifact_started
    artifact_profile: dict[str, float] = {}

    def _record_artifact_stage(name: str) -> None:
        nonlocal stage_started
        if not profile_artifacts:
            return
        now = time.perf_counter()
        artifact_profile[f"{name}_time_s"] = float(now - stage_started)
        stage_started = now

    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    _record_artifact_stage("setup")
    for k, class_mrc in enumerate(class_mrcs):
        write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    _record_artifact_stage("class_mrc")
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    _write_model_star(model_star, state, class_mrcs)
    _record_artifact_stage("model_star")
    meta_path = f"{output_prefix}_it{iteration:03d}_recovar_meta.json"
    with open(meta_path, "w") as f:
        json.dump(_json_ready(meta), f, indent=2, sort_keys=True)
    _record_artifact_stage("meta_json")
    if main_star is not None and dataset is not None and particle_state is not None:
        _write_data_star(
            f"{output_prefix}_it{iteration:03d}_data.star",
            main_star,
            optics_star,
            dataset,
            particle_state,
        )
    _record_artifact_stage("data_star")
    if profile_artifacts:
        artifact_profile["total_time_s"] = float(time.perf_counter() - artifact_started)
        print(
            f"VDAM iteration {iteration} artifact profile: "
            f"{json.dumps(artifact_profile, sort_keys=True)}",
            flush=True,
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
    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, class_mrc in enumerate(class_mrcs):
        if not os.path.exists(class_mrc):
            write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    final_mrc = _initial_model_mrc_from_prefix(output_prefix)
    best_class = int(np.argmax(np.asarray(state.pdf_class)))
    write_relion_mrc(final_mrc, np.asarray(state.Iref[best_class]), voxel_size=float(state.pixel_size))
    return final_mrc, class_mrcs
