"""Input-STAR pose initialization in matched half-local particle order."""

import argparse
import numpy as np

from recovar.em.relion import relion_metadata


def _load_input_star_previous_best_poses(
    input_particles,
    relion_halfset_particles,
    half1_idx,
    half2_idx,
    *,
    voxel_size: float,
):
    """Load deposited poses in the exact half-local order used by refinement.

    ``half1_idx`` and ``half2_idx`` index the RECOVAR input particle table,
    whereas ``--relion_half_sets`` may be in a different row order.  Bind the
    two tables by full RELION image identity and fail closed if the supplied
    split is stale, incomplete, or inconsistent with the half-local layout.
    Translations are returned in pixels, matching ``ReplayState`` and RELION's
    previous-best-pose convention.
    """

    input_rows = relion_metadata._particle_identity_rows(
        input_particles,
        label="RECOVAR input STAR",
    )
    halfset_rows = relion_metadata._particle_identity_rows(
        relion_halfset_particles,
        label="RELION half-set STAR",
    )
    if set(input_rows) != set(halfset_rows):
        missing = len(set(input_rows) - set(halfset_rows))
        extra = len(set(halfset_rows) - set(input_rows))
        raise ValueError(
            "RELION half-set STAR and RECOVAR input STAR do not contain the "
            "same rlnImageName/stack identities "
            f"(missing={missing}, extra={extra})",
        )

    n_particles = len(input_particles)
    if len(input_rows) != n_particles:
        raise ValueError("RECOVAR input STAR identity count does not match its particle rows")

    half_indices = []
    for label, values in (("half1_idx", half1_idx), ("half2_idx", half2_idx)):
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional, got {indices.shape}")
        if indices.size and (
            int(np.min(indices)) < 0 or int(np.max(indices)) >= n_particles
        ):
            raise ValueError(f"{label} contains an out-of-bounds RECOVAR particle row")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{label} contains duplicate RECOVAR particle rows")
        half_indices.append(indices)
    if np.intersect1d(half_indices[0], half_indices[1]).size:
        raise ValueError("half1_idx and half2_idx overlap")
    combined_indices = np.concatenate(half_indices)
    if combined_indices.size != n_particles or not np.array_equal(
        np.sort(combined_indices),
        np.arange(n_particles, dtype=np.int64),
    ):
        raise ValueError(
            "half1_idx and half2_idx must form an exact partition of the input STAR rows",
        )

    if "rlnRandomSubset" not in relion_halfset_particles.columns:
        raise ValueError("RELION half-set STAR is missing rlnRandomSubset")
    random_subsets = np.asarray(
        relion_halfset_particles["rlnRandomSubset"],
        dtype=np.int64,
    ).reshape(-1)
    if random_subsets.shape != (len(relion_halfset_particles),):
        raise ValueError("RELION half-set rlnRandomSubset has an invalid shape")
    if not np.all(np.isin(random_subsets, (1, 2))):
        raise ValueError("RELION half-set rlnRandomSubset values must be 1 or 2")

    identities_by_input_row = [None] * n_particles
    for identity, row in input_rows.items():
        identities_by_input_row[row] = identity
    for half, indices in enumerate(half_indices, start=1):
        supplied_subsets = np.asarray(
            [random_subsets[halfset_rows[identities_by_input_row[int(row)]]] for row in indices],
            dtype=np.int64,
        )
        if not np.all(supplied_subsets == half):
            bad_rows = indices[supplied_subsets != half]
            raise ValueError(
                f"half{half}_idx disagrees with RELION rlnRandomSubset for "
                f"{bad_rows.size} input rows",
            )

    def _numeric_columns(columns, *, field: str) -> np.ndarray:
        missing = [column for column in columns if column not in input_particles.columns]
        if missing:
            raise ValueError(
                f"RECOVAR input STAR is missing {field} columns: {', '.join(missing)}",
            )
        try:
            values = np.stack(
                [np.asarray(input_particles[column], dtype=np.float64) for column in columns],
                axis=1,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RECOVAR input STAR {field} columns must be numeric") from exc
        expected_shape = (n_particles, len(columns))
        if values.shape != expected_shape:
            raise ValueError(
                f"RECOVAR input STAR {field} array has shape {values.shape}, "
                f"expected {expected_shape}",
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"RECOVAR input STAR {field} values must be finite")
        return values

    eulers = _numeric_columns(
        ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"),
        field="Euler-angle",
    )

    angstrom_columns = ("rlnOriginXAngst", "rlnOriginYAngst")
    pixel_columns = ("rlnOriginX", "rlnOriginY")
    has_angstrom = [column in input_particles.columns for column in angstrom_columns]
    has_pixels = [column in input_particles.columns for column in pixel_columns]
    if any(has_angstrom) and not all(has_angstrom):
        raise ValueError("RECOVAR input STAR must provide both rlnOriginXAngst and rlnOriginYAngst")
    if any(has_pixels) and not all(has_pixels):
        raise ValueError("RECOVAR input STAR must provide both rlnOriginX and rlnOriginY")
    if all(has_angstrom):
        if not np.isfinite(voxel_size) or float(voxel_size) <= 0.0:
            raise ValueError("voxel_size must be positive and finite for Angstrom origins")
        translations = _numeric_columns(
            angstrom_columns,
            field="Angstrom-origin",
        ) / float(voxel_size)
        translation_units = "angstrom"
    elif all(has_pixels):
        translations = _numeric_columns(pixel_columns, field="pixel-origin")
        translation_units = "pixel"
    else:
        translations = np.zeros((n_particles, 2), dtype=np.float64)
        translation_units = "implicit_zero"

    eulers_per_half = [
        np.ascontiguousarray(eulers[indices], dtype=np.float32)
        for indices in half_indices
    ]
    translations_per_half = [
        np.ascontiguousarray(translations[indices], dtype=np.float32)
        for indices in half_indices
    ]
    for half, (half_eulers, half_translations, indices) in enumerate(
        zip(eulers_per_half, translations_per_half, half_indices, strict=True),
        start=1,
    ):
        if half_eulers.shape != (indices.size, 3):
            raise ValueError(f"half-{half} input Euler array has an invalid shape")
        if half_translations.shape != (indices.size, 2):
            raise ValueError(f"half-{half} input translation array has an invalid shape")
        if not np.all(np.isfinite(half_eulers)) or not np.all(np.isfinite(half_translations)):
            raise ValueError(f"half-{half} input poses are not finite after float32 conversion")

    return {
        "iteration": "input_star",
        "previous_best_rotation_eulers": eulers_per_half,
        "previous_best_translations": translations_per_half,
        "translation_units": translation_units,
    }


def _add_initial_pose_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-pose-source",
        choices=("auto", "input-star", "none"),
        default="auto",
        help=(
            "Initial previous-best poses for a fresh K=1 refinement. 'auto' "
            "loads Euler angles and origins from <data_dir>/particles.star when "
            "--relion_half_sets supplies the matched random halves; 'input-star' "
            "requires that production path explicitly; 'none' preserves an "
            "unseeded search. Diagnostic/replay pose sources retain ownership."
        ),
    )


def _resolve_input_star_pose_seed(
    requested_source: str,
    *,
    n_classes: int,
    init_relion_iteration: int,
    has_relion_half_sets: bool,
    has_competing_pose_source: bool,
    diagnostic_single_half: bool,
) -> bool:
    """Resolve the production input-STAR pose seed or reject ambiguous use."""

    source = str(requested_source).strip().lower()
    if source not in {"auto", "input-star", "none"}:
        raise ValueError(f"unsupported initial pose source {requested_source!r}")
    if source == "none":
        return False

    incompatibilities = []
    if int(n_classes) != 1:
        incompatibilities.append("it is K=1-only")
    if int(init_relion_iteration) != 0:
        incompatibilities.append("it requires a fresh --init_relion_iteration 0 run")
    if not bool(has_relion_half_sets):
        incompatibilities.append("it requires --relion_half_sets")
    if bool(has_competing_pose_source):
        incompatibilities.append("a diagnostic/replay pose source already owns initialization")
    if bool(diagnostic_single_half):
        incompatibilities.append("it requires both gold-standard halves")

    if source == "auto":
        return not incompatibilities
    if incompatibilities:
        raise ValueError("--initial-pose-source input-star " + "; ".join(incompatibilities))
    return True


def _kclass_firstiter_translation_seed(
    initial_override,
    *,
    n_classes,
    init_relion_iteration,
):
    """Select only RELION's input origins for a fresh Class3D search.

    RELION Class3D does not center its first global angular search on the
    input orientations, but it does round and apply the input origins before
    taking the image FFT.  Keep those two pieces of state independent: this
    helper deliberately returns translations only and cannot expose the
    orientations, normalization corrections, priors, or noise carried by the
    broader replay override.
    """

    if int(n_classes) <= 1 or int(init_relion_iteration) != 0:
        return None
    if initial_override is None:
        raise ValueError("fresh Class3D translation initialization is missing run_it000 state")
    translations = initial_override.get("previous_best_translations")
    if not isinstance(translations, (list, tuple)) or len(translations) != 2:
        raise ValueError("fresh Class3D translation initialization requires two half arrays")

    selected = []
    for half_index, values in enumerate(translations, start=1):
        if values is None:
            raise ValueError(
                "fresh Class3D translation initialization is missing "
                f"half-{half_index} input origins"
            )
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            # A single all-data Class3D process owns an intentionally empty
            # second accumulator. Generic replay extraction loses the
            # trailing coordinate dimension when indexing that empty half.
            array = np.empty((0, 2), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                "fresh Class3D half-"
                f"{half_index} input origins have shape {array.shape}; expected (N, 2)"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"fresh Class3D half-{half_index} input origins contain non-finite values"
            )
        selected.append(np.ascontiguousarray(array).copy())
    return selected
