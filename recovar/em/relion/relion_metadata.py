"""RELION STAR readers, translation metadata and noise-shell helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from recovar.core import fourier_transform_utils
from recovar.em.helpers.orientation_priors import relion_translation_search_base


def _relion_metadata_translations(
    previous_best_translations, selected_relative_translations, *, dtype: np.dtype = np.float32
):
    """Return RELION-style metadata offsets after selecting relative shifts.

    RELION applies the rounded previous offset to the image before scoring,
    evaluates the search grid as a relative sampled translation, then writes
    ``rounded_old_offset + sampled_translation`` back to metadata. Keeping
    that absolute value is required for the next iteration's pre-shift and
    sigma-offset sufficient statistic.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running a genuine double-precision comparison should pass
    ``np.float64``. RELION's own per-particle offset metadata
    (``exp_metadata``, ``EMDL_ORIENT_ORIGIN_X/Y_ANGSTROM``) is never narrowed
    to float -- ``exp_metadata`` is declared ``MultidimArray<RFLOAT>`` and
    ``EMDL_ORIENT_ORIGIN_X_ANGSTROM`` is registered ``EMDL_DOUBLE``, backed by
    ``std::vector<double>`` in ``MetaDataContainer`` (RELION
    ``src/ml_optimiser.h``, ``src/metadata_label.h``, ``src/metadata_container.h``).
    """
    selected = np.asarray(selected_relative_translations, dtype=dtype)
    base = relion_translation_search_base(previous_best_translations, dtype=dtype)
    if base is None:
        return selected
    return (np.asarray(base, dtype=dtype).reshape(selected.shape) + selected).astype(dtype)


def _relion_half_plane_shell_counts(image_shape):
    """Count RELION's non-redundant FFTW half-plane shell pixels."""
    height, width = int(image_shape[0]), int(image_shape[1])
    n_shells = height // 2 + 1
    counts = np.zeros(n_shells, dtype=np.float64)
    for iy in range(height):
        ky = iy if iy <= height // 2 else iy - height
        for ix in range(width // 2 + 1):
            # RELION excludes redundant jp==0, ip<0 FFTW half-plane entries.
            if ix == 0 and ky < 0:
                continue
            shell = int(np.rint(np.sqrt(float(ky * ky + ix * ix))))
            if shell < n_shells:
                counts[shell] += 1.0
    return counts


def _radial_profile_from_noise_variance(noise_variance, image_shape):
    """Average an image-shaped noise vector into integer radial shells."""
    n_shells = image_shape[0] // 2 + 1
    radial_dist = np.clip(
        fourier_transform_utils.get_grid_of_radial_distances(
            image_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1),
        0,
        n_shells - 1,
    )
    noise_np = np.asarray(noise_variance, dtype=np.float64).reshape(-1)
    radial = np.zeros(n_shells, dtype=np.float64)
    counts = np.zeros(n_shells, dtype=np.float64)
    np.add.at(radial, radial_dist[: noise_np.size], noise_np)
    np.add.at(counts, radial_dist[: noise_np.size], 1.0)
    return radial / np.maximum(counts, 1.0)


def _required_relion_scalar(text, path, name, cast=float):
    """Read the first matching replay scalar, preserving legacy regex semantics."""
    import re

    m = re.search(rf"_{name}\s+(\S+)", text)
    if not m:
        raise ValueError(f"Missing {name} in {path}")
    return cast(m.group(1))


def read_relion_sampling_metadata(sampling_star_path):
    """Read the full set of RELION sampling metadata needed for replay:
    ``(random_perturbation, perturbation_factor, healpix_order, offset_range, offset_step)``.

    ``offset_range`` and ``offset_step`` are in the same units RELION writes
    (Angstroms at scale-0, or as configured). ``healpix_order`` is the order
    RELION actually used at that iter.
    """
    text = open(sampling_star_path).read()

    return dict(
        random_perturbation=_required_relion_scalar(text, sampling_star_path, "rlnSamplingPerturbInstance"),
        perturbation_factor=_required_relion_scalar(text, sampling_star_path, "rlnSamplingPerturbFactor"),
        healpix_order=_required_relion_scalar(text, sampling_star_path, "rlnHealpixOrder", int),
        psi_step=_required_relion_scalar(text, sampling_star_path, "rlnPsiStep"),
        offset_range=_required_relion_scalar(text, sampling_star_path, "rlnOffsetRange"),
        offset_step=_required_relion_scalar(text, sampling_star_path, "rlnOffsetStep"),
    )


def read_relion_model_metadata(model_star_path):
    """Read RELION model star fields needed for replay.

    Returns ``current_image_size`` and ``current_resolution`` from the
    model star file.  These are written by ``updateCurrentResolution`` +
    ``updateImageSizeAndResolutionPointers`` at the start of each RELION
    iteration and stored in the ``data_model_general`` table.

    Local-search replay also needs the orientational prior widths from
    the same table.  Older/minimal fixtures may omit those fields, so the
    sigma values are optional and returned as ``None`` when absent.
    """
    import re

    text = open(model_star_path).read()

    def _grab_optional(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            return None
        return cast(m.group(1))

    return dict(
        current_image_size=_required_relion_scalar(text, model_star_path, "rlnCurrentImageSize", int),
        current_resolution=_required_relion_scalar(text, model_star_path, "rlnCurrentResolution"),
        orientational_prior_mode=_grab_optional("rlnOrientationalPriorMode", int),
        sigma_prior_rot_angle=_grab_optional("rlnSigmaPriorRotAngle"),
        sigma_prior_tilt_angle=_grab_optional("rlnSigmaPriorTiltAngle"),
        sigma_prior_psi_angle=_grab_optional("rlnSigmaPriorPsiAngle"),
    )


def read_relion_optimiser_metadata(optimiser_star_path):
    """Read RELION optimiser fields needed for exact replay control flow."""
    import re

    text = open(optimiser_star_path).read()

    def _grab(name, cast=float, default=None):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            return default
        return cast(m.group(1))

    return dict(
        random_seed=_grab("rlnRandomSeed", int),
        current_iteration=_grab("rlnCurrentIteration", int),
        number_iterations=_grab("rlnNumberOfIterations", int),
        overall_accuracy_rotations=_grab("rlnOverallAccuracyRotations"),
        overall_accuracy_translations_angst=_grab("rlnOverallAccuracyTranslationsAngst"),
        has_converged=_grab("rlnHasConverged", int),
        number_iter_without_resolution_gain=_grab("rlnNumberOfIterWithoutResolutionGain", int),
        number_iter_without_changing_assignments=_grab("rlnNumberOfIterWithoutChangingAssignments", int),
        changes_optimal_orientations=_grab("rlnChangesOptimalOrientations"),
        changes_optimal_offsets=_grab("rlnChangesOptimalOffsets"),
        changes_optimal_classes=_grab("rlnChangesOptimalClasses"),
        smallest_changes_orientations=_grab("rlnSmallestChangesOrientations"),
        smallest_changes_offsets=_grab("rlnSmallestChangesOffsets"),
        smallest_changes_classes=_grab("rlnSmallestChangesClasses"),
        do_correct_ctf=_grab("rlnDoCorrectCtf", int),
        gradient_refine=_grab("rlnDoGradientRefine", int),
        do_grad=_grab("rlnDoStochasticGradientDescent", int),
        grad_em_iters=_grab("rlnGradEmIters", int),
        grad_has_converged=_grab("rlnGradHasConverged", int),
        maximum_significants_arg=_grab("rlnMaximumSignificantPoses", int),
    )


def read_relion_direction_prior(model_star_path, *, dtype=np.float32):
    """Read RELION's saved orientation distribution from ``model.star``."""
    import numpy as np
    import starfile

    data = starfile.read(str(model_star_path))
    if not isinstance(data, dict) or "model_pdf_orient_class_1" not in data:
        raise ValueError(f"Missing model_pdf_orient_class_1 in {model_star_path}")
    df = data["model_pdf_orient_class_1"]
    if "rlnOrientationDistribution" not in df.columns:
        raise ValueError(f"Missing rlnOrientationDistribution in {model_star_path}")
    return np.asarray(df["rlnOrientationDistribution"], dtype=dtype)


def read_relion_direction_priors(model_star_path, n_classes=None, *, dtype=np.float32):
    """Read all RELION per-class orientation distributions from ``model.star``."""
    import re

    import numpy as np
    import starfile

    data = starfile.read(str(model_star_path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected STAR dictionary in {model_star_path}")
    if n_classes is None:
        class_keys = sorted(
            (
                key
                for key in data
                if re.fullmatch(r"model_pdf_orient_class_\d+", str(key))
            ),
            key=lambda key: int(str(key).rsplit("_", 1)[1]),
        )
    else:
        class_keys = [f"model_pdf_orient_class_{idx + 1}" for idx in range(int(n_classes))]
    if not class_keys:
        raise ValueError(f"Missing model_pdf_orient_class_* tables in {model_star_path}")

    priors = []
    for key in class_keys:
        if key not in data:
            raise ValueError(f"Missing {key} in {model_star_path}")
        df = data[key]
        if "rlnOrientationDistribution" not in df.columns:
            raise ValueError(f"Missing rlnOrientationDistribution in {key} of {model_star_path}")
        priors.append(np.asarray(df["rlnOrientationDistribution"], dtype=dtype))
    return np.stack(priors, axis=0)


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


def _load_relion_mask_params(optimiser_star_path):
    """Extract RELION image-mask parameters from an optimiser STAR file."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    particle_match = re.search(r"rlnParticleDiameter\s+([0-9]+(?:\.[0-9]+)?)", text)
    if particle_match is None:
        particle_match = re.search(r"particle_diameter\s+([0-9]+(?:\.[0-9]+)?)", text)

    width_match = re.search(r"rlnWidthMaskEdge\s+([0-9]+(?:\.[0-9]+)?)", text)
    if width_match is None:
        width_match = re.search(r"width_mask_edge\s+([0-9]+(?:\.[0-9]+)?)", text)

    if particle_match is None or width_match is None:
        return None

    return float(particle_match.group(1)), float(width_match.group(1))



def _load_relion_max_significants(optimiser_star_path):
    """Extract RELION's saved maximum-significant-poses argument from an optimiser STAR."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    match = re.search(r"rlnMaximumSignificantPoses\s+(-?[0-9]+)", text)
    if match is None:
        match = re.search(r"maximum_significant_poses\s+(-?[0-9]+)", text)
    if match is None:
        return None
    return int(match.group(1))



def _parse_relion_cli_ini_high(text):
    """Extract a positive RELION ``--ini_high`` value from an optimiser STAR header."""
    cli_line = ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "--" in stripped:
            cli_line = stripped.lstrip("#").strip()
            break
    match = re.search(r"(?:^|\s)--ini_high(?:\s+|=)(\S+)", cli_line)
    if match is None:
        return None
    val = float(match.group(1))
    if val <= 0.0:
        return None
    return val



def _read_relion_mrc_model_pixel_size(path):
    """Read RELION's binary64 sampling rate from MRC cell length/grid size.

    ``mrcfile.voxel_size`` performs the division in float32.  RELION retains
    the float32 header cell length and integer grid size, then divides them in
    ``RFLOAT`` (binary64 in the parity build).  Keeping that division boundary
    matters for marginal first-iteration normalized-CC winners.
    """

    import mrcfile

    with mrcfile.open(path, permissive=False, header_only=True) as handle:
        cell_lengths = np.asarray(
            [handle.header.cella.x, handle.header.cella.y, handle.header.cella.z],
            dtype=np.float64,
        )
        grid_sizes = np.asarray(
            [handle.header.mx, handle.header.my, handle.header.mz],
            dtype=np.int64,
        )
    if np.any(grid_sizes <= 0) or not np.all(np.isfinite(cell_lengths)):
        raise ValueError(f"invalid MRC sampling header: {path}")
    sampling = cell_lengths / grid_sizes.astype(np.float64)
    if np.any(sampling <= 0.0) or not np.allclose(sampling, sampling[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"K=1 RELION model requires isotropic MRC sampling: {path}")
    return float(sampling[0])



def _parse_relion_tau2_fudge(text):
    """Extract RELION's tau2_fudge from a model or optimiser STAR text block.

    ``_rlnTau2FudgeFactor`` (model.star) is the value RELION actually used.
    ``_rlnTau2FudgeArg`` (optimiser.star) is the user's --tau2_fudge CLI
    value, or -1 when the user did not pass --tau2_fudge (RELION binary
    default kicks in: 1.0 for auto-refine, 4.0 for Class3D). Passing -1
    downstream inverts the Wiener regularization (``inv_tau = 1 /
    (pf^3 * tau2_fudge * tau)``) — that produces a corrupt iter-1
    reconstruction and collapses iter-2+ ``ave_Pmax`` even though iter-1
    Pmax is at RELION parity. Prefer ``Factor`` over ``Arg`` and treat
    a non-positive ``Arg`` as "unset" so ``_resolve_tau2_fudge`` falls
    back to the K-class default.
    """
    match = re.search(r"_?rlnTau2FudgeFactor\s+(\S+)", text)
    if match is not None:
        return float(match.group(1))
    match = re.search(r"_?rlnTau2FudgeArg\s+(\S+)", text)
    if match is None:
        return None
    val = float(match.group(1))
    if val <= 0.0:
        return None
    return val



def _load_relion_it000_model_stars(relion_init_dir, n_classes):
    """Load RELION iter-0 model STARs for strict cold-start replay.

    Class3D writes a shared ``run_it000_model.star``. AutoRefine writes
    half-specific ``run_it000_half{1,2}_model.star`` files instead; preserve
    the shared path when present, and fall back to the half pair for K=1.
    """
    import starfile as _starfile

    relion_init_dir = Path(relion_init_dir)
    shared_model_path = relion_init_dir / "run_it000_model.star"
    if shared_model_path.exists():
        model = _starfile.read(str(shared_model_path))
        return {
            "models": [model],
            "model_paths": [shared_model_path],
            "reference_model": model,
            "reference_model_path": shared_model_path,
            "source": "shared",
        }

    half_model_paths = [
        relion_init_dir / "run_it000_half1_model.star",
        relion_init_dir / "run_it000_half2_model.star",
    ]
    if int(n_classes) == 1 and all(path.exists() for path in half_model_paths):
        models = [_starfile.read(str(path)) for path in half_model_paths]
        return {
            "models": models,
            "model_paths": half_model_paths,
            "reference_model": models[0],
            "reference_model_path": half_model_paths[0],
            "source": "half-specific",
        }

    expected = [shared_model_path, *half_model_paths]
    missing = [str(path) for path in expected if not path.exists()]
    raise SystemExit(
        "--relion_init_dir given but no compatible iter-0 model STAR was found; "
        f"missing candidates: {', '.join(missing)}",
    )



def _relion_image_identity(name, *, label: str) -> tuple[int, str]:
    """Return the exact ``(<1-based index>, <stack>)`` RELION image identity."""

    match = re.fullmatch(r"(\d+)@(.+)", str(name))
    if match is None:
        raise ValueError(f"{label} image names must use the '<index>@<stack>' form; got {name!r}")
    return int(match.group(1)), match.group(2)



def _particle_identity_rows(particles, *, label: str) -> dict[tuple[int, str], int]:
    if "rlnImageName" not in particles.columns:
        raise ValueError(f"{label} is missing rlnImageName")
    identities = [
        _relion_image_identity(name, label=label)
        for name in np.asarray(particles["rlnImageName"]).reshape(-1)
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} contains duplicate rlnImageName/stack identities")
    return {identity: row for row, identity in enumerate(identities)}


def relion_do_grad_for_iteration(
    *,
    gradient_refine: bool,
    has_converged: bool,
    iteration: int,
    number_iterations: int,
    grad_em_iters: int,
    do_firstiter_cc: bool,
    grad_has_converged: bool,
) -> bool:
    """Mirror RELION's per-iteration ``do_grad`` decision.

    RELION recomputes this flag at the start of every optimiser iteration.
    Keeping the calculation separate from the serialized
    ``rlnDoStochasticGradientDescent`` value is important: an iteration-0
    optimiser commonly stores zero there even though iteration 1 will run in
    gradient mode.
    """

    if not bool(gradient_refine):
        return False
    return not (
        bool(has_converged)
        or int(iteration) > int(number_iterations) - int(grad_em_iters)
        or (bool(do_firstiter_cc) and int(iteration) == 1)
        or bool(grad_has_converged)
    )


def relion_active_max_significants(
    maximum_significants_arg: int,
    *,
    do_grad: bool,
    n_classes: int,
    reference_dimension: int = 3,
) -> int:
    """Resolve RELION's active coarse significant-pose cap.

    ``rlnMaximumSignificantPoses`` stores the user argument, not necessarily
    the value used by the expectation step.  When that argument is ``-1`` and
    gradient refinement is active, RELION substitutes 5 poses per class for
    2-D references or 100 poses per class for 3-D references.
    """

    maximum_significants_arg = int(maximum_significants_arg)
    n_classes = int(n_classes)
    reference_dimension = int(reference_dimension)
    if n_classes < 1:
        raise ValueError(f"n_classes must be positive, got {n_classes}")
    if reference_dimension not in {2, 3}:
        raise ValueError(
            "reference_dimension must be 2 or 3, "
            f"got {reference_dimension}"
        )
    if maximum_significants_arg != -1:
        return maximum_significants_arg
    if not bool(do_grad):
        return -1
    per_class = 5 if reference_dimension == 2 else 100
    return per_class * n_classes


def resolve_relion_runtime_max_significants(
    *,
    override: int | None,
    optimiser_metadata: dict[str, object],
    target_iteration: int,
    do_firstiter_cc: bool,
    n_classes: int,
    reference_dimension: int = 3,
) -> dict[str, object]:
    """Resolve an optimiser argument into the cap used by one RELION iteration.

    An explicit override is an active-value override, so ``-1`` can still be
    used to request an uncapped diagnostic. Without an override, the function
    mirrors RELION's gradient control flow and automatic per-class cap.
    """

    gradient_refine = bool(optimiser_metadata.get("gradient_refine") or False)
    if gradient_refine:
        number_iterations = optimiser_metadata.get("number_iterations")
        if number_iterations is None:
            raise ValueError(
                "Gradient replay requires rlnNumberOfIterations in the RELION optimiser"
            )
        grad_em_iters = optimiser_metadata.get("grad_em_iters")
        do_grad = relion_do_grad_for_iteration(
            gradient_refine=True,
            has_converged=bool(optimiser_metadata.get("has_converged") or False),
            iteration=int(target_iteration),
            number_iterations=int(number_iterations),
            grad_em_iters=1 if grad_em_iters is None else int(grad_em_iters),
            do_firstiter_cc=bool(do_firstiter_cc),
            grad_has_converged=bool(
                optimiser_metadata.get("grad_has_converged") or False
            ),
        )
    else:
        do_grad = False

    saved_argument = optimiser_metadata.get("maximum_significants_arg")
    if override is not None:
        active = int(override)
        source = "cli_override"
    else:
        argument = -1 if saved_argument is None else int(saved_argument)
        active = relion_active_max_significants(
            argument,
            do_grad=do_grad,
            n_classes=n_classes,
            reference_dimension=reference_dimension,
        )
        source = (
            "relion_gradient_runtime_default"
            if argument == -1 and do_grad
            else "relion_optimiser_argument"
        )

    return {
        "maximum_significants_argument": (
            None if saved_argument is None else int(saved_argument)
        ),
        "active_max_significants": int(active),
        "source": source,
        "gradient_refine": bool(gradient_refine),
        "do_grad": bool(do_grad),
        "target_iteration": int(target_iteration),
    }
