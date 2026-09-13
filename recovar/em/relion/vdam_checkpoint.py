"""Read and validate RELION VDAM checkpoints for diagnostic continuation.

This adapter restores native optimizer, sampling and momentum state. Fresh
VDAM refinement uses its own initialization and does not call this loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.em.vdam.native_options import NativeInitialModelOptions
from recovar.em.vdam.native_sampling import (
    RELION_ORIENTATIONAL_PRIOR_NOPRIOR,
    RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI,
    NativeSamplingState,
)
from recovar.em.vdam.star_io import _relion_star_list_value
from recovar.em.vdam.state import InitialModelState


@dataclass(frozen=True)
class NativeContinuationCheckpoint:
    """Fully materialized native VDAM state for one diagnostic restart."""

    optimiser_star: Path
    model_star: Path
    data_star: Path
    sampling_star: Path
    iteration: int
    state: InitialModelState
    sampling_state: NativeSamplingState
    grad_ini_subset_size: int
    grad_fin_subset_size: int
    grad_ini_frac: float
    grad_fin_frac: float
    grad_suspended_local_searches_iter: int


def _resolve_relion_checkpoint_path(value: str, *, owner: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _read_relion_complex_moment(path: Path, *, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Read RELION's interleaved real/imaginary gradient-moment MRC."""

    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        raw = np.asarray(mrc.data, dtype=np.float32).copy()
    expected_storage_shape = (*expected_shape[:-1], 2 * expected_shape[-1])
    if raw.shape != expected_storage_shape:
        raise ValueError(
            f"RELION gradient moment {path} has shape {raw.shape}; "
            f"expected interleaved shape {expected_storage_shape}"
        )
    if not raw.flags.c_contiguous:
        raw = np.ascontiguousarray(raw)
    values = raw.view(np.complex64).reshape(expected_shape).astype(np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"RELION gradient moment {path} contains non-finite values")
    return values


def _second_pseudo_half_moment_path(first_path: Path, *, nr_classes: int) -> Path:
    """Mirror ``MlModel::readStar``'s filename arithmetic for pseudo-half moment 1."""

    import re

    match = re.fullmatch(r"(.*?)(\d{3})(\.[^.]+)", first_path.name)
    if match is None:
        raise ValueError(
            "native VDAM diagnostic continuation requires a three-digit moment filename, "
            f"got {first_path}"
        )
    second_index = int(match.group(2)) + int(nr_classes)
    return first_path.with_name(f"{match.group(1)}{second_index:03d}{match.group(3)}").resolve(strict=True)


def _load_native_vdam_continuation(
    optimiser_star: str | Path,
    *,
    expected_data_star: str | Path,
    opts: NativeInitialModelOptions,
    dataset,
) -> NativeContinuationCheckpoint:
    """Load every state field RELION consumes at a VDAM continuation boundary.

    This loader is intentionally fail closed and is only used by the bounded
    one-next-iteration profiler.  In particular, it requires native gradient
    moment files rather than reconstructing or approximating momentum.
    """

    import starfile

    from recovar.utils.helpers import load_relion_volume

    optimiser_path = Path(optimiser_star).expanduser().resolve(strict=True)
    optimiser_text = optimiser_path.read_text()
    iteration = _relion_star_list_value(optimiser_text, "rlnCurrentIteration", int)
    nr_iter = _relion_star_list_value(optimiser_text, "rlnNumberOfIterations", int)
    data_path = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnExperimentalDataStarFile"),
        owner=optimiser_path,
    )
    model_path = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnModelStarFile"),
        owner=optimiser_path,
    )
    sampling_path = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnOrientSamplingStarFile"),
        owner=optimiser_path,
    )
    if data_path != Path(expected_data_star).expanduser().resolve(strict=True):
        raise ValueError(
            "--i must be the exact data STAR named by --diagnostic_continue_optimiser: "
            f"{expected_data_star} != {data_path}"
        )
    if nr_iter != int(opts.nr_iter):
        raise ValueError(f"checkpoint nr_iter={nr_iter} differs from requested nr_iter={opts.nr_iter}")
    required_optimizer_values = {
        "rlnDoGradientRefine": 1,
        "rlnDoStochasticGradientDescent": 1,
        "rlnDoSplitRandomHalves": 0,
        "rlnRandomSeed": int(opts.random_seed),
        "rlnAdaptiveOversampleOrder": int(opts.oversampling),
    }
    for label, expected in required_optimizer_values.items():
        observed = _relion_star_list_value(optimiser_text, label, int)
        if observed != expected:
            raise ValueError(f"checkpoint _{label}={observed} differs from requested value {expected}")
    if _relion_star_list_value(optimiser_text, "rlnGradEmIters", int) != int(opts.grad_em_iters):
        raise ValueError("checkpoint grad_em_iters differs from requested schedule")
    if _relion_star_list_value(optimiser_text, "rlnParticleDiameter", float) != float(opts.particle_diameter):
        raise ValueError("checkpoint particle diameter differs from requested value")
    unsupported_subset_modes = {
        "rlnDoFastSubsetOptimisation": _relion_star_list_value(
            optimiser_text,
            "rlnDoFastSubsetOptimisation",
            int,
        ),
        "rlnGradSubsetOrder": _relion_star_list_value(
            optimiser_text,
            "rlnGradSubsetOrder",
            int,
        ),
    }
    enabled_subset_modes = [
        name for name, value in unsupported_subset_modes.items() if int(value) != 0
    ]
    if enabled_subset_modes:
        raise NotImplementedError(
            "diagnostic native VDAM continuation does not support "
            + ", ".join(enabled_subset_modes)
        )
    grad_suspended_local_searches_iter = _relion_star_list_value(
        optimiser_text,
        "rlnGradSuspendLocalSamplingIter",
        int,
    )
    grad_ini_subset_size = _relion_star_list_value(
        optimiser_text,
        "rlnSgdInitialSubsetSize",
        int,
    )
    grad_fin_subset_size = _relion_star_list_value(
        optimiser_text,
        "rlnSgdFinalSubsetSize",
        int,
    )
    grad_ini_frac = _relion_star_list_value(
        optimiser_text,
        "rlnSgdInitialIterationsFraction",
        float,
    )
    grad_fin_frac = _relion_star_list_value(
        optimiser_text,
        "rlnSgdFinalIterationsFraction",
        float,
    )

    model = starfile.read(model_path, always_dict=True)
    general = model.get("model_general")
    classes = model.get("model_classes")
    if not isinstance(general, dict) or classes is None:
        raise ValueError(f"{model_path} lacks native model_general/model_classes tables")

    def _general(name: str, cast=float):
        if name not in general:
            raise ValueError(f"{model_path} lacks {name}")
        return cast(general[name])

    ori_size = _general("rlnOriginalImageSize", int)
    current_size = _general("rlnCurrentImageSize", int)
    pixel_size = _general("rlnPixelSize", float)
    nr_classes = _general("rlnNrClasses", int)
    padding_factor = _general("rlnPaddingFactor", float)
    if nr_classes != int(opts.nr_classes):
        raise ValueError(f"checkpoint K={nr_classes} differs from requested K={opts.nr_classes}")
    if nr_classes != 1:
        raise NotImplementedError("diagnostic native VDAM continuation is currently K=1-only")
    if ori_size != int(dataset.grid_size) or not np.isclose(pixel_size, float(dataset.voxel_size)):
        raise ValueError("checkpoint model geometry differs from the input particle stack")
    if not np.isclose(padding_factor, float(opts.padding_factor)):
        raise ValueError("checkpoint padding factor differs from requested value")
    n_shells = ori_size // 2 + 1
    moment_shape = (ori_size * int(opts.padding_factor),) * 2 + (
        (ori_size * int(opts.padding_factor)) // 2 + 1,
    )

    references = []
    moment1_first = []
    moment1_second = []
    moment2 = []
    pdf_class = []
    tau2 = []
    sigma2 = []
    fsc = []
    coverage = []
    data_vs_prior = []
    direction_priors = []
    for class_index in range(nr_classes):
        row = classes.iloc[class_index]
        reference_path = _resolve_relion_checkpoint_path(str(row["rlnReferenceImage"]), owner=model_path)
        moment1_path = _resolve_relion_checkpoint_path(str(row["rlnGradMoment1"]), owner=model_path)
        moment2_path = _resolve_relion_checkpoint_path(str(row["rlnGradMoment2"]), owner=model_path)
        second_moment1_path = _second_pseudo_half_moment_path(
            moment1_path,
            nr_classes=nr_classes,
        )
        references.append(np.asarray(load_relion_volume(reference_path), dtype=np.float64))
        moment1_first.append(_read_relion_complex_moment(moment1_path, expected_shape=moment_shape))
        moment1_second.append(_read_relion_complex_moment(second_moment1_path, expected_shape=moment_shape))
        moment2.append(_read_relion_complex_moment(moment2_path, expected_shape=moment_shape))
        pdf_class.append(float(row["rlnClassDistribution"]))

        spectrum = model.get(f"model_class_{class_index + 1}")
        direction = model.get(f"model_pdf_orient_class_{class_index + 1}")
        if spectrum is None or direction is None or len(spectrum) != n_shells:
            raise ValueError(f"{model_path} lacks complete class-{class_index + 1} spectra/prior")
        tau2.append(np.asarray(spectrum["rlnReferenceTau2"], dtype=np.float64))
        sigma2.append(np.asarray(spectrum["rlnReferenceSigma2"], dtype=np.float64))
        fsc.append(np.asarray(spectrum["rlnGoldStandardFsc"], dtype=np.float64))
        coverage.append(np.asarray(spectrum["rlnFourierCompleteness"], dtype=np.float64))
        data_vs_prior.append(np.asarray(spectrum["rlnSsnrMap"], dtype=np.float64))
        direction_priors.append(np.asarray(direction["rlnOrientationDistribution"], dtype=np.float64))

    optics_tables = sorted(key for key in model if str(key).startswith("model_optics_group_"))
    if len(optics_tables) != 1:
        raise NotImplementedError("diagnostic native VDAM continuation requires one optics group")
    noise_table = model[optics_tables[0]]
    if len(noise_table) != n_shells:
        raise ValueError("checkpoint sigma2_noise spectrum has the wrong shell count")
    sigma2_noise = np.asarray(noise_table["rlnSigma2Noise"], dtype=np.float64)[None, :]
    current_resolution_angstrom = _general("rlnCurrentResolution", float)
    current_resolution = 1.0 / current_resolution_angstrom
    state = InitialModelState(
        iter=iteration,
        nr_iter=nr_iter,
        K=nr_classes,
        ori_size=ori_size,
        pixel_size=pixel_size,
        pseudo_halfsets=True,
        Iref=np.stack(references, axis=0),
        Igrad1=np.concatenate(
            [np.stack(moment1_first, axis=0), np.stack(moment1_second, axis=0)],
            axis=0,
        ),
        Igrad2=np.stack(moment2, axis=0),
        sigma2_noise=sigma2_noise,
        tau2_class=np.stack(tau2, axis=0),
        sigma2_class=np.stack(sigma2, axis=0),
        fsc_halves_class=np.stack(fsc, axis=0),
        fourier_coverage_class=np.stack(coverage, axis=0),
        data_vs_prior_class=np.stack(data_vs_prior, axis=0),
        pdf_class=np.asarray(pdf_class, dtype=np.float64),
        pdf_direction=np.stack(direction_priors, axis=0),
        sigma2_offset=_general("rlnSigmaOffsetsAngst", float) ** 2,
        current_resolution=current_resolution,
        current_resolution_shell=int(np.floor(current_resolution * pixel_size * ori_size + 0.5)),
        current_size=current_size,
        incr_size=_relion_star_list_value(optimiser_text, "rlnIncrementImageSize", int),
        ave_Pmax=_general("rlnAveragePmax", float),
        has_high_fsc_at_limit=bool(
            _relion_star_list_value(optimiser_text, "rlnHasHighFscAtResolLimit", int)
        ),
        grad_current_stepsize=_relion_star_list_value(
            optimiser_text,
            "rlnGradCurrentStepsize",
            float,
        ),
        tau2_fudge_factor=_general("rlnTau2FudgeFactor", float),
        subset_size=_relion_star_list_value(optimiser_text, "rlnSgdSubsetSize", int),
        has_converged=bool(_relion_star_list_value(optimiser_text, "rlnHasConverged", int)),
        grad_has_converged=bool(
            _relion_star_list_value(optimiser_text, "rlnGradHasConverged", int)
        ),
    )
    for name in (
        "Iref",
        "Igrad1",
        "Igrad2",
        "sigma2_noise",
        "tau2_class",
        "sigma2_class",
        "fsc_halves_class",
        "fourier_coverage_class",
        "data_vs_prior_class",
        "pdf_class",
        "pdf_direction",
    ):
        if not np.all(np.isfinite(np.asarray(getattr(state, name)))):
            raise ValueError(f"checkpoint state field {name} contains non-finite values")

    sampling_text = sampling_path.read_text()
    sampling_state = NativeSamplingState(
        healpix_order=_relion_star_list_value(sampling_text, "rlnHealpixOrder", int),
        adaptive_oversampling=_relion_star_list_value(
            optimiser_text,
            "rlnAdaptiveOversampleOrder",
            int,
        ),
        offset_range_angstrom=_relion_star_list_value(sampling_text, "rlnOffsetRange", float),
        offset_step_angstrom=_relion_star_list_value(sampling_text, "rlnOffsetStep", float),
        offset_range_ori_angstrom=_relion_star_list_value(
            sampling_text,
            "rlnOffsetRangeOriginal",
            float,
        ),
        offset_step_ori_angstrom=_relion_star_list_value(
            sampling_text,
            "rlnOffsetStepOriginal",
            float,
        ),
        pixel_size=pixel_size,
        auto_local_healpix_order=_relion_star_list_value(
            optimiser_text,
            "rlnAutoLocalSearchesHealpixOrder",
            int,
        ),
        acc_rot=_relion_star_list_value(optimiser_text, "rlnOverallAccuracyRotations", float),
        acc_trans_angstrom=_relion_star_list_value(
            optimiser_text,
            "rlnOverallAccuracyTranslationsAngst",
            float,
        ),
        current_changes_optimal_offsets_angstrom=_relion_star_list_value(
            optimiser_text,
            "rlnChangesOptimalOffsets",
            float,
        ),
        current_changes_optimal_orientations=_relion_star_list_value(
            optimiser_text,
            "rlnChangesOptimalOrientations",
            float,
        ),
        current_changes_optimal_classes=_relion_star_list_value(
            optimiser_text,
            "rlnChangesOptimalClasses",
            float,
        ),
        smallest_changes_optimal_offsets_angstrom=_relion_star_list_value(
            optimiser_text,
            "rlnSmallestChangesOffsets",
            float,
        ),
        smallest_changes_optimal_orientations=_relion_star_list_value(
            optimiser_text,
            "rlnSmallestChangesOrientations",
            float,
        ),
        smallest_changes_optimal_classes=_relion_star_list_value(
            optimiser_text,
            "rlnSmallestChangesClasses",
            float,
        ),
        nr_iter_wo_resol_gain=_relion_star_list_value(
            optimiser_text,
            "rlnNumberOfIterWithoutResolutionGain",
            int,
        ),
        nr_iter_wo_large_hidden_variable_changes=_relion_star_list_value(
            optimiser_text,
            "rlnNumberOfIterWithoutChangingAssignments",
            int,
        ),
        # updateCurrentResolution compares against mymodel.current_resolution,
        # not best_resol_thus_far.  The former lives in the model checkpoint.
        last_current_resolution=current_resolution,
        orientational_prior_mode=_general("rlnOrientationalPriorMode", int),
        uniform_local_orientation_prior=(
            _general("rlnOrientationalPriorMode", int) == RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI
            and _general("rlnSigmaPriorRotAngle", float) == 0.0
            and _general("rlnSigmaPriorTiltAngle", float) == 0.0
            and _general("rlnSigmaPriorPsiAngle", float) == 0.0
        ),
    )
    return NativeContinuationCheckpoint(
        optimiser_star=optimiser_path,
        model_star=model_path,
        data_star=data_path,
        sampling_star=sampling_path,
        iteration=iteration,
        state=state,
        sampling_state=sampling_state,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        grad_ini_frac=grad_ini_frac,
        grad_fin_frac=grad_fin_frac,
        grad_suspended_local_searches_iter=grad_suspended_local_searches_iter,
    )


def _validate_continuation_order_replay(checkpoint: NativeContinuationCheckpoint) -> None:
    """Fail closed when native subset-order history is not reconstructible."""

    if (
        int(checkpoint.sampling_state.orientational_prior_mode)
        != RELION_ORIENTATIONAL_PRIOR_NOPRIOR
        or int(checkpoint.grad_suspended_local_searches_iter) != -1
    ):
        raise NotImplementedError(
            "diagnostic native VDAM continuation can reconstruct particle "
            "order only before the first local-search transition"
        )
