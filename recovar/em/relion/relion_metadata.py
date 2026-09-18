"""RELION STAR readers, translation metadata and noise-shell helpers."""

from __future__ import annotations

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


def read_relion_sampling_metadata(sampling_star_path):
    """Read the full set of RELION sampling metadata needed for replay:
    ``(random_perturbation, perturbation_factor, healpix_order, offset_range, offset_step)``.

    ``offset_range`` and ``offset_step`` are in the same units RELION writes
    (Angstroms at scale-0, or as configured). ``healpix_order`` is the order
    RELION actually used at that iter.
    """
    import re

    text = open(sampling_star_path).read()

    def _grab(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            raise ValueError(f"Missing {name} in {sampling_star_path}")
        return cast(m.group(1))

    return dict(
        random_perturbation=_grab("rlnSamplingPerturbInstance"),
        perturbation_factor=_grab("rlnSamplingPerturbFactor"),
        healpix_order=_grab("rlnHealpixOrder", int),
        psi_step=_grab("rlnPsiStep"),
        offset_range=_grab("rlnOffsetRange"),
        offset_step=_grab("rlnOffsetStep"),
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

    def _grab(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            raise ValueError(f"Missing {name} in {model_star_path}")
        return cast(m.group(1))

    def _grab_optional(name, cast=float):
        m = re.search(rf"_{name}\s+(\S+)", text)
        if not m:
            return None
        return cast(m.group(1))

    return dict(
        current_image_size=_grab("rlnCurrentImageSize", int),
        current_resolution=_grab("rlnCurrentResolution"),
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
