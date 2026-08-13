"""RELION parity-replay helpers.

Extracted from ``iteration_loop.py``: replay-iteration index mapping,
per-half float32 normalizers, the ``_RelionHalfInputState`` dataclass
carrying per-half image corrections / scale corrections / direction priors
through the iteration loop, and ``apply_iter_replay_overrides`` which
applies the per-iteration replay state overrides (read from RELION
sampling/model/direction-prior dumps and/or an explicit
``iter_replay_override`` dict) onto the in-flight iteration state.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import (
    healpix_angular_step,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    class_weights_from_direction_prior,
    infer_direction_prior_healpix_order,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    remap_direction_prior_to_healpix_order,
)
from recovar.em.dense_single_volume.mean_helpers import (
    _mean_noise_variance,
    _normalize_noise_variance_per_half,
)
from recovar.em.dense_single_volume.relion_metadata import (
    _radial_profile_from_noise_variance,
)

# Sampling-module symbols (read_relion_*, get_translation_grid) are resolved
# lazily through ``recovar.em.dense_single_volume.iteration_loop`` inside
# ``apply_iter_replay_overrides`` so that test monkeypatches on the
# iteration_loop module surface win without a per-test setattr on
# ``relion_replay``. See tests/unit/test_refine_relion_mode.py:5408.

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelionProjectorReplayState:
    """Exact per-half RELION ``Projector::data`` captured at one boundary.

    The source manifest binds the in-memory arrays to a sealed capture.  The
    iteration loop validates the remaining geometry metadata against the live
    replay before handing these slabs to the production scorer.
    """

    projector_half_by_half: tuple[np.ndarray, np.ndarray]
    projector_r_max_by_half: tuple[int, int]
    current_size: int
    padding_factor: int
    volume_shape: tuple[int, int, int]
    n_classes: int
    source_manifest_sha256: str


def _parse_relion_projector_replay_state(value, *, n_classes: int) -> RelionProjectorReplayState | None:
    """Validate an explicit captured-projector replay override.

    This intentionally accepts one atomic mapping instead of independent
    arrays and scalars.  A partial projector override would otherwise fall
    back to rebuilding some state from the resident half-map and no longer be
    an exact frozen-boundary replay.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("relion_projector_state must be a mapping")
    required = {
        "projector_half_by_half",
        "projector_r_max_by_half",
        "current_size",
        "padding_factor",
        "volume_shape",
        "n_classes",
        "source_manifest_sha256",
    }
    missing = sorted(required.difference(value))
    extra = sorted(set(value).difference(required))
    if missing or extra:
        raise ValueError(
            "relion_projector_state keys must match the version-1 contract exactly; "
            f"missing={missing}, extra={extra}"
        )

    captured_n_classes = int(value["n_classes"])
    if captured_n_classes != int(n_classes) or captured_n_classes <= 0:
        raise ValueError(
            "captured RELION projector class count does not match replay: "
            f"captured={captured_n_classes}, replay={int(n_classes)}"
        )
    current_size = int(value["current_size"])
    padding_factor = int(value["padding_factor"])
    if current_size <= 0 or padding_factor <= 0:
        raise ValueError("captured RELION projector current_size and padding_factor must be positive")

    volume_shape_values = tuple(int(item) for item in value["volume_shape"])
    if len(volume_shape_values) != 3 or any(item <= 0 for item in volume_shape_values):
        raise ValueError("captured RELION projector volume_shape must contain three positive dimensions")

    manifest_sha256 = str(value["source_manifest_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("captured RELION projector source_manifest_sha256 must be 64 lowercase hex digits")

    projectors = value["projector_half_by_half"]
    r_max_values = value["projector_r_max_by_half"]
    if not isinstance(projectors, (list, tuple)) or len(projectors) != 2:
        raise ValueError("captured RELION projector state must contain exactly two half-set slabs")
    if not isinstance(r_max_values, (list, tuple)) or len(r_max_values) != 2:
        raise ValueError("captured RELION projector state must contain exactly two half-set r_max values")

    normalized_projectors = []
    normalized_r_max = []
    for half_idx, (projector, r_max) in enumerate(zip(projectors, r_max_values, strict=True), start=1):
        array = np.asarray(projector)
        if array.dtype != np.dtype(np.complex64):
            raise TypeError(
                f"captured RELION half-{half_idx} projector must be complex64, got {array.dtype}"
            )
        if array.ndim != 4 or int(array.shape[0]) != captured_n_classes:
            raise ValueError(
                "captured RELION projector slabs must have shape "
                f"(n_classes, z, y, x_half); half-{half_idx} has {array.shape}"
            )
        if any(int(size) <= 0 for size in array.shape[1:]):
            raise ValueError(f"captured RELION half-{half_idx} projector has an empty spatial dimension")
        if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
            raise ValueError(f"captured RELION half-{half_idx} projector contains non-finite values")
        r_max_int = int(r_max)
        if r_max_int < 0 or r_max_int >= min(int(size) for size in array.shape[1:]):
            raise ValueError(
                f"captured RELION half-{half_idx} projector r_max={r_max_int} is outside its slab"
            )
        normalized = np.ascontiguousarray(array).copy()
        normalized.setflags(write=False)
        normalized_projectors.append(normalized)
        normalized_r_max.append(r_max_int)

    if normalized_projectors[0].shape != normalized_projectors[1].shape:
        raise ValueError("captured RELION half-set projector slab shapes must match")

    return RelionProjectorReplayState(
        projector_half_by_half=(normalized_projectors[0], normalized_projectors[1]),
        projector_r_max_by_half=(normalized_r_max[0], normalized_r_max[1]),
        current_size=current_size,
        padding_factor=padding_factor,
        volume_shape=volume_shape_values,
        n_classes=captured_n_classes,
        source_manifest_sha256=manifest_sha256,
    )


def read_relion_single_optics_sigma2_noise(model, *, context):
    """Read the sole supported RELION optics-group noise spectrum.

    RELION carries one ``sigma2_noise`` spectrum per optics group. RECOVAR's
    current EM scorer carries only one spectrum per random half, so silently
    selecting optics group 1 would produce incorrect strict-parity results for
    multi-optics data. Fail closed until scoring is optics-group indexed.
    """

    if not isinstance(model, dict):
        return None
    noise_keys = sorted(
        key
        for key, table in model.items()
        if re.fullmatch(r"model_optics_group_\d+", str(key))
        and hasattr(table, "columns")
        and "rlnSigma2Noise" in table.columns
    )
    if len(noise_keys) > 1:
        raise NotImplementedError(
            f"Strict RELION replay does not yet support {len(noise_keys)} optics-group "
            f"sigma2_noise tables in {context}: {noise_keys}"
        )
    if not noise_keys:
        return None
    return np.asarray(model[noise_keys[0]]["rlnSigma2Noise"], dtype=np.float64)


def relion_mpi_process_start_scoring_noise_pair(noise_half1, noise_half2, *, split_random_halves):
    """Return the noise arrays that RELION MPI uses at process-start scoring.

    AutoRefine reads a model for each random subset, but MPI initialisation
    then calls ``initialiseSigma2Noise`` only on follower rank 1 and broadcasts
    that rank's ``mymodel.sigma2_noise`` to every follower. Consequently both
    random subsets score with the half-1 spectrum at process start. Later
    uninterrupted iterations update each follower independently. Class3D has
    one shared model and does not need this emulation.
    """

    # RELION keeps sigma2_noise in RFLOAT and casts only its reciprocal to
    # XFLOAT when constructing Minvsigma2.  Preserve the caller's dtype here:
    # an early float32 cast changes that reciprocal by one ULP on some shells.
    first = np.asarray(noise_half1)
    second = np.asarray(noise_half2)
    if split_random_halves:
        second = first.copy()
    return [first, second]


def _replay_control_model_iteration(init_relion_iteration: int, loop_iteration: int) -> int:
    """Return the RELION model.star index whose control state governs this replay step."""
    return int(init_relion_iteration) + int(loop_iteration) + 1


def _optional_float32_half_pair(values):
    """Return optional per-half arrays normalized to float32."""
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=np.float32) if values[0] is not None else None,
        np.asarray(values[1], dtype=np.float32) if values[1] is not None else None,
    ]


def _optional_int64_half_pair(values):
    """Return optional per-half integer arrays."""
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=np.int64) if values[0] is not None else None,
        np.asarray(values[1], dtype=np.int64) if values[1] is not None else None,
    ]


def _optional_group_count_half_pair(values):
    """Return an optional explicit group cardinality for each half-set."""
    if values is None:
        return [None, None]
    arr = np.asarray(values).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if arr.size != 2:
        raise ValueError(
            "init_group_count must be a scalar or contain exactly two values; "
            f"got shape {np.asarray(values).shape}"
        )
    counts = []
    for value in arr:
        if value is None:
            counts.append(None)
            continue
        count = int(value)
        if count < 0 or float(value) != float(count):
            raise ValueError(f"init_group_count values must be non-negative integers, got {value!r}")
        counts.append(count)
    return counts


def _normalize_sigma_offset_per_half(values):
    """Return a strict two-element float list for half-specific sigma offsets."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            "translation_sigma_angstrom_per_half must contain exactly two values; "
            f"got shape {np.asarray(values).shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("translation_sigma_angstrom_per_half must be finite")
    return [float(arr[0]), float(arr[1])]


def _as_sigma_offset_half_pair(values):
    """Return a scalar or explicit pair as a strict two-half sigma list."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    return _normalize_sigma_offset_per_half(arr)


def _mean_sigma_offset_per_half(values):
    per_half = _normalize_sigma_offset_per_half(values)
    if per_half is None:
        return None
    return float(0.5 * (per_half[0] + per_half[1]))


def _normalize_logged_float32_half_pair(values, *, label: str):
    """Normalize per-half correction arrays and log summary statistics."""
    per_half = _optional_float32_half_pair(values)
    for k, arr in enumerate(per_half):
        if arr is None:
            continue
        if arr.size:
            logger.info(
                "RELION mode: %s half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                label,
                k + 1,
                arr.mean(),
                arr.std(),
                arr.min(),
                arr.max(),
                len(arr),
            )
        else:
            logger.info("RELION mode: %s half-%d: empty", label, k + 1)
    return per_half


@dataclass
class _RelionHalfInputState:
    """Mutable per-half inputs carried across replay and local-search iterations."""

    previous_best_translations: list
    previous_best_rotation_eulers: list
    image_corrections: list
    scale_corrections: list
    group_ids: list
    group_count: list

    @classmethod
    def from_initial_values(
        cls,
        *,
        previous_best_translations,
        previous_best_rotation_eulers,
        image_corrections,
        scale_corrections,
        group_ids=None,
        group_count=None,
    ):
        return cls(
            previous_best_translations=_optional_float32_half_pair(previous_best_translations),
            previous_best_rotation_eulers=_optional_float32_half_pair(previous_best_rotation_eulers),
            image_corrections=_normalize_logged_float32_half_pair(
                image_corrections,
                label="image_corrections",
            ),
            scale_corrections=_normalize_logged_float32_half_pair(
                scale_corrections,
                label="scale_corrections",
            ),
            group_ids=_optional_int64_half_pair(group_ids),
            group_count=_optional_group_count_half_pair(group_count),
        )


def _apply_replay_correction_overrides(*, relion_half_inputs, replay_override) -> list[str]:
    """Apply replay norm/scale state while distinguishing serialized and live scale."""

    replay_image_value = replay_override.get("image_corrections")
    serialized_scale_value = replay_override.get("serialized_scale_corrections")
    scoring_scale_value = replay_override.get("scoring_scale_corrections")
    legacy_scale_value = replay_override.get("scale_corrections")
    if legacy_scale_value is not None:
        if serialized_scale_value is not None or scoring_scale_value is not None:
            raise ValueError(
                "Legacy scale_corrections cannot be combined with serialized_scale_corrections "
                "or scoring_scale_corrections"
            )
        logger.warning(
            "Replay override: scale_corrections is deprecated; treating it as an explicit "
            "scoring scale and, when paired with image_corrections, its source scale"
        )
        scoring_scale_value = legacy_scale_value
        if replay_image_value is not None:
            # Historical callers supplied image and scale as one paired state.
            # Treating the legacy scale as both source and target preserves
            # those exact arrays instead of rescaling against resident state.
            serialized_scale_value = legacy_scale_value

    replay_images = _optional_float32_half_pair(replay_image_value)
    serialized_scales = _optional_float32_half_pair(serialized_scale_value)
    scoring_scales = _optional_float32_half_pair(scoring_scale_value)

    for half_idx in range(2):
        resident_image = relion_half_inputs.image_corrections[half_idx]
        resident_scale = relion_half_inputs.scale_corrections[half_idx]
        override_image = replay_images[half_idx]
        serialized_scale = serialized_scales[half_idx]
        explicit_scoring_scale = scoring_scales[half_idx]

        # A model STAR is leader-serialized provenance, not necessarily the
        # scale resident on the scorer rank. Preserve a live native scale
        # unless an explicit scoring oracle is supplied. A cold start has no
        # resident scale and therefore falls back to the serialized one.
        target_scale = (
            explicit_scoring_scale
            if explicit_scoring_scale is not None
            else resident_scale
            if resident_scale is not None
            else serialized_scale
        )
        base_image = override_image if override_image is not None else resident_image
        base_scale = (
            serialized_scale
            if override_image is not None and serialized_scale is not None
            else resident_scale
        )

        if target_scale is not None:
            target_scale = np.asarray(target_scale, dtype=np.float32)
            if not np.all(np.isfinite(target_scale)) or np.any(target_scale <= 0.0):
                raise ValueError("scoring scale corrections must be finite and positive")
        if base_scale is not None:
            base_scale = np.asarray(base_scale, dtype=np.float32)
            if not np.all(np.isfinite(base_scale)) or np.any(base_scale <= 0.0):
                raise ValueError("source scale corrections must be finite and positive")
        if base_image is not None:
            base_image = np.asarray(base_image, dtype=np.float32)
            if target_scale is not None and base_scale is None:
                raise ValueError("Cannot preserve image_corrections/scale without a source scale")
            if target_scale is not None and base_scale is not None:
                if base_image.shape != base_scale.shape or base_image.shape != target_scale.shape:
                    raise ValueError("image, source-scale, and scoring-scale corrections must have matching shapes")
                base_image = base_image * (target_scale / base_scale)
            relion_half_inputs.image_corrections[half_idx] = base_image
        if explicit_scoring_scale is not None or resident_scale is None:
            relion_half_inputs.scale_corrections[half_idx] = target_scale

    applied_fields = []
    if replay_image_value is not None:
        applied_fields.append("image_corrections")
        logger.info("Replay override: image_corrections <- norm state rescaled to live scoring scale")
    if serialized_scale_value is not None and legacy_scale_value is None:
        applied_fields.append("serialized_scale_corrections")
        logger.info("Replay provenance: serialized_scale_corrections recorded; resident scoring scale preserved")
    if scoring_scale_value is not None:
        applied_fields.append("scale_corrections" if legacy_scale_value is not None else "scoring_scale_corrections")
        logger.info("Replay override: scoring scale corrections <- explicit scorer oracle")
    return applied_fields


@dataclass
class ReplayOverrideResult:
    """Iteration-state values touched by replay overrides.

    ``state``, ``relion_half_inputs``, and the four ``*_direction_prior_*``
    lists are mutated in place by ``apply_iter_replay_overrides`` (they are
    object/list references) and do not appear in this result. Scalars and
    array refs that need to be reassigned by the caller appear here.
    """

    cs: int
    prior_translations: Any  # jnp.ndarray or None — used downstream by local-search prior
    previous_best_rotations: list
    noise_variance_per_half: list
    noise_variance: Any
    previous_noise_radial_per_half: list
    previous_noise_radial: Any
    current_sigma_offset_angstrom: float
    replay_meta: dict | None  # parsed sampling.star (or None); used downstream by perturbation apply
    current_sigma_offset_angstrom_per_half: list[float] | None = None
    class_weights: np.ndarray | None = None
    relion_projector_state: RelionProjectorReplayState | None = None


def apply_iter_replay_overrides(
    *,
    iter_replay_override: dict | None,
    perturb_replay_relion_dir: str | None,
    perturb_replay_relion_prefix: str = "run",
    init_relion_iteration: int,
    iteration: int,
    state,
    cs: int,
    cryo,
    k_class_enabled: bool,
    n_classes: int,
    relion_half_inputs: _RelionHalfInputState,
    previous_best_rotations: list,
    noise_variance_per_half: list,
    noise_variance,
    previous_noise_radial_per_half: list,
    previous_noise_radial,
    current_sigma_offset_angstrom: float,
    current_sigma_offset_angstrom_per_half: list[float] | None = None,
    class_direction_prior_per_half: list,
    class_direction_prior_order_per_half: list,
    global_direction_prior_per_half: list,
    global_direction_prior_order_per_half: list,
    preserve_existing_direction_prior: bool = False,
    sealed_sampling_state: dict | None = None,
) -> ReplayOverrideResult:
    """Apply per-iteration replay overrides to the in-flight iteration state.

    See ``docs/math/em_parity_program.md`` under the 2026-07-15 targeted
    posterior discriminators for the serialized-versus-runtime scale contract.

    Mutates ``state``, ``relion_half_inputs``, and the four direction-prior
    lists in place. Returns explicit new values for everything else.

    Two override sources, applied in order:

    1. ``perturb_replay_relion_dir``: read RELION's per-iter sampling.star +
       (control) model.star + (previous-iter) half-model.star, override
       healpix order, local-search activation, sigma priors, translation
       range/step, current_size, and direction priors.
    2. ``iter_replay_override`` dict: explicit overrides for sigma_offset,
       previous-best poses, image corrections, serialized/scoring scale
       corrections, noise variance, direction priors, and a sealed exact
       per-half RELION projector state.
    """

    # Resolve sampling-module helpers through iteration_loop so test
    # monkeypatches (``monkeypatch.setattr(refine_mod, "read_relion_*", ...)``)
    # win without monkeypatching this module too. Import is lazy to avoid a
    # circular import at module-load time (iteration_loop imports this module).
    from recovar.em.dense_single_volume import iteration_loop as _il

    _replay_prior_translations = None
    _model_star = None
    _model_meta = None
    _replay_meta = None
    _replay_class_weights = None
    _replay_projector_state = None
    _current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
        current_sigma_offset_angstrom
        if current_sigma_offset_angstrom_per_half is None
        else current_sigma_offset_angstrom_per_half
    )

    if sealed_sampling_state is not None:
        if int(iteration) != 0:
            raise ValueError("sealed frozen-boundary sampling currently owns exactly one iteration")
        _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
        _relion_hp = int(sealed_sampling_state["healpix_order_original"])
        if _relion_hp > int(state.max_healpix_order):
            raise ValueError(
                "sealed sampling HEALPix order exceeds runtime maximum: "
                f"sealed={_relion_hp} max={state.max_healpix_order}"
            )
        state.healpix_order = _relion_hp
        state.do_local_search = bool(state.healpix_order >= state.auto_local_healpix_order)
        state.sigma_rot = np.deg2rad(float(sealed_sampling_state["sigma_rot_deg"]))
        state.sigma_psi = np.deg2rad(float(sealed_sampling_state["sigma_psi_deg"]))
        state.translation_range = float(sealed_sampling_state["offset_range_angstrom"]) / _px
        state.translation_step = float(sealed_sampling_state["offset_step_angstrom"]) / _px
        sealed_x = np.asarray(sealed_sampling_state["translations_x_angstrom"], dtype=np.float64)
        sealed_y = np.asarray(sealed_sampling_state["translations_y_angstrom"], dtype=np.float64)
        _replay_prior_translations = jnp.asarray(
            np.stack([sealed_x / _px, sealed_y / _px], axis=1).astype(np.float32),
            dtype=jnp.float32,
        )
        cs = int(sealed_sampling_state["current_size"])
        _replay_meta = {
            "healpix_order": _relion_hp,
            "psi_step": float(sealed_sampling_state["psi_step_deg"]),
            "offset_range": float(sealed_sampling_state["offset_range_angstrom"]),
            "offset_step": float(sealed_sampling_state["offset_step_angstrom"]),
            "perturbation_factor": float(sealed_sampling_state["perturbation_factor"]),
            "random_perturbation": float(sealed_sampling_state["random_perturbation"]),
            "sealed_v3": True,
        }
        logger.info(
            "Frozen-boundary v3 owns sampling: consumer_iter=%d hp=%d current/coarse=%d/%d "
            "translations=%d rp=%+.12g",
            int(sealed_sampling_state["consumer_relion_iteration"]),
            _relion_hp,
            cs,
            int(sealed_sampling_state["coarse_size"]),
            int(_replay_prior_translations.shape[0]),
            float(sealed_sampling_state["random_perturbation"]),
        )
    elif perturb_replay_relion_dir is not None:
        _star = os.path.join(
            perturb_replay_relion_dir,
            f"{perturb_replay_relion_prefix}_it{init_relion_iteration + iteration + 1:03d}_sampling.star",
        )
        _replay_meta = _il.read_relion_sampling_metadata(_star)
        _relion_hp = int(_replay_meta["healpix_order"])
        _relion_psi_step_deg = float(_replay_meta.get("psi_step", healpix_angular_step(_relion_hp)))
        # RELION stores offset_{range,step} in Angstroms; convert to px.
        _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
        _relion_offset_range = float(_replay_meta["offset_range"]) / _px
        _relion_offset_step = float(_replay_meta["offset_step"]) / _px
        _replay_prior_translations_np = _il._translation_grid_for_class_count(
            _relion_offset_range,
            _relion_offset_step,
            n_classes=n_classes,
        ).astype(np.float32)
        _state_prior_translations = _il._translation_grid_for_class_count(
            float(state.translation_range),
            float(state.translation_step),
            n_classes=n_classes,
        ).astype(np.float32)
        _translation_grid_differs = _state_prior_translations.shape != _replay_prior_translations_np.shape
        if not _translation_grid_differs:
            _translation_grid_differs = not np.allclose(
                _state_prior_translations,
                _replay_prior_translations_np,
                rtol=0.0,
                atol=1e-6,
            )
        _translation_params_differ = (
            abs(float(state.translation_range) - _relion_offset_range) > 1e-6
            or abs(float(state.translation_step) - _relion_offset_step) > 1e-6
        )
        if _translation_grid_differs and not _translation_params_differ:
            logger.info(
                "Replay override: preserving current translation grid for sub-tolerance "
                "RELION replay rounding: range %.9g -> %.9g px, step %.9g -> %.9g px "
                "(translation grid n=%d vs rounded n=%d)",
                float(state.translation_range),
                _relion_offset_range,
                float(state.translation_step),
                _relion_offset_step,
                int(_state_prior_translations.shape[0]),
                int(_replay_prior_translations_np.shape[0]),
            )
            _replay_prior_translations_np = _state_prior_translations
        _replay_prior_translations = jnp.array(_replay_prior_translations_np)
        _capped_hp = min(_relion_hp, state.max_healpix_order)
        if state.healpix_order != _capped_hp:
            if _capped_hp < _relion_hp:
                logger.info(
                    "Replay override: healpix_order %d -> %d (RELION %d capped by max_healpix_order=%d, from %s)",
                    state.healpix_order,
                    _capped_hp,
                    _relion_hp,
                    state.max_healpix_order,
                    _star,
                )
            else:
                logger.info(
                    "Replay override: healpix_order %d -> %d (from %s)",
                    state.healpix_order,
                    _capped_hp,
                    _star,
                )
            state.healpix_order = _capped_hp
        _replay_do_local = bool(state.healpix_order >= state.auto_local_healpix_order)
        if state.do_local_search != _replay_do_local:
            logger.info(
                "Replay override: local_search %s -> %s (healpix_order=%d, auto_local_healpix_order=%d)",
                state.do_local_search,
                _replay_do_local,
                state.healpix_order,
                state.auto_local_healpix_order,
            )
            state.do_local_search = _replay_do_local
            if _replay_do_local:
                state.sigma_rot = 0.0
                state.sigma_psi = 0.0
        # RELION's run_itNNN_model.star is written after iteration N, but its
        # current_size and local-prior sigma fields are the controls used by
        # that same iteration's E-step.  Other fields in the same file, such
        # as current resolution and average Pmax, are post-iteration state.
        # Sampling perturbation uses the same N suffix.
        # Reuse it for both current_size and local-prior sigmas.
        _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
        _model_star_candidates = [
            os.path.join(
                perturb_replay_relion_dir,
                f"{perturb_replay_relion_prefix}_it{_cs_iter:03d}_half1_model.star",
            ),
            os.path.join(
                perturb_replay_relion_dir,
                f"{perturb_replay_relion_prefix}_it{_cs_iter:03d}_model.star",
            ),
        ]
        _model_star = next((path for path in _model_star_candidates if os.path.exists(path)), None)
        if _model_star is not None:
            _model_meta = _il.read_relion_model_metadata(_model_star)
        if _replay_do_local:
            _relion_sigma_rot_deg = None
            _relion_sigma_psi_deg = None
            if _model_meta is not None:
                _sigma_rot_deg = _model_meta.get("sigma_prior_rot_angle")
                _sigma_tilt_deg = _model_meta.get("sigma_prior_tilt_angle")
                _sigma_psi_deg = _model_meta.get("sigma_prior_psi_angle")
                _dir_candidates = [
                    float(value)
                    for value in (_sigma_rot_deg, _sigma_tilt_deg)
                    if value is not None and float(value) > 0.0
                ]
                if _dir_candidates:
                    _relion_sigma_rot_deg = max(_dir_candidates)
                if _sigma_psi_deg is not None and float(_sigma_psi_deg) > 0.0:
                    _relion_sigma_psi_deg = float(_sigma_psi_deg)
            if _relion_sigma_rot_deg is None:
                _relion_sigma_rot_deg = _relion_psi_step_deg
                logger.info(
                    "Replay override: model local prior sigma missing; falling back to RELION psi_step %.3f deg",
                    _relion_psi_step_deg,
                )
            if _relion_sigma_psi_deg is None:
                _relion_sigma_psi_deg = _relion_sigma_rot_deg
            _relion_sigma_rot_rad = np.deg2rad(_relion_sigma_rot_deg)
            _relion_sigma_psi_rad = np.deg2rad(_relion_sigma_psi_deg)
            if (
                abs(float(state.sigma_rot) - _relion_sigma_rot_rad) > 1e-8
                or abs(float(state.sigma_psi) - _relion_sigma_psi_rad) > 1e-8
            ):
                logger.info(
                    "Replay override: local prior sigma %.3f/%.3f deg -> %.3f/%.3f deg (from %s)",
                    float(np.rad2deg(state.sigma_rot)),
                    float(np.rad2deg(state.sigma_psi)),
                    _relion_sigma_rot_deg,
                    _relion_sigma_psi_deg,
                    _model_star if _model_star is not None else _star,
                )
            state.sigma_rot = _relion_sigma_rot_rad
            state.sigma_psi = _relion_sigma_psi_rad
        if _translation_params_differ:
            logger.info(
                "Replay override: translation_range %.9g -> %.9g px, step %.9g -> %.9g px "
                "(translation grid n=%d -> %d)",
                float(state.translation_range),
                _relion_offset_range,
                float(state.translation_step),
                _relion_offset_step,
                int(_state_prior_translations.shape[0]),
                int(_replay_prior_translations_np.shape[0]),
            )
            state.translation_range = _relion_offset_range
            state.translation_step = _relion_offset_step

        # Override current_size from the RELION model star for the replayed
        # iteration's E-step controls.
        if _model_meta is not None:
            _relion_cs = int(_model_meta["current_image_size"])
            if _relion_cs <= 0:
                logger.info(
                    "Replay override: ignoring non-positive current_size=%d from %s",
                    _relion_cs,
                    _model_star,
                )
            elif cs != _relion_cs:
                logger.info(
                    "Replay override: current_size %d -> %d (from %s)",
                    cs,
                    _relion_cs,
                    _model_star,
                )
                cs = _relion_cs

        if iteration > 0 and not preserve_existing_direction_prior:
            _prior_iter = init_relion_iteration + iteration
            if iter_replay_override is None or iter_replay_override.get("direction_prior") is None:
                for _half_idx in range(2):
                    _prior_star = os.path.join(
                        perturb_replay_relion_dir,
                        f"{perturb_replay_relion_prefix}_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                    )
                    if not os.path.exists(_prior_star):
                        if not k_class_enabled:
                            continue
                        # Class3D writes one shared model.star rather than
                        # auto-refine-style half-model STAR files.  During
                        # strict replay, use that shared direction prior for
                        # both RECOVAR halfsets.
                        _prior_star = os.path.join(
                            perturb_replay_relion_dir,
                            f"{perturb_replay_relion_prefix}_it{_prior_iter:03d}_model.star",
                        )
                        if not os.path.exists(_prior_star):
                            continue
                    _relion_direction_prior = (
                        _il.read_relion_direction_priors(_prior_star, n_classes)
                        if k_class_enabled
                        else _il.read_relion_direction_prior(_prior_star)
                    )
                    if k_class_enabled:
                        inferred_weights = class_weights_from_direction_prior(_relion_direction_prior, n_classes)
                        if inferred_weights is not None:
                            _replay_class_weights = inferred_weights
                    _relion_direction_prior_order = infer_direction_prior_healpix_order(
                        _relion_direction_prior[0] if k_class_enabled else _relion_direction_prior
                    )
                    if _relion_direction_prior_order != state.healpix_order:
                        logger.info(
                            "Replay override: remapping half-%d direction prior from healpix_order=%d to %d",
                            _half_idx + 1,
                            _relion_direction_prior_order,
                            state.healpix_order,
                        )
                        if k_class_enabled:
                            _relion_direction_prior = np.stack(
                                [
                                    remap_direction_prior_to_healpix_order(
                                        _relion_direction_prior[class_idx],
                                        _relion_direction_prior_order,
                                        state.healpix_order,
                                    )
                                    for class_idx in range(n_classes)
                                ],
                                axis=0,
                            )
                        else:
                            _relion_direction_prior = remap_direction_prior_to_healpix_order(
                                _relion_direction_prior,
                                _relion_direction_prior_order,
                                state.healpix_order,
                            )
                        _relion_direction_prior_order = state.healpix_order
                    if k_class_enabled:
                        class_direction_prior_per_half[_half_idx] = normalize_class_direction_prior_per_half(
                            [_relion_direction_prior, None] if _half_idx == 0 else [None, _relion_direction_prior],
                            n_classes,
                        )[_half_idx]
                        class_direction_prior_order_per_half[_half_idx] = _relion_direction_prior_order
                        logger.info(
                            "Replay override: class direction prior half-%d <- %s (%d classes, %d directions)",
                            _half_idx + 1,
                            _prior_star,
                            class_direction_prior_per_half[_half_idx].shape[0],
                            class_direction_prior_per_half[_half_idx].shape[1],
                        )
                    else:
                        global_direction_prior_per_half[_half_idx] = _relion_direction_prior
                        global_direction_prior_order_per_half[_half_idx] = _relion_direction_prior_order
                        logger.info(
                            "Replay override: direction prior half-%d <- %s (%d directions, range=[%.6f, %.6f], zeros=%d)",
                            _half_idx + 1,
                            _prior_star,
                            len(_relion_direction_prior),
                            float(_relion_direction_prior.min()),
                            float(_relion_direction_prior.max()),
                            int(np.sum(_relion_direction_prior == 0)),
                        )
        elif preserve_existing_direction_prior:
            logger.info(
                "Replay control: preserving sealed existing direction priors; "
                "external model prior reload suppressed"
            )

    if iter_replay_override is not None:
        _replay_projector_state = _parse_relion_projector_replay_state(
            iter_replay_override.get("relion_projector_state"),
            n_classes=n_classes,
        )
        if _replay_projector_state is not None:
            logger.info(
                "Replay override: exact RELION Projector::data <- manifest %s",
                _replay_projector_state.source_manifest_sha256,
            )
        _replay_sigma_per_half = iter_replay_override.get("translation_sigma_angstrom_per_half")
        if _replay_sigma_per_half is not None:
            _current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(_replay_sigma_per_half)
            current_sigma_offset_angstrom = float(
                0.5
                * (
                    _current_sigma_offset_angstrom_per_half[0]
                    + _current_sigma_offset_angstrom_per_half[1]
                )
            )
            logger.info(
                "Replay override: sigma_offset <- half1 %.4f A, half2 %.4f A, mean %.4f A (iter=%d)",
                _current_sigma_offset_angstrom_per_half[0],
                _current_sigma_offset_angstrom_per_half[1],
                current_sigma_offset_angstrom,
                iteration + 1,
            )
        _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
        if _replay_sigma is not None and _replay_sigma_per_half is None:
            current_sigma_offset_angstrom = float(_replay_sigma)
            _current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(_replay_sigma)
            logger.info(
                "Replay override: sigma_offset <- %.4f A (iter=%d)",
                current_sigma_offset_angstrom,
                iteration + 1,
            )
        _replay_prev_trans = iter_replay_override.get("previous_best_translations")
        if _replay_prev_trans is not None:
            relion_half_inputs.previous_best_translations = _optional_float32_half_pair(_replay_prev_trans)
            logger.info(
                "Replay override: previous_best_translations <- half1=%s half2=%s",
                "set" if relion_half_inputs.previous_best_translations[0] is not None else "none",
                "set" if relion_half_inputs.previous_best_translations[1] is not None else "none",
            )
        _replay_prev_rots = iter_replay_override.get("previous_best_rotations")
        if _replay_prev_rots is not None:
            previous_best_rotations = _optional_float32_half_pair(_replay_prev_rots)
            logger.info(
                "Replay override: previous_best_rotations <- half1=%s half2=%s",
                "set" if previous_best_rotations[0] is not None else "none",
                "set" if previous_best_rotations[1] is not None else "none",
            )
        _replay_prev_eulers = iter_replay_override.get("previous_best_rotation_eulers")
        if _replay_prev_eulers is not None:
            relion_half_inputs.previous_best_rotation_eulers = _optional_float32_half_pair(_replay_prev_eulers)
            logger.info(
                "Replay override: previous_best_rotation_eulers <- half1=%s half2=%s",
                "set" if relion_half_inputs.previous_best_rotation_eulers[0] is not None else "none",
                "set" if relion_half_inputs.previous_best_rotation_eulers[1] is not None else "none",
            )
        _apply_replay_correction_overrides(
            relion_half_inputs=relion_half_inputs,
            replay_override=iter_replay_override,
        )
        _replay_noise = iter_replay_override.get("noise_variance")
        if _replay_noise is not None:
            noise_variance_per_half = _normalize_noise_variance_per_half(_replay_noise, n_halves=2)
            noise_variance = _mean_noise_variance(noise_variance_per_half)
            previous_noise_radial_per_half = [
                _radial_profile_from_noise_variance(noise_k, cryo.image_shape) for noise_k in noise_variance_per_half
            ]
            previous_noise_radial = jnp.asarray(
                np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
                dtype=jnp.float32,
            )
            logger.info("Replay override: sigma2_noise <- per-half model.star arrays")
        _replay_dir_prior = iter_replay_override.get("direction_prior")
        if _replay_dir_prior is not None:
            if k_class_enabled:
                inferred_weights = class_weights_from_direction_prior(_replay_dir_prior, n_classes)
                if inferred_weights is not None:
                    _replay_class_weights = inferred_weights
            if k_class_enabled:
                replay_priors = normalize_class_direction_prior_per_half(_replay_dir_prior, n_classes)
            else:
                replay_priors = normalize_direction_prior_per_half(_replay_dir_prior)
            for _half_idx in range(2):
                if replay_priors[_half_idx] is None:
                    continue
                prior_k = np.asarray(replay_priors[_half_idx], dtype=np.float32)
                prior_order_k = infer_direction_prior_healpix_order(prior_k[0] if k_class_enabled else prior_k)
                if prior_order_k != state.healpix_order:
                    logger.info(
                        "Replay override: remapping provided half-%d direction prior from healpix_order=%d to %d",
                        _half_idx + 1,
                        prior_order_k,
                        state.healpix_order,
                    )
                    if k_class_enabled:
                        prior_k = np.stack(
                            [
                                remap_direction_prior_to_healpix_order(
                                    prior_k[class_idx],
                                    prior_order_k,
                                    state.healpix_order,
                                )
                                for class_idx in range(n_classes)
                            ],
                            axis=0,
                        )
                    else:
                        prior_k = remap_direction_prior_to_healpix_order(
                            prior_k,
                            prior_order_k,
                            state.healpix_order,
                        )
                    prior_order_k = state.healpix_order
                if k_class_enabled:
                    class_direction_prior_per_half[_half_idx] = normalize_class_direction_prior(prior_k, n_classes)
                    class_direction_prior_order_per_half[_half_idx] = prior_order_k
                    logger.info(
                        "Replay override: class direction prior half-%d <- provided override (%d classes, %d directions)",
                        _half_idx + 1,
                        class_direction_prior_per_half[_half_idx].shape[0],
                        class_direction_prior_per_half[_half_idx].shape[1],
                    )
                else:
                    global_direction_prior_per_half[_half_idx] = prior_k
                    global_direction_prior_order_per_half[_half_idx] = prior_order_k
                    logger.info(
                        "Replay override: direction prior half-%d <- provided override (%d directions, range=[%.6f, %.6f], zeros=%d)",
                        _half_idx + 1,
                        len(prior_k),
                        float(prior_k.min()),
                        float(prior_k.max()),
                        int(np.sum(prior_k == 0)),
                    )

    return ReplayOverrideResult(
        cs=cs,
        prior_translations=_replay_prior_translations,
        previous_best_rotations=previous_best_rotations,
        noise_variance_per_half=noise_variance_per_half,
        noise_variance=noise_variance,
        previous_noise_radial_per_half=previous_noise_radial_per_half,
        previous_noise_radial=previous_noise_radial,
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        replay_meta=_replay_meta,
        current_sigma_offset_angstrom_per_half=_current_sigma_offset_angstrom_per_half,
        class_weights=_replay_class_weights,
        relion_projector_state=_replay_projector_state,
    )
