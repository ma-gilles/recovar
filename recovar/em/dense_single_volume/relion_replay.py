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
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import (
    healpix_angular_step,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
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

    @classmethod
    def from_initial_values(
        cls,
        *,
        previous_best_translations,
        previous_best_rotation_eulers,
        image_corrections,
        scale_corrections,
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
        )


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


def apply_iter_replay_overrides(
    *,
    iter_replay_override: dict | None,
    perturb_replay_relion_dir: str | None,
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
    class_direction_prior_per_half: list,
    class_direction_prior_order_per_half: list,
    global_direction_prior_per_half: list,
    global_direction_prior_order_per_half: list,
) -> ReplayOverrideResult:
    """Apply per-iteration replay overrides to the in-flight iteration state.

    Mutates ``state``, ``relion_half_inputs``, and the four direction-prior
    lists in place. Returns explicit new values for everything else.

    Two override sources, applied in order:

    1. ``perturb_replay_relion_dir``: read RELION's per-iter sampling.star +
       (control) model.star + (previous-iter) half-model.star, override
       healpix order, local-search activation, sigma priors, translation
       range/step, current_size, and direction priors.
    2. ``iter_replay_override`` dict: explicit overrides for sigma_offset,
       previous-best poses, image/scale corrections, noise variance, and
       direction priors.
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

    if perturb_replay_relion_dir is not None:
        _star = os.path.join(
            perturb_replay_relion_dir,
            f"run_it{init_relion_iteration + iteration + 1:03d}_sampling.star",
        )
        _replay_meta = _il.read_relion_sampling_metadata(_star)
        _relion_hp = int(_replay_meta["healpix_order"])
        _relion_psi_step_deg = float(_replay_meta.get("psi_step", healpix_angular_step(_relion_hp)))
        # RELION stores offset_{range,step} in Angstroms; convert to px.
        _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
        _relion_offset_range = float(_replay_meta["offset_range"]) / _px
        _relion_offset_step = float(_replay_meta["offset_step"]) / _px
        _replay_prior_translations = jnp.array(
            _il.get_translation_grid(
                _relion_offset_range,
                _relion_offset_step,
            ).astype(np.float32)
        )
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
        # The model star records the control state for the replayed E-step.
        # Reuse it for both current_size and local-prior sigmas.
        _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
        _model_star_candidates = [
            os.path.join(perturb_replay_relion_dir, f"run_it{_cs_iter:03d}_half1_model.star"),
            os.path.join(perturb_replay_relion_dir, f"run_it{_cs_iter:03d}_model.star"),
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
        if (
            abs(float(state.translation_range) - _relion_offset_range) > 1e-6
            or abs(float(state.translation_step) - _relion_offset_step) > 1e-6
        ):
            logger.info(
                "Replay override: translation_range %.3f -> %.3f px, step %.3f -> %.3f px",
                float(state.translation_range),
                _relion_offset_range,
                float(state.translation_step),
                _relion_offset_step,
            )
            state.translation_range = _relion_offset_range
            state.translation_step = _relion_offset_step

        # Override current_size from the RELION model star that records the
        # control state for the replayed E-step. Empirically, replaying
        # RELION iter N+1 against the saved benchmark trajectory requires
        # reading run_it{N+1}_model.star, not run_it{N}_model.star:
        # the saved model star already carries the control variables
        # (current_size, sigma_offset) used by that E-step.
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

        if iteration > 0:
            _prior_iter = init_relion_iteration + iteration
            if iter_replay_override is None or iter_replay_override.get("direction_prior") is None:
                for _half_idx in range(2):
                    _prior_star = os.path.join(
                        perturb_replay_relion_dir,
                        f"run_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                    )
                    if not os.path.exists(_prior_star):
                        continue
                    _relion_direction_prior = (
                        _il.read_relion_direction_priors(_prior_star, n_classes)
                        if k_class_enabled
                        else _il.read_relion_direction_prior(_prior_star)
                    )
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

    if iter_replay_override is not None:
        _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
        if _replay_sigma is not None:
            current_sigma_offset_angstrom = float(_replay_sigma)
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
        _replay_img_corr = iter_replay_override.get("image_corrections")
        if _replay_img_corr is not None:
            relion_half_inputs.image_corrections = _optional_float32_half_pair(_replay_img_corr)
            logger.info(
                "Replay override: image_corrections <- half1=%s half2=%s",
                "set" if relion_half_inputs.image_corrections[0] is not None else "none",
                "set" if relion_half_inputs.image_corrections[1] is not None else "none",
            )
        _replay_scale_corr = iter_replay_override.get("scale_corrections")
        if _replay_scale_corr is not None:
            relion_half_inputs.scale_corrections = _optional_float32_half_pair(_replay_scale_corr)
            logger.info(
                "Replay override: scale_corrections <- half1=%s half2=%s",
                "set" if relion_half_inputs.scale_corrections[0] is not None else "none",
                "set" if relion_half_inputs.scale_corrections[1] is not None else "none",
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
    )
