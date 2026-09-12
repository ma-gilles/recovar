"""Snapshot and restore selected state at a RELION replay boundary.

The controller decides when a probe runs. This module owns copying, component
selection and optional amplitude scaling; CLI validation and variant names
live in ``state_swap_probe``. Array conversions, snapshot ownership and the
ordered return tuple retain the controller's existing semantics.
"""

import logging
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics.state_swap_probe import _STATE_SWAP_VARIANT_COMPONENTS

# Keep the existing diagnostic log namespace for configured handlers/filters.
logger = logging.getLogger(__name__)


_STATE_SWAP_MAP_SCALE_VARIANTS = {
    "recovar_maps_global_scale_to_relion": ("recovar", "relion", "global"),
    "recovar_maps_shell_scale_to_relion": ("recovar", "relion", "shell"),
    "relion_maps_global_scale_to_recovar": ("relion", "recovar", "global"),
    "relion_maps_shell_scale_to_recovar": ("relion", "recovar", "shell"),
}


_STATE_SWAP_STATE_FIELD_GROUPS = {
    "state_sampling_grid": {
        "translation_range",
        "translation_step",
        "adaptive_oversampling",
    },
    "state_local_priors": {
        "do_local_search",
        "sigma_rot",
        "sigma_psi",
    },
    "state_convergence_only": {
        "current_resolution",
        "previous_resolution",
        "ave_Pmax",
        "acc_rot",
        "acc_trans",
        "fraction_changed",
        "changes_optimal_offsets",
        "current_changes_optimal_orientations",
        "current_changes_optimal_offsets_angstrom",
        "current_changes_optimal_classes",
        "smallest_changes_optimal_orientations",
        "smallest_changes_optimal_offsets_angstrom",
        "smallest_changes_optimal_classes",
        "nr_iter_wo_resol_gain",
        "nr_iter_wo_assignment_changes",
        "nr_iter_wo_large_hidden_variable_changes",
        "mpi_leader_hidden_variable_angular_step_deg",
        "mpi_leader_hidden_variable_translation_step_angstrom",
        "suppress_hidden_variable_increment_once",
        "has_converged",
    },
}


_STATE_SWAP_STATE_NO_GRID_EXCLUDE = (
    _STATE_SWAP_STATE_FIELD_GROUPS["state_sampling_grid"] | _STATE_SWAP_STATE_FIELD_GROUPS["state_local_priors"]
)


def _copy_optional_array(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _copy_half_pair(values):
    return [_copy_optional_array(value) for value in values]


def _copy_direction_prior_state(values, orders):
    return _copy_half_pair(values), [None if order is None else int(order) for order in orders]


def _copy_optional_float_pair(values):
    if values is None:
        return None
    return [float(values[0]), float(values[1])]


def _restore_state_fields(state, state_fields, fields):
    for field_name in fields:
        if field_name in state_fields:
            setattr(state, field_name, state_fields[field_name])


def _snapshot_state_swap_inputs(
    *,
    state,
    cs,
    means,
    mean_variance,
    noise_variance_per_half,
    noise_variance,
    previous_noise_radial_per_half,
    previous_noise_radial,
    relion_half_inputs,
    previous_best_rotations,
    current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half,
    class_direction_prior_per_half,
    class_direction_prior_order_per_half,
    global_direction_prior_per_half,
    global_direction_prior_order_per_half,
):
    class_priors, class_prior_orders = _copy_direction_prior_state(
        class_direction_prior_per_half,
        class_direction_prior_order_per_half,
    )
    global_priors, global_prior_orders = _copy_direction_prior_state(
        global_direction_prior_per_half,
        global_direction_prior_order_per_half,
    )
    return {
        "state_fields": dict(state.__dict__),
        "cs": int(cs),
        "means": [_copy_optional_array(mean) for mean in means],
        "mean_variance": _copy_optional_array(mean_variance),
        "noise_variance_per_half": _copy_half_pair(noise_variance_per_half),
        "noise_variance": _copy_optional_array(noise_variance),
        "previous_noise_radial_per_half": _copy_half_pair(previous_noise_radial_per_half),
        "previous_noise_radial": _copy_optional_array(previous_noise_radial),
        "image_corrections": _copy_half_pair(relion_half_inputs.image_corrections),
        "scale_corrections": _copy_half_pair(relion_half_inputs.scale_corrections),
        "previous_best_translations": _copy_half_pair(relion_half_inputs.previous_best_translations),
        "previous_best_rotation_eulers": _copy_half_pair(relion_half_inputs.previous_best_rotation_eulers),
        "previous_best_rotations": _copy_half_pair(previous_best_rotations),
        "current_sigma_offset_angstrom": float(current_sigma_offset_angstrom),
        "current_sigma_offset_angstrom_per_half": _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
        "class_direction_prior_per_half": class_priors,
        "class_direction_prior_order_per_half": class_prior_orders,
        "global_direction_prior_per_half": global_priors,
        "global_direction_prior_order_per_half": global_prior_orders,
    }


class _StateSwapValues(NamedTuple):
    """Iteration state the RELION replay override may hand back to the controller.

    The controller unpacks it positionally in this order; the state-swap probe
    returns the inputs unchanged unless a diagnostic variant restores RECOVAR
    components at its target iteration.
    """

    cs: object
    means: object
    mean_variance: object
    noise_variance_per_half: object
    noise_variance: object
    previous_noise_radial_per_half: object
    previous_noise_radial: object
    previous_best_rotations: object
    current_sigma_offset_angstrom: object
    current_sigma_offset_angstrom_per_half: object
    class_direction_prior_per_half: object
    class_direction_prior_order_per_half: object
    global_direction_prior_per_half: object
    global_direction_prior_order_per_half: object



def _state_swap_map_shell_labels(volume_shape):
    """Return unshifted integer-radius labels for a full Fourier volume."""

    shape = tuple(int(size) for size in volume_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"State-swap map scaling requires a positive three-dimensional volume shape, got {shape}")
    axes = [np.fft.fftfreq(size) * size for size in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.rint(np.sqrt(sum(grid * grid for grid in grids))).astype(np.int32).reshape(-1)


def _scale_state_swap_reference_maps(
    source_means,
    target_means,
    *,
    mode,
    volume_shape,
):
    """Scale source-map amplitudes toward target maps without changing phase."""

    if mode not in {"global", "shell"}:
        raise ValueError(f"Unknown state-swap map scaling mode {mode!r}")
    if len(source_means) != len(target_means) or not source_means:
        raise ValueError("State-swap map scaling requires equal non-empty source and target map lists")

    expected_size = int(np.prod(tuple(int(size) for size in volume_shape)))
    shell_labels = _state_swap_map_shell_labels(volume_shape) if mode == "shell" else None
    scaled_means = []
    summaries = []
    for map_index, (source_value, target_value) in enumerate(zip(source_means, target_means, strict=True)):
        if source_value is None or target_value is None:
            raise ValueError(f"State-swap map scaling requires map {map_index + 1} in both states")
        source = np.asarray(source_value)
        target = np.asarray(target_value)
        if source.shape != target.shape or source.size != expected_size:
            raise ValueError(
                "State-swap map scaling shape mismatch for map "
                f"{map_index + 1}: source={source.shape}, target={target.shape}, "
                f"expected_size={expected_size}"
            )
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError(f"State-swap map scaling requires finite map {map_index + 1} values")

        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        before_norm = float(np.linalg.norm(source_flat - target_flat))
        target_norm = float(np.linalg.norm(target_flat))
        relative_before = before_norm / target_norm if target_norm > 0.0 else before_norm

        if mode == "global":
            denominator = float(np.vdot(source_flat, source_flat).real)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(f"State-swap map {map_index + 1} has zero or invalid source energy")
            scale = float(np.vdot(source_flat, target_flat).real / denominator)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"State-swap map {map_index + 1} has invalid global scale {scale}")
            scaled_flat = source_flat * scale
            scale_values = np.asarray([scale], dtype=np.float64)
        else:
            scaled_flat = source_flat.copy()
            scale_values = np.ones(int(np.max(shell_labels)) + 1, dtype=np.float64)
            for shell in np.unique(shell_labels):
                shell_mask = shell_labels == shell
                source_shell = source_flat[shell_mask]
                target_shell = target_flat[shell_mask]
                denominator = float(np.vdot(source_shell, source_shell).real)
                target_energy = float(np.vdot(target_shell, target_shell).real)
                if denominator <= 0.0:
                    if target_energy > 0.0:
                        raise ValueError(
                            "State-swap map scaling cannot create target energy in "
                            f"empty source shell {int(shell)} for map {map_index + 1}"
                        )
                    scale = 1.0
                else:
                    scale = float(np.vdot(source_shell, target_shell).real / denominator)
                    if not np.isfinite(scale) or scale <= 0.0:
                        raise ValueError(
                            "State-swap map scaling found invalid scale "
                            f"{scale} in shell {int(shell)} for map {map_index + 1}"
                        )
                scaled_flat[shell_mask] = source_shell * scale
                scale_values[int(shell)] = scale

        scaled = scaled_flat.reshape(source.shape).astype(source.dtype, copy=False)
        after_norm = float(np.linalg.norm(scaled.reshape(-1) - target_flat))
        relative_after = after_norm / target_norm if target_norm > 0.0 else after_norm
        scaled_means.append(scaled)
        summaries.append(
            {
                "map_index": map_index,
                "mode": mode,
                "scale_min": float(np.min(scale_values)),
                "scale_max": float(np.max(scale_values)),
                "relative_l2_before": relative_before,
                "relative_l2_after": relative_after,
            }
        )
    return scaled_means, summaries


def _apply_state_swap_probe(
    *,
    probe,
    iteration,
    recovar_snapshot,
    state,
    cs,
    volume_shape,
    means,
    mean_variance,
    noise_variance_per_half,
    noise_variance,
    previous_noise_radial_per_half,
    previous_noise_radial,
    relion_half_inputs,
    previous_best_rotations,
    current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half,
    class_direction_prior_per_half,
    class_direction_prior_order_per_half,
    global_direction_prior_per_half,
    global_direction_prior_order_per_half,
):
    """Restore selected RECOVAR-produced state after RELION replay override."""

    unchanged = _StateSwapValues(
        cs,
        means,
        mean_variance,
        noise_variance_per_half,
        noise_variance,
        previous_noise_radial_per_half,
        previous_noise_radial,
        previous_best_rotations,
        current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half,
        class_direction_prior_per_half,
        class_direction_prior_order_per_half,
        global_direction_prior_per_half,
        global_direction_prior_order_per_half,
    )
    if not probe or recovar_snapshot is None:
        return unchanged
    target_iteration = int(probe.get("iteration", 1))
    if int(iteration) != target_iteration:
        return unchanged

    variant = str(probe.get("variant", "all_relion"))
    components = _STATE_SWAP_VARIANT_COMPONENTS.get(variant)
    if components is None:
        raise ValueError(
            f"Unknown state_swap_probe variant {variant!r}; expected one of "
            f"{sorted(['all_relion', *_STATE_SWAP_VARIANT_COMPONENTS])}",
        )
    logger.warning(
        "STATE-SWAP diagnostic: iteration=%d variant=%s restoring RECOVAR components=%s after RELION replay",
        int(iteration) + 1,
        variant,
        ",".join(sorted(components)),
    )

    recovar_state_fields = recovar_snapshot["state_fields"]
    if "state" in components:
        state.__dict__.update(recovar_state_fields)
    if "state_sampling_grid" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_sampling_grid"],
        )
    if "state_local_priors" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_local_priors"],
        )
    if "state_convergence_only" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_convergence_only"],
        )
    if "state_no_grid" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            set(recovar_state_fields) - _STATE_SWAP_STATE_NO_GRID_EXCLUDE,
        )
    if "maps" in components:
        means = [jnp.asarray(mean) if mean is not None else None for mean in recovar_snapshot["means"]]
    if "map_scale" in components:
        source_name, target_name, scale_mode = _STATE_SWAP_MAP_SCALE_VARIANTS[variant]
        relion_means = means
        recovar_means = recovar_snapshot["means"]
        source_means = recovar_means if source_name == "recovar" else relion_means
        target_means = recovar_means if target_name == "recovar" else relion_means
        scaled_means, scale_summaries = _scale_state_swap_reference_maps(
            source_means,
            target_means,
            mode=scale_mode,
            volume_shape=volume_shape,
        )
        means = [jnp.asarray(mean) for mean in scaled_means]
        for summary in scale_summaries:
            logger.warning(
                "STATE-SWAP map amplitude: variant=%s map=%d mode=%s scale=[%.9g, %.9g] relative_l2=%.9g->%.9g",
                variant,
                int(summary["map_index"]) + 1,
                summary["mode"],
                summary["scale_min"],
                summary["scale_max"],
                summary["relative_l2_before"],
                summary["relative_l2_after"],
            )
    if "tau2_noise" in components or "tau2" in components:
        mean_variance = jnp.asarray(recovar_snapshot["mean_variance"])
    if "tau2_noise" in components or "noise_variance" in components:
        noise_variance_per_half = [
            jnp.asarray(noise_k) if noise_k is not None else None
            for noise_k in recovar_snapshot["noise_variance_per_half"]
        ]
        noise_variance = jnp.asarray(recovar_snapshot["noise_variance"])
    if "tau2_noise" in components or "previous_noise_radial" in components:
        previous_noise_radial_per_half = _copy_half_pair(recovar_snapshot["previous_noise_radial_per_half"])
        previous_noise_radial = _copy_optional_array(recovar_snapshot["previous_noise_radial"])
    if "image_scale" in components or "image_correction" in components:
        relion_half_inputs.image_corrections = _copy_half_pair(recovar_snapshot["image_corrections"])
    if "image_scale" in components or "scale_correction" in components:
        relion_half_inputs.scale_corrections = _copy_half_pair(recovar_snapshot["scale_corrections"])
    if "poses" in components:
        relion_half_inputs.previous_best_translations = _copy_half_pair(
            recovar_snapshot["previous_best_translations"],
        )
        relion_half_inputs.previous_best_rotation_eulers = _copy_half_pair(
            recovar_snapshot["previous_best_rotation_eulers"],
        )
        previous_best_rotations = _copy_half_pair(recovar_snapshot["previous_best_rotations"])
    if "direction_prior" in components:
        class_direction_prior_per_half = _copy_half_pair(recovar_snapshot["class_direction_prior_per_half"])
        class_direction_prior_order_per_half = list(recovar_snapshot["class_direction_prior_order_per_half"])
        global_direction_prior_per_half = _copy_half_pair(recovar_snapshot["global_direction_prior_per_half"])
        global_direction_prior_order_per_half = list(recovar_snapshot["global_direction_prior_order_per_half"])
    if "sigma_offset" in components:
        current_sigma_offset_angstrom = float(recovar_snapshot["current_sigma_offset_angstrom"])
        current_sigma_offset_angstrom_per_half = _copy_optional_float_pair(
            recovar_snapshot["current_sigma_offset_angstrom_per_half"]
        )
    if "current_size" in components:
        cs = int(recovar_snapshot["cs"])

    return _StateSwapValues(
        cs,
        means,
        mean_variance,
        noise_variance_per_half,
        noise_variance,
        previous_noise_radial_per_half,
        previous_noise_radial,
        previous_best_rotations,
        current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half,
        class_direction_prior_per_half,
        class_direction_prior_order_per_half,
        global_direction_prior_per_half,
        global_direction_prior_order_per_half,
    )
