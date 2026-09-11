"""Noise and class-probability updates from the InitialModel E-step metadata.

RELION's ``MlOptimiser::maximization`` noise (sigma2) and class-probability
(pdf_class) updates for the native VDAM InitialModel, computed from the
E-step accumulator metadata, with the env-gated noise-boundary dump.
``iteration_loop`` calls these once per iteration.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .schedules import DEFAULT_GRAD_MU
from .state import InitialModelState

MIN_SIGMA2_OFFSET_ANGSTROM2: float = 2.0


def _halfset_values(meta: dict, key: str) -> list:
    suffix = f"_{key}"
    return [meta[name] for name in sorted(meta) if name.startswith("halfset_") and name.endswith(suffix)]


def _posterior_sums_from_meta(meta: dict, key: str) -> np.ndarray | None:
    if (value := meta.get(key)) is not None:
        return np.asarray(value, dtype=np.float64)
    values = [np.asarray(v, dtype=np.float64) for v in _halfset_values(meta, key)]
    return sum(values[1:], values[0].copy()) if values else None


def _scalar_sum_from_meta(meta: dict, key: str) -> float | None:
    if (value := meta.get(key)) is not None:
        return float(value)
    values = _halfset_values(meta, key)
    return float(sum(float(v) for v in values)) if values else None


def _my_mu(mu: float, do_grad: bool, subset_size: int) -> float:
    my_mu = float(mu) if do_grad and subset_size != -1 else 0.0
    if my_mu < 0.0 or my_mu > 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    return my_mu


def _array_finite_summary(name: str, value: object, *, max_indices: int = 5) -> str:
    arr = np.asarray(value)
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        values = arr.astype(np.float64, copy=False).reshape(-1)
        if values.size == 0:
            return f"{name}: shape={arr.shape}, empty"
        return f"{name}: shape={arr.shape}, all finite, min={float(np.min(values)):.6g}, max={float(np.max(values)):.6g}"
    finite_values = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    finite_range = (
        f"finite_min={float(np.min(finite_values)):.6g}, finite_max={float(np.max(finite_values)):.6g}"
        if finite_values.size
        else "no finite values"
    )
    sample_indices = [tuple(int(x) for x in idx) for idx in bad[:max_indices]]
    return f"{name}: shape={arr.shape}, nonfinite={int(bad.shape[0])}/{arr.size}, {finite_range}, first_bad={sample_indices}"


def _dump_noise_failure_meta(state: InitialModelState, meta: dict, summaries: Sequence[str]) -> str | None:
    dump_root = os.environ.get("RECOVAR_INITIALMODEL_NOISE_FAILURE_DUMP_DIR")
    if not dump_root:
        return None
    path = Path(dump_root)
    path.mkdir(parents=True, exist_ok=True)
    dump_path = path / f"noise_failure_iter_{getattr(state, 'iter', 'unknown')}.npz"
    payload: dict[str, np.ndarray] = {
        "state_iter": np.asarray([getattr(state, "iter", -1)], dtype=np.int64),
        "state_subset_size": np.asarray([getattr(state, "subset_size", -1)], dtype=np.int64),
        "state_ori_size": np.asarray([getattr(state, "ori_size", -1)], dtype=np.int64),
        "summaries": np.asarray(list(summaries), dtype=str),
    }
    for key, value in sorted(meta.items()):
        if not any(token in key for token in ("noise", "wsum")):
            continue
        try:
            payload[key] = np.asarray(value)
        except Exception:
            payload[f"{key}_repr"] = np.asarray([repr(value)], dtype=str)
    np.savez_compressed(dump_path, **payload)
    return str(dump_path)


def _maybe_dump_noise_update_boundary(
    state: InitialModelState,
    updated_state: InitialModelState,
    *,
    wsum_sigma2_noise: np.ndarray,
    wsum_img_power: np.ndarray,
    noise_sumw: float,
    wsum_noise_a2: np.ndarray | None = None,
    wsum_noise_xa: np.ndarray | None = None,
) -> str | None:
    """Write VDAM noise sufficient statistics only when explicitly requested."""

    dump_root = os.environ.get("RECOVAR_INITIALMODEL_NOISE_UPDATE_DUMP_DIR")
    if not dump_root:
        return None
    requested = os.environ.get("RECOVAR_INITIALMODEL_NOISE_UPDATE_DUMP_ITERATION")
    if requested:
        requested_iterations = {int(token.strip()) for token in requested.split(",") if token.strip()}
        if int(state.iter) not in requested_iterations:
            return None

    from recovar.em.dense_single_volume.relion_metadata import (
        _relion_half_plane_shell_counts,
    )

    dump_dir = Path(dump_root)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"initialmodel_noise_update_it{int(state.iter):03d}.npz"
    if dump_path.exists():
        raise ValueError(f"refusing to overwrite {dump_path}")
    n4 = float(int(state.ori_size) ** 4)
    old_noise = np.asarray(state.sigma2_noise, dtype=np.float64)[0] * n4
    new_noise = np.asarray(updated_state.sigma2_noise, dtype=np.float64)[0] * n4
    residual = np.asarray(wsum_sigma2_noise, dtype=np.float64)
    image_power = np.asarray(wsum_img_power, dtype=np.float64)
    payload = {
        "schema": np.asarray("recovar.initialmodel.noise_update_boundary.v1"),
        "iteration": np.asarray([int(state.iter)], dtype=np.int32),
        "current_size": np.asarray([int(state.current_size)], dtype=np.int32),
        "image_shape": np.asarray([int(state.ori_size), int(state.ori_size)], dtype=np.int32),
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts(
            (int(state.ori_size), int(state.ori_size))
        ),
        "half0_wsum_sigma2_noise": residual,
        "half0_wsum_img_power": image_power,
        "half0_wsum_total": residual + image_power,
        "half0_sumw": np.asarray([float(noise_sumw)], dtype=np.float64),
        "half0_previous_sigma2_noise": old_noise,
        "half0_sigma2_noise": new_noise,
    }
    if wsum_noise_a2 is not None or wsum_noise_xa is not None:
        if wsum_noise_a2 is None or wsum_noise_xa is None:
            raise ValueError("noise split diagnostics require both wsum_noise_a2 and wsum_noise_xa")
        noise_a2 = np.asarray(wsum_noise_a2, dtype=np.float64)
        noise_xa = np.asarray(wsum_noise_xa, dtype=np.float64)
        if noise_a2.shape != residual.shape or noise_xa.shape != residual.shape:
            raise ValueError("noise split diagnostics must match the noise shell topology")
        payload["half0_wsum_noise_a2"] = noise_a2
        payload["half0_wsum_noise_xa"] = noise_xa
    np.savez_compressed(dump_path, **payload)
    return str(dump_path)


def update_noise_from_estep_meta(
    state: InitialModelState,
    meta: dict,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """Update ``sigma2_noise`` from E-step weighted sums (engine units → RELION /N⁴)."""
    wsum_sigma2_noise = _posterior_sums_from_meta(meta, "wsum_sigma2_noise")
    wsum_img_power = _posterior_sums_from_meta(meta, "wsum_img_power")
    wsum_noise_a2 = _posterior_sums_from_meta(meta, "wsum_noise_a2")
    wsum_noise_xa = _posterior_sums_from_meta(meta, "wsum_noise_xa")
    noise_sumw = _scalar_sum_from_meta(meta, "noise_sumw")
    if wsum_sigma2_noise is None or wsum_img_power is None or noise_sumw is None:
        return state
    if noise_sumw <= 0.0 or not np.isfinite(noise_sumw):
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    if wsum_sigma2_noise.shape != wsum_img_power.shape:
        raise ValueError(
            f"wsum_sigma2_noise and wsum_img_power shape mismatch: {wsum_sigma2_noise.shape} vs {wsum_img_power.shape}"
        )
    expected_shells = int(state.ori_size) // 2 + 1
    if wsum_sigma2_noise.shape != (expected_shells,):
        raise ValueError(f"noise weighted sums must have shape ({expected_shells},), got {wsum_sigma2_noise.shape}")
    if not np.all(np.isfinite(wsum_sigma2_noise)) or not np.all(np.isfinite(wsum_img_power)):
        summaries = [
            _array_finite_summary("wsum_sigma2_noise", wsum_sigma2_noise),
            _array_finite_summary("wsum_img_power", wsum_img_power),
            f"noise_sumw={noise_sumw!r}",
        ]
        if dump_path := _dump_noise_failure_meta(state, meta, summaries):
            summaries.append(f"dump={dump_path}")
        raise ValueError("noise weighted sums must be finite: " + "; ".join(summaries))

    from recovar.reconstruction import noise

    sigma2_relion_units = np.asarray(
        noise.normalize_wsum_to_sigma2_noise(
            wsum_sigma2_noise, wsum_img_power, float(noise_sumw), (int(state.ori_size), int(state.ori_size))
        ),
        dtype=np.float64,
    ) / float(int(state.ori_size) ** 4)
    if not np.all(np.isfinite(sigma2_relion_units)) or np.any(sigma2_relion_units <= 0.0):
        raise ValueError("updated sigma2_noise must be positive and finite")

    new_state = replace(state)
    new_sigma2 = np.asarray(state.sigma2_noise, dtype=np.float64).copy()
    if new_sigma2.ndim != 2 or new_sigma2.shape[1] != expected_shells:
        raise ValueError(f"sigma2_noise must have shape (G, {expected_shells}), got {new_sigma2.shape}")
    new_state.sigma2_noise = new_sigma2 * my_mu + (1.0 - my_mu) * sigma2_relion_units[None, :]
    _maybe_dump_noise_update_boundary(
        state,
        new_state,
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        noise_sumw=float(noise_sumw),
        wsum_noise_a2=wsum_noise_a2,
        wsum_noise_xa=wsum_noise_xa,
    )
    return new_state


def update_probabilities_from_estep_meta(
    state: InitialModelState,
    meta: dict,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """``MlOptimiser::maximizationOtherParameters`` for pdf_class / pdf_direction / sigma2_offset."""
    class_sums = _posterior_sums_from_meta(meta, "class_posterior_sums")
    if class_sums is None:
        return state
    class_sums = np.asarray(class_sums, dtype=np.float64)
    if class_sums.shape != (state.K,):
        raise ValueError(f"class_posterior_sums must have shape ({state.K},), got {class_sums.shape}")
    if not np.all(np.isfinite(class_sums)) or np.any(class_sums < 0.0):
        raise ValueError("class_posterior_sums must be non-negative and finite")
    sum_weight = float(np.sum(class_sums))
    if sum_weight <= 0.0:
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    new_state = replace(state)
    new_pdf_class = np.asarray(state.pdf_class, dtype=np.float64) * my_mu
    new_pdf_class += (1.0 - my_mu) * class_sums / sum_weight
    pdf_class_sum = float(np.sum(new_pdf_class))
    if pdf_class_sum > 0.0:
        new_pdf_class /= pdf_class_sum
    new_state.pdf_class = new_pdf_class

    direction_sums = _posterior_sums_from_meta(meta, "class_direction_posterior_sums")
    if direction_sums is not None and state.pdf_direction is not None:
        direction_sums = np.asarray(direction_sums, dtype=np.float64)
        if direction_sums.ndim != 2 or direction_sums.shape[0] != state.K:
            raise ValueError(
                f"class_direction_posterior_sums must have shape ({state.K}, n_directions), got {direction_sums.shape}"
            )
        if not np.all(np.isfinite(direction_sums)) or np.any(direction_sums < 0.0):
            raise ValueError("class_direction_posterior_sums must be non-negative and finite")
        pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
        if pdf_direction.shape != direction_sums.shape:
            # RELION resizes pdf_direction to the new sampling.NrDirections()
            # and fills it uniformly when angular sampling changes.
            pdf_direction = np.full(direction_sums.shape, 1.0 / float(state.K * direction_sums.shape[1]))
        new_pdf_direction = pdf_direction * my_mu
        new_pdf_direction += (1.0 - my_mu) * direction_sums / sum_weight
        new_state.pdf_direction = new_pdf_direction

    wsum_sigma2_offset = meta.get("wsum_sigma2_offset")
    if wsum_sigma2_offset is not None:
        wsum_sigma2_offset = float(wsum_sigma2_offset)
        if not np.isfinite(wsum_sigma2_offset) or wsum_sigma2_offset < 0.0:
            raise ValueError("wsum_sigma2_offset must be non-negative and finite")
        sigma2_offset_sumw = float(meta.get("sigma2_offset_sumw", sum_weight))
        if not np.isfinite(sigma2_offset_sumw) or sigma2_offset_sumw <= 0.0:
            raise ValueError("sigma2_offset_sumw must be positive and finite")
        sigma2_offset = float(state.sigma2_offset) * my_mu
        # RELION divides by 2*sum_weight for 2D particle translations.
        # Its sum_weight is accumulated from the same significant-pruned
        # reconstruction weights as wsum_sigma2_offset, rather than from the
        # unpruned per-image class responsibilities.
        sigma2_offset += (1.0 - my_mu) * wsum_sigma2_offset / (2.0 * sigma2_offset_sumw)
        new_state.sigma2_offset = max(float(sigma2_offset), MIN_SIGMA2_OFFSET_ANGSTROM2)

    return new_state
