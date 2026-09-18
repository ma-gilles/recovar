"""Optional VDAM noise-boundary captures and nonfinite-input reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from recovar.em.vdam.state import InitialModelState


def _array_finite_summary(name: str, value: object, *, max_indices: int = 5) -> str:
    arr = np.asarray(value)
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        values = arr.astype(np.float64, copy=False).reshape(-1)
        if values.size == 0:
            return f"{name}: shape={arr.shape}, empty"
        return (
            f"{name}: shape={arr.shape}, all finite, min={float(np.min(values)):.6g}, max={float(np.max(values)):.6g}"
        )
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

    from recovar.em.relion.relion_metadata import _relion_half_plane_shell_counts

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
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts((int(state.ori_size), int(state.ori_size))),
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
