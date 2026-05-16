#!/usr/bin/env python
"""Prepare a PPCA initializer with GT mean and randomized W shell power."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.initialization import real_volume_to_centered_fourier_half


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _half_shell_labels(volume_shape) -> np.ndarray:
    labels = np.asarray(
        ftu.get_grid_of_radial_distances_real(tuple(volume_shape), scaled=False, frequency_shift=0),
        dtype=np.int64,
    ).reshape(-1)
    expected = int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))
    if labels.size != expected:
        raise AssertionError(f"shell labels have {labels.size} entries, expected {expected}")
    return labels


def _half_to_real(half_volume, volume_shape) -> np.ndarray:
    full = ftu.half_volume_to_full_volume(jnp.asarray(half_volume), tuple(volume_shape))
    return np.asarray(ftu.get_idft3(full.reshape(tuple(volume_shape))).real, dtype=np.float32)


def _shell_power(half_columns: np.ndarray, labels: np.ndarray) -> np.ndarray:
    half_columns = np.asarray(half_columns)
    if half_columns.ndim == 1:
        half_columns = half_columns[:, None]
    shell_count = int(labels.max(initial=0)) + 1
    powers = []
    for k in range(half_columns.shape[1]):
        powers.append(
            np.bincount(labels, weights=np.abs(half_columns[:, k]) ** 2, minlength=shell_count).astype(np.float64)
        )
    return np.stack(powers, axis=1) if powers else np.zeros((shell_count, 0), dtype=np.float64)


def _match_component_shell_power(random_half: np.ndarray, target_half: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shell_count = int(labels.max(initial=0)) + 1
    target_power = np.bincount(labels, weights=np.abs(target_half) ** 2, minlength=shell_count).astype(np.float64)
    random_power = np.bincount(labels, weights=np.abs(random_half) ** 2, minlength=shell_count).astype(np.float64)
    scale = np.divide(
        target_power,
        random_power,
        out=np.zeros_like(target_power),
        where=random_power > np.finfo(np.float64).eps,
    )
    return (random_half * np.sqrt(scale)[labels]).astype(np.complex64)


def prepare_gt_mean_random_w_init(
    *,
    source_init_npz: Path,
    output_dir: Path,
    q: int | None = None,
    seed: int = 20260507,
) -> dict[str, Any]:
    source = np.load(source_init_npz, allow_pickle=True)
    if "mu" not in source or "W" not in source:
        raise ValueError(f"{source_init_npz} must contain real-space mu and W arrays")
    mu = np.asarray(source["mu"], dtype=np.float32)
    source_W = np.asarray(source["W"], dtype=np.float32)
    if mu.ndim != 3 or source_W.ndim != 4:
        raise ValueError(f"expected mu [N,N,N] and W [q,N,N,N], got {mu.shape} and {source_W.shape}")
    q_resolved = int(source_W.shape[0]) if q is None else int(q)
    if q_resolved < 0 or q_resolved > int(source_W.shape[0]):
        raise ValueError(f"q={q_resolved} is outside source W component count {source_W.shape[0]}")

    volume_shape = tuple(int(x) for x in mu.shape)
    labels = _half_shell_labels(volume_shape)
    target_half = np.stack(
        [np.asarray(real_volume_to_centered_fourier_half(source_W[k]), dtype=np.complex64) for k in range(q_resolved)],
        axis=1,
    )

    rng = np.random.default_rng(int(seed))
    random_half_columns = []
    random_real_columns = []
    for k in range(q_resolved):
        random_real = rng.standard_normal(volume_shape).astype(np.float32)
        random_half = np.asarray(real_volume_to_centered_fourier_half(random_real), dtype=np.complex64)
        matched_half = _match_component_shell_power(random_half, target_half[:, k], labels)
        matched_real = _half_to_real(matched_half, volume_shape)
        random_real_columns.append(matched_real)
        random_half_columns.append(np.asarray(real_volume_to_centered_fourier_half(matched_real), dtype=np.complex64))
    random_W = np.stack(random_real_columns, axis=0) if q_resolved else np.zeros((0,) + volume_shape, dtype=np.float32)
    random_half = np.stack(random_half_columns, axis=1) if q_resolved else target_half[:, :0]

    target_shell = _shell_power(target_half, labels)
    random_shell = _shell_power(random_half, labels)
    denom = np.maximum(np.abs(target_shell), np.finfo(np.float64).eps)
    rel_err = np.divide(np.abs(random_shell - target_shell), denom, out=np.zeros_like(target_shell), where=denom > 0)
    significant = np.abs(target_shell) > np.maximum(np.max(np.abs(target_shell), axis=0, keepdims=True) * 1.0e-8, 1.0e-12)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "ppca_init.npz"
    np.savez_compressed(
        npz_path,
        mu=mu.astype(np.float32),
        W=random_W.astype(np.float32),
        q=np.asarray(q_resolved, dtype=np.int64),
        source_init_npz=np.asarray(str(source_init_npz)),
        seed=np.asarray(int(seed), dtype=np.int64),
    )
    summary = {
        "passed": bool(np.all(np.isfinite(mu)) and np.all(np.isfinite(random_W))),
        "source_init_npz": source_init_npz,
        "output_init": npz_path,
        "q": int(q_resolved),
        "seed": int(seed),
        "mu_rms": float(np.sqrt(np.mean(mu**2))),
        "source_W_rms": float(np.sqrt(np.mean(source_W[:q_resolved] ** 2))) if q_resolved else 0.0,
        "random_W_rms": float(np.sqrt(np.mean(random_W**2))) if q_resolved else 0.0,
        "source_W_component_norms": [float(np.linalg.norm(source_W[k].reshape(-1))) for k in range(q_resolved)],
        "random_W_component_norms": [float(np.linalg.norm(random_W[k].reshape(-1))) for k in range(q_resolved)],
        "max_relative_shell_power_error": float(np.max(rel_err[significant])) if np.any(significant) else 0.0,
        "max_relative_shell_power_error_all_shells": float(np.max(rel_err)) if rel_err.size else 0.0,
        "shell_power_error_significance_threshold": "max(1e-8 * component_max_shell_power, 1e-12)",
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _jsonable(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-init-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260507)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_gt_mean_random_w_init(
        source_init_npz=Path(args.source_init_npz),
        output_dir=Path(args.output_dir),
        q=args.q,
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
