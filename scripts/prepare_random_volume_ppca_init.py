#!/usr/bin/env python
"""Prepare a PPCA initializer by sampling and downsampling a volume bank."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from recovar.em.ppca_refinement.initialization import initialize_ppca_from_gt_volumes
from recovar.utils import helpers


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


def _source_index(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)(?=\.mrc$)", path.name)
    return int(match.group(1)) if match else int(fallback)


def _load_volume(path: Path, *, frame: str):
    if frame == "recovar":
        return helpers.load_mrc(path, return_voxel_size=True)
    if frame == "relion":
        return helpers.load_relion_volume(path, return_voxel_size=True)
    raise ValueError("frame must be 'recovar' or 'relion'")


def _downsample_recovar_volume(volume: np.ndarray, target_grid_size: int) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"expected a cubic 3D volume, got {volume.shape}")
    input_grid_size = int(volume.shape[0])
    target_grid_size = int(target_grid_size)
    if target_grid_size == input_grid_size:
        return np.array(volume, dtype=np.float32, copy=True)
    if target_grid_size > input_grid_size:
        raise ValueError(
            f"target_grid_size={target_grid_size} is larger than input grid size {input_grid_size}; "
            "this helper intentionally only downsamples."
        )
    return np.real(helpers.downsample_vol_by_fourier_truncation(volume, target_grid_size)).astype(np.float32)


def prepare_random_volume_ppca_init(
    volume_paths: list[Path],
    *,
    output_dir: Path,
    k: int,
    q: int | None,
    target_grid_size: int,
    seed: int,
    frame: str,
    write_maps: bool = True,
) -> dict[str, Any]:
    """Sample ``k`` volumes, downsample them, and write PPCA init artifacts."""

    if not volume_paths:
        raise ValueError("no input volumes matched")
    if k < 1:
        raise ValueError("k must be positive")
    if k > len(volume_paths):
        raise ValueError(f"k={k} exceeds number of available volumes ({len(volume_paths)})")

    output_dir = Path(output_dir)
    map_dir = output_dir / "sampled_volumes"
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    selected_positions = rng.choice(len(volume_paths), size=int(k), replace=False)
    selected_paths = [Path(volume_paths[int(pos)]) for pos in selected_positions]
    selected_source_indices = [_source_index(path, int(pos)) for pos, path in zip(selected_positions, selected_paths)]

    downsampled = []
    written_maps = []
    voxel_sizes = []
    input_shapes = []
    output_voxel_size = None
    for out_idx, path in enumerate(selected_paths):
        volume, voxel_size = _load_volume(path, frame=frame)
        volume = np.asarray(volume, dtype=np.float32)
        input_shapes.append(tuple(int(x) for x in volume.shape))
        voxel_size_value = float(voxel_size.x) if hasattr(voxel_size, "x") else float(voxel_size)
        voxel_sizes.append(voxel_size_value)
        down = _downsample_recovar_volume(volume, int(target_grid_size))
        downsampled.append(down)
        output_voxel_size = voxel_size_value * (float(volume.shape[0]) / float(target_grid_size))
        if write_maps:
            out_path = map_dir / f"sample{out_idx:03d}_source{selected_source_indices[out_idx]:04d}.mrc"
            helpers.write_mrc(out_path, down, voxel_size=output_voxel_size)
            written_maps.append(out_path)

    volume_stack = np.stack(downsampled, axis=0)
    q_resolved = int(k - 1) if q is None else int(q)
    init = initialize_ppca_from_gt_volumes(
        volume_stack,
        q=q_resolved,
        weights=None,
        frame="recovar",
    )

    npz_path = output_dir / "ppca_init.npz"
    np.savez_compressed(
        npz_path,
        mu=init.mu.astype(np.float32),
        W=init.W.astype(np.float32),
        aligned_volumes=init.aligned_volumes.astype(np.float32),
        weights=init.weights.astype(np.float64),
        selected_positions=np.asarray(selected_positions, dtype=np.int64),
        selected_source_indices=np.asarray(selected_source_indices, dtype=np.int64),
        selected_volume_paths=np.asarray([str(path) for path in selected_paths]),
        written_volume_paths=np.asarray([str(path) for path in written_maps]),
        target_grid_size=np.asarray(target_grid_size, dtype=np.int64),
        q=np.asarray(q_resolved, dtype=np.int64),
        k=np.asarray(k, dtype=np.int64),
    )

    W_flat = init.W.reshape(init.W.shape[0], -1)
    covariance_trace = float(np.sum(np.abs(W_flat) ** 2))
    centered = volume_stack.reshape(k, -1) - init.mu.reshape(1, -1)
    empirical_trace = float(np.mean(np.sum(centered**2, axis=1)))
    retained_fraction = 1.0 if empirical_trace == 0.0 else float(covariance_trace / empirical_trace)
    summary = {
        "passed": bool(np.all(np.isfinite(init.mu)) and np.all(np.isfinite(init.W))),
        "volume_glob_count": len(volume_paths),
        "selected_volume_paths": selected_paths,
        "selected_positions": selected_positions,
        "selected_source_indices": selected_source_indices,
        "written_volume_paths": written_maps,
        "input_frame": frame,
        "input_shapes": input_shapes,
        "target_grid_size": int(target_grid_size),
        "input_voxel_sizes": voxel_sizes,
        "output_voxel_size": output_voxel_size,
        "k": int(k),
        "q": q_resolved,
        "seed": int(seed),
        "npz_path": npz_path,
        "initializer_diagnostics": init.diagnostics,
        "stats": {
            "mu_shape": list(init.mu.shape),
            "W_shape": list(init.W.shape),
            "mu_rms": float(np.sqrt(np.mean(init.mu**2))),
            "W_rms": float(np.sqrt(np.mean(init.W**2))) if init.W.size else 0.0,
            "empirical_centered_trace": empirical_trace,
            "loading_covariance_trace": covariance_trace,
            "retained_covariance_fraction": retained_fraction,
            "rank_truncated": bool(q_resolved < max(int(k) - 1, 0)),
            "trace_relative_error": float(
                abs(covariance_trace - empirical_trace) / max(abs(empirical_trace), np.finfo(np.float64).eps)
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _jsonable(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-glob", required=True, help="Input MRC glob, for example '/path/vols/vol*.mrc'")
    parser.add_argument("--output-dir", required=True, help="Directory for sampled maps and ppca_init.npz")
    parser.add_argument("--k", type=int, default=10, help="Number of volumes to sample")
    parser.add_argument("--q", type=int, default=None, help="PPCA components; default K-1")
    parser.add_argument("--target-grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--frame", choices=("recovar", "relion"), default="recovar")
    parser.add_argument("--no-write-maps", action="store_true", help="Only write ppca_init.npz and summary.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    volume_paths = [Path(path) for path in sorted(glob.glob(args.volume_glob))]
    summary = prepare_random_volume_ppca_init(
        [Path(path) for path in volume_paths],
        output_dir=Path(args.output_dir),
        k=int(args.k),
        q=args.q,
        target_grid_size=int(args.target_grid_size),
        seed=int(args.seed),
        frame=args.frame,
        write_maps=not args.no_write_maps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
