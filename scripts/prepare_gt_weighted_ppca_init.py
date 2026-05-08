#!/usr/bin/env python
"""Prepare a PPCA initializer from simulator GT volumes and image assignments."""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.initialization import initialize_ppca_from_gt_volumes
from recovar.simulation import synthetic_dataset
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


def _load_simulation_info(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_volume(path: Path, *, frame: str):
    if frame == "recovar":
        return helpers.load_mrc(path, return_voxel_size=True)
    if frame == "relion":
        return helpers.load_relion_volume(path, return_voxel_size=True)
    raise ValueError("frame must be 'recovar' or 'relion'")


def _load_simulation_scaled_recovar_volumes(simulation_info) -> np.ndarray:
    """Load GT volumes exactly in the simulator frame used for particles.

    ``simulation_info['scale_vol']`` is not just a cosmetic metadata field: the
    simulator multiplies volumes by it, and it may also Fourier-resize asset
    maps to ``simulation_info['grid_size']`` before projection.  Reuse the
    synthetic-dataset loader so PPCA GT initializers match the particle stack
    amplitude and box size instead of reimplementing that convention here.
    """

    heterogeneous = synthetic_dataset.load_heterogeneous_reconstruction(simulation_info)
    volume_shape = tuple(int(x) for x in heterogeneous.volume_shape)
    volumes_fourier = np.asarray(heterogeneous.volumes).reshape((-1,) + volume_shape)
    volumes_real = [np.asarray(ftu.get_idft3(volume).real, dtype=np.float32) for volume in volumes_fourier]
    return np.stack(volumes_real, axis=0)


def prepare_gt_weighted_ppca_init(
    *,
    volume_paths: list[Path],
    simulation_info_path: Path,
    output_dir: Path,
    q: int,
    frame: str,
    write_maps: bool,
    apply_simulation_scale: bool = False,
):
    if not volume_paths:
        raise ValueError("no input volumes matched")
    simulation_info = _load_simulation_info(simulation_info_path)
    assignments = np.asarray(simulation_info["image_assignment"], dtype=np.int64)
    if assignments.size == 0:
        raise ValueError("simulation_info image_assignment is empty")
    if assignments.min(initial=0) < 0 or assignments.max(initial=-1) >= len(volume_paths):
        raise ValueError("image_assignment values are outside the provided volume bank")
    counts = np.bincount(assignments, minlength=len(volume_paths)).astype(np.float64)
    weights = counts / np.sum(counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "gt_weighted_maps"
    if write_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    voxel_sizes = []
    if apply_simulation_scale:
        if frame != "recovar":
            raise ValueError("--apply-simulation-scale uses the simulator's RECOVAR-frame loader; use --frame recovar")
        volume_stack = _load_simulation_scaled_recovar_volumes(simulation_info)
        input_shapes = [tuple(int(x) for x in vol.shape) for vol in volume_stack]
        amplitude_scale = float(simulation_info.get("scale_vol", 1.0))
        volume_loader = "synthetic_dataset.load_heterogeneous_reconstruction"
        for path in volume_paths:
            if path.exists():
                _vol, voxel_size = _load_volume(path, frame=frame)
                voxel_sizes.append(float(voxel_size.x) if hasattr(voxel_size, "x") else float(voxel_size))
    else:
        volumes = []
        input_shapes = []
        for path in volume_paths:
            vol, voxel_size = _load_volume(path, frame=frame)
            vol = np.asarray(vol, dtype=np.float32)
            volumes.append(vol)
            input_shapes.append(tuple(int(x) for x in vol.shape))
            voxel_sizes.append(float(voxel_size.x) if hasattr(voxel_size, "x") else float(voxel_size))
        volume_stack = np.stack(volumes, axis=0)
        amplitude_scale = None
        volume_loader = "explicit_volume_paths"

    init = initialize_ppca_from_gt_volumes(
        volume_stack,
        q=int(q),
        weights=weights,
        frame="recovar",
        amplitude_scale=None,
    )

    npz_path = output_dir / "ppca_init.npz"
    np.savez_compressed(
        npz_path,
        mu=init.mu.astype(np.float32),
        W=init.W.astype(np.float32),
        aligned_volumes=init.aligned_volumes.astype(np.float32),
        weights=init.weights.astype(np.float64),
        volume_paths=np.asarray([str(path) for path in volume_paths]),
        q=np.asarray(q, dtype=np.int64),
        k=np.asarray(len(volume_paths), dtype=np.int64),
        assignment_counts=counts.astype(np.int64),
    )

    written_maps = []
    if write_maps:
        voxel_size = voxel_sizes[0] if voxel_sizes else 1.0
        mu_path = map_dir / "init_mu.mrc"
        helpers.write_mrc(mu_path, init.mu.astype(np.float32), voxel_size=voxel_size)
        written_maps.append(mu_path)
        for pc_idx in range(int(q)):
            pc_path = map_dir / f"init_W{pc_idx + 1:02d}.mrc"
            helpers.write_mrc(pc_path, init.W[pc_idx].astype(np.float32), voxel_size=voxel_size)
            written_maps.append(pc_path)

    summary = {
        "passed": bool(np.all(np.isfinite(init.mu)) and np.all(np.isfinite(init.W))),
        "npz_path": npz_path,
        "simulation_info": simulation_info_path,
        "volume_glob_count": len(volume_paths),
        "volume_paths": volume_paths,
        "input_frame": frame,
        "apply_simulation_scale": bool(apply_simulation_scale),
        "amplitude_scale": amplitude_scale,
        "volume_loader": volume_loader,
        "input_shapes": input_shapes,
        "input_voxel_sizes": voxel_sizes,
        "n_images_weighted": int(assignments.size),
        "nonzero_weight_count": int(np.count_nonzero(weights)),
        "assignment_count_minmax": [int(counts.min(initial=0)), int(counts.max(initial=0))],
        "q": int(q),
        "initializer_diagnostics": init.diagnostics,
        "stats": {
            "mu_shape": list(init.mu.shape),
            "W_shape": list(init.W.shape),
            "mu_rms": float(np.sqrt(np.mean(init.mu**2))),
            "W_rms": float(np.sqrt(np.mean(init.W**2))) if init.W.size else 0.0,
        },
        "written_maps": written_maps,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _jsonable(summary)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-glob", required=True)
    parser.add_argument("--simulation-info", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--frame", choices=("recovar", "relion"), default="recovar")
    parser.add_argument("--no-write-maps", action="store_true")
    parser.add_argument(
        "--apply-simulation-scale",
        action="store_true",
        help="Apply simulation_info['scale_vol'] so GT volume amplitudes match generated particles.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_gt_weighted_ppca_init(
        volume_paths=[Path(path) for path in sorted(glob.glob(args.volume_glob))],
        simulation_info_path=Path(args.simulation_info),
        output_dir=Path(args.output_dir),
        q=int(args.q),
        frame=args.frame,
        write_maps=not args.no_write_maps,
        apply_simulation_scale=bool(args.apply_simulation_scale),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
