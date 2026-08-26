#!/usr/bin/env python3
"""Replay a short captured K=1 BPref particle prefix through the CUDA kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from scripts.analyze_k1_bpref_prefixes import _metrics, _native_prefix, _require


SCHEMA = "recovar-k1-bpref-prefix-replay-v1"


def _image_shape(pixel_count: int) -> tuple[int, int]:
    for height in range(2, 4097, 2):
        if height * (height // 2 + 1) == pixel_count:
            return height, height
    raise ValueError(f"cannot infer an even FFTW image size from {pixel_count} pixels")


def _capture(directory: Path, half: int, original_index: int) -> dict:
    path = directory / f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz"
    _require(path.is_file(), f"missing RECOVAR operand capture {path}")
    with np.load(path, allow_pickle=False) as values:
        _require(
            str(np.asarray(values["schema"]).reshape(()))
            == "recovar-bpref-accumulator-delta-v2",
            "RECOVAR capture schema changed",
        )
        volume_shape = tuple(map(int, np.asarray(values["volume_shape"]).reshape(-1)))
        return {
            "path": str(path.resolve()),
            "ordinal": int(values["particle_launch_ordinal"]),
            "volume_shape": volume_shape,
            "image": np.asarray(values["operand_source_image"], dtype=np.complex64),
            "ctf": np.asarray(values["operand_ctf"], dtype=np.float32),
            "minvsigma2": np.asarray(values["operand_minvsigma2"], dtype=np.float32),
            "posterior": np.asarray(values["operand_posterior"], dtype=np.float32),
            "translation_angles": np.asarray(
                values["operand_translation_angles"], dtype=np.float32
            ),
            "eulers": np.asarray(values["operand_eulers"], dtype=np.float32),
            "threshold": np.asarray(values["operand_threshold"], dtype=np.float32).reshape(1),
            "weight_norm": np.asarray(values["operand_weight_norm"], dtype=np.float32).reshape(1),
            "image_shape": _image_shape(int(np.asarray(values["operand_source_image"]).size)),
            "max_r": float(np.asarray(values["max_r"]).reshape(())),
            "captured_after_data": -np.asarray(
                values["after_data"], dtype=np.complex64
            ).reshape(-1),
            "captured_after_weight": np.asarray(
                values["after_weight"], dtype=np.float32
            ).reshape(-1),
        }


def replay(
    selection_path: Path,
    native_directory: Path,
    recovar_directory: Path,
    half: int,
    max_particles: int,
    compare_final_only: bool = False,
) -> dict:
    selection = json.loads(selection_path.read_text())
    selected = selection[f"half{half}"]
    rows = list(
        zip(
            map(int, selected["half_local_ordinals"]),
            map(int, selected["original_indices"]),
            map(int, selected["native_internal_indices"]),
            strict=True,
        )
    )
    rows = [row for row in rows if row[0] < max_particles]
    _require(
        [row[0] for row in rows] == list(range(max_particles)),
        "prefix replay requires every consecutive particle from zero",
    )
    first = _capture(recovar_directory, half, rows[0][1])
    volume_shape = first["volume_shape"]
    volume_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    data = jnp.zeros(volume_size, dtype=jnp.complex64)
    weight = jnp.zeros(volume_size, dtype=jnp.float32)
    reports = []
    for ordinal, original_index, internal_index in rows:
        capture = _capture(recovar_directory, half, original_index)
        _require(capture["ordinal"] == ordinal, "capture ordinal changed")
        _require(capture["volume_shape"] == volume_shape, "capture volume shape changed")
        data, weight = cuda_backproject.relion_firstiter_bpref_fused_x_half(
            data,
            weight,
            jnp.asarray(capture["image"]),
            jnp.asarray(capture["ctf"]),
            jnp.asarray(capture["minvsigma2"]),
            jnp.asarray(capture["posterior"]),
            jnp.asarray(capture["translation_angles"]),
            jnp.asarray(capture["eulers"]),
            jnp.asarray(capture["threshold"]),
            jnp.asarray(capture["weight_norm"]),
            capture["image_shape"],
            volume_shape,
            capture["max_r"],
        )
        if compare_final_only and ordinal + 1 != max_particles:
            continue
        replay_data = -np.asarray(data, dtype=np.complex64).reshape(-1)
        replay_weight = np.asarray(weight, dtype=np.float32).reshape(-1)
        native = _native_prefix(native_directory, half, internal_index)
        reports.append(
            {
                "particle_count": ordinal + 1,
                "original_index": original_index,
                "native_internal_index": internal_index,
                "capture": capture["path"],
                "replay_vs_native": {
                    "data": _metrics(replay_data, native["data"]),
                    "weight": _metrics(replay_weight, native["weight"]),
                },
                "replay_vs_captured_recovar": {
                    "data": _metrics(replay_data, capture["captured_after_data"]),
                    "weight": _metrics(replay_weight, capture["captured_after_weight"]),
                },
            }
        )
    report = {
        "schema": SCHEMA,
        "selection": str(selection_path.resolve()),
        "native_directory": str(native_directory.resolve()),
        "recovar_directory": str(recovar_directory.resolve()),
        "half": half,
        "max_particles": max_particles,
        "rows": reports,
    }
    if compare_final_only:
        report["comparison_scope"] = "final_prefix_only"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-directory", required=True, type=Path)
    parser.add_argument("--half", required=True, type=int, choices=(1, 2))
    parser.add_argument("--max-particles", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--compare-final-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _require(args.repetitions > 0, "repetitions must be positive")
    reports = [
        replay(
            args.selection,
            args.native_directory,
            args.recovar_directory,
            args.half,
            args.max_particles,
            args.compare_final_only,
        )
        for _ in range(args.repetitions)
    ]
    report = reports[0] if args.repetitions == 1 else {
        "schema": "recovar-k1-bpref-prefix-replay-repeats-v1",
        "repetitions": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
