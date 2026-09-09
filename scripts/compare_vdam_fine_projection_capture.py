#!/usr/bin/env python
"""Compare RECOVAR's CUDA projection of a captured RELION PPref operand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_flat(path: Path, dtype) -> np.ndarray:
    count = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    values = np.fromfile(path, dtype=dtype, offset=4)
    if values.size != count:
        raise ValueError(f"{path}: header says {count} values, found {values.size}")
    return values


def _complex_flat(capture: Path, stem: str) -> np.ndarray:
    real = _read_flat(capture / f"{stem}_real.bin", np.float64)
    imag = _read_flat(capture / f"{stem}_imag.bin", np.float64)
    return (real + 1j * imag).astype(np.complex64)


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    difference = actual - expected
    worst = np.argsort(np.abs(difference))[-4:][::-1]
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "unequal_count": int(np.count_nonzero(actual != expected)),
        "worst_pixels": [
            {
                "compact_index": int(index),
                "expected_real": float(expected[index].real),
                "expected_imag": float(expected[index].imag),
                "actual_real": float(actual[index].real),
                "actual_imag": float(actual[index].imag),
                "abs_error": float(abs(difference[index])),
            }
            for index in worst
            if difference[index] != 0
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion-capture", type=Path, required=True)
    parser.add_argument("--recovar-score-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument("--current-size", type=int, default=52)
    args = parser.parse_args()

    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    capture = args.relion_capture.resolve()
    score_dump = np.load(args.recovar_score_dump)
    dims = _read_flat(capture / "pass1_class0_ppref_dims.bin", np.int32)
    projector = _complex_flat(capture, "pass1_class0_ppref").reshape(
        int(dims[2]), int(dims[1]), int(dims[0])
    )
    rotations = np.asarray(score_dump["local_rotation_matrices"], dtype=np.float32)
    score_indices, _ = make_fourier_window_indices_np(
        (args.box_size, args.box_size), args.current_size
    )
    projected, _ = compute_relion_projector_projections_block(
        projector,
        rotations,
        (args.box_size, args.box_size),
        r_max=args.current_size // 2,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=args.current_size,
        pixel_indices=score_indices,
        relion_texture_interp=True,
        # InitialModel scores RELION's rounded current-size crop, including
        # its few pixels just outside the exact Euclidean-radius disk.
        mask_current_image_disk=False,
    )
    projected = np.asarray(projected)

    candidate_rotations = _read_flat(capture / "pass1_acc_rot_idx.bin", np.int32)
    native_all = _complex_flat(capture, "pass1_class0_fine_ref").reshape(
        candidate_rotations.size, args.current_size * (args.current_size // 2 + 1)
    )
    full_size = args.box_size
    native_take = []
    for full_index in score_indices:
        full_row, column = divmod(int(full_index), full_size // 2 + 1)
        ky = full_row - full_size // 2
        native_row = ky if ky >= 0 else ky + args.current_size
        native_take.append(native_row * (args.current_size // 2 + 1) + column)
    native_all = native_all[:, np.asarray(native_take, dtype=np.int32)] * (-(full_size**2))

    unique_metrics = {}
    for rotation_index in np.unique(candidate_rotations):
        candidate = int(np.flatnonzero(candidate_rotations == rotation_index)[0])
        unique_metrics[str(int(rotation_index))] = _metrics(
            native_all[candidate], projected[int(rotation_index)]
        )
    payload = {
        "relion_capture": str(capture),
        "recovar_score_dump": str(args.recovar_score_dump.resolve()),
        "projector_shape": list(projector.shape),
        "score_indices": score_indices.tolist(),
        "rotation_metrics": unique_metrics,
        "worst_relative_l2": max(value["relative_l2"] for value in unique_metrics.values()),
        "worst_max_abs": max(value["max_abs"] for value in unique_metrics.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
