#!/usr/bin/env python3
"""Localize a native RELION/RECOVAR K=1 fine-score residual by operand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)


def _flat_memmap(path: Path, dtype=np.float64) -> np.memmap:
    count = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    return np.memmap(path, dtype=dtype, mode="r", offset=4, shape=(count,))


def _tree_sum(values: np.ndarray) -> np.ndarray:
    """Reproduce RELION's 256-lane float32 fine-score reduction."""

    values = np.asarray(values, dtype=np.float32)
    lanes = np.zeros(values.shape[:-1] + (256,), dtype=np.float32)
    for start in range(0, values.shape[-1], 256):
        width = min(256, values.shape[-1] - start)
        lanes[..., :width] += values[..., start : start + width]
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    return lanes[..., 0]


def _diff2(reference: np.ndarray, shifted: np.ndarray, weight: np.ndarray) -> np.ndarray:
    real = np.asarray(reference.real - shifted.real, dtype=np.float32)
    imag = np.asarray(reference.imag - shifted.imag, dtype=np.float32)
    pixels = np.asarray(real * real + imag * imag, dtype=np.float32)
    pixels = np.asarray(pixels * np.float32(0.5), dtype=np.float32)
    pixels = np.asarray(pixels * np.asarray(weight, dtype=np.float32), dtype=np.float32)
    return _tree_sum(pixels)


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(left).reshape(-1)))
    return float(np.linalg.norm((np.asarray(left) - np.asarray(right)).reshape(-1)) / denominator)


def _full_to_compact(window_indices: np.ndarray, *, full_size: int, current_size: int) -> np.ndarray:
    full_half = full_size // 2 + 1
    current_half = current_size // 2 + 1
    window_indices = np.asarray(window_indices, dtype=np.int64)
    rows = window_indices // full_half
    columns = window_indices % full_half
    ky = rows - full_size // 2
    relion_rows = np.where(ky < 0, ky + current_size, ky)
    relion_indices = relion_rows * current_half + columns
    lookup = np.full(current_size * current_half, -1, dtype=np.int32)
    lookup[relion_indices] = np.arange(window_indices.size, dtype=np.int32)
    return lookup


def analyze(
    dump_dir: Path,
    recovar_npz: Path,
    *,
    full_image_size: int,
    chunk_size: int,
) -> dict:
    dump_dir = Path(dump_dir)
    with np.load(recovar_npz, allow_pickle=False) as payload:
        rec = {name: np.array(payload[name]) for name in payload.files}

    native_eulers = _flat_memmap(dump_dir / "pass1_class0_fine_eulers.bin").reshape(-1, 3, 3)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        native_eulers,
        rec["rotations"],
    )
    native_rotation_row = np.asarray(_flat_memmap(dump_dir / "pass1_acc_rot_idx.bin", np.int32))
    translation = np.asarray(_flat_memmap(dump_dir / "pass1_acc_trans_idx.bin", np.int32))
    rec_rotation_row = nearest[native_rotation_row]
    native_raw = np.asarray(_flat_memmap(dump_dir / "pass1_exp_Mweight_raw_preprior.bin"))
    candidate_count = native_raw.size
    if not (native_rotation_row.size == translation.size == candidate_count):
        raise ValueError("native candidate arrays have different lengths")

    rec_mask = np.asarray(rec["candidate_mask"], dtype=bool)
    expected_keys = set(map(tuple, np.argwhere(rec_mask)))
    native_keys = set(zip(rec_rotation_row.tolist(), translation.tolist(), strict=True))
    if native_keys != expected_keys:
        raise ValueError(
            f"candidate keys differ: native={len(native_keys)} RECOVAR={len(expected_keys)} "
            f"common={len(native_keys & expected_keys)}"
        )

    current_size = int(rec["current_size"])
    native_pixel_count = current_size * (current_size // 2 + 1)
    lookup = _full_to_compact(
        rec["window_indices"],
        full_size=full_image_size,
        current_size=current_size,
    )
    valid = lookup >= 0
    compact = lookup[valid]
    scale = np.float32(full_image_size**2)

    native_corr = np.asarray(_flat_memmap(dump_dir / "pass1_img0_corr_img.bin"), dtype=np.float32)
    if native_corr.size != native_pixel_count:
        raise ValueError("native corr_img has the wrong current-size topology")
    native_weight = native_corr / np.float32(full_image_size**4)
    rec_weight = np.zeros(native_pixel_count, dtype=np.float32)
    rec_weight[valid] = np.asarray(
        rec["ctf2_over_nv_score"] * rec["half_weights"],
        dtype=np.float32,
    )[compact]

    native_ref_real = _flat_memmap(dump_dir / "pass1_class0_fine_ref_real.bin")
    native_ref_imag = _flat_memmap(dump_dir / "pass1_class0_fine_ref_imag.bin")
    native_shift_real = _flat_memmap(dump_dir / "pass1_class0_fine_shifted_real.bin")
    native_shift_imag = _flat_memmap(dump_dir / "pass1_class0_fine_shifted_imag.bin")
    expected_values = candidate_count * native_pixel_count
    if any(item.size != expected_values for item in (native_ref_real, native_ref_imag, native_shift_real, native_shift_imag)):
        raise ValueError("native fine operand tensor size does not match candidate topology")

    labels = (
        "native",
        "recovar_reference_only",
        "recovar_shifted_image_only",
        "recovar_weight_only",
        "recovar_reference_and_shifted_image",
        "recovar_reference_and_weight",
        "recovar_shifted_image_and_weight",
        "recovar_all",
    )
    costs = {label: np.empty(candidate_count, dtype=np.float32) for label in labels}
    operand_error = {"reference_num": 0.0, "reference_den": 0.0, "shifted_num": 0.0, "shifted_den": 0.0}

    for start in range(0, candidate_count, chunk_size):
        stop = min(candidate_count, start + chunk_size)
        shape = (stop - start, native_pixel_count)
        sl = slice(start * native_pixel_count, stop * native_pixel_count)
        native_reference = -scale * (
            np.asarray(native_ref_real[sl], dtype=np.float32).reshape(shape)
            + np.complex64(1j) * np.asarray(native_ref_imag[sl], dtype=np.float32).reshape(shape)
        )
        native_shifted = -scale * (
            np.asarray(native_shift_real[sl], dtype=np.float32).reshape(shape)
            + np.complex64(1j) * np.asarray(native_shift_imag[sl], dtype=np.float32).reshape(shape)
        )
        rec_reference = np.zeros(shape, dtype=np.complex64)
        rec_shifted = np.zeros(shape, dtype=np.complex64)
        rec_reference[:, valid] = rec["proj_half"][rec_rotation_row[start:stop]][:, compact]
        rec_shifted[:, valid] = rec["shifted_corrected"][translation[start:stop]][:, compact]

        # Pixels outside RECOVAR's compact radial window have zero score
        # weight. Including their unconstrained projected values would make a
        # raw operand norm look large despite contributing exactly zero diff2.
        operand_error["reference_num"] += float(
            np.sum(np.abs(native_reference[:, valid] - rec_reference[:, valid]) ** 2)
        )
        operand_error["reference_den"] += float(np.sum(np.abs(native_reference[:, valid]) ** 2))
        operand_error["shifted_num"] += float(
            np.sum(np.abs(native_shifted[:, valid] - rec_shifted[:, valid]) ** 2)
        )
        operand_error["shifted_den"] += float(np.sum(np.abs(native_shifted[:, valid]) ** 2))

        arms = {
            "native": (native_reference, native_shifted, native_weight),
            "recovar_reference_only": (rec_reference, native_shifted, native_weight),
            "recovar_shifted_image_only": (native_reference, rec_shifted, native_weight),
            "recovar_weight_only": (native_reference, native_shifted, rec_weight),
            "recovar_reference_and_shifted_image": (rec_reference, rec_shifted, native_weight),
            "recovar_reference_and_weight": (rec_reference, native_shifted, rec_weight),
            "recovar_shifted_image_and_weight": (native_reference, rec_shifted, rec_weight),
            "recovar_all": (rec_reference, rec_shifted, rec_weight),
        }
        for label, operands in arms.items():
            costs[label][start:stop] = _diff2(*operands)

    rec_score = np.asarray(rec["scores_pre_prior"], dtype=np.float64)[rec_rotation_row, translation]
    captured_residual = _center(rec_score + native_raw)
    native_replay_error = _center(costs["native"] - native_raw)
    recovar_replay_error = _center(-costs["recovar_all"] - rec_score)
    baseline = _center(costs["recovar_all"] - costs["native"])
    baseline_energy = float(np.sum(baseline * baseline))
    interventions = {}
    for label in labels:
        residual = _center(costs["recovar_all"] - costs[label])
        energy = float(np.sum(residual * residual))
        interventions[label] = {
            "residual": _stats(residual),
            "centered_energy": energy,
            "baseline_energy_removal_fraction": 1.0 - energy / baseline_energy,
        }

    native_translation = np.stack(
        [
            np.asarray(_flat_memmap(dump_dir / "pass1_candidate_translation_x.bin")),
            np.asarray(_flat_memmap(dump_dir / "pass1_candidate_translation_y.bin")),
        ],
        axis=1,
    )
    rec_translation = np.asarray(rec["fine_translations"])[translation]
    return {
        "schema": "em-k1-native-fine-operands-v1",
        "status": "complete",
        "relion_part_id": int(np.fromfile(dump_dir / "pass1_acc_part_id.bin", dtype=np.float64, count=1)[0]),
        "recovar_original_index": int(rec["original_index"]),
        "candidate_count": int(candidate_count),
        "candidate_keys_exact": True,
        "rotation_matrix_orientation": orientation,
        "rotation_matrix_median_frobenius": float(np.median(rotation_distance)),
        "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
        "translation_max_abs": float(np.max(np.abs(native_translation - rec_translation))),
        "operand_relative_l2": {
            "projected_reference": float(np.sqrt(operand_error["reference_num"] / operand_error["reference_den"])),
            "shifted_corrected_image": float(np.sqrt(operand_error["shifted_num"] / operand_error["shifted_den"])),
            "pixel_weight": _relative_l2(native_weight, rec_weight),
        },
        "captured_centered_score_residual": _stats(captured_residual),
        "native_cost_replay_error": _stats(native_replay_error),
        "recovar_cost_replay_error": _stats(recovar_replay_error),
        "interventions": interventions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-pass2-npz", required=True, type=Path)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(
        args.relion_dump_dir,
        args.recovar_pass2_npz,
        full_image_size=args.full_image_size,
        chunk_size=args.chunk_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
