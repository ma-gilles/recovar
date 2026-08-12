#!/usr/bin/env python3
"""Join one native RELION coarse-weight capture to a RECOVAR significance dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table  # noqa: E402


@dataclass(frozen=True)
class NativeCoarseCapture:
    header: np.ndarray
    raw_diff2: np.ndarray
    orientation_prior: np.ndarray
    translation_prior: np.ndarray
    orientation_zero: np.ndarray
    translation_zero: np.ndarray
    preexponent: np.ndarray
    postexponent: np.ndarray
    sorted_weights: np.ndarray
    cumulative_weights: np.ndarray


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_native_coarse_capture(path: Path) -> NativeCoarseCapture:
    data = path.read_bytes()
    if data[:16].rstrip(b"\0") != b"RLNCOARSEV1":
        raise ValueError("not a RELION coarse-v1 capture")
    header = np.frombuffer(data, dtype="<u8", count=32, offset=16).copy()
    if int(header[0]) != 1 or int(header[15]) != 4:
        raise ValueError("unsupported RELION coarse capture schema or XFLOAT size")
    count = int(header[6])
    filtered_count = int(header[7])
    n_rot = int(header[16])
    n_trans = int(header[17])
    if count != int(header[4]) * int(header[5]) or n_rot * n_trans != count:
        raise ValueError("RELION coarse capture dimensions are inconsistent")
    offset = 16 + 32 * 8

    def floats(length: int) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(data, dtype="<f4", count=length, offset=offset).copy()
        offset += length * 4
        return result

    def bytes_(length: int) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(data, dtype="u1", count=length, offset=offset).copy()
        offset += length
        return result

    result = NativeCoarseCapture(
        header=header,
        raw_diff2=floats(count),
        orientation_prior=floats(n_rot),
        translation_prior=floats(n_trans),
        orientation_zero=bytes_(n_rot).astype(bool),
        translation_zero=bytes_(n_trans).astype(bool),
        preexponent=floats(count),
        postexponent=floats(count),
        sorted_weights=floats(filtered_count),
        cumulative_weights=floats(filtered_count),
    )
    if offset != len(data):
        raise ValueError(f"RELION coarse capture has {len(data) - offset} trailing bytes")
    return result


def _target_record(
    *,
    native: NativeCoarseCapture,
    recovar_path: Path,
    native_rotation: int,
    native_translation: int,
    recovar_rotation: int,
    recovar_translation: int,
) -> dict[str, object]:
    native_n_trans = int(native.header[5])
    native_index = native_rotation * native_n_trans + native_translation
    threshold = _float32_from_bits(int(native.header[13]))
    native_descending_rank = int(
        np.count_nonzero(native.postexponent > native.postexponent[native_index])
    )
    with np.load(recovar_path, allow_pickle=False) as recovar:
        recovar_n_rot = int(np.asarray(recovar["n_rot"]).item())
        recovar_n_trans = int(np.asarray(recovar["n_trans"]).item())
        recovar_index = recovar_rotation * recovar_n_trans + recovar_translation
        recovar_pre = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32).reshape(-1)
        recovar_with = np.asarray(recovar["scores_with_prior_per_class"], dtype=np.float32).reshape(-1)
        recovar_weights = np.asarray(recovar["weights_full"], dtype=np.float32).reshape(-1)
        recovar_mask = np.asarray(recovar["significant_mask"], dtype=bool).reshape(-1)
        rotation_prior = np.asarray(recovar["rotation_log_prior"], dtype=np.float32).reshape(-1)
        translation_prior = np.asarray(recovar["translation_log_prior"], dtype=np.float32).reshape(-1)
    if recovar_pre.size != recovar_n_rot * recovar_n_trans:
        raise ValueError("RECOVAR coarse dump dimensions are inconsistent")
    recovar_descending_rank = int(np.count_nonzero(recovar_weights > recovar_weights[recovar_index]))
    native_pre_no_prior = (
        _float32_from_bits(int(native.header[10])) - float(native.raw_diff2[native_index])
    )
    native_pre_relative = float(native.preexponent[native_index] - np.max(native.preexponent))
    recovar_pre_relative = float(recovar_with[recovar_index] - np.max(recovar_with))
    return {
        "coordinates": {
            "native_rotation": native_rotation,
            "native_translation": native_translation,
            "recovar_rotation": recovar_rotation,
            "recovar_translation": recovar_translation,
        },
        "native": {
            "raw_diff2": float(native.raw_diff2[native_index]),
            "score_without_prior": native_pre_no_prior,
            "orientation_prior": float(native.orientation_prior[native_rotation]),
            "translation_prior": float(native.translation_prior[native_translation]),
            "preexponent": float(native.preexponent[native_index]),
            "preexponent_relative_to_best": native_pre_relative,
            "postexponent_weight": float(native.postexponent[native_index]),
            "significant_threshold": threshold,
            "selected": bool(native.postexponent[native_index] >= threshold),
            "descending_rank_zero_based": native_descending_rank,
        },
        "recovar": {
            "score_without_prior": float(recovar_pre[recovar_index]),
            "orientation_prior": float(rotation_prior[recovar_rotation]),
            "translation_prior": float(translation_prior[recovar_translation]),
            "score_with_prior": float(recovar_with[recovar_index]),
            "score_with_prior_relative_to_best": recovar_pre_relative,
            "posterior_weight": float(recovar_weights[recovar_index]),
            "selected": bool(recovar_mask[recovar_index]),
            "descending_rank_zero_based": recovar_descending_rank,
        },
        "relative_preexponent_recovar_minus_native": recovar_pre_relative - native_pre_relative,
    }


def analyze(
    *,
    native_path: Path,
    recovar_path: Path,
    native_rotation: int,
    native_translation: int,
    recovar_rotation: int,
    recovar_translation: int,
    native_directions: int,
    native_psi: int,
) -> dict[str, object]:
    native = load_native_coarse_capture(native_path)
    threshold_index = int(native.header[8])
    native_count = int(native.header[6])
    threshold = _float32_from_bits(int(native.header[13]))
    selected = native.postexponent >= threshold
    if native_directions * native_psi != int(native.header[4]):
        raise ValueError("native direction/psi factors do not match the captured rotation count")

    def mapped(values: np.ndarray) -> np.ndarray:
        table = np.asarray(values).reshape(
            int(native.header[16]), int(native.header[17])
        )
        return _map_relion_table(
            table,
            n_directions=native_directions,
            n_psi=native_psi,
            relion_to_recovar_translation=np.arange(int(native.header[5]), dtype=np.int64),
        )

    native_raw = mapped(native.raw_diff2)
    native_pre = mapped(native.preexponent)
    native_probability = mapped(native.postexponent) / np.float32(
        _float32_from_bits(int(native.header[12]))
    )
    native_mask = mapped(selected).astype(bool)
    with np.load(recovar_path, allow_pickle=False) as recovar:
        recovar_raw = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)[0]
        recovar_pre = np.asarray(recovar["scores_with_prior_per_class"], dtype=np.float32)[0]
        recovar_probability = np.asarray(
            recovar["weights_per_class"], dtype=np.float32
        )[0].reshape(recovar_raw.shape)
        recovar_mask = np.asarray(recovar["significant_mask"], dtype=bool).reshape(
            recovar_raw.shape
        )
        recovar_rotation_prior = np.asarray(
            recovar["rotation_log_prior"], dtype=np.float32
        ).reshape(-1)
        recovar_translation_prior = np.asarray(
            recovar["translation_log_prior"], dtype=np.float32
        ).reshape(-1)
    native_rotation_prior = _map_relion_table(
        native.orientation_prior.reshape(int(native.header[16]), 1),
        n_directions=native_directions,
        n_psi=native_psi,
        relion_to_recovar_translation=np.asarray([0], dtype=np.int64),
    )[:, 0]
    native_rotation_zero = _map_relion_table(
        native.orientation_zero.reshape(int(native.header[16]), 1),
        n_directions=native_directions,
        n_psi=native_psi,
        relion_to_recovar_translation=np.asarray([0], dtype=np.int64),
    )[:, 0].astype(bool)
    raw_valid = (
        (native_raw != -np.finfo(np.float32).max)
        & np.isfinite(native_raw)
        & np.isfinite(recovar_raw)
    )
    combined_valid = raw_valid & np.isfinite(native_pre) & np.isfinite(recovar_pre)
    raw_offset = np.median(
        recovar_raw[raw_valid].astype(np.float64)
        + native_raw[raw_valid].astype(np.float64)
    )
    raw_residual = (
        recovar_raw[raw_valid].astype(np.float64)
        + native_raw[raw_valid].astype(np.float64)
        - raw_offset
    )
    recovar_pre_max = np.max(recovar_pre[combined_valid])
    native_pre_max = np.max(native_pre[combined_valid])
    pre_residual = (
        recovar_pre[combined_valid].astype(np.float64) - float(recovar_pre_max)
    ) - (native_pre[combined_valid].astype(np.float64) - float(native_pre_max))
    probability_residual = (
        recovar_probability.astype(np.float64) - native_probability.astype(np.float64)
    )
    mask_mismatches = np.argwhere(recovar_mask != native_mask)
    finite_orientation_prior = (
        (~native_rotation_zero)
        & np.isfinite(native_rotation_prior)
        & np.isfinite(recovar_rotation_prior)
    )
    orientation_prior_residual = (
        recovar_rotation_prior[finite_orientation_prior].astype(np.float64)
        - native_rotation_prior[finite_orientation_prior].astype(np.float64)
    )
    translation_prior_residual = (
        recovar_translation_prior.astype(np.float64)
        - native.translation_prior.astype(np.float64)
    )

    native_best = np.unravel_index(
        np.argmax(np.where(combined_valid, native_pre, -np.inf)), native_pre.shape
    )
    recovar_best = np.unravel_index(
        np.argmax(np.where(combined_valid, recovar_pre, -np.inf)), recovar_pre.shape
    )
    recovar_target = (recovar_rotation, recovar_translation)
    target_raw_relative_recovar = float(
        recovar_raw[recovar_target] - recovar_raw[recovar_best]
    )
    target_raw_relative_native = float(
        -(native_raw[recovar_target] - native_raw[native_best])
    )

    def residual_metrics(values: np.ndarray) -> dict[str, float]:
        absolute = np.abs(np.asarray(values, dtype=np.float64))
        return {
            "median_abs": float(np.median(absolute)),
            "p95_abs": float(np.percentile(absolute, 95)),
            "max_abs": float(np.max(absolute)),
        }

    return {
        "schema": "recovar.em.k1_native_coarse_boundary.v1",
        "status": "complete",
        "artifacts": {
            "native": str(native_path.resolve()),
            "native_sha256": _sha256(native_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
        "native_summary": {
            "iteration": int(native.header[1]),
            "stack_index_one_based": int(native.header[2]),
            "part_id": int(native.header[3]),
            "candidate_count": native_count,
            "filtered_positive_count": int(native.header[7]),
            "threshold_index_ascending_zero_based": threshold_index,
            "cutoff_count": int(native.header[7]) - threshold_index,
            "serialized_cutoff_count": int(native.header[9]),
            "selected_count": int(np.count_nonzero(selected)),
            "sum_weight": _float32_from_bits(int(native.header[12])),
            "significant_weight": threshold,
            "adaptive_fraction": _float32_from_bits(int(native.header[14])),
            "scan_last": float(native.cumulative_weights[-1]),
            "scan_before_threshold": (
                None if threshold_index == 0 else float(native.cumulative_weights[threshold_index - 1])
            ),
            "scan_at_threshold": float(native.cumulative_weights[threshold_index]),
        },
        "full_boundary": {
            "candidate_topology_exact": bool(native_raw.shape == recovar_raw.shape),
            "raw_valid_count": int(np.count_nonzero(raw_valid)),
            "raw_score_centered_residual": residual_metrics(raw_residual),
            "preexponent_centered_residual": residual_metrics(pre_residual),
            "orientation_prior_support_exact": bool(
                np.array_equal(native_rotation_zero, ~np.isfinite(recovar_rotation_prior))
            ),
            "orientation_prior_residual": residual_metrics(orientation_prior_residual),
            "translation_prior_residual": residual_metrics(translation_prior_residual),
            "posterior_residual": residual_metrics(probability_residual),
            "posterior_total_variation": float(
                0.5 * np.sum(np.abs(probability_residual), dtype=np.float64)
            ),
            "support_mismatch_count": int(mask_mismatches.shape[0]),
            "support_mismatches_first": mask_mismatches[:64].astype(int).tolist(),
        },
        "shared_best": {
            "native_recovar_coordinates": [int(native_best[0]), int(native_best[1])],
            "recovar_coordinates": [int(recovar_best[0]), int(recovar_best[1])],
            "exact": bool(native_best == recovar_best),
        },
        "target": _target_record(
            native=native,
            recovar_path=recovar_path,
            native_rotation=native_rotation,
            native_translation=native_translation,
            recovar_rotation=recovar_rotation,
            recovar_translation=recovar_translation,
        )
        | {
            "raw_score_relative_to_best_recovar": target_raw_relative_recovar,
            "raw_score_relative_to_best_native": target_raw_relative_native,
            "raw_score_relative_recovar_minus_native": (
                target_raw_relative_recovar - target_raw_relative_native
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native-rotation", type=int, required=True)
    parser.add_argument("--native-translation", type=int, required=True)
    parser.add_argument("--recovar-rotation", type=int, required=True)
    parser.add_argument("--recovar-translation", type=int, required=True)
    parser.add_argument("--native-directions", type=int, required=True)
    parser.add_argument("--native-psi", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        native_path=args.native,
        recovar_path=args.recovar,
        native_rotation=args.native_rotation,
        native_translation=args.native_translation,
        recovar_rotation=args.recovar_rotation,
        recovar_translation=args.recovar_translation,
        native_directions=args.native_directions,
        native_psi=args.native_psi,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
