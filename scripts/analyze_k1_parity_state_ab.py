#!/usr/bin/env python3
"""Compare two RECOVAR iteration parity dumps at one immutable particle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _f32(value: float) -> dict[str, object]:
    rounded = np.float32(value)
    return {
        "value": float(rounded),
        "bits_hex": f"0x{struct.unpack('<I', rounded.tobytes())[0]:08x}",
    }


def _array_metrics(control: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(control)
    rhs = np.asarray(candidate)
    if lhs.shape != rhs.shape:
        raise ValueError(f"array shapes differ: {lhs.shape} != {rhs.shape}")
    delta = rhs.astype(np.float64) - lhs.astype(np.float64)
    lhs_norm = float(np.linalg.norm(lhs.astype(np.float64).reshape(-1)))
    return {
        "shape": list(lhs.shape),
        "bit_equal_fraction": float(np.mean(lhs == rhs)),
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "relative_l2": (
            float(np.linalg.norm(delta.reshape(-1)) / lhs_norm)
            if lhs_norm > 0.0
            else 0.0 if not np.any(delta) else float("inf")
        ),
    }


def analyze(
    *,
    control_path: Path,
    candidate_path: Path,
    half: int,
    original_index: int,
) -> dict[str, object]:
    prefix = f"half{half}_"
    with np.load(control_path, allow_pickle=False) as control, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        for scalar in ("iteration", "relion_iteration", "current_size"):
            if int(np.asarray(control[scalar]).item()) != int(np.asarray(candidate[scalar]).item()):
                raise ValueError(f"{scalar} differs between parity dumps")

        control_ids = np.asarray(control[prefix + "original_image_indices"], dtype=np.int64)
        candidate_ids = np.asarray(candidate[prefix + "original_image_indices"], dtype=np.int64)
        if not np.array_equal(control_ids, candidate_ids):
            raise ValueError("physical particle identity order differs between parity dumps")
        positions = np.flatnonzero(control_ids == int(original_index))
        if positions.size != 1:
            raise ValueError(
                f"expected original index {original_index} exactly once in half {half}, "
                f"found {positions.size}"
            )
        position = int(positions[0])

        target_fields: dict[str, object] = {}
        for name in (
            "wsum_norm_correction",
            "norm_corrections",
            "image_corrections",
            "scale_corrections",
            "max_posterior",
            "best_log_score",
            "hard_assignment",
            "coarse_hard_assignment",
        ):
            key = prefix + name
            control_value = np.asarray(control[key])[position].item()
            candidate_value = np.asarray(candidate[key])[position].item()
            record: dict[str, object] = {
                "control": control_value,
                "candidate": candidate_value,
                "candidate_minus_control": candidate_value - control_value,
            }
            if name in {
                "wsum_norm_correction",
                "norm_corrections",
                "image_corrections",
                "scale_corrections",
                "max_posterior",
                "best_log_score",
            }:
                record["control_f32"] = _f32(float(control_value))
                record["candidate_f32"] = _f32(float(candidate_value))
            target_fields[name] = record

        global_fields: dict[str, object] = {}
        for name in ("avg_norm_correction", "sumw", "Ft_y_total", "Ft_ctf_total"):
            key = prefix + name
            control_value = np.asarray(control[key]).item()
            candidate_value = np.asarray(candidate[key]).item()
            global_fields[name] = {
                "control": control_value,
                "candidate": candidate_value,
                "candidate_minus_control": candidate_value - control_value,
                "control_f32": _f32(float(control_value)),
                "candidate_f32": _f32(float(candidate_value)),
            }

        array_fields = {
            name: _array_metrics(control[name], candidate[name])
            for name in (
                "sigma2_noise",
                prefix + "wsum_sigma2_noise",
                prefix + "wsum_img_power",
                prefix + "norm_corrections",
                prefix + "image_corrections",
                prefix + "mean_real_ds",
                prefix + "unreg_mean_real_ds",
            )
        }

    image_delta = float(target_fields["image_corrections"]["candidate_minus_control"])
    norm_delta = float(target_fields["norm_corrections"]["candidate_minus_control"])
    avg_delta = float(global_fields["avg_norm_correction"]["candidate_minus_control"])
    scale_delta = float(target_fields["scale_corrections"]["candidate_minus_control"])
    if image_delta == 0.0:
        classification = "target scoring correction is unchanged"
    elif norm_delta != 0.0:
        classification = "target per-particle norm state changes before scoring"
    elif avg_delta != 0.0:
        classification = "half-average norm state changes before scoring"
    elif scale_delta != 0.0:
        classification = "target scale state changes before scoring"
    else:
        classification = "image correction changes at an unrecorded boundary"

    return {
        "schema": "recovar.em.k1_parity_state_ab.v1",
        "status": "complete",
        "classification": classification,
        "half": int(half),
        "original_index": int(original_index),
        "physical_position": position,
        "target": target_fields,
        "half_state": global_fields,
        "arrays": array_fields,
        "artifacts": {
            "control": str(control_path.resolve()),
            "control_sha256": _sha256(control_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--original-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        control_path=args.control,
        candidate_path=args.candidate,
        half=args.half,
        original_index=args.original_index,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
