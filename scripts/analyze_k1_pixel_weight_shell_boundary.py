#!/usr/bin/env python3
"""Compare native RELION and RECOVAR coarse pixel weights shell by shell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_values_on_recovar_window,
)
from scripts.analyze_k1_native_reference_score_counterfactual import (
    _pixel_weight_shell_stats,
)
from scripts.validate_relion_coarse_operand_capture import load_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(operands_path: Path, recovar_path: Path, *, image_size: int) -> dict[str, object]:
    operands = load_artifact(operands_path)
    with np.load(recovar_path, allow_pickle=False) as payload:
        original_index = int(payload["original_index"])
        current_size = int(payload["current_size"])
        score_indices = np.asarray(payload["coarse_gaussian_score_indices"], dtype=np.int32)
        recovar_weight = np.asarray(payload["coarse_gaussian_pixel_weight"], dtype=np.float32)
    if operands.stack_index - 1 != original_index:
        raise ValueError("cross-engine particle identity mismatch")
    native_correction = relion_values_on_recovar_window(
        operands.correction[np.newaxis, :],
        score_indices,
        full_image_size=image_size,
        current_size=current_size,
    )[0].real.astype(np.float32)
    native_weight = (native_correction / np.float32(image_size**4)).astype(np.float32)
    delta = recovar_weight.astype(np.float64) - native_weight.astype(np.float64)
    return {
        "schema": "recovar.em.k1_pixel_weight_shell_boundary.v1",
        "particle": {
            "part_id": operands.part_id,
            "stack_index_one_based": operands.stack_index,
            "original_index_zero_based": original_index,
        },
        "image_size": image_size,
        "current_size": current_size,
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(native_weight.astype(np.float64))),
        "shell_boundary": _pixel_weight_shell_stats(
            recovar_weight,
            native_weight,
            score_indices,
            (image_size, image_size),
        ),
        "artifacts": {
            "operands": str(operands_path.resolve()),
            "operands_sha256": _sha256(operands_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(args.operands, args.recovar, image_size=args.image_size)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
