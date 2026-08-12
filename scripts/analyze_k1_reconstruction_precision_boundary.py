#!/usr/bin/env python3
"""Measure the K=1 reconstruction precision boundary against native RELION."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from recovar.utils import helpers
from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_native_real(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"truncated native reconstruction dump: {path}")
    z, y, x, *_ = struct.unpack_from("<8q", raw)
    expected = 64 + z * y * x * np.dtype(np.float64).itemsize
    if len(raw) != expected:
        raise ValueError(f"native reconstruction dump size mismatch: {path}")
    return np.frombuffer(raw, dtype=np.float64, offset=64).reshape((z, y, x)).copy()


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    denominator = float(np.linalg.norm(expected))
    if denominator == 0.0:
        raise ValueError("reference has zero norm")
    return float(np.linalg.norm(actual - expected) / denominator)


def _fsc_auc(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(normalized_fsc_auc(np.asarray(shell_fsc(actual, expected), dtype=np.float64)))


def _premask_metrics(path: Path, native_recovar: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        fourier = np.asarray(archive["means_premask"])
        real = np.asarray(archive["means_premask_real"])
    return {
        "fourier_dtype": str(fourier.dtype),
        "real_dtype": str(real.dtype),
        "relative_l2": _relative_l2(real, native_recovar),
        "max_absolute": float(np.max(np.abs(real - native_recovar))),
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-reconstruct-dir", type=Path, required=True)
    parser.add_argument("--native-relion-dir", type=Path, required=True)
    parser.add_argument("--baseline-premask-dir", type=Path, required=True)
    parser.add_argument("--candidate-premask-dir", type=Path, required=True)
    parser.add_argument("--baseline-intermediates", type=Path, required=True)
    parser.add_argument("--candidate-intermediates", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    halves: dict[str, object] = {}
    for half in (1, 2):
        native_dump = (
            args.native_reconstruct_dir
            / f"reconstruct_rank{half:02d}_call0000_volume_after_gridding.bin"
        )
        native_real = _load_native_real(native_dump)
        native_recovar = np.asarray(helpers.relion_volume_to_recovar(native_real), dtype=np.float64)
        baseline_premask = args.baseline_premask_dir / f"recovar_premask_it001_half{half}.npz"
        candidate_premask = args.candidate_premask_dir / f"recovar_premask_it001_half{half}.npz"
        baseline_boundary = _premask_metrics(baseline_premask, native_recovar)
        candidate_boundary = _premask_metrics(candidate_premask, native_recovar)

        native_map_path = args.native_relion_dir / f"run_it001_half{half}_class001.mrc"
        baseline_map_path = args.baseline_intermediates / f"it000_half{half}_reg.mrc"
        candidate_map_path = args.candidate_intermediates / f"it000_half{half}_reg.mrc"
        native_map = np.asarray(helpers.load_relion_volume(str(native_map_path)), dtype=np.float64)
        baseline_map = np.asarray(helpers.load_mrc(str(baseline_map_path)), dtype=np.float64)
        candidate_map = np.asarray(helpers.load_mrc(str(candidate_map_path)), dtype=np.float64)
        baseline_rel = _relative_l2(baseline_map, native_map)
        candidate_rel = _relative_l2(candidate_map, native_map)

        halves[str(half)] = {
            "pre_mask_reconstruction": {
                "baseline": baseline_boundary,
                "candidate": candidate_boundary,
                "relative_l2_improvement_factor": (
                    baseline_boundary["relative_l2"] / candidate_boundary["relative_l2"]
                ),
            },
            "post_lowpass_mask_map": {
                "baseline": {
                    "relative_l2": baseline_rel,
                    "signed_fsc_auc_non_dc": _fsc_auc(baseline_map, native_map),
                    "sha256": _sha256(baseline_map_path),
                },
                "candidate": {
                    "relative_l2": candidate_rel,
                    "signed_fsc_auc_non_dc": _fsc_auc(candidate_map, native_map),
                    "sha256": _sha256(candidate_map_path),
                },
                "relative_l2_improvement_factor": baseline_rel / candidate_rel,
            },
            "native_artifacts": {
                "reconstruction_dump": str(native_dump.resolve()),
                "reconstruction_dump_sha256": _sha256(native_dump),
                "numbered_map": str(native_map_path.resolve()),
                "numbered_map_sha256": _sha256(native_map_path),
            },
        }

    report = {
        "schema": "recovar.em.k1_reconstruction_precision_boundary.v1",
        "metric_policy": "scale-sensitive relative-L2 and signed non-DC FSC-AUC; no correlation gate",
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
