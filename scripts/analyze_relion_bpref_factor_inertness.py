#!/usr/bin/env python3
"""Qualify a selected-factor RELION capture against a sealed repeat envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mrcfile
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bpref(path: Path, *, complex_values: bool) -> np.ndarray:
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
        if shape.size != 3 or np.any(shape <= 0):
            raise ValueError(f"invalid BPref shape header: {path}")
        values = np.fromfile(stream, dtype=np.complex128 if complex_values else np.float64)
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"BPref payload count mismatch: {path}")
    return values.reshape(tuple(int(value) for value in shape))


def _array_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError("inertness array contract changed")
    left = lhs.astype(np.complex128, copy=False)
    right = rhs.astype(np.complex128, copy=False)
    delta = right - left
    denominator = max(float(np.linalg.norm(left)), np.finfo(np.float64).tiny)
    return {
        "shape": list(lhs.shape),
        "dtype": str(lhs.dtype),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "mismatch_count": int(np.count_nonzero(lhs != rhs)),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.shape != rhs.shape or lhs.ndim != 3 or len(set(lhs.shape)) != 1:
        raise ValueError("expected equal cubic maps")
    n = lhs.shape[0]
    left = np.fft.rfftn(np.asarray(lhs, dtype=np.float64))
    right = np.fft.rfftn(np.asarray(rhs, dtype=np.float64))
    full = np.fft.fftfreq(n) * n
    half = np.fft.rfftfreq(n) * n
    yy, xx = np.meshgrid(full, half, indexing="ij")
    packed_weights = np.full(half.shape, 2.0, dtype=np.float64)
    packed_weights[0] = 1.0
    if n % 2 == 0:
        packed_weights[-1] = 1.0
    _, shell_weights = np.meshgrid(full, packed_weights, indexing="ij")
    numerator = np.zeros(n // 2 + 1, dtype=np.float64)
    lhs_power = np.zeros_like(numerator)
    rhs_power = np.zeros_like(numerator)
    for z_index, z_frequency in enumerate(full):
        shell = np.rint(np.sqrt(z_frequency**2 + yy**2 + xx**2)).astype(np.int32)
        keep = shell <= n // 2
        indices = shell[keep].reshape(-1)
        weights = shell_weights[keep].reshape(-1)
        cross = (left[z_index] * np.conj(right[z_index]))[keep].reshape(-1)
        numerator += np.bincount(indices, weights=cross.real * weights, minlength=n // 2 + 1)
        lhs_power += np.bincount(
            indices,
            weights=(np.abs(left[z_index][keep]) ** 2).reshape(-1) * weights,
            minlength=n // 2 + 1,
        )
        rhs_power += np.bincount(
            indices,
            weights=(np.abs(right[z_index][keep]) ** 2).reshape(-1) * weights,
            minlength=n // 2 + 1,
        )
    denominator = np.sqrt(lhs_power * rhs_power)
    return np.clip(
        np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0),
        -1.0,
        1.0,
    )


def _map_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    fsc = _shell_fsc(lhs, rhs)
    finite = fsc[1:][np.isfinite(fsc[1:])]
    if finite.size == 0:
        raise ValueError("map FSC has no finite non-DC shells")
    return {
        "fsc": fsc.tolist(),
        "fsc_auc_non_dc": float(np.mean(finite)),
        "fsc_min_non_dc": float(np.min(finite)),
    }


def _within_envelope(value: float, reference: float, multiplier: float) -> bool:
    if reference == 0:
        return value == 0
    return value <= multiplier * reference


def analyze(
    control_root: Path,
    capture_root: Path,
    reference_report: Path,
    *,
    iteration: int,
    multiplier: float,
) -> dict[str, object]:
    if iteration < 1:
        raise ValueError("capture iteration must be positive")
    reference = json.loads(reference_report.read_text())
    if reference.get("capture_inertness_qualified") is not True:
        raise ValueError("sealed reference inertness report did not qualify")
    arrays = {}
    hashes = {}
    array_gates = []
    for rank in (1, 2):
        prefix = f"mstep_it{iteration:03d}_rank{rank}_half{rank}_c0_pre_lowres_join_bpref"
        for field, complex_values in (("data", True), ("weight", False)):
            paths = {
                "control": control_root / "dumps" / f"{prefix}_{field}.bin",
                "capture": capture_root / "dumps" / f"{prefix}_{field}.bin",
            }
            current = _array_metrics(
                _read_bpref(paths["control"], complex_values=complex_values),
                _read_bpref(paths["capture"], complex_values=complex_values),
            )
            reference_key = f"rank{rank}_half{rank}_{field}"
            reference_arrays = reference["array_comparisons"]
            if reference_key not in reference_arrays:
                reference_key = f"pre_lowres_join_rank{rank}_half{rank}_{field}"
            reference_repeat = reference_arrays[reference_key]["control_a_vs_control_b"]
            reference_value = float(
                reference_repeat.get("relative_l2", reference_repeat.get("relative_l2_over_lhs"))
            )
            qualified = _within_envelope(float(current["relative_l2"]), reference_value, multiplier)
            current["sealed_control_repeat_relative_l2"] = reference_value
            current["within_sealed_repeat_envelope"] = qualified
            arrays[f"rank{rank}_half{rank}_{field}"] = current
            array_gates.append(qualified)
            hashes.update({str(path.resolve()): _sha256(path) for path in paths.values()})

    maps = {}
    map_gates = []
    for half in (1, 2):
        paths = {
            "control": control_root / "relion" / f"run_it{iteration:03d}_half{half}_class001.mrc",
            "capture": capture_root / "relion" / f"run_it{iteration:03d}_half{half}_class001.mrc",
        }
        loaded = {}
        for name, path in paths.items():
            with mrcfile.open(path, permissive=False) as handle:
                loaded[name] = np.asarray(handle.data, dtype=np.float32).copy()
            hashes[str(path.resolve())] = _sha256(path)
        current = _map_metrics(loaded["control"], loaded["capture"])
        reference_auc = float(
            reference["map_fsc_comparisons"][f"half{half}"]["control_a_vs_control_b"]
            ["fsc_auc_non_dc"]
        )
        current_defect = 1.0 - float(current["fsc_auc_non_dc"])
        reference_defect = 1.0 - reference_auc
        qualified = _within_envelope(current_defect, reference_defect, multiplier)
        current["sealed_control_repeat_fsc_auc_non_dc"] = reference_auc
        current["within_sealed_repeat_envelope"] = qualified
        maps[f"half{half}"] = current
        map_gates.append(qualified)

    return {
        "schema": "relion-bpref-factor-capture-inertness-v1",
        "metric_policy": "exact/array metrics for intermediates; FSC/FSC-AUC only for maps; no correlation",
        "control_root": str(control_root.resolve()),
        "capture_root": str(capture_root.resolve()),
        "capture_iteration": iteration,
        "sealed_reference_report": str(reference_report.resolve()),
        "sealed_reference_sha256": _sha256(reference_report),
        "repeat_envelope_multiplier": multiplier,
        "array_comparisons": arrays,
        "map_fsc_comparisons": maps,
        "artifact_sha256": hashes,
        "capture_inertness_qualified": bool(all(array_gates) and all(map_gates)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--sealed-reference-report", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--repeat-envelope-multiplier", type=float, default=2.0)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite inertness artifact: {args.output_json}")
    report = analyze(
        args.control_root,
        args.capture_root,
        args.sealed_reference_report,
        iteration=args.iteration,
        multiplier=args.repeat_envelope_multiplier,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
