#!/usr/bin/env python3
"""Compare matched RELION and RECOVAR VDAM iteration-one M-step stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = "recovar.vdam_mstep_boundary.v1"


def _read_relion_array(path: Path, *, complex_values: bool) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
        if shape.size != 3 or np.any(shape <= 0):
            raise ValueError(f"{path}: invalid three-int64 shape header")
        values = np.fromfile(stream, dtype=np.float64)
    count = int(np.prod(shape, dtype=np.int64))
    if complex_values:
        if values.size != 2 * count:
            raise ValueError(f"{path}: expected {2 * count} float64 components, got {values.size}")
        values = values.view(np.complex128)
    elif values.size != count:
        raise ValueError(f"{path}: expected {count} float64 values, got {values.size}")
    return values.reshape(tuple(int(value) for value in shape))


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} vs {reference.shape}")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("comparison contains non-finite values")
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    difference = candidate - reference
    return {
        "shape": list(candidate.shape),
        "value_count": int(candidate.size),
        "exact_count": int(np.count_nonzero(candidate == reference)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": (
            float(np.linalg.norm(difference.reshape(-1)) / denominator)
            if denominator > 0.0
            else float(np.linalg.norm(difference.reshape(-1)))
        ),
    }


_STAGES = (
    ("input_gradient_half0", "pipe_it1_c0_Igrad1_pre.bin", "Igrad1_in_h0.npy", True),
    ("input_gradient_half1", "pipe_it1_c0_Igrad1_h_pre.bin", "Igrad1_in_h1.npy", True),
    ("input_second_moment", "pipe_it1_c0_Igrad2_pre.bin", "Igrad2_in.npy", True),
    ("raw_accumulator_data_half0", "pipe_it1_c0_bp_data_pre_reweight.bin", "accum_h0_data.npy", True),
    ("raw_accumulator_weight_half0", "pipe_it1_c0_bp_weight.bin", "accum_h0_weight.npy", False),
    ("raw_accumulator_data_half1", "pipe_it1_c0_bp_data_h_pre_reweight.bin", "accum_h1_data.npy", True),
    ("raw_accumulator_weight_half1", "pipe_it1_c0_bp_weight_h.bin", "accum_h1_weight.npy", False),
    ("post_reweight_data_half0", "pipe_it1_c0_bp_data_post_reweight.bin", "data_h0_post_reweight.npy", True),
    ("post_reweight_data_half1", "pipe_it1_c0_bp_data_h_post_reweight.bin", "data_h1_post_reweight.npy", True),
    ("post_first_moment_half0", "pipe_it1_c0_Igrad1_post.bin", "m1_h0_post.npy", True),
    ("post_first_moment_half1", "pipe_it1_c0_Igrad1_h_post.bin", "m1_h1_post.npy", True),
    ("post_second_moment", "pipe_it1_c0_Igrad2_post.bin", "m2_post.npy", True),
    ("reference_before_reconstruct", "mstep_it1_c0_iref_before.bin", "iref_relion_in.npy", False),
    ("reference_after_reconstruct", "mstep_it1_c0_iref_after.bin", "iref_out_relion_frame.npy", False),
)


def analyze(native_directory: Path, recovar_directory: Path) -> dict:
    native_directory = Path(native_directory).resolve()
    recovar_directory = Path(recovar_directory).resolve()
    comparisons = {}
    first_nonexact = None
    for name, native_name, recovar_name, complex_values in _STAGES:
        native = _read_relion_array(
            native_directory / native_name,
            complex_values=complex_values,
        )
        recovar = np.load(recovar_directory / recovar_name, allow_pickle=False)
        metric = _metric(recovar, native)
        comparisons[name] = metric
        if first_nonexact is None and metric["exact_count"] != metric["value_count"]:
            first_nonexact = name
    return {
        "schema": SCHEMA,
        "status": "complete",
        "native_directory": str(native_directory),
        "recovar_directory": str(recovar_directory),
        "first_nonexact_stage": first_nonexact,
        "all_stages_bitwise_exact": first_nonexact is None,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.native_directory, args.recovar_directory)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
