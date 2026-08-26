#!/usr/bin/env python3
"""Build immutable RELION/RECOVAR identities for K=1 BPref prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import starfile


SCHEMA = "recovar-k1-bpref-prefix-selection-v2"


def _particles(path: Path):
    payload = starfile.read(path)
    if isinstance(payload, dict):
        if "particles" not in payload:
            raise ValueError(f"missing particles block in {path}")
        return payload["particles"]
    return payload


def _unique_image_names(path: Path) -> np.ndarray:
    particles = _particles(path)
    if "rlnImageName" not in particles:
        raise ValueError(f"missing rlnImageName in {path}")
    names = np.asarray(particles["rlnImageName"], dtype=str).reshape(-1)
    if len(set(names.tolist())) != names.size:
        raise ValueError(f"rlnImageName values are not unique in {path}")
    return names


def _stack_index_one_based(image_name: str) -> int:
    token, separator, _ = str(image_name).partition("@")
    if not separator:
        raise ValueError(f"invalid RELION image identity: {image_name!r}")
    index = int(token)
    if index <= 0:
        raise ValueError(f"RELION stack indices must be one-based: {image_name!r}")
    return index


def _parse_ordinals(value: str) -> np.ndarray:
    ordinals = np.asarray([int(item) for item in value.split(",") if item], dtype=np.int64)
    if ordinals.size == 0 or np.any(ordinals < 0):
        raise argparse.ArgumentTypeError("ordinals must be non-negative integers")
    if np.unique(ordinals).size != ordinals.size or np.any(np.diff(ordinals) <= 0):
        raise argparse.ArgumentTypeError("ordinals must be unique and strictly increasing")
    return ordinals


def _half_payload(
    half_indices: np.ndarray,
    ordinals: np.ndarray,
    input_names: np.ndarray,
    native_row_by_name: dict[str, int],
) -> dict[str, list[int]]:
    half_indices = np.asarray(half_indices, dtype=np.int64).reshape(-1)
    if ordinals[-1] >= half_indices.size:
        raise ValueError(
            f"prefix ordinal {int(ordinals[-1])} exceeds half size {half_indices.size}"
        )
    original_indices = half_indices[ordinals]
    if np.any(original_indices < 0) or np.any(original_indices >= input_names.size):
        raise ValueError("RECOVAR half indices are outside the input STAR row range")
    selected_names = input_names[original_indices]
    missing = [name for name in selected_names.tolist() if name not in native_row_by_name]
    if missing:
        raise ValueError(f"selected identities are absent from RELION data STAR: {missing[:3]}")
    return {
        "half_local_ordinals": ordinals.tolist(),
        "original_indices": original_indices.tolist(),
        "native_internal_indices": [native_row_by_name[name] for name in selected_names.tolist()],
        "stack_indices_1based": [_stack_index_one_based(name) for name in selected_names.tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--input-particles-star", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--relion-source-commit", required=True)
    parser.add_argument("--physical-iteration", type=int, default=1)
    parser.add_argument(
        "--ordinals",
        type=_parse_ordinals,
        default=_parse_ordinals("0,1,3,7,15,31,63,127,255,511"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_names = _unique_image_names(args.input_particles_star.resolve())
    native_names = _unique_image_names(args.relion_data_star.resolve())
    if set(input_names.tolist()) != set(native_names.tolist()):
        raise ValueError("input and RELION data STAR files have different particle identities")
    native_row_by_name = {name: row for row, name in enumerate(native_names.tolist())}

    with np.load(args.refinement_results.resolve(), allow_pickle=False) as archive:
        half1 = np.asarray(archive["half1_indices"], dtype=np.int64)
        half2 = np.asarray(archive["half2_indices"], dtype=np.int64)
    if np.intersect1d(half1, half2).size or np.union1d(half1, half2).size != input_names.size:
        raise ValueError("RECOVAR half indices are not a disjoint complete particle partition")

    report = {
        "schema": SCHEMA,
        "case": args.case_name,
        "physical_iteration": int(args.physical_iteration),
        "relion_source_commit": args.relion_source_commit,
        "source_refinement_results": str(args.refinement_results.resolve()),
        "half1": _half_payload(half1, args.ordinals, input_names, native_row_by_name),
        "half2": _half_payload(half2, args.ordinals, input_names, native_row_by_name),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
