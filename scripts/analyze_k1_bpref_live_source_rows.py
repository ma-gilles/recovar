#!/usr/bin/env python3
"""Compare live RECOVAR BPref source rows with passive RELION rows.

This boundary is later than primitive replay and earlier than scatter.  It
uses the rows that production actually handed to RECOVAR's scatter kernel and
RELION's passive in-kernel pre-scatter capture.  The comparison therefore does
not infer a live operand from a rounded diagnostic noise array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
else:
    from validate_relion_bpref_factor_capture import load_factor_capture  # type: ignore[no-redef]


SCHEMA = "recovar.em.k1_bpref_live_source_rows.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "metric arrays must be nonempty and aligned")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _exact_rotation_map(native_matrices: np.ndarray, active_rotations: np.ndarray) -> np.ndarray:
    """Map native matrix rows to RECOVAR rows, rejecting convention drift."""

    native = np.asarray(native_matrices, dtype=np.float32).reshape(-1, 9)
    # RECOVAR stores projection matrices in the transpose convention used by
    # its slicer; RELION serializes the matrix consumed by BP.cuh.
    active = np.asarray(active_rotations, dtype=np.float32).transpose(0, 2, 1).reshape(-1, 9)
    errors = np.max(np.abs(native[:, None, :] - active[None, :, :]), axis=2)
    mapping = np.argmin(errors, axis=1).astype(np.int64)
    _require(np.all(errors[np.arange(native.shape[0]), mapping] == 0.0), "rotation matrices are not bitwise aligned")
    _require(np.unique(mapping).size == mapping.size, "rotation mapping is not one-to-one")
    return mapping


def _load_contribution(directory: Path, original_index: int) -> tuple[Path, dict[str, np.ndarray]]:
    matches: list[tuple[Path, dict[str, np.ndarray]]] = []
    for path in sorted(directory.glob("bpref_contribution_rows_it001_h*_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            originals = np.asarray(archive["original_indices"], dtype=np.int64)
            rows = np.flatnonzero(originals == int(original_index))
            if rows.size:
                _require(rows.size == 1 and originals.size == 1, f"ambiguous particle ownership in {path}")
                matches.append((path, {name: archive[name] for name in archive.files}))
    _require(len(matches) == 1, f"expected one contribution for source row {original_index}")
    return matches[0]


def _compare_particle(
    *,
    target: dict[str, Any],
    factor_directory: Path,
    contribution_directory: Path,
) -> dict[str, Any]:
    stack = int(target["stack_index_one_based"])
    original = int(target["original_index_zero_based"])
    factor_paths = sorted(factor_directory.glob(f"part*_stack{stack}_img*_class*.bpre-v2.bin"))
    _require(len(factor_paths) == 1, f"expected one native factor capture for stack {stack}")
    factor = load_factor_capture(factor_paths[0])
    contribution_path, contribution = _load_contribution(contribution_directory, original)

    _require(bool(contribution["high_precision_operand_bundle"]), "high-precision bundle is absent")
    _require(int(contribution["iteration"]) == 1, "comparison is not physical iteration 1")
    _require(int(contribution["stack_indices_1based"][0]) == stack, "particle identity changed")
    _require(int(contribution["current_size"]) == int(factor.header[17]), "current size changed")
    _require(str(contribution["noise_variance_half"].dtype) == "float64", "live variance lost binary64 precision")

    rotation_map = _exact_rotation_map(factor.rotations["matrix"], contribution["active_rotations"])
    accepted = factor.hypotheses[(factor.hypotheses["flags"] & np.uint32(1)) != 0]
    _require(accepted.size == 1, f"stack {stack}: expected one firstiter-CC hypothesis")
    native_orientation = int(accepted[0]["orientation_local"])
    active_row = int(rotation_map[native_orientation])

    image_size = int(contribution["image_shape"][0])
    _require(tuple(contribution["image_shape"].tolist()) == (image_size, image_size), "image is not square")
    half_width = image_size // 2 + 1
    standard_indices = np.asarray(contribution["window_indices"], dtype=np.int64)
    standard_rows = standard_indices // half_width
    coordinates = list(
        zip(
            (standard_indices % half_width).tolist(),
            np.where(standard_rows <= image_size // 2, standard_rows, standard_rows - image_size).tolist(),
            strict=True,
        )
    )
    coordinate_to_recovar = {coordinate: row for row, coordinate in enumerate(coordinates)}
    native_rows = factor.summaries[
        factor.summaries["orientation_local"] == np.uint32(native_orientation)
    ]
    _require(native_rows.size > 0, f"stack {stack}: native live source support is empty")
    recovar_rows = np.asarray(
        [coordinate_to_recovar[(int(row["x"]), int(row["y"]))] for row in native_rows],
        dtype=np.int64,
    )
    _require(np.unique(recovar_rows).size == recovar_rows.size, "native source coordinates are not unique")

    n2 = np.float32(image_size**2)
    n4 = np.float32(image_size**4)
    native_numerator = (
        -(native_rows["source_re"] + np.complex64(1j) * native_rows["source_im"]) / n2
    ).astype(np.complex64)
    native_denominator = (native_rows["source_weight"] / n4).astype(np.float32)
    recovar_numerator = np.asarray(
        contribution["active_summed"][active_row, recovar_rows], dtype=np.complex64
    )
    recovar_denominator = np.asarray(
        contribution["active_ctf_probs"][active_row, recovar_rows], dtype=np.float32
    )

    return {
        "stack_index_one_based": stack,
        "original_index_zero_based": original,
        "half": int(contribution["half"]),
        "native_orientation_local": native_orientation,
        "recovar_active_rotation_row": active_row,
        "rotation_map": rotation_map.tolist(),
        "native_scattered_source_rows": int(native_rows.size),
        "recovar_positive_source_rows_before_radius_gate": int(
            np.count_nonzero(contribution["active_ctf_probs"][active_row] > 0.0)
        ),
        "comparisons": {
            "live_prescatter_numerator": _metric(native_numerator, recovar_numerator),
            "live_prescatter_denominator": _metric(native_denominator, recovar_denominator),
        },
        "factor_capture": str(factor_paths[0].resolve()),
        "factor_capture_sha256": factor.sha256,
        "contribution_bundle": str(contribution_path.resolve()),
        "contribution_bundle_sha256": _sha256(contribution_path),
    }


def _distribution(particles: list[dict[str, Any]], name: str) -> dict[str, float | int]:
    values = np.asarray(
        [particle["comparisons"][name]["relative_l2_over_reference"] for particle in particles],
        dtype=np.float64,
    )
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "exact_count": int(sum(particle["comparisons"][name]["exact_equal"] for particle in particles)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--contribution-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_json.read_text())
    targets = selection.get("targets")
    _require(isinstance(targets, list) and bool(targets), "selection has no targets")
    particles = [
        _compare_particle(
            target=target,
            factor_directory=args.factor_directory,
            contribution_directory=args.contribution_directory,
        )
        for target in targets
    ]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact and scale-sensitive relative-L2 source-row metrics; no correlation",
        "selection_json": str(args.selection_json.resolve()),
        "selection_sha256": _sha256(args.selection_json),
        "particle_count": len(particles),
        "summary": {
            name: _distribution(particles, name)
            for name in ("live_prescatter_numerator", "live_prescatter_denominator")
        },
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
