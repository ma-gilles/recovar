#!/usr/bin/env python3
"""Compare one-iteration K=1 BPref contribution operands across two RECOVAR arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_k1_bpref_factor_boundary import _metric, _pixel_coordinates, _rotation_map
from .validate_relion_bpref_factor_capture import load_factor_capture

SCHEMA = "recovar.em.k1_bpref_contribution_ab.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contribution(directory: Path, original: int) -> tuple[Path, dict[str, np.ndarray], int]:
    matches = []
    for path in sorted(directory.glob("bpref_contribution_rows_it001_h*_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            rows = np.flatnonzero(np.asarray(archive["original_indices"]) == original)
            if rows.size:
                _require(rows.size == 1, f"particle {original} is duplicated in {path}")
                matches.append((path, {name: archive[name] for name in archive.files}, int(rows[0])))
    _require(len(matches) == 1, f"expected one contribution bundle for particle {original}")
    return matches[0]


def _active_row(contribution: dict[str, np.ndarray], particle_row: int, rotation_row: int) -> int:
    rows = np.flatnonzero(
        (np.asarray(contribution["active_particle_rows"]) == particle_row)
        & (np.asarray(contribution["active_rotation_rows"]) == rotation_row)
    )
    _require(rows.size == 1, "accepted particle/rotation does not map to one active row")
    return int(rows[0])


def _improvement(control: dict[str, Any], candidate: dict[str, Any]) -> float:
    before = float(control["relative_l2_over_reference"])
    after = float(candidate["relative_l2_over_reference"])
    return 0.0 if before == 0.0 else float((before - after) / before)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--pass2-directory", type=Path, required=True)
    parser.add_argument("--control-contribution-directory", type=Path, required=True)
    parser.add_argument("--candidate-contribution-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")

    selection = json.loads(args.selection_json.read_text())
    image_size = int(selection["physical_image_size"])
    factors = {
        factor.stack_index: factor
        for factor in (
            load_factor_capture(path) for path in args.factor_directory.glob("*.bpre-v2.bin")
        )
    }
    particles = []
    for target in selection["targets"]:
        stack = int(target["stack_index_one_based"])
        original = int(target["original_index_zero_based"])
        factor = factors[stack]
        pass2_paths = sorted(args.pass2_directory.glob(f"pass2_orig{original:06d}_cs*.npz"))
        _require(len(pass2_paths) == 1, f"stack {stack}: expected one pass-2 identity dump")
        with np.load(pass2_paths[0], allow_pickle=False) as archive:
            pass2 = {name: archive[name] for name in archive.files}
        rotation_map, rotation_error = _rotation_map(factor.rotations, pass2["rotations"])
        accepted_rows = np.flatnonzero((factor.hypotheses["flags"] & 1) != 0)
        _require(accepted_rows.size == 1, f"stack {stack}: expected one accepted hypothesis")
        accepted = factor.hypotheses[int(accepted_rows[0])]
        rotation_row = int(rotation_map[int(accepted["orientation_local"])])

        coordinates = _pixel_coordinates(pass2["recon_window_indices"], image_size)
        native_lookup = {
            (int(x), int(y)): row
            for row, (x, y) in enumerate(zip(factor.pixels["x"], factor.pixels["y"], strict=True))
        }
        native_rows = np.asarray([native_lookup[key] for key in coordinates], dtype=np.int64)
        terms = factor.terms.reshape(accepted_rows.size, factor.pixels.size)[0][native_rows]
        n2 = np.float32(image_size**2)
        n4 = np.float32(image_size**4)
        native_numerator = (
            -(terms["term_re"] + np.complex64(1j) * terms["term_im"]) / n2
        ).astype(np.complex64)
        native_denominator = (terms["weight_term"] / n4).astype(np.float32)

        arms = {}
        for arm, directory in (
            ("control", args.control_contribution_directory),
            ("candidate", args.candidate_contribution_directory),
        ):
            path, contribution, particle_row = _load_contribution(directory, original)
            active_row = _active_row(contribution, particle_row, rotation_row)
            arms[arm] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "preprocess_backend": str(np.asarray(contribution["preprocess_backend"]).item()),
                "numerator": _metric(native_numerator, contribution["active_summed"][active_row]),
                "denominator": _metric(native_denominator, contribution["active_ctf_probs"][active_row]),
            }
        particles.append(
            {
                "stack_index_one_based": stack,
                "original_index_zero_based": original,
                "rotation_map_max_abs": rotation_error,
                "control": arms["control"],
                "candidate": arms["candidate"],
                "relative_l2_fraction_removed": {
                    name: _improvement(arms["control"][name], arms["candidate"][name])
                    for name in ("numerator", "denominator")
                },
            }
        )

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact and relative-L2 intermediates; no correlation",
        "selection_json": str(args.selection_json.resolve()),
        "selection_sha256": _sha256(args.selection_json),
        "particle_count": len(particles),
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
