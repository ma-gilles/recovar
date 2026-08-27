#!/usr/bin/env python3
"""Compare iteration-1 state and M-step boundaries for a coarse-prior-order A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "recovar.em.k1_case04_coarse_prior_order_it1_ab.v1"
ARRAY_STEMS = (
    "it000_Ft_y_0.npy",
    "it000_Ft_y_1.npy",
    "it000_Ft_ctf_0.npy",
    "it000_Ft_ctf_1.npy",
    "it000_fsc.npy",
    "it000_noise_half1.npy",
    "it000_noise_half2.npy",
    "it000_tau2.npy",
)
PARTICLE_FIELDS = (
    "original_image_indices",
    "max_posterior",
    "significant_counts",
    "fine_hard_assignment",
    "coarse_hard_assignment",
    "rotation_matrices",
    "rotation_eulers_deg",
    "relative_translations_pixels",
    "absolute_translations_pixels",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(
            "comparison operands differ in shape or dtype: "
            f"{reference.shape}/{reference.dtype} versus {candidate.shape}/{candidate.dtype}"
        )
    mismatch = reference != candidate
    if mismatch.ndim > 1:
        mismatch_rows = np.any(mismatch.reshape(mismatch.shape[0], -1), axis=1)
    else:
        mismatch_rows = mismatch
    delta = candidate.astype(
        np.complex128 if np.iscomplexobj(candidate) else np.float64,
    ) - reference.astype(
        np.complex128 if np.iscomplexobj(reference) else np.float64,
    )
    denominator = float(np.linalg.norm(reference.reshape(-1).astype(delta.dtype)))
    numerator = float(np.linalg.norm(delta.reshape(-1)))
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact_equal": bool(not np.any(mismatch)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "mismatch_row_count": int(np.count_nonzero(mismatch_rows)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_reference": numerator / denominator if denominator else numerator,
    }


def _load_particle_state(directory: Path, half: int) -> dict[str, np.ndarray]:
    path = directory / f"it000_particle_state_half{half}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [field for field in PARTICLE_FIELDS if field not in payload.files]
        if missing:
            raise ValueError(f"{path} is missing particle-state fields {missing}")
        return {field: np.asarray(payload[field]) for field in PARTICLE_FIELDS}


def _compare(reference_dir: Path, candidate_dir: Path) -> dict[str, object]:
    arrays: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    for stem in ARRAY_STEMS:
        reference_path = reference_dir / stem
        candidate_path = candidate_dir / stem
        if not reference_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"missing A/B array {reference_path} or {candidate_path}")
        arrays[stem] = _metric(np.load(reference_path), np.load(candidate_path))
        artifacts[stem] = {
            "reference_sha256": _sha256(reference_path),
            "candidate_sha256": _sha256(candidate_path),
        }

    particle_paths = {
        "reference": [reference_dir / f"it000_particle_state_half{half}.npz" for half in (1, 2)],
        "candidate": [candidate_dir / f"it000_particle_state_half{half}.npz" for half in (1, 2)],
    }
    missing_particle_paths = [
        str(path)
        for paths in particle_paths.values()
        for path in paths
        if not path.is_file()
    ]
    particles: dict[str, object] = {
        "status": "unavailable" if missing_particle_paths else "complete",
        "missing_paths": missing_particle_paths,
    }
    if not missing_particle_paths:
        for half in (1, 2):
            reference = _load_particle_state(reference_dir, half)
            candidate = _load_particle_state(candidate_dir, half)
            if not np.array_equal(
                reference["original_image_indices"],
                candidate["original_image_indices"],
            ):
                raise ValueError(f"half {half} immutable particle order changed")
            particles[f"half{half}"] = {
                field: _metric(reference[field], candidate[field])
                for field in PARTICLE_FIELDS
            }
    return {
        "arrays": arrays,
        "particle_state": particles,
        "artifacts": artifacts,
    }


def analyze(*, old_dir: Path, current_dir: Path, legacy_dir: Path) -> dict[str, object]:
    for directory in (old_dir, current_dir, legacy_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "hypothesis": (
            "If native coarse prior/min-diff ordering is the only iteration-1 "
            "source change, the legacy diagnostic should reproduce the old "
            "M-step arrays while differing from the current control."
        ),
        "old_vs_current": _compare(old_dir, current_dir),
        "old_vs_legacy": _compare(old_dir, legacy_dir),
        "current_vs_legacy": _compare(current_dir, legacy_dir),
        "paths": {
            "old": str(old_dir.resolve()),
            "current": str(current_dir.resolve()),
            "legacy": str(legacy_dir.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        old_dir=args.old_dir,
        current_dir=args.current_dir,
        legacy_dir=args.legacy_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
