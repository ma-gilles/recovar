#!/usr/bin/env python3
"""Build a K=1 final-pass manifest with selected fields from a donor boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_final_manifest_ab import PARTICLE_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _half_indices(results_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(results_path, allow_pickle=True) as archive:
        missing = [name for name in ("half1_indices", "half2_indices") if name not in archive.files]
        if missing:
            raise ValueError(f"{results_path} is missing particle identities: {missing}")
        halves = tuple(
            np.asarray(archive[name], dtype=np.int64).reshape(-1)
            for name in ("half1_indices", "half2_indices")
        )
    for half, indices in enumerate(halves, start=1):
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{results_path} half {half} contains duplicate particle identities")
    if np.intersect1d(halves[0], halves[1]).size:
        raise ValueError(f"{results_path} half sets overlap")
    return halves


def _donor_gather(base_indices: np.ndarray, donor_indices: np.ndarray, *, half: int) -> np.ndarray:
    if base_indices.size != donor_indices.size or set(base_indices.tolist()) != set(donor_indices.tolist()):
        raise ValueError(f"half {half} base and donor particle identity sets differ")
    donor_position = {int(source_id): position for position, source_id in enumerate(donor_indices)}
    gather = np.asarray([donor_position[int(source_id)] for source_id in base_indices], dtype=np.int64)
    if not np.array_equal(np.sort(gather), np.arange(gather.size, dtype=np.int64)):
        raise ValueError(f"half {half} donor gather is not bijective")
    return gather


def _gather_sha256(gather: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(gather, dtype=np.int64).tobytes()).hexdigest()


def build_hybrid(
    *,
    base_manifest_dir: Path,
    base_results: Path,
    donor_manifest_dir: Path,
    donor_results: Path,
    output_dir: Path,
    fields: tuple[str, ...],
) -> dict[str, object]:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("at least one unique --field is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}")
    base_indices = _half_indices(base_results)
    donor_indices = _half_indices(donor_results)
    prepared_halves = []
    report_halves = []
    for half_index in range(2):
        half = half_index + 1
        base_path = base_manifest_dir / f"manifest_final_half{half_index}.npz"
        donor_path = donor_manifest_dir / f"manifest_final_half{half_index}.npz"
        output_path = output_dir / f"manifest_final_half{half_index}.npz"
        gather = _donor_gather(base_indices[half_index], donor_indices[half_index], half=half)
        with np.load(base_path, allow_pickle=False) as base_archive, np.load(
            donor_path, allow_pickle=False
        ) as donor_archive:
            for label, archive in (("base", base_archive), ("donor", donor_archive)):
                if "half_index" not in archive.files:
                    raise ValueError(f"half {half} {label} manifest is missing half_index")
                if int(np.asarray(archive["half_index"]).item()) != half_index:
                    raise ValueError(f"half {half} {label} manifest has the wrong half_index")
            payload = {name: np.asarray(base_archive[name]) for name in base_archive.files}
            field_rows = {}
            for field in fields:
                if field not in base_archive.files or field not in donor_archive.files:
                    raise ValueError(f"half {half} missing requested field {field}")
                base_value = np.asarray(base_archive[field])
                donor_value = np.asarray(donor_archive[field])
                aligned_donor = donor_value
                if field in PARTICLE_FIELDS and donor_value.ndim > 0 and donor_value.shape[0] > 0:
                    if donor_value.shape[0] != gather.size:
                        raise ValueError(
                            f"half {half} donor {field} has {donor_value.shape[0]} rows, expected {gather.size}"
                        )
                    aligned_donor = donor_value[gather]
                if aligned_donor.shape != base_value.shape or aligned_donor.dtype != base_value.dtype:
                    raise ValueError(
                        f"half {half} {field} topology differs: "
                        f"base={base_value.shape}/{base_value.dtype}, "
                        f"donor={aligned_donor.shape}/{aligned_donor.dtype}"
                    )
                if not (
                    np.issubdtype(base_value.dtype, np.number)
                    or np.issubdtype(base_value.dtype, np.bool_)
                ):
                    raise ValueError(f"half {half} {field} must be numeric or boolean")
                payload[field] = aligned_donor
                comparison_dtype = np.complex128 if np.iscomplexobj(aligned_donor) else np.float64
                residual = aligned_donor.astype(comparison_dtype) - base_value.astype(comparison_dtype)
                denominator = float(np.linalg.norm(base_value.reshape(-1)))
                field_rows[field] = {
                    "shape": list(base_value.shape),
                    "dtype": str(base_value.dtype),
                    "changed_count": int(np.count_nonzero(aligned_donor != base_value)),
                    "relative_l2_donor_minus_base": (
                        float(np.linalg.norm(residual.reshape(-1)) / denominator)
                        if denominator > 0.0
                        else None
                    ),
                }
        prepared_halves.append((output_path, payload))
        report_halves.append(
            {
                "half": half,
                "base_manifest": str(base_path.resolve()),
                "base_manifest_sha256": _sha256(base_path),
                "donor_manifest": str(donor_path.resolve()),
                "donor_manifest_sha256": _sha256(donor_path),
                "output_manifest": str(output_path.resolve()),
                "donor_gather_sha256": _gather_sha256(gather),
                "fields": field_rows,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for row, (output_path, payload) in zip(report_halves, prepared_halves, strict=True):
        np.savez(output_path, **payload)
        row["output_manifest_sha256"] = _sha256(output_path)
    return {
        "schema": "recovar.em.k1_final_manifest_hybrid.v1",
        "status": "complete",
        "fields": list(fields),
        "identity_semantics": "base physical row -> immutable source row -> donor row",
        "base_results": str(base_results.resolve()),
        "base_results_sha256": _sha256(base_results),
        "donor_results": str(donor_results.resolve()),
        "donor_results_sha256": _sha256(donor_results),
        "halves": report_halves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest-dir", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--donor-manifest-dir", type=Path, required=True)
    parser.add_argument("--donor-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_hybrid(
        base_manifest_dir=args.base_manifest_dir.resolve(),
        base_results=args.base_results.resolve(),
        donor_manifest_dir=args.donor_manifest_dir.resolve(),
        donor_results=args.donor_results.resolve(),
        output_dir=args.output_dir.resolve(),
        fields=tuple(args.field),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
