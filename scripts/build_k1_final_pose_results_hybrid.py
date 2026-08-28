#!/usr/bin/env python3
"""Build a final-replay results archive with RELION poses in RECOVAR row order."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import starfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _particles(path: Path):
    tables = starfile.read(path)
    return tables["particles"] if isinstance(tables, dict) else tables


def _pixel_size(path: Path) -> float:
    tables = starfile.read(path)
    if not isinstance(tables, dict) or "optics" not in tables:
        raise ValueError(f"{path} has no optics table")
    optics = tables["optics"]
    if "rlnImagePixelSize" not in optics.columns or len(optics) != 1:
        raise ValueError(f"{path} must contain one rlnImagePixelSize value")
    value = float(np.asarray(optics["rlnImagePixelSize"], dtype=np.float64)[0])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{path} has invalid rlnImagePixelSize={value}")
    return value


def _image_identity(name: object, *, label: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)@(.+)", str(name))
    if match is None:
        raise ValueError(f"{label} image name must use '<index>@<stack>': {name!r}")
    return int(match.group(1)), match.group(2)


def _identity_rows(particles, *, label: str) -> dict[tuple[int, str], int]:
    if "rlnImageName" not in particles.columns:
        raise ValueError(f"{label} is missing rlnImageName")
    identities = [_image_identity(value, label=label) for value in particles["rlnImageName"]]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} contains duplicate particle identities")
    return {identity: row for row, identity in enumerate(identities)}


def _metrics(candidate: np.ndarray, base: np.ndarray) -> dict[str, object]:
    comparison_dtype = np.complex128 if np.iscomplexobj(candidate) else np.float64
    residual = candidate.astype(comparison_dtype) - base.astype(comparison_dtype)
    denominator = float(np.linalg.norm(base.reshape(-1)))
    return {
        "shape": list(base.shape),
        "dtype": str(base.dtype),
        "changed_count": int(np.count_nonzero(candidate != base)),
        "relative_l2_candidate_minus_base": (
            float(np.linalg.norm(residual.reshape(-1)) / denominator)
            if denominator > 0.0
            else None
        ),
        "max_absolute_candidate_minus_base": float(np.max(np.abs(residual), initial=0.0)),
    }


def build_pose_results_hybrid(
    *,
    base_results: Path,
    input_particle_star: Path,
    relion_data_star: Path,
    aligned_manifest_dir: Path,
    output_results: Path,
) -> dict[str, object]:
    """Replace only the last-numbered pose arrays in a sealed results archive."""

    if output_results.exists():
        raise FileExistsError(f"refusing to overwrite {output_results}")
    input_particles = _particles(input_particle_star)
    relion_particles = _particles(relion_data_star)
    input_rows = _identity_rows(input_particles, label="input particle STAR")
    relion_rows = _identity_rows(relion_particles, label="RELION data STAR")
    if set(input_rows) != set(relion_rows):
        missing = len(set(input_rows) - set(relion_rows))
        extra = len(set(relion_rows) - set(input_rows))
        raise ValueError(
            "input and RELION particle identities differ "
            f"(missing={missing}, extra={extra})"
        )
    angle_columns = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    missing_angles = [name for name in angle_columns if name not in relion_particles.columns]
    if missing_angles:
        raise ValueError(f"RELION data STAR is missing pose columns: {missing_angles}")
    origin_columns = ("rlnOriginXAngst", "rlnOriginYAngst")
    missing_origins = [name for name in origin_columns if name not in relion_particles.columns]
    if missing_origins:
        raise ValueError(f"RELION data STAR is missing origin columns: {missing_origins}")

    pixel_size = _pixel_size(input_particle_star)
    relion_eulers = np.stack(
        [np.asarray(relion_particles[name], dtype=np.float64) for name in angle_columns], axis=1
    ).astype(np.float32)
    relion_translations = (
        np.stack(
            [np.asarray(relion_particles[name], dtype=np.float64) for name in origin_columns],
            axis=1,
        )
        / pixel_size
    ).astype(np.float32)
    input_identities = tuple(input_rows)

    with np.load(base_results, allow_pickle=True) as base_archive:
        payload = {name: np.asarray(base_archive[name]) for name in base_archive.files}
    current_sizes = np.asarray(payload.get("current_sizes"), dtype=np.int64)
    if current_sizes.ndim != 1 or current_sizes.size == 0:
        raise ValueError(f"{base_results} has no numbered refinement iterations")
    pose_label = f"{current_sizes.size - 1:03d}"
    half_indices = tuple(
        np.asarray(payload[f"half{half}_indices"], dtype=np.int64).reshape(-1)
        for half in (1, 2)
    )
    if np.intersect1d(*half_indices).size:
        raise ValueError("base results half sets overlap")
    if sum(indices.size for indices in half_indices) != len(input_identities):
        raise ValueError("base results half sets do not cover the input STAR")

    pose_halves = []
    translation_halves = []
    report_halves = []
    for half_index, indices in enumerate(half_indices):
        identities = [input_identities[int(row)] for row in indices]
        relion_gather = np.asarray([relion_rows[identity] for identity in identities], dtype=np.int64)
        eulers = np.ascontiguousarray(relion_eulers[relion_gather], dtype=np.float32)
        translations = np.ascontiguousarray(relion_translations[relion_gather], dtype=np.float32)
        euler_key = f"best_rotation_eulers_iter_{pose_label}_half{half_index}"
        translation_key = f"best_translations_iter_{pose_label}_half{half_index}"
        if euler_key not in payload or translation_key not in payload:
            raise ValueError(f"base results are missing {euler_key} or {translation_key}")
        manifest_path = aligned_manifest_dir / f"manifest_final_half{half_index}.npz"
        with np.load(manifest_path, allow_pickle=False) as manifest:
            manifest_translations = np.asarray(
                manifest["absolute_previous_translations"], dtype=np.float32
            )
        if not np.array_equal(translations, manifest_translations):
            max_abs = float(np.max(np.abs(translations - manifest_translations), initial=0.0))
            raise ValueError(
                f"half {half_index + 1} RELION translations do not match the aligned "
                f"manifest (max_abs={max_abs})"
            )
        report_halves.append(
            {
                "half": half_index + 1,
                "relion_row_gather_sha256": hashlib.sha256(relion_gather.tobytes()).hexdigest(),
                "eulers": _metrics(eulers, np.asarray(payload[euler_key], dtype=np.float32)),
                "translations": _metrics(
                    translations, np.asarray(payload[translation_key], dtype=np.float32)
                ),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
        payload[euler_key] = eulers
        payload[translation_key] = translations
        pose_halves.append(eulers)
        translation_halves.append(translations)

    def _replace_derived(prefix: str, halves: list[np.ndarray]) -> None:
        concatenated = np.concatenate(halves, axis=0)
        source_order = np.empty((len(input_identities), halves[0].shape[1]), dtype=np.float32)
        for indices, values in zip(half_indices, halves, strict=True):
            source_order[indices] = values
        direct_key = f"{prefix}_iter_{pose_label}"
        by_image_key = f"{prefix}_by_image_iter_{pose_label}"
        final_by_image_key = f"{prefix}_final_by_image"
        for key, value in (
            (direct_key, concatenated),
            (by_image_key, source_order),
            (final_by_image_key, source_order),
        ):
            if key in payload:
                payload[key] = value

    _replace_derived("best_rotation_eulers", pose_halves)
    _replace_derived("best_translations", translation_halves)
    output_results.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_results, **payload)
    return {
        "schema": "recovar.em.k1_final_pose_results_hybrid.v1",
        "status": "complete",
        "pose_iteration_label": pose_label,
        "pixel_size_angstrom": pixel_size,
        "base_results": str(base_results.resolve()),
        "base_results_sha256": _sha256(base_results),
        "input_particle_star": str(input_particle_star.resolve()),
        "input_particle_star_sha256": _sha256(input_particle_star),
        "relion_data_star": str(relion_data_star.resolve()),
        "relion_data_star_sha256": _sha256(relion_data_star),
        "output_results": str(output_results.resolve()),
        "output_results_sha256": _sha256(output_results),
        "halves": report_halves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--input-particle-star", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--aligned-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_pose_results_hybrid(
        base_results=args.base_results.resolve(),
        input_particle_star=args.input_particle_star.resolve(),
        relion_data_star=args.relion_data_star.resolve(),
        aligned_manifest_dir=args.aligned_manifest_dir.resolve(),
        output_results=args.output_results.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
