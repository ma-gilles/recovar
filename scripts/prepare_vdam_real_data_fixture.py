#!/usr/bin/env python3
"""Create an immutable balanced-half real-particle VDAM fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import starfile


def _particles_table(star_data):
    if isinstance(star_data, dict):
        if "particles" in star_data:
            return star_data["particles"]
        if len(star_data) == 1:
            return next(iter(star_data.values()))
        raise ValueError(f"STAR has no particles table; tables={sorted(star_data)}")
    return star_data


def _column(table, name: str):
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return candidate
    raise ValueError(f"particle STAR is missing {name}")


def select_balanced_half_indices(particles, *, particles_per_half: int, seed: int) -> np.ndarray:
    if int(particles_per_half) <= 0:
        raise ValueError("particles_per_half must be positive")
    subset_column = _column(particles, "rlnRandomSubset")
    subsets = np.asarray(particles[subset_column], dtype=np.int64)
    if not np.all(np.isin(subsets, (1, 2))):
        raise ValueError("rlnRandomSubset must contain only RELION half identifiers 1 and 2")

    rng = np.random.RandomState(int(seed))
    selected = []
    for half in (1, 2):
        candidates = np.flatnonzero(subsets == half)
        if candidates.size < int(particles_per_half):
            raise ValueError(
                f"half {half} contains {candidates.size} particles; "
                f"requested {int(particles_per_half)}"
            )
        selected.append(rng.choice(candidates, size=int(particles_per_half), replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_relative_stacks(particles, *, source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    image_column = _column(particles, "rlnImageName")
    stack_names = sorted({str(value).split("@", 1)[1] for value in particles[image_column]})
    linked = []
    for stack_name in stack_names:
        stack_path = Path(stack_name)
        if stack_path.is_absolute():
            if not stack_path.is_file():
                raise FileNotFoundError(stack_path)
            linked.append({"fixture_path": str(stack_path), "source_path": str(stack_path)})
            continue
        source_path = (source_dir / stack_path).resolve(strict=True)
        fixture_path = output_dir / stack_path
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        if fixture_path.exists() or fixture_path.is_symlink():
            if fixture_path.resolve() != source_path:
                raise FileExistsError(f"fixture stack path already targets another file: {fixture_path}")
        else:
            fixture_path.symlink_to(source_path)
        linked.append({"fixture_path": str(fixture_path), "source_path": str(source_path)})
    return linked


def _constant_particle_value(particles, name: str) -> float:
    column = _column(particles, name)
    values = np.asarray(particles[column], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"legacy STAR column {name} must be finite and nonempty")
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"legacy STAR column {name} varies; explicit optics grouping is required")
    return float(values[0])


def promote_legacy_optics(particles, *, image_size: int):
    """Return a RELION 3.1 optics/particles pair from a legacy single table."""

    detector_pixel_um = _constant_particle_value(particles, "rlnDetectorPixelSize")
    magnification = _constant_particle_value(particles, "rlnMagnification")
    if detector_pixel_um <= 0.0 or magnification <= 0.0:
        raise ValueError("legacy detector pixel size and magnification must be positive")
    image_pixel_size = detector_pixel_um * 10_000.0 / magnification
    optics_values = {
        "rlnOpticsGroup": [1],
        "rlnOpticsGroupName": ["opticsGroup1"],
        "rlnImagePixelSize": [image_pixel_size],
        "rlnImageSize": [int(image_size)],
        "rlnImageDimensionality": [2],
        "rlnVoltage": [_constant_particle_value(particles, "rlnVoltage")],
        "rlnSphericalAberration": [_constant_particle_value(particles, "rlnSphericalAberration")],
        "rlnAmplitudeContrast": [_constant_particle_value(particles, "rlnAmplitudeContrast")],
    }
    promoted = particles.copy()
    for name in (
        "rlnDetectorPixelSize",
        "rlnMagnification",
        "rlnVoltage",
        "rlnSphericalAberration",
        "rlnAmplitudeContrast",
    ):
        for candidate in (name, f"_{name}"):
            if candidate in promoted.columns:
                promoted = promoted.drop(columns=candidate)
    promoted["rlnOpticsGroup"] = np.ones(len(promoted), dtype=np.int64)
    if not any(candidate in promoted.columns for candidate in ("rlnPhaseShift", "_rlnPhaseShift")):
        promoted["rlnPhaseShift"] = np.zeros(len(promoted), dtype=np.float64)
    return {"optics": pd.DataFrame(optics_values), "particles": promoted}


def _stack_image_size(linked_stacks: list[dict[str, str]]) -> int:
    import mrcfile

    sizes = set()
    for linked in linked_stacks:
        with mrcfile.open(linked["source_path"], mode="r", header_only=True, permissive=True) as mrc:
            sizes.add((int(mrc.header.ny), int(mrc.header.nx)))
    if len(sizes) != 1:
        raise ValueError(f"fixture particle stacks must share one image shape; got {sorted(sizes)}")
    image_h, image_w = next(iter(sizes))
    if image_h != image_w:
        raise ValueError(f"fixture particle images must be square; got {(image_h, image_w)}")
    return image_h


def prepare_fixture(
    *,
    source_star: Path,
    output_dir: Path,
    dataset: str,
    particles_per_half: int,
    seed: int,
) -> dict:
    source_star = source_star.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required_absent in ("particles.star", "source_indices.npy", "fixture_manifest.json"):
        if (output_dir / required_absent).exists():
            raise FileExistsError(f"refusing to overwrite existing fixture file: {output_dir / required_absent}")

    star_data = starfile.read(source_star)
    particles = _particles_table(star_data)
    indices = select_balanced_half_indices(
        particles,
        particles_per_half=int(particles_per_half),
        seed=int(seed),
    )
    subset = particles.iloc[indices].reset_index(drop=True)
    linked_stacks = _link_relative_stacks(subset, source_dir=source_star.parent, output_dir=output_dir)
    image_size = _stack_image_size(linked_stacks)
    if isinstance(star_data, dict) and "optics" in star_data:
        output_star_data = dict(star_data)
        output_star_data["particles"] = subset.copy()
        if not any(
            candidate in output_star_data["particles"].columns
            for candidate in ("rlnPhaseShift", "_rlnPhaseShift")
        ):
            output_star_data["particles"]["rlnPhaseShift"] = np.zeros(len(subset), dtype=np.float64)
        optics_promoted = False
    else:
        output_star_data = promote_legacy_optics(subset, image_size=image_size)
        optics_promoted = True
    output_star = output_dir / "particles.star"
    source_indices = output_dir / "source_indices.npy"
    starfile.write(output_star_data, output_star, overwrite=False)
    np.save(source_indices, indices, allow_pickle=False)

    subset_column = _column(subset, "rlnRandomSubset")
    subset_ids = np.asarray(subset[subset_column], dtype=np.int64)
    manifest = {
        "schema": "recovar.vdam_real_data_fixture.v1",
        "dataset": str(dataset),
        "source_star": str(source_star),
        "source_star_sha256": _sha256(source_star),
        "source_particle_count": int(len(particles)),
        "selection_seed": int(seed),
        "particles_per_half": int(particles_per_half),
        "image_size": int(image_size),
        "legacy_optics_promoted": bool(optics_promoted),
        "selected_particle_count": int(indices.size),
        "selected_random_subset_counts": {
            str(half): int(np.sum(subset_ids == half)) for half in (1, 2)
        },
        "first_source_indices_zero_based": indices[:20].tolist(),
        "source_indices_array_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "source_indices_file_sha256": _sha256(source_indices),
        "particles_star_sha256": _sha256(output_star),
        "linked_stacks": linked_stacks,
    }
    (output_dir / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-star", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--particles-per-half", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    manifest = prepare_fixture(
        source_star=args.source_star,
        output_dir=args.output_dir,
        dataset=args.dataset,
        particles_per_half=args.particles_per_half,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
