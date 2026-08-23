#!/usr/bin/env python3
"""Create an immutable balanced-half real-particle VDAM fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
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
    output_star_data = dict(star_data) if isinstance(star_data, dict) else subset
    if isinstance(output_star_data, dict):
        if "particles" in output_star_data:
            output_star_data["particles"] = subset
        else:
            only_key = next(iter(output_star_data))
            output_star_data[only_key] = subset

    linked_stacks = _link_relative_stacks(subset, source_dir=source_star.parent, output_dir=output_dir)
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
