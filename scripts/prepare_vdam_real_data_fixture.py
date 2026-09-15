#!/usr/bin/env python3
"""Create an immutable balanced-half real-particle VDAM fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
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


def _optional_column(table, name: str):
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return candidate
    return None


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


def select_synthetic_half_indices(
    particle_count: int,
    *,
    particles_per_half: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select particles and assign deterministic balanced pseudo-halfsets."""

    requested = 2 * int(particles_per_half)
    if int(particles_per_half) <= 0:
        raise ValueError("particles_per_half must be positive")
    if requested > int(particle_count):
        raise ValueError(f"source contains {int(particle_count)} particles; requested {requested}")
    rng = np.random.RandomState(int(seed))
    unsorted = rng.choice(int(particle_count), size=requested, replace=False).astype(np.int64)
    half_by_source = {
        int(source_index): 1 if rank < int(particles_per_half) else 2
        for rank, source_index in enumerate(unsorted)
    }
    indices = np.sort(unsorted)
    subset_ids = np.asarray([half_by_source[int(index)] for index in indices], dtype=np.int64)
    return indices, subset_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_cryodrgn_source_star(
    *,
    particles_path: Path,
    poses_path: Path,
    ctf_path: Path,
    output_star: Path,
    source_indices_path: Path | None = None,
) -> dict:
    """Materialize an optional indexed cryoDRGN dataset as a RELION STAR."""

    from recovar.utils.helpers import write_starfile_from_cryodrgn_format

    particles_path = particles_path.expanduser().resolve(strict=True)
    poses_path = poses_path.expanduser().resolve(strict=True)
    ctf_path = ctf_path.expanduser().resolve(strict=True)
    output_star = output_star.expanduser().resolve()
    if output_star.exists():
        raise FileExistsError(f"refusing to overwrite materialized STAR: {output_star}")
    output_star.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vdam-source-star-", dir=output_star.parent) as tmp_dir:
        full_star = Path(tmp_dir) / "particles.star"
        write_starfile_from_cryodrgn_format(
            ctf_path,
            poses_path,
            particles_path,
            full_star,
        )
        star_data = starfile.read(full_star)

    particles = _particles_table(star_data)
    source_particle_count = len(particles)
    source_indices = None
    if source_indices_path is not None:
        source_indices_path = source_indices_path.expanduser().resolve(strict=True)
        with source_indices_path.open("rb") as handle:
            source_indices = np.asarray(pickle.load(handle))
        if source_indices.ndim != 1 or not np.issubdtype(source_indices.dtype, np.integer):
            raise ValueError("source index file must contain a one-dimensional integer array")
        source_indices = source_indices.astype(np.int64, copy=False)
        if source_indices.size == 0:
            raise ValueError("source index file must not be empty")
        if np.any(source_indices < 0) or np.any(source_indices >= source_particle_count):
            raise ValueError("source index file contains an out-of-range particle index")
        if np.unique(source_indices).size != source_indices.size:
            raise ValueError("source index file contains duplicate particle indices")
        selected = particles.iloc[source_indices].reset_index(drop=True)
    else:
        selected = particles.copy().reset_index(drop=True)

    if isinstance(star_data, dict):
        output_data = dict(star_data)
        output_data["particles"] = selected
    else:
        output_data = selected
    starfile.write(output_data, output_star, overwrite=False)

    manifest = {
        "schema": "recovar.vdam_cryodrgn_source_star.v1",
        "particles_path": str(particles_path),
        "poses_path": str(poses_path),
        "ctf_path": str(ctf_path),
        "particles_sha256": _sha256(particles_path),
        "poses_sha256": _sha256(poses_path),
        "ctf_sha256": _sha256(ctf_path),
        "source_particle_count": int(source_particle_count),
        "selected_particle_count": int(len(selected)),
        "source_indices_path": None if source_indices_path is None else str(source_indices_path),
        "source_indices_file_sha256": (
            None if source_indices_path is None else _sha256(source_indices_path)
        ),
        "selected_source_indices_sha256": (
            None if source_indices is None else hashlib.sha256(source_indices.tobytes()).hexdigest()
        ),
        "output_star": str(output_star),
        "output_star_sha256": _sha256(output_star),
        "translation_contract": "cryoDRGN box fractions converted to RELION Angstrom origins",
    }
    manifest_path = output_star.with_suffix(".materialization.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


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


def promote_legacy_optics(particles, *, image_size: int, pixel_size: float | None = None):
    """Return a RELION 3.1 optics/particles pair from a legacy single table."""

    detector_column = _optional_column(particles, "rlnDetectorPixelSize")
    magnification_column = _optional_column(particles, "rlnMagnification")
    if (detector_column is None) != (magnification_column is None):
        raise ValueError("legacy STAR must provide both detector pixel size and magnification")
    if detector_column is not None:
        detector_pixel_um = _constant_particle_value(particles, "rlnDetectorPixelSize")
        magnification = _constant_particle_value(particles, "rlnMagnification")
        if detector_pixel_um <= 0.0 or magnification <= 0.0:
            raise ValueError("legacy detector pixel size and magnification must be positive")
        derived_pixel_size = detector_pixel_um * 10_000.0 / magnification
        if pixel_size is not None and not np.isclose(float(pixel_size), derived_pixel_size, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"explicit pixel size {float(pixel_size)} disagrees with legacy metadata {derived_pixel_size}"
            )
        image_pixel_size = derived_pixel_size
    else:
        if pixel_size is None or float(pixel_size) <= 0.0:
            raise ValueError(
                "legacy STAR has no detector pixel size/magnification; provide an explicit positive pixel size"
            )
        image_pixel_size = float(pixel_size)
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
    origin_x = _optional_column(promoted, "rlnOriginX")
    origin_y = _optional_column(promoted, "rlnOriginY")
    if (origin_x is None) != (origin_y is None):
        raise ValueError("legacy STAR must provide both rlnOriginX and rlnOriginY")
    origin_x_angst = _optional_column(promoted, "rlnOriginXAngst")
    origin_y_angst = _optional_column(promoted, "rlnOriginYAngst")
    if (origin_x_angst is None) != (origin_y_angst is None):
        raise ValueError("legacy STAR must provide both rlnOriginXAngst and rlnOriginYAngst")
    if origin_x_angst is None and origin_x is not None:
        # Once an optics table is present RELION uses the Angstrom origin
        # columns. Merely retaining the legacy pixel columns silently resets
        # the active offsets to zero, so preserve their physical meaning.
        promoted["rlnOriginXAngst"] = np.asarray(promoted[origin_x], dtype=np.float64) * image_pixel_size
        promoted["rlnOriginYAngst"] = np.asarray(promoted[origin_y], dtype=np.float64) * image_pixel_size

    pmax_column = _optional_column(promoted, "rlnMaxValueProbDistribution")
    if pmax_column is not None:
        # This is a fresh InitialModel run. Old optimizer posteriors are not
        # acquisition metadata and make unvisited rows look active in RELION's
        # trajectory STAR files.
        promoted[pmax_column] = np.zeros(len(promoted), dtype=np.float64)
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
    pixel_size: float | None = None,
    synthesize_random_subsets: bool = False,
) -> dict:
    source_star = source_star.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required_absent in ("particles.star", "source_indices.npy", "fixture_manifest.json"):
        if (output_dir / required_absent).exists():
            raise FileExistsError(f"refusing to overwrite existing fixture file: {output_dir / required_absent}")

    star_data = starfile.read(source_star)
    particles = _particles_table(star_data)
    source_subset_column = _optional_column(particles, "rlnRandomSubset")
    synthetic_subset_ids = None
    if source_subset_column is None:
        if not synthesize_random_subsets:
            raise ValueError(
                "particle STAR has no rlnRandomSubset; pass --synthesize-random-subsets to create deterministic halves"
            )
        indices, synthetic_subset_ids = select_synthetic_half_indices(
            len(particles),
            particles_per_half=int(particles_per_half),
            seed=int(seed),
        )
    else:
        indices = select_balanced_half_indices(
            particles,
            particles_per_half=int(particles_per_half),
            seed=int(seed),
        )
    subset = particles.iloc[indices].reset_index(drop=True)
    if synthetic_subset_ids is not None:
        subset["rlnRandomSubset"] = synthetic_subset_ids
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
        output_star_data = promote_legacy_optics(subset, image_size=image_size, pixel_size=pixel_size)
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
        "explicit_pixel_size": None if pixel_size is None else float(pixel_size),
        "random_subsets_synthesized": synthetic_subset_ids is not None,
        "legacy_origins_converted_to_angstrom": bool(
            optics_promoted
            and _optional_column(subset, "rlnOriginX") is not None
            and _optional_column(subset, "rlnOriginXAngst") is None
        ),
        "stale_max_posterior_reset": bool(
            optics_promoted and _optional_column(subset, "rlnMaxValueProbDistribution") is not None
        ),
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-star", type=Path)
    source.add_argument("--particles", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--ctf", type=Path)
    parser.add_argument("--source-ind", type=Path)
    parser.add_argument("--materialized-source-star", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--particles-per-half", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--pixel-size", type=float)
    parser.add_argument("--synthesize-random-subsets", action="store_true")
    args = parser.parse_args()
    source_star = args.source_star
    if args.particles is not None:
        if args.poses is None or args.ctf is None:
            parser.error("--particles requires both --poses and --ctf")
        source_star = args.materialized_source_star
        if source_star is None:
            source_star = args.output_dir.resolve().parent / "source" / "particles.star"
        materialize_cryodrgn_source_star(
            particles_path=args.particles,
            poses_path=args.poses,
            ctf_path=args.ctf,
            output_star=source_star,
            source_indices_path=args.source_ind,
        )
    elif any(value is not None for value in (args.poses, args.ctf, args.source_ind, args.materialized_source_star)):
        parser.error("cryoDRGN source options require --particles")
    assert source_star is not None
    manifest = prepare_fixture(
        source_star=source_star,
        output_dir=args.output_dir,
        dataset=args.dataset,
        particles_per_half=args.particles_per_half,
        seed=args.seed,
        pixel_size=args.pixel_size,
        synthesize_random_subsets=args.synthesize_random_subsets,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
