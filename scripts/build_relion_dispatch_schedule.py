#!/usr/bin/env python
"""Convert a captured RELION MPI leader dispatch log to a strict replay NPZ.

The current v2 input is whitespace-separated with five integer columns::

    schema_version iteration follower_rank sorted_position original_part_id

The original particle ID makes the sorted-position namespace unambiguous.
Legacy four-column range logs are rejected because they cannot populate the
schema-v3 identity mapping; callers must never interpret ranges as original IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.relion_worker_scale import (
    make_relion_dispatch_schedule_from_chunks,
    relion_dispatch_metadata_relative_path,
    relion_oracle_id,
    relion_oracle_manifest_sha256,
    relion_ordered_particle_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-particles", required=True, type=int)
    parser.add_argument("--n-followers", required=True, type=int)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--random-seed", required=True, type=int)
    parser.add_argument(
        "--oracle-dir",
        required=True,
        help="Completed RELION run directory whose state produced the dispatch capture.",
    )
    parser.add_argument(
        "--particle-star",
        default=None,
        help=(
            "Particle data STAR relative to --oracle-dir. Defaults to the earliest "
            "run_it*_data.star and is hashed in row order."
        ),
    )
    parser.add_argument(
        "--oracle-artifact",
        action="append",
        default=None,
        help=(
            "Additional state artifact relative to --oracle-dir to include in the content "
            "manifest. Repeat as needed. All top-level run_it*.star/run_it*.mrc state "
            "files plus particles.star/reference_init.mrc when present are always included."
        ),
    )
    return parser


def _relative_oracle_path(root: Path, value: str | Path) -> str:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SystemExit(f"RELION oracle artifact must be inside {root}: {path}") from exc


def _default_oracle_artifacts(root: Path) -> list[str]:
    selected = {
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file()
        and (
            (path.name.startswith("run_it") and path.suffix in {".star", ".mrc"})
            or path.name
            in {
                "particles.star",
                "reference_init.mrc",
                "run_optimiser.star",
                "run_sampling.star",
                "run_data.star",
                "run_model.star",
            }
        )
    }
    if not selected:
        raise SystemExit(f"No RELION run state artifacts found in {root}")
    return sorted(selected)


def main() -> None:
    args = _parser().parse_args()
    oracle_dir = Path(args.oracle_dir).expanduser().resolve()
    if not oracle_dir.is_dir():
        raise SystemExit(f"--oracle-dir is not a directory: {oracle_dir}")
    dispatch_relative = _relative_oracle_path(oracle_dir, args.dispatch_log)
    log_path = oracle_dir / dispatch_relative
    if not log_path.is_file():
        raise SystemExit(f"RELION dispatch log does not exist: {log_path}")
    if args.particle_star is None:
        particle_candidates = sorted(oracle_dir.glob("run_it*_data.star"))
        if not particle_candidates:
            particle_candidates = [oracle_dir / "particles.star"]
        particle_path = particle_candidates[0]
    else:
        particle_path = oracle_dir / args.particle_star
    particle_relative = _relative_oracle_path(oracle_dir, particle_path)
    particle_path = oracle_dir / particle_relative
    if not particle_path.is_file():
        raise SystemExit(f"RELION oracle particle STAR does not exist: {particle_path}")

    rows = np.loadtxt(log_path, dtype=np.int64, comments="#", ndmin=2)
    if rows.ndim != 2 or rows.shape[1] not in (4, 5):
        raise SystemExit(
            f"{log_path} must contain legacy four-column ranges or v2 five-column "
            f"identity records; observed shape {rows.shape}"
        )
    if rows.shape[1] == 5:
        if not np.all(rows[:, 0] == 2):
            raise SystemExit(f"{log_path} contains a non-v2 dispatch schema record")
        chunk_iterations = rows[:, 1]
        chunk_ranks = rows[:, 2]
        chunk_first = rows[:, 3]
        chunk_last = rows[:, 3]
        original_ids = rows[:, 4]
        for iteration in np.unique(chunk_iterations):
            selected = chunk_iterations == iteration
            expected = np.arange(int(args.n_particles), dtype=np.int64)
            if not np.array_equal(np.sort(chunk_first[selected]), expected):
                raise SystemExit(
                    f"v2 dispatch sorted positions are not a bijection at iteration {iteration}"
                )
            if not np.array_equal(np.sort(original_ids[selected]), expected):
                raise SystemExit(
                    f"v2 dispatch original particle IDs are not a bijection at iteration {iteration}"
                )
    else:
        raise SystemExit(
            "legacy four-column dispatch ranges cannot produce schema v3 because they lack "
            "the authoritative original_particle_id_by_sorted_position mapping"
        )
    dispatch_metadata_relative = relion_dispatch_metadata_relative_path(dispatch_relative)
    dispatch_metadata_path = oracle_dir / dispatch_metadata_relative
    dispatch_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dispatch_metadata = {
        "schema_version": 2,
        "dispatch_log_schema_version": 2,
        "schedule_schema_version": 3,
        "dispatch_log_relative_path": dispatch_relative,
        "n_particles": int(args.n_particles),
        "n_followers": int(args.n_followers),
        "pool_size": int(args.pool_size),
        "random_seed": int(args.random_seed),
    }
    dispatch_metadata_path.write_text(
        json.dumps(dispatch_metadata, sort_keys=True, indent=2) + "\n"
    )

    artifact_paths = _default_oracle_artifacts(oracle_dir)
    if args.oracle_artifact:
        artifact_paths = sorted(
            set(artifact_paths)
            | {_relative_oracle_path(oracle_dir, value) for value in args.oracle_artifact}
        )
    if particle_relative not in artifact_paths:
        artifact_paths.append(particle_relative)
        artifact_paths.sort()
    if dispatch_relative not in artifact_paths:
        artifact_paths.append(dispatch_relative)
        artifact_paths.sort()
    if dispatch_metadata_relative not in artifact_paths:
        artifact_paths.append(dispatch_metadata_relative)
        artifact_paths.sort()

    import starfile

    particle_data = starfile.read(particle_path)
    particles = particle_data["particles"] if isinstance(particle_data, dict) else particle_data
    if len(particles) != int(args.n_particles):
        raise SystemExit(
            f"--n-particles={args.n_particles} does not match {particle_path} "
            f"row count {len(particles)}"
        )
    manifest_sha256 = relion_oracle_manifest_sha256(oracle_dir, artifact_paths)
    particle_order_sha256 = relion_ordered_particle_sha256(particles)
    oracle_id = relion_oracle_id(
        manifest_sha256=manifest_sha256,
        particle_order_sha256=particle_order_sha256,
    )
    iterations = np.unique(chunk_iterations)
    original_by_sorted = np.empty((iterations.size, int(args.n_particles)), dtype=np.int64)
    for row_idx, iteration in enumerate(iterations):
        selected = chunk_iterations == iteration
        original_by_sorted[row_idx, chunk_first[selected]] = original_ids[selected]
    schedule = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=iterations,
        chunk_iterations=chunk_iterations,
        chunk_ranks=chunk_ranks,
        chunk_first=chunk_first,
        chunk_last=chunk_last,
        n_particles=args.n_particles,
        original_particle_id_by_sorted_position=original_by_sorted,
        n_followers=args.n_followers,
        pool_size=args.pool_size,
        random_seed=args.random_seed,
        oracle_id=oracle_id,
        oracle_manifest_sha256=manifest_sha256,
        oracle_artifact_paths=artifact_paths,
        particle_order_sha256=particle_order_sha256,
        particle_star_relative_path=particle_relative,
        dispatch_log_relative_path=dispatch_relative,
        source=str(log_path),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.int64(3),
        relion_iterations=schedule.relion_iterations,
        owner_by_sorted_position=schedule.owner_by_sorted_position,
        original_particle_id_by_sorted_position=(
            schedule.original_particle_id_by_sorted_position
        ),
        n_followers=np.int64(schedule.n_followers),
        pool_size=np.int64(schedule.pool_size),
        random_seed=np.int64(schedule.random_seed),
        oracle_id=np.asarray(schedule.oracle_id),
        oracle_manifest_sha256=np.asarray(schedule.oracle_manifest_sha256),
        oracle_artifact_paths=np.asarray(schedule.oracle_artifact_paths),
        particle_order_sha256=np.asarray(schedule.particle_order_sha256),
        particle_star_relative_path=np.asarray(schedule.particle_star_relative_path),
        dispatch_log_relative_path=np.asarray(schedule.dispatch_log_relative_path),
        source=np.asarray(schedule.source),
    )
    print(
        f"wrote {output}: iterations={schedule.relion_iterations.tolist()} "
        f"particles={schedule.owner_by_sorted_position.shape[1]} "
        f"followers={schedule.n_followers} pool={schedule.pool_size}"
        f" oracle_id={schedule.oracle_id}"
    )


if __name__ == "__main__":
    main()
