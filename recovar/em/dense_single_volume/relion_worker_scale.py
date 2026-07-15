"""Strict RELION MPI-follower group-scale emulation.

RELION's segmented ``MlWsumModel::pack`` path uses the optics-group count
when packing group-scale XA/AA statistics.  In an all-data Class3D run with
more scale groups than optics groups, only the leading optics-group-sized
prefix is MPI-reduced; the remaining scale statistics and resulting scale
vectors stay follower-local.  These helpers reproduce that behavior without
leaking it into ordinary RECOVAR refinement.

Expectation ownership is *not* a static equal partition.  RELION's leader
hands each next ``--pool`` chunk to whichever follower requests work next.
Exact replay therefore requires a schedule captured from the same RELION run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RelionFollowerScaleState:
    """Follower-local physical-group scale vectors and normalization counts."""

    scales: np.ndarray
    group_counts: np.ndarray
    n_optics_groups: int

    @property
    def n_followers(self) -> int:
        return int(self.scales.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.scales.shape[1])


@dataclass(frozen=True)
class RelionDispatchSchedule:
    """Captured dynamic follower ownership in shuffled-position order."""

    relion_iterations: np.ndarray
    owner_by_sorted_position: np.ndarray
    original_particle_id_by_sorted_position: np.ndarray
    n_followers: int
    pool_size: int
    random_seed: int
    oracle_id: str
    oracle_manifest_sha256: str
    oracle_artifact_paths: tuple[str, ...]
    particle_order_sha256: str
    particle_star_relative_path: str
    dispatch_log_relative_path: str
    source: str


@dataclass(frozen=True)
class RelionFollowerScaleReplay:
    """Complete follower-scale matrices injected at selected numbered iterations."""

    relion_iterations: np.ndarray
    follower_scales: np.ndarray
    oracle_id: str
    schema_version: int
    boundary: str
    source_artifact_relative_paths: tuple[str, ...]
    source: str


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RELION_ORACLE_SCHEMA_VERSION = 3
_RELION_FOLLOWER_REPLAY_SCHEMA_VERSION = 1
_RELION_FOLLOWER_REPLAY_BOUNDARY = "numbered_pre_score"


def relion_dispatch_metadata_relative_path(dispatch_log_relative_path: str) -> str:
    """Return the manifest-bound metadata sidecar for a dispatch capture."""

    dispatch = _validate_oracle_artifact_paths([dispatch_log_relative_path])[0]
    return f"{dispatch}.recovar_schedule.json"


def _load_relion_dispatch_rows(path: Path) -> np.ndarray:
    try:
        rows = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot parse RELION dispatch log {path}: {exc}") from exc
    if rows.ndim != 2 or rows.shape[1] not in (4, 5) or rows.shape[0] < 1:
        raise ValueError(
            f"RELION dispatch log {path} must contain legacy four-column ranges "
            "or v2 five-column identity records; "
            f"observed shape {rows.shape}"
        )
    if rows.shape[1] == 5 and not np.all(rows[:, 0] == 2):
        raise ValueError(f"RELION dispatch log {path} contains a non-v2 schema record")
    return rows


def _relion_dispatch_chunk_columns(rows: np.ndarray, n_particles: int):
    """Return replay chunk columns without conflating sorted and original IDs."""

    if rows.shape[1] == 4:
        raise ValueError(
            "legacy four-column dispatch ranges do not contain the authoritative "
            "sorted-position to original-particle mapping"
        )
    iterations = rows[:, 1]
    ranks = rows[:, 2]
    sorted_positions = rows[:, 3]
    original_ids = rows[:, 4]
    expected = np.arange(int(n_particles), dtype=np.int64)
    for iteration in np.unique(iterations):
        selected = iterations == iteration
        if not np.array_equal(np.sort(sorted_positions[selected]), expected):
            raise ValueError(
                f"v2 dispatch sorted positions are not a bijection at iteration {iteration}"
            )
        if not np.array_equal(np.sort(original_ids[selected]), expected):
            raise ValueError(
                f"v2 dispatch original particle IDs are not a bijection at iteration {iteration}"
            )
    return iterations, ranks, sorted_positions, sorted_positions, original_ids


def _load_relion_follower_scale_sources(
    oracle_dir: str | Path,
    artifact_paths,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse canonical RELION follower post-scale TSV dumps."""

    root = Path(oracle_dir).expanduser().resolve()
    records: dict[tuple[int, int, int], float] = {}
    for relative in _validate_oracle_artifact_paths(artifact_paths):
        path = root / relative
        try:
            table = np.genfromtxt(path, names=True, delimiter="\t", encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot parse follower-scale source {path}: {exc}") from exc
        required = {"iteration", "mpi_rank", "group_index", "scale_post"}
        names = set(table.dtype.names or ())
        if not required.issubset(names):
            raise ValueError(
                f"follower-scale source {path} is missing columns {sorted(required - names)}"
            )
        rows = np.atleast_1d(table)
        for row in rows:
            key = (int(row["iteration"]), int(row["mpi_rank"]), int(row["group_index"]))
            if key in records:
                raise ValueError(f"duplicate follower-scale source row {key}")
            records[key] = float(row["scale_post"])
    if not records:
        raise ValueError("follower-scale source artifacts contain no rows")
    iterations = np.asarray(sorted({key[0] for key in records}), dtype=np.int64)
    ranks = sorted({key[1] for key in records})
    groups = sorted({key[2] for key in records})
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("follower-scale source mpi_rank values must be contiguous and 1-based")
    if groups != list(range(len(groups))):
        raise ValueError("follower-scale source group_index values must be contiguous and 0-based")
    scales = np.empty((iterations.size, len(ranks), len(groups)), dtype=np.float64)
    for iteration_idx, iteration in enumerate(iterations):
        for rank in ranks:
            for group in groups:
                key = (int(iteration), int(rank), int(group))
                if key not in records:
                    raise ValueError(f"follower-scale sources are missing row {key}")
                scales[iteration_idx, rank - 1, group] = records[key]
    return iterations, scales


def _validate_sha256(value: str, *, name: str) -> str:
    value = str(value)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_oracle_artifact_paths(paths) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in np.asarray(paths).reshape(-1).tolist())
    if not normalized:
        raise ValueError("oracle_artifact_paths must be non-empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("oracle_artifact_paths must be sorted and unique")
    for value in normalized:
        path = Path(value)
        if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
            raise ValueError(
                "oracle_artifact_paths must contain normalized relative POSIX paths"
            )
    return normalized


def relion_oracle_manifest_sha256(
    oracle_dir: str | Path,
    artifact_paths,
) -> str:
    """Hash exact RELION state-file names and bytes without binding to a host path."""

    root = Path(oracle_dir).expanduser().resolve()
    paths = _validate_oracle_artifact_paths(artifact_paths)
    digest = hashlib.sha256(b"recovar-relion-oracle-manifest-v1\0")
    for relative in paths:
        artifact = root / relative
        if not artifact.is_file():
            raise ValueError(f"RELION oracle artifact is missing or not a file: {artifact}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(artifact.stat().st_size).to_bytes(8, "little"))
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def relion_ordered_particle_sha256(particles) -> str:
    """Hash RELION particle identities and parity-relevant labels in row order."""

    if "rlnImageName" not in particles.columns:
        raise ValueError("RELION oracle particle STAR is missing rlnImageName")
    columns = [
        name
        for name in (
            "rlnImageName",
            "rlnOpticsGroup",
            "rlnRandomSubset",
            "rlnGroupNumber",
        )
        if name in particles.columns
    ]
    digest = hashlib.sha256(b"recovar-relion-ordered-particles-v1\0")
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(int(len(particles)).to_bytes(8, "little"))
    for row in particles[columns].itertuples(index=False, name=None):
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def relion_oracle_id(*, manifest_sha256: str, particle_order_sha256: str) -> str:
    """Derive the portable identity shared by schedule and follower replay."""

    manifest = _validate_sha256(manifest_sha256, name="oracle_manifest_sha256")
    order = _validate_sha256(particle_order_sha256, name="particle_order_sha256")
    digest = hashlib.sha256(b"recovar-relion-oracle-id-v1\0")
    digest.update(manifest.encode("ascii"))
    digest.update(order.encode("ascii"))
    return digest.hexdigest()


def verify_relion_dispatch_schedule_oracle(
    schedule: RelionDispatchSchedule,
    oracle_dir: str | Path,
) -> None:
    """Fail closed unless the supplied RELION directory is the captured oracle."""

    root = Path(oracle_dir).expanduser().resolve()
    observed_manifest = relion_oracle_manifest_sha256(
        root,
        schedule.oracle_artifact_paths,
    )
    if observed_manifest != schedule.oracle_manifest_sha256:
        raise ValueError(
            "RELION oracle state manifest does not match the dispatch schedule "
            f"({observed_manifest} != {schedule.oracle_manifest_sha256})"
        )
    particle_path = root / schedule.particle_star_relative_path
    try:
        import starfile

        table = starfile.read(particle_path)
    except Exception as exc:
        raise ValueError(f"cannot read RELION oracle particle STAR {particle_path}: {exc}") from exc
    particles = table["particles"] if isinstance(table, dict) else table
    observed_order = relion_ordered_particle_sha256(particles)
    if observed_order != schedule.particle_order_sha256:
        raise ValueError(
            "ordered RELION particle identity does not match the dispatch schedule "
            f"({observed_order} != {schedule.particle_order_sha256})"
        )

    metadata_relative = relion_dispatch_metadata_relative_path(
        schedule.dispatch_log_relative_path
    )
    if metadata_relative not in schedule.oracle_artifact_paths:
        raise ValueError(
            "dispatch metadata sidecar is absent from the verified oracle manifest"
        )
    metadata_path = root / metadata_relative
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read dispatch metadata {metadata_path}: {exc}") from exc
    expected_metadata = {
        "schema_version": 2,
        "dispatch_log_schema_version": 2,
        "schedule_schema_version": 3,
        "dispatch_log_relative_path": schedule.dispatch_log_relative_path,
        "n_particles": int(schedule.owner_by_sorted_position.shape[1]),
        "n_followers": int(schedule.n_followers),
        "pool_size": int(schedule.pool_size),
        "random_seed": int(schedule.random_seed),
    }
    if metadata != expected_metadata:
        raise ValueError(
            "manifest-bound dispatch metadata does not match the schedule payload: "
            f"observed={metadata!r} expected={expected_metadata!r}"
        )
    rows = _load_relion_dispatch_rows(root / schedule.dispatch_log_relative_path)
    chunk_iterations, chunk_ranks, chunk_first, chunk_last, chunk_original_ids = _relion_dispatch_chunk_columns(
        rows, expected_metadata["n_particles"]
    )
    original_by_sorted = np.empty(
        (schedule.relion_iterations.size, expected_metadata["n_particles"]), dtype=np.int64
    )
    for row_idx, iteration in enumerate(schedule.relion_iterations):
        selected = chunk_iterations == iteration
        original_by_sorted[row_idx, chunk_first[selected]] = chunk_original_ids[selected]
    reconstructed = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=np.unique(chunk_iterations),
        chunk_iterations=chunk_iterations,
        chunk_ranks=chunk_ranks,
        chunk_first=chunk_first,
        chunk_last=chunk_last,
        n_particles=expected_metadata["n_particles"],
        original_particle_id_by_sorted_position=original_by_sorted,
        n_followers=expected_metadata["n_followers"],
        pool_size=expected_metadata["pool_size"],
        random_seed=expected_metadata["random_seed"],
        oracle_id=schedule.oracle_id,
        oracle_manifest_sha256=schedule.oracle_manifest_sha256,
        oracle_artifact_paths=schedule.oracle_artifact_paths,
        particle_order_sha256=schedule.particle_order_sha256,
        particle_star_relative_path=schedule.particle_star_relative_path,
        dispatch_log_relative_path=schedule.dispatch_log_relative_path,
        source=str(root / schedule.dispatch_log_relative_path),
    )
    if not np.array_equal(reconstructed.relion_iterations, schedule.relion_iterations):
        raise ValueError("schedule iterations were not derived from the bound dispatch log")
    if not np.array_equal(
        reconstructed.owner_by_sorted_position,
        schedule.owner_by_sorted_position,
    ):
        raise ValueError("schedule follower owners were not derived from the bound dispatch log")
    if not np.array_equal(
        reconstructed.original_particle_id_by_sorted_position,
        schedule.original_particle_id_by_sorted_position,
    ):
        raise ValueError("schedule particle identities were not derived from the bound dispatch log")


def validate_relion_follower_scale_replay(
    replay: RelionFollowerScaleReplay,
    *,
    n_followers: int | None = None,
    n_groups: int | None = None,
    schedule_iterations=None,
    schedule_oracle_id: str | None = None,
    schedule_artifact_paths=None,
    numbered_iterations=None,
    first_numbered_iteration: int | None = None,
    oracle_dir: str | Path | None = None,
) -> None:
    """Validate replay contents and, optionally, the active strict topology."""

    iterations_raw = np.asarray(replay.relion_iterations)
    if not np.issubdtype(iterations_raw.dtype, np.integer):
        raise ValueError("relion_iterations must have an integer dtype")
    iterations = np.asarray(iterations_raw, dtype=np.int64).reshape(-1)
    scales_raw = np.asarray(replay.follower_scales)
    if not (
        np.issubdtype(scales_raw.dtype, np.floating)
        or np.issubdtype(scales_raw.dtype, np.integer)
    ):
        raise ValueError("follower_scales must have a real numeric dtype")
    scales = np.asarray(scales_raw, dtype=np.float64)
    _validate_sha256(replay.oracle_id, name="follower replay oracle_id")
    if int(replay.schema_version) != _RELION_FOLLOWER_REPLAY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported follower-scale replay schema_version "
            f"{replay.schema_version}; expected {_RELION_FOLLOWER_REPLAY_SCHEMA_VERSION}"
        )
    if replay.boundary != _RELION_FOLLOWER_REPLAY_BOUNDARY:
        raise ValueError(
            "follower-scale replay boundary must be "
            f"{_RELION_FOLLOWER_REPLAY_BOUNDARY!r}"
        )
    source_artifacts = _validate_oracle_artifact_paths(
        replay.source_artifact_relative_paths
    )
    if iterations.size < 1 or np.unique(iterations).size != iterations.size:
        raise ValueError("relion_iterations must be a non-empty unique vector")
    if np.any(iterations < 1) or np.any(np.diff(iterations) <= 0):
        raise ValueError("relion_iterations must be positive and strictly increasing")
    if scales.ndim != 3 or scales.shape[0] != iterations.size:
        raise ValueError(
            "follower_scales must have shape (n_iterations, n_followers, n_groups)"
        )
    if scales.shape[1] < 1 or scales.shape[2] < 1:
        raise ValueError("follower_scales follower and group dimensions must be positive")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("follower_scales must be finite and strictly positive")
    if n_followers is not None and scales.shape[1] != int(n_followers):
        raise ValueError(
            "follower_scales follower dimension does not match the dispatch topology "
            f"({scales.shape[1]} != {int(n_followers)})"
        )
    if n_groups is not None and scales.shape[2] != int(n_groups):
        raise ValueError(
            "follower_scales group dimension does not match the physical group axis "
            f"({scales.shape[2]} != {int(n_groups)})"
        )
    if schedule_iterations is not None:
        captured = set(np.asarray(schedule_iterations, dtype=np.int64).reshape(-1).tolist())
        missing = sorted(set(iterations.tolist()) - captured)
        if missing:
            raise ValueError(
                "follower-scale replay iterations are absent from the captured dispatch schedule: "
                f"{missing}"
            )
    if schedule_oracle_id is not None and replay.oracle_id != str(schedule_oracle_id):
        raise ValueError(
            "follower-scale replay oracle_id does not match the dispatch schedule "
            f"({replay.oracle_id} != {schedule_oracle_id})"
        )
    if schedule_artifact_paths is not None:
        schedule_artifacts = set(_validate_oracle_artifact_paths(schedule_artifact_paths))
        missing_artifacts = sorted(set(source_artifacts) - schedule_artifacts)
        if missing_artifacts:
            raise ValueError(
                "follower-scale replay source artifacts are absent from the verified "
                f"oracle manifest: {missing_artifacts}"
            )
    if oracle_dir is not None:
        source_iterations, source_scales = _load_relion_follower_scale_sources(
            oracle_dir,
            source_artifacts,
        )
        # The raw dumps are written after M-step n; this replay boundary is
        # the next numbered pre-score step n+1.
        if not np.array_equal(source_iterations + 1, iterations):
            raise ValueError(
                "follower-scale replay iterations were not derived from the bound "
                "source artifacts"
            )
        if not np.array_equal(source_scales, scales):
            raise ValueError(
                "follower-scale replay matrix was not derived from the bound source artifacts"
            )
    if numbered_iterations is not None:
        numbered = set(np.asarray(numbered_iterations, dtype=np.int64).reshape(-1).tolist())
        outside_numbered = sorted(set(iterations.tolist()) - numbered)
        if outside_numbered:
            raise ValueError(
                "follower-scale replay iterations are outside the requested numbered "
                f"refinement range: {outside_numbered}"
            )
    if first_numbered_iteration is not None and int(first_numbered_iteration) in iterations:
        raise ValueError(
            "follower-scale replay cannot target the first numbered RELION iteration "
            f"{int(first_numbered_iteration)}: resident image-normalization state is not "
            "initialized before its scoring boundary"
        )


def validate_relion_follower_scale_replay_application(
    replay: RelionFollowerScaleReplay,
    *,
    applied_iterations,
) -> tuple[np.ndarray, np.ndarray]:
    """Require every requested replay row to have been applied exactly once and in order."""

    requested = np.asarray(replay.relion_iterations, dtype=np.int64).reshape(-1)
    applied_raw = np.asarray(applied_iterations)
    if applied_raw.size and not np.issubdtype(applied_raw.dtype, np.integer):
        raise RuntimeError("applied follower-scale replay iterations must have an integer dtype")
    applied = np.asarray(applied_raw, dtype=np.int64).reshape(-1)
    if np.array_equal(applied, requested):
        return requested.copy(), applied.copy()

    requested_set = set(requested.tolist())
    applied_set = set(applied.tolist())
    missing = sorted(requested_set - applied_set)
    unexpected = sorted(applied_set - requested_set)
    duplicate = sorted(
        int(value)
        for value, count in zip(*np.unique(applied, return_counts=True), strict=True)
        if int(count) > 1
    )
    raise RuntimeError(
        "RELION follower-scale replay was not applied exactly once at every requested "
        "numbered iteration: "
        f"requested={requested.tolist()} applied={applied.tolist()} missing={missing} "
        f"unexpected={unexpected} duplicate={duplicate}"
    )


def load_relion_follower_scale_replay(path: str | Path) -> RelionFollowerScaleReplay:
    """Load selected complete follower-scale states from a diagnostic NPZ."""

    replay_path = Path(path).expanduser().resolve()
    with np.load(replay_path, allow_pickle=False) as values:
        required = {
            "schema_version",
            "boundary",
            "source_artifact_relative_paths",
            "relion_iterations",
            "follower_scales",
            "oracle_id",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(
                f"RELION follower-scale replay {replay_path} is missing keys: "
                f"{', '.join(missing)}"
            )
        iterations_raw = np.asarray(values["relion_iterations"])
        if not np.issubdtype(iterations_raw.dtype, np.integer):
            raise ValueError("relion_iterations must have an integer dtype")
        scales_raw = np.asarray(values["follower_scales"])
        if not (
            np.issubdtype(scales_raw.dtype, np.floating)
            or np.issubdtype(scales_raw.dtype, np.integer)
        ):
            raise ValueError("follower_scales must have a real numeric dtype")
        schema_raw = np.asarray(values["schema_version"])
        if not np.issubdtype(schema_raw.dtype, np.integer) or schema_raw.size != 1:
            raise ValueError("schema_version must be an integer scalar")
        replay = RelionFollowerScaleReplay(
            relion_iterations=np.asarray(iterations_raw, dtype=np.int64).reshape(-1),
            follower_scales=np.asarray(scales_raw, dtype=np.float64),
            oracle_id=str(np.asarray(values["oracle_id"]).reshape(())),
            schema_version=int(schema_raw.reshape(())),
            boundary=str(np.asarray(values["boundary"]).reshape(())),
            source_artifact_relative_paths=_validate_oracle_artifact_paths(
                values["source_artifact_relative_paths"]
            ),
            source=(
                str(np.asarray(values["source"]).reshape(()))
                if "source" in values.files
                else str(replay_path)
            ),
        )
    validate_relion_follower_scale_replay(replay)
    return replay


def load_relion_dispatch_schedule(path: str | Path) -> RelionDispatchSchedule:
    """Load and validate an exact per-iteration RELION dispatch schedule."""

    schedule_path = Path(path).expanduser().resolve()
    with np.load(schedule_path, allow_pickle=False) as values:
        required = {
            "schema_version",
            "relion_iterations",
            "owner_by_sorted_position",
            "original_particle_id_by_sorted_position",
            "n_followers",
            "pool_size",
            "random_seed",
            "oracle_id",
            "oracle_manifest_sha256",
            "oracle_artifact_paths",
            "particle_order_sha256",
            "particle_star_relative_path",
            "dispatch_log_relative_path",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(
                f"RELION dispatch schedule {schedule_path} is missing keys: {', '.join(missing)}"
            )
        schema_raw = np.asarray(values["schema_version"])
        if not np.issubdtype(schema_raw.dtype, np.integer) or schema_raw.size != 1:
            raise ValueError("schema_version must be an integer scalar")
        schema_version = int(schema_raw.reshape(()))
        if schema_version != _RELION_ORACLE_SCHEMA_VERSION:
            if schema_version == 2:
                raise ValueError(
                    "RELION dispatch schedule schema v2 lacks the authoritative "
                    "original_particle_id_by_sorted_position mapping; rebuild schema v3 "
                    "from a five-column RELION text-log schema v2 capture. Legacy "
                    "four-column logs cannot be migrated exactly."
                )
            raise ValueError(
                "unsupported RELION dispatch schedule schema_version "
                f"{schema_version}; expected {_RELION_ORACLE_SCHEMA_VERSION}"
            )
        iterations_raw = np.asarray(values["relion_iterations"])
        if not np.issubdtype(iterations_raw.dtype, np.integer):
            raise ValueError("relion_iterations must have an integer dtype")
        iterations = np.asarray(iterations_raw, dtype=np.int64).reshape(-1)
        owners_raw = np.asarray(values["owner_by_sorted_position"])
        if not np.issubdtype(owners_raw.dtype, np.integer):
            raise ValueError("owner_by_sorted_position must have an integer dtype")
        owners = np.asarray(owners_raw, dtype=np.int64)
        originals_raw = np.asarray(values["original_particle_id_by_sorted_position"])
        if not np.issubdtype(originals_raw.dtype, np.integer):
            raise ValueError("original_particle_id_by_sorted_position must have an integer dtype")
        originals = np.asarray(originals_raw, dtype=np.int64)
        integer_scalars = {}
        for key in ("n_followers", "pool_size", "random_seed"):
            raw_value = np.asarray(values[key])
            if not np.issubdtype(raw_value.dtype, np.integer):
                raise ValueError(f"{key} must have an integer dtype")
            if raw_value.size != 1:
                raise ValueError(f"{key} must be a scalar")
            integer_scalars[key] = int(raw_value.reshape(()))
        n_followers = integer_scalars["n_followers"]
        pool_size = integer_scalars["pool_size"]
        random_seed = integer_scalars["random_seed"]
        source = (
            str(np.asarray(values["source"]).reshape(()))
            if "source" in values.files
            else str(schedule_path)
        )
        oracle_id_value = str(np.asarray(values["oracle_id"]).reshape(()))
        manifest_sha256 = _validate_sha256(
            str(np.asarray(values["oracle_manifest_sha256"]).reshape(())),
            name="oracle_manifest_sha256",
        )
        particle_order_sha256 = _validate_sha256(
            str(np.asarray(values["particle_order_sha256"]).reshape(())),
            name="particle_order_sha256",
        )
        artifact_paths = _validate_oracle_artifact_paths(values["oracle_artifact_paths"])
        particle_star_relative_path = str(
            np.asarray(values["particle_star_relative_path"]).reshape(())
        )
        _validate_oracle_artifact_paths([particle_star_relative_path])
        if particle_star_relative_path not in artifact_paths:
            raise ValueError(
                "particle_star_relative_path must be included in oracle_artifact_paths"
            )
        dispatch_log_relative_path = str(
            np.asarray(values["dispatch_log_relative_path"]).reshape(())
        )
        _validate_oracle_artifact_paths([dispatch_log_relative_path])
        if dispatch_log_relative_path not in artifact_paths:
            raise ValueError(
                "dispatch_log_relative_path must be included in oracle_artifact_paths"
            )
        dispatch_metadata_relative_path = relion_dispatch_metadata_relative_path(
            dispatch_log_relative_path
        )
        if dispatch_metadata_relative_path not in artifact_paths:
            raise ValueError(
                "dispatch metadata sidecar must be included in oracle_artifact_paths"
            )
        expected_oracle_id = relion_oracle_id(
            manifest_sha256=manifest_sha256,
            particle_order_sha256=particle_order_sha256,
        )
        if oracle_id_value != expected_oracle_id:
            raise ValueError(
                "oracle_id is inconsistent with the captured manifest and particle-order hashes"
            )

    if iterations.size < 1 or np.unique(iterations).size != iterations.size:
        raise ValueError("relion_iterations must be a non-empty unique vector")
    if np.any(iterations < 1) or np.any(np.diff(iterations) <= 0):
        raise ValueError("relion_iterations must be positive and strictly increasing")
    if owners.ndim != 2 or owners.shape[0] != iterations.size or owners.shape[1] < 1:
        raise ValueError(
            "owner_by_sorted_position must have shape (n_iterations, n_particles)"
        )
    if originals.shape != owners.shape:
        raise ValueError(
            "original_particle_id_by_sorted_position must match owner_by_sorted_position shape"
        )
    expected_originals = np.arange(owners.shape[1], dtype=np.int64)
    for row_idx in range(originals.shape[0]):
        if not np.array_equal(np.sort(originals[row_idx]), expected_originals):
            raise ValueError(
                "original_particle_id_by_sorted_position must be a permutation in every iteration"
            )
    if n_followers < 1:
        raise ValueError("n_followers must be positive")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if np.any(owners < 0) or np.any(owners >= n_followers):
        raise ValueError("captured follower owners are out of bounds")
    return RelionDispatchSchedule(
        relion_iterations=iterations,
        owner_by_sorted_position=owners,
        original_particle_id_by_sorted_position=originals,
        n_followers=n_followers,
        pool_size=pool_size,
        random_seed=random_seed,
        oracle_id=oracle_id_value,
        oracle_manifest_sha256=manifest_sha256,
        oracle_artifact_paths=artifact_paths,
        particle_order_sha256=particle_order_sha256,
        particle_star_relative_path=particle_star_relative_path,
        dispatch_log_relative_path=dispatch_log_relative_path,
        source=source,
    )


def make_relion_dispatch_schedule_from_chunks(
    *,
    relion_iterations,
    chunk_iterations,
    chunk_first,
    chunk_last,
    chunk_ranks,
    n_particles: int,
    original_particle_id_by_sorted_position,
    n_followers: int,
    pool_size: int,
    random_seed: int,
    oracle_id: str,
    oracle_manifest_sha256: str,
    oracle_artifact_paths,
    particle_order_sha256: str,
    particle_star_relative_path: str,
    dispatch_log_relative_path: str,
    source: str,
) -> RelionDispatchSchedule:
    """Validate leader dispatch chunks and materialize sorted-position owners."""

    iterations = np.asarray(relion_iterations, dtype=np.int64).reshape(-1)
    chunk_iters = np.asarray(chunk_iterations, dtype=np.int64).reshape(-1)
    first = np.asarray(chunk_first, dtype=np.int64).reshape(-1)
    last = np.asarray(chunk_last, dtype=np.int64).reshape(-1)
    ranks = np.asarray(chunk_ranks, dtype=np.int64).reshape(-1)
    originals = np.asarray(original_particle_id_by_sorted_position)
    if not (chunk_iters.shape == first.shape == last.shape == ranks.shape):
        raise ValueError("dispatch chunk columns must have identical shapes")
    n_particles = int(n_particles)
    n_followers = int(n_followers)
    pool_size = int(pool_size)
    if iterations.size < 1 or np.unique(iterations).size != iterations.size:
        raise ValueError("relion_iterations must be non-empty and unique")
    if n_particles < 1 or n_followers < 1 or pool_size < 1:
        raise ValueError("n_particles, n_followers, and pool_size must be positive")
    if not np.issubdtype(originals.dtype, np.integer):
        raise ValueError("original_particle_id_by_sorted_position must have an integer dtype")
    originals = np.asarray(originals, dtype=np.int64)
    if originals.shape != (iterations.size, n_particles):
        raise ValueError(
            "original_particle_id_by_sorted_position must have shape "
            "(n_iterations, n_particles)"
        )
    expected_originals = np.arange(n_particles, dtype=np.int64)
    for row in originals:
        if not np.array_equal(np.sort(row), expected_originals):
            raise ValueError(
                "original_particle_id_by_sorted_position must be a permutation in every iteration"
            )
    if np.any(first < 0) or np.any(last < first) or np.any(last >= n_particles):
        raise ValueError("dispatch chunks contain invalid sorted-position bounds")
    if np.any(last - first + 1 > pool_size):
        raise ValueError("dispatch chunk exceeds the recorded RELION pool size")
    if np.any(ranks < 1) or np.any(ranks > n_followers):
        raise ValueError("dispatch ranks must be RELION follower ranks 1..n_followers")
    artifact_paths = _validate_oracle_artifact_paths(oracle_artifact_paths)
    particle_star_relative_path = _validate_oracle_artifact_paths(
        [particle_star_relative_path]
    )[0]
    if particle_star_relative_path not in artifact_paths:
        raise ValueError("particle_star_relative_path must be included in oracle_artifact_paths")
    dispatch_log_relative_path = _validate_oracle_artifact_paths(
        [dispatch_log_relative_path]
    )[0]
    if dispatch_log_relative_path not in artifact_paths:
        raise ValueError("dispatch_log_relative_path must be included in oracle_artifact_paths")
    dispatch_metadata_relative_path = relion_dispatch_metadata_relative_path(
        dispatch_log_relative_path
    )
    if dispatch_metadata_relative_path not in artifact_paths:
        raise ValueError("dispatch metadata sidecar must be included in oracle_artifact_paths")
    manifest_sha256 = _validate_sha256(
        oracle_manifest_sha256,
        name="oracle_manifest_sha256",
    )
    order_sha256 = _validate_sha256(
        particle_order_sha256,
        name="particle_order_sha256",
    )
    expected_oracle_id = relion_oracle_id(
        manifest_sha256=manifest_sha256,
        particle_order_sha256=order_sha256,
    )
    if str(oracle_id) != expected_oracle_id:
        raise ValueError("oracle_id is inconsistent with manifest and particle-order hashes")

    owners = np.full((iterations.size, n_particles), -1, dtype=np.int64)
    for row_idx, relion_iteration in enumerate(iterations):
        selected = np.flatnonzero(chunk_iters == relion_iteration)
        if selected.size < 1:
            raise ValueError(f"RELION iteration {int(relion_iteration)} has no dispatch chunks")
        for chunk_idx in selected:
            positions = slice(int(first[chunk_idx]), int(last[chunk_idx]) + 1)
            if np.any(owners[row_idx, positions] >= 0):
                raise ValueError(
                    f"RELION iteration {int(relion_iteration)} has overlapping dispatch chunks"
                )
            owners[row_idx, positions] = int(ranks[chunk_idx]) - 1
        missing = np.flatnonzero(owners[row_idx] < 0)
        if missing.size:
            raise ValueError(
                f"RELION iteration {int(relion_iteration)} leaves {missing.size} particles undispatched"
            )

    return RelionDispatchSchedule(
        relion_iterations=iterations.copy(),
        owner_by_sorted_position=owners,
        original_particle_id_by_sorted_position=originals.copy(),
        n_followers=n_followers,
        pool_size=pool_size,
        random_seed=int(random_seed),
        oracle_id=expected_oracle_id,
        oracle_manifest_sha256=manifest_sha256,
        oracle_artifact_paths=artifact_paths,
        particle_order_sha256=order_sha256,
        particle_star_relative_path=particle_star_relative_path,
        dispatch_log_relative_path=dispatch_log_relative_path,
        source=str(source),
    )


def validate_relion_follower_scale_start(*, n_followers: int, init_relion_iteration: int) -> None:
    """Reject continuations whose follower-local scale state is unavailable.

    A RELION model STAR serializes only follower 1's scale vector.  Therefore
    it cannot initialize the complete MPI follower state after iteration 0.
    Supporting that case requires an explicit checkpoint containing every
    follower vector; no such input is currently accepted by the refinement
    API.
    """

    if int(n_followers) > 0 and int(init_relion_iteration) > 0:
        raise ValueError(
            "RELION follower-local scale emulation cannot cold-start at "
            f"init_relion_iteration={int(init_relion_iteration)} from a leader-serialized STAR; "
            "start from iteration 0 (full follower-scale checkpoint input is not implemented)"
        )


def make_relion_follower_scale_state(
    *,
    n_followers: int,
    group_counts,
    n_optics_groups: int,
    initial_group_scales=None,
) -> RelionFollowerScaleState:
    """Construct a validated follower state, initially equal on all followers."""

    n_followers = int(n_followers)
    if n_followers < 1:
        raise ValueError(f"n_followers must be positive, got {n_followers}")
    counts = np.asarray(group_counts, dtype=np.float64).reshape(-1)
    if counts.size < 1 or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("group_counts must be a non-empty finite non-negative vector")
    n_optics_groups = int(n_optics_groups)
    if n_optics_groups < 0 or n_optics_groups > counts.size:
        raise ValueError(
            f"n_optics_groups must be in [0, {counts.size}], got {n_optics_groups}",
        )
    if initial_group_scales is None:
        initial = np.ones(counts.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_group_scales, dtype=np.float64).reshape(-1)
        if initial.shape != counts.shape:
            raise ValueError(
                f"initial_group_scales has shape {initial.shape}, expected {counts.shape}",
            )
        if np.any(~np.isfinite(initial)) or np.any(initial <= 0.0):
            raise ValueError("initial_group_scales must be finite and positive")
    return RelionFollowerScaleState(
        scales=np.repeat(initial[None, :], n_followers, axis=0),
        group_counts=counts.copy(),
        n_optics_groups=n_optics_groups,
    )


def relion_class3d_sorted_particle_ids(
    *,
    particle_ids_by_image,
    optics_group_ids_by_image,
    random_seed: int,
) -> np.ndarray:
    """Return RELION's exact one-time Class3D shuffled internal-particle order.

    RELION shuffles the initial ``sorted_idx`` once at the original first
    expectation using ``std::mt19937(random_seed + 1)``, then stable-sorts by
    optics group. Its function-static guard retains that order on later
    iterations and diagnostic continuations. The binding supplies the
    source-faithful ``std::shuffle``.
    """

    particle_ids = np.asarray(particle_ids_by_image, dtype=np.int64).reshape(-1)
    optics_by_image = np.asarray(optics_group_ids_by_image, dtype=np.int64).reshape(-1)
    if optics_by_image.shape != particle_ids.shape:
        raise ValueError("optics_group_ids_by_image must match particle_ids_by_image")
    n_particles = int(particle_ids.size)
    if np.unique(particle_ids).size != n_particles or (
        n_particles and (int(np.min(particle_ids)) < 0 or int(np.max(particle_ids)) >= n_particles)
    ):
        raise ValueError("particle_ids_by_image must be a permutation of [0, n_particles)")
    optics_by_particle = np.empty(n_particles, dtype=np.int64)
    optics_by_particle[particle_ids] = optics_by_image

    from recovar.relion_bind import _relion_bind_core as bind

    if not hasattr(bind, "vdam_randomise_particles_order"):
        raise RuntimeError("RELION binding lacks vdam_randomise_particles_order; rebuild recovar/relion_bind")
    sorted_particle_ids = np.asarray(
        bind.vdam_randomise_particles_order(
            n_particles,
            int(random_seed) + 1,
        ),
        dtype=np.int64,
    )
    sorted_particle_ids = sorted_particle_ids[
        np.argsort(optics_by_particle[sorted_particle_ids], kind="stable")
    ]
    return sorted_particle_ids


def relion_class3d_follower_owners_from_schedule(
    schedule: RelionDispatchSchedule,
    *,
    particle_ids_by_image,
    optics_group_ids_by_image,
    random_seed: int,
    relion_iteration: int,
) -> np.ndarray:
    """Map one captured dynamic dispatch row into RECOVAR image order."""

    if int(random_seed) != int(schedule.random_seed):
        raise ValueError(
            "RELION dispatch schedule random_seed does not match the refinement "
            f"({schedule.random_seed} != {int(random_seed)})"
        )
    matches = np.flatnonzero(schedule.relion_iterations == int(relion_iteration))
    if matches.size != 1:
        raise ValueError(
            f"RELION dispatch schedule does not contain iteration {int(relion_iteration)}"
        )
    particle_ids = np.asarray(particle_ids_by_image, dtype=np.int64).reshape(-1)
    row = np.asarray(schedule.owner_by_sorted_position[int(matches[0])], dtype=np.int64)
    sorted_particle_ids = np.asarray(
        schedule.original_particle_id_by_sorted_position[int(matches[0])], dtype=np.int64
    )
    if row.shape != sorted_particle_ids.shape or row.shape != particle_ids.shape:
        raise ValueError(
            "RELION dispatch schedule particle count does not match the authoritative data STAR "
            f"({row.size} != {sorted_particle_ids.size})"
        )
    owner_by_particle = np.empty(row.size, dtype=np.int64)
    owner_by_particle[sorted_particle_ids] = row
    return owner_by_particle[particle_ids]


def relion_worker_group_ids(group_ids, follower_owners, *, n_groups: int) -> np.ndarray:
    """Encode ``(follower, physical group)`` as one accumulator group axis."""

    groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    owners = np.asarray(follower_owners, dtype=np.int64).reshape(-1)
    if groups.shape != owners.shape:
        raise ValueError("group_ids and follower_owners must have the same shape")
    n_groups = int(n_groups)
    if n_groups < 1:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    if groups.size and (int(np.min(groups)) < 0 or int(np.max(groups)) >= n_groups):
        raise ValueError("group_ids are out of bounds")
    if owners.size and int(np.min(owners)) < 0:
        raise ValueError("follower_owners must be non-negative")
    return owners * n_groups + groups


def select_relion_follower_scales(
    state: RelionFollowerScaleState,
    *,
    group_ids,
    follower_owners,
) -> np.ndarray:
    """Select the current runtime scale for each particle."""

    groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    owners = np.asarray(follower_owners, dtype=np.int64).reshape(-1)
    if groups.shape != owners.shape:
        raise ValueError("group_ids and follower_owners must have the same shape")
    if groups.size and (int(np.min(groups)) < 0 or int(np.max(groups)) >= state.n_groups):
        raise ValueError("group_ids are out of bounds")
    if owners.size and (int(np.min(owners)) < 0 or int(np.max(owners)) >= state.n_followers):
        raise ValueError("follower_owners are out of bounds")
    return np.asarray(state.scales[owners, groups], dtype=np.float64)


def update_relion_follower_scales(
    state: RelionFollowerScaleState,
    *,
    wsum_signal_product,
    wsum_reference_power,
    relion_firstiter_cc_this_iter: bool = False,
    scale_relaxation_mu: float = 0.0,
) -> RelionFollowerScaleState:
    """Apply RELION's follower-local scale update and segmented-pack boundary."""

    if relion_firstiter_cc_this_iter:
        return state
    mu = float(scale_relaxation_mu)
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"scale_relaxation_mu must be in [0, 1], got {mu}")
    expected_shape = (state.n_followers, state.n_groups)
    xa = np.asarray(wsum_signal_product, dtype=np.float64).reshape(expected_shape).copy()
    aa = np.asarray(wsum_reference_power, dtype=np.float64).reshape(expected_shape).copy()
    if np.any(~np.isfinite(xa)) or np.any(~np.isfinite(aa)) or np.any(aa < 0.0):
        raise ValueError("scale XA/AA must be finite and AA must be non-negative")

    combined_count = min(int(state.n_optics_groups), state.n_groups)
    if combined_count:
        xa_combined = np.sum(xa[:, :combined_count], axis=0)
        aa_combined = np.sum(aa[:, :combined_count], axis=0)
        xa[:, :combined_count] = xa_combined[None, :]
        aa[:, :combined_count] = aa_combined[None, :]

    target = np.ones_like(xa)
    np.divide(xa, aa, out=target, where=aa > 0.0)
    updated = mu * np.asarray(state.scales, dtype=np.float64) + (1.0 - mu) * target
    for follower in range(state.n_followers):
        median = float(np.sort(updated[follower])[state.n_groups // 2])
        if np.isfinite(median) and median > 0.0:
            updated[follower] = np.clip(updated[follower], median / 5.0, 5.0 * median)
        count_sum = float(np.sum(state.group_counts))
        if count_sum > 0.0:
            avg_scale = float(np.sum(state.group_counts * updated[follower]) / count_sum)
            if avg_scale > 0.0 and np.isfinite(avg_scale):
                updated[follower] /= avg_scale

    return RelionFollowerScaleState(
        scales=updated,
        group_counts=np.asarray(state.group_counts, dtype=np.float64).copy(),
        n_optics_groups=int(state.n_optics_groups),
    )


def relion_rank1_serialized_scales(state: RelionFollowerScaleState) -> np.ndarray:
    """Return a copy of the scale vector written by RELION MPI follower 1."""

    return np.asarray(state.scales[0], dtype=np.float64).copy()
