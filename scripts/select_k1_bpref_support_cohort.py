#!/usr/bin/env python3
"""Select a bounded BPref operand cohort from a sealed particle-state audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<16s40Q")
HEADER_MAGIC = b"RLNBPREV1HEADER\0"
DEFAULT_SEED = "k1-bpref-support-cohort-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_relion_identities(directory: Path) -> dict[int, tuple[int, int]]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"RELION reference capture directory does not exist: {directory}")
    incomplete = sorted(directory.glob("*.tmp.*"))
    if incomplete:
        raise ValueError(f"incomplete RELION capture artifact remains: {incomplete[0]}")
    paths = sorted(directory.glob("*.bpre-v1.bin"))
    if not paths:
        raise ValueError(f"no sealed RELION capture artifacts in {directory}")

    identities: dict[int, tuple[int, int]] = {}
    part_ids: set[int] = set()
    for path in paths:
        with path.open("rb") as stream:
            raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"truncated RELION capture header: {path}")
        magic, *values = HEADER.unpack(raw)
        if magic != HEADER_MAGIC:
            raise ValueError(f"invalid RELION capture header magic: {path}")
        if values[0] != 1 or values[1] != HEADER.size:
            raise ValueError(f"unsupported RELION capture header schema: {path}")
        part_id = int(values[7])
        stack_index = int(values[8])
        mpi_rank = int(values[10])
        if stack_index in identities:
            raise ValueError(f"duplicate RELION stack identity: {stack_index}")
        if part_id in part_ids:
            raise ValueError(f"duplicate RELION part identity: {part_id}")
        identities[stack_index] = (part_id, mpi_rank)
        part_ids.add(part_id)
    return identities


def _load_stack_indices(image_names_path: Path) -> np.ndarray:
    image_names = np.load(image_names_path.expanduser().resolve(), allow_pickle=False)
    if image_names.ndim != 1:
        raise ValueError("image_names must be a one-dimensional array")
    stack_indices = np.empty(image_names.size, dtype=np.int64)
    for index, raw_identity in enumerate(image_names.astype(str).tolist()):
        stack_index, separator, _ = raw_identity.partition("@")
        if not separator or not stack_index.isdigit() or int(stack_index) <= 0:
            raise ValueError(f"invalid image identity at row {index}: {raw_identity!r}")
        stack_indices[index] = int(stack_index)
    if np.unique(stack_indices).size != stack_indices.size:
        raise ValueError("image_names contains duplicate stack identities")
    return stack_indices


def _hash_rank(*, seed: str, label: str, stack_index: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{stack_index}".encode()).digest()


def _take_hashed(
    rows: list[dict[str, int | str]],
    count: int,
    *,
    seed: str,
    label: str,
) -> list[dict[str, int | str]]:
    if len(rows) < count:
        raise ValueError(f"cohort stratum {label} has {len(rows)} rows, fewer than requested {count}")
    ranked = sorted(
        rows,
        key=lambda row: _hash_rank(seed=seed, label=label, stack_index=int(row["stack_index_one_based"])),
    )
    return ranked[:count]


def select_cohort(
    *,
    trajectory_path: Path,
    image_names_path: Path,
    reference_capture_dir: Path,
    iteration: int,
    mpi_rank: int,
    total_count: int,
    minus_two_count: int,
    minus_one_count: int,
    seed: str = DEFAULT_SEED,
) -> dict[str, object]:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    if mpi_rank < 0:
        raise ValueError("MPI rank must be nonnegative")
    if total_count <= 0 or minus_two_count < 0 or minus_one_count < 0:
        raise ValueError("cohort counts must be nonnegative and total_count must be positive")

    trajectory_path = trajectory_path.expanduser().resolve()
    image_names_path = image_names_path.expanduser().resolve()
    reference_capture_dir = reference_capture_dir.expanduser().resolve()
    support_key = f"it{iteration:03d}_support_delta"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        if "identity_row_index" not in trajectory or support_key not in trajectory:
            raise ValueError(f"trajectory is missing identity_row_index or {support_key}")
        identity_rows = np.asarray(trajectory["identity_row_index"], dtype=np.int64)
        support_delta_raw = np.asarray(trajectory[support_key], dtype=np.float64)
    if identity_rows.ndim != 1 or support_delta_raw.shape != identity_rows.shape:
        raise ValueError("trajectory identity/support arrays are inconsistent")
    if np.unique(identity_rows).size != identity_rows.size:
        raise ValueError("trajectory identity rows are duplicated")
    if not np.all(np.isfinite(support_delta_raw)):
        raise ValueError("trajectory support deltas must be finite")
    support_delta = support_delta_raw.astype(np.int64)
    if not np.array_equal(support_delta_raw, support_delta.astype(np.float64)):
        raise ValueError("trajectory support deltas must be integral")

    stack_indices = _load_stack_indices(image_names_path)
    if np.any(identity_rows < 0) or np.any(identity_rows >= stack_indices.size):
        raise ValueError("trajectory identity row is outside image_names")
    relion_identities = _load_relion_identities(reference_capture_dir)

    pools: dict[str, list[dict[str, int | str]]] = {
        "support_delta_le_minus_3": [],
        "support_delta_minus_2": [],
        "support_delta_minus_1": [],
        "exact_support_control": [],
    }
    for identity_row, delta in zip(identity_rows.tolist(), support_delta.tolist()):
        stack_index = int(stack_indices[identity_row])
        if stack_index != identity_row + 1:
            raise ValueError(
                "image_names row-to-stack mapping is not the required canonical i -> i+1 mapping"
            )
        try:
            part_id, observed_rank = relion_identities[stack_index]
        except KeyError as error:
            raise ValueError(f"stack identity {stack_index} is missing from RELION capture") from error
        if observed_rank != mpi_rank:
            continue
        if delta <= -3:
            label = "support_delta_le_minus_3"
        elif delta == -2:
            label = "support_delta_minus_2"
        elif delta == -1:
            label = "support_delta_minus_1"
        elif delta == 0:
            label = "exact_support_control"
        else:
            raise ValueError(f"unexpected positive support delta at identity row {identity_row}: {delta}")
        pools[label].append(
            {
                "group": label,
                "original_index_zero_based": int(identity_row),
                "stack_index_one_based": stack_index,
                "relion_part_id": part_id,
                "mpi_rank": observed_rank,
                "support_delta": int(delta),
            }
        )

    selected = sorted(
        pools["support_delta_le_minus_3"],
        key=lambda row: int(row["stack_index_one_based"]),
    )
    if len(selected) > total_count:
        raise ValueError("deepest support-loss stratum exceeds total cohort size")
    selected += _take_hashed(
        pools["support_delta_minus_2"],
        minus_two_count,
        seed=seed,
        label="support_delta_minus_2",
    )
    selected += _take_hashed(
        pools["support_delta_minus_1"],
        minus_one_count,
        seed=seed,
        label="support_delta_minus_1",
    )
    control_count = total_count - len(selected)
    if control_count < 0:
        raise ValueError("requested support-loss strata exceed total cohort size")
    selected += _take_hashed(
        pools["exact_support_control"],
        control_count,
        seed=seed,
        label="exact_support_control",
    )
    selected.sort(key=lambda row: int(row["stack_index_one_based"]))

    canonical_rows = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    group_counts = {
        label: sum(row["group"] == label for row in selected)
        for label in pools
    }
    return {
        "schema": "recovar-k1-bpref-support-cohort-v1",
        "iteration": iteration,
        "mpi_rank": mpi_rank,
        "selection_seed": seed,
        "selection_policy": {
            "all_support_delta_le_minus_3": True,
            "support_delta_minus_2_count": minus_two_count,
            "support_delta_minus_1_count": minus_one_count,
            "fill_exact_support_controls_to_total": total_count,
        },
        "source_pool_counts_on_rank": {label: len(rows) for label, rows in pools.items()},
        "selected_group_counts": group_counts,
        "selected_particle_count": len(selected),
        "cohort_rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "relion_part_ids_csv": ",".join(str(row["relion_part_id"]) for row in selected),
        "recovar_original_indices_csv": ",".join(
            str(row["original_index_zero_based"]) for row in selected
        ),
        "selected_stack_indices_one_based": [
            int(row["stack_index_one_based"]) for row in selected
        ],
        "rows": selected,
        "sources": {
            "trajectory_npz": str(trajectory_path),
            "trajectory_npz_sha256": _sha256_file(trajectory_path),
            "image_names_npy": str(image_names_path),
            "image_names_npy_sha256": _sha256_file(image_names_path),
            "reference_capture_dir": str(reference_capture_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--image-names-npy", type=Path, required=True)
    parser.add_argument("--reference-capture-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--mpi-rank", type=int, default=1)
    parser.add_argument("--total-count", type=int, default=64)
    parser.add_argument("--minus-two-count", type=int, default=16)
    parser.add_argument("--minus-one-count", type=int, default=24)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = select_cohort(
        trajectory_path=args.trajectory_npz,
        image_names_path=args.image_names_npy,
        reference_capture_dir=args.reference_capture_dir,
        iteration=args.iteration,
        mpi_rank=args.mpi_rank,
        total_count=args.total_count,
        minus_two_count=args.minus_two_count,
        minus_one_count=args.minus_one_count,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("schema", "selected_group_counts", "selected_particle_count", "cohort_rows_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
