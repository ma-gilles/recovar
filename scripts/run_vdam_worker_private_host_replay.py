#!/usr/bin/env python3
"""Replay sealed VDAM callbacks with persistent worker-private accumulators.

The production RECOVAR replay has one accumulator per reconstruction group.
RELION instead gives each host worker a private backprojector and reduces those
backprojectors after the particle loop.  This diagnostic preserves the sealed
RECOVAR operands and concurrent worker streams, but remaps every
``(group, worker)`` pair to its own persistent accumulator.  By default native
physical grid cardinality is left unchanged so the two hypotheses remain
separable.  An optional RELION topology capture can replace both worker
ownership and per-particle launch cardinality for the combined causal arm.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
)
from recovar.em.initial_model.layout import relion_bpref_frame_scales
from scripts import analyze_vdam_mstep_boundary, run_vdam_exact_native_host_replay

SCHEMA = "recovar.vdam_worker_private_host_replay.v3"
_CALL_RE = re.compile(r"-call-(\d+)-input\.npz$")


def _scalar(bundle: dict[str, np.ndarray], name: str) -> int:
    value = np.asarray(bundle[name])
    if value.size != 1:
        raise ValueError(f"{name} must be scalar")
    return int(value.reshape(()))


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _ordered_inputs(input_directory: Path) -> list[Path]:
    rows = []
    for path in Path(input_directory).glob("*-input.npz"):
        match = _CALL_RE.search(path.name)
        if match is not None:
            rows.append((int(match.group(1)), path.resolve()))
    if not rows:
        raise FileNotFoundError(f"no sealed callback inputs in {input_directory}")
    rows.sort()
    call_ids = [call_id for call_id, _path in rows]
    expected = list(range(call_ids[0], call_ids[0] + len(call_ids)))
    if call_ids != expected:
        raise ValueError(f"callback IDs are not contiguous: {call_ids}")
    return [path for _call_id, path in rows]


def _original_index_from_image_name(value: object) -> int:
    token = str(value).split("@", maxsplit=1)[0]
    try:
        stack_index = int(token)
    except ValueError as error:
        raise ValueError(f"invalid RELION rlnImageName {value!r}") from error
    if stack_index <= 0:
        raise ValueError(f"RELION stack index must be one-based: {value!r}")
    return stack_index - 1


def _load_native_topology(
    topology_path: Path,
    data_star: Path,
    *,
    iteration: int,
) -> dict[int, tuple[int, int]]:
    """Map original stack index to native ``(worker, launch_count)``."""
    import starfile

    by_part: dict[int, tuple[int, int]] = {}
    for line_number, line in enumerate(topology_path.read_text().splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(
                f"topology line {line_number} has {len(fields)} fields, expected 6"
            )
        row_iteration, part_id, image_id, worker, orientation_count, _offset = (
            int(field) for field in fields
        )
        if row_iteration != iteration:
            continue
        if image_id != 0:
            raise ValueError(
                f"topology line {line_number} has image_id {image_id}, expected 0"
            )
        if part_id in by_part:
            raise ValueError(f"duplicate native topology part_id {part_id}")
        if worker < 0 or orientation_count <= 0:
            raise ValueError(f"invalid native topology line {line_number}: {line!r}")
        by_part[part_id] = (worker, orientation_count)

    document = starfile.read(data_star)
    particles = document["particles"] if isinstance(document, dict) else document
    if "rlnImageName" not in particles:
        raise ValueError("RELION data STAR has no rlnImageName column")
    if set(by_part) != set(range(len(by_part))):
        raise ValueError(
            "native topology part IDs must be contiguous from zero"
        )
    if len(by_part) > len(particles):
        raise ValueError("native topology has more parts than the RELION data STAR")
    result: dict[int, tuple[int, int]] = {}
    for part_id in range(len(by_part)):
        image_name = particles["rlnImageName"].iloc[part_id]
        original_index = _original_index_from_image_name(image_name)
        if original_index in result:
            raise ValueError(f"duplicate original stack index {original_index}")
        result[original_index] = by_part[part_id]
    return result


def _apply_native_topology(
    source: dict[str, np.ndarray],
    topology: dict[int, tuple[int, int]],
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in source.items()}
    trace_ids = np.asarray(source["particle_trace_ids"], dtype=np.int64)
    replay_order = np.asarray(source["rotation_replay_order"])
    if replay_order.ndim != 2 or replay_order.shape[0] != trace_ids.size:
        raise ValueError("sealed rotation replay order has inconsistent shape")
    workers = np.empty(trace_ids.size, dtype=np.int32)
    counts = np.empty(trace_ids.size, dtype=np.int32)
    for row, trace_id_value in enumerate(trace_ids):
        trace_id = int(trace_id_value)
        if trace_id not in topology:
            raise ValueError(f"no native topology for particle trace {trace_id}")
        worker, count = topology[trace_id]
        if count > replay_order.shape[1]:
            raise ValueError(
                f"native launch count {count} exceeds sealed replay width "
                f"{replay_order.shape[1]} for particle trace {trace_id}"
            )
        workers[row] = worker
        counts[row] = count
    result["worker_lane_ids"] = workers
    result["rotation_replay_counts"] = counts
    return result


def _private_bundle(
    source: dict[str, np.ndarray],
    private_real: np.ndarray,
    private_imag: np.ndarray,
    private_weight: np.ndarray,
    *,
    worker_count: int,
) -> dict[str, np.ndarray]:
    group_count = _scalar(source, "reconstruction_group_count")
    n_particles = _scalar(source, "n_particles")
    worker_lanes = np.asarray(source["worker_lane_ids"], dtype=np.int32)
    groups = np.asarray(source["reconstruction_group_ids"], dtype=np.int32)
    if worker_lanes.shape != (n_particles,) or groups.shape != (n_particles,):
        raise ValueError("particle worker/group arrays have inconsistent shapes")
    if np.any(worker_lanes < 0) or np.any(worker_lanes >= worker_count):
        raise ValueError("worker lane is outside the requested private topology")
    if np.any(groups < 0) or np.any(groups >= group_count):
        raise ValueError("reconstruction group is outside the sealed topology")
    expected_shape = (group_count, worker_count, private_real.shape[-1])
    for name, value in (
        ("private_real", private_real),
        ("private_imag", private_imag),
        ("private_weight", private_weight),
    ):
        if value.shape != expected_shape or value.dtype != np.float32:
            raise ValueError(
                f"{name} has shape/dtype {value.shape}/{value.dtype}, "
                f"expected {expected_shape}/float32"
            )

    result = {name: np.asarray(value) for name, value in source.items()}
    result["reconstruction_group_ids"] = (
        groups * np.int32(worker_count) + worker_lanes
    ).astype(np.int32, copy=False)
    result["reconstruction_group_count"] = np.int32(group_count * worker_count)
    result["data_real_volume"] = np.ascontiguousarray(
        private_real.reshape(group_count * worker_count, -1)
    )
    result["data_imag_volume"] = np.ascontiguousarray(
        private_imag.reshape(group_count * worker_count, -1)
    )
    result["weight_volume"] = np.ascontiguousarray(
        private_weight.reshape(group_count * worker_count, -1)
    )
    result["parallel_worker_replay"] = np.int32(1)
    return result


def _serial_reduce(private: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    if private.ndim != 3 or private.dtype != np.float32:
        raise ValueError("private accumulator must be float32 (group, worker, voxel)")
    if tuple(sorted(order)) != tuple(range(private.shape[1])):
        raise ValueError("reduction order must be a permutation of all worker lanes")
    reduced = np.zeros((private.shape[0], private.shape[2]), dtype=np.float32)
    for worker in order:
        np.add(reduced, private[:, worker], out=reduced)
    return reduced


def _native_metrics(
    native_directory: Path,
    recovar_directory: Path,
    reduced_real: np.ndarray,
    reduced_imag: np.ndarray,
    reduced_weight: np.ndarray,
    *,
    iteration: int,
    ori_size: int,
    recon_volume_shape: tuple[int, int, int],
) -> dict:
    token = f"it{iteration}"
    native_names = (
        f"pipe_{token}_c0_bp_data_pre_reweight.bin",
        f"pipe_{token}_c0_bp_weight.bin",
        f"pipe_{token}_c0_bp_data_h_pre_reweight.bin",
        f"pipe_{token}_c0_bp_weight_h.bin",
    )
    if reduced_real.shape[0] != 2:
        raise ValueError("native comparison currently requires two pseudo-halfsets")
    native_data = [
        analyze_vdam_mstep_boundary._read_relion_array(
            native_directory / native_names[index], complex_values=True
        )
        for index in (0, 2)
    ]
    native_weight = [
        analyze_vdam_mstep_boundary._read_relion_array(
            native_directory / native_names[index], complex_values=False
        )
        for index in (1, 3)
    ]
    production_data = [
        np.load(recovar_directory / f"accum_h{group}_data.npy", allow_pickle=False)
        for group in range(2)
    ]
    production_weight = [
        np.load(recovar_directory / f"accum_h{group}_weight.npy", allow_pickle=False)
        for group in range(2)
    ]
    rows = {}
    data_scale, weight_scale = relion_bpref_frame_scales(ori_size)
    for group in range(2):
        enforced_data, enforced_weight = enforce_half_volume_x0(
            reduced_real[group] + 1j * reduced_imag[group],
            reduced_weight[group],
            recon_volume_shape,
            logger=logging.getLogger(__name__),
            label=f"Worker-private half {group}",
            force_host=True,
        )
        candidate_data = (
            data_scale * np.asarray(enforced_data)
        ).reshape(native_data[group].shape)
        candidate_weight = (weight_scale * np.asarray(enforced_weight)).reshape(
            native_weight[group].shape
        )
        private_data_metric = analyze_vdam_mstep_boundary._metric(
            candidate_data, native_data[group]
        )
        private_weight_metric = analyze_vdam_mstep_boundary._metric(
            candidate_weight, native_weight[group]
        )
        production_data_metric = analyze_vdam_mstep_boundary._metric(
            production_data[group], native_data[group]
        )
        production_weight_metric = analyze_vdam_mstep_boundary._metric(
            production_weight[group], native_weight[group]
        )
        rows[f"half{group}"] = {
            "worker_private_data": private_data_metric,
            "worker_private_weight": private_weight_metric,
            "production_shared_data": production_data_metric,
            "production_shared_weight": production_weight_metric,
            "data_relative_l2_ratio_private_over_shared": (
                private_data_metric["relative_l2"]
                / production_data_metric["relative_l2"]
            ),
            "weight_relative_l2_ratio_private_over_shared": (
                private_weight_metric["relative_l2"]
                / production_weight_metric["relative_l2"]
            ),
        }
    return rows


def replay(
    input_directory: Path,
    output_directory: Path,
    library_path: Path,
    native_directory: Path,
    recovar_directory: Path,
    *,
    worker_count: int,
    iteration: int,
    topology_path: Path | None = None,
    topology_data_star: Path | None = None,
) -> dict:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    for name in (
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR",
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_PARTICLE_ID",
    ):
        if os.environ.get(name, "").strip():
            raise RuntimeError(f"worker-private replay cannot mix with {name}")
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    input_paths = _ordered_inputs(input_directory)
    if (topology_path is None) != (topology_data_star is None):
        raise ValueError("topology path and topology data STAR must be supplied together")
    topology = None
    if topology_path is not None and topology_data_star is not None:
        topology = _load_native_topology(
            topology_path, topology_data_star, iteration=iteration
        )
    first = _load_bundle(input_paths[0])
    group_count = _scalar(first, "reconstruction_group_count")
    source_accumulator = np.asarray(first["data_real_volume"])
    if source_accumulator.ndim != 2 or source_accumulator.shape[0] != group_count:
        raise ValueError("sealed source accumulator has invalid shape")
    private_shape = (group_count, worker_count, source_accumulator.shape[1])
    private_real = np.zeros(private_shape, dtype=np.float32)
    private_imag = np.zeros(private_shape, dtype=np.float32)
    private_weight = np.zeros(private_shape, dtype=np.float32)
    callbacks = []
    ori_size = _scalar(first, "physical_image_size")
    recon_volume_shape = tuple(
        _scalar(first, name) for name in ("volume_n0", "volume_n1", "volume_n2")
    )

    with tempfile.TemporaryDirectory(
        prefix="vdam-worker-private-", dir=output_directory
    ) as temporary_text:
        temporary = Path(temporary_text)
        for call_index, input_path in enumerate(input_paths):
            source = _load_bundle(input_path)
            if topology is not None:
                source = _apply_native_topology(source, topology)
            if _scalar(source, "reconstruction_group_count") != group_count:
                raise ValueError("reconstruction group count changes across callbacks")
            before_real = _serial_reduce(
                private_real, tuple(range(worker_count))
            )
            before_imag = _serial_reduce(
                private_imag, tuple(range(worker_count))
            )
            before_weight = _serial_reduce(
                private_weight, tuple(range(worker_count))
            )
            sealed_real = np.asarray(source["data_real_volume"], dtype=np.float32)
            sealed_imag = np.asarray(source["data_imag_volume"], dtype=np.float32)
            sealed_weight = np.asarray(source["weight_volume"], dtype=np.float32)
            start_state_comparison = {
                "data_real": analyze_vdam_mstep_boundary._metric(
                    before_real, sealed_real
                ),
                "data_imag": analyze_vdam_mstep_boundary._metric(
                    before_imag, sealed_imag
                ),
                "weight": analyze_vdam_mstep_boundary._metric(
                    before_weight, sealed_weight
                ),
            }
            packed = _private_bundle(
                source,
                private_real,
                private_imag,
                private_weight,
                worker_count=worker_count,
            )
            packed_input = temporary / f"call-{call_index:04d}-input.npz"
            packed_output = temporary / f"call-{call_index:04d}-output.npz"
            np.savez(packed_input, **packed)
            callback = run_vdam_exact_native_host_replay.run_replay(
                packed_input, packed_output, library_path
            )
            with np.load(packed_output, allow_pickle=False) as result:
                shape = (group_count, worker_count, source_accumulator.shape[1])
                private_real = np.asarray(result["data_real_volume"]).reshape(shape)
                private_imag = np.asarray(result["data_imag_volume"]).reshape(shape)
                private_weight = np.asarray(result["weight_volume"]).reshape(shape)
            callbacks.append(
                {
                    "call_index": call_index,
                    "source": str(input_path),
                    "n_particles": callback["n_particles"],
                    "rotation_count": callback["rotation_count"],
                    "pixel_count": callback["pixel_count"],
                    "native_worker_lanes": (
                        sorted(set(np.asarray(source["worker_lane_ids"]).tolist()))
                        if topology is not None
                        else None
                    ),
                    "native_rotation_count_range": (
                        [
                            int(np.min(source["rotation_replay_counts"])),
                            int(np.max(source["rotation_replay_counts"])),
                        ]
                        if topology is not None
                        else None
                    ),
                    "start_state_private_vs_production_shared": (
                        start_state_comparison
                    ),
                }
            )

    order = tuple(range(worker_count))
    reduced_real = _serial_reduce(private_real, order)
    reduced_imag = _serial_reduce(private_imag, order)
    reduced_weight = _serial_reduce(private_weight, order)
    accumulator_path = output_directory / "worker_private_accumulators.npz"
    np.savez(
        accumulator_path,
        private_real=private_real,
        private_imag=private_imag,
        private_weight=private_weight,
        reduced_real=reduced_real,
        reduced_imag=reduced_imag,
        reduced_weight=reduced_weight,
        reduction_order=np.asarray(order, dtype=np.int32),
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "hypothesis": (
            "persistent_worker_private_accumulators_with_native_topology"
            if topology is not None
            else "persistent_worker_private_accumulators"
        ),
        "native_physical_grid_replayed": topology is not None,
        "native_worker_owners_replayed": topology is not None,
        "native_topology_path": str(topology_path) if topology_path else None,
        "native_topology_data_star": (
            str(topology_data_star) if topology_data_star else None
        ),
        "worker_count": worker_count,
        "reconstruction_group_count": group_count,
        "callback_count": len(callbacks),
        "callbacks": callbacks,
        "accumulators": str(accumulator_path),
        "native_comparison": _native_metrics(
            native_directory,
            recovar_directory,
            reduced_real,
            reduced_imag,
            reduced_weight,
            iteration=iteration,
            ori_size=ori_size,
            recon_volume_shape=recon_volume_shape,
        ),
    }
    report_path = output_directory / "worker_private_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--topology-data-star", type=Path)
    args = parser.parse_args()
    report = replay(
        args.input_directory,
        args.output_directory,
        args.library,
        args.native_directory,
        args.recovar_directory,
        worker_count=args.worker_count,
        iteration=args.iteration,
        topology_path=args.topology,
        topology_data_star=args.topology_data_star,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
