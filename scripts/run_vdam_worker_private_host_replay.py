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

SCHEMA = "recovar.vdam_worker_private_host_replay.v6"
_CALL_RE = re.compile(r"-call-(\d+)-input\.npz$")
_PANEL_MAGIC = 0x5644414D42504631
_PANEL_HEADER_WORDS = 10


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


def _part_to_original_indices(data_star: Path, part_count: int) -> list[int]:
    import starfile

    document = starfile.read(data_star)
    particles = document["particles"] if isinstance(document, dict) else document
    if "rlnImageName" not in particles:
        raise ValueError("RELION data STAR has no rlnImageName column")
    if part_count > len(particles):
        raise ValueError("native capture has more parts than the RELION data STAR")
    result = [
        _original_index_from_image_name(particles["rlnImageName"].iloc[part_id])
        for part_id in range(part_count)
    ]
    if len(set(result)) != len(result):
        raise ValueError("RELION data STAR contains duplicate captured identities")
    return result


def _load_native_topology(
    topology_path: Path,
    data_star: Path,
    *,
    iteration: int,
) -> dict[int, tuple[int, int]]:
    """Map original stack index to native ``(worker, launch_count)``."""
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

    if set(by_part) != set(range(len(by_part))):
        raise ValueError(
            "native topology part IDs must be contiguous from zero"
        )
    original_indices = _part_to_original_indices(data_star, len(by_part))
    result: dict[int, tuple[int, int]] = {}
    for part_id, original_index in enumerate(original_indices):
        if original_index in result:
            raise ValueError(f"duplicate original stack index {original_index}")
        result[original_index] = by_part[part_id]
    return result


def _load_native_launch_order(
    topology_path: Path,
    data_star: Path,
    *,
    iteration: int,
) -> list[int]:
    part_ids = []
    for line_number, line in enumerate(topology_path.read_text().splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(
                f"topology line {line_number} has {len(fields)} fields, expected 6"
            )
        row_iteration, part_id = int(fields[0]), int(fields[1])
        if row_iteration == iteration:
            part_ids.append(part_id)
    if set(part_ids) != set(range(len(part_ids))) or len(set(part_ids)) != len(
        part_ids
    ):
        raise ValueError("native launch order must cover each contiguous part once")
    original_indices = _part_to_original_indices(data_star, len(part_ids))
    return [original_indices[part_id] for part_id in part_ids]


def _read_native_panel(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=_PANEL_HEADER_WORDS)
        if header.size != _PANEL_HEADER_WORDS:
            raise ValueError(f"truncated native BPref panel header: {path}")
        if int(header[0]) != _PANEL_MAGIC or int(header[1]) != 1:
            raise ValueError(f"unsupported native BPref panel schema: {path}")
        xfloat_bytes = int(header[9])
        if xfloat_bytes not in (4, 8):
            raise ValueError(f"unsupported XFLOAT size {xfloat_bytes}: {path}")
        dtype = np.dtype("<f4" if xfloat_bytes == 4 else "<f8")
        orientation_count = int(header[7])
        translation_count = int(header[8])
        payload = np.fromfile(stream, dtype=dtype)
    euler_values = orientation_count * 9
    expected = euler_values + orientation_count * translation_count
    if payload.size != expected:
        raise ValueError(
            f"native BPref panel {path} has {payload.size} values, expected {expected}"
        )
    return {
        "path": str(path.resolve()),
        "iteration": int(header[2]),
        "part_id": int(header[3]),
        "image_id": int(header[4]),
        "class_id": int(header[5]),
        "iproj_offset": int(header[6]),
        "orientation_count": orientation_count,
        "translation_count": translation_count,
        "eulers": np.asarray(
            payload[:euler_values].reshape(orientation_count, 9), dtype=np.float32
        ),
        "weights": np.asarray(
            payload[euler_values:].reshape(orientation_count, translation_count),
            dtype=np.float32,
        ),
    }


def _load_native_panels(
    panel_directory: Path,
    data_star: Path,
    *,
    iteration: int,
) -> dict[int, dict[str, object]]:
    by_part: dict[int, dict[str, object]] = {}
    for path in sorted(panel_directory.glob(f"it{iteration}_part*_img*_class*.bin")):
        panel = _read_native_panel(path)
        if panel["iteration"] != iteration:
            raise ValueError(f"panel iteration mismatch: {path}")
        if panel["image_id"] != 0 or panel["class_id"] != 0:
            raise ValueError("K=1 native panel replay requires image 0 and class 0")
        part_id = int(panel["part_id"])
        if part_id in by_part:
            raise ValueError(f"duplicate native BPref panel for part {part_id}")
        by_part[part_id] = panel
    if not by_part:
        raise FileNotFoundError(f"no iteration {iteration} panels in {panel_directory}")
    if set(by_part) != set(range(len(by_part))):
        raise ValueError("native BPref panel part IDs must be contiguous from zero")
    original_indices = _part_to_original_indices(data_star, len(by_part))
    return {
        original_indices[part_id]: by_part[part_id] for part_id in range(len(by_part))
    }


def _apply_native_topology(
    source: dict[str, np.ndarray],
    topology: dict[int, tuple[int, int]],
    panels: dict[int, dict[str, object]] | None = None,
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
        if count > replay_order.shape[1] and panels is None:
            raise ValueError(
                f"native launch count {count} exceeds sealed replay width "
                f"{replay_order.shape[1]} for particle trace {trace_id}"
            )
        workers[row] = worker
        counts[row] = count
    result["worker_lane_ids"] = workers
    result["rotation_replay_counts"] = counts
    if panels is None:
        return result

    posterior = np.asarray(source["posterior_over_weight_norm"], dtype=np.float32)
    eulers = np.asarray(source["projector_eulers"], dtype=np.float32)
    compact = np.asarray(source["compact_rotations"], dtype=np.float32)
    n_particles, sealed_width = replay_order.shape
    translation_count = _scalar(source, "translation_count")
    if posterior.shape != (n_particles, sealed_width, translation_count):
        raise ValueError("sealed posterior has inconsistent shape")
    if eulers.shape != (n_particles, sealed_width, 9):
        raise ValueError("sealed Euler panel has inconsistent shape")
    if compact.shape != (n_particles, sealed_width, 6):
        raise ValueError("sealed compact rotation panel has inconsistent shape")
    identity = np.arange(sealed_width, dtype=np.int32)
    if not np.all(replay_order == identity[None, :]):
        raise ValueError("native panel remapping requires identity sealed replay order")

    replay_width = max(sealed_width, int(np.max(counts)))
    remapped_posterior = np.zeros(
        (n_particles, replay_width, translation_count), dtype=np.float32
    )
    remapped_eulers = np.zeros((n_particles, replay_width, 9), dtype=np.float32)
    remapped_compact = np.zeros((n_particles, replay_width, 6), dtype=np.float32)
    for row, trace_id_value in enumerate(trace_ids):
        trace_id = int(trace_id_value)
        if trace_id not in panels:
            raise ValueError(f"no native BPref panel for particle trace {trace_id}")
        panel = panels[trace_id]
        native_count = int(panel["orientation_count"])
        if native_count != int(counts[row]):
            raise ValueError(
                f"native topology/panel count mismatch for particle trace {trace_id}"
            )
        if int(panel["translation_count"]) != translation_count:
            raise ValueError(
                f"native translation count mismatch for particle trace {trace_id}"
            )
        native_eulers = np.asarray(panel["eulers"], dtype=np.float32)
        if native_eulers.shape != (native_count, 9):
            raise ValueError(f"invalid native Euler shape for particle trace {trace_id}")
        remapped_eulers[row, :native_count] = native_eulers

        native_rows_by_euler: dict[bytes, list[int]] = {}
        for native_row in range(native_count):
            native_rows_by_euler.setdefault(
                native_eulers[native_row].tobytes(), []
            ).append(native_row)
        active_rows = np.flatnonzero(np.any(posterior[row] != 0.0, axis=1))
        used_native_rows: set[int] = set()
        for candidate_row_value in active_rows:
            candidate_row = int(candidate_row_value)
            key = eulers[row, candidate_row].tobytes()
            choices = native_rows_by_euler.get(key, [])
            native_row = next(
                (choice for choice in choices if choice not in used_native_rows), None
            )
            if native_row is None:
                raise ValueError(
                    "active candidate Euler has no exact native row for particle "
                    f"trace {trace_id}, candidate row {candidate_row}"
                )
            used_native_rows.add(native_row)
            remapped_posterior[row, native_row] = posterior[row, candidate_row]
            remapped_compact[row, native_row] = compact[row, candidate_row]

    result["rotation_count"] = np.asarray(source["rotation_count"]).dtype.type(
        replay_width
    )
    result["posterior_over_weight_norm"] = remapped_posterior
    result["projector_eulers"] = remapped_eulers
    result["compact_rotations"] = remapped_compact
    result["rotation_replay_order"] = np.tile(
        np.arange(replay_width, dtype=np.int32), (n_particles, 1)
    )
    return result


def _merge_callback_sources(
    sources: list[dict[str, np.ndarray]],
    launch_order: list[int],
) -> dict[str, np.ndarray]:
    if not sources:
        raise ValueError("cannot merge an empty callback list")
    particle_keys = (
        "images",
        "ctf",
        "minvsigma2",
        "reconstruction_group_ids",
        "worker_lane_ids",
        "particle_trace_ids",
        "rotation_replay_counts",
        "particle_start_offsets_ns",
    )
    rotation_keys = (
        "posterior_over_weight_norm",
        "projector_eulers",
        "compact_rotations",
    )
    varying_keys = set(particle_keys) | set(rotation_keys) | {
        "rotation_replay_order",
        "n_particles",
        "rotation_count",
        "data_real_volume",
        "data_imag_volume",
        "weight_volume",
    }
    first = sources[0]
    for source in sources[1:]:
        if set(source) != set(first):
            raise ValueError("sealed callback schemas differ")
        for name in set(first) - varying_keys:
            if not np.array_equal(source[name], first[name]):
                raise ValueError(f"sealed static callback field changes: {name}")

    width = max(_scalar(source, "rotation_count") for source in sources)
    merged = {name: np.asarray(value) for name, value in first.items()}
    for name in particle_keys:
        merged[name] = np.concatenate([np.asarray(source[name]) for source in sources])
    for name in rotation_keys:
        rows = []
        for source in sources:
            value = np.asarray(source[name])
            padding = [(0, 0), (0, width - value.shape[1])]
            padding.extend((0, 0) for _axis in range(value.ndim - 2))
            rows.append(np.pad(value, padding, mode="constant"))
        merged[name] = np.concatenate(rows)
    n_particles = sum(_scalar(source, "n_particles") for source in sources)
    merged["rotation_replay_order"] = np.tile(
        np.arange(width, dtype=np.int32), (n_particles, 1)
    )
    merged["n_particles"] = np.asarray(first["n_particles"]).dtype.type(n_particles)
    merged["rotation_count"] = np.asarray(first["rotation_count"]).dtype.type(width)
    merged["particle_start_offsets_ns"] = np.zeros(n_particles, dtype=np.int32)
    merged["parallel_worker_replay"] = np.int32(1)

    trace_ids = np.asarray(merged["particle_trace_ids"], dtype=np.int64)
    if len(set(trace_ids.tolist())) != n_particles:
        raise ValueError("merged callback particle traces are not unique")
    if set(trace_ids.tolist()) != set(launch_order):
        raise ValueError("merged callback identities differ from native launch order")
    row_by_trace = {int(trace_id): row for row, trace_id in enumerate(trace_ids)}
    order = np.asarray([row_by_trace[trace_id] for trace_id in launch_order])
    for name in particle_keys + rotation_keys + ("rotation_replay_order",):
        merged[name] = np.ascontiguousarray(merged[name][order])
    return merged


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
    native_panel_directory: Path | None = None,
    merge_callbacks: bool = False,
    worker_private_accumulators: bool = True,
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
    if native_panel_directory is not None and topology_data_star is None:
        raise ValueError("native panels require topology and topology data STAR")
    topology = None
    panels = None
    launch_order = None
    if topology_path is not None and topology_data_star is not None:
        topology = _load_native_topology(
            topology_path, topology_data_star, iteration=iteration
        )
        if merge_callbacks:
            launch_order = _load_native_launch_order(
                topology_path, topology_data_star, iteration=iteration
            )
        if native_panel_directory is not None:
            panels = _load_native_panels(
                native_panel_directory, topology_data_star, iteration=iteration
            )
            if set(panels) != set(topology):
                raise ValueError("native topology and panel identities differ")
    first = _load_bundle(input_paths[0])
    group_count = _scalar(first, "reconstruction_group_count")
    source_accumulator = np.asarray(first["data_real_volume"])
    if source_accumulator.ndim != 2 or source_accumulator.shape[0] != group_count:
        raise ValueError("sealed source accumulator has invalid shape")
    state_shape = (
        (group_count, worker_count, source_accumulator.shape[1])
        if worker_private_accumulators
        else (group_count, source_accumulator.shape[1])
    )
    state_real = np.zeros(state_shape, dtype=np.float32)
    state_imag = np.zeros(state_shape, dtype=np.float32)
    state_weight = np.zeros(state_shape, dtype=np.float32)
    callbacks = []
    ori_size = _scalar(first, "physical_image_size")
    recon_volume_shape = tuple(
        _scalar(first, name) for name in ("volume_n0", "volume_n1", "volume_n2")
    )
    prepared_sources = []
    for input_path in input_paths:
        source = _load_bundle(input_path)
        if topology is not None:
            source = _apply_native_topology(source, topology, panels)
        prepared_sources.append(source)
    if merge_callbacks:
        if launch_order is None:
            raise ValueError("merged callback replay requires native topology")
        callback_sources = [
            (
                "merged_native_launch_order",
                _merge_callback_sources(prepared_sources, launch_order),
            )
        ]
    else:
        callback_sources = list(
            zip((str(path) for path in input_paths), prepared_sources)
        )

    with tempfile.TemporaryDirectory(
        prefix="vdam-worker-private-", dir=output_directory
    ) as temporary_text:
        temporary = Path(temporary_text)
        for call_index, (source_label, source) in enumerate(callback_sources):
            if _scalar(source, "reconstruction_group_count") != group_count:
                raise ValueError("reconstruction group count changes across callbacks")
            if worker_private_accumulators:
                before_real = _serial_reduce(state_real, tuple(range(worker_count)))
                before_imag = _serial_reduce(state_imag, tuple(range(worker_count)))
                before_weight = _serial_reduce(
                    state_weight, tuple(range(worker_count))
                )
            else:
                before_real = state_real
                before_imag = state_imag
                before_weight = state_weight
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
            if worker_private_accumulators:
                packed = _private_bundle(
                    source,
                    state_real,
                    state_imag,
                    state_weight,
                    worker_count=worker_count,
                )
            else:
                packed = {name: np.asarray(value) for name, value in source.items()}
                packed["data_real_volume"] = state_real
                packed["data_imag_volume"] = state_imag
                packed["weight_volume"] = state_weight
                packed["parallel_worker_replay"] = np.int32(1)
            packed_input = temporary / f"call-{call_index:04d}-input.npz"
            packed_output = temporary / f"call-{call_index:04d}-output.npz"
            np.savez(packed_input, **packed)
            callback = run_vdam_exact_native_host_replay.run_replay(
                packed_input, packed_output, library_path
            )
            with np.load(packed_output, allow_pickle=False) as result:
                state_real = np.asarray(result["data_real_volume"]).reshape(state_shape)
                state_imag = np.asarray(result["data_imag_volume"]).reshape(state_shape)
                state_weight = np.asarray(result["weight_volume"]).reshape(state_shape)
            callbacks.append(
                {
                    "call_index": call_index,
                    "source": source_label,
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
    if worker_private_accumulators:
        reduced_real = _serial_reduce(state_real, order)
        reduced_imag = _serial_reduce(state_imag, order)
        reduced_weight = _serial_reduce(state_weight, order)
    else:
        reduced_real = state_real
        reduced_imag = state_imag
        reduced_weight = state_weight
    accumulator_path = output_directory / "worker_private_accumulators.npz"
    accumulator_bundle = dict(
        reduced_real=reduced_real,
        reduced_imag=reduced_imag,
        reduced_weight=reduced_weight,
        reduction_order=np.asarray(order, dtype=np.int32),
    )
    if worker_private_accumulators:
        accumulator_bundle.update(
            private_real=state_real,
            private_imag=state_imag,
            private_weight=state_weight,
        )
    else:
        accumulator_bundle.update(
            shared_real=state_real,
            shared_imag=state_imag,
            shared_weight=state_weight,
        )
    np.savez(accumulator_path, **accumulator_bundle)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "hypothesis": "_with_".join(
            filter(
                None,
                (
                    (
                        "persistent_worker_private_accumulators"
                        if worker_private_accumulators
                        else "shared_accumulators"
                    ),
                    "native_topology" if topology is not None else "",
                    "native_panels" if panels is not None else "",
                    "merged_callbacks" if merge_callbacks else "",
                ),
            )
        ),
        "native_physical_grid_replayed": topology is not None,
        "native_worker_owners_replayed": topology is not None,
        "native_topology_path": str(topology_path) if topology_path else None,
        "native_topology_data_star": (
            str(topology_data_star) if topology_data_star else None
        ),
        "native_euler_panels_replayed": panels is not None,
        "native_panel_directory": (
            str(native_panel_directory) if native_panel_directory else None
        ),
        "callbacks_merged_in_native_launch_order": merge_callbacks,
        "worker_private_accumulators": worker_private_accumulators,
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
    parser.add_argument("--native-panel-directory", type=Path)
    parser.add_argument("--merge-callbacks", action="store_true")
    parser.add_argument("--shared-accumulators", action="store_true")
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
        native_panel_directory=args.native_panel_directory,
        merge_callbacks=args.merge_callbacks,
        worker_private_accumulators=not args.shared_accumulators,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
