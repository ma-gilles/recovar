"""Sealed VDAM worker/chronology replay and passive trace selection.

This opt-in diagnostic owner validates native capture schemas and maps stack IDs,
worker lanes, launch order and timing into the current image/rotation buckets.
It preserves captured ordering and the four process-local LRU caches; loading
this module does not import the local engine or run a replay. Kernel execution
and candidate block-map publication stay with the engine.
"""

from __future__ import annotations

import functools
import os

import jax.numpy as jnp
import numpy as np

RELION_VDAM_WORKER_SCHEDULE_ENV = "RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ"
RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV = "RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY"
RELION_VDAM_WORKER_REPLAY_ITER_ENV = "RECOVAR_RELION_VDAM_WORKER_REPLAY_ITER"
RELION_VDAM_WORKER_STREAM_COUNT = 8
RELION_VDAM_BLOCK_CHRONOLOGY_ENV = "RECOVAR_RELION_VDAM_BLOCK_CHRONOLOGY_NPZ"
VDAM_CANDIDATE_BLOCK_TRACE_ENV = "RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE"
VDAM_CANDIDATE_BLOCK_TRACE_ITER_ENV = "RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_ITER"
VDAM_WAVG_BPREF_HOST_GAP_TRACE_ENV = "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE"
VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR_ENV = (
    "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR"
)
VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR_ENV = (
    "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR"
)


@functools.lru_cache(maxsize=8)
def _load_relion_vdam_worker_schedule(path: str) -> np.ndarray:
    """Return a dense stack-index to worker-lane map from a sealed v2 trace."""

    with np.load(path, allow_pickle=False) as payload:
        schema_version = int(np.asarray(payload["schema_version"]).item())
        dataset_particles = int(np.asarray(payload["dataset_particles"]).item())
        n_threads = int(np.asarray(payload["n_threads"]).item())
        stack_indices = np.asarray(
            payload["stack_index_by_sorted_position"], dtype=np.int64
        )
        owners = np.asarray(payload["owner_by_sorted_position"], dtype=np.int64)
    if schema_version != 2:
        raise ValueError(
            "VDAM worker replay requires a v2 trace with stack-image join keys"
        )
    if dataset_particles <= 0 or n_threads != 8:
        raise ValueError("VDAM worker replay requires a positive dataset and eight workers")
    if stack_indices.ndim != 1 or owners.shape != stack_indices.shape:
        raise ValueError("VDAM worker replay arrays must be matching vectors")
    if (
        np.unique(stack_indices).size != stack_indices.size
        or np.any(stack_indices < 0)
        or np.any(stack_indices >= dataset_particles)
        or np.any(owners < 0)
        or np.any(owners >= n_threads)
    ):
        raise ValueError("VDAM worker replay schedule contains invalid stack IDs or owners")
    owner_by_stack_index = np.full(dataset_particles, -1, dtype=np.int32)
    owner_by_stack_index[stack_indices] = owners.astype(np.int32)
    owner_by_stack_index.setflags(write=False)
    return owner_by_stack_index


@functools.lru_cache(maxsize=8)
def _load_relion_vdam_block_start_orders(
    worker_schedule_path: str,
    chronology_path: str,
) -> tuple[int, int, tuple[np.ndarray | None, ...]]:
    """Join a sealed native block chronology to zero-based stack-image IDs."""

    with np.load(worker_schedule_path, allow_pickle=False) as schedule:
        schedule_schema = int(np.asarray(schedule["schema_version"]).item())
        schedule_iteration = int(np.asarray(schedule["iteration"]).item())
        dataset_particles = int(np.asarray(schedule["dataset_particles"]).item())
        internal_ids = np.asarray(
            schedule["internal_particle_id_by_sorted_position"], dtype=np.int64
        )
        stack_indices = np.asarray(
            schedule["stack_index_by_sorted_position"], dtype=np.int64
        )
    with np.load(chronology_path, allow_pickle=False) as chronology:
        chronology_schema = int(np.asarray(chronology["schema_version"]).item())
        chronology_iteration = int(np.asarray(chronology["iteration"]).item())
        chronology_particles = int(np.asarray(chronology["n_particles"]).item())
        records = np.asarray(chronology["records"])
    if schedule_schema != 2 or chronology_schema != 1:
        raise ValueError("captured block replay requires worker v2 and chronology v1")
    if schedule_iteration != chronology_iteration:
        raise ValueError("worker schedule and block chronology iterations differ")
    if dataset_particles <= 0 or chronology_particles != internal_ids.size:
        raise ValueError("captured block replay particle counts differ")
    if (
        internal_ids.ndim != 1
        or stack_indices.shape != internal_ids.shape
        or np.unique(internal_ids).size != internal_ids.size
        or np.unique(stack_indices).size != stack_indices.size
        or np.any(stack_indices < 0)
        or np.any(stack_indices >= dataset_particles)
    ):
        raise ValueError("captured block replay join keys are invalid")
    required_fields = {
        "particle_id",
        "block_start_globaltimer",
        "orientation_row",
        "image_count",
    }
    if records.ndim != 1 or records.dtype.names is None or not required_fields.issubset(
        records.dtype.names
    ):
        raise ValueError("captured block replay records have an invalid schema")
    if not np.array_equal(np.unique(records["particle_id"]), np.sort(internal_ids)):
        raise ValueError("captured block replay internal particle IDs differ")

    stack_by_internal = {
        int(internal_id): int(stack_index)
        for internal_id, stack_index in zip(internal_ids.tolist(), stack_indices.tolist())
    }
    orders: list[np.ndarray | None] = [None] * dataset_particles
    for internal_id in internal_ids.tolist():
        rows = records[records["particle_id"] == internal_id]
        image_counts = np.unique(rows["image_count"])
        if image_counts.size != 1:
            raise ValueError("captured block replay launch image counts differ")
        image_count = int(image_counts[0])
        if rows.size != image_count or not np.array_equal(
            np.sort(rows["orientation_row"]),
            np.arange(image_count, dtype=rows["orientation_row"].dtype),
        ):
            raise ValueError("captured block replay orientation rows are not a bijection")
        order = rows[
            np.lexsort((rows["orientation_row"], rows["block_start_globaltimer"]))
        ]["orientation_row"].astype(np.int32, copy=True)
        order.setflags(write=False)
        orders[stack_by_internal[int(internal_id)]] = order
    return schedule_iteration, dataset_particles, tuple(orders)


@functools.lru_cache(maxsize=8)
def _load_relion_vdam_particle_issue_ranks(
    worker_schedule_path: str,
    chronology_path: str,
) -> tuple[int, int, np.ndarray]:
    """Join native launch sequence to a dense stack-index issue-rank map."""

    with np.load(worker_schedule_path, allow_pickle=False) as schedule:
        schedule_schema = int(np.asarray(schedule["schema_version"]).item())
        schedule_iteration = int(np.asarray(schedule["iteration"]).item())
        dataset_particles = int(np.asarray(schedule["dataset_particles"]).item())
        n_threads = int(np.asarray(schedule["n_threads"]).item())
        internal_ids = np.asarray(
            schedule["internal_particle_id_by_sorted_position"], dtype=np.int64
        )
        stack_indices = np.asarray(
            schedule["stack_index_by_sorted_position"], dtype=np.int64
        )
        owners = np.asarray(
            schedule["owner_by_sorted_position"], dtype=np.int64
        )
    with np.load(chronology_path, allow_pickle=False) as chronology:
        chronology_schema = int(np.asarray(chronology["schema_version"]).item())
        chronology_iteration = int(np.asarray(chronology["iteration"]).item())
        chronology_particles = int(np.asarray(chronology["n_particles"]).item())
        chronology_threads = int(np.asarray(chronology["n_threads"]).item())
        records = np.asarray(chronology["records"])
    if schedule_schema != 2 or chronology_schema != 1:
        raise ValueError("captured particle issue replay requires worker v2 and chronology v1")
    if schedule_iteration != chronology_iteration:
        raise ValueError("worker schedule and particle chronology iterations differ")
    if (
        dataset_particles <= 0
        or chronology_particles != internal_ids.size
        or n_threads != RELION_VDAM_WORKER_STREAM_COUNT
        or chronology_threads != n_threads
    ):
        raise ValueError("captured particle issue replay topology differs")
    if (
        internal_ids.ndim != 1
        or stack_indices.shape != internal_ids.shape
        or owners.shape != internal_ids.shape
        or np.unique(internal_ids).size != internal_ids.size
        or np.unique(stack_indices).size != stack_indices.size
        or np.any(stack_indices < 0)
        or np.any(stack_indices >= dataset_particles)
        or np.any(owners < 0)
        or np.any(owners >= n_threads)
    ):
        raise ValueError("captured particle issue replay join keys are invalid")
    required_fields = {"launch_sequence", "particle_id", "worker_id"}
    if records.ndim != 1 or records.dtype.names is None or not required_fields.issubset(
        records.dtype.names
    ):
        raise ValueError("captured particle issue chronology has an invalid schema")
    if not np.array_equal(np.unique(records["particle_id"]), np.sort(internal_ids)):
        raise ValueError("captured particle issue internal particle IDs differ")

    stack_by_internal = {
        int(internal_id): int(stack_index)
        for internal_id, stack_index in zip(internal_ids.tolist(), stack_indices.tolist())
    }
    owner_by_internal = {
        int(internal_id): int(owner)
        for internal_id, owner in zip(internal_ids.tolist(), owners.tolist())
    }
    issue_rank_by_stack_index = np.full(dataset_particles, -1, dtype=np.int32)
    seen_launch_sequences: list[int] = []
    for internal_id in internal_ids.tolist():
        rows = records[records["particle_id"] == internal_id]
        launch_sequences = np.unique(rows["launch_sequence"])
        worker_ids = np.unique(rows["worker_id"])
        if launch_sequences.size != 1 or worker_ids.size != 1:
            raise ValueError("captured particle issue launch identity is not unique")
        if int(worker_ids[0]) != owner_by_internal[int(internal_id)]:
            raise ValueError("captured particle issue worker owner differs from schedule")
        launch_sequence = int(launch_sequences[0])
        seen_launch_sequences.append(launch_sequence)
        issue_rank_by_stack_index[stack_by_internal[int(internal_id)]] = launch_sequence
    if not np.array_equal(
        np.sort(np.asarray(seen_launch_sequences, dtype=np.int64)),
        np.arange(internal_ids.size, dtype=np.int64),
    ):
        raise ValueError("captured particle issue launch sequences are not a bijection")
    issue_rank_by_stack_index.setflags(write=False)
    return schedule_iteration, dataset_particles, issue_rank_by_stack_index


@functools.lru_cache(maxsize=8)
def _load_relion_vdam_particle_start_offsets_ns(
    worker_schedule_path: str,
    chronology_path: str,
) -> tuple[int, int, np.ndarray]:
    """Join native first-block timestamps to dense stack-index offsets."""

    trace_iteration, dataset_particles, issue_ranks = (
        _load_relion_vdam_particle_issue_ranks(
            worker_schedule_path,
            chronology_path,
        )
    )
    with np.load(worker_schedule_path, allow_pickle=False) as schedule:
        internal_ids = np.asarray(
            schedule["internal_particle_id_by_sorted_position"], dtype=np.int64
        )
        stack_indices = np.asarray(
            schedule["stack_index_by_sorted_position"], dtype=np.int64
        )
    with np.load(chronology_path, allow_pickle=False) as chronology:
        records = np.asarray(chronology["records"])
    if records.dtype.names is None or "block_start_globaltimer" not in records.dtype.names:
        raise ValueError("captured particle timing chronology has no block-start timestamps")

    starts_by_issue_rank = np.empty(internal_ids.size, dtype=np.uint64)
    stack_by_issue_rank = np.empty(internal_ids.size, dtype=np.int64)
    for internal_id, stack_index in zip(internal_ids.tolist(), stack_indices.tolist()):
        rows = records[records["particle_id"] == internal_id]
        starts = rows["block_start_globaltimer"]
        starts = starts[starts > 0]
        if starts.size == 0:
            raise ValueError("captured particle timing has no valid block start")
        issue_rank = int(issue_ranks[int(stack_index)])
        starts_by_issue_rank[issue_rank] = np.min(starts)
        stack_by_issue_rank[issue_rank] = int(stack_index)

    # NVIDIA's %globaltimer timestamps are nanoseconds.  The first native
    # launch is also the earliest launch in the sealed trace; retain the tiny
    # measured cross-stream inversions rather than changing their identities.
    first_start = int(starts_by_issue_rank[0])
    offsets_by_issue_rank = np.asarray(
        [int(start) - first_start for start in starts_by_issue_rank],
        dtype=np.int64,
    )
    if np.any(offsets_by_issue_rank < 0):
        raise ValueError("captured particle timing precedes the first launch")
    if np.any(offsets_by_issue_rank > np.iinfo(np.int32).max):
        raise ValueError("captured particle timing exceeds the int32 nanosecond range")
    offsets_by_stack_index = np.full(dataset_particles, -1, dtype=np.int32)
    offsets_by_stack_index[stack_by_issue_rank] = offsets_by_issue_rank.astype(np.int32)
    offsets_by_stack_index.setflags(write=False)
    return trace_iteration, dataset_particles, offsets_by_stack_index


def _relion_vdam_particle_issue_order_for_images(
    experiment_dataset,
    image_indices,
    *,
    debug_iteration: int | None,
) -> np.ndarray | None:
    """Return the current particles ordered by native global launch sequence."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    if topology not in {
        "captured_particle_issue",
        "captured_particle_issue_native_count",
        "captured_particle_issue_native_grid",
        "captured_particle_timing",
        "captured_particle_timing_native_count",
        "captured_particle_timing_native_grid",
    }:
        return None
    schedule_path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    chronology_path = os.environ.get(RELION_VDAM_BLOCK_CHRONOLOGY_ENV, "").strip()
    if not schedule_path or not chronology_path:
        raise ValueError(
            "captured particle issue replay requires sealed worker schedule and chronology NPZs"
        )
    trace_iteration, dataset_particles, issue_ranks = (
        _load_relion_vdam_particle_issue_ranks(schedule_path, chronology_path)
    )
    if debug_iteration is None or int(debug_iteration) != trace_iteration:
        return None
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if original_indices.shape != np.asarray(image_indices).shape:
        raise ValueError("captured particle issue image-index mapping returned an invalid shape")
    if np.any(original_indices < 0) or np.any(original_indices >= dataset_particles):
        raise ValueError("captured particle issue image index is outside the traced dataset")
    selected_ranks = issue_ranks[original_indices]
    if np.any(selected_ranks < 0):
        missing = original_indices[selected_ranks < 0]
        raise ValueError(
            "captured particle issue replay is missing selected stack indices "
            f"{missing[:8].tolist()}"
        )
    if np.unique(selected_ranks).size != selected_ranks.size:
        raise ValueError("captured particle issue ranks are not unique")
    return np.argsort(selected_ranks, kind="stable").astype(np.int32, copy=False)


def _relion_vdam_particle_start_offsets_for_images(
    experiment_dataset,
    image_indices,
    *,
    debug_iteration: int | None,
) -> np.ndarray | None:
    """Return native first-block offsets for the exact sealed iteration."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    if topology not in {
        "captured_particle_timing",
        "captured_particle_timing_native_count",
        "captured_particle_timing_native_grid",
    }:
        return None
    schedule_path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    chronology_path = os.environ.get(RELION_VDAM_BLOCK_CHRONOLOGY_ENV, "").strip()
    if not schedule_path or not chronology_path:
        raise ValueError(
            "captured particle timing requires sealed worker schedule and chronology NPZs"
        )
    trace_iteration, dataset_particles, offsets = (
        _load_relion_vdam_particle_start_offsets_ns(schedule_path, chronology_path)
    )
    if debug_iteration is None or int(debug_iteration) != trace_iteration:
        return None
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if original_indices.shape != np.asarray(image_indices).shape:
        raise ValueError("captured particle timing image-index mapping returned an invalid shape")
    if np.any(original_indices < 0) or np.any(original_indices >= dataset_particles):
        raise ValueError("captured particle timing image index is outside the traced dataset")
    selected_offsets = offsets[original_indices]
    if np.any(selected_offsets < 0):
        missing = original_indices[selected_offsets < 0]
        raise ValueError(
            "captured particle timing is missing selected stack indices "
            f"{missing[:8].tolist()}"
        )
    return selected_offsets.astype(np.int32, copy=False)


def _relion_vdam_block_start_orders_for_images(
    experiment_dataset,
    image_indices,
    *,
    rotation_count: int,
    valid_rotation_counts,
    debug_iteration: int | None,
) -> np.ndarray | None:
    """Resolve captured native block-start order for its exact traced iteration."""

    if not _relion_vdam_block_start_replay_active(debug_iteration=debug_iteration):
        return None
    schedule_path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    chronology_path = os.environ.get(RELION_VDAM_BLOCK_CHRONOLOGY_ENV, "").strip()
    _, dataset_particles, orders = _load_relion_vdam_block_start_orders(
        schedule_path,
        chronology_path,
    )
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if original_indices.shape != np.asarray(image_indices).shape:
        raise ValueError("captured block replay image-index mapping returned an invalid shape")
    if np.any(original_indices < 0) or np.any(original_indices >= dataset_particles):
        raise ValueError("captured block replay image index is outside the traced dataset")
    selected_orders = [orders[int(index)] for index in original_indices.tolist()]
    if any(order is None for order in selected_orders):
        missing = original_indices[
            np.asarray([order is None for order in selected_orders], dtype=bool)
        ]
        raise ValueError(
            "captured block replay is missing selected stack indices "
            f"{missing[:8].tolist()}"
        )
    valid_rotation_counts = np.asarray(valid_rotation_counts, dtype=np.int64)
    if valid_rotation_counts.shape != original_indices.shape:
        raise ValueError("captured block replay valid-row counts have an invalid shape")
    native_counts_array = np.asarray(
        [order.size for order in selected_orders],
        dtype=np.int64,
    )
    if np.any(valid_rotation_counts <= 0) or np.any(
        native_counts_array < valid_rotation_counts
    ) or np.any(native_counts_array - valid_rotation_counts >= 8):
        raise ValueError(
            "captured block replay cannot prove a native-grid prefix: "
            f"native={np.unique(native_counts_array).tolist()} "
            f"valid={np.unique(valid_rotation_counts).tolist()}"
        )
    if any(order.size > rotation_count for order in selected_orders):
        native_counts = sorted({int(order.size) for order in selected_orders})
        raise ValueError(
            "captured block replay native rotation count exceeds the current bucket: "
            f"native={native_counts} candidate={rotation_count}"
        )
    expanded_orders = []
    for order in selected_orders:
        # RECOVAR's static bucket can be larger than RELION's per-particle
        # eight-row-padded launch. Rows beyond the native launch carry zero
        # posterior; append them after the complete measured native order.
        if order.size < rotation_count:
            order = np.concatenate(
                (
                    order,
                    np.arange(order.size, rotation_count, dtype=np.int32),
                )
            )
        expanded_orders.append(order)
    return np.stack(expanded_orders, axis=0).astype(np.int32, copy=False)


def _relion_vdam_block_start_replay_active(*, debug_iteration: int | None) -> bool:
    """Return whether this call is the exact iteration represented by the seal."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    if topology not in {
        "captured_block_start",
        "captured_block_grid",
        "captured_native_grid",
        "captured_native_count",
        "captured_native_trace_shape",
        "captured_native_grid_trace_shape",
        "materialized_native_grid_trace_shape",
        "captured_particle_issue_native_count",
        "captured_particle_timing_native_count",
        "captured_particle_issue_native_grid",
        "captured_particle_timing_native_grid",
    }:
        return False
    schedule_path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    chronology_path = os.environ.get(RELION_VDAM_BLOCK_CHRONOLOGY_ENV, "").strip()
    if not schedule_path or not chronology_path:
        raise ValueError(
            "captured block replay requires sealed worker schedule and block chronology NPZs"
        )
    trace_iteration, _, _ = _load_relion_vdam_block_start_orders(
        schedule_path,
        chronology_path,
    )
    return debug_iteration is not None and int(debug_iteration) == trace_iteration


def _relion_vdam_worker_lanes_for_images(
    experiment_dataset,
    image_indices,
    *,
    debug_iteration: int | None = None,
):
    """Resolve optional native worker owners into the current physical bucket order."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    replay_iteration_raw = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_ITER_ENV,
        "",
    ).strip()
    if replay_iteration_raw and (topology == "round_robin" or path):
        try:
            replay_iteration = int(replay_iteration_raw)
        except ValueError as exc:
            raise ValueError(
                "VDAM worker replay iteration must be a positive integer"
            ) from exc
        if replay_iteration <= 0:
            raise ValueError(
                "VDAM worker replay iteration must be a positive integer"
            )
        if debug_iteration is None or int(debug_iteration) != replay_iteration:
            return None
    if topology == "round_robin":
        if path:
            raise ValueError(
                "round-robin VDAM worker replay cannot also use a captured schedule"
            )
        image_indices_array = np.asarray(image_indices)
        return (
            np.arange(image_indices_array.size, dtype=np.int32)
            .reshape(image_indices_array.shape)
            % RELION_VDAM_WORKER_STREAM_COUNT
        )
    if not path:
        return None
    owner_by_stack_index = _load_relion_vdam_worker_schedule(path)
    if topology in {
        "captured_particle_issue",
        "captured_particle_issue_native_count",
        "captured_particle_issue_native_grid",
        "captured_particle_timing",
        "captured_particle_timing_native_count",
        "captured_particle_timing_native_grid",
        "captured_block_start",
        "captured_block_grid",
        "captured_native_grid",
        "captured_native_count",
        "captured_native_trace_shape",
        "captured_native_grid_trace_shape",
        "materialized_native_grid_trace_shape",
    }:
        if topology in {
            "captured_particle_issue",
            "captured_particle_issue_native_count",
            "captured_particle_issue_native_grid",
            "captured_particle_timing",
            "captured_particle_timing_native_count",
            "captured_particle_timing_native_grid",
        }:
            if (
                _relion_vdam_particle_issue_order_for_images(
                    experiment_dataset,
                    image_indices,
                    debug_iteration=debug_iteration,
                )
                is None
            ):
                return None
        elif not _relion_vdam_block_start_replay_active(
            debug_iteration=debug_iteration
        ):
            return None
    if topology in {
        "single_rotation",
        "single_rotation_f64",
        "single_rotation_reverse",
        "single_rotation_sm132",
    }:
        # This diagnostic intentionally discards captured owners and serializes
        # every particle and orientation block.  The v2 trace still gets fully
        # schema-validated above, but later VDAM iterations may select particles
        # that were not present in the iteration-1 trace.
        return np.zeros(np.asarray(image_indices).shape, dtype=np.int32)
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if original_indices.shape != np.asarray(image_indices).shape:
        raise ValueError("VDAM worker replay image-index mapping returned an invalid shape")
    if np.any(original_indices < 0) or np.any(original_indices >= owner_by_stack_index.size):
        raise ValueError("VDAM worker replay image index is outside the traced dataset")
    owners = owner_by_stack_index[original_indices]
    if np.any(owners < 0):
        missing = original_indices[owners < 0]
        raise ValueError(
            "VDAM worker replay is missing selected stack indices "
            f"{missing[:8].tolist()}"
        )
    if topology == "single":
        owners = np.zeros_like(owners)
    elif topology not in {
        "captured",
        "captured_particle_issue",
        "captured_particle_issue_native_count",
        "captured_particle_issue_native_grid",
        "captured_particle_timing",
        "captured_particle_timing_native_count",
        "captured_particle_timing_native_grid",
        "captured_block_start",
        "captured_block_grid",
        "captured_native_grid",
        "captured_native_count",
        "captured_native_trace_shape",
        "captured_native_grid_trace_shape",
        "materialized_native_grid_trace_shape",
    }:
        raise ValueError(
            "VDAM worker replay topology must be 'captured', 'single', or "
            "'single_rotation', 'single_rotation_f64', 'single_rotation_reverse', "
            "'single_rotation_sm132', 'captured_particle_issue', 'captured_block_start', "
            "'captured_particle_timing', "
            "'captured_particle_issue_native_count', "
            "'captured_particle_issue_native_grid', "
            "'captured_particle_timing_native_count', "
            "'captured_particle_timing_native_grid', "
            "'captured_block_grid', 'captured_native_grid', or "
            "'captured_native_count', 'captured_native_trace_shape', or "
            "'captured_native_grid_trace_shape', or "
            "'materialized_native_grid_trace_shape'"
        )
    return owners.astype(np.int32, copy=False)


def _relion_vdam_native_grid_counts_for_images(
    experiment_dataset,
    image_indices,
    *,
    rotation_count: int,
    valid_rotation_counts,
    debug_iteration: int | None,
) -> np.ndarray | None:
    """Return sealed native per-particle grid sizes for the exact-grid replay."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    if topology not in {
        "captured_native_grid",
        "captured_native_count",
        "captured_native_trace_shape",
        "captured_native_grid_trace_shape",
        "materialized_native_grid_trace_shape",
        "captured_particle_issue_native_count",
        "captured_particle_timing_native_count",
        "captured_particle_issue_native_grid",
        "captured_particle_timing_native_grid",
    }:
        return None
    orders = _relion_vdam_block_start_orders_for_images(
        experiment_dataset,
        image_indices,
        rotation_count=rotation_count,
        valid_rotation_counts=valid_rotation_counts,
        debug_iteration=debug_iteration,
    )
    if orders is None:
        return None
    schedule_path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    chronology_path = os.environ.get(RELION_VDAM_BLOCK_CHRONOLOGY_ENV, "").strip()
    _, dataset_particles, captured_orders = _load_relion_vdam_block_start_orders(
        schedule_path,
        chronology_path,
    )
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if np.any(original_indices < 0) or np.any(original_indices >= dataset_particles):
        raise ValueError("native-grid replay image index is outside the traced dataset")
    counts = np.asarray(
        [captured_orders[int(index)].size for index in original_indices.tolist()],
        dtype=np.int32,
    )
    if counts.shape != np.asarray(image_indices).shape:
        raise ValueError("native-grid replay counts have an invalid shape")
    if np.any(counts <= 0) or np.any(counts > rotation_count):
        raise ValueError("native-grid replay count is outside the candidate bucket")
    return counts


def _relion_vdam_identity_native_grid_replay() -> bool:
    """Return whether native grid counts should keep identity physical rows."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return topology in {
        "captured_native_count",
        "captured_native_trace_shape",
        "captured_particle_issue_native_count",
        "captured_particle_timing_native_count",
    }


def _relion_vdam_materialized_native_grid_replay(
    *, debug_iteration: int | None
) -> bool:
    """Materialize captured logical rows before the native identity-grid launch."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return topology == "materialized_native_grid_trace_shape" and (
        _relion_vdam_block_start_replay_active(debug_iteration=debug_iteration)
    )


def _relion_vdam_native_trace_shape_replay(
    *, debug_iteration: int | None
) -> bool:
    """Retain native trace instructions only at the sealed trace iteration."""

    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return topology in {
        "captured_native_trace_shape",
        "captured_native_grid_trace_shape",
        "materialized_native_grid_trace_shape",
    } and (
        _relion_vdam_block_start_replay_active(debug_iteration=debug_iteration)
    )


def _materialize_relion_vdam_rotation_rows(value, replay_order):
    """Gather logical rotation rows into their captured physical block order."""

    value = jnp.asarray(value)
    replay_order = jnp.asarray(replay_order, dtype=jnp.int32)
    if value.ndim < 2 or value.shape[:2] != replay_order.shape:
        raise ValueError(
            "materialized VDAM operand must match the particle/rotation order axes"
        )
    index = replay_order.reshape(
        replay_order.shape + (1,) * (value.ndim - replay_order.ndim)
    )
    return jnp.take_along_axis(
        value,
        jnp.broadcast_to(index, value.shape),
        axis=1,
    )


def _relion_vdam_candidate_trace_active(*, debug_iteration: int | None) -> bool:
    """Gate passive candidate tracing to its explicitly sealed iteration."""

    if not os.environ.get(VDAM_CANDIDATE_BLOCK_TRACE_ENV, "").strip():
        return False
    iteration_text = os.environ.get(
        VDAM_CANDIDATE_BLOCK_TRACE_ITER_ENV,
        "",
    ).strip()
    if not iteration_text:
        raise ValueError("candidate block trace requires an explicit iteration")
    try:
        target_iteration = int(iteration_text)
    except ValueError as exc:
        raise ValueError("candidate block trace iteration must be an integer") from exc
    if target_iteration <= 0:
        raise ValueError("candidate block trace iteration must be positive")
    return debug_iteration is not None and int(debug_iteration) == target_iteration


def _relion_vdam_candidate_trace_ids_for_images(
    experiment_dataset,
    image_indices,
    *,
    debug_iteration: int | None,
):
    """Return stable stack IDs when a passive launch diagnostic needs them."""

    block_trace_active = _relion_vdam_candidate_trace_active(
        debug_iteration=debug_iteration
    )
    host_gap_trace_active = bool(
        os.environ.get(VDAM_WAVG_BPREF_HOST_GAP_TRACE_ENV, "").strip()
    )
    host_replay_capture_active = bool(
        os.environ.get(VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR_ENV, "").strip()
    )
    quiesced_prelaunch_capture_active = bool(
        os.environ.get(VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR_ENV, "").strip()
    )
    if not (
        block_trace_active
        or host_gap_trace_active
        or host_replay_capture_active
        or quiesced_prelaunch_capture_active
    ):
        return None
    original_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(image_indices),
        dtype=np.int64,
    )
    if original_indices.shape != np.asarray(image_indices).shape:
        raise ValueError("candidate block trace image-index mapping returned an invalid shape")
    if np.any(original_indices < 0) or np.any(
        original_indices > np.iinfo(np.int32).max
    ):
        raise ValueError("candidate block trace stack index is outside int32 range")
    return original_indices.astype(np.int32, copy=False)


def _relion_vdam_serial_rotation_replay() -> bool:
    """Return whether the opt-in worker replay also serializes rotations."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return bool(path) and topology in {
        "single_rotation",
        "single_rotation_f64",
        "single_rotation_reverse",
        "single_rotation_sm132",
    }


def _relion_vdam_captured_block_serial_replay() -> bool:
    """Return whether captured native block order uses one-block launches."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return bool(path) and topology == "captured_block_start"


def _relion_vdam_float64_accumulator_replay() -> bool:
    """Return whether the diagnostic uses binary64 accumulator storage."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return bool(path) and topology == "single_rotation_f64"


def _relion_vdam_reverse_rotation_replay() -> bool:
    """Return whether serialized orientation blocks run in reverse order."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return bool(path) and topology == "single_rotation_reverse"


def _relion_vdam_rotation_replay_stride() -> int:
    """Return the logical native-SM stride for serialized orientation blocks."""

    path = os.environ.get(RELION_VDAM_WORKER_SCHEDULE_ENV, "").strip()
    topology = os.environ.get(
        RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    ).strip().lower()
    return 132 if path and topology == "single_rotation_sm132" else 0
