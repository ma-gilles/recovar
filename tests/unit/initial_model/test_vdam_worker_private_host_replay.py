from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import run_vdam_worker_private_host_replay as worker_private

ROOT = Path(__file__).resolve().parents[3]


def _bundle() -> dict[str, np.ndarray]:
    return {
        "n_particles": np.int64(4),
        "reconstruction_group_count": np.int32(2),
        "reconstruction_group_ids": np.asarray([0, 1, 0, 1], dtype=np.int32),
        "worker_lane_ids": np.asarray([2, 0, 1, 2], dtype=np.int32),
        "data_real_volume": np.zeros((2, 5), dtype=np.float32),
        "data_imag_volume": np.zeros((2, 5), dtype=np.float32),
        "weight_volume": np.zeros((2, 5), dtype=np.float32),
        "parallel_worker_replay": np.int32(0),
    }


def test_private_bundle_maps_each_group_worker_pair_to_one_accumulator() -> None:
    source = _bundle()
    private = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    packed = worker_private._private_bundle(
        source, private, private + 1, private + 2, worker_count=3
    )

    assert packed["reconstruction_group_count"] == 6
    assert packed["reconstruction_group_ids"].tolist() == [2, 3, 1, 5]
    assert packed["parallel_worker_replay"] == 1
    assert np.array_equal(packed["data_real_volume"], private.reshape(6, 5))
    assert np.array_equal(source["reconstruction_group_ids"], [0, 1, 0, 1])


def test_private_bundle_rejects_worker_outside_topology() -> None:
    source = _bundle()
    source["worker_lane_ids"][0] = 3
    private = np.zeros((2, 3, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="worker lane"):
        worker_private._private_bundle(
            source, private, private, private, worker_count=3
        )


def test_serial_reduce_uses_explicit_float32_worker_order() -> None:
    private = np.asarray([[[1.0e8], [1.0], [-1.0e8]]], dtype=np.float32)
    forward = worker_private._serial_reduce(private, (0, 1, 2))
    reverse = worker_private._serial_reduce(private, (2, 1, 0))

    assert forward.dtype == np.float32
    assert forward.item() == 0.0
    assert reverse.item() == 0.0
    with pytest.raises(ValueError, match="permutation"):
        worker_private._serial_reduce(private, (0, 0, 2))


def test_ordered_inputs_is_fail_closed_on_missing_callback(tmp_path) -> None:
    np.savez(tmp_path / "pid-1-call-0000-input.npz", value=np.int32(0))
    np.savez(tmp_path / "pid-1-call-0002-input.npz", value=np.int32(2))
    with pytest.raises(ValueError, match="not contiguous"):
        worker_private._ordered_inputs(tmp_path)


def test_relion_continuation_can_capture_complete_bpref_topology() -> None:
    source = (ROOT / "scripts/run_vdam_relion_continuation_capture.sbatch").read_text()
    assert "VDAM_RELION_CONT_BPREF_TOPOLOGY_PATH" in source
    assert "VDAM_RELION_CONT_BPREF_TOPOLOGY_EXPECTED_ROWS" in source
    assert "RELION_ACC_DUMP_BPREF_TOPOLOGY" in source
    assert "VDAM_RELION_CONT_MSTEP_DUMP_DIR" in source
    assert "VDAM_RELION_CONT_TARGET_GPU_UUID" in source
    assert "VDAM_TARGET_GPU_MISS" in source


def test_apply_native_topology_replaces_owner_and_physical_count() -> None:
    source = _bundle()
    source["particle_trace_ids"] = np.asarray([10, 11, 12, 13], dtype=np.int32)
    source["rotation_replay_order"] = np.tile(
        np.arange(8, dtype=np.int32), (4, 1)
    )
    source["rotation_replay_counts"] = np.full(4, 8, dtype=np.int32)
    topology = {10: (1, 4), 11: (0, 7), 12: (2, 6), 13: (1, 5)}

    result = worker_private._apply_native_topology(source, topology)

    assert result["worker_lane_ids"].tolist() == [1, 0, 2, 1]
    assert result["rotation_replay_counts"].tolist() == [4, 7, 6, 5]
    assert source["worker_lane_ids"].tolist() == [2, 0, 1, 2]


def test_apply_native_topology_rejects_count_beyond_sealed_grid() -> None:
    source = _bundle()
    source["particle_trace_ids"] = np.asarray([10, 11, 12, 13], dtype=np.int32)
    source["rotation_replay_order"] = np.tile(
        np.arange(8, dtype=np.int32), (4, 1)
    )
    topology = {10: (1, 9), 11: (0, 7), 12: (2, 6), 13: (1, 5)}
    with pytest.raises(ValueError, match="exceeds sealed replay width"):
        worker_private._apply_native_topology(source, topology)


def test_load_native_topology_maps_part_ids_through_star_identity(
    tmp_path, monkeypatch
) -> None:
    import pandas as pd
    import starfile

    topology = tmp_path / "topology.tsv"
    topology.write_text(
        "1\t1\t0\t3\t7\t0\n"
        "2\t0\t0\t6\t4\t0\n"
        "1\t0\t0\t2\t6\t0\n"
        "2\t1\t0\t7\t5\t0\n"
    )
    particles = pd.DataFrame({"rlnImageName": ["0011@x.mrcs", "0021@x.mrcs"]})
    monkeypatch.setattr(starfile, "read", lambda _path: {"particles": particles})

    result = worker_private._load_native_topology(
        topology, tmp_path / "data.star", iteration=1
    )

    assert result == {10: (2, 6), 20: (3, 7)}


def test_native_panel_remap_places_active_rows_at_exact_native_eulers() -> None:
    candidate_eulers = np.arange(36, dtype=np.float32).reshape(4, 9)
    native_eulers = np.vstack(
        [
            candidate_eulers[2],
            np.full(9, 101, dtype=np.float32),
            np.full(9, 102, dtype=np.float32),
            np.full(9, 103, dtype=np.float32),
            candidate_eulers[0],
        ]
    )
    posterior = np.zeros((1, 4, 2), dtype=np.float32)
    posterior[0, 0] = [0.25, 0.5]
    posterior[0, 2] = [0.75, 0.0]
    source = {
        "particle_trace_ids": np.asarray([10], dtype=np.int32),
        "worker_lane_ids": np.asarray([0], dtype=np.int32),
        "rotation_replay_order": np.arange(4, dtype=np.int32)[None, :],
        "rotation_replay_counts": np.asarray([4], dtype=np.int32),
        "rotation_count": np.int64(4),
        "translation_count": np.int64(2),
        "posterior_over_weight_norm": posterior,
        "projector_eulers": candidate_eulers[None, :, :],
        "compact_rotations": np.arange(24, dtype=np.float32).reshape(1, 4, 6),
    }
    panel = {
        "orientation_count": 5,
        "translation_count": 2,
        "eulers": native_eulers,
        "weights": np.zeros((5, 2), dtype=np.float32),
    }

    result = worker_private._apply_native_topology(
        source, {10: (3, 5)}, {10: panel}
    )

    assert int(result["rotation_count"]) == 5
    assert result["worker_lane_ids"].tolist() == [3]
    assert result["rotation_replay_counts"].tolist() == [5]
    assert np.array_equal(result["projector_eulers"][0], native_eulers)
    assert np.array_equal(result["posterior_over_weight_norm"][0, 0], [0.75, 0.0])
    assert np.array_equal(result["posterior_over_weight_norm"][0, 4], [0.25, 0.5])
    assert np.count_nonzero(result["posterior_over_weight_norm"][0, 1:4]) == 0


def test_read_native_panel_is_fail_closed_and_preserves_float32(tmp_path) -> None:
    path = tmp_path / "it1_part0_img0_class0.bin"
    header = (
        worker_private._PANEL_MAGIC,
        1,
        1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
    )
    eulers = np.arange(18, dtype="<f4")
    weights = np.arange(6, dtype="<f4") + 20
    path.write_bytes(struct.pack("<10Q", *header) + eulers.tobytes() + weights.tobytes())

    panel = worker_private._read_native_panel(path)

    assert panel["orientation_count"] == 2
    assert panel["translation_count"] == 3
    assert np.array_equal(panel["eulers"], eulers.reshape(2, 9))
    assert np.array_equal(panel["weights"], weights.reshape(2, 3))
