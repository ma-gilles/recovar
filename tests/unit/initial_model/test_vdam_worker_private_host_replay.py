from __future__ import annotations

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
