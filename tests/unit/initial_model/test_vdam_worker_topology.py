from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine

pytestmark = pytest.mark.unit


def _clear_worker_replay(monkeypatch):
    monkeypatch.delenv(local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV, raising=False)
    monkeypatch.delenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        local_em_engine.RELION_VDAM_BLOCK_CHRONOLOGY_ENV,
        raising=False,
    )
    local_em_engine._load_relion_vdam_worker_schedule.cache_clear()
    local_em_engine._load_relion_vdam_particle_issue_ranks.cache_clear()


class _IndexDataset:
    def __init__(self, original_indices):
        self.original_indices = np.asarray(original_indices, dtype=np.int64)

    def original_image_indices_from_local(self, image_indices):
        return self.original_indices[np.asarray(image_indices)]


def _write_particle_issue_seals(tmp_path, *, launch_sequences=(2, 0, 1)):
    schedule_path = tmp_path / "worker_schedule.npz"
    chronology_path = tmp_path / "block_chronology.npz"
    internal_ids = np.asarray([30, 10, 20], dtype=np.int64)
    stack_indices = np.asarray([4, 1, 7], dtype=np.int64)
    owners = np.asarray([2, 0, 1], dtype=np.int64)
    np.savez(
        schedule_path,
        schema_version=np.asarray(2),
        iteration=np.asarray(58),
        dataset_particles=np.asarray(8),
        n_particles=np.asarray(3),
        n_threads=np.asarray(8),
        pool_size=np.asarray(24),
        internal_particle_id_by_sorted_position=internal_ids,
        stack_index_by_sorted_position=stack_indices,
        owner_by_sorted_position=owners,
    )
    records = np.empty(
        6,
        dtype=[
            ("launch_sequence", "<u8"),
            ("particle_id", "<i8"),
            ("worker_id", "<i4"),
        ],
    )
    for position, (internal_id, owner, launch_sequence) in enumerate(
        zip(internal_ids, owners, launch_sequences)
    ):
        records[2 * position : 2 * position + 2] = (
            launch_sequence,
            internal_id,
            owner,
        )
    np.savez(
        chronology_path,
        schema_version=np.asarray(1),
        iteration=np.asarray(58),
        n_particles=np.asarray(3),
        n_threads=np.asarray(8),
        records=records,
    )
    return schedule_path, chronology_path


def _enable_particle_issue_replay(monkeypatch, schedule_path, chronology_path):
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured_particle_issue",
    )
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV,
        str(schedule_path),
    )
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_BLOCK_CHRONOLOGY_ENV,
        str(chronology_path),
    )


def test_default_vdam_worker_topology_keeps_single_controller(monkeypatch):
    _clear_worker_replay(monkeypatch)

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(10, dtype=np.int64),
    )

    assert owners is None


def test_round_robin_vdam_worker_topology_selects_eight_host_workers(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(10, dtype=np.int64),
    )

    np.testing.assert_array_equal(
        owners,
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.int32),
    )


def test_round_robin_vdam_worker_topology_preserves_bucket_shape(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )

    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        object(),
        np.arange(12, dtype=np.int64).reshape(3, 4),
    )

    assert owners.shape == (3, 4)
    np.testing.assert_array_equal(owners.ravel(), np.arange(12, dtype=np.int32) % 8)


def test_round_robin_vdam_worker_topology_can_target_one_iteration(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, "58")

    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )
    np.testing.assert_array_equal(
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=58,
        ),
        np.arange(8, dtype=np.int32),
    )


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_round_robin_vdam_worker_topology_rejects_invalid_iteration(
    monkeypatch,
    value,
):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=58,
        )


def test_round_robin_vdam_worker_topology_rejects_captured_schedule(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "round_robin",
    )
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV,
        "/tmp/must-not-be-read.npz",
    )

    with pytest.raises(ValueError, match="cannot also use a captured schedule"):
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
        )


def test_captured_vdam_worker_topology_can_skip_non_target_iteration(monkeypatch):
    _clear_worker_replay(monkeypatch)
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_REPLAY_TOPOLOGY_ENV,
        "captured",
    )
    monkeypatch.setenv(local_em_engine.RELION_VDAM_WORKER_REPLAY_ITER_ENV, "58")
    monkeypatch.setenv(
        local_em_engine.RELION_VDAM_WORKER_SCHEDULE_ENV,
        "/tmp/must-not-be-read-before-target.npz",
    )

    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            object(),
            np.arange(8, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )


def test_captured_particle_issue_replay_joins_stack_ids_and_orders_bucket(
    monkeypatch,
    tmp_path,
):
    _clear_worker_replay(monkeypatch)
    schedule_path, chronology_path = _write_particle_issue_seals(tmp_path)
    _enable_particle_issue_replay(monkeypatch, schedule_path, chronology_path)
    dataset = _IndexDataset([4, 1, 7])

    order = local_em_engine._relion_vdam_particle_issue_order_for_images(
        dataset,
        np.arange(3, dtype=np.int64),
        debug_iteration=58,
    )
    owners = local_em_engine._relion_vdam_worker_lanes_for_images(
        dataset,
        np.arange(3, dtype=np.int64),
        debug_iteration=58,
    )

    np.testing.assert_array_equal(order, np.asarray([1, 2, 0], dtype=np.int32))
    np.testing.assert_array_equal(owners, np.asarray([2, 0, 1], dtype=np.int32))


def test_captured_particle_issue_replay_only_targets_sealed_iteration(
    monkeypatch,
    tmp_path,
):
    _clear_worker_replay(monkeypatch)
    schedule_path, chronology_path = _write_particle_issue_seals(tmp_path)
    _enable_particle_issue_replay(monkeypatch, schedule_path, chronology_path)
    dataset = _IndexDataset([4, 1, 7])

    assert (
        local_em_engine._relion_vdam_particle_issue_order_for_images(
            dataset,
            np.arange(3, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )
    assert (
        local_em_engine._relion_vdam_worker_lanes_for_images(
            dataset,
            np.arange(3, dtype=np.int64),
            debug_iteration=57,
        )
        is None
    )


def test_captured_particle_issue_replay_rejects_nonbijective_launches(
    monkeypatch,
    tmp_path,
):
    _clear_worker_replay(monkeypatch)
    schedule_path, chronology_path = _write_particle_issue_seals(
        tmp_path,
        launch_sequences=(0, 0, 2),
    )
    _enable_particle_issue_replay(monkeypatch, schedule_path, chronology_path)

    with pytest.raises(ValueError, match="launch sequences are not a bijection"):
        local_em_engine._relion_vdam_particle_issue_order_for_images(
            _IndexDataset([4, 1, 7]),
            np.arange(3, dtype=np.int64),
            debug_iteration=58,
        )


def test_captured_particle_issue_replay_rejects_untraced_selected_particle(
    monkeypatch,
    tmp_path,
):
    _clear_worker_replay(monkeypatch)
    schedule_path, chronology_path = _write_particle_issue_seals(tmp_path)
    _enable_particle_issue_replay(monkeypatch, schedule_path, chronology_path)

    with pytest.raises(ValueError, match="missing selected stack indices"):
        local_em_engine._relion_vdam_particle_issue_order_for_images(
            _IndexDataset([4, 0, 7]),
            np.arange(3, dtype=np.int64),
            debug_iteration=58,
        )


def test_particle_issue_order_reorders_every_particle_operand_and_serializes_controller(
    monkeypatch,
):
    from recovar import cuda_backproject

    captured = {}

    def fake_fused_projector(*args, **kwargs):
        captured["images"] = np.asarray(args[2])
        captured["ctf"] = np.asarray(args[3])
        captured["posterior"] = np.asarray(args[5])
        captured["scoring_rotations"] = np.asarray(args[9])
        captured["groups"] = np.asarray(kwargs["reconstruction_group_ids"])
        captured["workers"] = np.asarray(kwargs["worker_lane_ids"])
        captured["trace_ids"] = np.asarray(kwargs["particle_trace_ids"])
        captured["rotation_counts"] = np.asarray(kwargs["rotation_replay_counts"])
        captured["parallel"] = kwargs["parallel_worker_replay"]
        return args[0], args[1], None

    monkeypatch.setattr(
        cuda_backproject,
        "relion_vdam_mstep_fused_projector_x_half",
        fake_fused_projector,
    )
    particle_ids = np.asarray([10, 20, 30], dtype=np.float32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (3, 1, 3, 3),
    ).copy()
    rotations[:, 0, 0, 0] = particle_ids

    local_em_engine._accumulate_relion_vdam_physical_particle_grid(
        images=np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.complex64),
        ctf=np.asarray([[10, 10], [20, 20], [30, 30]], dtype=np.float32),
        minvsigma2=np.ones((3, 2), dtype=np.float32),
        posterior_over_weight_norm=particle_ids[:, None, None],
        translation_angles=np.zeros((1, 2), dtype=np.float32),
        reference=np.zeros((3, 1, 2), dtype=np.complex64),
        rotations=rotations,
        row_mask=np.ones((3, 1), dtype=bool),
        Ft_y=np.zeros((2, 1), dtype=np.complex64),
        Ft_ctf=np.zeros((2, 1), dtype=np.float32),
        projector_full=np.zeros((3, 3, 3), dtype=np.complex64),
        scoring_rotations=rotations,
        projector_r_max=1,
        pixel_indices=np.asarray([0, 1], dtype=np.int32),
        image_shape=(2, 2),
        volume_shape=(3, 3, 3),
        max_r=1,
        reconstruction_group_ids=np.asarray([0, 1, 0], dtype=np.int32),
        worker_lane_ids=np.asarray([4, 5, 6], dtype=np.int32),
        particle_trace_ids=np.asarray([100, 200, 300], dtype=np.int32),
        rotation_replay_counts=np.asarray([11, 22, 33], dtype=np.int32),
        particle_replay_order=np.asarray([1, 2, 0], dtype=np.int32),
    )

    np.testing.assert_array_equal(captured["images"][:, 0].real, [20, 30, 10])
    np.testing.assert_array_equal(captured["ctf"][:, 0], [20, 30, 10])
    np.testing.assert_array_equal(captured["posterior"][:, 0, 0], [20, 30, 10])
    np.testing.assert_array_equal(
        captured["scoring_rotations"][:, 0, 0, 0], [20, 30, 10]
    )
    np.testing.assert_array_equal(captured["groups"], [1, 0, 0])
    np.testing.assert_array_equal(captured["workers"], [5, 6, 4])
    np.testing.assert_array_equal(captured["trace_ids"], [200, 300, 100])
    np.testing.assert_array_equal(captured["rotation_counts"], [22, 33, 11])
    assert captured["parallel"] is False


def test_particle_issue_order_rejects_nonbijective_particle_axis():
    with pytest.raises(ValueError, match="particle-axis bijection"):
        local_em_engine._accumulate_relion_vdam_physical_particle_grid(
            images=np.zeros((2, 1), dtype=np.complex64),
            ctf=np.zeros((2, 1), dtype=np.float32),
            minvsigma2=np.zeros((2, 1), dtype=np.float32),
            posterior_over_weight_norm=np.ones((2, 1, 1), dtype=np.float32),
            translation_angles=np.zeros((1, 2), dtype=np.float32),
            reference=np.zeros((2, 1, 1), dtype=np.complex64),
            rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 1, 3, 3)),
            row_mask=np.ones((2, 1), dtype=bool),
            Ft_y=np.zeros(1, dtype=np.complex64),
            Ft_ctf=np.zeros(1, dtype=np.float32),
            pixel_indices=np.zeros(1, dtype=np.int32),
            image_shape=(1, 1),
            volume_shape=(3, 3, 3),
            max_r=1,
            particle_replay_order=np.asarray([0, 0], dtype=np.int32),
        )
