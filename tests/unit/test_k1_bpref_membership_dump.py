from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed


def test_k1_bpref_membership_dump_preserves_identity_padding_and_weights(
    monkeypatch,
    tmp_path,
):
    dump_dir = tmp_path / "membership"
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_ITERATION", "2")
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_HALF", "1")
    monkeypatch.setattr(sparse_pass2_bucketed, "_bpref_membership_dump_counter", 0)
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=2, half=1)

    dataset = SimpleNamespace(dataset_indices=np.asarray([41, 73], dtype=np.int64))
    posterior = np.asarray(
        [
            [[0.1, 0.2, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.1, 0.0], [0.2, 0.2, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    reconstruction = np.where(posterior >= np.float32(0.2), posterior, 0.0)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (2, 3, 3, 3),
    ).copy()

    try:
        sparse_pass2_bucketed._maybe_dump_k1_bpref_membership(
            experiment_dataset=dataset,
            image_indices=np.asarray([0, 1], dtype=np.int64),
            current_size=60,
            actual_counts=np.asarray([2, 2], dtype=np.int64),
            rotations=rotations,
            rotation_indices=np.asarray([[10, 11, -1], [20, 21, -1]], dtype=np.int64),
            fine_translations=np.zeros((3, 2), dtype=np.float32),
            candidate_mask=posterior > 0,
            posterior_probs=posterior,
            reconstruction_probs=reconstruction,
            reconstruction_mask=reconstruction > 0,
            reconstruction_sum_weight=np.asarray([0.6, 1.0], dtype=np.float32),
            reconstruction_threshold=np.asarray([0.2, 0.2], dtype=np.float32),
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()

    paths = sorted(dump_dir.glob("*.npz"))
    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=False) as payload:
        assert str(payload["schema"].item()) == "recovar-bpref-rotation-mass-v2"
        assert int(payload["iteration"]) == 2
        assert int(payload["half"]) == 1
        assert int(payload["current_size"]) == 60
        assert np.array_equal(payload["original_indices"], [41, 73])
        assert np.array_equal(payload["stack_indices_1based"], [42, 74])
        assert np.array_equal(payload["actual_counts"], [2, 2])
        assert np.array_equal(
            payload["candidate_translation_count"],
            np.sum(posterior > 0, axis=-1),
        )
        assert np.array_equal(
            payload["posterior_rotation_mass"],
            np.sum(posterior, axis=-1),
        )
        assert np.array_equal(
            payload["reconstruction_rotation_mass"],
            np.sum(reconstruction, axis=-1),
        )
        assert np.array_equal(
            payload["significant_translation_count"],
            np.sum(reconstruction > 0, axis=-1),
        )
        assert payload["posterior_rotation_mass"].dtype == np.float32
        assert payload["rotations"].shape == (2, 3, 3, 3)


def test_k1_bpref_membership_dump_respects_physical_context(monkeypatch, tmp_path):
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_ITERATION", "2")
    monkeypatch.setenv("RECOVAR_BPREF_MEMBERSHIP_DUMP_HALF", "1")
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        sparse_pass2_bucketed._maybe_dump_k1_bpref_membership(
            experiment_dataset=SimpleNamespace(dataset_indices=np.asarray([0])),
            image_indices=np.asarray([0]),
            current_size=60,
            actual_counts=np.asarray([1]),
            rotations=np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
            rotation_indices=np.asarray([[0]]),
            fine_translations=np.zeros((1, 2), dtype=np.float32),
            candidate_mask=np.ones((1, 1, 1), dtype=bool),
            posterior_probs=np.ones((1, 1, 1), dtype=np.float32),
            reconstruction_probs=np.ones((1, 1, 1), dtype=np.float32),
            reconstruction_mask=np.ones((1, 1, 1), dtype=bool),
            reconstruction_sum_weight=np.ones((1,), dtype=np.float32),
            reconstruction_threshold=np.zeros((1,), dtype=np.float32),
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()
    assert not list(tmp_path.glob("*.npz"))
