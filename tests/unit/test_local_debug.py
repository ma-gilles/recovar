import numpy as np

from recovar.em.dense_single_volume.local_debug import (
    maybe_write_dense_per_pose_score_dump,
    parse_dense_per_pose_score_dump_request,
)


def test_dense_per_pose_score_dump_uses_safe_label(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_TARGET", "9")
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL", "coarse/class 0")

    request = parse_dense_per_pose_score_dump_request()
    scores = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    maybe_write_dense_per_pose_score_dump(
        request=request,
        indices=np.array([7, 9], dtype=np.int64),
        scores=scores,
        block_index=3,
    )

    out = tmp_path / "target000009_coarse_class_0_block0003.npy"
    assert out.exists()
    np.testing.assert_array_equal(np.load(out), scores[1].astype(np.float64))


def test_dense_per_pose_score_dump_keeps_legacy_name_without_label(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_TARGET", "7")
    monkeypatch.setenv("RECOVAR_DEBUG_PER_POSE_DUMP_PREPRIOR", "1")

    request = parse_dense_per_pose_score_dump_request()

    maybe_write_dense_per_pose_score_dump(
        request=request,
        indices=np.array([7], dtype=np.int64),
        scores=np.ones((1, 2, 3), dtype=np.float32),
        block_index=2,
        preprior=True,
    )

    assert (tmp_path / "target000007_block0002_preprior.npy").exists()
