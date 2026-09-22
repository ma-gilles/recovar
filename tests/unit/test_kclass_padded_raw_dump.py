"""Padded diagnostic rows retain only the real rotation support."""
from types import SimpleNamespace
import numpy as np
from recovar.em.diagnostics import pass2 as sparse_pass2_mod

def test_kclass_dense_pass2_dump_trims_padded_raw_diff2(monkeypatch, tmp_path):
    n_rot = 2
    bucket_n_rot = 4
    n_trans = 3
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.asarray([42], dtype=np.int64),
    )
    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    per_image_inputs = {
        "oversampled_rots": [rotations],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32)],
        "log_prior": [np.asarray([0.1, -0.2], dtype=np.float32)],
    }
    candidate_mask = np.asarray(
        [
            [
                [True, False, True],
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        ],
        dtype=bool,
    )
    scores = np.arange(
        bucket_n_rot * n_trans,
        dtype=np.float32,
    ).reshape(1, bucket_n_rot, n_trans)
    raw_diff2 = (
        np.arange(bucket_n_rot * n_trans, dtype=np.float32)
        .reshape(bucket_n_rot, n_trans)
        + np.float32(500.0)
    )

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    sparse_pass2_mod._maybe_dump_k_class_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([0], dtype=np.int64),
        class_index=0,
        per_image_inputs=per_image_inputs,
        class_bucket_arrays={"candidate_mask": candidate_mask},
        compact_pair_arrays=None,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=np.full_like(scores, 1.0 / scores.size),
        bucket_translation_prior=np.zeros((1, n_trans), dtype=np.float32),
        compact_pairs=False,
        raw_diff2_by_batch_row={0: raw_diff2},
        relion_min_diff2=np.asarray([499.0], dtype=np.float32),
    )

    payload = np.load(dump_dir / "pass2_orig000042_class001_cs014.npz")
    np.testing.assert_array_equal(payload["relion_raw_diff2"], raw_diff2[:n_rot])
    assert payload["relion_min_diff2"] == np.float32(499.0)
