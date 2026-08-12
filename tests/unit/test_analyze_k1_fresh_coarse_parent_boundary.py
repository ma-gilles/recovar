from types import SimpleNamespace

import numpy as np

from scripts.analyze_k1_fresh_coarse_parent_boundary import (
    _native_parent_keys,
    _recovar_parent_keys,
)
from scripts.validate_relion_fine_score_capture import ACTIVE


def test_native_parent_keys_collapse_eight_by_four_fine_children():
    rotation_dtype = np.dtype(
        [("orientation_class_key", np.uint64), ("orientation_local", np.uint32)]
    )
    rotations = np.zeros(16, dtype=rotation_dtype)
    rotations["orientation_class_key"] = np.repeat([7, 11], 8)
    rotations["orientation_local"] = np.arange(16)
    translation_dtype = np.dtype([("translation", np.uint32)])
    translations = np.zeros(8, dtype=translation_dtype)
    translations["translation"] = np.arange(8)
    candidate_dtype = np.dtype(
        [("flags", np.uint32), ("rotation_local", np.uint64), ("translation_id", np.uint64)]
    )
    candidates = np.zeros(64, dtype=candidate_dtype)
    candidates["flags"] = ACTIVE
    candidates["rotation_local"] = np.repeat(np.arange(16), 4)
    candidates["translation_id"] = np.tile(np.arange(4), 16)

    keys, counts = _native_parent_keys(
        SimpleNamespace(rotations=rotations, translations=translations),
        SimpleNamespace(candidates=candidates),
    )

    np.testing.assert_array_equal(keys, np.asarray([[7, 0], [11, 0]]))
    np.testing.assert_array_equal(counts, np.asarray([32, 32]))


def test_recovar_parent_keys_decode_flat_significant_mask(tmp_path):
    path = tmp_path / "coarse.npz"
    mask = np.zeros(12, dtype=bool)
    mask[[1, 11]] = True
    np.savez(
        path,
        n_classes=np.int64(1),
        n_rot=np.int64(3),
        n_trans=np.int64(4),
        significant_mask=mask,
        n_significant=np.int64(2),
        adaptive_fraction=np.float64(0.999),
    )

    keys, metadata = _recovar_parent_keys(path)

    np.testing.assert_array_equal(keys, np.asarray([[0, 1], [2, 3]]))
    assert metadata["n_significant"] == 2
