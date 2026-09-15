"""``_flat_parts`` is the one concatenate-or-empty of per-image layout parts in ``local_layout``."""

import inspect

import numpy as np

from recovar.em.local import local_layout


def test_owner_concatenates_casts_and_returns_typed_empties():
    assert local_layout._flat_parts([], empty_shape=(0, 3, 3), dtype=np.float32).shape == (0, 3, 3)
    assert local_layout._flat_parts([], empty_shape=0, dtype=np.int64).dtype == np.int64
    flat = local_layout._flat_parts([np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)], empty_shape=0, dtype=np.int64, cast=np.int64)
    assert flat.dtype == np.int64 and flat.tolist() == [1, 2, 3]
    same = local_layout._flat_parts([np.ones((2, 4), bool)], empty_shape=(0, 4), dtype=bool)
    assert same.dtype == bool and same.shape == (2, 4)


def test_no_inline_concatenate_or_empty_remains():
    src = inspect.getsource(local_layout)
    assert src.count("_flat_parts(") == 18  # the definition and seventeen sites
    assert "axis=0) if rotation_ids_parts else np.zeros" not in src and "if rotations_parts else np.zeros" not in src
