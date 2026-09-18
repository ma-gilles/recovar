"""Both `_postprocess_local_bucket` calls take the bucket's unpadded rows from one owner."""

import types

import numpy as np

from recovar.em.local import local_em_engine


def test_owner_limits_every_row_array_and_keeps_optional_none():
    bucket = types.SimpleNamespace(local_rotation_ids=np.arange(6).reshape(3, 2), local_rotation_mask=np.ones((3, 2), bool), local_rotations=np.zeros((3, 2, 3, 3)), local_source_eulers=None, local_rotation_posterior_ids=np.arange(6).reshape(3, 2))
    rows = local_em_engine._unpadded_bucket_rows(bucket, 2)
    assert list(rows) == ["local_rotation_ids", "local_rotation_mask", "local_rotations", "local_source_eulers", "local_rotation_posterior_ids"]
    assert rows["local_rotation_ids"].shape == (2, 2) and rows["local_rotations"].shape == (2, 2, 3, 3)
    assert rows["local_source_eulers"] is None and rows["local_rotation_posterior_ids"].shape == (2, 2)
