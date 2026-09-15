"""Both `_postprocess_local_bucket` calls take the bucket's unpadded rows from one owner."""

import inspect
import types

import numpy as np

from recovar.em.local import local_em_engine


def test_owner_limits_every_row_array_and_keeps_optional_none():
    bucket = types.SimpleNamespace(local_rotation_ids=np.arange(6).reshape(3, 2), local_rotation_mask=np.ones((3, 2), bool), local_rotations=np.zeros((3, 2, 3, 3)), local_source_eulers=None, local_rotation_posterior_ids=np.arange(6).reshape(3, 2))
    rows = local_em_engine._unpadded_bucket_rows(bucket, 2)
    assert list(rows) == ["local_rotation_ids", "local_rotation_mask", "local_rotations", "local_source_eulers", "local_rotation_posterior_ids"]
    assert rows["local_rotation_ids"].shape == (2, 2) and rows["local_rotations"].shape == (2, 2, 3, 3)
    assert rows["local_source_eulers"] is None and rows["local_rotation_posterior_ids"].shape == (2, 2)


def test_both_postprocess_calls_use_the_owner():
    from recovar.em.local import local_bucket_stages

    # one call site is the engine's own, the other is in the bucket stage that owns it
    src = inspect.getsource(local_em_engine) + inspect.getsource(local_bucket_stages)
    assert src.count("**_unpadded_bucket_rows(bucket, unpadded_batch_size),") == 2
    for fn in (local_bucket_stages._postprocess_fixed_capacity_whole_score_calls, local_em_engine.run_local_em_exact):
        assert "**_unpadded_bucket_rows(bucket, unpadded_batch_size)," in inspect.getsource(fn)
    # only the owner slices the mask (the VDAM replay capture passes the rotation ids alone)
    assert src.count("local_rotation_mask=bucket.local_rotation_mask[:unpadded_batch_size]") == 1
