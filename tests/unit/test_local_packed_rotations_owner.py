"""``_packed_bucket_rotations`` is the one host gather of packed bucket rotations in the local engine."""

import inspect
import types

import numpy as np

from recovar.em.dense_single_volume import local_em_engine


def test_owner_gathers_rows_and_limits_them_to_the_unpadded_batch():
    rot = np.arange(3 * 4 * 9, dtype=np.float64).reshape(3, 4, 3, 3)
    bucket = types.SimpleNamespace(local_rotations=rot, local_mstep_rotations=None)
    take = np.array([[3, 0], [1, 1], [2, 0]])
    mask = np.array([[True, False], [True, True], [False, False]])
    take_jnp, mask_jnp, packed, packed_mstep = local_em_engine._packed_bucket_rotations(bucket, take[:2], mask[:2], batch_rows=2)
    assert packed.shape == (2, 2, 3, 3) and np.array_equal(packed[0, 0], rot[0, 3]) and np.array_equal(packed[1, 1], rot[1, 1])
    assert np.array_equal(packed_mstep, packed) and str(take_jnp.dtype) == "int32" and mask_jnp.shape == (2, 2)
    full = local_em_engine._packed_bucket_rotations(bucket, take, mask)
    assert full[2].shape == (3, 2, 3, 3) and full[2].dtype == np.float64
    cast = local_em_engine._packed_bucket_rotations(bucket, take, mask, rotations_dtype=np.float32)
    assert cast[2].dtype == np.float32 and cast[3].dtype == np.float64


def test_local_engine_sites_use_the_owner():
    src = inspect.getsource(local_em_engine)
    assert src.count("= _packed_bucket_rotations(bucket, reconstruction_take_indices, reconstruction_pack_mask_np") == 4
    assert "reconstruction_take_indices[:, :, None, None]" not in inspect.getsource(local_em_engine.run_local_em_exact)
