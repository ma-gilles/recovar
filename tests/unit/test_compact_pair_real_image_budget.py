"""Actual image chunks must obey the gathered diff2 footprint."""
import numpy as np
import pytest
from recovar.em.sparse_pass2.sparse_pass2_adjoint import _split_compact_pair_buckets_by_projection_gather_budget

@pytest.mark.parametrize('skip,expected_max', [(False, 2), (True, 10)])
def test_real_images_obey_diff2_budget(monkeypatch, skip, expected_max):
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE', '0')
    monkeypatch.setenv('RECOVAR_SPARSE_PASS2_LADDER_CHUNKS', '0')
    buckets = [dict(image_indices=np.arange(10), pair_bucket_size=100, class_bucket_sizes=(1,))]
    result = _split_compact_pair_buckets_by_projection_gather_budget(
        buckets, [], n_score_pixels=8, n_recon_pixels=8,
        projection_complex_dtype=np.complex64, max_gather_bytes=25600,
        rotation_block_size_for_quantization=1, skip_diff2_gather_budget=skip,
    )
    assert max(len(b['image_indices']) for b in result) == expected_max
    np.testing.assert_array_equal(np.concatenate([b['image_indices'] for b in result]), np.arange(10))
