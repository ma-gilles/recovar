import numpy as np
import pytest
from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _max_images_for_mstep_output_budget, _split_sparse_pass2_buckets_by_mstep_output_budget,
    _max_adjoint_block_bytes_for_pass,
)
pytestmark = pytest.mark.unit

def test_sparse_pass2_mstep_output_cap_matches_empiar10202_n800_boundary():
    device_memory_bytes = 80 * 1024**3
    output_budget = _max_adjoint_block_bytes_for_pass(device_memory_bytes)
    n_recon_pixels = 320800

    assert _max_images_for_mstep_output_budget(
        64,
        n_recon_pixels,
        max_output_bytes=output_budget,
    ) == 2
    assert _max_images_for_mstep_output_budget(
        128,
        n_recon_pixels,
        max_output_bytes=output_budget,
    ) == 1
    assert _max_images_for_mstep_output_budget(
        256,
        n_recon_pixels,
        max_output_bytes=output_budget,
    ) == 1

    failed_request_bytes = 5 * 128 * n_recon_pixels * np.dtype(np.complex64).itemsize
    assert failed_request_bytes == 1_642_496_000


def test_sparse_pass2_mstep_output_budget_splits_buckets_without_reordering():
    buckets = [
        {
            "bucket_size": 128,
            "image_indices": np.asarray([7, 2, 9], dtype=np.int64),
            "sentinel": "preserved",
        },
        {
            "bucket_size": 256,
            "image_indices": np.asarray([4, 1], dtype=np.int64),
            "sentinel": "preserved",
        },
    ]
    split = _split_sparse_pass2_buckets_by_mstep_output_budget(
        buckets,
        n_recon_pixels=320800,
        max_output_bytes=_max_adjoint_block_bytes_for_pass(80 * 1024**3),
    )

    assert [int(bucket["bucket_size"]) for bucket in split] == [128, 128, 128, 256, 256]
    assert [int(bucket["image_indices"][0]) for bucket in split] == [7, 2, 9, 4, 1]
    assert all(bucket["sentinel"] == "preserved" for bucket in split)
