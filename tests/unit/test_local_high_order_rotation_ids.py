"""Local-search rotation ids must survive sampling orders whose id space exceeds int32.

The parent-expanded local layout encodes a child orientation as
``psi_index * n_pixels + healpix_pixel`` on the unreduced fine lattice. At
HEALPix order 9 that is about 3072 * 3.1e6 = 9.7e9 > 2**31, so int32 ids wrap
to negative values, which the exact local engine then treats as padding and
aborts on (observed on EMPIAR-10202 at RELION iteration 21). These tests pin
the int64 contract at every host-side boundary of the id flow.
"""

import numpy as np
import pytest

from recovar.em.local import local_bucket_stages, local_em_engine, local_layout, local_projection_cache


def test_parent_expanded_child_ids_exceeding_int32_stay_positive_int64():
    fine_order = 9
    # RELION eulers (rot, tilt, psi) in degrees: psi near 360 selects the last
    # psi index so the encoded ids land near the top of the id space.
    prior_eulers = np.array([[12.0, 80.0, 359.5]], dtype=np.float64)
    entries = local_layout._build_parent_expanded_local_entries(
        prior_eulers,
        fine_order,
        sigma_rot=0.3,
        sigma_psi=0.3,
        oversampling_order=1,
    )
    rotation_ids_flat = np.asarray(entries[2])
    assert rotation_ids_flat.dtype == np.int64
    assert rotation_ids_flat.size > 0
    assert np.all(rotation_ids_flat >= 0)
    assert int(rotation_ids_flat.max()) > np.iinfo(np.int32).max


def test_projection_cache_rows_map_large_ids_and_padding():
    cache = local_projection_cache.LocalRelionProjectionCache(
        projections=None,
        id_map=None,
        enabled=True,
        row_count=3,
        id_map_row_count=3,
        unique_ids=np.array([5, 2**31 + 7, 2**33], dtype=np.int64),
    )
    ids = np.array([[2**33, 5, -1], [2**31 + 7, -1, -1]], dtype=np.int64)
    rows = local_projection_cache.rows_for_bucket(cache, ids)
    assert rows.dtype == np.int32
    np.testing.assert_array_equal(rows, np.array([[2, 0, 0], [1, 0, 0]], dtype=np.int32))
    with pytest.raises(RuntimeError, match="missing from the RELION projection cache"):
        local_projection_cache.rows_for_bucket(cache, np.array([[6]], dtype=np.int64))


def test_hard_assignment_encoding_is_int64():
    rotation_ids = np.array([2**33, 3], dtype=np.int64)
    trans = np.array([83, 0], dtype=np.int32)
    encoded = local_bucket_stages.encode_hard_assignment(rotation_ids, trans, 84)
    assert encoded.dtype == np.int64
    assert encoded.tolist() == [2**33 * 84 + 83, 3 * 84]
    assert np.all(encoded // 84 == rotation_ids)
