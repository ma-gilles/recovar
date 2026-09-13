"""Coarse support encodings retain exact IDs and stored NamedTuple identity."""

import pickle

import numpy as np
import pytest

from recovar.em.scoring.significant_samples import (
    ComplementSignificantSampleIndices,
    compact_significant_sample_indices_from_mask,
    significant_sample_count,
    significant_sample_ids,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mask, expected_ids, encoding",
    [
        ([], [], type(None)),
        ([True, True], [0, 1], type(None)),
        ([False, False], [], np.ndarray),
        ([False, True, False, True], [1, 3], np.ndarray),
        ([True, False, True, True], [0, 2, 3], ComplementSignificantSampleIndices),
    ],
)
def test_coarse_support_encoding_preserves_ids_and_count(mask, expected_ids, encoding):
    samples = compact_significant_sample_indices_from_mask(mask)
    assert type(samples) is encoding
    assert significant_sample_count(samples, len(mask)) == len(expected_ids)
    ids = significant_sample_ids(samples, len(mask))
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, expected_ids)


@pytest.mark.parametrize("protocol", [4, 5])
def test_complement_support_roundtrip(protocol):
    samples = ComplementSignificantSampleIndices(np.array([1, 4], dtype=np.int32), 6)
    assert samples._fields == ("excluded_indices", "total_size")
    assert samples.__annotations__ == {"excluded_indices": np.ndarray, "total_size": int}
    packed = pickle.dumps(samples, protocol=protocol)
    restored = pickle.loads(packed)
    assert type(restored) is ComplementSignificantSampleIndices
    assert restored.size == 4
    assert restored.excluded_indices.dtype == np.int32
    np.testing.assert_array_equal(significant_sample_ids(restored, 6), [0, 2, 3, 5])
