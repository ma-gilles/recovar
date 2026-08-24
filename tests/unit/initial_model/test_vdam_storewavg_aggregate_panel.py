from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_vdam_storewavg_aggregate_panel import (
    _align_native_rows,
    _particle_locations,
    _residual_geometry,
)

pytestmark = pytest.mark.unit


def test_align_native_rows_applies_complete_rotation_permutation():
    native = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.float32)

    aligned = _align_native_rows(native, np.asarray([2, 0, 1]))

    np.testing.assert_array_equal(aligned, np.asarray([[20, 21], [30, 31], [10, 11]]))


def test_align_native_rows_rejects_incomplete_rotation_map():
    with pytest.raises(ValueError, match="complete permutation"):
        _align_native_rows(np.ones((3, 2)), np.asarray([0, 0, 2]))


def test_particle_locations_rejects_cross_shard_identity_overlap():
    class Capture:
        def __init__(self, values):
            self.original_indices = np.asarray(values, dtype=np.int64)

    with pytest.raises(ValueError, match="multiple captures"):
        _particle_locations([Capture([1, 2]), Capture([2, 3])])


def test_residual_geometry_separates_parallel_and_orthogonal_components():
    reference = np.asarray([1.0, 0.0])
    candidate = np.asarray([2.0, 3.0])

    geometry = _residual_geometry(reference, candidate)

    assert geometry["candidate_projection_on_reference"] == pytest.approx(2.0)
    assert geometry["candidate_orthogonal_over_reference"] == pytest.approx(3.0)
    assert geometry["cosine"] == pytest.approx(2.0 / np.sqrt(13.0))
