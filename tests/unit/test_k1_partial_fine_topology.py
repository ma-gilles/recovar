import numpy as np
import pytest

from scripts.analyze_k1_partial_fine_topology import partial_rotation_map


@pytest.mark.unit
def test_partial_rotation_map_reports_exact_overlap_and_unmatched_rows():
    recovar = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        ]
    )
    native_matrices = np.asarray(
        [
            recovar[2],
            np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
            recovar[0],
        ]
    )
    factor = np.empty(3, dtype=[("matrix", np.float32, (9,))])
    factor["matrix"] = native_matrices.transpose(0, 2, 1).reshape(3, 9)

    mapping, native_only, recovar_only = partial_rotation_map(factor, recovar)

    np.testing.assert_array_equal(mapping, np.asarray([2, -1, 0]))
    np.testing.assert_array_equal(native_only, np.asarray([1]))
    np.testing.assert_array_equal(recovar_only, np.asarray([1]))
