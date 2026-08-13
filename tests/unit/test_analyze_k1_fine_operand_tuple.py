import numpy as np
import pytest

from scripts.analyze_k1_fine_operand_tuple import _largest_mismatches


@pytest.mark.unit
def test_largest_mismatches_preserves_full_indices_for_selected_pixels():
    relion = np.zeros(6, dtype=np.float32)
    recovar = np.asarray([100.0, 2.0, 200.0, 4.0, 300.0, 1.0], dtype=np.float32)

    mismatches = _largest_mismatches(
        relion,
        recovar,
        flat_indices=np.asarray([1, 3, 5]),
        limit=2,
    )

    assert [row["flat_index"] for row in mismatches] == [3, 1]
    assert [row["abs_delta"] for row in mismatches] == [4.0, 2.0]


@pytest.mark.unit
def test_largest_mismatches_rejects_selected_pixels_outside_operand():
    values = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="outside the flattened operand"):
        _largest_mismatches(values, values, flat_indices=np.asarray([3]))
