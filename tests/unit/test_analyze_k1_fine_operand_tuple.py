import numpy as np
import pytest

from scripts.analyze_k1_fine_operand_tuple import (
    _largest_mismatches,
    _score_window_rows_from_relion_full,
)


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


@pytest.mark.unit
def test_score_window_mapping_handles_even_y_nyquist_and_permuted_rows():
    # current_size=4 uses FFTW rows ky=[0, 1, +2, -1].  The physical score
    # window is centered in an 8x8 image, so these selected rows map to
    # physical half-plane indices [20, 26, 32, 16].  In particular, ky=+2
    # must not be interpreted as ky=-2, which is the projector convention.
    supported_full = np.asarray([0, 4, 8, 10], dtype=np.int64)
    window_indices = np.asarray([32, 20, 16, 26], dtype=np.int64)

    rows = _score_window_rows_from_relion_full(
        supported_full=supported_full,
        window_indices=window_indices,
        image_shape=(8, 8),
        current_size=4,
    )

    assert rows.tolist() == [1, 3, 0, 2]
    assert window_indices[rows].tolist() == [20, 26, 32, 16]


@pytest.mark.unit
def test_score_window_mapping_rejects_missing_or_duplicate_pixels():
    supported_full = np.asarray([0, 8], dtype=np.int64)

    with pytest.raises(ValueError, match="duplicate physical pixels"):
        _score_window_rows_from_relion_full(
            supported_full=supported_full,
            window_indices=np.asarray([20, 20], dtype=np.int64),
            image_shape=(8, 8),
            current_size=4,
        )

    with pytest.raises(ValueError, match="score pixels and RECOVAR score window differ"):
        _score_window_rows_from_relion_full(
            supported_full=supported_full,
            window_indices=np.asarray([20, 31], dtype=np.int64),
            image_shape=(8, 8),
            current_size=4,
        )
