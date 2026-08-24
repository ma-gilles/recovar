from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_bpref_live_source_rows import _exact_rotation_map, _metric


@pytest.mark.unit
def test_exact_rotation_map_accounts_for_recovar_transpose_convention() -> None:
    native = np.asarray(
        [
            np.arange(9, dtype=np.float32),
            np.arange(9, dtype=np.float32) + np.float32(20.0),
        ]
    )
    recovar = native[[1, 0]].reshape(2, 3, 3).transpose(0, 2, 1)

    np.testing.assert_array_equal(_exact_rotation_map(native, recovar), np.asarray([1, 0]))


@pytest.mark.unit
def test_exact_rotation_map_rejects_nonexact_or_duplicate_rows() -> None:
    duplicate_native = np.stack((np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)))
    duplicate_recovar = duplicate_native.copy().transpose(0, 2, 1)

    with pytest.raises(ValueError, match="one-to-one"):
        _exact_rotation_map(duplicate_native, duplicate_recovar)

    native = np.stack(
        (np.eye(3, dtype=np.float32), np.diag(np.asarray([1.0, -1.0, -1.0], dtype=np.float32)))
    )
    recovar = native.copy().transpose(0, 2, 1)
    recovar[1, 0, 0] += np.float32(1.0e-4)
    with pytest.raises(ValueError, match="bitwise aligned"):
        _exact_rotation_map(native, recovar)


@pytest.mark.unit
def test_live_source_metric_is_scale_sensitive() -> None:
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    exact = _metric(reference, reference.copy())
    scaled = _metric(reference, reference * np.float32(2.0))

    assert exact["exact_equal"] is True
    assert exact["relative_l2_over_reference"] == 0.0
    assert scaled["exact_equal"] is False
    assert scaled["relative_l2_over_reference"] == pytest.approx(1.0)
