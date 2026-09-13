import numpy as np
import pytest

from recovar.em.helpers.shape_buckets import (
    coarse_bucket,
    pad_axis,
    power_bucket,
    power_of_two_bucket,
    round_up_to_multiple,
)


def test_round_up_to_multiple():
    assert round_up_to_multiple(0, 8) == 0
    assert round_up_to_multiple(1, 8) == 8
    assert round_up_to_multiple(16, 8) == 16
    assert round_up_to_multiple(17, 8) == 24


def test_power_of_two_bucket():
    assert power_of_two_bucket(1, minimum=16, maximum=4096) == 16
    assert power_of_two_bucket(17, minimum=16, maximum=4096) == 32
    assert power_of_two_bucket(4097, minimum=16, maximum=4096) == 4097


def test_power_bucket_can_use_low_cardinality_radix_four_classes():
    assert power_bucket(17, base=4, minimum=16, maximum=4096) == 64
    assert power_bucket(65, base=4, minimum=16, maximum=4096) == 256
    assert power_bucket(257, base=4, minimum=16, maximum=4096) == 1024


def test_coarse_bucket():
    assert coarse_bucket(1, small_power2_max=64, large_multiple=16, minimum=16) == 16
    assert coarse_bucket(33, small_power2_max=64, large_multiple=16) == 64
    assert coarse_bucket(65, small_power2_max=64, large_multiple=16) == 80


def test_pad_axis_preserves_values_and_fills_constant():
    arr = np.arange(6).reshape(2, 3)
    padded = pad_axis(arr, 1, 5, value=-1)
    np.testing.assert_array_equal(padded[:, :3], arr)
    np.testing.assert_array_equal(padded[:, 3:], -np.ones((2, 2), dtype=arr.dtype))


def test_pad_axis_rejects_truncation():
    with pytest.raises(ValueError, match="cannot pad"):
        pad_axis(np.zeros((2, 3)), 1, 2)
