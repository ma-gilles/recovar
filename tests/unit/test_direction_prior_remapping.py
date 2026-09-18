"""Global and class-axis contracts for shared replay-prior remapping."""

import numpy as np
import pytest

from recovar.em.helpers import orientation_priors as priors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_classes", [None, 1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_remapping_keeps_class_order_and_explicit_dtype(monkeypatch, n_classes, dtype):
    original = np.arange(12 * (n_classes or 1), dtype=np.float64)
    if n_classes is not None:
        original = original.reshape(n_classes, 12)
    calls = []
    outputs = []

    def remap(row, src_order, dst_order, *, dtype):
        assert np.shares_memory(row, original)
        calls.append((row.copy(), src_order, dst_order, dtype))
        output = np.array([len(calls), -len(calls)], dtype=dtype)
        outputs.append(output)
        return output

    monkeypatch.setattr(priors, "remap_direction_prior_to_healpix_order", remap)
    result = priors.remap_half_direction_prior_to_healpix_order(
        original,
        2,
        3,
        n_classes=n_classes,
        dtype=dtype,
    )
    assert len(calls) == (n_classes or 1)
    for index, (row, src, dst, actual_dtype) in enumerate(calls):
        np.testing.assert_array_equal(row, original if n_classes is None else original[index])
        assert (src, dst, actual_dtype) == (2, 3, dtype)
    if n_classes is None:
        assert result is outputs[0]
    else:
        np.testing.assert_array_equal(result, [[i + 1, -i - 1] for i in range(n_classes)])
        assert result.shape == (n_classes, 2)
    assert result.dtype == dtype


@pytest.mark.parametrize("n_classes", [None, 1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("src_order,dst_order", [(0, 0), (0, 1), (1, 0)])
def test_uniform_class_conditionals_remain_uniform(n_classes, dtype, src_order, dst_order):
    source = np.ones(12 * 4**src_order, dtype=np.float64)
    if n_classes is not None:
        # Unequal class masses must not affect the normalized conditional rows.
        source = np.arange(1, n_classes + 1)[:, None] * source
    result = priors.remap_half_direction_prior_to_healpix_order(
        source,
        src_order,
        dst_order,
        n_classes=n_classes,
        dtype=dtype,
    )
    size = 12 * 4**dst_order
    shape = (size,) if n_classes is None else (n_classes, size)
    np.testing.assert_array_equal(result, np.full(shape, 1 / size, dtype=dtype))
    assert result.dtype == dtype


def test_omitted_dtype_preserves_float32_file_replay_default():
    result = priors.remap_half_direction_prior_to_healpix_order(np.ones((4, 12), np.float64), 0, 1, n_classes=4)
    assert result.dtype == np.float32


def test_class_failure_stops_before_later_rows(monkeypatch):
    seen = []

    def remap(row, src_order, dst_order, *, dtype):
        seen.append(int(row[0]))
        if row[0] == 1:
            raise ValueError("bad class prior")
        return row

    monkeypatch.setattr(priors, "remap_direction_prior_to_healpix_order", remap)
    with pytest.raises(ValueError, match="bad class prior"):
        priors.remap_half_direction_prior_to_healpix_order(np.arange(4)[:, None], 0, 1, n_classes=4)
    assert seen == [0, 1]
