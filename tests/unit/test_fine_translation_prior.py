"""Host prior mapping preserves image identity, parent order and precision."""

import numpy as np
import pytest

from recovar.em.helpers.translation_prior import expand_fine_translation_prior

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_fine_prior_preserves_parent_order_duplicates_and_input(dtype, shared):
    prior = np.array([[0.0, -1.25, -3.5], [-2.0, -4.5, -6.0]], dtype=dtype)
    if shared:
        prior = prior[0]
    before = prior.copy()
    fine = expand_fine_translation_prior(
        prior, np.array([2, 0, 2, 1], dtype=np.int32), n_images=2, n_fine_trans=4, dtype=dtype
    )
    expected = np.array([[-3.5, 0.0, -3.5, -1.25], [-6.0, -2.0, -6.0, -4.5]], dtype=dtype)
    if shared:
        expected[1] = expected[0]
    np.testing.assert_array_equal(fine, expected)
    np.testing.assert_array_equal(prior, before)
    assert fine.dtype == dtype
    assert not np.shares_memory(fine, prior)
    assert fine.flags.writeable is not shared


def test_fine_prior_keeps_host_double_values():
    prior = np.array([-1.0 - 2**-40, 0.0], dtype=np.float64)
    fine = expand_fine_translation_prior(
        prior, np.array([0, 0]), n_images=3, n_fine_trans=2, dtype=np.float64
    )
    assert np.all(fine == -1.0 - 2**-40)
    assert np.all(fine != np.float64(np.float32(prior[0])))


@pytest.mark.parametrize("shared", [False, True])
def test_fine_prior_empty_children(shared):
    prior = np.zeros(2 if shared else (3, 2), dtype=np.float32)
    fine = expand_fine_translation_prior(
        prior, np.array([], dtype=np.int32), n_images=3, n_fine_trans=0, dtype=np.float32
    )
    assert fine.shape == (3, 0)
    assert fine.dtype == np.float32


@pytest.mark.parametrize("shape", [(), (1, 1, 1)])
def test_fine_prior_rejects_unsupported_rank(shape):
    with pytest.raises(ValueError, match=f"translation_log_prior must be 1D or 2D, got {len(shape)} dimensions"):
        expand_fine_translation_prior(
            np.zeros(shape), np.array([0]), n_images=1, n_fine_trans=1, dtype=np.float64
        )


def test_fine_prior_rejects_out_of_range_parent():
    with pytest.raises(IndexError):
        expand_fine_translation_prior(
            np.zeros(2), np.array([2]), n_images=1, n_fine_trans=1, dtype=np.float64
        )
