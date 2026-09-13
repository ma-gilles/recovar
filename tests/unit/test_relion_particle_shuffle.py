"""Pin the distinct legacy and f2c1a384 AutoRefine particle orders."""
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.helpers.expected_accuracy import relion_auto_refine_half_orders

pytestmark = pytest.mark.unit


def test_mt19937_paired_reference():
    from recovar.relion_bind import _relion_bind_core as bind

    # Independent GCC 11 transcription of RELION f2c1a384 exp_model.cpp;
    # importantly, half 2 continues the generator consumed by half 1.
    first, second = bind.auto_refine_randomise_half_orders_mt19937(10, 7, 1712)
    np.testing.assert_array_equal(first, [5, 6, 8, 4, 7, 0, 2, 1, 9, 3])
    np.testing.assert_array_equal(second, [5, 0, 2, 3, 4, 1, 6])


def test_legacy_paired_reference_is_unchanged():
    from recovar.relion_bind import _relion_bind_core as bind

    first, second = bind.auto_refine_randomise_half_orders(10, 7, 1712)
    np.testing.assert_array_equal(first, [0, 4, 3, 9, 1, 8, 2, 5, 6, 7])
    np.testing.assert_array_equal(second, [2, 3, 6, 0, 1, 5, 4])


@pytest.mark.parametrize('algorithm', ['legacy', 'mt19937'])
def test_python_dispatch_and_stable_optics_sort(monkeypatch, algorithm):
    from recovar import relion_bind

    calls = []

    def shuffled(n1, n2, seed):
        calls.append((n1, n2, seed))
        return [2, 0, 1], [1, 0]

    name = 'auto_refine_randomise_half_orders' + ('_mt19937' if algorithm == 'mt19937' else '')
    monkeypatch.setattr(relion_bind, '_relion_bind_core', SimpleNamespace(**{name: shuffled}))
    first, second = relion_auto_refine_half_orders(
        [1, 2, 1, 2, 1], 42, optics_group_ids=[2, 1, 1, 1, 2], shuffle_algorithm=algorithm,
    )
    assert calls == [(3, 2, 43)]
    np.testing.assert_array_equal(first, [2, 4, 0])
    np.testing.assert_array_equal(second, [3, 1])


def test_modern_choice_never_falls_back_to_old_binary(monkeypatch):
    from recovar import relion_bind

    monkeypatch.setattr(relion_bind, '_relion_bind_core', SimpleNamespace())
    with pytest.raises(RuntimeError, match='mt19937'):
        relion_auto_refine_half_orders([1, 2], 42, shuffle_algorithm='mt19937')


def test_invalid_shuffle_is_rejected():
    with pytest.raises(ValueError, match='shuffle_algorithm'):
        relion_auto_refine_half_orders([1, 2], 42, shuffle_algorithm='auto')
