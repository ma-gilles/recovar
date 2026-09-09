"""Trial-order setup keeps explicit validation and native fallback separate."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import expected_accuracy as owner

pytestmark = pytest.mark.unit
LOG = logging.getLogger("trial_order_test")


def options(explicit=None, base=None, optics=None):
    return SimpleNamespace(half1_trial_order_local=explicit, half1_base_order_local=base, half1_optics_group_ids=optics)


def prepare(opts, *, size=3, seed=7, initial=0):
    return owner.prepare_relion_half1_trial_order(
        expected_accuracy=opts,
        half1_dataset=SimpleNamespace(n_units=size),
        optimizer_random_seed=seed,
        init_relion_iteration=initial,
        log=LOG,
    )


@pytest.mark.parametrize("order", [[2, 0, 1], [[2, 0, 1]], np.array([2, 0, 1], np.int32)])
def test_explicit_order_preserves_particle_identity_and_bypasses_native(monkeypatch, caplog, order):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit order must not call native setup")

    monkeypatch.setattr(owner, "relion_half1_trial_order", forbidden)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        result = prepare(options(explicit=order))
    np.testing.assert_array_equal(result, [2, 0, 1])
    assert result.dtype == np.int64
    assert "explicit physical trial order (3 particles)" in caplog.text


@pytest.mark.parametrize(
    "order,message",
    [([0, 1], "must have shape"), ([0, 0, 2], "must be a permutation"), ([0, 1, 3], "must be a permutation")],
)
def test_invalid_explicit_metadata_raises_outside_native_fallback(order, message):
    with pytest.raises(ValueError, match=message):
        prepare(options(explicit=order))


def test_empty_explicit_order_stays_an_empty_int64_array():
    result = prepare(options(explicit=[]), size=0)
    assert result.shape == (0,) and result.dtype == np.int64


@pytest.mark.parametrize("size,seed", [(3, None), (0, 7), (0, None)])
def test_unrequested_or_empty_native_order_is_not_called(monkeypatch, size, seed):
    def forbidden(*args, **kwargs):
        pytest.fail("native order is not requested")

    monkeypatch.setattr(owner, "relion_half1_trial_order", forbidden)
    assert prepare(options(), size=size, seed=seed) is None


@pytest.mark.parametrize("initial,first_iteration", [(-3, 1), (0, 1), (5, 6)])
def test_native_order_gets_original_seed_schedule_and_arrays(monkeypatch, initial, first_iteration):
    base = np.array([1, 2, 0], dtype=np.int64)
    optics = np.ones(3, dtype=np.int64)
    native_result = np.array([2, 1, 0], dtype=np.int64)
    calls = []

    def native(n, seed, **kwargs):
        calls.append((n, seed, kwargs))
        return native_result

    monkeypatch.setattr(owner, "relion_half1_trial_order", native)
    result = prepare(options(base=base, optics=optics), initial=initial)
    assert result is native_result
    assert len(calls) == 1
    n, seed, kwargs = calls[0]
    assert (n, seed, kwargs["first_iteration"]) == (3, 7, first_iteration)
    assert kwargs["base_order_local"] is base
    assert kwargs["optics_group_ids"] is optics


@pytest.mark.parametrize("error_type", [ModuleNotFoundError, RuntimeError, ValueError])
def test_native_failure_retains_warning_and_none(monkeypatch, caplog, error_type):
    def unavailable(*args, **kwargs):
        raise error_type("unavailable sentinel")

    monkeypatch.setattr(owner, "relion_half1_trial_order", unavailable)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert prepare(options()) is None
    assert "RELION exact expected-accuracy particle order unavailable: unavailable sentinel" in caplog.text


def test_multiple_optics_groups_fail_closed_after_existing_native_call(monkeypatch, caplog):
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs)
        return np.arange(3, dtype=np.int64)

    monkeypatch.setattr(owner, "relion_half1_trial_order", native)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert prepare(options(optics=np.array([1, 2, 1]))) is None
    assert len(calls) == 1
    assert "exact expected accuracy currently supports one RELION optics group" in caplog.text
