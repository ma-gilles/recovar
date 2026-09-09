"""Initial half/class layouts preserve canonical values and array ownership."""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.mean_helpers import prepare_initial_mean_variance
from recovar.em.dense_single_volume.projector_preparation import prepare_initial_real_references

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)
SHAPE = (3, 3, 3)


@pytest.mark.parametrize("classes", [1, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["shared", "stacked", "pair"])
def test_real_reference_half_class_layout_and_aliases(classes, dtype, layout):
    source = (np.arange(classes * 27).reshape((classes,) + SHAPE) / 17).astype(dtype)
    if layout == "shared":
        value = source[0] if classes == 1 else source
        expected = [source, source]
    elif layout == "stacked":
        value = np.stack([source, -source])
        expected = [source, -source]
    else:
        value = (source.copy(), -source)
        expected = list(value)
    result = prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=classes, log=LOG)
    for actual, wanted in zip(result, expected):
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, wanted)
    if layout == "shared":
        assert result[0] is result[1]
    if dtype == np.float64:
        for index in range(2):
            origin = value[index] if layout == "pair" else value
            assert np.shares_memory(result[index], origin)


def test_absent_real_reference_keeps_fourier_fallback(caplog):
    assert prepare_initial_real_references(None, volume_shape=SHAPE, n_classes=4, log=LOG) == [None, None]
    assert not caplog.records


def test_single_class_half_stack_without_class_axis():
    value = np.arange(54, dtype=np.float64).reshape((2,) + SHAPE)
    result = prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=1, log=LOG)
    assert all(v.shape == (1,) + SHAPE for v in result)
    np.testing.assert_array_equal(result[0][0], value[0])
    np.testing.assert_array_equal(result[1][0], value[1])


@pytest.mark.parametrize("value,classes", [(np.zeros(27), 1), (np.zeros(SHAPE), 4), ([np.zeros(SHAPE)] * 2, 2)])
def test_incompatible_real_reference_fails_without_broadcast(value, classes):
    with pytest.raises(ValueError, match="init_reference_real must be"):
        prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=classes, log=LOG)


@pytest.mark.parametrize("classes", [False, True])
def test_shared_tau2_keeps_original_object(classes):
    initial = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    shared, halves = prepare_initial_mean_variance(
        initial, use_per_half_mean_variance=False, k_class_enabled=classes, log=LOG
    )
    assert shared is initial
    assert halves[0] is initial and halves[1] is initial


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_half_tau2_values_and_existing_promoted_average(dtype, monkeypatch):
    monkeypatch.setenv("RECOVAR_USE_FLOAT64_SCORING", "0")
    monkeypatch.setenv("RECOVAR_USE_FLOAT64_PROJECTIONS", "0")
    source = np.array([[1, 2**24, 3], [2**-20, -(2**24), 5]], dtype=dtype)
    initial = jnp.asarray(source)
    shared, halves = prepare_initial_mean_variance(
        initial, use_per_half_mean_variance=True, k_class_enabled=False, log=LOG
    )
    expected = ((source[0].astype(np.float64) + source[1].astype(np.float64)) * 0.5).astype(np.float32)
    assert shared.dtype == np.float32
    np.testing.assert_array_equal(shared, expected)
    for index, half in enumerate(halves):
        assert half.dtype == dtype
        np.testing.assert_array_equal(half, source[index])


@pytest.mark.parametrize(
    "shape,kclass,message",
    [((2, 3), True, "only for K=1"), ((3,), False, "leading half axis 2"), ((3, 2), False, "leading half axis 2")],
)
def test_half_tau2_rejects_unsupported_inputs(shape, kclass, message):
    with pytest.raises(ValueError, match=message):
        prepare_initial_mean_variance(jnp.ones(shape), use_per_half_mean_variance=True, k_class_enabled=kclass, log=LOG)
