"""Local slab normalization preserves shape, dtype and error boundaries."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.projector_preparation import prepare_local_projector_slab

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("class_axis", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_slab_preserves_values_dtype_and_input(dtype, class_axis, strided):
    slab = np.arange(120).reshape(4, 6, 5).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        slab += 3j
    if strided:
        slab = slab[::-1, ::2, ::-1]
    supplied = slab[None] if class_axis else slab
    before = supplied.copy()
    actual = prepare_local_projector_slab(supplied)
    assert actual.dtype == slab.dtype
    np.testing.assert_array_equal(actual, slab)
    np.testing.assert_array_equal(supplied, before)


@pytest.mark.parametrize("shape", [(), (4,), (4, 3), (2, 4, 4, 3), (0, 4, 4, 3), (1, 1, 4, 4, 3)])
@pytest.mark.parametrize("path_label", ["local RELION projector path", "local RELION projector big-JIT path"])
def test_invalid_slab_preserves_exact_path_error(shape, path_label):
    reason = "a single-class projector slab" if len(shape) == 4 else "Projector::data shape (z, y, x_half)"
    with pytest.raises(ValueError) as error:
        prepare_local_projector_slab(np.zeros(shape, np.complex64), path_label=path_label)
    assert str(error.value) == f"{path_label} expected {reason}, got {shape}"


def test_existing_device_slab_is_not_copied():
    slab = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    assert prepare_local_projector_slab(slab) is slab


def test_zero_spatial_extent_retains_existing_shape_only_contract():
    slab = jnp.empty((0, 4, 3), dtype=jnp.complex64)
    assert prepare_local_projector_slab(slab) is slab
