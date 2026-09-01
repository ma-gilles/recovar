from types import SimpleNamespace

import pytest

import numpy as np

from scripts.analyze_k1_wavg_posterior_boundary import (
    _gather_native_table_in_recovar_order,
    _native_to_recovar_rows,
    _physical_image_shape,
)


@pytest.mark.unit
def test_physical_image_shape_comes_from_native_preprocess_header():
    artifact = SimpleNamespace(header=(0,) * 12 + (256, 192))

    assert _physical_image_shape(artifact, stack_index=17) == (256, 192)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(0, 256), (256, -1)])
def test_physical_image_shape_rejects_invalid_native_header(shape):
    artifact = SimpleNamespace(header=(0,) * 12 + shape)

    with pytest.raises(ValueError, match="stack 17"):
        _physical_image_shape(artifact, stack_index=17)


@pytest.mark.unit
def test_native_geometry_map_and_table_gather_are_bijective():
    native = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    recovar = native[[1, 0]]

    row_map, maximum_error = _native_to_recovar_rows(
        native,
        recovar,
        name="test rows",
        tolerance=0.0,
    )

    np.testing.assert_array_equal(row_map, [1, 0])
    assert maximum_error == 0.0
    table = np.asarray([[10, 11], [20, 21]])
    np.testing.assert_array_equal(
        _gather_native_table_in_recovar_order(table, row_map, row_map),
        [[21, 20], [11, 10]],
    )


@pytest.mark.unit
def test_native_geometry_map_rejects_non_bijection():
    native = np.asarray([[1.0], [1.0]], dtype=np.float32)
    recovar = np.asarray([[1.0], [2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="not bijective"):
        _native_to_recovar_rows(
            native,
            recovar,
            name="test rows",
            tolerance=0.0,
        )
