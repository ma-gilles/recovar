from __future__ import annotations

import numpy as np

from types import SimpleNamespace

import pytest

from scripts.analyze_k1_bpref_scatter_geometry import (
    _factor_runtime_geometry,
    _fma_rn_f32,
    _mul_rn_f32,
)


def _bits(value) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def test_coordinate_oracle_emulates_deployed_cuda_fma_rounding():
    """Keep the host oracle from reporting a false one-ULP scatter mismatch."""

    matrix_x = np.float32(-0.19866076111793518)
    matrix_y = np.float32(0.1579185426235199)
    pixel_x = np.float32(3.0)
    pixel_y = np.float32(1.0)
    padding = np.float32(2.0)

    second_term = _mul_rn_f32(matrix_y, pixel_y)
    cuda_sum = _fma_rn_f32(matrix_x, pixel_x, second_term)
    cuda_coordinate = _mul_rn_f32(cuda_sum, padding)
    separate_coordinate = _mul_rn_f32(
        _mul_rn_f32(matrix_x, pixel_x) + second_term,
        padding,
    )

    assert _bits(cuda_coordinate) == 3210758628
    assert _bits(separate_coordinate) == 3210758629


def test_coordinate_oracle_preserves_vector_shape_and_float32_dtype():
    left = np.asarray([[-0.25], [0.5]], dtype=np.float32)
    right = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    addend = np.asarray([[0.125], [-0.25]], dtype=np.float32)

    result = _fma_rn_f32(left, right, addend)

    assert result.dtype == np.float32
    assert result.shape == (2, 3)
    assert np.array_equal(
        result,
        np.asarray(
            [
                [-0.125, -0.375, -0.625],
                [0.25, 0.75, 1.25],
            ],
            dtype=np.float32,
        ),
    )


def test_factor_runtime_geometry_supports_case10_score_and_bpref_sizes():
    header = [0] * 64
    header[16:19] = [30, 58, 1]
    header[22] = 28
    header[37:40] = [58, 115, 115]

    current_size, accumulator_shape, max_r = _factor_runtime_geometry(
        SimpleNamespace(header=tuple(header))
    )

    assert current_size == 58
    assert accumulator_shape == (115, 115, 115)
    assert max_r == 28.0


def test_factor_runtime_geometry_rejects_non_cubic_half_layout():
    header = [0] * 64
    header[16:19] = [30, 58, 1]
    header[22] = 28
    header[37:40] = [58, 115, 113]

    with pytest.raises(ValueError, match="cubic x-half accumulator"):
        _factor_runtime_geometry(SimpleNamespace(header=tuple(header)))
