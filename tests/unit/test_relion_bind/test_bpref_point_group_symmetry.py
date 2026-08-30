"""RELION CPU oracle tests for BPref point-group symmetry."""

from __future__ import annotations

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import apply_point_group_symmetry_to_bpref

pytestmark = pytest.mark.unit

_ORI_SIZE = 4
_PADDING_FACTOR = 1
_CURRENT_SIZE = 4
_R_MAX = 2
_PAD_SIZE = 7
_HALF_SHAPE = (_PAD_SIZE, _PAD_SIZE, _PAD_SIZE // 2 + 1)
_CENTER = _PAD_SIZE // 2


def _apply(data, weight, symmetry):
    return apply_point_group_symmetry_to_bpref(
        np.ascontiguousarray(data, dtype=np.complex128),
        np.ascontiguousarray(weight, dtype=np.float64),
        symmetry=symmetry,
        ori_size=_ORI_SIZE,
        padding_factor=_PADDING_FACTOR,
        current_size=_CURRENT_SIZE,
        r_max=_R_MAX,
        enforce_hermitian=True,
    )


def test_cpu_oracle_negative_x_uses_complex_conjugate():
    data = np.zeros(_HALF_SHAPE, dtype=np.complex128)
    weight = np.zeros(_HALF_SHAPE, dtype=np.float64)
    target = (_CENTER, _CENTER, 1)
    data[target] = 2.0 + 3.0j
    weight[target] = 5.0

    data_out, weight_out = _apply(data, weight, "C2")

    assert data_out[target] == 4.0 + 0.0j
    assert weight_out[target] == 10.0


def test_cpu_oracle_sums_weights_without_group_average():
    data = np.zeros(_HALF_SHAPE, dtype=np.complex128)
    weight = np.ones(_HALF_SHAPE, dtype=np.float64)
    target = (_CENTER, _CENTER, 1)

    _, weight_out = _apply(data, weight, "D2")

    assert weight_out[target] == 4.0


def test_cpu_oracle_support_radius_includes_boundary_and_excludes_margin():
    data = np.zeros(_HALF_SHAPE, dtype=np.complex128)
    weight = np.zeros(_HALF_SHAPE, dtype=np.float64)
    boundary = (_CENTER, _CENTER, _R_MAX)
    outside = (_CENTER, _CENTER, _R_MAX + 1)
    data[boundary] = 1.0 + 2.0j
    data[outside] = 5.0 + 7.0j
    weight[boundary] = 2.0
    weight[outside] = 3.0

    data_out, weight_out = _apply(data, weight, "C2")

    assert data_out[boundary] == 2.0 + 0.0j
    assert weight_out[boundary] == 4.0
    assert data_out[outside] == data[outside]
    assert weight_out[outside] == weight[outside]


def test_cpu_oracle_x0_is_summed_once_and_center_is_untouched():
    data = np.zeros(_HALF_SHAPE, dtype=np.complex128)
    weight = np.zeros(_HALF_SHAPE, dtype=np.float64)
    first = (_CENTER - 1, _CENTER, 0)
    partner = (_CENTER + 1, _CENTER, 0)
    center = (_CENTER, _CENTER, 0)
    data[first] = 1.0 + 2.0j
    data[partner] = 3.0 + 5.0j
    data[center] = 7.0 + 11.0j
    weight[first] = 13.0
    weight[partner] = 17.0
    weight[center] = 19.0

    data_out, weight_out = _apply(data, weight, "C1")

    expected = data[first] + np.conj(data[partner])
    assert data_out[first] == expected
    assert data_out[partner] == np.conj(expected)
    assert weight_out[first] == 30.0
    assert weight_out[partner] == 30.0
    assert data_out[center] == data[center]
    assert weight_out[center] == weight[center]
