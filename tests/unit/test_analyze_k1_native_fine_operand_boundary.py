from __future__ import annotations

import numpy as np

from scripts.analyze_k1_native_fine_operand_boundary import (
    _metric,
    _native_gaussian_components,
    _native_to_recovar_compact,
)


def test_metric_accepts_noncontiguous_panel_slices():
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    panel = values[:, ::2]
    assert not panel.flags.c_contiguous

    report = _metric(panel, panel.copy())

    assert report["bit_equal_fraction"] == 1.0
    assert report["value_mismatch_count"] == 0


def test_native_gaussian_components_use_half_tree_sum_and_highres_constant():
    reference = np.asarray([[3.0 + 4.0j, 2.0 - 1.0j]], dtype=np.complex64)
    shifted = np.asarray([[1.0 + 1.0j, -1.0 + 1.0j]], dtype=np.complex64)
    corr = np.asarray([2.0, 0.5], dtype=np.float32)

    terms, lanes, raw = _native_gaussian_components(
        reference,
        shifted,
        corr,
        np.float32(0.25),
    )

    np.testing.assert_array_equal(terms, np.asarray([[13.0, 3.25]], dtype=np.float32))
    np.testing.assert_array_equal(lanes[0, :2], terms[0])
    np.testing.assert_array_equal(lanes[0, 2:], np.zeros(254, dtype=np.float32))
    np.testing.assert_array_equal(raw, np.asarray([16.5], dtype=np.float32))


def test_native_to_recovar_compact_preserves_signed_fftw_rows():
    native_size = 6
    recovar_size = 8
    recovar_xdim = recovar_size // 2 + 1
    lookup = np.arange(recovar_size * recovar_xdim, dtype=np.int64)

    mapping = _native_to_recovar_compact(
        native_image_size=native_size * (native_size // 2 + 1),
        recovar_full_to_compact=lookup,
    ).reshape(native_size, native_size // 2 + 1)

    np.testing.assert_array_equal(mapping[0], lookup[:4])
    np.testing.assert_array_equal(mapping[3], lookup[3 * recovar_xdim : 3 * recovar_xdim + 4])
    np.testing.assert_array_equal(mapping[4], lookup[6 * recovar_xdim : 6 * recovar_xdim + 4])
    np.testing.assert_array_equal(mapping[5], lookup[7 * recovar_xdim : 7 * recovar_xdim + 4])
