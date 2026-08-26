from __future__ import annotations

import numpy as np

from scripts.analyze_k1_native_fine_operand_boundary import (
    _align_recovar_operands_to_native_units,
    _metric,
    _native_gaussian_components,
    _native_to_recovar_compact,
)


def test_strict_score_operands_are_not_rescaled_twice():
    native_preprocessed = np.asarray([1 + 2j], dtype=np.complex64)
    native_corrected = np.asarray([3 - 4j], dtype=np.complex64)
    native_reference = np.asarray([[5 + 6j]], dtype=np.complex64)
    native_shifted = np.asarray([[7 - 8j]], dtype=np.complex64)
    native_corr = np.asarray([9], dtype=np.float32)
    n2 = np.float32(16)

    actual = _align_recovar_operands_to_native_units(
        direct_preprocessed=native_preprocessed * n2,
        direct_corrected=-native_corrected * n2,
        score_reference=-native_reference,
        score_shifted=-native_shifted,
        score_corr=native_corr,
        physical_image_size=4,
    )

    for observed, expected in zip(
        actual,
        (
            native_preprocessed,
            native_corrected,
            native_reference,
            native_shifted,
            native_corr,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)


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
