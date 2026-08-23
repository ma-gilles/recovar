from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_k1_native_wavg_pixels import (
    _comparison,
    _load_counted,
    _native_standard_half_indices,
    _normalise_native_weights,
    _particle_row_for_stack,
    _recovar_rows_in_native_order,
    _replace_window_with_native_preprocess,
    _wavg_components,
)


@pytest.mark.parametrize("identity_column", ("rlnImageName", "_rlnImageName"))
def test_particle_row_lookup_uses_stack_identity_not_star_row_order(identity_column):
    particles = pd.DataFrame(
        {
            identity_column: ["1204@particles.mrcs", "2081@particles.mrcs"],
            "rlnDefocusU": [17421.9, 14270.1],
        }
    )

    row = _particle_row_for_stack(particles, 2081)

    assert row[identity_column] == "2081@particles.mrcs"
    assert row["rlnDefocusU"] == 14270.1


def test_load_counted_round_trip(tmp_path):
    path = tmp_path / "values.bin"
    values = np.asarray([1.25, -2.5, 3.75], dtype="<f8")
    path.write_bytes(struct.pack("<Q", values.size) + values.tobytes())

    np.testing.assert_array_equal(_load_counted(path, "<f8"), values)


def test_current_rectangle_mapping_matches_centered_window():
    image_size = 8
    current_size = 4
    half_width = image_size // 2 + 1
    logical_rows = np.asarray([-1, 0, 1, 2])
    centered_rows = logical_rows + image_size // 2
    centered = (
        centered_rows[:, None] * half_width
        + np.arange(current_size // 2 + 1)[None, :]
    ).reshape(-1)

    rows = _recovar_rows_in_native_order(
        centered[::-1],
        current_size=current_size,
        image_size=image_size,
    )
    native_standard = _native_standard_half_indices(current_size, image_size)
    recovar_standard = (
        ((centered[::-1] // half_width - image_size // 2) % image_size) * half_width
        + centered[::-1] % half_width
    )
    np.testing.assert_array_equal(recovar_standard[rows], native_standard)


def test_wavg_components_and_bit_comparison():
    projections = np.asarray([[1 + 2j, -1 + 0.5j]], dtype=np.complex64)
    images = np.asarray([[0.25 - 1j, 2 + 1j]], dtype=np.complex64)
    ctf = np.asarray([2, -0.5], dtype=np.float32)
    probabilities = np.asarray([[1]], dtype=np.float32)

    result = _wavg_components(projections, images, ctf, probabilities)
    # These literals follow the kernel's separate float32 real/imag multiplies
    # and add, rather than NumPy's complex absolute-value implementation.
    np.testing.assert_array_equal(result["wdiff2"], np.asarray([28.0625, 3.8125], np.float32))
    np.testing.assert_array_equal(result["aa"], np.asarray([20.0, 0.3125], np.float32))
    np.testing.assert_array_equal(result["xa"], np.asarray([-3.5, 0.75], np.float32))

    comparison = _comparison(result["wdiff2"], result["wdiff2"], np.asarray([True, True]))
    assert comparison["bit_exact_count"] == 2
    assert comparison["mismatch_count"] == 0


def test_replace_window_with_native_preprocess_scatter_direction():
    processed = np.asarray([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], np.complex64)
    window = np.asarray([3, 1, 0], np.int32)
    native_rows = np.asarray([2, 0, 1], np.int32)
    native = np.asarray([1 + 5j, 2 + 6j, 3 + 7j], np.complex64)

    replaced = _replace_window_with_native_preprocess(
        processed,
        window,
        native_rows,
        native,
        np.float32(0.5),
    )

    np.testing.assert_array_equal(
        replaced,
        np.asarray([2 + 10j, 6 + 14j, 30 + 3j, 4 + 12j], np.complex64),
    )


def test_normalise_native_weights_replaces_relion_sentinel():
    sentinel = np.finfo(np.float32).min
    raw = np.asarray([sentinel, 1.0, 3.0, 0.0, sentinel, sentinel], np.float32)

    result = _normalise_native_weights(raw, 2, 3)

    np.testing.assert_array_equal(
        result,
        np.asarray([[0.0, 0.25, 0.75], [0.0, 0.0, 0.0]], np.float32),
    )
