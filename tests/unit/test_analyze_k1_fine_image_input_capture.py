"""Structural tests for the bounded RELION fine-image-input artifact."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_fine_image_input_capture import (
    FOOTER,
    FOOTER_MAGIC,
    HEADER,
    HEADER_MAGIC,
    PIXEL_DTYPE,
    _load_capture,
)


pytestmark = pytest.mark.unit


def _write_capture(path, *, pixel_ids=(0, 1)) -> None:
    pixels = np.zeros(len(pixel_ids), dtype=PIXEL_DTYPE)
    pixels["pixel"] = pixel_ids
    pixels["local_fctf"] = np.asarray([0.5, -0.25], dtype=np.float32)
    pixels["pixel_correction"] = np.asarray([2.0, -4.0], dtype=np.float32)
    pixels["fourier_real"] = np.asarray([3.0, 5.0], dtype=np.float32)
    pixels["corrected_real"] = pixels["fourier_real"] * pixels["pixel_correction"]
    artifact_bytes = HEADER.size + pixels.nbytes + FOOTER.size
    values = [0] * 40
    values[0] = 1
    values[1] = HEADER.size
    values[2] = PIXEL_DTYPE.itemsize
    values[3] = FOOTER.size
    values[4:12] = [2, 639, 1574, 1, 0, 0, len(pixels), 1]
    values[16] = artifact_bytes
    values[20:22] = [1, 1]
    path.write_bytes(
        HEADER.pack(HEADER_MAGIC, *values)
        + pixels.tobytes()
        + FOOTER.pack(FOOTER_MAGIC, len(pixels), 0)
    )


def test_load_fine_image_input_capture_accepts_dense_ordered_pixels(tmp_path):
    path = tmp_path / "fine-image-input-v1.bin"
    _write_capture(path)

    header, pixels = _load_capture(path)

    assert header[5:7] == (639, 1574)
    np.testing.assert_array_equal(pixels["pixel"], np.asarray([0, 1]))
    np.testing.assert_array_equal(pixels["corrected_real"], np.asarray([6.0, -20.0]))


def test_load_fine_image_input_capture_rejects_permuted_pixel_ids(tmp_path):
    path = tmp_path / "fine-image-input-v1.bin"
    _write_capture(path, pixel_ids=(1, 0))

    with pytest.raises(ValueError, match="dense and ordered"):
        _load_capture(path)
