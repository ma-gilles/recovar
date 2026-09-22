from __future__ import annotations

import struct

import numpy as np
import pytest

from scripts.analyze_k1_bpref_integrated_kernel import _float32_from_bits, _metric


@pytest.mark.unit
def test_float32_from_bits_preserves_native_controls() -> None:
    expected = np.float32(0.999)
    bits = struct.unpack("<I", struct.pack("<f", expected))[0]
    assert _float32_from_bits(bits).tobytes() == expected.tobytes()


@pytest.mark.unit
def test_integrated_kernel_metric_is_exact_and_scale_sensitive() -> None:
    reference = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    exact = _metric(reference, reference.copy())
    scaled = _metric(reference, reference * np.complex64(2.0))

    assert exact["exact_equal"] is True
    assert exact["support_mismatch_count"] == 0
    assert scaled["relative_l2_over_reference"] == pytest.approx(1.0)
