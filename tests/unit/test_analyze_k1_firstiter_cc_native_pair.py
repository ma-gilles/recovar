from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_firstiter_cc_native_pair import _float_record


pytestmark = pytest.mark.unit


def test_float_record_preserves_binary32_bits() -> None:
    value = np.nextafter(np.float32(0.25), np.float32(np.inf))
    record = _float_record(value)
    assert record["float32_bits_hex"] == f"0x{value.view(np.uint32).item():08x}"
