import numpy as np
import pytest

from scripts.analyze_k1_native_norm_factor_panel import _float32_ulp_distance, _stack_index


def test_float32_ulp_distance_for_adjacent_positive_values():
    value = np.float32(1.170881986618042)
    adjacent = np.nextafter(value, np.float32(np.inf), dtype=np.float32)

    assert _float32_ulp_distance(value, adjacent) == 1
    assert _float32_ulp_distance(adjacent, value) == 1
    assert _float32_ulp_distance(value, value) == 0


def test_float32_ulp_distance_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite and positive"):
        _float32_ulp_distance(0.0, 1.0)


def test_stack_index_parses_relion_identity():
    assert _stack_index("67@particles.128.mrcs") == 67
    with pytest.raises(ValueError, match="invalid RELION image identity"):
        _stack_index("particles.128.mrcs")
