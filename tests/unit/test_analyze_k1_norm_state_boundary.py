import numpy as np

from scripts.analyze_k1_norm_state_boundary import (
    _complex_scale,
    _positive_float32_ulp_distance,
)


def test_positive_float32_ulp_distance_counts_neighbors():
    value = np.float32(1.25)
    neighbor = np.nextafter(value, np.float32(np.inf), dtype=np.float32)

    assert _positive_float32_ulp_distance(value, neighbor) == 1
    assert _positive_float32_ulp_distance(neighbor, value) == 1


def test_complex_scale_recovers_common_amplitude():
    reference = np.asarray([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    candidate = reference / np.float32(1.25)

    result = _complex_scale(reference, candidate)

    assert np.isclose(result["native_over_recovar_optimal_real_scale"], 1.25)
    assert np.isclose(result["scale_error_energy_removal_fraction"], 1.0)
