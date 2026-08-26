import numpy as np

from scripts.analyze_k1_ppref_vs_pass2_projection import _metric


def test_metric_reports_exact_complex_components() -> None:
    values = np.asarray([1 + 2j, -3 + 4j], dtype=np.complex64)
    report = _metric(values, values.copy())
    assert report["float32_component_count"] == 4
    assert report["bitwise_equal_float32_component_count"] == 4
    assert report["first_unequal_float32_component"] is None
    assert report["relative_l2"] == 0.0


def test_metric_localizes_first_float32_component() -> None:
    reference = np.asarray([1 + 2j, -3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] = np.complex64(-2.5 + 4j)
    report = _metric(candidate, reference)
    assert report["float32_component_count"] == 4
    assert report["bitwise_equal_float32_component_count"] == 3
    assert report["first_unequal_float32_component"] == 2
    assert report["relative_l2"] > 0.0
