import numpy as np
import pytest

from scripts.analyze_k1_bpref_factor_boundary import (
    _classify_localization,
    _metric,
    _translation_map,
)
from scripts.validate_relion_bpref_factor_capture import TRANSLATION_DTYPE


@pytest.mark.unit
def test_classify_k1_bpref_factor_boundary_in_causal_order():
    common = {
        "capture_self_closes": True,
        "same_posterior_operands_close": True,
        "sequential_summary_closes": True,
        "highest_summary_closes": True,
    }
    assert _classify_localization(**{**common, "capture_self_closes": False}) == (
        "relion_factor_capture_does_not_reproduce_relion_summary"
    )
    assert _classify_localization(**{**common, "same_posterior_operands_close": False}) == (
        "bpref_operand_mismatch_before_translation_reduction"
    )
    assert _classify_localization(**{**common, "highest_summary_closes": False}) == (
        "recovar_translation_reduction_order_mismatch"
    )
    assert _classify_localization(**common) == "particle_prescatter_boundary_closes"


@pytest.mark.unit
def test_translation_map_matches_relion_phase_units():
    physical_size = 128
    recovar = np.asarray([[0.0, 0.0], [1.5, -2.0], [-0.5, 0.25]], dtype=np.float32)
    relion = np.zeros(recovar.shape[0], dtype=TRANSLATION_DTYPE)
    relion["translation"] = np.arange(recovar.shape[0], dtype=np.uint32)
    relion["x"] = (-2.0 * np.pi * recovar[:, 0] / physical_size).astype(np.float32)
    relion["y"] = (-2.0 * np.pi * recovar[:, 1] / physical_size).astype(np.float32)
    mapping, error = _translation_map(relion, recovar[[2, 0, 1]], physical_image_size=physical_size)
    assert np.array_equal(mapping, np.asarray([1, 2, 0]))
    assert error <= 1.0e-7


@pytest.mark.unit
def test_metric_uses_exact_and_relative_l2_without_correlation():
    reference = np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] += np.complex64(1.0e-4j)
    metric = _metric(reference, candidate)
    assert metric["exact_equal"] is False
    assert metric["mismatch_count"] == 1
    assert metric["relative_l2_over_reference"] > 0.0
    assert "correlation" not in metric
