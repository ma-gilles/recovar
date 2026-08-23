import numpy as np
import pytest

from scripts.analyze_k1_bpref_factor_boundary import (
    _capture_stack_indices,
    _classify_localization,
    _first_cross_engine_boundary,
    _first_primitive_boundary,
    _metric,
    _recovar_capture_path,
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename",
    [
        "pass2_orig000192_cs060.npz",
        "pass2_orig000192_class001_cs060.npz",
    ],
)
def test_recovar_capture_path_accepts_k1_filename_variants(tmp_path, filename):
    expected = tmp_path / filename
    expected.touch()
    assert _recovar_capture_path(tmp_path, original_index=192, current_size=60) == expected


@pytest.mark.unit
def test_capture_stack_indices_allow_qualified_target_subset():
    assert _capture_stack_indices(
        {"capture_stack_indices_one_based": [11, 13, 17]},
        [11, 17],
    ) == [11, 13, 17]


@pytest.mark.unit
def test_capture_stack_indices_reject_missing_target():
    with pytest.raises(ValueError, match="not a subset"):
        _capture_stack_indices(
            {"capture_stack_indices_one_based": [11, 13]},
            [11, 17],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"support_exact": False}, "support"),
        ({"posterior_rel_l2": 2.0e-5}, "posterior"),
        ({"same_posterior_operands_close": False}, "bpref_primitive:inverse_noise"),
        (
            {"sequential_summary_closes": False, "highest_summary_closes": False},
            "translation_reduction_or_unmeasured_operand",
        ),
        ({"highest_summary_closes": False}, "translation_reduction_order"),
        ({}, "accumulator_destination_and_inter_particle_reduction"),
    ],
)
def test_first_cross_engine_boundary_preserves_causal_order(override, expected):
    particle = {
        "support_exact": True,
        "comparisons": {
            "posterior_common_support": {"relative_l2_over_reference": 0.0},
        },
        "same_posterior_operands_close": True,
        "first_primitive_boundary": "inverse_noise",
        "sequential_summary_closes": True,
        "highest_summary_closes": True,
    }
    posterior_rel_l2 = override.pop("posterior_rel_l2", None)
    particle.update(override)
    if posterior_rel_l2 is not None:
        particle["comparisons"]["posterior_common_support"][
            "relative_l2_over_reference"
        ] = posterior_rel_l2
    assert _first_cross_engine_boundary([particle]) == expected


@pytest.mark.unit
def test_first_primitive_boundary_uses_observer_qualified_tolerance():
    comparisons = {
        name: {"relative_l2_over_reference": 0.0}
        for name in (
            "ctf_with_scale",
            "inverse_noise",
            "weighted_ctf",
            "translated_fourier_image",
            "same_posterior_numerator_terms",
            "same_posterior_denominator_terms",
        )
    }
    comparisons["inverse_noise"]["relative_l2_over_reference"] = 1.01e-7
    comparisons["weighted_ctf"]["relative_l2_over_reference"] = 2.0e-7
    assert _first_primitive_boundary(comparisons) == "inverse_noise"
