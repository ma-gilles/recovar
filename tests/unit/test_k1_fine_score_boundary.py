from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_k1_fine_score_boundary import (
    _classify_particle,
    _first_exact_boundary,
    _first_mismatch_record,
    _float32_ulp_distance,
    _geometry_only_significant_count,
    _metric,
    _raw_diff2_terms,
    _reduce_relion_fine_lanes,
    _rotation_map,
    _stable_top_n_mask,
)


@pytest.mark.unit
def test_stable_top_n_mask_preserves_tie_order():
    weights = np.asarray([0.5, 0.5, 0.25, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(
        _stable_top_n_mask(weights, 2),
        np.asarray([True, True, False, False]),
    )


@pytest.mark.unit
def test_rotation_map_uses_exact_transposed_matrix_permutation():
    recovar = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        ]
    )
    permutation = np.asarray([2, 0, 1])
    factor = np.empty(3, dtype=[("matrix", np.float32, (9,))])
    factor["matrix"] = recovar[permutation].transpose(0, 2, 1).reshape(3, 9)

    mapping, error = _rotation_map(factor, recovar)

    np.testing.assert_array_equal(mapping, permutation)
    assert error == 0.0


@pytest.mark.unit
def test_geometry_only_support_uses_native_bpref_header_count():
    header = np.zeros(54, dtype=np.uint64)
    header[45] = 227
    factor = SimpleNamespace(
        header=header,
        rotations=np.empty(80),
        translations=np.empty(116),
    )

    assert _geometry_only_significant_count(factor) == 227


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("active_tuple_subset", "active_candidate_tuple_mismatch"),
        ("raw_diff2_close", "raw_fine_diff2_mismatch"),
        ("priors_close", "fine_prior_mismatch"),
        ("centered_log_weight_close", "fine_log_weight_arithmetic_mismatch"),
        ("posterior_close", "fine_normalized_posterior_mismatch"),
        ("support_exact", "fine_significant_support_mismatch"),
    ],
)
def test_classification_reports_first_unequal_boundary(changed, expected):
    values = {
        "active_tuple_subset": True,
        "raw_diff2_close": True,
        "priors_close": True,
        "centered_log_weight_close": True,
        "posterior_close": True,
        "support_exact": True,
    }
    values[changed] = False
    assert _classify_particle(**values) == expected


@pytest.mark.unit
def test_metric_uses_exact_and_relative_l2_without_correlation():
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1.0, 2.001], dtype=np.float32)
    metric = _metric(reference, candidate)
    assert metric["exact_equal"] is False
    assert metric["relative_l2_over_reference"] > 0.0
    assert "correlation" not in metric


@pytest.mark.unit
def test_first_mismatch_record_reports_stable_tuple_and_float32_ulp():
    reference = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    candidate = reference.copy()
    candidate[1] = np.nextafter(candidate[1], np.float32(np.inf))
    keys = np.asarray([[7, 1], [7, 2], [8, 0]], dtype=np.int64)

    record = _first_mismatch_record(reference, candidate, tuple_keys=keys)

    assert record is not None
    assert record["flat_index"] == 1
    assert record["recovar_rotation_row"] == 7
    assert record["recovar_translation_row"] == 2
    assert record["float32_ulp_distance"] == 1
    assert record["candidate_float32_bits"] - record["reference_float32_bits"] == 1


@pytest.mark.unit
def test_first_mismatch_record_returns_none_for_exact_arrays():
    values = np.asarray([-2.0, 0.0], dtype=np.float64)
    keys = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    assert _first_mismatch_record(values, values.copy(), tuple_keys=keys) is None


@pytest.mark.unit
def test_float32_ulp_distance_handles_negative_neighbors():
    value = np.float32(-2.0)
    neighbor = np.nextafter(value, np.float32(np.inf))
    assert _float32_ulp_distance(float(value), float(neighbor)) == 1


@pytest.mark.unit
def test_first_exact_boundary_uses_scientific_stage_order():
    stages = {
        "active_candidate_tuples": True,
        "raw_diff2": True,
        "orientation_log_prior": False,
        "translation_log_prior": False,
        "combined_log_weight_centered": False,
        "normalized_posterior_native_active": False,
        "significant_support": True,
    }
    assert _first_exact_boundary(stages) == "orientation_log_prior"


@pytest.mark.unit
def test_raw_diff2_terms_preserve_relion_float32_operation_order():
    reference = np.asarray([1.25 + 0.5j], dtype=np.complex64)
    shifted = np.asarray([-0.25 + 0.75j], dtype=np.complex64)
    weight = np.asarray([3.0], dtype=np.float32)
    expected = np.float32(
        np.float32(
            np.float32(np.float32(1.25) - np.float32(-0.25)) ** 2
            + np.float32(np.float32(0.5) - np.float32(0.75)) ** 2
        )
        * np.float32(0.5)
        * weight[0]
    )
    result = _raw_diff2_terms(reference, shifted, weight)
    assert result.dtype == np.float32
    assert result[0].view(np.uint32) == expected.view(np.uint32)


@pytest.mark.unit
def test_raw_diff2_terms_replay_relion_cuda_contracted_square_sum():
    diff_real = np.float32(-5.3695326)
    diff_imag = np.float32(5.811181)
    reference = np.asarray([diff_real + np.complex64(1j) * diff_imag], dtype=np.complex64)
    shifted = np.zeros(1, dtype=np.complex64)
    weight = np.asarray([2.0], dtype=np.float32)
    imag_squared = np.multiply(diff_imag, diff_imag, dtype=np.float32)
    contracted = np.asarray(
        np.float64(diff_real) * np.float64(diff_real) + np.float64(imag_squared),
        dtype=np.float32,
    )
    expected = np.multiply(
        np.multiply(contracted, np.float32(0.5), dtype=np.float32),
        weight,
        dtype=np.float32,
    )

    result = _raw_diff2_terms(reference, shifted, weight)

    assert result[0].view(np.uint32) == expected[0].view(np.uint32)


@pytest.mark.unit
def test_reduce_relion_fine_lanes_matches_fixed_tree():
    lanes = np.arange(256, dtype=np.float32)
    reduced, levels = _reduce_relion_fine_lanes(lanes)
    assert [level.size for level in levels] == [128, 64, 32, 16, 8, 4, 2, 1]
    assert reduced.view(np.uint32) == np.sum(lanes, dtype=np.float32).view(np.uint32)
