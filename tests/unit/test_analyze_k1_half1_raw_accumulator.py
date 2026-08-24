from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_half1_raw_accumulator import (
    _centered_real_inner_correlation,
    _intervention_projection_on_gap,
    _load_native_bpref,
    _load_recovar,
    _metric,
    _paired_downsampled_residual_structure,
    _paired_raw_residual_structure,
    _raw_accumulator_region_metrics,
    _require_matching_bpref_stage,
)


def _write_bpref(path: Path, *, schema: str = "recovar-bpref-prejoin-v2") -> None:
    np.savez(
        path,
        schema=np.asarray(schema),
        iteration=np.int32(1),
        current_size=np.int32(56),
        padding_factor=np.int32(2),
        grid_size=np.int32(128),
        volume_shape=np.asarray((128, 128, 128), dtype=np.int32),
        mstep_accumulator_shape=np.asarray((123, 123, 123), dtype=np.int32),
        Ft_y_0=np.asarray([1 + 2j], dtype=np.complex64),
        Ft_y_1=np.asarray([3 + 4j], dtype=np.complex64),
        Ft_ctf_0=np.asarray([5], dtype=np.float32),
        Ft_ctf_1=np.asarray([6], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("half", "expected_numerator", "expected_weight"),
    ((1, 1 + 2j, 5.0), (2, 3 + 4j, 6.0)),
)
def test_load_recovar_selects_requested_half(
    tmp_path: Path,
    half: int,
    expected_numerator: complex,
    expected_weight: float,
) -> None:
    path = tmp_path / "prejoin.npz"
    _write_bpref(path)

    loaded = _load_recovar(path, half=half)

    assert loaded["half"] == half
    assert loaded["numerator"].item() == expected_numerator
    assert loaded["weight"].item() == expected_weight
    assert loaded["stage"] == "pre_lowres_join"


def test_load_recovar_accepts_post_lowres_join_dump(tmp_path: Path) -> None:
    path = tmp_path / "postjoin.npz"
    _write_bpref(path, schema="recovar-bpref-accum-v2")

    loaded = _load_recovar(path, half=1)

    assert loaded["stage"] == "post_lowres_join"
    assert loaded["numerator"].item() == 1 + 2j


def test_load_recovar_rejects_invalid_half(tmp_path: Path) -> None:
    path = tmp_path / "prejoin.npz"
    _write_bpref(path)

    with pytest.raises(ValueError, match="half must be 1 or 2"):
        _load_recovar(path, half=0)


def test_load_native_bpref_state_v1(tmp_path: Path) -> None:
    path = tmp_path / "bpref_data.bin"
    shape = np.asarray((3, 3, 2), dtype=np.int64)
    values = np.arange(18, dtype=np.float64).astype(np.complex128).reshape(3, 3, 2)
    with path.open("wb") as stream:
        shape.tofile(stream)
        values.tofile(stream)

    loaded_shape, loaded_values = _load_native_bpref(path, value_dtype=np.complex128)

    assert np.array_equal(loaded_shape, shape)
    assert np.array_equal(loaded_values, values)


def test_metric_reports_exact_first_mismatch_telemetry() -> None:
    source = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    target = np.asarray([1.0, 2.5, 4.0], dtype=np.float32)

    result = _metric(source, target, allow_sign=False)

    assert result["exact_equal"] is False
    assert result["mismatch_count"] == 1
    assert result["first_mismatch_flat_index"] == 1
    assert result["max_absolute"] == pytest.approx(0.5)


def test_require_matching_bpref_stage_rejects_cross_stage_comparison() -> None:
    _require_matching_bpref_stage("post_lowres_join", "post_lowres_join")

    with pytest.raises(ValueError, match="BPref stage mismatch"):
        _require_matching_bpref_stage("pre_lowres_join", "post_lowres_join")


def test_centered_real_inner_correlation_handles_complex_values_and_mask() -> None:
    source = np.asarray([1 + 2j, 2 + 1j, 4 - 3j, 8 + 2j])
    target = 3 * source + (7 - 2j)

    assert _centered_real_inner_correlation(source, target) == pytest.approx(1.0)
    assert _centered_real_inner_correlation(
        source, target, mask=np.asarray([True, True, True, False])
    ) == pytest.approx(1.0)


def test_paired_raw_residual_structure_separates_signal_and_error_coherence() -> None:
    native_numerator1 = np.asarray([1 + 1j, 2 + 0j, 4 - 1j, 8 + 2j])
    native_numerator2 = 2 * native_numerator1 + (3 + 1j)
    native_denominator1 = np.asarray([1.0, 2.0, 4.0, 8.0])
    native_denominator2 = 2 * native_denominator1 + 3
    numerator_error1 = np.asarray([1 + 0j, -1 + 1j, 2 - 1j, -2 + 0j]) * 1e-4
    numerator_error2 = np.asarray([-2 + 1j, 1 + 0j, 1 - 2j, 0 + 1j]) * 1e-4
    denominator_error1 = np.asarray([1.0, -1.0, 2.0, -2.0]) * 1e-4
    denominator_error2 = np.asarray([-2.0, 1.0, 1.0, 0.0]) * 1e-4

    result = _paired_raw_residual_structure(
        -native_numerator1 - numerator_error1,
        native_numerator1,
        native_denominator1 + denominator_error1,
        native_denominator1,
        -native_numerator2 - numerator_error2,
        native_numerator2,
        native_denominator2 + denominator_error2,
        native_denominator2,
    )

    assert result["numerator_sign_applied_to_recovar"] == [-1, -1]
    assert result["native_signal"]["numerator_centered_correlation"] == pytest.approx(1.0)
    assert result["native_signal"]["denominator_centered_correlation"] == pytest.approx(1.0)
    assert abs(
        result["recovar_minus_native_residual"]["numerator_centered_correlation"]
    ) < 0.8
    assert abs(
        result["recovar_minus_native_residual"]["denominator_centered_correlation"]
    ) < 0.8


def test_paired_downsampled_residual_structure_reports_average_numerator_and_weight() -> None:
    native_average1 = np.asarray([1 + 1j, 2 + 0j, 4 - 1j, 8 + 2j])
    native_average2 = 2 * native_average1 + (3 + 1j)
    native_weight1 = np.asarray([1.0, 2.0, 4.0, 8.0])
    native_weight2 = np.asarray([2.0, 4.0, 8.0, 16.0])
    average_error1 = np.asarray([1 + 0j, -1 + 1j, 2 - 1j, -2 + 0j]) * 1e-4
    average_error2 = np.asarray([-2 + 1j, 1 + 0j, 1 - 2j, 0 + 1j]) * 1e-4
    weight_error1 = np.asarray([1.0, -1.0, 2.0, -2.0]) * 1e-4
    weight_error2 = np.asarray([-2.0, 1.0, 1.0, 0.0]) * 1e-4

    result = _paired_downsampled_residual_structure(
        -native_average1 - average_error1,
        native_weight1 + weight_error1,
        native_average1,
        native_weight1,
        -native_average2 - average_error2,
        native_weight2 + weight_error2,
        native_average2,
        native_weight2,
    )

    assert set(result) == {"policy", "average", "numerator", "denominator"}
    assert result["average"]["sign_applied_to_recovar"] == [-1, -1]
    assert result["denominator"]["sign_applied_to_recovar"] == [1, 1]
    for field in ("average", "numerator", "denominator"):
        assert np.isfinite(result[field]["residual_centered_correlation"])


def test_raw_accumulator_region_metrics_localizes_x0_residual() -> None:
    shape = (3, 2, 2)
    native_numerator = np.arange(1, 13, dtype=np.float64).astype(np.complex128)
    native_denominator = np.arange(11, 23, dtype=np.float64)
    recovar_numerator = -native_numerator.copy()
    recovar_denominator = native_denominator.copy()
    x0 = np.zeros(shape, dtype=bool)
    x0[shape[0] // 2, :, :] = True
    x0 = x0.reshape(-1)
    recovar_numerator[x0] -= 2.0
    recovar_denominator[x0] += 3.0

    result = _raw_accumulator_region_metrics(
        recovar_numerator,
        native_numerator,
        recovar_denominator,
        native_denominator,
        accumulator_shape=shape,
    )

    assert result["x0_public_axis"] == 0
    assert result["x0_public_index"] == 1
    assert result["coordinate_count"] == {"x0": 4, "off_x0": 8}
    for field in ("numerator", "denominator"):
        assert result["regions"][field]["x0"]["residual_l2_fraction_of_total"] == pytest.approx(1.0)
        assert result["regions"][field]["off_x0"]["exact_equal"] is True


def test_intervention_projection_reports_exact_gap_closure() -> None:
    native = np.asarray([1.0, 2.0, 4.0, 8.0])
    recovar = native + np.asarray([0.5, -0.25, 0.125, -0.0625])

    result = _intervention_projection_on_gap(
        recovar,
        native,
        recovar.copy(),
        allow_sign=False,
    )

    assert result["real_inner_product_cosine"] == pytest.approx(1.0)
    assert result["least_squares_intervention_scale_to_gap"] == pytest.approx(1.0)
    assert result["gap_norm_ratio_after_full_intervention"] == pytest.approx(0.0)
    assert result["squared_gap_fraction_removed_by_full_intervention"] == pytest.approx(1.0)
