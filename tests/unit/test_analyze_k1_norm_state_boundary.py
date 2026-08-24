import numpy as np

from scripts.analyze_k1_norm_state_boundary import (
    _complex_scale,
    _load_native_factor_report,
    _parse_native_norm_operands,
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


def test_parse_native_norm_operands_filters_particle_and_decodes_hex(tmp_path):
    log = tmp_path / "relion.stderr"
    log.write_text(
        "RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=515 "
        "previous_norm=0x1p+0 previous_avg=0x1p+0 old_norm_over_avg=0x1p+0 "
        "wsum_norm=0x1.8p+3 sqrt_2_wsum=0x1.3988e1409212ep+2 "
        "new_norm=0x1.3988e1409212ep+2\n"
        "RELION_P1_NORMALIZATION_OPERANDS_V1 part_id=999 avg_norm=0x1p+0 "
        "particle_norm=0x1p+0 quotient=0x1p+0 quotient_f32_bits=3f800000\n"
        "RELION_P1_NORMALIZATION_OPERANDS_V1 part_id=515 avg_norm=0x1.ep+2 "
        "particle_norm=0x1p+3 quotient=0x1.ep-1 quotient_f32_bits=3f700000\n"
    )

    result = _parse_native_norm_operands(log, part_id=515)

    assert result["expectation"] == {
        "part_id": 515,
        "avg_norm_float64": 7.5,
        "particle_norm_float64": 8.0,
        "quotient_float64": 0.9375,
        "quotient_float32_bits": "0x3f700000",
    }
    assert len(result["updates"]) == 1
    assert result["updates"][0]["iteration"] == 1
    assert result["updates"][0]["wsum_norm"] == 12.0


def test_load_native_factor_report_requires_exact_source_aligned_factor(tmp_path):
    report = tmp_path / "factor.json"
    report.write_text(
        """{
  "schema": "recovar.em.k1_native_normalization_factor.v1",
  "source_index_zero_based": 68955,
  "recovered": {
    "factor_float32": 1.0047231912612915,
    "factor_float32_bits": "0x3f809ac5",
    "mismatch_count": 0
  }
}\n"""
    )

    result = _load_native_factor_report(report, source_index=68955)

    assert result["normalization_factor_float32"] == 1.0047231912612915
    assert result["normalization_factor_bits"] == "0x3f809ac5"
