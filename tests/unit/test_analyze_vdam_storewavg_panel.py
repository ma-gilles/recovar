from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_vdam_storewavg_panel import (
    _native_prefixes,
    _original_index_from_image_name,
    _pooled_relative_l2,
    _quantiles,
)

pytestmark = pytest.mark.unit


def test_original_index_from_image_name_converts_one_based_relion_identity():
    assert _original_index_from_image_name("1060@particles.128.mrcs") == 1059


def test_original_index_from_image_name_rejects_invalid_identity():
    with pytest.raises(ValueError, match="invalid RELION image identity"):
        _original_index_from_image_name("particles.128.mrcs")
    with pytest.raises(ValueError, match="must be positive"):
        _original_index_from_image_name("0@particles.128.mrcs")


def test_quantiles_reports_distribution_and_mean():
    result = _quantiles([0.0, 1.0, 2.0, 3.0, 4.0])
    assert result == {
        "min": 0.0,
        "p50": 2.0,
        "p90": pytest.approx(3.6),
        "p99": pytest.approx(3.96),
        "max": 4.0,
        "mean": 2.0,
    }


def test_pooled_relative_l2_uses_summed_squared_norms():
    assert _pooled_relative_l2(25.0, 1.0) == pytest.approx(0.2)


def test_native_prefixes_separates_complete_and_racy_partial_captures(tmp_path):
    required = {
        "orientation_num.bin",
        "translation_num.bin",
        "sorted_weights.bin",
        "sum_weight.bin",
        "significant_weight.bin",
        "eulers.bin",
        "trans_xyz.bin",
        "ctfs.bin",
    }
    for suffix in required:
        (tmp_path / f"img0_part7_storeWavg_{suffix}").touch()
    (tmp_path / "img0_part8_storeWavg_sorted_weights.bin").touch()

    complete, incomplete = _native_prefixes(tmp_path)

    assert complete == {7: "img0_part7_storeWavg_"}
    assert set(incomplete) == {8}
    assert "orientation_num.bin" in incomplete[8]


def test_production_score_gradient_override_uses_supplied_native_probabilities():
    from scripts.analyze_vdam_storewavg_boundary import _production_score_gradient_rows

    posterior = np.asarray([[[0.25, 0.75], [0.5, 0.5]]], dtype=np.float32)
    mask = np.ones_like(posterior, dtype=bool)
    override = np.asarray([[1.0, 0.0], [0.0, 0.25]], dtype=np.float32)
    shifted = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    ctf2 = np.asarray([2.0, 3.0], dtype=np.float32)
    projections = np.asarray(
        [[0.5 + 0.25j, 1 + 0.5j], [2 + 1j, 3 + 1.5j]], dtype=np.complex64
    )

    data, weight, probabilities = _production_score_gradient_rows(
        {
            "posterior": posterior,
            "reconstruction_sample_mask": mask,
            "debug_shifted_recon": shifted,
            "debug_ctf2_over_nv_recon": ctf2,
            "debug_proj_for_recon": projections,
        },
        reconstruction_probs_override=override,
    )

    mass = override.sum(axis=-1, dtype=np.float32)
    expected_weight = mass[:, None] * ctf2[None, :]
    expected_data = override @ shifted - projections * expected_weight
    np.testing.assert_array_equal(probabilities, override)
    np.testing.assert_array_equal(weight, expected_weight.astype(np.float32))
    np.testing.assert_allclose(data, expected_data.astype(np.complex64), rtol=1.0e-7, atol=1.0e-7)
