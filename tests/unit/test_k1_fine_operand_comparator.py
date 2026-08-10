import json

import numpy as np

from scripts.compare_k1_relion_recovar_fine_operands import (
    _expanded_score_components,
    _json_default,
    _metric,
    _score_terms,
)
from scripts.validate_relion_fine_operand_capture import (
    _cuda_fine_contribution,
    _cuda_fine_production_lanes,
    _reduce_lanes,
    _replay_lanes,
)


def test_score_terms_replay_cuda_contribution_and_lane_tree():
    reference = np.asarray([1 + 2j, -3 + 0.5j, 0.25 - 4j], dtype=np.complex64)
    shifted = np.asarray([-2 + 1j, 0.5 - 1j, 1.25 + 2j], dtype=np.complex64)
    corr = np.asarray([0.5, 2.0, 0.125], dtype=np.float32)
    sum_init = np.float32(0.75)

    result = _score_terms(reference, shifted, corr, sum_init)
    contribution = _cuda_fine_contribution(
        np.subtract(reference.real, shifted.real, dtype=np.float32),
        np.subtract(reference.imag, shifted.imag, dtype=np.float32),
        corr,
    )
    lanes = _replay_lanes(contribution)

    assert np.array_equal(result["contribution"], contribution)
    production_lanes = _cuda_fine_production_lanes(
        np.subtract(reference.real, shifted.real, dtype=np.float32),
        np.subtract(reference.imag, shifted.imag, dtype=np.float32),
        corr,
    )
    assert np.array_equal(result["replay_lanes"], lanes)
    assert np.array_equal(result["production_lanes"], production_lanes)
    assert result["replay_raw"] == np.add(
        _reduce_lanes(lanes), sum_init, dtype=np.float32
    )
    assert result["production_raw"] == np.add(
        _reduce_lanes(production_lanes), sum_init, dtype=np.float32
    )


def test_metric_records_first_complex_mismatch():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] = np.complex64(3 + 5j)

    report = _metric(reference, candidate)

    assert not report["exact_equal"]
    assert report["mismatch_count"] == 1
    assert report["first_mismatch"]["flat_index"] == 1
    assert report["first_mismatch"]["reference"] == {"real": 3.0, "imag": 4.0}
    assert report["first_mismatch"]["candidate"] == {"real": 3.0, "imag": 5.0}


def test_json_default_encodes_numpy_scalars_and_arrays():
    value = {"scalar": np.int64(7), "array": np.asarray([1, 2], dtype=np.int32)}
    assert json.loads(json.dumps(value, default=_json_default)) == {
        "scalar": 7,
        "array": [1, 2],
    }


def test_expanded_score_components_recombine_squared_difference():
    reference = np.asarray([1 + 2j, -0.5 + 0.25j], dtype=np.complex64)
    shifted = np.asarray([-2 + 1j, 0.75 - 1j], dtype=np.complex64)
    corr = np.asarray([0.5, 3.0], dtype=np.float32)
    sum_init = np.float32(0.125)

    components, _ = _expanded_score_components(reference, shifted, corr, sum_init)
    expected = np.sum(
        0.5
        * corr.astype(np.float64)
        * np.abs(reference.astype(np.complex128) - shifted.astype(np.complex128)) ** 2,
        dtype=np.float64,
    ) + np.float64(sum_init)

    assert np.isclose(components["total"], expected, rtol=0, atol=1e-14)
