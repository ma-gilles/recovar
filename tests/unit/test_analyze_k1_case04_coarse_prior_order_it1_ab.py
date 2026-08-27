import numpy as np
import pytest

from scripts.analyze_k1_case04_coarse_prior_order_it1_ab import ARRAY_STEMS, _compare, _metric


def test_metric_reports_exact_and_one_entry_float_delta():
    reference = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    exact = _metric(reference, reference.copy())
    assert exact["exact_equal"]
    assert exact["mismatch_count"] == 0

    candidate = reference.copy()
    candidate[-1] = np.nextafter(candidate[-1], np.float32(np.inf))
    changed = _metric(reference, candidate)
    assert not changed["exact_equal"]
    assert changed["mismatch_count"] == 1
    assert changed["mismatch_row_count"] == 1
    assert changed["max_abs"] > 0.0


def test_metric_counts_complex_rows_and_rejects_dtype_changes():
    reference = np.asarray([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1, 0] += np.complex64(1.0j)
    changed = _metric(reference, candidate)
    assert changed["mismatch_count"] == 1
    assert changed["mismatch_row_count"] == 1
    assert changed["max_abs"] == 1.0

    with pytest.raises(ValueError, match="shape or dtype"):
        _metric(reference, candidate.astype(np.complex128))


def test_compare_keeps_array_evidence_when_particle_state_is_unavailable(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    for stem in ARRAY_STEMS:
        values = np.asarray([1.0, 2.0], dtype=np.float32)
        np.save(reference_dir / stem, values)
        np.save(candidate_dir / stem, values)

    comparison = _compare(reference_dir, candidate_dir)

    assert all(metric["exact_equal"] for metric in comparison["arrays"].values())
    assert comparison["particle_state"]["status"] == "unavailable"
    assert len(comparison["particle_state"]["missing_paths"]) == 4
