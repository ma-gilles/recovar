import numpy as np

from scripts.analyze_k1_exact_ppref_coarse_boundary import (
    _centered_stats,
    _posterior_metrics,
    _top_mass,
)


def test_centered_stats_ignore_additive_score_constant():
    reference = np.asarray([2.0, 4.0, -3.0], dtype=np.float32)
    candidate = reference + np.float32(11.0)
    report = _centered_stats(candidate, reference, np.ones(3, dtype=bool))
    assert report == {"median_abs": 0.0, "p95_abs": 0.0, "max_abs": 0.0, "rms": 0.0}


def test_posterior_metrics_normalize_inputs():
    report = _posterior_metrics(np.asarray([1.0, 3.0]), np.asarray([2.0, 6.0]))
    assert report == {"total_variation": 0.0, "max_abs": 0.0}


def test_posterior_metrics_flattens_equivalent_candidate_domains():
    candidate = np.asarray([0.1, 0.2, 0.3, 0.4])
    reference = candidate.reshape(2, 2)
    assert _posterior_metrics(candidate, reference) == {
        "total_variation": 0.0,
        "max_abs": 0.0,
    }


def test_top_mass_uses_stable_largest_values():
    assert np.isclose(_top_mass(np.asarray([0.1, 0.6, 0.3]), 2), 0.9)
