from __future__ import annotations

import numpy as np

from scripts import analyze_em_k1_bpref_membership_cohort as analyzer


def _rotations(count: int) -> np.ndarray:
    values = np.zeros((count, 3, 3), dtype=np.float32)
    for index in range(count):
        values[index] = np.eye(3, dtype=np.float32)
        values[index, 0, 0] += np.float32(index)
    return values


def test_compare_particle_exact_membership_and_mass() -> None:
    rotations = _rotations(2)
    relion = np.asarray([[0.6, 0.3], [0.09, 0.01]], dtype=np.float32)
    recovar = relion.astype(np.float64)
    reconstruction = np.where(recovar >= 0.1, recovar, 0.0)
    report = analyzer.compare_particle(
        relion_rotations=rotations,
        relion_weights=relion,
        relion_significant_weight=0.1,
        relion_weight_norm=1.0,
        recovar_rotations=rotations,
        recovar_posterior=recovar,
        recovar_reconstruction=reconstruction,
    )
    assert report["candidate_sets_exact"]
    assert report["positive_rotation_sets_exact"]
    assert report["significant_sample_count_exact"]
    assert report["reconstruction_mass_gate_passed"]
    assert report["strict_particle_passed"]


def test_compare_particle_detects_translation_support_difference() -> None:
    rotations = _rotations(1)
    report = analyzer.compare_particle(
        relion_rotations=rotations,
        relion_weights=np.asarray([[0.6, 0.3]], dtype=np.float32),
        relion_significant_weight=0.1,
        relion_weight_norm=1.0,
        recovar_rotations=rotations,
        recovar_posterior=np.asarray([[0.6, 0.3]], dtype=np.float64),
        recovar_reconstruction=np.asarray([[0.6, 0.0]], dtype=np.float64),
    )
    assert report["positive_rotation_sets_exact"]
    assert not report["significant_sample_count_exact"]
    assert not report["strict_particle_passed"]


def test_compare_particle_detects_candidate_difference() -> None:
    report = analyzer.compare_particle(
        relion_rotations=_rotations(2),
        relion_weights=np.asarray([[0.6], [0.4]], dtype=np.float32),
        relion_significant_weight=0.1,
        relion_weight_norm=1.0,
        recovar_rotations=_rotations(1),
        recovar_posterior=np.asarray([[1.0]], dtype=np.float64),
        recovar_reconstruction=np.asarray([[1.0]], dtype=np.float64),
    )
    assert not report["candidate_sets_exact"]
    assert not report["positive_rotation_sets_exact"]
    assert report["relion_unmatched_candidate_count"] == 1


def test_compare_particle_detects_mass_difference() -> None:
    rotations = _rotations(1)
    report = analyzer.compare_particle(
        relion_rotations=rotations,
        relion_weights=np.asarray([[0.6, 0.4]], dtype=np.float32),
        relion_significant_weight=0.1,
        relion_weight_norm=1.0,
        recovar_rotations=rotations,
        recovar_posterior=np.asarray([[0.6, 0.2]], dtype=np.float64),
        recovar_reconstruction=np.asarray([[0.6, 0.2]], dtype=np.float64),
    )
    assert report["candidate_sets_exact"]
    assert report["positive_rotation_sets_exact"]
    assert report["significant_sample_count_exact"]
    assert not report["reconstruction_mass_gate_passed"]
