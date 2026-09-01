from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_em_real_kclass_initialmodel as audit
from scripts.summarize_em_completion_bench import shell_fsc

pytestmark = pytest.mark.unit


def _volume(seed: int, size: int = 16) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((size, size, size))


def test_fsc_assignment_recovers_swapped_classes_without_correlation():
    first = _volume(1)
    second = _volume(2)

    scores, _ = audit._pairwise_fsc_auc([first, second], [second, first])
    permutation = audit._best_class_permutation(scores)

    assert permutation == (1, 0)
    np.testing.assert_allclose(scores[np.arange(2), permutation], 1.0, atol=1e-12)


def test_pairwise_fsc_cache_is_exact_and_transforms_each_k4_map_once(monkeypatch):
    candidate = [_volume(seed) for seed in range(10, 14)]
    reference = [_volume(seed) for seed in range(20, 24)]
    expected_curves = {
        (candidate_class, reference_class): np.asarray(
            shell_fsc(candidate_map, reference_map),
            dtype=np.float64,
        )
        for candidate_class, candidate_map in enumerate(candidate)
        for reference_class, reference_map in enumerate(reference)
    }
    expected_scores = np.asarray(
        [
            [
                audit.normalized_fsc_auc(expected_curves[(candidate_class, reference_class)])
                for reference_class in range(4)
            ]
            for candidate_class in range(4)
        ],
        dtype=np.float64,
    )
    original_fftn = np.fft.fftn
    original_meshgrid = np.meshgrid
    fft_shapes: list[tuple[int, ...]] = []
    meshgrid_calls = 0

    def counted_fftn(array):
        fft_shapes.append(np.asarray(array).shape)
        return original_fftn(array)

    def counted_meshgrid(*args, **kwargs):
        nonlocal meshgrid_calls
        meshgrid_calls += 1
        return original_meshgrid(*args, **kwargs)

    monkeypatch.setattr(audit.np.fft, "fftn", counted_fftn)
    monkeypatch.setattr(audit.np, "meshgrid", counted_meshgrid)

    actual_scores, actual_curves = audit._pairwise_fsc_auc(candidate, reference)

    assert fft_shapes == [(16, 16, 16)] * 8
    assert meshgrid_calls == 1
    np.testing.assert_array_equal(actual_scores, expected_scores)
    assert actual_curves.keys() == expected_curves.keys()
    for pair, expected in expected_curves.items():
        np.testing.assert_array_equal(actual_curves[pair], expected)


def test_assignment_accuracy_applies_map_permutation():
    candidate = np.asarray([0, 0, 1, 1, 1])
    reference = np.asarray([1, 1, 0, 0, 0])

    assert audit._class_assignment_accuracy(candidate, reference, (1, 0)) == 1.0
    assert audit._class_assignment_accuracy(candidate, reference, (0, 1)) == 0.0


def test_population_flags_underfilled_class_without_hiding_counts():
    labels = np.asarray([0] * 90 + [1] * 8 + [2] + [3])

    population = audit._population(labels, 4, minimum_class_fraction=0.02)

    assert population["counts"] == [90, 8, 1, 1]
    assert population["fractions"] == [0.9, 0.08, 0.01, 0.01]
    assert population["collapsed_class_ids_one_based"] == [3, 4]
    assert population["collapse_detected"] is True


def test_unassigned_iteration_zero_is_explicit_and_not_a_false_collapse(tmp_path: Path):
    star = tmp_path / "particles.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
        "1@particles.mrcs 0\n2@particles.mrcs 0\n"
    )

    labels = np.fromiter(audit._assignments_by_image(star, 4).values(), dtype=np.int64)
    population = audit._population(labels, 4, minimum_class_fraction=0.01)

    assert labels.tolist() == [-1, -1]
    assert population["assigned_particles"] == 0
    assert population["unassigned_particles"] == 2
    assert population["minimum_fraction"] is None
    assert population["collapse_detected"] is False


def test_assignment_star_rejects_duplicate_image_identity(tmp_path: Path):
    star = tmp_path / "particles.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
        "1@particles.mrcs 1\n1@particles.mrcs 2\n"
    )

    with pytest.raises(audit.AuditError, match="duplicate image identities"):
        audit._assignments_by_image(star, 2)


def test_matched_assignments_exposes_identity_drift_and_uses_full_populations(tmp_path: Path):
    candidate = tmp_path / "candidate.star"
    reference = tmp_path / "reference.star"
    header = "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
    candidate.write_text(header + "1@particles.mrcs 1\n2@particles.mrcs 1\n")
    reference.write_text(header + "1@particles.mrcs 1\n3@particles.mrcs 2\n")

    result = audit._matched_assignments(candidate, reference, (0, 1), 0.01)

    assert result["image_identity_sets_match"] is False
    assert result["candidate_only_particle_count"] == 1
    assert result["reference_only_particle_count"] == 1
    assert result["candidate_population"]["counts"] == [2, 0]
    assert result["reference_population"]["counts"] == [1, 1]


def test_matched_unassigned_status_must_agree(tmp_path: Path):
    candidate = tmp_path / "candidate.star"
    reference = tmp_path / "reference.star"
    header = "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
    candidate.write_text(header + "1@particles.mrcs 0\n2@particles.mrcs 1\n")
    reference.write_text(header + "1@particles.mrcs 1\n2@particles.mrcs 1\n")

    result = audit._matched_assignments(candidate, reference, (0, 1), 0.01)

    assert result["assignment_status_match"] is False
    assert result["common_assigned_particles"] == 1
    assert result["accuracy"] == 1.0


def test_class_score_matrix_must_be_square_and_finite():
    with pytest.raises(ValueError, match="square"):
        audit._best_class_permutation(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        audit._best_class_permutation(np.asarray([[1.0, np.nan], [0.0, 1.0]]))
