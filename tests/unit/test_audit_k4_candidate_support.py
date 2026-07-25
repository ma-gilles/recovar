from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_k4_relion_recovar_candidate_support as auditor

ROTATION_DTYPE = np.dtype(
    [
        ("orientation_class_key", "<u8"),
        ("oversampled_rotation", "<u8"),
        ("matrix", "<f4", (9,)),
    ]
)
ROW_DTYPE = np.dtype([("orientation_local", "<u8")])


def _rotations(class_keys, oversampled):
    rows = np.zeros(len(class_keys), dtype=ROTATION_DTYPE)
    rows["orientation_class_key"] = class_keys
    rows["oversampled_rotation"] = oversampled
    return rows


def test_relion_global_indices_remove_class_offset_and_canonicalize_order():
    rotations = _rotations([144, 144, 145, 145], [0, 1, 0, 1])

    actual = auditor._relion_global_indices(
        rotations,
        class_index=1,
        coarse_rotation_count=144,
        oversampling_factor=2,
        healpix_order=0,
    )

    np.testing.assert_array_equal(actual, np.array([0, 1, 24, 25]))


def test_relion_global_indices_reject_wrong_class():
    rotations = _rotations([143], [0])

    with pytest.raises(RuntimeError, match="outside the requested class"):
        auditor._relion_global_indices(
            rotations,
            class_index=1,
            coarse_rotation_count=144,
            oversampling_factor=2,
            healpix_order=0,
        )


def test_particle_report_localizes_complete_expansion_to_coarse_parents():
    rotations = _rotations(
        [144, 144, 145, 145, 146, 146],
        [0, 1, 0, 1, 0, 1],
    )
    rows = np.zeros(1, dtype=ROW_DTYPE)
    rows["orientation_local"] = 4
    artifact = SimpleNamespace(
        stack_index=17,
        rotations=rotations,
        rows=rows,
    )
    recovar = {
        "candidate_globals": np.array([0, 1, 48, 49]),
        "contributor_globals": np.array([48]),
    }

    report = auditor._particle_report(
        artifact,
        recovar,
        "control",
        class_index=1,
        coarse_rotation_count=144,
        oversampling_factor=2,
        healpix_order=0,
    )

    assert report["relion_is_complete_oversampled_expansion"]
    assert report["recovar_is_complete_oversampled_expansion"]
    assert report["relion_coarse_parent_count"] == 3
    assert report["recovar_coarse_parent_count"] == 2
    assert report["coarse_parent_overlap_count"] == 2
    assert not report["coarse_parent_sets_exact"]
    assert report["all_relion_contributors_in_recovar_candidates"]
    assert report["all_recovar_contributors_in_relion_candidates"]
    assert report["contributor_sets_exact"]
    assert report["contributor_overlap_count"] == 1


def test_relion_geometry_gate_requires_transposed_capture_convention():
    expected = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    rotations = _rotations([144, 144], [0, 1])
    rotations["matrix"] = expected.transpose(0, 2, 1).reshape(2, 9)
    artifact = SimpleNamespace(rotations=rotations)

    actual = auditor._relion_geometry_max_abs(
        [artifact],
        class_index=1,
        coarse_rotation_count=144,
        oversampling_factor=2,
        expected_relion_fine_rotations=expected,
    )

    assert actual == 0.0


def test_sampling_perturbation_mismatch_fails_closed():
    status, classification, comparable = auditor._classify_support(
        all_complete=True,
        any_parent_difference=True,
        relion_random_perturbation=-0.12306,
        recovar_random_perturbation=0.096421,
        perturbation_tolerance=5e-7,
    )

    assert status == "invalid_comparison"
    assert classification == "incomparable_sampling_perturbation_precludes_cross_engine_support_claim"
    assert not comparable


def test_classification_localizes_fine_rotation_contributor_difference():
    status, classification, comparable = auditor._classify_support(
        all_complete=True,
        any_parent_difference=False,
        all_fine_candidate_sets_exact=True,
        any_contributor_difference=True,
        relion_random_perturbation=-0.12306,
        recovar_random_perturbation=-0.12306,
        perturbation_tolerance=5e-7,
    )

    assert status == "complete"
    assert classification == "fine_rotation_contributor_support_difference_after_candidate_generation"
    assert comparable


def test_classification_reports_exact_candidate_and_contributor_support():
    status, classification, comparable = auditor._classify_support(
        all_complete=True,
        any_parent_difference=False,
        all_fine_candidate_sets_exact=True,
        any_contributor_difference=False,
        relion_random_perturbation=-0.12306,
        recovar_random_perturbation=-0.12306,
        perturbation_tolerance=5e-7,
    )

    assert status == "complete"
    assert classification == "candidate_and_rotation_contributor_support_exact"
    assert comparable
