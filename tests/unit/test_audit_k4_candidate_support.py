from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_k4_relion_recovar_candidate_support as auditor

ROTATION_DTYPE = np.dtype(
    [
        ("orientation_class_key", "<u8"),
        ("oversampled_rotation", "<u8"),
    ]
)
ROW_DTYPE = np.dtype([("orientation_local", "<u8")])


def _rotations(class_keys, oversampled):
    rows = np.zeros(len(class_keys), dtype=ROTATION_DTYPE)
    rows["orientation_class_key"] = class_keys
    rows["oversampled_rotation"] = oversampled
    return rows


def test_relion_global_indices_remove_class_offset():
    rotations = _rotations([4, 4, 5, 5], [0, 1, 0, 1])

    actual = auditor._relion_global_indices(
        rotations,
        class_index=1,
        coarse_rotation_count=4,
        oversampling_factor=2,
    )

    np.testing.assert_array_equal(actual, np.array([0, 1, 2, 3]))


def test_relion_global_indices_reject_wrong_class():
    rotations = _rotations([3], [0])

    with pytest.raises(RuntimeError, match="outside the requested class"):
        auditor._relion_global_indices(
            rotations,
            class_index=1,
            coarse_rotation_count=4,
            oversampling_factor=2,
        )


def test_particle_report_localizes_complete_expansion_to_coarse_parents():
    rotations = _rotations(
        [4, 4, 5, 5, 6, 6],
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
        "candidate_globals": np.array([0, 1, 4, 5]),
        "contributor_globals": np.array([4]),
    }

    report = auditor._particle_report(
        artifact,
        recovar,
        "control",
        class_index=1,
        coarse_rotation_count=4,
        oversampling_factor=2,
    )

    assert report["relion_is_complete_oversampled_expansion"]
    assert report["recovar_is_complete_oversampled_expansion"]
    assert report["relion_coarse_parent_count"] == 3
    assert report["recovar_coarse_parent_count"] == 2
    assert report["coarse_parent_overlap_count"] == 2
    assert not report["coarse_parent_sets_exact"]
    assert report["all_relion_contributors_in_recovar_candidates"]
    assert report["all_recovar_contributors_in_relion_candidates"]
