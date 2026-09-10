from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_vdam_kclass_trajectory as audit

pytestmark = pytest.mark.unit


def _volume(seed: int, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((size, size, size))


def test_fsc_assignment_recovers_swapped_classes_without_correlation():
    first = _volume(1)
    second = _volume(2)

    scores, _ = audit._pairwise_fsc_auc([first, second], [second, first])
    permutation = audit._best_class_permutation(scores)

    assert permutation == (1, 0)
    np.testing.assert_allclose(scores[np.arange(2), permutation], 1.0, atol=1e-12)


def test_assignment_accuracy_applies_map_permutation():
    candidate = np.asarray([0, 0, 1, 1, 1])
    reference = np.asarray([1, 1, 0, 0, 0])

    assert audit._class_assignment_accuracy(candidate, reference, (1, 0)) == 1.0
    assert audit._class_assignment_accuracy(candidate, reference, (0, 1)) == 0.0


def test_class_score_matrix_must_be_square_and_finite():
    with pytest.raises(ValueError, match="square"):
        audit._best_class_permutation(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        audit._best_class_permutation(np.asarray([[1.0, np.nan], [0.0, 1.0]]))


def test_trajectory_audit_supports_k1(monkeypatch, tmp_path):
    volume = _volume(3)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(audit, "_load_relion_volume", lambda _path: volume)
    monkeypatch.setattr(
        audit,
        "_matched_assignments",
        lambda _candidate, _reference, permutation: {
            "common_particles": 4,
            "common_assigned_particles": 4,
            "candidate_particles": 4,
            "reference_particles": 4,
            "accuracy": 1.0,
        },
    )

    report, shellwise = audit.audit_trajectory(
        candidate_dir=tmp_path / "candidate",
        reference_dir=tmp_path / "reference",
        K=1,
        checkpoints=(90,),
        minimum_fsc_auc=0.999,
        minimum_assignment_accuracy=0.999,
    )

    assert report["result"] == "pass"
    assert report["iterations"][0]["permutation_candidate_to_reference"] == [0]
    assert report["minimum_matched_fsc_auc"] == pytest.approx(1.0)
    assert len(shellwise) == 1


def test_trajectory_audit_rejects_zero_classes(tmp_path):
    with pytest.raises(ValueError, match="K >= 1"):
        audit.audit_trajectory(
            candidate_dir=tmp_path / "candidate",
            reference_dir=tmp_path / "reference",
            K=0,
            checkpoints=(1,),
            minimum_fsc_auc=0.999,
            minimum_assignment_accuracy=0.999,
        )


def _assignment_star(path, rows):
    path.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
        + "".join(f"{image} {label}\n" for image, label in rows)
    )
    return path


@pytest.mark.parametrize("side", ["candidate", "reference"])
@pytest.mark.parametrize(
    "bad_rows",
    [
        [("1@stack.mrcs", 1)],  # Silent intersection used to discard the other row.
        [("1@stack.mrcs", 1), ("2@stack.mrcs", 2), ("1@stack.mrcs", 1)],
        [("1@stack.mrcs", 1), ("2@stack.mrcs", 5)],
        [("1@stack.mrcs", 1), ("2@stack.mrcs", -1)],
        [("1@stack.mrcs", 1), ("2@stack.mrcs", 2.5)],
    ],
)
def test_assignment_audit_rejects_incomplete_or_invalid_rows(tmp_path, side, bad_rows):
    paths = {}
    for arm in ("candidate", "reference"):
        rows = bad_rows if arm == side else [("1@stack.mrcs", 1), ("2@stack.mrcs", 2)]
        paths[arm] = _assignment_star(tmp_path / f"{arm}.star", rows)
    with pytest.raises(audit.AuditError):
        audit._matched_assignments(paths["candidate"], paths["reference"], (0, 1, 2, 3))


def test_assignment_audit_keeps_unassigned_rows_and_matches_by_identity(tmp_path):
    candidate = _assignment_star(tmp_path / "candidate.star", [("2@s", 1), ("1@s", 2), ("3@s", 0)])
    reference = _assignment_star(tmp_path / "reference.star", [("1@s", 1), ("2@s", 2), ("3@s", 0)])
    assert audit._matched_assignments(candidate, reference, (1, 0)) == {
        "common_particles": 3,
        "common_assigned_particles": 2,
        "candidate_particles": 3,
        "reference_particles": 3,
        "accuracy": 1.0,
    }
