from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_fine_capture_ab import ORDERED_FIELDS, analyze


def _write_capture(path: Path, *, changed_field: str | None = None) -> None:
    values: dict[str, np.ndarray] = {
        "iteration": np.asarray(1, dtype=np.int64),
        "half": np.asarray(1, dtype=np.int64),
        "original_index": np.asarray(94083, dtype=np.int64),
        "current_size": np.asarray(56, dtype=np.int64),
        "n_fine_trans": np.asarray(2, dtype=np.int64),
    }
    for index, field in enumerate(ORDERED_FIELDS):
        if field in {"candidate_mask", "reconstruction_mask"}:
            value = np.asarray([[True, True], [True, False]], dtype=bool)
        elif field in {"oversampled_rot_indices", "parent_map", "window_indices", "recon_window_indices"}:
            value = np.arange(4, dtype=np.int32).reshape(2, 2)
        elif field in {"fine_translations"}:
            value = np.arange(4, dtype=np.float32).reshape(2, 2)
        elif field == "rotations":
            value = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        elif field in {"scores_pre_prior", "scores_with_prior", "probs", "reconstruction_probs"}:
            value = np.asarray([[0.1, 0.4], [0.3, 0.0]], dtype=np.float32)
        else:
            value = np.full((2, 2), index + 1, dtype=np.float32)
        if field == changed_field:
            value = value.copy()
            value.reshape(-1)[0] = False if value.dtype == bool else value.reshape(-1)[0] + 1
        values[field] = value
    np.savez(path, **values)


@pytest.mark.parametrize(
    ("changed_field", "expected"),
    [
        (None, None),
        ("direct_score_input", "direct_score_input"),
        ("raw_operand_corr_img_score", "raw_operand_corr_img_score"),
        ("probs", "probs"),
    ],
)
def test_fine_capture_ab_reports_first_ordered_difference(tmp_path, changed_field, expected):
    control = tmp_path / "control.npz"
    candidate = tmp_path / "candidate.npz"
    _write_capture(control)
    _write_capture(candidate, changed_field=changed_field)

    report = analyze(control_path=control, candidate_path=candidate)

    assert report["first_non_bit_exact_field"] == expected
    assert report["summary"]["control_candidate_count"] == 3


def test_fine_capture_ab_fails_closed_on_missing_field(tmp_path):
    control = tmp_path / "control.npz"
    candidate = tmp_path / "candidate.npz"
    _write_capture(control)
    _write_capture(candidate)
    with np.load(candidate, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files if name != "probs"}
    np.savez(candidate, **values)

    with pytest.raises(ValueError, match="missing ordered fields:.*probs"):
        analyze(control_path=control, candidate_path=candidate)


def test_fine_capture_ab_can_allow_only_iteration_context_mismatch(tmp_path):
    control = tmp_path / "control.npz"
    candidate = tmp_path / "candidate.npz"
    _write_capture(control)
    _write_capture(candidate)
    with np.load(candidate, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["iteration"] = np.asarray(2, dtype=np.int64)
    np.savez(candidate, **values)

    with pytest.raises(ValueError, match="capture scalar iteration differs"):
        analyze(control_path=control, candidate_path=candidate)

    report = analyze(
        control_path=control,
        candidate_path=candidate,
        allow_iteration_mismatch=True,
    )
    assert report["first_non_bit_exact_field"] is None
    assert report["iteration_mismatch_allowed"] is True
    assert report["scalar_context"]["iteration"] == {
        "control": 1,
        "candidate": 2,
    }
