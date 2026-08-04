from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_em_k4_joint_f32_support_counterfactual import build_report


def _write_capture(
    root: Path,
    class_one_based: int,
    scores: list[float],
    support: list[bool],
) -> None:
    score_array = np.asarray(scores, dtype=np.float32)[None, :]
    support_array = np.asarray(support, dtype=bool)[None, :]
    np.savez(
        root / f"pass2_orig000007_class{class_one_based:03d}_cs038.npz",
        original_index=np.int64(7),
        class_index=np.int64(class_one_based - 1),
        current_size=np.int64(38),
        scores_with_prior=score_array.astype(np.float64),
        candidate_mask=np.ones_like(support_array),
        reconstruction_mask=support_array,
        reconstruction_n_significant=np.int64(np.count_nonzero(support_array)),
    )


def _capture_root(tmp_path: Path, *, extra_current_support: bool = False) -> Path:
    root = tmp_path / "capture"
    root.mkdir()
    _write_capture(root, 1, [0.0, -10.0], [True, extra_current_support])
    _write_capture(root, 2, [-20.0], [False])
    _write_capture(root, 3, [-30.0], [False])
    _write_capture(root, 4, [-40.0], [False])
    return root


@pytest.mark.unit
def test_reports_exact_joint_support(tmp_path: Path) -> None:
    report = build_report(
        capture_root=_capture_root(tmp_path),
        adaptive_fraction=0.999,
        repetitions=2,
    )

    assert report["classification"] == "joint_f32_raw_weight_support_matches_current_probability_support"
    assert report["current_total_significant"] == 1
    assert report["f32_raw_weight_total_significant"] == 1
    assert report["joint_mask_mismatch_count"] == 0
    assert report["repeat_mask_exact"] is True
    assert report["correlation_used"] is False


@pytest.mark.unit
def test_reports_joint_support_difference(tmp_path: Path) -> None:
    report = build_report(
        capture_root=_capture_root(tmp_path, extra_current_support=True),
        adaptive_fraction=0.999,
        repetitions=1,
    )

    assert report["classification"] == "joint_f32_raw_weight_support_differs_from_current_probability_support"
    assert report["joint_mask_mismatch_count"] == 1
    assert report["current_only_count"] == 1
    assert report["f32_only_count"] == 0


@pytest.mark.unit
def test_rejects_non_float32_score_capture(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    path = root / "pass2_orig000007_class001_cs038.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["scores_with_prior"] = values["scores_with_prior"].copy()
    values["scores_with_prior"][0, 0] = 0.1
    np.savez(path, **values)

    with pytest.raises(ValueError, match="exact float32 capture"):
        build_report(
            capture_root=root,
            adaptive_fraction=0.999,
            repetitions=1,
        )
