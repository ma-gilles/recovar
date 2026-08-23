from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_native_cc_translation_tie import (
    _float_record,
    _pass_field,
    _recovar_score_panel,
    _relion_atomic_score,
)


pytestmark = pytest.mark.unit


def test_pass_field_requires_one_exact_pass_layout() -> None:
    payload = {
        "pass0_firstiter_cc_raw_trans_idx": np.asarray([7], dtype=np.int32),
        "pass1_firstiter_cc_raw_trans_idx": np.asarray([89, 91], dtype=np.int32),
    }
    np.testing.assert_array_equal(
        _pass_field(payload, "firstiter_cc_raw_trans_idx"),
        np.asarray([89, 91], dtype=np.int32),
    )


def test_relion_atomic_score_records_exact_binary32_result() -> None:
    score = _relion_atomic_score(np.float32(1.25), np.float32(2.5))
    record = _float_record(score)
    assert np.asarray(score).dtype == np.float32
    assert record["float32_bits_hex"] == f"0x{score.view(np.uint32).item():08x}"
    assert np.float32(record["float32"]).view(np.uint32) == score.view(np.uint32)


def test_recovar_panel_uses_matrix_matched_local_rotation_row(tmp_path) -> None:
    old_path = tmp_path / "old.npz"
    current_path = tmp_path / "current.npz"
    old_scores = np.full((2, 4), -np.inf, dtype=np.float64)
    old_scores[1, 1] = np.float32(0.25)
    old_scores[1, 3] = np.nextafter(np.float32(0.25), np.float32(np.inf))
    np.savez(
        old_path,
        oversampled_rot_indices=np.asarray([1000, 2000], dtype=np.int64),
        scores_pre_prior=old_scores,
    )
    current_scores = np.full((1, 3, 4), -np.inf, dtype=np.float64)
    current_scores[0, 2, 1] = np.float32(0.25)
    current_scores[0, 2, 3] = np.float32(0.25)
    np.savez(
        current_path,
        original_indices=np.asarray([38594], dtype=np.int64),
        active_global_rotation_indices=np.asarray([2000], dtype=np.int64),
        active_rotation_rows=np.asarray([2], dtype=np.int64),
        candidate_preprior_scores=current_scores,
    )

    report = _recovar_score_panel(
        old_path,
        current_path,
        matched_rotation_row=1,
        translations=(1, 3),
    )
    assert report["historical"]["global_rotation"] == 2000
    assert report["historical"]["winner_translation"] == 3
    assert report["historical"]["exact_tie"] is False
    assert report["current"]["global_rotation"] == 2000
    assert report["current"]["winner_translation"] == 1
    assert report["current"]["exact_tie"] is True
