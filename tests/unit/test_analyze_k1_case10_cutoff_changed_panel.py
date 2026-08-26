from pathlib import Path

import pytest

from scripts.analyze_k1_case10_cutoff_changed_panel import parse_change_log


pytestmark = pytest.mark.unit


def test_parse_change_log_requires_records_to_match_terminal_counts(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "K=1 cutoff-row support change: original_index=17 half_local_index=9 "
        "removed=[31, 32] added=[48]\n"
        "K=1 deterministic cutoff-row rescore complete: margin=0.001 "
        "examined=100 ambiguous=4 support_changed_images=1 "
        "support_changed_candidates=3\n"
    )

    records, summary = parse_change_log(log)

    assert summary == {
        "margin": 0.001,
        "examined_images": 100,
        "ambiguous_images": 4,
        "support_changed_images": 1,
        "support_changed_candidates": 3,
    }
    assert records == [
        {
            "original_index": 17,
            "half_local_index": 9,
            "removed_flat_pose_ids": [31, 32],
            "added_flat_pose_ids": [48],
        }
    ]


def test_parse_change_log_rejects_missing_changed_rows(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "K=1 deterministic cutoff-row rescore complete: margin=0.001 "
        "examined=100 ambiguous=4 support_changed_images=1 "
        "support_changed_candidates=2\n"
    )

    with pytest.raises(ValueError, match="record count"):
        parse_change_log(log)
