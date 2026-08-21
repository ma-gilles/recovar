from pathlib import Path

import pytest

from scripts.analyze_k1_reconstruction_stage_boundary import _stage_path


def test_stage_path_selects_requested_uninterrupted_call(tmp_path: Path) -> None:
    expected = tmp_path / "reconstruct_rank01_pid123_call0002_tau2.bin"
    expected.touch()

    assert _stage_path(tmp_path, half=1, stage="tau2", call_index=2) == expected


def test_stage_path_does_not_silently_fall_back_to_continuation_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "reconstruct_rank01_pid123_call0000_tau2.bin").touch()

    with pytest.raises(ValueError, match="call-2"):
        _stage_path(tmp_path, half=1, stage="tau2", call_index=2)
