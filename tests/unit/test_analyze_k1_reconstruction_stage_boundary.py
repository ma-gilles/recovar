from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_reconstruction_stage_boundary import (
    _select_accumulator_targets,
    _stage_path,
)


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


def test_select_accumulator_targets_uses_explicit_joined_fields() -> None:
    archive = {
        "Ft_y": np.asarray([1.0 + 2.0j]),
        "Ft_ctf": np.asarray([3.0]),
        "Ft_y_0": np.asarray([4.0 + 5.0j]),
        "Ft_ctf_0": np.asarray([6.0]),
    }

    targets = _select_accumulator_targets(archive, joined=True, halves=(1,))

    assert len(targets) == 1
    rank, numerator, weight = targets[0]
    assert rank == 1
    np.testing.assert_array_equal(numerator, archive["Ft_y"])
    np.testing.assert_array_equal(weight, archive["Ft_ctf"])


def test_select_accumulator_targets_rejects_ambiguous_joined_rank() -> None:
    archive = {"Ft_y": np.asarray([1.0]), "Ft_ctf": np.asarray([2.0])}

    with pytest.raises(ValueError, match="--halves 1"):
        _select_accumulator_targets(archive, joined=True, halves=(1, 2))
