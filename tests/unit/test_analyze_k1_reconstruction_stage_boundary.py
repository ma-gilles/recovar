from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.analyze_k1_reconstruction_stage_boundary import (
    _load,
    _relion_projector_centered_to_fftw_half,
    _select_accumulator_targets,
    _stage_path,
)


def test_load_memory_maps_exact_stage_payload(tmp_path: Path) -> None:
    expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    path = tmp_path / "stage.bin"
    path.write_bytes(struct.pack("<3q", *expected.shape) + bytes(40) + expected.tobytes())

    actual = _load(path, np.dtype("<f8"))

    assert isinstance(actual, np.memmap)
    np.testing.assert_array_equal(actual, expected)


def test_load_rejects_payload_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "truncated.bin"
    path.write_bytes(struct.pack("<3q", 2, 3, 4) + bytes(40) + bytes(8))

    with pytest.raises(ValueError, match="payload size mismatch"):
        _load(path, np.dtype("<f8"))


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


def test_relion_projector_centered_to_fftw_half_matches_logical_gather() -> None:
    side = 5
    half_x = side // 2 + 1
    centered = np.empty((side, side, half_x), dtype=np.complex128)
    for z_index in range(side):
        for y_index in range(side):
            for x_index in range(half_x):
                centered[z_index, y_index, x_index] = (
                    100 * z_index + 10 * y_index + x_index
                ) + 1j * (z_index - y_index)

    actual = _relion_projector_centered_to_fftw_half(centered, max_radius=2)
    logical = [0, 1, 2, -2, -1]
    for z_raw, kz in enumerate(logical):
        for y_raw, ky in enumerate(logical):
            for kx in range(half_x):
                if kz * kz + ky * ky + kx * kx <= 4:
                    assert actual[z_raw, y_raw, kx] == centered[kz + 2, ky + 2, kx]
                else:
                    assert actual[z_raw, y_raw, kx] == 0


def test_relion_projector_centered_to_fftw_half_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="cubic RELION x-half"):
        _relion_projector_centered_to_fftw_half(
            np.zeros((5, 4, 3), dtype=np.float64),
            max_radius=2,
        )
