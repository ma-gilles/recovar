from pathlib import Path

import numpy as np
import pytest

from scripts.seal_k1_reference_lifecycle_boundary import seal_boundary


def _write_capture(path: Path, *, iteration: int = 1, half: int = 1, stage: str = "post_mask"):
    values = (np.arange(8) + 1j * np.arange(8)[::-1]).astype(np.complex128)
    np.savez(
        path,
        iteration=np.asarray(iteration, dtype=np.int32),
        half=np.asarray(half, dtype=np.int32),
        stage=np.asarray(stage),
        value_fourier=values,
    )
    return values


def test_seal_boundary_preserves_process_resident_fourier_values(tmp_path):
    source = tmp_path / "lifecycle.npz"
    output = tmp_path / "boundary.npz"
    expected = _write_capture(source)

    seal_boundary(source, output, expected_size=2)

    with np.load(output, allow_pickle=False) as payload:
        assert np.array_equal(payload["mean_vol_ft"], expected)
        assert str(payload["source_path"]) == str(source.resolve())
        assert int(payload["source_iteration"]) == 1
        assert int(payload["source_half"]) == 1
        assert str(payload["source_stage"]) == "post_mask"
        assert int(payload["volume_size"]) == 2


@pytest.mark.parametrize(
    ("iteration", "half", "stage"),
    [(0, 1, "post_mask"), (1, 3, "post_mask"), (1, 1, "presave")],
)
def test_seal_boundary_rejects_wrong_lifecycle_boundary(
    tmp_path, iteration, half, stage
):
    source = tmp_path / "lifecycle.npz"
    _write_capture(source, iteration=iteration, half=half, stage=stage)

    with pytest.raises(ValueError, match="expected iteration-1"):
        seal_boundary(source, tmp_path / "boundary.npz", expected_size=2)


def test_seal_boundary_refuses_overwrite(tmp_path):
    source = tmp_path / "lifecycle.npz"
    output = tmp_path / "boundary.npz"
    _write_capture(source)
    output.touch()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        seal_boundary(source, output, expected_size=2)
