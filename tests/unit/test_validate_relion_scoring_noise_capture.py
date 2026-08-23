import numpy as np
import pytest

from scripts.validate_relion_scoring_noise_capture import (
    HEADER_SIZE,
    MAGIC,
    MAGIC_SIZE,
    load_capture,
    validate_capture,
)


def _write_capture(path, *, corrupt_inverse=False):
    sigma2 = np.asarray([0.25, 0.5, 1.0], dtype="<f8")
    fudge = np.float32(2.0)
    inverse = np.asarray(1.0 / (np.float64(fudge) * sigma2), dtype="<f4")
    if corrupt_inverse:
        inverse[1] = np.nextafter(inverse[1], np.float32(np.inf))
    header = np.zeros(16, dtype="<u8")
    header[:9] = [
        1,
        3,
        1,
        0,
        sigma2.size,
        8,
        8,
        4,
        fudge.view(np.uint32),
    ]
    payload = (
        MAGIC
        + bytes(MAGIC_SIZE - len(MAGIC))
        + header.tobytes()
        + sigma2.tobytes()
        + inverse.tobytes()
    )
    assert len(payload) == HEADER_SIZE + sigma2.size * 12
    path.write_bytes(payload)


@pytest.mark.unit
def test_scoring_noise_capture_replays_inverse_bits(tmp_path):
    path = tmp_path / "noise.bin"
    _write_capture(path)

    capture = load_capture(path)
    report = validate_capture(capture)

    assert capture.iteration == 3
    assert capture.rank == 1
    assert capture.sigma2_fudge == np.float32(2.0)
    assert report["inverse_sigma2_replay_exact"] is True


@pytest.mark.unit
def test_scoring_noise_capture_rejects_changed_inverse_word(tmp_path):
    path = tmp_path / "noise.bin"
    _write_capture(path, corrupt_inverse=True)

    with pytest.raises(ValueError, match="does not replay exactly"):
        load_capture(path)
