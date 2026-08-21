from pathlib import Path

import numpy as np
import pytest

from scripts.capture_k1_focused_native_support import native_scoring_noise_radial
from scripts.validate_relion_scoring_noise_capture import ScoringNoiseCapture


def _capture(*, iteration=2, rank=1, optics_group=0, shell_count=65):
    sigma2 = np.linspace(0.25, 1.0, shell_count, dtype=np.float64)
    fudge = np.float32(1.5)
    return ScoringNoiseCapture(
        path=Path("noise.bin"),
        iteration=iteration,
        rank=rank,
        optics_group_zero_based=optics_group,
        sigma2_fudge=fudge,
        sigma2=sigma2,
        inverse_sigma2_f32=np.asarray(1.0 / (np.float64(fudge) * sigma2), dtype=np.float32),
    )


@pytest.mark.unit
def test_native_scoring_noise_radial_converts_to_recovar_fourier_units():
    capture = _capture()

    actual = native_scoring_noise_radial(
        capture,
        consumer_iteration=2,
        half=1,
        image_shape=(128, 128),
    )

    expected = capture.sigma2 * np.float64(capture.sigma2_fudge) * np.float64(128**4)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (_capture(iteration=3), "iteration"),
        (_capture(rank=2), "MPI rank"),
        (_capture(optics_group=1), "one optics group"),
        (_capture(shell_count=64), "shell count"),
    ],
)
def test_native_scoring_noise_radial_rejects_misaligned_capture(capture, message):
    with pytest.raises(ValueError, match=message):
        native_scoring_noise_radial(
            capture,
            consumer_iteration=2,
            half=1,
            image_shape=(128, 128),
        )
