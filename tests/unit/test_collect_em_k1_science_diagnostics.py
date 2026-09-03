from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "collect_em_k1_science_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("collect_em_k1_science_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.unit


def _asymmetric_volume(size: int = 24) -> np.ndarray:
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    volume = np.exp(-((z - 0.30 * size) ** 2 + (y - 0.46 * size) ** 2 + (x - 0.65 * size) ** 2) / 7.0)
    volume += 0.7 * np.exp(
        -((z - 0.68 * size) ** 2 + (y - 0.28 * size) ** 2 + (x - 0.38 * size) ** 2) / 5.0
    )
    volume += 0.3 * np.exp(
        -((z - 0.52 * size) ** 2 + (y - 0.72 * size) ** 2 + (x - 0.22 * size) ** 2) / 3.0
    )
    return np.asarray(volume, dtype=np.float32)


def test_continuous_proper_fit_recovers_small_rigid_drift() -> None:
    source = _asymmetric_volume()
    truth_rotation = Rotation.from_euler("zyx", [3.0, -2.0, 1.0], degrees=True).as_matrix()
    truth_translation = np.array([0.7, -0.4, 0.3])
    target = MODULE.apply_proper_rigid_transform(source, truth_rotation, truth_translation)

    fit = MODULE.fit_proper_rigid_transform(
        source,
        target,
        fit_max_shell=6,
        refine_healpix_orders=(),
        max_continuous_rotation_degrees=10.0,
        max_translation_fit_voxels=2.0,
        seed_rotation_matrix=np.eye(3),
    )

    fitted_rotation = np.asarray(fit["rotation_matrix_recovar_to_relion"])
    assert fit["optimizer_success"] is True
    assert fit["fit_correlation"] > 0.97
    assert np.linalg.det(fitted_rotation) == pytest.approx(1.0, abs=1.0e-10)
    assert np.linalg.norm(fitted_rotation.T @ fitted_rotation - np.eye(3), ord="fro") < 1.0e-10


def test_global_fit_always_considers_canonical_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _asymmetric_volume()
    distant_seed = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()[None, ...]
    monkeypatch.setattr(MODULE, "relion_alignment_rotations", lambda _order: distant_seed)

    fit = MODULE.fit_proper_rigid_transform(
        source,
        source,
        fit_max_shell=6,
        refine_healpix_orders=(),
        max_continuous_rotation_degrees=5.0,
        max_translation_fit_voxels=1.0,
    )

    assert fit["seed_source"] == "identity_augmented_RELION_HEALPix_grid"
    np.testing.assert_array_equal(fit["seed_rotation_matrix"], np.eye(3))
    assert fit["fit_correlation"] > 0.999


def test_transform_rejects_reflection() -> None:
    volume = _asymmetric_volume(16)

    with pytest.raises(ValueError, match="proper rotation"):
        MODULE.apply_proper_rigid_transform(volume, np.diag([-1.0, 1.0, 1.0]), (0.0, 0.0, 0.0))


def test_common_mask_is_engine_symmetric_and_nontrivial() -> None:
    first = _asymmetric_volume(24)
    second = 2.5 * first + np.float32(0.2)

    mask_forward, metadata = MODULE.construct_common_soft_mask(first, second)
    mask_reverse, _ = MODULE.construct_common_soft_mask(second, first)

    np.testing.assert_array_equal(mask_forward, mask_reverse)
    assert metadata["engine_symmetric"] is True
    assert np.any(mask_forward > 0.5)
    assert np.any(mask_forward < 0.5)


def test_shell_fsc_identical_inputs_is_one() -> None:
    volume = _asymmetric_volume(16)
    calculator = MODULE.ShellFscCalculator(16)
    transform = calculator.fourier(volume)

    curve = calculator.curve_from_fourier(transform, transform)

    np.testing.assert_allclose(curve[np.isfinite(curve)], 1.0, rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize("size", [15, 16])
def test_shell_fsc_half_spectrum_matches_full_complex_fft(size: int) -> None:
    rng = np.random.default_rng(10202 + size)
    left = rng.standard_normal((size, size, size), dtype=np.float32)
    right = rng.standard_normal((size, size, size), dtype=np.float32)
    calculator = MODULE.ShellFscCalculator(size)

    observed = calculator.curve_from_fourier(
        calculator.fourier(left),
        calculator.fourier(right),
    )

    frequencies = (np.fft.fftfreq(size) * size).astype(np.float32)
    squared = frequencies[:, None] ** 2 + frequencies[None, :] ** 2
    shells = np.empty((size, size, size), dtype=np.int16)
    for first_axis_index, first_axis_frequency in enumerate(frequencies):
        shells[first_axis_index] = np.rint(
            np.sqrt(squared + first_axis_frequency * first_axis_frequency)
        ).astype(np.int16)
    left_full = np.fft.fftn(left)
    right_full = np.fft.fftn(right)
    numerator = np.bincount(
        shells.reshape(-1),
        weights=np.real(left_full * np.conj(right_full)).reshape(-1),
    )
    left_power = np.bincount(shells.reshape(-1), weights=(np.abs(left_full) ** 2).reshape(-1))
    right_power = np.bincount(shells.reshape(-1), weights=(np.abs(right_full) ** 2).reshape(-1))
    expected = numerator / np.sqrt(left_power * right_power)
    expected = expected[: size // 2 - 1]

    np.testing.assert_allclose(observed, expected, rtol=2.0e-6, atol=2.0e-7)


def test_load_volume_accepts_structured_mrc_voxel_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "volume.mrc"
    path.touch()
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    voxel = np.array((1.31, 1.31, 1.31), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])[()]
    monkeypatch.setattr(MODULE.helpers, "load_mrc", lambda *_args, **_kwargs: (volume, voxel))

    observed_volume, observed_voxel = MODULE._load_volume(path, "recovar")

    np.testing.assert_array_equal(observed_volume, volume)
    assert observed_voxel == pytest.approx(1.31)


def test_load_volume_rejects_anisotropic_voxel_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "volume.mrc"
    path.touch()
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    voxel = np.array((1.0, 1.0, 1.1), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])[()]
    monkeypatch.setattr(MODULE.helpers, "load_mrc", lambda *_args, **_kwargs: (volume, voxel))

    with pytest.raises(ValueError, match="anisotropic voxel size"):
        MODULE._load_volume(path, "recovar")
