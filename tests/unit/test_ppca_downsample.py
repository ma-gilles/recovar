"""Numerical checks for the fitting and output grids of fixed-pose PPCA."""

import mrcfile
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import CryoEMDataset
from recovar.data_io.image_metadata import ImageMetadata
from recovar.data_io.image_sources import create_image_source
from recovar.ppca import downsample

pytestmark = pytest.mark.unit


def test_mean_fsc_excludes_signal_outside_the_solvent_mask():
    size = 32
    shape = (size,) * 3
    axis = np.arange(size) / size * 2 * np.pi
    solvent_mask = np.zeros(shape, dtype=np.float32)
    taper = np.hanning(18)[1:-1]
    solvent_mask[8:24, 8:24, 8:24] = taper[:, None, None] * taper[None, :, None] * taper[None, None, :]
    common_signal = 10 * np.cos(axis)[:, None, None]
    rng = np.random.default_rng(38)
    half1 = common_signal + rng.normal(size=shape)
    half2 = common_signal + rng.normal(size=shape)
    solvent = (solvent_mask == 0) * rng.normal(0, 20, size=shape)

    def resolve(first, second):
        return downsample.masked_mean_resolution(ftu.get_dft3(first), ftu.get_dft3(second), solvent_mask, shape, 1.5)

    resolution = resolve(half1, half2)
    assert 3 < resolution < 48
    assert resolve(half1 + solvent, half2 - solvent) == pytest.approx(resolution)
    assert resolve(half1, half1) is None
    assert resolve(half1, -half1) is None


@pytest.mark.parametrize("resolution, box", [(None, 64), (2.2, 64), (8.0, 22), (200.0, 8)])
def test_ppca_box_retains_target_frequency(resolution, box):
    assert downsample.fitting_box_size(64, 1.25, resolution) == box


def test_crop_preserves_translations_slices_and_signal_to_noise(tmp_path):
    size, target = 16, 8
    n_images = 4
    w = np.arange(size, dtype=np.float32) * 2 * np.pi / size
    image = np.cos(w)[:, None] + np.cos(2 * w)[None, :]
    volume = np.broadcast_to(image[None, :, :] / size, (size,) * 3)
    image_stack = np.arange(1, n_images + 1, dtype=np.float32)[:, None, None] * image
    path = tmp_path / "original.mrcs"
    with mrcfile.new(path) as mrc:
        mrc.set_data(image_stack)

    rotations = np.tile(np.eye(3), (n_images, 1, 1))
    translations = np.arange(2 * n_images, dtype=np.float32).reshape(-1, 2)
    metadata = ImageMetadata(rotations, translations, np.ones((n_images, 9), dtype=np.float32))
    dataset = CryoEMDataset(create_image_source(str(path), ind=[1, 0, 3, 2]), 1.5, metadata)
    dataset.data_multiplier = -1
    dataset.halfset_indices = [np.array([0, 2]), np.array([1, 3])]
    dataset.set_radial_noise_model(np.full(size // 2 - 1, 16, dtype=np.float32))
    fit = downsample.make_fitting_dataset(dataset, target, tmp_path, batch_size=2, gpu_memory=1)
    np.testing.assert_array_equal(fit.rotation_matrices, rotations)
    np.testing.assert_array_equal(fit.translations, translations / 2)
    np.testing.assert_array_equal(fit.halfset_indices, dataset.halfset_indices)
    np.testing.assert_array_equal(fit.noise.noise_variance_radial, np.ones(target // 2 - 1))
    assert fit.voxel_size * target == dataset.voxel_size * size
    assert fit.data_multiplier == -1

    full_mask = np.ones((size,) * 3, dtype=np.float32)
    mean, [mask] = downsample.crop_mean_and_masks(ftu.get_dft3(volume), [full_mask], (size,) * 3, target)
    np.testing.assert_allclose(mask, 1.0)
    images, *_ = next(iter(fit.iter_batches(2, prefetch=False)))
    projected_mean = mean.reshape((target,) * 3)[target // 2]
    cropped_first_image = ftu.get_dft2(np.asarray(images)[0])
    np.testing.assert_allclose(projected_mean * 2, cropped_first_image, atol=1e-5)


def test_lift_preserves_hermitian_symmetry_and_fourier_gram_matrix():
    source, target = 8, 16
    rng = np.random.default_rng(8)
    volumes = rng.normal(size=(2, source, source, source)).astype(np.float32)
    packed = ftu.get_dft3_real(volumes).reshape(2, -1).T
    lifted = downsample.pad_half_spectra(packed, source, target)
    lifted = ftu.half_volume_to_full_volume(lifted.T, (target,) * 3)
    original = ftu.get_dft3(volumes).reshape(2, -1)
    np.testing.assert_allclose(lifted.conj() @ lifted.T, original.conj() @ original.T, rtol=1e-5)
    np.testing.assert_allclose(ftu.get_idft3(lifted.reshape(2, target, target, target)).imag, 0, atol=1e-7)
