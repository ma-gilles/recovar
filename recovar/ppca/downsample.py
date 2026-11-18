"""Reduce the fixed-pose PPCA fitting grid while preserving its field of view."""

import logging
from pathlib import Path

import numpy as np

from recovar.core import fourier_transform_utils as ftu

logger = logging.getLogger(__name__)

FSC_THRESHOLD = 1 / 7


def masked_mean_resolution(half1, half2, mask, volume_shape, voxel_size):
    """Return the first 0.143 crossing in Å, or None if it is unmeasurable."""
    from recovar.heterogeneity.locres import find_fsc_resol
    from recovar.reconstruction.regularization import get_fsc

    masked1 = ftu.get_dft3(ftu.get_idft3(half1.reshape(volume_shape)).real * mask)
    masked2 = ftu.get_dft3(ftu.get_idft3(half2.reshape(volume_shape)).real * mask)
    fsc = np.asarray(get_fsc(masked1, masked2, volume_shape))
    if fsc[1] < FSC_THRESHOLD or np.all(fsc[1:] >= FSC_THRESHOLD):
        return None

    radius = float(find_fsc_resol(fsc, threshold=FSC_THRESHOLD))
    return float(volume_shape[0] * voxel_size / radius)


def fitting_box_size(grid_size, voxel_size, resolution):
    """Round up to an even box whose PPCA projection radius covers the target."""
    if resolution is None or resolution <= 2 * voxel_size:
        return grid_size
    # PPCA projects to D//2 - 1. Reserve the Nyquist shell for interpolation.
    radius = np.ceil(grid_size * voxel_size / resolution)
    return min(grid_size, max(8, 2 * (int(radius) + 1)))


def make_fitting_dataset(dataset, target_size, directory, batch_size, gpu_memory):
    """Write the selected particles once and map the smaller stack for EM."""
    import mrcfile

    from recovar.data_io.cryoem_dataset import CryoEMDataset
    from recovar.data_io.downsample import downsample_images, get_downsample_batch_size
    from recovar.data_io.image_backends import ParticleImageDataset
    from recovar.data_io.image_loader import MRCLoader
    from recovar.data_io.image_metadata import ImageMetadata
    from recovar.data_io.image_sources import BackendImageSource, ImageSourceInfo
    from recovar.reconstruction.noise import VariableRadialNoiseModel

    ratio = target_size / dataset.grid_size
    voxel_size = dataset.voxel_size / ratio
    path = str(Path(directory) / "particles.mrcs")
    batch_size = min(batch_size, get_downsample_batch_size(dataset.grid_size, gpu_memory))
    with mrcfile.new_mmap(path, shape=(dataset.n_images, target_size, target_size), mrc_mode=2) as stack:
        stack.voxel_size = voxel_size
        for images, _, _, _, _, _, indices in dataset.iter_batches(batch_size, prefetch=False):
            stack.data[indices] = downsample_images(np.asarray(images), target_size)

    loader = MRCLoader(path, skip_staging=True)
    source = BackendImageSource(
        ParticleImageDataset(path, image_loader=loader), info=ImageSourceInfo(particles_file=path)
    )
    source.data_multiplier = dataset.data_multiplier
    metadata = ImageMetadata(dataset.rotation_matrices, dataset.translations * ratio, dataset.CTF_params)
    result = CryoEMDataset(
        source,
        voxel_size,
        metadata,
        ctf_evaluator=dataset.ctf_evaluator,
        premultiplied_ctf=dataset.premultiplied_ctf,
    )
    result.halfset_indices = dataset.halfset_indices

    # The retained Fourier samples have unchanged physical frequencies. Image
    # coefficients scale as ratio**2, so their noise variance scales as ratio**4.
    source_noise = dataset.noise
    n_shells = target_size // 2 - 1
    if isinstance(source_noise, VariableRadialNoiseModel):
        radials = source_noise.noise_variance_radials[:, :n_shells] * ratio**4
        result.noise = VariableRadialNoiseModel(radials, source_noise.dose_indices, image_shape=result.image_shape)
    else:
        radial = source_noise.noise_variance_radial[:n_shells] * ratio**4
        result.set_radial_noise_model(radial)
    return result


def crop_mean_and_masks(mean, masks, source_shape, target_size):
    """Crop a projected-density mean and spatial masks onto the fitting grid."""
    source_size = source_shape[0]
    ratio = target_size / source_size
    start = (source_size - target_size) // 2
    window = (slice(start, start + target_size),) * 3

    # A projection and its mean slice must have the same Fourier amplitude.
    cropped_mean = mean.reshape(source_shape)[window] * ratio**2
    cropped_mean = ftu.get_dft3(ftu.get_idft3(cropped_mean).real).reshape(-1)

    # Masks stay on their original real-space amplitude scale.
    cropped_masks = []
    for mask in masks:
        cropped = ftu.get_dft3(mask.reshape(source_shape))[window]
        cropped_masks.append(np.asarray(ftu.get_idft3(cropped).real * ratio**3).clip(0, 1))
    return cropped_mean, cropped_masks


def pad_half_spectra(spectra, source_size, target_size):
    """Embed packed spectra on a larger grid, preserving their Fourier norms."""
    q = spectra.shape[1]
    spectra = np.array(spectra.T).reshape(q, source_size, source_size, source_size // 2 + 1)
    start = (target_size - source_size) // 2
    # A small-grid Nyquist bin represents both signs. Split its coefficient
    # equally over the two large-grid bins in the full-spectrum L2 norm.
    spectra[:, 0, :, :] *= 2**-0.5
    spectra[:, :, 0, :] *= 2**-0.5
    spectra[..., -1] *= 2**-0.5
    padded = np.zeros((q, target_size, target_size, target_size // 2 + 1), dtype=spectra.dtype)
    stop = start + source_size
    n_freqs = source_size // 2 + 1
    padded[:, start:stop, start:stop, :n_freqs] = spectra
    padded[:, stop, start:stop, :n_freqs] = spectra[:, 0, :, :]
    padded[:, start:stop, stop, :n_freqs] = spectra[:, :, 0, :]
    padded[:, stop, stop, :n_freqs] = spectra[:, 0, 0, :]
    return padded.reshape(q, -1).T
