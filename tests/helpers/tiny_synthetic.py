import numpy as np

import recovar.simulator as simulator
import recovar.synthetic_dataset as synthetic_dataset
from recovar import dataset
from recovar import core
import recovar.fourier_transform_utils as fourier_transform_utils


def tiny_ctf_pose_generator(n_images, grid_size):
    ctf = np.zeros((n_images, 9), dtype=np.float32)
    ctf[:, core.CTFParamIndex.DFU] = 15000.0
    ctf[:, core.CTFParamIndex.DFV] = 15000.0
    ctf[:, core.CTFParamIndex.DFANG] = 0.0
    ctf[:, core.CTFParamIndex.VOLT] = 300.0
    ctf[:, core.CTFParamIndex.CS] = 2.7
    ctf[:, core.CTFParamIndex.W] = 0.1
    ctf[:, core.CTFParamIndex.BFACTOR] = 50.0
    ctf[:, core.CTFParamIndex.CONTRAST] = 1.0
    rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    trans = np.zeros((n_images, 2), dtype=np.float32)
    return ctf, rots, trans


def make_tiny_fourier_volumes(grid_size=4):
    volume_size = int(grid_size**3)
    return np.array(
        [
            np.linspace(0.1, 1.0, volume_size),
            np.linspace(1.0, 0.2, volume_size),
        ],
        dtype=np.float32,
    ).astype(np.complex64)


def make_tiny_simulation(grid_size=4, n_images=8, seed=0):
    np.random.seed(seed)
    vols = make_tiny_fourier_volumes(grid_size=grid_size)
    return simulator.generate_simulated_dataset(
        volumes=vols,
        voxel_size=1.5,
        volume_distribution=np.array([0.5, 0.5], dtype=np.float32),
        n_images=n_images,
        noise_variance=np.ones(grid_size // 2 - 1, dtype=np.float32) * 1e-6,
        noise_scale_std=0.0,
        contrast_std=0.0,
        put_extra_particles=False,
        percent_outliers=0.0,
        dataset_param_generator=tiny_ctf_pose_generator,
        disc_type="linear_interp",
        image_offset_n_std=0.0,
        per_particle_contrast=True,
        premultiplied_ctf=False,
    )


def make_tiny_hvd_from_simulation(grid_size=4, n_images=8, seed=0):
    vols = make_tiny_fourier_volumes(grid_size=grid_size)
    _, _, _, _, simulation_info, _, _ = make_tiny_simulation(grid_size=grid_size, n_images=n_images, seed=seed)
    volume_size = int(grid_size**3)
    hvd = synthetic_dataset.HeterogeneousVolumeDistribution(
        volumes=vols.copy(),
        image_assignments=simulation_info["image_assignment"],
        contrasts=simulation_info["per_image_contrast"],
        valid_indices=np.ones(volume_size, dtype=np.float32),
        vol_batch_size=1,
    )
    return hvd, simulation_info, vols


def make_tiny_cryo_dataset(grid_size=4, n_images=8, seed=0):
    _, ctf_params, rots, trans, _, voxel_size, _ = make_tiny_simulation(
        grid_size=grid_size, n_images=n_images, seed=seed
    )
    cryo = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=voxel_size,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        CTF_fun=core.evaluate_ctf_wrapper,
        dataset_indices=None,
        grid_size=grid_size,
    )
    return cryo


class TinyFTImageStack:
    """In-memory Fourier-domain image stack for tiny end-to-end unit tests."""

    def __init__(self, images_real):
        images_real = np.asarray(images_real)
        if images_real.ndim != 3:
            raise ValueError("images_real must have shape (n_images, D, D)")
        self.n_images = images_real.shape[0]
        self.D = int(images_real.shape[-1])
        self.unpadded_D = self.D
        self.padding = 0
        self.image_shape = (self.D, self.D)
        self.mask = np.ones(self.image_shape, dtype=np.float32)
        self.Np = self.n_images
        self._images_fourier = np.asarray(fourier_transform_utils.get_dft2(images_real)).reshape(self.n_images, -1).astype(np.complex64)

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        for start in range(0, self.n_images, batch_size):
            end = min(start + batch_size, self.n_images)
            idx = np.arange(start, end, dtype=np.int32)
            yield self._images_fourier[idx], idx, idx

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        subset_indices = np.asarray(subset_indices, dtype=np.int32)
        for start in range(0, subset_indices.size, batch_size):
            idx = subset_indices[start : start + batch_size]
            yield self._images_fourier[idx], idx, idx

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers=num_workers)

    def process_images(self, image, apply_image_mask=True):
        # Keep this trace-friendly for jitted covariance code paths.
        return image


class TinyRadialNoise:
    def __init__(self, image_size):
        self._noise = np.ones((image_size,), dtype=np.float32)

    def get(self, indices):
        return np.tile(self._noise[None], (len(indices), 1))


def make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8, seed=0):
    images, ctf_params, rots, trans, _, voxel_size, _ = make_tiny_simulation(
        grid_size=grid_size,
        n_images=n_images,
        seed=seed,
    )
    image_stack = TinyFTImageStack(images)
    cryo = dataset.CryoEMDataset(
        image_stack=image_stack,
        voxel_size=voxel_size,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        CTF_fun=core.evaluate_ctf_wrapper,
        dataset_indices=None,
        grid_size=grid_size,
    )
    cryo.noise = TinyRadialNoise(image_size=np.prod(cryo.image_shape))
    return cryo
