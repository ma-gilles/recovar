import numpy as np
from pathlib import Path
import pandas as pd

import recovar.simulation.simulator as simulator
import recovar.simulation.synthetic_dataset as synthetic_dataset
from recovar.data_io import cryoem_dataset as dataset
from recovar import core
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.data_io import starfile


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
    metadata = dataset.ImageMetadata(rots, trans, ctf_params)
    cryo = dataset.CryoEMDataset(
        image_source=None,
        voxel_size=voxel_size,
        metadata=metadata,
        ctf_evaluator=core.CTFEvaluator(),
        grid_size=grid_size,
    )
    return cryo


class TinyFTImageStack:
    """In-memory Fourier-domain stack that implements the image-backend contract."""

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
        self._images_fourier = (
            np.asarray(fourier_transform_utils.get_dft2(images_real)).reshape(self.n_images, -1).astype(np.complex64)
        )

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        for start in range(0, self.n_images, batch_size):
            end = min(start + batch_size, self.n_images)
            idx = np.arange(start, end, dtype=np.int32)
            yield self._images_fourier[idx], idx, idx

    def get_image_generator(self, batch_size, num_workers=0):
        return self.get_dataset_generator(batch_size, num_workers=num_workers)

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

    def get_half(self, indices):
        return self.get(indices)


def make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8, seed=0):
    images, ctf_params, rots, trans, _, voxel_size, _ = make_tiny_simulation(
        grid_size=grid_size,
        n_images=n_images,
        seed=seed,
    )
    image_stack = TinyFTImageStack(images)
    metadata = dataset.ImageMetadata(rots, trans, ctf_params)
    cryo = dataset.CryoEMDataset(
        image_source=image_stack,
        voxel_size=voxel_size,
        metadata=metadata,
        ctf_evaluator=core.CTFEvaluator(),
        dataset_indices=np.arange(image_stack.n_images, dtype=np.int32),
        grid_size=grid_size,
    )
    cryo.noise = TinyRadialNoise(image_size=np.prod(cryo.image_shape))
    return cryo


def make_tiny_loader_files(
    out_dir,
    grid_size=8,
    n_images=6,
    n_particles=3,
):
    """Create tiny on-disk files for dataset loading tests.

    Returns dict with:
      particles_mrcs, particles_star, poses_pkl, ctf_pkl
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tiny real-space stack (n_images, D, D)
    images = np.arange(n_images * grid_size * grid_size, dtype=np.float32).reshape(n_images, grid_size, grid_size)
    particles_mrcs = out_dir / "particles.mrcs"
    utils.write_mrc(str(particles_mrcs), images)

    # Build a tiny STAR where rows map to the same MRCS stack and grouped particles.
    # Particles are assigned cyclically to keep deterministic groups.
    groups = [f"g{(i % n_particles) + 1}" for i in range(n_images)]
    image_names = [f"{i + 1}@{particles_mrcs.name}" for i in range(n_images)]
    df = pd.DataFrame(
        {
            "_rlnImageName": image_names,
            "_rlnGroupName": groups,
            "_rlnMicrographPreExposure": np.linspace(1.0, float(n_images), n_images, dtype=np.float32),
            "_rlnCtfScalefactor": np.ones(n_images, dtype=np.float32),
            "_rlnCtfBfactor": -np.linspace(1.0, float(n_images), n_images, dtype=np.float32),
        }
    )
    particles_star = out_dir / "particles.star"
    starfile.write_star(str(particles_star), data=df)

    # CTF pickle expected by load_utils.load_ctf_params: (N, 9)
    # [D, Apix, DFU, DFV, DFANG, VOLT, CS, W, phase]
    ctf = np.zeros((n_images, 9), dtype=np.float32)
    ctf[:, 0] = float(grid_size)
    ctf[:, 1] = 1.5
    ctf[:, 2] = 15000.0
    ctf[:, 3] = 15000.0
    ctf[:, 4] = 0.0
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1
    ctf[:, 8] = 0.0
    ctf_pkl = out_dir / "ctf.pkl"
    utils.pickle_dump(ctf, str(ctf_pkl))

    # Pose pickle expected by load_utils.load_poses:
    # (rots, trans_frac) with trans in fractional units.
    rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    trans = np.zeros((n_images, 2), dtype=np.float32)
    poses_pkl = out_dir / "poses.pkl"
    utils.pickle_dump((rots, trans), str(poses_pkl))

    return {
        "particles_mrcs": str(particles_mrcs),
        "particles_star": str(particles_star),
        "poses_pkl": str(poses_pkl),
        "ctf_pkl": str(ctf_pkl),
        "n_images": n_images,
        "grid_size": grid_size,
    }


def make_tiny_tilt_loader_files_from_simulator(
    out_dir,
    grid_size=8,
    n_images=24,
    n_tilts=3,
    n_volumes=4,
    voxel_size=1.5,
):
    """Create tiny on-disk cryo-ET files via simulator for robust loader tests.

    Returns dict with:
      particles_star, particles_mrcs, poses_pkl, ctf_pkl, simulation_info_pkl, n_images, grid_size, n_tilts
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real-space compact volumes with a moving blob.
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    vol_prefix = out_dir / "vol"
    for i in range(n_volumes):
        t = i / max(n_volumes - 1, 1)
        static = np.exp(-((xx + 0.25) ** 2 + (yy + 0.1) ** 2 + (zz + 0.05) ** 2) / (2 * 0.18**2))
        moving = np.exp(-((xx - (-0.45 + 0.9 * t)) ** 2 + (yy - 0.25) ** 2 + (zz - 0.2) ** 2) / (2 * 0.16**2))
        vol = (static + 0.8 * moving).astype(np.float32)
        vol -= vol.mean()
        denom = np.linalg.norm(vol.ravel())
        if denom > 0:
            vol /= denom
        utils.write_mrc(f"{vol_prefix}{i:04d}.mrc", vol, voxel_size=voxel_size)

    sim_out = out_dir / "simulated_dataset"
    sim_out.mkdir(parents=True, exist_ok=True)
    _, sim_info = simulator.generate_synthetic_dataset(
        str(sim_out),
        voxel_size,
        str(vol_prefix),
        int(n_images),
        outlier_file_input=None,
        grid_size=grid_size,
        volume_distribution=np.ones(n_volumes, dtype=np.float32) / float(n_volumes),
        dataset_params_option="uniform",
        noise_level=0.5,
        noise_model="radial1",
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=0.0,
        disc_type="linear_interp",
        n_tilts=int(n_tilts),
    )

    particles_star = sim_out / "particles.star"
    particles_mrcs = sim_out / f"particles.{grid_size}.mrcs"
    ctf_pkl = sim_out / "ctf.pkl"
    poses_pkl = sim_out / "poses.pkl"
    sim_info_pkl = sim_out / "simulation_info.pkl"

    return {
        "particles_star": str(particles_star),
        "particles_mrcs": str(particles_mrcs),
        "poses_pkl": str(poses_pkl),
        "ctf_pkl": str(ctf_pkl),
        "simulation_info_pkl": str(sim_info_pkl),
        "n_images": int(n_images),
        "grid_size": int(grid_size),
        "n_tilts": int(n_tilts),
        "sim_info": sim_info,
    }
