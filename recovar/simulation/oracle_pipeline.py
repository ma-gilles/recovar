"""Generate a fake RECOVAR pipeline output from a synthetic dataset.

This is a teaching/diagnostic helper. Given a simulated dataset whose
ground-truth volumes and noise model are known, it bypasses the actual
pipeline and writes the on-disk layout that ``recovar pipeline`` would
produce, so downstream commands like ``recovar compute_state`` can be
run on it directly.

Concretely:

- ``mean``, principal components and eigenvalues come from the
  simulator's heterogeneous reconstruction (oracle PCA).
- Per-image latent coordinates ``zs`` and their posterior precision
  ``cov_zs`` are computed by running the standard RECOVAR embedding step
  on the actual noisy images, but using the oracle mean / PCs /
  eigenvalues / noise variance.

The output isolates the variance produced by image noise on the latent
coordinates from the bias of imperfect mean / PCs / noise estimation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io import halfsets as halfset_io
from recovar.heterogeneity import embedding
from recovar.output import output_paths
from recovar.simulation import synthetic_dataset

logger = logging.getLogger(__name__)


_PARAMS_VERSION = "0.7"


def compute_oracle_basis(sim_info: dict) -> dict:
    """Compute the oracle Fourier mean, PCs and eigenvalues for a sim.

    Returns a dict with:
      - ``mean``: Fourier-space mean, shape ``(volume_size,)``, complex64.
      - ``u``: Fourier-space PCs, shape ``(volume_size, n_pcs)``, complex64.
      - ``s``: Eigenvalues of the covariance, shape ``(n_pcs,)``, float32.
      - ``volume_shape``: 3-tuple.
      - ``mean_real``: real-space mean volume for diagnostics, shape volume_shape.
      - ``u_real``: real-space PCs, shape ``(n_pcs, *volume_shape)`` (float32).
      - ``probs``: per-state probabilities used to build the basis.
    """
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    volume_shape = tuple(hvd.volume_shape)
    mean = np.asarray(hvd.get_mean(), dtype=np.complex64).reshape(-1)
    u = np.asarray(hvd.get_u(), dtype=np.complex64)
    s = np.asarray(hvd.get_s(), dtype=np.float32)

    mean_real = np.asarray(ftu.get_idft3(mean.reshape(volume_shape)).real, dtype=np.float32)
    n_pcs = u.shape[1]
    u_real = np.empty((n_pcs, *volume_shape), dtype=np.float32)
    for k in range(n_pcs):
        u_real[k] = np.asarray(ftu.get_idft3(u[:, k].reshape(volume_shape)).real, dtype=np.float32)

    return {
        "mean": mean,
        "u": u,
        "s": s,
        "volume_shape": volume_shape,
        "mean_real": mean_real,
        "u_real": u_real,
        "probs": np.asarray(hvd.get_probs_of_state(), dtype=np.float32),
    }


def _build_oracle_input_args(
    *,
    pipeline_dir: Path,
    particles_path: Path,
    poses_path: Path,
    ctf_path: Path,
    halfsets_pickle_path: Path,
    zdims: list[int],
    premultiplied_ctf: bool,
    correct_contrast: bool,
    noise_model: str,
    volume_mask_path: Path,
    focus_mask_path: Path | None,
) -> argparse.Namespace:
    """Build a Namespace mirroring the subset of pipeline args RECOVAR consumers use."""
    ns = argparse.Namespace()

    # Inputs needed by HalfsetDatasetSpec.from_args / load_halfset_dataset_from_args.
    ns.particles = str(particles_path)
    ns.poses = str(poses_path)
    ns.ctf = str(ctf_path)
    ns.datadir = None
    ns.strip_prefix = None
    ns.n_images = -1
    ns.padding = 0
    ns.tilt_series = False
    ns.tilt_series_ctf = "cryoem"
    ns.angle_per_tilt = None
    ns.dose_per_tilt = None
    ns.premultiplied_ctf = bool(premultiplied_ctf)
    ns.uninvert_data = "false"
    ns.downsample = None
    ns.ind = None
    ns.tilt_ind = None
    ns.ntilts = None

    # Where the pipeline would have stored the halfsets pickle. Pipeline.py
    # rewrites args.halfsets to this path before saving params, so we mirror
    # that here.
    ns.halfsets = str(halfsets_pickle_path)

    # Pipeline-flavoured flags downstream tooling occasionally inspects.
    ns.outdir = str(pipeline_dir)
    ns.zdim = list(zdims)
    ns.correct_contrast = bool(correct_contrast)
    ns.shared_contrast_across_tilts = True
    ns.noise_model = noise_model
    ns.mask = str(volume_mask_path)
    ns.focus_mask = None if focus_mask_path is None else str(focus_mask_path)
    ns.use_complement_mask = False
    ns.ignore_zero_frequency = False
    ns.keep_intermediate = False
    ns.lazy = False
    ns.do_over_with_contrast = False
    ns.only_mean = False
    ns.gpu_gb = None
    return ns


def _nan_pad_to_original(arr: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
    """Scatter dataset-local rows into a NaN-padded original-index array."""
    n_original = int(np.max(sorted_indices)) + 1 if sorted_indices.size else 0
    padded = np.full((n_original, *arr.shape[1:]), np.nan, dtype=arr.dtype)
    padded[sorted_indices] = arr
    return padded


def _save_embeddings_per_zdim(model_dir: Path, embedding_dict: dict, sorted_particle_indices: np.ndarray) -> None:
    """Write one ``model/zdim_K`` directory per zdim with the standard ``.npy`` files."""
    fields = (
        "latent_coords",
        "latent_coords_noreg",
        "latent_precision",
        "latent_precision_noreg",
        "contrasts",
        "contrasts_noreg",
    )
    zdims = sorted({z for f in fields if f in embedding_dict for z in embedding_dict[f].keys()})
    for zdim in zdims:
        zdim_dir = model_dir / f"zdim_{zdim}"
        zdim_dir.mkdir(parents=True, exist_ok=True)
        for field in fields:
            if field in embedding_dict and zdim in embedding_dict[field]:
                arr = np.asarray(embedding_dict[field][zdim])
                padded = _nan_pad_to_original(arr, sorted_particle_indices)
                np.save(zdim_dir / f"{field}.npy", padded)


def _save_volume_files(
    paths: output_paths.ResultPaths,
    *,
    voxel_size: float,
    mean_real: np.ndarray,
    u_real: np.ndarray,
    volume_mask: np.ndarray,
    focus_mask: np.ndarray | None,
) -> None:
    """Write the oracle MRC files needed by PipelineOutput accessors."""
    paths.ensure_volumes_dir()
    utils.write_mrc(paths.mean_volume, mean_real.astype(np.float32), voxel_size=voxel_size)
    utils.write_mrc(paths.mean_half1_unfil, mean_real.astype(np.float32), voxel_size=voxel_size)
    utils.write_mrc(paths.mean_half2_unfil, mean_real.astype(np.float32), voxel_size=voxel_size)
    utils.write_mrc(paths.mask_volume, np.asarray(volume_mask, dtype=np.float32), voxel_size=voxel_size)
    utils.write_mrc(paths.dilated_mask_volume, np.asarray(volume_mask, dtype=np.float32), voxel_size=voxel_size)
    if focus_mask is not None:
        utils.write_mrc(paths.focus_mask_volume, np.asarray(focus_mask, dtype=np.float32), voxel_size=voxel_size)
    for k in range(u_real.shape[0]):
        utils.write_mrc(paths.eigenvector(k), u_real[k].astype(np.float32), voxel_size=voxel_size)


def write_oracle_pipeline_output(
    *,
    pipeline_dir: str | os.PathLike,
    dataset_dir: str | os.PathLike,
    voxel_size: float,
    volume_mask: np.ndarray,
    sim_info: dict,
    zdims: list[int] | None = None,
    gpu_memory: float | None = None,
    disc_type: str = "linear_interp",
    focus_mask: np.ndarray | None = None,
    premultiplied_ctf: bool = False,
    noise_model: str = "radial",
    split_random_seed: int = 0,
    lazy: bool = False,
) -> dict:
    """Write a fake pipeline output that ``compute_state`` can consume.

    Parameters
    ----------
    pipeline_dir : path
        Output directory. Will be populated with ``model/`` and
        ``output/volumes/`` subtrees as if produced by ``recovar pipeline``.
    dataset_dir : path
        Directory that contains the simulated dataset
        (``particles.star``, ``poses.pkl``, ``ctf.pkl``,
        ``simulation_info.pkl``).
    voxel_size : float
        Voxel size in Angstroms.
    volume_mask : ndarray
        Solvent mask used by the embedding step and saved as ``mask.mrc``.
    sim_info : dict
        Simulator metadata, typically from ``simulation_info.pkl`` or the
        return value of ``simulator.generate_synthetic_dataset``. Must
        contain ``image_assignment`` and ``noise_variance``.
    zdims : list[int], optional
        Latent dimensions to compute embeddings for. Defaults to a small
        student-friendly ladder ``[1, 2, 4]``, capped at the rank of the
        oracle PCA basis.
    gpu_memory : float, optional
        GPU memory budget (GB) for the embedding batch sizing.
    disc_type : str
        Discretization passed to ``embedding.get_per_image_embedding``.
    focus_mask : ndarray, optional
        If supplied, written as ``output/volumes/focus_mask.mrc``.
    premultiplied_ctf : bool
        Whether the simulator pre-multiplied images by CTF.
    noise_model : str
        Which noise-model branch to advertise in the saved ``input_args``.
    split_random_seed : int
        Seed for the pipeline-style random halfset split.
    lazy : bool
        If true, keep particle images on disk and load batches on demand.

    Returns
    -------
    dict
        Summary describing what was written.
    """
    pipeline_dir = Path(pipeline_dir)
    dataset_dir = Path(dataset_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths.ResultPaths(str(pipeline_dir))
    paths.ensure_dirs()

    particles_path = dataset_dir / next(p.name for p in dataset_dir.glob("particles.*.mrcs"))
    star_path = dataset_dir / "particles.star"
    poses_path = dataset_dir / "poses.pkl"
    ctf_path = dataset_dir / "ctf.pkl"
    if not star_path.exists():
        raise FileNotFoundError(f"Expected {star_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Expected {poses_path}")
    if not ctf_path.exists():
        raise FileNotFoundError(f"Expected {ctf_path}")

    oracle = compute_oracle_basis(sim_info)
    volume_shape = oracle["volume_shape"]
    n_pcs_available = int(oracle["u"].shape[1])
    if n_pcs_available <= 0:
        raise ValueError("Oracle basis has zero principal components; cannot embed.")

    requested_zdims = list(zdims) if zdims else [min(z, n_pcs_available) for z in (1, 2, 4) if z <= n_pcs_available]
    if not requested_zdims:
        raise ValueError("No usable zdims; oracle basis is too small.")
    requested_zdims = sorted({int(min(z, n_pcs_available)) for z in requested_zdims if z > 0})

    image_assignment = np.asarray(sim_info["image_assignment"], dtype=np.int64).reshape(-1)
    valid_indices = np.flatnonzero(image_assignment >= 0).astype(np.int32)
    if valid_indices.size == 0:
        raise ValueError("All particles are flagged as outliers (image_assignment < 0).")

    halfsets_split = halfset_io.split_index_list(valid_indices, split_random_seed=split_random_seed)
    sorted_particle_indices = np.sort(np.concatenate(halfsets_split))
    if not np.array_equal(sorted_particle_indices, valid_indices):
        # Should not happen, but keep the layout invariant explicit.
        raise RuntimeError("halfset split and valid indices disagree")

    utils.pickle_dump([np.asarray(h, dtype=np.int32) for h in halfsets_split], paths.halfsets)
    utils.pickle_dump([np.asarray(h, dtype=np.int32) for h in halfsets_split], paths.particles_halfsets)

    input_args = _build_oracle_input_args(
        pipeline_dir=pipeline_dir,
        particles_path=star_path,
        poses_path=poses_path,
        ctf_path=ctf_path,
        halfsets_pickle_path=Path(paths.particles_halfsets),
        zdims=requested_zdims,
        premultiplied_ctf=premultiplied_ctf,
        correct_contrast=False,
        noise_model=noise_model,
        volume_mask_path=Path(paths.mask_volume),
        focus_mask_path=None if focus_mask is None else Path(paths.focus_mask_volume),
    )

    dataset = halfset_io.load_halfset_dataset_from_args(input_args, lazy=lazy, ind_split=halfsets_split)

    noise_variance_radial = np.asarray(sim_info["noise_variance"], dtype=np.float32).reshape(-1)
    if noise_variance_radial.size == 0:
        raise ValueError("simulation_info has empty noise_variance")
    dataset.set_radial_noise_model(noise_variance_radial)

    if gpu_memory is None:
        try:
            gpu_memory = float(utils.get_gpu_memory_total())
        except Exception:  # noqa: BLE001
            gpu_memory = 4.0

    _save_volume_files(
        paths,
        voxel_size=voxel_size,
        mean_real=oracle["mean_real"].reshape(volume_shape),
        u_real=oracle["u_real"],
        volume_mask=np.asarray(volume_mask, dtype=np.float32).reshape(volume_shape),
        focus_mask=None if focus_mask is None else np.asarray(focus_mask, dtype=np.float32).reshape(volume_shape),
    )

    flat_volume_mask = np.asarray(volume_mask, dtype=np.float32).reshape(-1)

    embedding_dict: dict[str, dict[int, np.ndarray]] = {
        "latent_coords": {},
        "latent_coords_noreg": {},
        "latent_precision": {},
        "latent_precision_noreg": {},
        "contrasts": {},
        "contrasts_noreg": {},
    }

    for zdim in requested_zdims:
        logger.info("Oracle regularized embedding for zdim=%d", zdim)
        zs, cov_zs, est_contrasts, _ = embedding.get_per_image_embedding(
            oracle["mean"],
            oracle["u"],
            oracle["s"],
            zdim,
            dataset,
            flat_volume_mask,
            gpu_memory,
            disc_type=disc_type,
            contrast_option="none",
            compute_covariances=True,
            ignore_zero_frequency=False,
        )
        zs = np.asarray(zs, dtype=np.float32)
        cov_zs = np.asarray(cov_zs, dtype=np.float32)
        contrasts_arr = np.asarray(est_contrasts, dtype=np.float32).reshape(-1)

        logger.info("Oracle noreg embedding for zdim=%d", zdim)
        zs_noreg, cov_zs_noreg, est_contrasts_noreg, _ = embedding.get_per_image_embedding(
            oracle["mean"],
            oracle["u"],
            np.full_like(oracle["s"], np.inf, dtype=np.float32),
            zdim,
            dataset,
            flat_volume_mask,
            gpu_memory,
            disc_type=disc_type,
            contrast_option="none",
            compute_covariances=True,
            ignore_zero_frequency=False,
        )
        zs_noreg = np.asarray(zs_noreg, dtype=np.float32)
        cov_zs_noreg = np.asarray(cov_zs_noreg, dtype=np.float32)
        contrasts_noreg_arr = np.asarray(est_contrasts_noreg, dtype=np.float32).reshape(-1)

        embedding_dict["latent_coords"][zdim] = zs
        embedding_dict["latent_coords_noreg"][zdim] = zs_noreg
        embedding_dict["latent_precision"][zdim] = cov_zs
        embedding_dict["latent_precision_noreg"][zdim] = cov_zs_noreg
        embedding_dict["contrasts"][zdim] = contrasts_arr
        embedding_dict["contrasts_noreg"][zdim] = contrasts_noreg_arr

    _save_embeddings_per_zdim(Path(paths.model_dir), embedding_dict, sorted_particle_indices)

    params = {
        "version": _PARAMS_VERSION,
        "volume_shape": tuple(int(d) for d in volume_shape),
        "voxel_size": float(voxel_size),
        "s": np.asarray(oracle["s"], dtype=np.float32),
        "noise_var_used": np.asarray(noise_variance_radial, dtype=np.float32),
        "noise_var_from_hf": np.asarray(noise_variance_radial, dtype=np.float32),
        "noise_var_from_het_residual": None,
        "input_args": input_args,
        "is_oracle_pipeline_output": True,
    }
    utils.pickle_dump(params, paths.params)

    summary = {
        "pipeline_dir": str(pipeline_dir),
        "zdims": [int(z) for z in requested_zdims],
        "n_pcs_available": int(n_pcs_available),
        "n_particles": int(valid_indices.size),
        "halfset_sizes": [int(len(h)) for h in halfsets_split],
        "noise_variance_length": int(noise_variance_radial.size),
        "voxel_size": float(voxel_size),
        "volume_shape": [int(d) for d in volume_shape],
    }
    return summary
