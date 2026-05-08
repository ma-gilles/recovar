#!/usr/bin/env python
"""Generate a high-resolution single-class PDB sanity dataset for RELION.

This benchmark is deliberately not based on the legacy ``recovar/assets/vol*.mrc``
density fixtures. It starts from atomic coordinates, generates a target-grid
scattering-potential volume at the requested voxel size, then simulates images
from that target-resolution volume.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.output.output import mkdir_safe, save_volume
from recovar.simulation import simulator, synthetic_dataset
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.utils.helpers import write_relion_mrc


logger = logging.getLogger(__name__)


def _required_dataset_files(output_dir: Path, grid_size: int) -> list[Path]:
    return [
        output_dir / "particles.star",
        output_dir / f"particles.{grid_size}.mrcs",
        output_dir / "poses.pkl",
        output_dir / "ctf.pkl",
        output_dir / "simulation_info.pkl",
    ]


def _write_references(output_dir: Path, grid_size: int, init_resolution_ang: float) -> None:
    dataset = load_dataset(
        str(output_dir / f"particles.{grid_size}.mrcs"),
        poses_file=str(output_dir / "poses.pkl"),
        ctf_file=str(output_dir / "ctf.pkl"),
        lazy=False,
    )
    sim_info = utils.pickle_load(output_dir / "simulation_info.pkl")
    reconstruction = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    gt_ft = np.asarray(reconstruction.get_mean())

    init_radius = max(1, int(round(dataset.voxel_size * grid_size / init_resolution_ang)))
    init_ft = gt_ft * dataset.get_valid_frequency_indices(rad=init_radius)

    save_volume(
        np.asarray(gt_ft.reshape(-1)),
        str(output_dir / "reference_gt"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )
    save_volume(
        np.asarray(init_ft.reshape(-1)),
        str(output_dir / "reference_init"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )

    gt_real = np.real(ftu.get_idft3(jnp.asarray(gt_ft).reshape(dataset.volume_shape)))
    write_relion_mrc(output_dir / "reference_gt_relion.mrc", np.asarray(gt_real), voxel_size=dataset.voxel_size)

    init_real = np.real(ftu.get_idft3(jnp.asarray(init_ft).reshape(dataset.volume_shape)))
    write_relion_mrc(output_dir / "reference_init_relion.mrc", np.asarray(init_real), voxel_size=dataset.voxel_size)


def prepare_benchmark(
    output_dir: Path,
    *,
    n_images: int,
    grid_size: int,
    voxel_size: float | None,
    noise_level: float,
    noise_model: str,
    init_resolution_ang: float,
    pdb_path: str | None,
    pdb_bfactor: float,
    relion_normalize: bool,
    streaming_mmap: bool,
    streaming_chunk_size: int,
    seed: int,
) -> None:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if init_resolution_ang <= 0:
        raise ValueError(f"init_resolution_ang must be positive, got {init_resolution_ang}")
    if streaming_mmap and not relion_normalize:
        raise ValueError("streaming_mmap requires relion_normalize=True in the simulator")
    if voxel_size is None:
        voxel_size = 4.25 * 128 / grid_size

    mkdir_safe(str(output_dir))
    np.random.seed(seed)

    volume_prefix = output_dir / "pdb_state" / "vol"
    volume_path = output_dir / "pdb_state" / "vol0000.mrc"
    if not volume_path.exists():
        logger.info(
            "Generating single PDB state: grid=%d voxel=%.6g B=%.6g source=%s",
            grid_size,
            voxel_size,
            pdb_bfactor,
            pdb_path or "recovar/assets/5nrl_atoms.npz",
        )
        generate_trajectory_volumes(
            output_dir=str(output_dir),
            grid_size=grid_size,
            n_volumes=1,
            voxel_size=voxel_size,
            Bfactor=pdb_bfactor,
            max_rotation_degrees=0.0,
            pdb_path=pdb_path,
            output_prefix=str(volume_prefix),
        )
    else:
        logger.info("Reusing existing target-grid PDB state at %s", volume_path)

    required = _required_dataset_files(output_dir, grid_size)
    if not all(path.exists() for path in required):
        logger.info(
            "Generating particles: output=%s n_images=%d grid=%d noise=%g model=%s relion_normalize=%s streaming=%s",
            output_dir,
            n_images,
            grid_size,
            noise_level,
            noise_model,
            relion_normalize,
            streaming_mmap,
        )
        simulator.generate_synthetic_dataset(
            str(output_dir),
            voxel_size,
            str(volume_prefix),
            n_images,
            grid_size=grid_size,
            volume_distribution=np.asarray([1.0], dtype=np.float64),
            dataset_params_option="uniform",
            noise_level=noise_level,
            noise_model=noise_model,
            put_extra_particles=False,
            percent_outliers=0.0,
            volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0.0,
            contrast_std=0.0,
            disc_type="nufft",
            n_tilts=-1,
            image_dtype=np.float32,
            outlier_file_input=None,
            relion_normalize=relion_normalize,
            streaming_mmap=streaming_mmap,
            streaming_chunk_size=streaming_chunk_size,
        )
    else:
        logger.info("Reusing existing dataset files in %s", output_dir)

    sim_info_path = output_dir / "simulation_info.pkl"
    sim_info = utils.pickle_load(sim_info_path)
    assignments = np.asarray(sim_info["image_assignment"])
    if set(np.unique(assignments).tolist()) != {0}:
        raise RuntimeError(f"Expected a single-class dataset, got assignments {np.unique(assignments).tolist()}")

    sim_info["pdb_k1_generation"] = {
        "source": "recovar.simulation.trajectory_generation.generate_trajectory_volumes",
        "pdb_path": pdb_path,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "pdb_bfactor": pdb_bfactor,
        "noise_level": noise_level,
        "noise_model": noise_model,
        "init_resolution_ang": init_resolution_ang,
        "seed": seed,
        "relion_normalize": relion_normalize,
        "streaming_mmap": streaming_mmap,
    }
    utils.pickle_dump(sim_info, sim_info_path)

    _write_references(output_dir, grid_size, init_resolution_ang)
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(sim_info["pdb_k1_generation"], f, indent=2)

    logger.info("Benchmark ready at %s", output_dir)
    logger.info("RELION --ref %s", output_dir / "reference_init_relion.mrc")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a high-resolution K=1 PDB sanity dataset for RELION/RECOVAR EM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for particles, references, and generation metadata.",
    )
    parser.add_argument("--n-images", type=int, default=100000)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=0.001)
    parser.add_argument("--noise-model", choices=("white", "radial1"), default="white")
    parser.add_argument("--init-resolution-ang", type=float, default=30.0)
    parser.add_argument(
        "--pdb-path",
        default=None,
        help="Optional .npz/.cif/.pdb input. Default uses recovar/assets/5nrl_atoms.npz.",
    )
    parser.add_argument(
        "--pdb-bfactor",
        type=float,
        default=0.0,
        help="B-factor for target-grid PDB volume generation. Use 0 for near-Nyquist sanity checks.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--relion-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RELION-style particle normalization; enabled by default.",
    )
    parser.add_argument(
        "--streaming-mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write particles through a streaming MRC mmap to avoid large resident memory use.",
    )
    parser.add_argument("--streaming-chunk-size", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_benchmark(
        args.output_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        noise_level=args.noise_level,
        noise_model=args.noise_model,
        init_resolution_ang=args.init_resolution_ang,
        pdb_path=args.pdb_path,
        pdb_bfactor=args.pdb_bfactor,
        relion_normalize=args.relion_normalize,
        streaming_mmap=args.streaming_mmap,
        streaming_chunk_size=args.streaming_chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
