#!/usr/bin/env python
"""Generate a high-resolution two-class PDB trajectory parity dataset.

This is the K=2 analogue of the PDB-volume path used by
``recovar.commands.run_test_all_metrics``: it generates two endpoint 5NRL
states with the trajectory/scattering code, simulates particles from those
states, and writes RELION-ready class reference maps.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import starfile

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


def _write_class_references(output_dir: Path, grid_size: int, init_radius: int) -> None:
    dataset = load_dataset(
        str(output_dir / f"particles.{grid_size}.mrcs"),
        poses_file=str(output_dir / "poses.pkl"),
        ctf_file=str(output_dir / "ctf.pkl"),
        lazy=False,
    )
    sim_info = utils.pickle_load(output_dir / "simulation_info.pkl")
    heterogeneous_volumes = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    class_vols = np.asarray(heterogeneous_volumes.volumes[:2])
    init_mask = dataset.get_valid_frequency_indices(rad=init_radius)

    class_rows = []
    for class_idx, gt_ft in enumerate(class_vols):
        class_no = class_idx + 1
        init_ft = gt_ft * init_mask

        save_volume(
            np.asarray(gt_ft.reshape(-1)),
            str(output_dir / f"reference_gt_class{class_no:03d}"),
            volume_shape=dataset.volume_shape,
            from_ft=True,
            voxel_size=dataset.voxel_size,
        )
        save_volume(
            np.asarray(init_ft.reshape(-1)),
            str(output_dir / f"reference_init_class{class_no:03d}"),
            volume_shape=dataset.volume_shape,
            from_ft=True,
            voxel_size=dataset.voxel_size,
        )

        init_real = np.real(ftu.get_idft3(jnp.asarray(init_ft).reshape(dataset.volume_shape)))
        relion_ref = output_dir / f"reference_init_class{class_no:03d}_relion.mrc"
        write_relion_mrc(relion_ref, np.asarray(init_real), voxel_size=dataset.voxel_size)
        class_rows.append(
            {
                "rlnReferenceImage": relion_ref.name,
                "rlnClassDistribution": 0.5,
            }
        )

    save_volume(
        np.asarray(heterogeneous_volumes.get_mean().reshape(-1)),
        str(output_dir / "reference_gt"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )
    starfile.write(
        {"model_classes": pd.DataFrame(class_rows)},
        output_dir / "reference_init_classes_relion.star",
        overwrite=True,
    )


def prepare_benchmark(
    output_dir: Path,
    *,
    n_images: int,
    grid_size: int,
    voxel_size: float | None,
    noise_level: float,
    init_radius: int,
    pdb_path: str | None,
    pdb_bfactor: float,
    pdb_max_rotation: float,
    relion_normalize: bool,
    streaming_mmap: bool,
    streaming_chunk_size: int,
    seed: int,
) -> None:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if init_radius <= 0:
        raise ValueError(f"init_radius must be positive, got {init_radius}")
    if voxel_size is None:
        voxel_size = 4.25 * 128 / grid_size

    mkdir_safe(str(output_dir))
    np.random.seed(seed)

    volume_prefix = output_dir / "pdb_states" / "vol"
    if not all((output_dir / f"pdb_states/vol{idx:04d}.mrc").exists() for idx in range(2)):
        logger.info(
            "Generating two PDB trajectory endpoint states: grid=%d voxel=%.4g B=%.4g max_rot=%.4g",
            grid_size,
            voxel_size,
            pdb_bfactor,
            pdb_max_rotation,
        )
        generate_trajectory_volumes(
            output_dir=str(output_dir),
            grid_size=grid_size,
            n_volumes=2,
            voxel_size=voxel_size,
            Bfactor=pdb_bfactor,
            max_rotation_degrees=pdb_max_rotation,
            pdb_path=pdb_path,
            output_prefix=str(volume_prefix),
        )
    else:
        logger.info("Reusing existing PDB states at %s####.mrc", volume_prefix)

    required = _required_dataset_files(output_dir, grid_size)
    if not all(path.exists() for path in required):
        logger.info(
            "Generating particles: output=%s n_images=%d grid=%d noise=%.4g relion_normalize=%s",
            output_dir,
            n_images,
            grid_size,
            noise_level,
            relion_normalize,
        )
        simulator.generate_synthetic_dataset(
            str(output_dir),
            voxel_size,
            str(volume_prefix),
            n_images,
            grid_size=grid_size,
            volume_distribution=np.asarray([0.5, 0.5], dtype=np.float64),
            dataset_params_option="uniform",
            noise_level=noise_level,
            noise_model="white",
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
    sim_info["pdb_k2_generation"] = {
        "source": "recovar.simulation.trajectory_generation.generate_trajectory_volumes",
        "pdb_path": pdb_path,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "pdb_bfactor": pdb_bfactor,
        "pdb_max_rotation_degrees": pdb_max_rotation,
        "trajectory_degrees": [0.0, pdb_max_rotation],
        "seed": seed,
        "streaming_mmap": streaming_mmap,
        "streaming_chunk_size": streaming_chunk_size,
    }
    utils.pickle_dump(sim_info, sim_info_path)

    _write_class_references(output_dir, grid_size, init_radius)
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(sim_info["pdb_k2_generation"], f, indent=2)

    logger.info("Benchmark ready at %s", output_dir)
    logger.info("RELION --ref %s", output_dir / "reference_init_classes_relion.star")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a high-resolution K=2 PDB trajectory dataset for RELION parity.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k2_5k_128"),
    )
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--init-radius", type=int, default=10)
    parser.add_argument(
        "--pdb-path",
        default=None,
        help="Optional .npz/.cif/.pdb input. Default uses recovar/assets/5nrl_atoms.npz.",
    )
    parser.add_argument("--pdb-bfactor", type=float, default=80.0)
    parser.add_argument("--pdb-max-rotation", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--relion-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RELION-style particle normalization; enabled by default.",
    )
    parser.add_argument(
        "--streaming-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write generated particles through an mmap-backed MRC stack to cap peak memory.",
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
        init_radius=args.init_radius,
        pdb_path=args.pdb_path,
        pdb_bfactor=args.pdb_bfactor,
        pdb_max_rotation=args.pdb_max_rotation,
        relion_normalize=args.relion_normalize,
        streaming_mmap=args.streaming_mmap,
        streaming_chunk_size=args.streaming_chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
