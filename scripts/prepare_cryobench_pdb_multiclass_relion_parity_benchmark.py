#!/usr/bin/env python
"""Prepare a multi-class RELION parity dataset from a directory of PDBs."""

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
from recovar.simulation import simulate_scattering_potential as ssp
from recovar.simulation.trajectory_generation import compute_bfactor_scaling
from recovar.utils.helpers import write_relion_mrc


logger = logging.getLogger(__name__)


def _pdb_files(pdb_dir: Path) -> list[Path]:
    files = sorted(path for path in pdb_dir.glob("*.pdb") if not path.name.startswith("."))
    if not files:
        raise ValueError(f"No .pdb files found in {pdb_dir}")
    return files


def _volume_prefix(output_dir: Path) -> Path:
    return output_dir / "pdb_volumes" / "vol"


def _generate_volumes_from_pdbs(
    pdb_dir: Path,
    output_dir: Path,
    *,
    grid_size: int,
    voxel_size: float,
    pdb_bfactor: float,
    force: bool,
) -> tuple[Path, list[dict]]:
    pdbs = _pdb_files(pdb_dir)
    prefix = _volume_prefix(output_dir)
    volume_dir = prefix.parent
    mkdir_safe(str(volume_dir))

    expected = [Path(f"{prefix}{idx:04d}.mrc") for idx in range(len(pdbs))]
    if not force and all(path.exists() for path in expected):
        logger.info("Reusing %d existing PDB volumes under %s", len(pdbs), volume_dir)
    else:
        scaling = compute_bfactor_scaling((grid_size, grid_size, grid_size), voxel_size, pdb_bfactor)
        for idx, pdb_path in enumerate(pdbs):
            out_path = Path(f"{prefix}{idx:04d}.mrc")
            logger.info("Generating volume %03d/%03d from %s", idx + 1, len(pdbs), pdb_path)
            ft_mol = ssp.generate_molecule_spectrum_from_pdb_id(
                str(pdb_path),
                voxel_size=voxel_size,
                grid_size=grid_size,
                force_symmetry=True,
            )
            ft_mol = ft_mol.reshape((grid_size, grid_size, grid_size)) * scaling
            vol = np.real(ftu.get_idft3(jnp.asarray(ft_mol))).astype(np.float32)
            utils.write_mrc(str(out_path), vol, voxel_size=voxel_size)

    manifest = [
        {
            "class_index": idx,
            "class_number": idx + 1,
            "pdb_path": str(pdb_path),
            "volume_path": str(expected[idx]),
        }
        for idx, pdb_path in enumerate(pdbs)
    ]
    with (output_dir / "class_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return prefix, manifest


def _required_dataset_files(output_dir: Path, grid_size: int) -> list[Path]:
    return [
        output_dir / "particles.star",
        output_dir / f"particles.{grid_size}.mrcs",
        output_dir / "poses.pkl",
        output_dir / "ctf.pkl",
        output_dir / "simulation_info.pkl",
    ]


def _write_class_references(output_dir: Path, grid_size: int, n_classes: int, init_radius: int) -> None:
    dataset = load_dataset(
        str(output_dir / f"particles.{grid_size}.mrcs"),
        poses_file=str(output_dir / "poses.pkl"),
        ctf_file=str(output_dir / "ctf.pkl"),
        lazy=True,
    )
    sim_info = utils.pickle_load(output_dir / "simulation_info.pkl")
    heterogeneous_volumes = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    init_mask = dataset.get_valid_frequency_indices(rad=init_radius)

    class_rows = []
    for class_idx in range(n_classes):
        class_no = class_idx + 1
        gt_ft = np.asarray(heterogeneous_volumes.volumes[class_idx])
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
                "rlnClassDistribution": 1.0 / float(n_classes),
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
    logger.info("Wrote %d RELION class references", n_classes)


def prepare_benchmark(
    output_dir: Path,
    *,
    pdb_dir: Path,
    n_images: int,
    grid_size: int,
    voxel_size: float | None,
    noise_level: float,
    init_radius: int,
    pdb_bfactor: float,
    relion_normalize: bool,
    streaming_mmap: bool,
    streaming_chunk_size: int,
    disc_type: str,
    seed: int,
    force_volumes: bool,
) -> None:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if voxel_size is None:
        voxel_size = 4.25 * 128 / grid_size

    mkdir_safe(str(output_dir))
    np.random.seed(seed)
    volume_prefix, manifest = _generate_volumes_from_pdbs(
        pdb_dir,
        output_dir,
        grid_size=grid_size,
        voxel_size=voxel_size,
        pdb_bfactor=pdb_bfactor,
        force=force_volumes,
    )
    n_classes = len(manifest)

    required = _required_dataset_files(output_dir, grid_size)
    if not all(path.exists() for path in required):
        logger.info(
            "Generating particles: output=%s classes=%d n_images=%d grid=%d noise=%.4g relion_normalize=%s streaming=%s disc_type=%s",
            output_dir,
            n_classes,
            n_images,
            grid_size,
            noise_level,
            relion_normalize,
            streaming_mmap,
            disc_type,
        )
        simulator.generate_synthetic_dataset(
            str(output_dir),
            voxel_size,
            str(volume_prefix),
            n_images,
            grid_size=grid_size,
            volume_distribution=np.full(n_classes, 1.0 / float(n_classes), dtype=np.float64),
            dataset_params_option="uniform",
            noise_level=noise_level,
            noise_model="white",
            put_extra_particles=False,
            percent_outliers=0.0,
            volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0.0,
            contrast_std=0.0,
            disc_type=disc_type,
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
    sim_info["cryobench_pdb_multiclass_generation"] = {
        "source": "recovar.simulation.simulate_scattering_potential.generate_molecule_spectrum_from_pdb_id",
        "pdb_dir": str(pdb_dir),
        "n_classes": n_classes,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "noise_level": noise_level,
        "pdb_bfactor": pdb_bfactor,
        "seed": seed,
        "disc_type": disc_type,
        "class_manifest": manifest,
    }
    utils.pickle_dump(sim_info, sim_info_path)

    _write_class_references(output_dir, grid_size, n_classes, init_radius)
    with (output_dir / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(sim_info["cryobench_pdb_multiclass_generation"], f, indent=2)
    logger.info("Benchmark ready at %s", output_dir)
    logger.info("RELION --ref %s", output_dir / "reference_init_classes_relion.star")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/home/mg6942/mytigress/cryobench2/Ribosembly/pdbs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=100000)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--noise-level", "--snr", dest="noise_level", type=float, default=1.0)
    parser.add_argument("--init-radius", type=int, default=10)
    parser.add_argument("--pdb-bfactor", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relion-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-chunk-size", type=int, default=1000)
    parser.add_argument(
        "--disc-type",
        choices=("cubic", "linear_interp", "nearest", "nufft"),
        default="cubic",
        help="Projection discretization for synthetic particles. Default cubic avoids slow NUFFT generation.",
    )
    parser.add_argument("--force-volumes", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_benchmark(
        args.output_dir,
        pdb_dir=args.pdb_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        noise_level=args.noise_level,
        init_radius=args.init_radius,
        pdb_bfactor=args.pdb_bfactor,
        relion_normalize=args.relion_normalize,
        streaming_mmap=args.streaming_mmap,
        streaming_chunk_size=args.streaming_chunk_size,
        disc_type=args.disc_type,
        seed=args.seed,
        force_volumes=args.force_volumes,
    )


if __name__ == "__main__":
    main()
