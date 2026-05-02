#!/usr/bin/env python
"""Generate a small K-class RELION parity benchmark dataset.

The single-class parity fixture starts RELION from one low-pass reference.
For K-class parity we instead write a RELION reference STAR with one initial
map per class, so the first validation measures joint class/pose EM behavior
rather than stochastic class discovery from one seed map.
"""

import argparse
import logging
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

import recovar
from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.output.output import mkdir_safe, save_volume
from recovar.simulation import simulator, synthetic_dataset
from recovar.utils.helpers import write_relion_mrc

import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _class_distribution(n_classes: int) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("multiclass benchmark requires n_classes >= 2")
    # recovar/assets currently ships three benchmark volumes. Keep unused
    # trailing assets at probability zero so the simulator can load the folder
    # unchanged while this fixture controls the active number of classes.
    dist = np.zeros(3, dtype=np.float64)
    if n_classes > dist.size:
        raise ValueError(f"n_classes={n_classes} exceeds available asset volumes ({dist.size})")
    dist[:n_classes] = 1.0 / float(n_classes)
    return dist


def _write_class_references(output_dir: str, grid_size: int, n_classes: int, init_radius: int):
    dataset = load_dataset(
        os.path.join(output_dir, f"particles.{grid_size}.mrcs"),
        poses_file=os.path.join(output_dir, "poses.pkl"),
        ctf_file=os.path.join(output_dir, "ctf.pkl"),
        lazy=False,
    )
    sim_info = utils.pickle_load(os.path.join(output_dir, "simulation_info.pkl"))
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    class_vols = np.asarray(hvd.volumes[:n_classes])
    valid_init = dataset.get_valid_frequency_indices(rad=init_radius)

    class_rows = []
    for class_idx in range(n_classes):
        class_no = class_idx + 1
        gt_ft = class_vols[class_idx]
        init_ft = gt_ft * valid_init

        save_volume(
            np.asarray(gt_ft.reshape(-1)),
            os.path.join(output_dir, f"reference_gt_class{class_no:03d}"),
            volume_shape=dataset.volume_shape,
            from_ft=True,
            voxel_size=dataset.voxel_size,
        )
        save_volume(
            np.asarray(init_ft.reshape(-1)),
            os.path.join(output_dir, f"reference_init_class{class_no:03d}"),
            volume_shape=dataset.volume_shape,
            from_ft=True,
            voxel_size=dataset.voxel_size,
        )

        init_real = np.real(ftu.get_idft3(jnp.asarray(init_ft).reshape(dataset.volume_shape)))
        relion_ref = Path(output_dir) / f"reference_init_class{class_no:03d}_relion.mrc"
        write_relion_mrc(relion_ref, np.asarray(init_real), voxel_size=dataset.voxel_size)
        class_rows.append(
            {
                "rlnReferenceImage": relion_ref.name,
                "rlnClassDistribution": 1.0 / float(n_classes),
            }
        )

    # RELION's MlModel::initialiseFromImages reads the model_classes table
    # and uses _rlnReferenceImage entries as the class references.
    starfile.write(
        {"model_classes": pd.DataFrame(class_rows)},
        Path(output_dir) / "reference_init_classes_relion.star",
        overwrite=True,
    )

    mean_ft = hvd.get_mean()
    save_volume(
        np.asarray(mean_ft.reshape(-1)),
        os.path.join(output_dir, "reference_gt"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )
    logger.info("Wrote %d class references in %s", n_classes, output_dir)


def prepare_benchmark(
    output_dir: str,
    *,
    n_images: int,
    grid_size: int,
    noise_level: float,
    n_classes: int,
    init_radius: int,
    relion_normalize: bool,
):
    mkdir_safe(output_dir)

    required = [
        Path(output_dir) / "particles.star",
        Path(output_dir) / f"particles.{grid_size}.mrcs",
        Path(output_dir) / "poses.pkl",
        Path(output_dir) / "ctf.pkl",
        Path(output_dir) / "simulation_info.pkl",
    ]
    if not all(path.exists() for path in required):
        voxel_size = 4.25 * 128 / grid_size
        volumes_path = str(files(recovar) / "assets" / "vol")
        logger.info(
            "Generating K-class benchmark: output=%s n_images=%d grid=%d noise=%.3f K=%d relion_normalize=%s",
            output_dir,
            n_images,
            grid_size,
            noise_level,
            n_classes,
            relion_normalize,
        )
        simulator.generate_synthetic_dataset(
            output_dir,
            voxel_size,
            volumes_path,
            n_images,
            grid_size=grid_size,
            volume_distribution=_class_distribution(n_classes),
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
            outlier_file_input=None,
            relion_normalize=relion_normalize,
        )
    else:
        logger.info("Dataset files already present in %s; reusing them", output_dir)

    _write_class_references(output_dir, grid_size, n_classes, init_radius)
    logger.info("Benchmark ready at %s", output_dir)
    logger.info("RELION --ref: %s", Path(output_dir) / "reference_init_classes_relion.star")


def main():
    parser = argparse.ArgumentParser(description="Prepare a K-class RELION parity benchmark dataset")
    parser.add_argument(
        "--output-dir",
        default="/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_k2_1k_64",
        help="Output directory for particles, metadata, and class references",
    )
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--init-radius", type=int, default=5)
    parser.add_argument("--relion-normalize", action="store_true")
    args = parser.parse_args()

    prepare_benchmark(
        args.output_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        noise_level=args.noise_level,
        n_classes=args.n_classes,
        init_radius=args.init_radius,
        relion_normalize=args.relion_normalize,
    )


if __name__ == "__main__":
    main()
