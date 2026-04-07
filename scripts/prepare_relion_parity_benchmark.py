#!/usr/bin/env python
"""Generate the reproducible 5k/128 RELION-parity benchmark dataset.

This prepares the synthetic single-volume dataset used by the recovar-vs-RELION
comparison scripts:

- ``particles.star`` / ``particles.128.mrcs``
- ``poses.pkl`` / ``ctf.pkl`` / ``simulation_info.pkl``
- ``reference_init.mrc`` (low-pass initial reference)
- ``reference_gt.mrc`` (ground-truth map)

The output layout matches what ``scripts/run_full_refinement.py``,
``scripts/run_comparison.py``, and ``scripts/compare_vs_relion.py`` expect.
"""

import argparse
import logging
import os
from importlib.resources import files

import mrcfile
import numpy as np

import recovar
from recovar import utils
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.output.output import mkdir_safe
from recovar.simulation import simulator, synthetic_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _write_reference_volumes(output_dir):
    """Write the benchmark init and ground-truth reference maps as MRC.

    Two pairs of files are written:

    * ``reference_init.mrc`` / ``reference_gt.mrc`` — recovar / cryoSPARC /
      cryoDRGN axis convention. These are read by recovar via ``load_mrc``
      and used by recovar's refinement and FSC scripts.

    * ``reference_init_relion.mrc`` / ``reference_gt_relion.mrc`` — RELION
      axis convention (``write_relion_mrc``). RELION reads MRC files raw
      and expects this frame; passing the cryosparc-frame file causes
      RELION to refine into the antipode basin (median pose error ~133°,
      see commit history around 2026-04-08). The RELION reference run
      must use ``--ref reference_init_relion.mrc``.
    """
    from recovar.output.output import save_volume
    from recovar.utils.helpers import write_relion_mrc
    from recovar.core import fourier_transform_utils as ftu
    import jax.numpy as jnp

    dataset = load_dataset(
        os.path.join(output_dir, "particles.128.mrcs"),
        poses_file=os.path.join(output_dir, "poses.pkl"),
        ctf_file=os.path.join(output_dir, "ctf.pkl"),
        lazy=False,
    )
    sim_info = utils.pickle_load(os.path.join(output_dir, "simulation_info.pkl"))
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info).get_mean()
    init = gt * dataset.get_valid_frequency_indices(rad=5)

    # --- recovar-frame MRCs (used by recovar's refine / FSC scripts) ---
    save_volume(
        np.asarray(init.reshape(-1)),
        os.path.join(output_dir, "reference_init"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )
    save_volume(
        np.asarray(gt.reshape(-1)),
        os.path.join(output_dir, "reference_gt"),
        volume_shape=dataset.volume_shape,
        from_ft=True,
        voxel_size=dataset.voxel_size,
    )

    # --- RELION-frame MRCs (used by ``--ref`` in the RELION reference run) ---
    init_real_recovar = np.real(
        ftu.get_idft3(jnp.asarray(init).reshape(dataset.volume_shape))
    )
    write_relion_mrc(
        os.path.join(output_dir, "reference_init_relion.mrc"),
        np.asarray(init_real_recovar),
        voxel_size=dataset.voxel_size,
    )

    gt_real_recovar = np.real(
        ftu.get_idft3(jnp.asarray(gt).reshape(dataset.volume_shape))
    )
    write_relion_mrc(
        os.path.join(output_dir, "reference_gt_relion.mrc"),
        np.asarray(gt_real_recovar),
        voxel_size=dataset.voxel_size,
    )


def prepare_benchmark(output_dir, *, n_images, grid_size, noise_level):
    mkdir_safe(output_dir)

    particles_star = os.path.join(output_dir, "particles.star")
    particles_mrcs = os.path.join(output_dir, f"particles.{grid_size}.mrcs")
    poses_pkl = os.path.join(output_dir, "poses.pkl")
    ctf_pkl = os.path.join(output_dir, "ctf.pkl")
    sim_info_pkl = os.path.join(output_dir, "simulation_info.pkl")

    dataset_ready = all(
        os.path.exists(path)
        for path in (particles_star, particles_mrcs, poses_pkl, ctf_pkl, sim_info_pkl)
    )

    if not dataset_ready:
        voxel_size = 4.25 * 128 / grid_size
        volumes_path = str(files(recovar) / "assets" / "vol")
        logger.info(
            "Generating benchmark dataset: output=%s n_images=%d grid=%d noise=%.3f",
            output_dir,
            n_images,
            grid_size,
            noise_level,
        )
        simulator.generate_synthetic_dataset(
            output_dir,
            voxel_size,
            volumes_path,
            n_images,
            grid_size=grid_size,
            volume_distribution=np.array([1.0, 0.0, 0.0]),
            dataset_params_option="uniform",
            noise_level=noise_level,
            noise_model="white",
            put_extra_particles=False,
            percent_outliers=0,
            volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0,
            contrast_std=0,
            disc_type="nufft",
            n_tilts=-1,
            outlier_file_input=None,
        )
    else:
        logger.info("Dataset files already present in %s; reusing them", output_dir)

    _write_reference_volumes(output_dir)
    logger.info("Benchmark ready at %s", output_dir)
    logger.info("Expected RELION output path: %s", os.path.join(output_dir, "relion_ref"))
    logger.info("Expected extracted reference path: %s", os.path.join(output_dir, "relion_ref_npz"))


def main():
    parser = argparse.ArgumentParser(description="Prepare the RELION parity benchmark dataset")
    parser.add_argument(
        "--output-dir",
        default="/scratch/gpfs/GILLES/mg6942/tmp/relion_parity_benchmark_5k_128_noise1p0",
        help="Output directory for the dataset and reference volumes",
    )
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--noise-level", type=float, default=1.0)
    args = parser.parse_args()

    prepare_benchmark(
        args.output_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        noise_level=args.noise_level,
    )


if __name__ == "__main__":
    main()
