#!/usr/bin/env python
"""Run the high-SNR comparison between our EM and RELION.

This script generates the synthetic dataset, runs our EM (with both own-FSC
and oracle modes), and computes FSC vs ground truth.

RELION must be run separately via Slurm (see run_relion_highsnr.sh).
After RELION finishes, run extract_relion_reference.py and then
compare_highsnr_results.py to generate the comparison tables.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONNOUSERSITE=1 pixi run python scripts/run_highsnr_comparison.py

Dataset output: /scratch/gpfs/GILLES/mg6942/tmp/em_comparison_highsnr/
"""

import logging
import time

import numpy as np
import jax.numpy as jnp
import mrcfile
import starfile

from importlib.resources import files
import recovar
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.output.output import mkdir_safe
from recovar.reconstruction import regularization, noise
from recovar.simulation import simulator, synthetic_dataset
from recovar import utils, em
from recovar.em.dense_single_volume.refine import refine_single_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUT = "/scratch/gpfs/GILLES/mg6942/tmp/em_comparison_highsnr/"


def generate_dataset():
    """Generate synthetic dataset with noise_level=0.1."""
    mkdir_safe(OUT)
    simulator.generate_synthetic_dataset(
        OUT,
        4.25 * 128 / 128,
        str(files(recovar) / "assets" / "vol"),
        5000,
        grid_size=128,
        volume_distribution=np.array([1, 0, 0]),
        dataset_params_option="uniform",
        noise_level=0.1,
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
    logger.info("Dataset generated at %s", OUT)


def write_reference_volumes():
    """Write low-pass init and GT reference volumes as MRC."""
    cryo = load_dataset(
        f"{OUT}particles.128.mrcs",
        poses_file=f"{OUT}poses.pkl",
        ctf_file=f"{OUT}ctf.pkl",
        lazy=False,
    )
    sim_info = utils.pickle_load(f"{OUT}simulation_info.pkl")
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info).get_mean()
    init = gt * cryo.get_valid_frequency_indices(rad=5)
    init_3d = np.real(
        np.fft.ifftn(np.fft.ifftshift(init.reshape(cryo.volume_shape)))
    ).astype(np.float32)
    with mrcfile.new(f"{OUT}reference_init.mrc", overwrite=True) as m:
        m.set_data(np.transpose(init_3d, (2, 1, 0)))
        m.voxel_size = cryo.voxel_size
    gt_3d = np.real(
        np.fft.ifftn(np.fft.ifftshift(gt.reshape(cryo.volume_shape)))
    ).astype(np.float32)
    with mrcfile.new(f"{OUT}reference_gt.mrc", overwrite=True) as m:
        m.set_data(np.transpose(gt_3d, (2, 1, 0)))
        m.voxel_size = cryo.voxel_size
    logger.info("Reference volumes written")


def run_our_code():
    """Run our EM with own-FSC and oracle modes."""
    cryo = load_dataset(
        f"{OUT}particles.128.mrcs",
        poses_file=f"{OUT}poses.pkl",
        ctf_file=f"{OUT}ctf.pkl",
        lazy=False,
    )
    sim_info = utils.pickle_load(f"{OUT}simulation_info.pkl")
    gt_vol = synthetic_dataset.load_heterogeneous_reconstruction(sim_info).get_mean()
    nv = noise.make_radial_noise(sim_info["noise_variance"], cryo.image_shape)
    sv = utils.make_radial_image(
        regularization.average_over_shells(np.abs(gt_vol) ** 2, cryo.volume_shape),
        cryo.volume_shape,
    )
    init_ft = (gt_vol * cryo.get_valid_frequency_indices(rad=5)).reshape(-1)

    rotations = utils.R_from_relion(em.sampling.get_rotation_grid(3))
    translations = em.sampling.get_translation_grid(3, 1)

    # Get RELION's half-set split
    data = starfile.read(f"{OUT}relion_ref/run_it001_data.star")
    subsets = data["particles"]["rlnRandomSubset"].values
    half1_idx = np.where(subsets == 1)[0]
    half2_idx = np.where(subsets == 2)[0]
    logger.info("Half-set 1: %d, Half-set 2: %d", len(half1_idx), len(half2_idx))

    cryo1 = cryo.subset(half1_idx)
    cryo2 = cryo.subset(half2_idx)

    # Run with own FSC
    logger.info("=== Run: own FSC ===")
    t0 = time.time()
    result_own = refine_single_volume(
        [cryo1, cryo2],
        init_ft, nv, sv,
        rotations, translations,
        "linear_interp",
        max_iter=10,
        image_batch_size=500,
        rotation_block_size=5000,
        adaptive_oversampling=0,
    )
    own_total = time.time() - t0
    logger.info("Own-FSC total: %.0fs", own_total)

    # Run with RELION's current_sizes (oracle)
    relion_sizes = []
    for it in range(9):
        d = dict(np.load(f"{OUT}relion_ref_npz/iteration_{it:03d}.npz", allow_pickle=True))
        relion_sizes.append(int(d["current_image_size"]))

    logger.info("=== Run: oracle ===")
    t0 = time.time()
    result_oracle = refine_single_volume(
        [cryo1, cryo2],
        init_ft, nv, sv,
        rotations, translations,
        "linear_interp",
        max_iter=10,
        image_batch_size=500,
        rotation_block_size=5000,
        adaptive_oversampling=0,
        relion_current_sizes=relion_sizes[1:],
    )
    oracle_total = time.time() - t0
    logger.info("Oracle total: %.0fs", oracle_total)

    # Compute FSC vs GT
    fsc_own = np.asarray(
        regularization.get_fsc_gpu(
            jnp.asarray(result_own["mean"]),
            jnp.asarray(gt_vol),
            cryo.volume_shape,
        )
    )
    fsc_oracle = np.asarray(
        regularization.get_fsc_gpu(
            jnp.asarray(result_oracle["mean"]),
            jnp.asarray(gt_vol),
            cryo.volume_shape,
        )
    )

    # Save
    np.savez(
        f"{OUT}our_results_halfset.npz",
        own_current_sizes=result_own["current_sizes"],
        own_pixel_resolutions=result_own["pixel_resolutions"],
        own_wall_times=result_own["wall_times"],
        own_fsc_history=np.array(result_own["fsc_history"], dtype=object),
        own_mean=np.asarray(result_own["mean"]),
        own_means_h1=np.asarray(result_own["means"][0]),
        own_means_h2=np.asarray(result_own["means"][1]),
        oracle_current_sizes=result_oracle["current_sizes"],
        oracle_pixel_resolutions=result_oracle["pixel_resolutions"],
        oracle_wall_times=result_oracle["wall_times"],
        oracle_fsc_history=np.array(result_oracle["fsc_history"], dtype=object),
        oracle_mean=np.asarray(result_oracle["mean"]),
        oracle_means_h1=np.asarray(result_oracle["means"][0]),
        oracle_means_h2=np.asarray(result_oracle["means"][1]),
        fsc_own_vs_gt=fsc_own,
        fsc_oracle_vs_gt=fsc_oracle,
        half1_idx=half1_idx,
        half2_idx=half2_idx,
    )
    logger.info("Results saved")

    # Print summary
    print("\nFSC vs GT:")
    print(f"{'Shell':>6s}  {'Own':>8s}  {'Oracle':>8s}")
    for s in range(0, 65, 5):
        if s < len(fsc_own):
            print(f"  {s:4d}  {fsc_own[s]:8.4f}  {fsc_oracle[s]:8.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_dataset()
        write_reference_volumes()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run_our_code()
    else:
        print("Usage: python scripts/run_highsnr_comparison.py [generate|run]")
        print("  generate: create synthetic dataset and reference volumes")
        print("  run: run our EM (requires RELION to have run first)")
