#!/usr/bin/env python
"""Generate regression baselines by running OLD ~/recovar on a canonical dataset.

This script:
1. Generates a canonical test dataset (PDB volumes, seed=42, GT union mask)
   using the current (new) code.
2. Runs the OLD ~/recovar pipeline on that dataset.
3. Computes metrics on the old pipeline output using the current code's
   metric functions (for consistency with test evaluation).
4. Saves the old-code scores as baseline JSON files.

The resulting baselines are committed to tests/baselines/ and represent
the ground truth that the new code must match or beat.

Usage (from within the new repo's pixi env):
    pixi run python scripts/generate_old_baselines.py \
        --output-dir /scratch/.../baseline_output \
        --old-conda-env recovar \
        [--grid-size 128] [--n-images 50000] [--tomo-tilts 7]

Requires:
    - ~/recovar with a working conda env
    - Current repo set up with pixi
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--old-conda-env", default="recovar",
                        help="Conda env name for ~/recovar (default: recovar)")
    parser.add_argument("--old-repo", default=os.path.expanduser("~/recovar"),
                        help="Path to old recovar repo (default: ~/recovar)")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-images", type=int, default=50000)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--contrast-std", type=float, default=0.1)
    parser.add_argument("--tomo-tilts", type=int, default=-1,
                        help="Number of tilts for cryo-ET (default -1 = SPA)")
    parser.add_argument("--noise-model", default="radial",
                        help="Noise model (default: radial)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import new code modules
    from recovar.commands import run_test_all_metrics as rtam
    from recovar.commands import pipeline
    from recovar.output import output, metrics, plot_utils
    from recovar.simulation import synthetic_dataset
    from recovar.commands.run_test_all_metrics import (
        make_big_test_dataset, load_u_real_for_metrics,
    )
    from recovar import utils
    import recovar.core.fourier_transform_utils as ftu

    grid_size = args.grid_size
    is_tomo = args.tomo_tilts > 0

    # ================================================================
    # Step 1: Generate canonical dataset with NEW code
    # ================================================================
    logger.info("=== Step 1: Generate canonical dataset ===")
    dataset_dir = str(out_dir / "test_dataset")

    # Generate PDB trajectory volumes
    from recovar.simulation.trajectory_generation import generate_trajectory_volumes
    gen_prefix = str(out_dir / "generated_volumes" / "vol")
    vol_input = generate_trajectory_volumes(
        output_dir=str(out_dir),
        grid_size=grid_size,
        n_volumes=50,
        voxel_size=4.25 * 128 / grid_size,
        Bfactor=80,
        max_rotation_degrees=10.0,
        output_prefix=gen_prefix,
    )
    logger.info("Generated PDB volumes at: %s", vol_input)

    # Generate dataset with fixed seed
    sim_info = make_big_test_dataset(
        vol_input, str(out_dir),
        noise_level=args.noise_level,
        grid_size=grid_size,
        n_images=args.n_images,
        contrast_std=args.contrast_std,
        n_tilts=args.tomo_tilts,
        noise_increase_per_tilt=None,
    )

    sim_info_path = os.path.join(dataset_dir, "simulation_info.pkl")
    logger.info("Dataset generated at: %s", dataset_dir)

    # Compute GT union mask
    gt_thing = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    volume_shape = (grid_size, grid_size, grid_size)
    gt_soft_mask, gt_binary_mask = metrics.make_union_gt_mask_from_hvd(gt_thing, volume_shape)
    gt_mask_path = os.path.join(dataset_dir, "gt_masks", "gt_union_mask.mrc")
    os.makedirs(os.path.dirname(gt_mask_path), exist_ok=True)
    utils.write_mrc(gt_mask_path, gt_soft_mask, voxel_size=4.25 * 128 / grid_size)
    logger.info("GT union mask saved: %s (%.1f%% voxels)",
                gt_mask_path, 100 * gt_binary_mask.sum() / gt_binary_mask.size)

    # ================================================================
    # Step 2: Run OLD pipeline via conda subprocess
    # ================================================================
    logger.info("=== Step 2: Run OLD pipeline ===")
    old_pipe_dir = os.path.join(dataset_dir, "pipeline_output_old")

    particles_path = (f"{dataset_dir}/particles.star" if is_tomo
                      else f"{dataset_dir}/particles.{grid_size}.mrcs")

    old_cmd = [
        "conda", "run", "-n", args.old_conda_env,
        "python", "-c", textwrap.dedent(f"""\
        import argparse, sys, os
        sys.path.insert(0, '{args.old_repo}')
        from recovar.commands import pipeline
        parser = pipeline.add_args(argparse.ArgumentParser())
        cmd = [
            '{particles_path}',
            '--poses', '{dataset_dir}/poses.pkl',
            '--ctf', '{dataset_dir}/ctf.pkl',
            '-o', '{old_pipe_dir}',
            '--mask', '{gt_mask_path}',
            '--lazy',
            '--zdim', '1,2,4,10,20',
        ]
        if {args.tomo_tilts} > 0:
            cmd += ['--noise-model', '{args.noise_model}']
        args = parser.parse_args(cmd)
        pipeline.standard_recovar_pipeline(args)
        print('OLD pipeline done')
        """),
    ]
    logger.info("Running: %s", " ".join(old_cmd[:6]) + " ...")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = subprocess.run(old_cmd, env=env, capture_output=False)
    if result.returncode != 0:
        logger.error("OLD pipeline failed with exit code %d", result.returncode)
        sys.exit(1)

    # ================================================================
    # Step 3: Compute metrics on OLD output using NEW code
    # ================================================================
    logger.info("=== Step 3: Compute metrics on OLD pipeline output ===")
    gt_mean = gt_thing.get_mean()
    u_gt, s_gt, _ = gt_thing.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)

    po = output.PipelineOutput(old_pipe_dir)
    ds = po.get("lazy_dataset")
    voxel_size = ds.voxel_size if not isinstance(ds, list) else ds[0].voxel_size
    vol_norm = np.sqrt(np.prod(volume_shape))

    scores = {}

    # Mean FSC
    _, scores["mean_fsc"] = plot_utils.plot_fsc_new(
        gt_mean, po.get("mean"), np.array(volume_shape), voxel_size,
        threshold=0.5, name="Mean FSC",
    )

    # Variance FSC: GT Fourier variance vs variance_est['combined']
    cov_sqrt = gt_thing.get_covariance_square_root(contrasted=False)
    gt_fourier_var = np.sum(np.abs(cov_sqrt) ** 2, axis=-1)
    est_var = np.asarray(po.get("variance_est")["combined"])
    _, scores["variance_fsc"] = plot_utils.plot_fsc_new(
        gt_fourier_var, est_var, np.array(volume_shape), voxel_size,
        threshold=0.5, name="Variance FSC",
    )

    # SVD relative variance
    u_real = load_u_real_for_metrics(po, 20)
    n_pcs = int(u_real.shape[0])
    u_est = np.array(u_real.reshape(n_pcs, -1)).T * vol_norm
    _, rel_var, _ = metrics.get_all_variance_scores(u_est, u_gt, s_gt)
    if rel_var.size > 4:
        scores["svd_relative_variance_4"] = float(rel_var[4])
    if rel_var.size > 10:
        scores["svd_relative_variance_10"] = float(rel_var[10])

    # Contrasts + embedding
    # Use unsorted (original image order) embeddings to match GT ordering.
    from recovar.commands.run_test_all_metrics import load_unsorted_embedding_component
    with open(sim_info_path, "rb") as f:
        si = pickle.load(f)
    pa = np.asarray(si["image_assignment"]).ravel()
    gt_contrasts = np.asarray(si["per_image_contrast"]).ravel()

    for zdim in [4, 10]:
        # Try new key names, fall back to old (zs/contrasts) for old pipeline output
        try:
            zs = np.asarray(load_unsorted_embedding_component(po, "latent_coords", zdim))
        except KeyError:
            zs = np.asarray(load_unsorted_embedding_component(po, "zs", zdim))
        _, ratio = metrics.variance_of_zs(zs, pa)
        scores[f"embedding_squared_error_{zdim}"] = float(ratio)
        try:
            c = np.asarray(load_unsorted_embedding_component(po, "contrasts_noreg", zdim)).ravel()
        except KeyError:
            c = np.asarray(load_unsorted_embedding_component(po, "contrasts", zdim)).ravel()
        scores[f"contrast_abs_error_{zdim}"] = float(np.mean(np.abs(gt_contrasts - c)))

    # Noise
    gt_noise = np.asarray(si["noise_variance"]).ravel()
    est_noise = np.asarray(po.get("noise_var_used"))
    n_sh = min(len(gt_noise), len(est_noise))
    scores["noise_correlation"] = float(np.corrcoef(gt_noise[:n_sh], est_noise[:n_sh])[0, 1])
    rel_err = np.abs(gt_noise[:n_sh] - est_noise[:n_sh]) / (gt_noise[:n_sh] + 1e-12)
    scores["noise_mean_relative_error"] = float(np.mean(rel_err))
    scores["noise_median_relative_error"] = float(np.median(rel_err))
    scores["noise_max_relative_error"] = float(np.max(rel_err))

    # State locres with GT union mask
    synt = gt_thing
    for l_idx in range(min(2, synt.volumes.shape[0])):
        gt_map = np.asarray(ftu.get_idft3(synt.volumes[l_idx].reshape(volume_shape)).real)
        state_path = os.path.join(old_pipe_dir, "state", f"state{l_idx:03d}.mrc")
        if os.path.exists(state_path):
            est_map = utils.load_mrc(state_path)
            em = metrics.compute_volume_error_metrics_from_gt(
                gt_map, est_map, voxel_size, gt_binary_mask,
            )
            scores[f"state_{l_idx}_locres_median"] = em.get("median_locres")
            scores[f"state_{l_idx}_locres_90pct"] = em.get("ninety_pc_locres")

    # Save
    scores_path = out_dir / "old_baseline_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    logger.info("OLD baseline scores saved to: %s", scores_path)

    logger.info("\n=== OLD BASELINE SCORES ===")
    for k, v in sorted(scores.items()):
        logger.info("  %s: %s", k, v)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
