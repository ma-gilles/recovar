#!/usr/bin/env python
"""Run N iterations of recovar in RELION mode, save results for diff comparison.

Usage:
  pixi run python scripts/run_multi_iter_parity.py \
    --relion_dir .../relion_ref_os0 \
    --data_star .../particles.star \
    --iter 3 --max_iter 15 \
    --output_dir .../recovar_15iter
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--data_star", required=True)
    parser.add_argument("--iter", type=int, default=3, help="RELION iteration to start from")
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_healpix_order", type=int, default=8)
    args = parser.parse_args()

    import jax.numpy as jnp
    import starfile

    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.refine import refine_single_volume
    from recovar.reconstruction import noise as recon_noise
    from recovar.utils import helpers

    relion_dir = Path(args.relion_dir)
    iteration = args.iter
    prefix = str(relion_dir / f"run_it{iteration:03d}")

    # ---- Load RELION state ----
    model_h1 = starfile.read(f"{prefix}_half1_model.star")
    N = int(model_h1["model_general"]["rlnOriginalImageSize"])
    current_size = int(model_h1["model_general"]["rlnCurrentImageSize"])
    pixel_size = float(model_h1["model_general"]["rlnPixelSize"])

    sigma2 = np.array(model_h1["model_optics_group_1"]["rlnSigma2Noise"])
    class1 = model_h1["model_class_1"]
    tau2_col = "rlnReferenceSigma2" if "rlnReferenceSigma2" in class1 else "rlnReferenceTau2"
    tau2 = np.array(class1[tau2_col])
    fsc_col = "rlnGoldStandardFsc" if "rlnGoldStandardFsc" in class1 else "rlnFourierShellCorrelationCorrected"
    fsc = np.array(class1[fsc_col])

    opt_text = (relion_dir / f"run_it{iteration:03d}_optimiser.star").read_text()
    m_pd = re.search(r"_rlnParticleDiameter\s+(\S+)", opt_text)
    particle_diameter = float(m_pd.group(1)) if m_pd else 544.0
    m_os = re.search(r"_rlnAdaptiveOversampleOrder\s+(\d+)", opt_text)
    oversampling = int(m_os.group(1)) if m_os else 0

    samp_text = (relion_dir / f"run_it{iteration:03d}_sampling.star").read_text()
    m_hp = re.search(r"_rlnHealpixOrder\s+(\d+)", samp_text)
    hp_order = int(m_hp.group(1)) if m_hp else 3
    m_range = re.search(r"_rlnOffsetRange\s+(\S+)", samp_text)
    m_step = re.search(r"_rlnOffsetStep\s+(\S+)", samp_text)
    offset_range = float(m_range.group(1)) if m_range else 12.75
    offset_step = float(m_step.group(1)) if m_step else 4.25

    # ave_Pmax from per-particle data
    relion_data = starfile.read(f"{prefix}_data.star")
    relion_df = relion_data["particles"] if isinstance(relion_data, dict) else relion_data
    ave_Pmax = float(np.mean(np.array(relion_df["rlnMaxValueProbDistribution"])))

    # has_high_fsc_at_limit (sticky flag)
    has_high_fsc_at_limit = False
    for it in range(1, iteration + 1):
        try:
            m = starfile.read(str(relion_dir / f"run_it{it:03d}_half1_model.star"))
            fc = np.array(m["model_class_1"][fsc_col])
            oc = (relion_dir / f"run_it{it:03d}_optimiser.star").read_text()
            cs_it = (
                int(re.search(r"_rlnCurrentImageSize\s+(\d+)", oc).group(1))
                if re.search(r"_rlnCurrentImageSize", oc)
                else None
            )
            if cs_it is None:
                mc = starfile.read(str(relion_dir / f"run_it{it:03d}_half1_model.star"))
                cs_it = int(mc["model_general"]["rlnCurrentImageSize"])
            shell_at_limit = cs_it // 2 - 1
            if shell_at_limit < len(fc) and fc[shell_at_limit] > 0.2:
                has_high_fsc_at_limit = True
        except Exception:
            pass

    print(f"RELION state: N={N}, hp={hp_order}, os={oversampling}, cs={current_size}")
    print(f"  pixel_size={pixel_size}, particle_diameter={particle_diameter}")
    print(f"  ave_Pmax={ave_Pmax:.4f}, has_high_fsc_at_limit={has_high_fsc_at_limit}")

    # ---- Init volumes ----
    noise_variance = jnp.asarray(recon_noise.make_radial_noise(sigma2, (N, N)))
    mean_variance = jnp.asarray(utils.make_radial_image(tau2, (N, N, N), extend_last_frequency=True))

    vol_h1 = helpers.load_relion_volume(f"{prefix}_half1_class001.mrc")
    vol_h2 = helpers.load_relion_volume(f"{prefix}_half2_class001.mrc")
    vol_ft_h1 = np.array(ftu.get_dft3(jnp.array(vol_h1)))
    vol_ft_h2 = np.array(ftu.get_dft3(jnp.array(vol_h2)))

    # ---- Dataset + half-set split ----
    ds = load_dataset(args.data_star)
    relion_subsets = np.array(relion_df["rlnRandomSubset"])
    relion_names = list(relion_df["rlnImageName"])
    our_particles = starfile.read(args.data_star)
    our_particles = our_particles["particles"] if isinstance(our_particles, dict) else our_particles
    our_names = list(our_particles["rlnImageName"])

    def _idx(name):
        m = re.match(r"(\d+)@", name)
        return int(m.group(1)) - 1 if m else -1

    relion_idx_map = {_idx(relion_names[i]): relion_subsets[i] for i in range(len(relion_names))}
    our_subsets = np.array([relion_idx_map.get(_idx(n), 0) for n in our_names])
    ds_half1 = ds.subset(np.where(our_subsets == 1)[0])
    ds_half2 = ds.subset(np.where(our_subsets == 2)[0])
    print(f"  Half-sets: {len(np.where(our_subsets == 1)[0])} + {len(np.where(our_subsets == 2)[0])}")

    # ---- Output directory ----
    out_dir = args.output_dir or str(relion_dir.parent / "_agent_scratch" / f"{args.max_iter}iter_parity")
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir).joinpath("SAFE_TO_DELETE").touch()

    # ---- Run ----
    print(f"\nRunning {args.max_iter} iterations...")
    t0 = time.time()
    result = refine_single_volume(
        experiment_datasets=[ds_half1, ds_half2],
        init_volume=[jnp.asarray(vol_ft_h1), jnp.asarray(vol_ft_h2)],
        init_noise_variance=noise_variance.reshape(-1),
        init_mean_variance=mean_variance.reshape(-1),
        rotations=None,
        translations=None,
        disc_type="linear_interp",
        mode="relion",
        max_iter=args.max_iter,
        image_batch_size=500,
        rotation_block_size=5000,
        init_current_size=current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=oversampling,
        init_healpix_order=hp_order,
        max_healpix_order=args.max_healpix_order,
        init_translation_range=offset_range / pixel_size,
        init_translation_step=offset_step / pixel_size,
        init_translation_sigma_angstrom=10.0,
        particle_diameter_ang=particle_diameter,
        tau2_fudge=1.0,
        perturb_factor=0.0,
        init_fsc=fsc,
        init_ave_Pmax=ave_Pmax,
        init_has_high_fsc_at_limit=has_high_fsc_at_limit,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted {args.max_iter} iterations in {elapsed:.1f}s")

    # ---- Save results ----
    save_dict = {
        "volume_shape": np.array([N, N, N]),
        "voxel_size": np.float64(pixel_size),
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
    }
    if result.get("ave_Pmax_trajectory"):
        save_dict["ave_Pmax_trajectory"] = np.array(result["ave_Pmax_trajectory"])
    if result.get("healpix_order_trajectory"):
        save_dict["healpix_order_trajectory"] = np.array(result["healpix_order_trajectory"])
    for traj_name, prefix_name in [
        ("fsc_history", "fsc_iter"),
        ("data_vs_prior_trajectory", "data_vs_prior_iter"),
        ("noise_radial_trajectory", "noise_radial_iter"),
        ("tau2_radial_trajectory", "tau2_radial_iter"),
        ("significant_counts", "sig_counts_iter"),
    ]:
        if result.get(traj_name):
            for i, arr_i in enumerate(result[traj_name]):
                save_dict[f"{prefix_name}_{i:03d}"] = np.array(arr_i)

    npz_path = os.path.join(out_dir, "refinement_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved: {npz_path}")

    # ---- Summary table ----
    n_iters = len(result["current_sizes"])
    print(f"\n{'iter':>4} {'cs':>4} {'pixres':>6} {'pmax':>8} {'hp':>3} {'FSC@0.5':>8} {'res(A)':>8}")
    print("-" * 50)
    for i in range(n_iters):
        cs_i = result["current_sizes"][i]
        pr_i = result["pixel_resolutions"][i]
        pmax_i = (
            result["ave_Pmax_trajectory"][i]
            if result.get("ave_Pmax_trajectory") and i < len(result["ave_Pmax_trajectory"])
            else 0
        )
        hp_i = (
            result["healpix_order_trajectory"][i]
            if result.get("healpix_order_trajectory") and i < len(result["healpix_order_trajectory"])
            else hp_order
        )
        fsc_i = (
            np.array(result["fsc_history"][i]) if result.get("fsc_history") and i < len(result["fsc_history"]) else None
        )
        fsc05 = 0
        if fsc_i is not None:
            for s in range(1, len(fsc_i)):
                if fsc_i[s] >= 0.5:
                    fsc05 = s
        res = (N * pixel_size) / max(fsc05, 1)
        print(f"{i + 1:4d} {cs_i:4d} {pr_i:6.1f} {pmax_i:8.4f} {hp_i:3d} {fsc05:8d} {res:8.1f}")

    # ---- Compare final volume with RELION ----
    last_relion_it = iteration + args.max_iter
    for k_half, label in [(0, "half1"), (1, "half2")]:
        target_path = str(relion_dir / f"run_it{last_relion_it:03d}_{label}_class001.mrc")
        if not Path(target_path).exists():
            print(f"  Final {label}: RELION it{last_relion_it:03d} not found")
            continue
        recovar_vol_ft = np.asarray(result["means"][k_half])
        recovar_vol_real = np.real(np.array(ftu.get_idft3(jnp.asarray(recovar_vol_ft.reshape(N, N, N)))))
        relion_vol = helpers.load_relion_volume(target_path)
        corr = float(np.corrcoef(recovar_vol_real.ravel(), relion_vol.ravel())[0, 1])
        print(f"  Final {label} vs RELION it{last_relion_it:03d}: corr={corr:.6f}")

    # ---- Run diff script ----
    print("\n=== Running diff_relion_recovar_per_iter.py ===")
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "scripts/diff_relion_recovar_per_iter.py",
            "--relion_dir",
            str(relion_dir),
            "--recovar_dir",
            out_dir,
            "--relion_start_iter",
            str(iteration),
            "--max_iter",
            str(args.max_iter + 1),
            "--tol",
            "0.05",
            "--shells",
            "10",
        ]
    )


if __name__ == "__main__":
    main()
