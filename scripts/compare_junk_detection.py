#!/usr/bin/env python
"""Compare junk detection intermediate outputs between OLD and NEW code.

Generates a cryo-ET dataset, runs pipeline + junk detection, and saves
per-cluster FSC scores. Run twice: once with NEW code, once with OLD code
on sys.path. Then compare the cluster-level scores to identify which
clusters are differently classified.

Usage:
    # NEW code
    python scripts/compare_junk_detection.py --output /path/to/output_new

    # OLD code (uses ~/recovar on sys.path)
    python scripts/compare_junk_detection.py --output /path/to/output_old --old

    # Compare
    python scripts/compare_junk_detection.py --compare /path/to/output_old /path/to/output_new
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np


def run_junk_detection(output_dir, use_old=False):
    """Run pipeline + junk detection on a fixed ET dataset."""
    if use_old:
        sys.path.insert(0, os.path.expanduser("~/recovar"))

    from recovar.commands import pipeline, make_test_dataset
    from recovar.commands import junk_particle_detection as jpd
    from recovar.output import output as o

    dataset_dir = os.path.join(output_dir, "dataset")
    test_dataset = os.path.join(dataset_dir, "test_dataset")
    pipe_output = os.path.join(test_dataset, "pipeline_output")
    junk_output = os.path.join(output_dir, "junk_output")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate dataset (same seed always)
    if not os.path.exists(os.path.join(test_dataset, "particles.star")):
        import subprocess
        subprocess.run([
            sys.executable, "-m", "recovar.commands.make_test_dataset",
            dataset_dir,
            "--image-size", "128",
            "--n-images", "10000",
            "--noise-level", "0.1",
            "--seed", "42",
            "--n-tilts", "7",
        ], check=True)

    # 2. Run pipeline
    if not os.path.exists(pipe_output):
        p_parser = pipeline.add_args(argparse.ArgumentParser())
        p_args = p_parser.parse_args([
            os.path.join(test_dataset, "particles.star"),
            "--ctf", os.path.join(test_dataset, "ctf.pkl"),
            "--poses", os.path.join(test_dataset, "poses.pkl"),
            "--mask", "from_halfmaps",
            "-o", pipe_output,
            "--zdim", "4",
            "--lazy",
            "--correct-contrast",
        ])
        pipeline.standard_recovar_pipeline(p_args)

    # 3. Run junk detection with verbose output
    po = o.PipelineOutput(pipe_output)
    cryos = po.get('dataset')
    zdim = 4
    coords_entry = 'latent_coords'
    zs = po.get(coords_entry)[zdim]

    # K-means
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=min(5000, len(zs)))
    km.fit(zs)
    centers = km.cluster_centers_
    labels = km.labels_

    print(f"Dataset: {len(zs)} particles, {cryos[0].n_units}+{cryos[1].n_units} per halfset")
    print(f"K-means: {len(centers)} clusters")

    # 4. Compute per-cluster FSC scores (replicating junk_particle_detection logic)
    from recovar.reconstruction import relion_functions
    from recovar.core import fourier_transform_utils
    from recovar.output import plot_utils

    volume_shape = po.get('volume_shape')
    mean_volume = po.get('mean').reshape(volume_shape)
    mean_real = np.real(fourier_transform_utils.get_idft3(mean_volume))

    batch_size = 500
    n_particles_per_cluster = min(100, max(10, len(zs) // 100))
    print(f"n_particles_per_cluster: {n_particles_per_cluster}")

    cluster_scores = {}
    for ci in range(len(centers)):
        zs_subsets = [zs[:cryos[0].n_units], zs[cryos[0].n_units:]]
        halfmaps = [None, None]

        for hi, zs_sub in enumerate(zs_subsets):
            distances = np.linalg.norm(zs_sub - centers[ci], axis=1)
            closest = np.argsort(distances)[:n_particles_per_cluster]

            Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(
                cryos[hi], None, batch_size,
                disc_type='linear_interp',
                index_subset=closest,
                upsampling_factor=2,
            )
            halfmap = relion_functions.post_process_from_filter_v2(
                Ft_ctf, F_ty, cryos[hi].volume_shape, 2,
                kernel='triangular', use_spherical_mask=True,
                grid_correct=True, gridding_correct="square",
            )
            halfmaps[hi] = halfmap

        # FSC between halfmaps
        h1_real = np.real(fourier_transform_utils.get_idft3(halfmaps[0].reshape(volume_shape)))
        h2_real = np.real(fourier_transform_utils.get_idft3(halfmaps[1].reshape(volume_shape)))
        combined_real = (h1_real + h2_real) / 2

        fsc_hh = plot_utils.compute_fsc(h1_real, h2_real)
        fsc_vs_mean = plot_utils.compute_fsc(combined_real, mean_real)

        fsc_auc = float(np.mean(fsc_hh[1:]))
        fsc_val = float(fsc_hh[len(fsc_hh)//4]) if len(fsc_hh) > 4 else float(fsc_hh[-1])
        fsc_mean_auc = float(np.mean(fsc_vs_mean[1:]))

        cluster_scores[ci] = {
            'fsc_auc': fsc_auc,
            'fsc_quarter': fsc_val,
            'fsc_vs_mean_auc': fsc_mean_auc,
            'n_particles_h0': int(min(n_particles_per_cluster, len(zs_subsets[0]))),
            'n_particles_h1': int(min(n_particles_per_cluster, len(zs_subsets[1]))),
        }

        if ci % 20 == 0:
            print(f"  Cluster {ci}: FSC_AUC={fsc_auc:.4f}, vs_mean={fsc_mean_auc:.4f}")

    # Save
    os.makedirs(junk_output, exist_ok=True)
    with open(os.path.join(junk_output, "cluster_scores.json"), "w") as f:
        json.dump(cluster_scores, f, indent=2)
    with open(os.path.join(junk_output, "cluster_centers.pkl"), "wb") as f:
        pickle.dump(centers, f)
    with open(os.path.join(junk_output, "cluster_labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    # Summary stats
    aucs = [v['fsc_auc'] for v in cluster_scores.values()]
    print(f"\nFSC AUC: mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}, "
          f"min={np.min(aucs):.4f}, max={np.max(aucs):.4f}")
    print(f"Saved to {junk_output}")


def compare_results(dir_old, dir_new):
    """Compare cluster FSC scores between OLD and NEW."""
    with open(os.path.join(dir_old, "junk_output", "cluster_scores.json")) as f:
        old = json.load(f)
    with open(os.path.join(dir_new, "junk_output", "cluster_scores.json")) as f:
        new = json.load(f)

    print(f"{'Cluster':>8} {'OLD_AUC':>10} {'NEW_AUC':>10} {'Diff':>10} {'OLD_vsMean':>12} {'NEW_vsMean':>12}")
    print("-" * 70)
    diffs = []
    for ci in sorted(old.keys(), key=int):
        o_auc = old[ci]['fsc_auc']
        n_auc = new.get(ci, {}).get('fsc_auc', float('nan'))
        o_mean = old[ci]['fsc_vs_mean_auc']
        n_mean = new.get(ci, {}).get('fsc_vs_mean_auc', float('nan'))
        diff = n_auc - o_auc
        diffs.append(diff)
        flag = " ***" if abs(diff) > 0.05 else ""
        print(f"{ci:>8} {o_auc:>10.4f} {n_auc:>10.4f} {diff:>+10.4f} {o_mean:>12.4f} {n_mean:>12.4f}{flag}")

    diffs = np.array(diffs)
    print(f"\nMean diff: {np.mean(diffs):+.4f}, Std: {np.std(diffs):.4f}")
    print(f"Clusters with >5% FSC drop: {np.sum(diffs < -0.05)}")
    print(f"Clusters with >5% FSC gain: {np.sum(diffs > 0.05)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output directory for run")
    parser.add_argument("--old", action="store_true", help="Use ~/recovar (old code)")
    parser.add_argument("--compare", nargs=2, metavar=("OLD_DIR", "NEW_DIR"),
                        help="Compare two output directories")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.output:
        run_junk_detection(args.output, use_old=args.old)
    else:
        parser.print_help()
