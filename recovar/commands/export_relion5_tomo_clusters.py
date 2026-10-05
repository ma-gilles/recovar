"""Export K-means particle subsets for native RELION 5 tomography refinement."""

import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="Pipeline/job_NNNN directory")
    parser.add_argument("--analysis", required=True, help="Analyze job directory containing data/kmeans_result.pkl")
    parser.add_argument("--particles", required=True, help="Native RELION 5 particle STAR to subset")
    parser.add_argument("--tomograms", required=True, help="Matching native RELION 5 tomograms.star")
    parser.add_argument("--outdir", required=True, help="New or empty output directory")
    parser.add_argument("--clusters", help="Comma-separated cluster IDs to export, e.g. 0,2 (default: all)")
    parser.add_argument("--datadir", help="RELION project root for relative stack paths (default: --particles folder)")
    args = parser.parse_args()
    clusters = None if args.clusters is None else [int(c) for c in args.clusters.split(",")]
    # Some historical params pickles contain JAX arrays; no GPU is needed to unpickle them.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from recovar.data_io.relion5_cluster_export import export_clusters

    summary = export_clusters(**{**vars(args), "clusters": clusters})
    print(summary.to_string(index=False))
    print(f"All classes (rlnClassNumber = cluster ID + 1): {os.path.abspath(args.outdir)}/particles_classes.star")


if __name__ == "__main__":
    main()
