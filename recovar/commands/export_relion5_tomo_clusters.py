"""Export K-means particle subsets for native RELION 5 tomography refinement."""

import argparse
import logging
import os


def _cluster_ids(value):
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer cluster IDs, e.g. 0,2") from exc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Only use trusted RECOVAR outputs (pickle files). Reuses original native stacks; "
            "does not copy pixels, reverse-convert poses, or submit jobs."
        ),
    )
    parser.add_argument("--pipeline", required=True, help="Actual Pipeline/job_NNNN directory containing model/")
    parser.add_argument("--analysis", required=True, help="Analyze job directory containing data/kmeans_result.pkl")
    parser.add_argument(
        "--particles", required=True, help="Native RELION 5 particle STAR to subset (e.g. final run_data.star)"
    )
    parser.add_argument("--tomograms", required=True, help="Matching native RELION 5 tomograms.star")
    parser.add_argument("--datadir", help="RELION project root for relative native stack and tilt-STAR references")
    parser.add_argument(
        "--outdir", required=True, help="New or empty output directory; existing results are never overwritten"
    )
    parser.add_argument(
        "--flat-particles", help="Explicit relocated RECOVAR flattened STAR; must preserve original group identities"
    )
    parser.add_argument(
        "--clusters", type=_cluster_ids, help="Export these clusters separately, e.g. 0,2 (default: all)"
    )
    parser.add_argument(
        "--reset-poses",
        action="store_true",
        help="Zero particle Euler angles and origins/priors for a fresh ab initio run",
    )
    parser.add_argument(
        "--reset-halfsets", action="store_true", help="Remove rlnRandomSubset so RELION assigns fresh halves"
    )
    parser.add_argument(
        "--trajectories", help="Optional named-particle RELION trajectories STAR; legacy row-indexed files are rejected"
    )
    args = parser.parse_args()
    # Some historical params pickles contain JAX arrays; this metadata utility
    # needs no GPU even when unpickling those arrays on a login/CPU-only host.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from recovar.data_io.relion5_cluster_export import export_clusters

    manifest = export_clusters(**vars(args))
    print(f"Exported {manifest['exported_particles']} physical particles into {len(manifest['clusters'])} clusters.")
    print(f"Summary: {os.path.abspath(args.outdir)}/summary.tsv")


if __name__ == "__main__":
    main()
