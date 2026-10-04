#!/usr/bin/env bash
# Native RELION-5 tomography subset export from a completed RECOVAR analysis.
# Run with the RECOVAR checkout that provides export_relion5_tomo_clusters.
# This does not submit jobs, copy image stacks, or convert flat STAR poses.
# Small nested tilt-STAR metadata files are copied with absolute micrograph
# references (preserving index@stack prefixes); geometry values are unchanged.
set -euo pipefail

# Edit these paths, or set the corresponding environment variables before running.
PIPELINE_DIR=${PIPELINE_DIR:-/path/to/Pipeline/job_0001}
ANALYZE_DIR=${ANALYZE_DIR:-/path/to/Analyze/k3/job_0001}
RELION_PARTICLES=${RELION_PARTICLES:-/path/to/RELION/Refine3D/job012/run_data.star}
RELION_TOMOGRAMS=${RELION_TOMOGRAMS:-/path/to/RELION/tomograms.star}
RELION_PROJECT_ROOT=${RELION_PROJECT_ROOT:-/path/to/RELION}
EXPORT_DIR=${EXPORT_DIR:-/path/to/new_relion_clusters}

# --particles must be a native one-row-per-physical-particle STAR for 2D tilt
# stacks, not a flat tilt-image STAR or extracted 3D pseudo-subtomograms.
# D64 RECOVAR labels may select native D128 particles via matching particle IDs.
# Use only trusted Pipeline/Analyze outputs: their pickle files are loaded.
args=(
    --pipeline "$PIPELINE_DIR"
    --analysis "$ANALYZE_DIR"
    --particles "$RELION_PARTICLES"
    --tomograms "$RELION_TOMOGRAMS"
    --datadir "$RELION_PROJECT_ROOT"
    --outdir "$EXPORT_DIR"
)

# Optional: relocated flat STAR used by the Pipeline; never a replacement dataset.
if [[ -n "${FLAT_PARTICLES:-}" ]]; then
    args+=(--flat-particles "$FLAT_PARTICLES")
fi
# Optional: CLUSTERS=0,2 exports those two clusters separately, not merged.
# Leave unset to export every cluster with its original zero-based number.
if [[ -n "${CLUSTERS:-}" ]]; then
    args+=(--clusters "$CLUSTERS")
fi
# Optional: motion/trajectory metadata is referenced only when explicitly given.
# It must use particle-named blocks, not legacy data_0/data_1 row-indexed blocks,
# with XYZ shifts for all native tomogram frames (not just visible images).
if [[ -n "${TRAJECTORIES:-}" ]]; then
    args+=(--trajectories "$TRAJECTORIES")
fi

# Default: preserve the supplied native poses, XYZ shifts, and halfsets.
# Explicitly append --reset-poses and/or --reset-halfsets below only if intended.
# --reset-poses zeroes poses/shifts/pose priors without changing tomo geometry.
# --reset-halfsets removes rlnRandomSubset so RELION can assign new halves.
recovar export_relion5_tomo_clusters "${args[@]}"

# Check EXPORT_DIR/manifest.json has status "completed", then inspect summary.tsv
# and cluster*/membership.tsv before RELION jobs.
# Each cluster has particles.star plus optimisation_set.star referencing the
# original image stacks and exported tomogram/tilt-STAR metadata. Referenced
# micrographs must remain available: RELION also reads the first image header.
