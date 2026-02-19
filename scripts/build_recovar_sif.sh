#!/bin/bash
# Build recovar.sif from the Docker image (run on a node that has Docker and Apptainer).
# Then copy the .sif to the project or scratch so batch jobs can use it.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if ! docker image inspect recovar:latest >/dev/null 2>&1; then
  echo "Building Docker image recovar:latest..."
  bash scripts/build_container.sh
fi
OUTPUT_SIF="${1:-$SCRIPT_DIR/../recovar.sif}"
echo "Building .sif: $OUTPUT_SIF"
apptainer build --force "$OUTPUT_SIF" docker-daemon://recovar:latest
echo "Done. Use: export CONTAINER_IMAGE=$OUTPUT_SIF"
echo "Or: export CONTAINER_SIF=$OUTPUT_SIF  (with CONTAINER_IMAGE=docker://recovar:latest)"
