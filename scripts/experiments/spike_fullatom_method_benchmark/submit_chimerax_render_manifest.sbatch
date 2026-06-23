#!/bin/bash
#SBATCH --job-name=spike-chimerax-render
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch $0 MANIFEST.csv OUTPUT_DIR [extra render_chimerax_manifest.py args...]" >&2
  exit 2
fi

MANIFEST="$1"
OUTPUT_DIR="$2"
shift 2

AGENT_ID="chimerax_render_${SLURM_JOB_ID}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$OUTPUT_DIR"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

REPO="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar"
cd "$REPO"

module purge
module load "chimerax/${CHIMERAX_VERSION:-1.9}"

echo "manifest=${MANIFEST}"
echo "output_dir=${OUTPUT_DIR}"
echo "chimerax=$(command -v chimerax)"
chimerax --version || true

chimerax --nogui --offscreen --script \
  "scripts/experiments/spike_fullatom_method_benchmark/render_chimerax_manifest.py --manifest ${MANIFEST} --output-dir ${OUTPUT_DIR} $*"
