#!/usr/bin/env bash
# Score ModelAngelo outputs for a dataset-size sweep. Use afterany dependency
# so failed low-count ModelAngelo tasks still produce a partial summary.

#SBATCH --job-name=score-ma-ds
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
ROOT="${ROOT:?Set ROOT to the ModelAngelo experiment root}"
WORKDIR="${WORKDIR:-$SUBMIT_DIR}"
REFERENCE_PDB="${REFERENCE_PDB:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs/morph_050.pdb}"
RESIDUE_RANGES="${RESIDUE_RANGES:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517/state25_masked_maps_scoring_models_metrics_20260517/models_scoring_region/moving_focus_mask_morph050_support_mask_gt_0p01_residue_ranges.txt}"
PYTHON="${PYTHON:-/projects/MOLBIO/local/pythonenv/relion-5.0.1-rhel9/bin/python}"

export PYTHONNOUSERSITE=1
mkdir -p "$ROOT/plots"

cd "$WORKDIR"
echo "start=$(date -Is)"
echo "root=$ROOT"
echo "python=$PYTHON"

"$PYTHON" scripts/experiments/modelangelo_spike_scoring/score_modelangelo_moving_region.py \
  --reference-pdb "$REFERENCE_PDB" \
  --residue-ranges "$RESIDUE_RANGES" \
  --model-root "$ROOT/modelangelo" \
  --csv "$ROOT/moving_region_all_atom_same_element_rmsd.csv"

"$PYTHON" scripts/experiments/modelangelo_spike_scoring/plot_modelangelo_rmsd_dataset_size.py \
  --csv "$ROOT/moving_region_all_atom_same_element_rmsd.csv" \
  --out-dir "$ROOT/plots"

echo "done=$(date -Is)"
