#!/usr/bin/env bash
# Score ModelAngelo outputs for a dataset-size sweep. Use afterany dependency
# so failed low-count ModelAngelo tasks still produce a partial summary.

#SBATCH --job-name=score-ma-ds
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.err

set -euo pipefail

ROOT="${ROOT:?Set ROOT to the ModelAngelo experiment root}"
WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
REFERENCE_PDB="${REFERENCE_PDB:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs/morph_050.pdb}"
RESIDUE_RANGES="${RESIDUE_RANGES:-/scratch/gpfs/GILLES/mg6942/runs/spike_consistency_grid256_noise10_scaling_20260516/modelangelo_rmsd_20260516/focus_mask_atom_extract_state50/moving_focus_mask_morph050_support_mask_gt_0p01_residue_ranges.txt}"
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
