# Spike ModelAngelo Scoring

This folder contains the scripts used for the full-atom spike dataset-size
experiment. The workflow is:

1. Run `relion_postprocess` on RECOVAR halfmaps with fixed true sharpening,
   using `--adhoc_bfac -100` for simulations rendered with B factor 100.
2. Run `model_angelo build` on the postprocessed map with the spike FASTA.
3. Score ModelAngelo CIFs against the state-50 morph model, restricted to
   the moving-region residue set.

The main measured quantity is a same-element nearest-atom RMSD after C-alpha
ICP alignment. It scores the full atoms in the chain-B residues that intersect
the state-50 moving focus-mask support.

## Inputs Used In The Current Run

Run root:

```bash
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516
```

Reference model and sequence:

```bash
/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs/morph_050.pdb
/scratch/gpfs/GILLES/mg6942/runs/spike_full_atom_morph_validation_20260516/spike_P0DTC2_2P_ectodomain_27_1147_trimer_ABC.fasta
```

Moving-region residue list:

```bash
/scratch/gpfs/GILLES/mg6942/runs/spike_consistency_grid256_noise10_scaling_20260516/modelangelo_rmsd_20260516/focus_mask_atom_extract_state50/moving_focus_mask_morph050_support_mask_gt_0p01_residue_ranges.txt
```

## Build A Run List

Create a tab-separated file with label and run directory. Each run directory
must contain:

```bash
07_compute_state/state000_half1_unfil.mrc
07_compute_state/state000_half2_unfil.mrc
05_masks/volume_mask_union.mrc
```

Example:

```bash
ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516
cat > "$ROOT/modelangelo_dataset_size_relion_adhoc_m100_20260516/runs_completed.tsv" <<EOF
n00010000	$ROOT/n00010000/runs/n00010000_seed0000
n00030000	$ROOT/n00030000/runs/n00030000_seed0000
n00100000	$ROOT/n00100000/runs/n00100000_seed0000
n00300000	$ROOT/n00300000/runs/n00300000_seed0000
EOF
```

## Submit Fixed-B ModelAngelo Jobs

Submit on A100 nodes because the RELION 5 ModelAngelo PyTorch environment used
here does not support H100 (`sm_90`) on this cluster.

```bash
ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/modelangelo_dataset_size_relion_adhoc_m100_20260516
sbatch \
  --array=0-3%4 \
  --export=ALL,ROOT="$ROOT",RUN_LIST="$ROOT/runs_completed.tsv" \
  scripts/experiments/modelangelo_spike_scoring/submit_modelangelo_fixed_b_dataset_size.sbatch
```

The default fixed B factor is `-100`, i.e. RELION sharpening convention for a
simulation rendered with B factor 100. Override with `ADHOC_BFAC` if needed.

## Score Models

```bash
ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/modelangelo_dataset_size_relion_adhoc_m100_20260516
python scripts/experiments/modelangelo_spike_scoring/score_modelangelo_moving_region.py \
  --reference-pdb /projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs/morph_050.pdb \
  --residue-ranges /scratch/gpfs/GILLES/mg6942/runs/spike_consistency_grid256_noise10_scaling_20260516/modelangelo_rmsd_20260516/focus_mask_atom_extract_state50/moving_focus_mask_morph050_support_mask_gt_0p01_residue_ranges.txt \
  --model-root "$ROOT/modelangelo" \
  --csv "$ROOT/moving_region_all_atom_same_element_rmsd.csv"
```

The scorer writes one row for each pruned and raw CIF. Empty or invalid CIFs
are retained as rows with `parse_error` instead of aborting the whole table.
