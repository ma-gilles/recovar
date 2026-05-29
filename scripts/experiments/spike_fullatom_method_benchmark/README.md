# Spike Full-Atom Method Benchmark

Benchmark cryoDRGN, CryoSPARC 3DFlex, and DynaMight on the same 100k
full-atom spike simulated dataset.

## Current Noise10/B100 Size Sweep

Dataset root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516`

Particle counts:

`10k, 30k, 100k, 300k, 1M, 3M`

States scored:

`0, 25, 50`

Output root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528`

RECOVAR results already scored from existing `state000_unfil.mrc` outputs:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/recovar`

Submitted jobs:

- cryoDRGN train array: `8915364_[0-11%12]`
- cryoDRGN decode/score array: `8915365_[0-5%6]`
- CryoSPARC 3DFlex submission array: `8915584_[0-5%6]`

The image-count sweep was stopped on May 29, 2026 in favor of a fixed 100k
sanity pass. Use:

```bash
bash scripts/experiments/spike_fullatom_method_benchmark/submit_method_sanity_100k_noise10_b100.sh
```

Output root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529`

The fixed pass runs cryoDRGN `zdim=1,8`, CryoSPARC 3DFlex `flex_K=1,2`, and a
real RECOVAR `pipeline.py` with `zdim=1,4`, followed by compute_state variants
`zdim1_reg_covdist`, `zdim1_noreg_covdist`, `zdim4_reg_covdist`,
`zdim4_noreg_covdist`, and `zdim4_reg_dist` for GT states `0,25,50`.

## Current Noise30/B80 RECOVAR Check

This is RECOVAR-only, reusing existing full-atom spike runs.

Output root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise30_b80_20260528/recovar`

State roots:

- state 0: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0000_reuse_20260519`
- state 25: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0025_reuse_20260519`
- state 50: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518`

The Slurm scoring script is `submit_recovar_noise30_b80_score.sbatch`; the
first score pass was run locally because it is only FFT/post-processing.

Source run:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000`

Benchmark root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517`

Policy:

- CryoSPARC 3DFlex jobs are created and queued through CryoSPARC lanes.
- DynaMight uses the RELION 5.0.1 GPU module on the `cryoem` Slurm partition.
- cryoDRGN has no method-specific scheduler here, so it uses the `cryoem`
  Slurm partition through the installed `cryodrgn/3.4.3` module.

Latent sweeps:

- cryoDRGN: `zdim=1`, `zdim=8` (cryoDRGN example/default convention).
- CryoSPARC 3DFlex: `flex_K=1`, `flex_K=2` (3DFlex default).
- DynaMight: `n_latent_dimensions=1`, `n_latent_dimensions=6`
  (DynaMight default).

After training, evaluate each method by taking the mean learned embedding for
each simulator GT state label, decoding/reconstructing at those 100 mean points,
and scoring against the matching `04_ground_truth/gt_vol*.mrc` inside
`05_masks/focus_mask_moving.mrc` and `05_masks/volume_mask_union.mrc`.
