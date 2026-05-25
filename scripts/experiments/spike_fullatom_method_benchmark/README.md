# Spike Full-Atom Method Benchmark

Benchmark cryoDRGN, CryoSPARC 3DFlex, and DynaMight on the same 100k
full-atom spike simulated dataset.

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

