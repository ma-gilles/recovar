# Spike Experiment Index

This file is the first place to check for the major spike simulation,
RECOVAR, method benchmark, and scoring outputs generated during the kernel
regression and full-atom consistency experiments.

## Maintained Full-Atom Sweeps

| label | status | root | notes |
|---|---|---|---|
| fullatom state50 noise10 B100 | completed through 3M | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516` | State 50 baseline, grid 256, voxel 1.25 A, zdim 4, standard kernel, all 50 estimates saved. |
| fullatom state25 noise10 B100 reuse | completed/scored | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517` | Reuses state50 source datasets and pipelines; compute_state target is state 25. |
| fullatom state0 noise10 B100 reuse | completed/scored | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0000_reuse_20260517` | Reuses state50 source datasets and pipelines; compute_state target is state 0. |
| fullatom state50 noise10 B60 | superseded | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b60_parallel_20260516` | Earlier render-B sweep; useful only for historical comparisons. |
| fullatom state50 noise100 B80 | completed through 1M; 3M queued as 8507272 | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518` | New noisier setting, image counts 10k/30k/100k/300k/1M complete; 3M source/postprocess/ModelAngelo chain queued as 8507272/8507273/8507274/8507275. Shared B80 raw volumes at `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_b80_shared_20260518/01_raw_volumes`. First submit 8374692 failed before work because the wrapper did not create the shared raw dir; fixed and resubmitted as 8374757. Combined FSC0.5 plots are in `<root>/plots/`; mean-subtracted metrics are in `<root>/mean_subtracted_metrics_20260518_r12/`; ModelAngelo RMSD outputs are in `<root>/modelangelo_dataset_size_relion_adhoc_m80_20260518/`. |
| fullatom state50 noise30 B80 | completed through 3M; 10M failed | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518` | Repeat of the noise100/B80 setup with `noise_level=30`, image counts 10k/30k/100k/300k/1M and 3M complete. The 10M task `8428659_1` failed after 20:09:42 during compute_state with a missing staged MRC under `/scratch/gpfs/GILLES/mg6942/tmp/spike-fa-noise30.0-b80-8428659-1/recovar_cache/`; dependent 10M postprocess/ModelAngelo jobs are `DependencyNeverSatisfied`. |
| fullatom state0 noise30 B80 reuse | completed through 3M; 10M blocked | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0000_reuse_20260519` | Reuses the state50 noise30/B80 source datasets/pipelines and computes target state 0. The 10M job 8457044 is blocked because source 10M failed. Scoring reference is `morph_001.pdb` because the morph PDB files are one-indexed. |
| fullatom state25 noise30 B80 reuse | completed through 3M; 10M blocked | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0025_reuse_20260519` | Reuses the state50 noise30/B80 source datasets/pipelines and computes target state 25. The 10M job 8457048 is blocked because source 10M failed. Scoring reference is `morph_025.pdb`. |
| fullatom state0 noise100 B80 reuse | running/queued as 8506487; 3M queued as 8507276 | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0000_reuse_20260520` | Reuses the state50 noise100/B80 source datasets/pipelines and computes target state 0. Jobs 8506487/8506488/8506489/8506490 cover 10k/30k/100k/300k/1M; jobs 8507276/8507277/8507279/8507280 cover 3M after the 3M source finishes. Scoring reference is `morph_001.pdb`. |
| fullatom state25 noise100 B80 reuse | queued as 8506491; 3M queued as 8507282 | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0025_reuse_20260520` | Reuses the state50 noise100/B80 source datasets/pipelines and computes target state 25. Jobs 8506491/8506492/8506493/8506494 cover 10k/30k/100k/300k/1M; jobs 8507282/8507283/8507284/8507285 cover 3M after the 3M source finishes. Scoring reference is `morph_025.pdb`. |

## Method Benchmarks

| label | status | root | notes |
|---|---|---|---|
| 100k fullatom method benchmark | partial | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517` | cryoDRGN zdim1/zdim8 completed and scored for state25; 3DFlex jobs were launched in CryoSPARC project P587; DynaMight failed with zero flexible points. |

## Controlled Embedding Diagnostics

Short current reference: `scripts/experiments/spike_fullatom_state_sweeps/CONTROLLED_EMBEDDING_DIAGNOSTICS.md`.

These experiments reuse the full-atom `noise_level=30`, render `B=80`,
grid-256 source datasets at
`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518`.
They target state 50, use standard kernel regression, save all 50 candidates,
and score estimate-vs-GT FSC/error inside the broad state-50 mask
`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_motion_union_mask_20260524/masks/state50_broad_seed_mask.mrc`.

| label | status | run outputs | plot outputs | notes |
|---|---|---|---|---|
| state-index + constant GT-noise proxy | completed; compute array `8758578`, plot job `8758697` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_standard_index_gtnoise_lazy` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_stateindex_gtnoise_20260525` | Controlled embedding is scaled GT state index plus white Gaussian noise with constant precision. Main plots: `stateindex_gtnoise_standard_fsc_and_error.png`, `stateindex_gtnoise_standard_resolution_vs_n.png`. |
| source-oracle zdim1 noreg | completed `2026-05-26`; compute array `8773567`, plot job `8773569` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_standard_oracle_noreg_gtnoise_lazy` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_oracle_noreg_20260526` | Uses the source `06_pipeline/model/zdim_1/latent_coords_noreg.npy` and `latent_precision_noreg.npy` directly; compute_state is called with `--zdim1 --no-z-regularization`. FSC0.5 vs GT at 3M is 6.18 A in `oracle_noreg_standard_summary.csv`. |
| state-index + per-image estimated precision | completed `2026-05-26`; compute array `8773568`, plot job `8773570` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_standard_index_estprec_gtnoise_lazy` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_stateindex_estprec_20260526` | Controlled embedding is scaled GT state index plus per-image Gaussian noise with `sigma_i = sqrt(1 / latent_precision_noreg_i)` from the source oracle run; the copied pipeline keeps that per-image precision. FSC0.5 vs GT at 3M is 5.89 A in `stateindex_estprec_standard_summary.csv`. |
| normal RECOVAR zdim4 regularized reference | completed; rescored `2026-05-26` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_controlled_embedding_compare_20260526` | Existing standard zdim4/regularized run rescored under the same broad state-50 mask for comparison. FSC0.5 vs GT at 3M is 4.13 A in `zdim4_regularized_broadmask_summary.csv`. Combined controlled-embedding plot: `controlled_embedding_plus_zdim4reg_resolution_vs_n.png`. |
| source-oracle zdim/reg grid | running; missing cells compute array `8777295`, plot job `8777367` | Missing cells write to `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_standard_oracle_zdim1_reg_lazy` and `.../07_compute_state_standard_oracle_zdim4_noreg_lazy`; existing cells reuse `07_compute_state_standard_oracle_noreg_gtnoise_lazy` and `07_compute_state` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_sourceoracle_zdim_reg_grid_20260526` | 2x2 source-oracle comparison requested to close the zdim1/zdim4 gap: zdim 1/4 crossed with regularized/noreg coordinates. Plots include estimate curves and nearest-1k latent-space GT-mixture curves. Submission scripts are `_agent_scratch/submit_spike_fullatom_noise30_sourceoracle_zdim_reg_grid_20260526.sbatch` and `_agent_scratch/postprocess_spike_fullatom_noise30_sourceoracle_zdim_reg_grid_20260526.sbatch`. |
| source-oracle zdim4 reg identity-distance | queued as `8779182`, combined plot job `8779206` waits for `8777295` and `8779182` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_standard_oracle_zdim4_reg_dist_lazy` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_sourceoracle_zdim_reg_grid_20260526` | Uses regularized zdim4 source-oracle coordinates and target point, but standard kernel distances use `--embedding-option dist`, i.e. identity latent precision instead of the stored regularized precision. Submission script: `_agent_scratch/submit_spike_fullatom_noise30_sourceoracle_zdim4_reg_dist_20260526.sbatch`. |
| fixed source-oracle reg/noreg grid | submitted `2026-05-26`; oracle pipeline array `8780075`, compute_state array `8780076`, plot job `8780077` | Rebuilt oracle pipelines write to `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/06_pipeline_oracle_regfix_20260526`; compute_state writes `07_compute_state_oracle_regfix_zdim1_reg_lazy`, `...zdim1_noreg_lazy`, `...zdim4_reg_lazy`, `...zdim4_noreg_lazy`, and `...zdim4_reg_dist_lazy` | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_oracle_regfix_zdim_grid_20260526` | Supersedes the stale source-oracle reg/noreg rows above after fixing `recovar/simulation/oracle_pipeline.py` so regularized embeddings use oracle eigenvalues and noreg embeddings use infinite eigenvalues. Plots include estimate curves plus nearest-1k latent-space GT-mixture curves across image counts. Submission scripts are `_agent_scratch/submit_spike_fullatom_noise30_regfix_oracle_pipeline_20260526.sbatch`, `_agent_scratch/submit_spike_fullatom_noise30_regfix_compute_state_20260526.sbatch`, and `_agent_scratch/postprocess_spike_fullatom_noise30_regfix_fsc_error_20260526.sbatch`. |
| pipeline zdim4 regularized rerun | submitted `2026-05-26`; compute array `8794988`, plot job `8794989` | `<source-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/07_compute_state_pipeline_zdim4_reg_lazy` | `<source-root>/plots/` and `<source-root>/mean_subtracted_pipeline_zdim4_reg_state50_broadmask_20260526` | Uses the ordinary RECOVAR-generated `06_pipeline/model/zdim_4/latent_coords.npy` and `latent_precision.npy`, target state 50 mean in that embedding, `--embedding-option cov_dist`, `--save-all-estimates`, and image counts 10k/30k/100k/300k/1M/3M/10M. Submission script: `scripts/experiments/spike_fullatom_state_sweeps/submit_pipeline_zdim4_reg_compute_state.sbatch`. |

## ModelAngelo / Atomic Scoring

| label | status | root | notes |
|---|---|---|---|
| state50 ModelAngelo fixed-B scoring | completed | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/modelangelo_dataset_size_relion_adhoc_m100_20260516` | Moving-region same-element RMSD scoring against the matching morph PDB. |
| state25 ModelAngelo fixed-B scoring | completed | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517/modelangelo_dataset_size_relion_adhoc_m100_state0025_20260517` | Moving-region same-element RMSD scoring for state25. |
| state0 ModelAngelo fixed-B scoring | completed | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0000_reuse_20260517/modelangelo_dataset_size_relion_adhoc_m100_state0000_20260517` | Moving-region same-element RMSD scoring for state0. |

## Earlier Kernel-Regression Development Runs

| label | status | root | notes |
|---|---|---|---|
| deconvolved vs standard 100k GT-noisy embedding | historical | `/scratch/gpfs/GILLES/mg6942/runs/compute_state_100k_gt_noisy_embedding_20260512` | Early lambda/h comparison and kernel diagnostics. |
| grid256 noise10 scaling scratch analysis | historical | `/scratch/gpfs/GILLES/mg6942/runs/spike_consistency_grid256_noise10_scaling_20260516` | Older plots, masks, ModelAngelo smoke outputs, and intermediate scoring artifacts. Treat as pre-full-atom-index scratch unless a path is explicitly referenced. |

## Repo Scripts

| script | purpose |
|---|---|
| `scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch` | Submit the new noise100/B80 RECOVAR dataset-size sweep through 1M images. |
| `scripts/experiments/spike_fullatom_state_sweeps/postprocess_fullatom_dataset_size.sbatch` | Generate FSC0.5-vs-GT, mean-subtracted FSC/locres, and low-frequency-cut residual plots after a dataset-size sweep. |
| `scripts/experiments/spike_fullatom_state_sweeps/score_dataset_size_fsc_vs_gt.py` | Compute unfiltered estimate-vs-GT FSC0.5 summary plots for a dataset-size sweep. |
| `scripts/experiments/spike_fullatom_state_sweeps/score_mean_subtracted_dataset_size.py` | Compute mean-subtracted residual FSC0.5 and sampled local resolution inside the focus mask. |
| `scripts/experiments/spike_fullatom_state_sweeps/plot_mean_subtracted_minfreq_sweep.py` | Replot mean-subtracted FSC curves after ignoring low-frequency shells. |
| `scripts/experiments/spike_fullatom_state_sweeps/submit_state25_compute_state_reuse.sbatch` | Reuse an existing source sweep and rerun compute_state for a different target state. |
| `scripts/experiments/spike_fullatom_state_sweeps/EXPERIMENTS.md` | More detailed notes for the state-sweep jobs and current full-atom roots. |
| `scripts/experiments/modelangelo_spike_scoring/score_modelangelo_dataset_size.sbatch` | Score ModelAngelo outputs with an `afterany` dependency so partial low-count failures still produce a summary. |
| `scripts/experiments/modelangelo_spike_scoring/plot_modelangelo_rmsd_dataset_size.py` | Build the best-available RMSD table and RMSD/p90/p99-vs-image-count plot. |
