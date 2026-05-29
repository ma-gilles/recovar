# Controlled Embedding Diagnostics

Quick reference for the state-50, noise30/B80, grid256 controlled-embedding runs.

## Common Setup

- Source datasets: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518`
- Image counts: `10k, 30k, 100k, 300k, 1M, 3M`
- Target: state 50
- Compute mode: standard kernel regression, `--embedding-option cov_dist`, `--save-all-estimates`
- Error/FSC mask: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_motion_union_mask_20260524/masks/state50_broad_seed_mask.mrc`
- Plot root: `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523`

## Current Variants

| short name | latent coordinates | precision used by compute_state | compute_state dir under each `nXXXXXXXX/runs/nXXXXXXXX_seed0000` | status |
|---|---|---|---|---|
| `oracle_zdim4_reg` | source-oracle `06_pipeline_oracle_regfix_20260526/model/zdim_4/latent_coords.npy` | source-oracle regularized precision | `07_compute_state_oracle_regfix_zdim4_reg_lazy` | complete through 3M |
| `gtpc_iid_trueunits` | true GT-PC coordinate `U^T(V_state-mean)` plus iid Gaussian noise | scalar precision matched to source-oracle residual after mapping to true GT-PC units | `07_compute_state_standard_gtpc_iidnoise_trueunits_zdim4_reg_lazy` | job `8832426`, complete through 300k, 1M/3M still running |
| `gtpc_covnoise_trueunits` | true GT-PC coordinate plus per-image Gaussian noise from transformed source-oracle covariance | matching transformed per-image precision | `07_compute_state_standard_gtpc_covnoise_trueunits_zdim4_reg_lazy` | job `8837864` running |
| `gtpc_covshuffle_trueunits` | same as `gtpc_covnoise_trueunits`, but source-oracle covariance matrices are shuffled across images before sampling | matching shuffled/transformed per-image precision | `07_compute_state_standard_gtpc_covshuffle_trueunits_zdim4_reg_lazy` | job `8838227` submitted |
| combined postprocess | scores the three GT-PC synthetic-noise variants and overlays them with source-oracle | n/a | plot-only | job `8838406`, dependency on `8832426`, `8837864`, `8838227` |

## Plot Outputs

| plot | path |
|---|---|
| source-oracle zdim4/reg curves | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_oracle_regfix_zdim4_reg_20260527/oracle_regfix_zdim4_reg_standard_fsc_and_error.png` |
| iid true-unit curves, partial 10k-300k | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_gtpc_iid_trueunits_zdim4_20260527_partial_10k_300k/gtpc_iid_trueunits_zdim4_standard_partial_10k_300k_fsc_and_error.png` |
| source-oracle vs iid true-unit overlay | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_oracle_vs_gtpc_iid_trueunits_zdim4_20260527/oracle_regfix_zdim4_vs_gtpc_iid_trueunits_partial_10k_300k.png` |
| all controlled variants overlay, after jobs finish | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/state50_broad_mask_controlled_gtpc_noise_variants_20260527/controlled_gtpc_noise_variants_oracle_iid_cov_covshuffle.png` |
| oracle covariance calibration check | `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/oracle_embedding_covariance_sanity_20260527/oracle_embedding_covariance_vs_empirical_error.png` |

The covariance calibration check says `trace(inv(P_source))` agrees with
`||source_z - affine(true_gtpc_z)||^2` in expectation. The same remains true
after transforming the covariance into true GT-PC coordinates.

## Scripts

| script | purpose |
|---|---|
| `_agent_scratch/prepare_gtpc_iid_noise_embedding.py` | Copies the source oracle pipeline and writes controlled zdim4 coordinates/precision. Supports iid, per-image covariance, and shuffled covariance modes. |
| `_agent_scratch/submit_spike_fullatom_noise30_gtpc_iid_trueunits_zdim4_standard_20260527.sbatch` | Runs `gtpc_iid_trueunits`. |
| `_agent_scratch/submit_spike_fullatom_noise30_gtpc_covnoise_zdim4_standard_20260527.sbatch` | Runs `gtpc_covnoise_trueunits`. |
| `_agent_scratch/submit_spike_fullatom_noise30_gtpc_covshuffle_zdim4_standard_20260527.sbatch` | Runs `gtpc_covshuffle_trueunits`. |
| `_agent_scratch/plot_stateindex_gtnoise_standard_fsc_error.py` | Scores one variant and writes FSC/error curves plus summary CSV. |
| `_agent_scratch/plot_oracle_vs_gtpc_noise_comparison.py` | Makes the current two-way source-oracle vs iid overlay. |
| `_agent_scratch/plot_controlled_embedding_variants_overlay.py` | Overlays source-oracle, iid, covariance-noise, and shuffled-covariance-noise curves after each variant has summary CSVs. |
| `_agent_scratch/postprocess_spike_fullatom_noise30_gtpc_noise_variants_20260527.sbatch` | Dependent postprocess job that scores the GT-PC variants and builds the combined overlay. |
| `_agent_scratch/compare_oracle_embedding_covariance.py` | Checks predicted covariance scale against empirical source-oracle embedding error. |

## Current Interpretation

So far, `oracle_zdim4_reg` is much better than `gtpc_iid_trueunits` at the
same image count. The next diagnostic is whether `gtpc_covnoise_trueunits`
recovers some of that gap, and whether shuffling covariances destroys it.
