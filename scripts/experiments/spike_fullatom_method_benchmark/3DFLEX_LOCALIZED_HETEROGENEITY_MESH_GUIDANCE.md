# 3DFlex Localized Heterogeneity Mesh Guidance

Date: 2026-06-10

This note records the current best-practice plan for the spike full-atom
localized-motion 3DFlex experiments.

## Sources Checked

- CryoSPARC Guide, "Tutorial: 3D Flex Mesh Preparation":
  https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-3d-flex-mesh-preparation
- CryoSPARC Guide, "Tutorial: 3D Flexible Refinement":
  https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-3d-flexible-refinement
- CryoSPARC Tools, "3D Flex: Custom Mesh Rigidity":
  https://tools.cryosparc.com/examples/3dflex-custom-mesh-rigidity-weights.html
- CryoSPARC Discuss, "3dflex custom meshes":
  https://discuss.cryosparc.com/t/3dflex-custom-meshes/12064

## Interpretation For This Dataset

The dataset has one dominant localized conformational motion. The moving region
is known from simulation masks, but 3DFlex should not be trained on only that
mask as the primary route. CryoSPARC's mesh-prep guidance says the mesh prep
job uses a consensus map and solvent mask, and warns that masking out real
density can produce artifacts because 3DFlex reconstructs in real space. For
this spike case, the best primary setup is therefore:

1. Use the full particle solvent mask in 3D Flex Mesh Prep.
2. Use a custom segmentation MRC to assign every voxel inside the solvent mask
   to exactly one segment:
   - segment 0: moving core,
   - segment 1: interface/buffer around the moving core,
   - segment 2: rest of spike body,
   - -1: solvent.
3. Fuse the body to the interface and the interface to the moving core:
   `2>1,1>0`.
4. Avoid making the body segment fully rigid as the primary setting. The docs
   warn that rigid segments propagate constraints through fused vertices; in our
   case, a rigid body fused through a small interface can easily clamp the
   localized motion we need to recover.
5. Prefer coarser meshes first. Use base tetra cells 12 or 20, not 40 as the
   first-choice production route. The docs note that finer meshes reduce
   regularization and increase cost; our tetra40 jobs are exactly the ones
   hitting long runtime/timeouts.
6. Use K=1 as the primary latent dimension for this simulated 1D trajectory.
   K=2 is diagnostic only.
7. Evaluate generated maps at the mean latent for each GT state. Do not judge
   3DFlex by arbitrary PC/coordinate traversals for method comparison.

## Preferred Variant Ranking

Primary variants:

- `seg3_moving_interface10_body`, `tetra_num_cells=20`,
  `tetra_segments_fuse_list=2>1,1>0`, no `tetra_rigid_list`,
  `flex_K=1`, `flex_latent_prior_lam=2.0`.
- Same mesh with `flex_sv_lam=0.1`, `flex_latent_samp_std=0.1`,
  `flex_latent_prior_lam=2.0`.
- Same topology with `tetra_num_cells=12` if tetra20 is still too slow.

Secondary diagnostics:

- `seg2_movingplusinterface10_body`, no hard-rigid body, tetra12/tetra20.
- K=2 runs only to check whether the model is using an extra dimension instead
  of the expected 1D motion.

Lower priority:

- Any variant with `tetra_rigid_list=2` as a hard-rigid body. Keep them as
  controls, but do not treat them as the likely best result.
- Tetra40 variants, unless a coarse mesh proves under-resolved and completes
  within the available 48-hour CryoSPARC lane.
- Focus-mask-only 3DFlex training. It is useful as a diagnostic, but the docs'
  real-space masking artifact warning makes it a poor primary method.

## Current Run Locations

Benchmark root:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604`

Input masks/segmentations:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604/cryosparc_3dflex_segmented_moving_sweep/inputs`

Main manifests:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604/cryosparc_3dflex_segmented_moving_sweep/cryosparc_3dflex_noise1_b80_n00300000_segmented_moving_sweep_jobs_COMBINED.json`

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604/cryosparc_3dflex_segmented_moving_followup/cryosparc_3dflex_noise1_b80_n00300000_segmented_followup_jobs.json`

Monitor:

`/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/monitor_3dflex_jobs.py`

Latest monitor snapshot:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604/monitor_3dflex_jobs_20260610_latest.json`

## Operational Notes

The CryoSPARC `168hrs` lane currently submits with `--account=hughson`, and
Slurm rejects that account/partition/time combination. Do not rely on that lane
for this benchmark. Make the experiments fit the working `48hrs-a100` lane by
using tetra12/tetra20, K=1, and the narrow primary variants above.

The continuous monitor should keep requeueing worker database-startup failures,
but it intentionally does not hide real timeouts or invalid Slurm submissions.
