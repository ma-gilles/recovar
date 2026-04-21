# Multi-Iter Parity Checkpoint

Date: 2026-04-21

This note records the current saved parity state before the longer 5k
convergence replay finishes.

## Saved output directories

Main 5k parity replay:
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/multi_iter_full_hp4_v4_childfix_gt`

1000-image adaptive stress replay:
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/multi_iter_1000_noise0p01_os1_v3_childfix_2iter`

Launched 5k convergence replay:
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/multi_iter_11_full_v1_childfix_gt`

Logs:
- 5k parity replay Slurm log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-5kgt-v4-7218937.out`
- 5k convergence replay Slurm log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-5k11-v1-7219478.out`

## Main 5k parity result

Source files:
- `pose_comparison_iter003.npz`
- `gt_comparison_final.npz`

RECOVAR vs RELION at iter 4:
- angular mean: `0.658646°`
- view-direction mean: `0.562604°`
- in-plane mean: `0.274411°`
- translation mean: `0.342820 px`

RECOVAR vs GT at iter 4:
- angular mean: `2.208353°`
- view-direction mean: `1.855104°`
- in-plane mean: `0.940941°`
- translation mean: `0.269876 px`

RELION vs GT at iter 4:
- angular mean: `2.224998°`
- view-direction mean: `1.868161°`
- in-plane mean: `0.948000°`
- translation mean: `0.220066 px`

Merged map vs GT:
- RECOVAR corr: `0.962886`
- RELION corr: `0.961718`
- RECOVAR FSC<0.5 shell: `37`
- RELION FSC<0.5 shell: `37`
- RECOVAR FSC<0.143 shell: `41`
- RELION FSC<0.143 shell: `41`

Interpretation:
- On the main 5k benchmark, parity is effectively closed at hp4.
- FSC and pose accuracy are essentially tied with RELION.

## 1000-image adaptive stress result

Source files:
- `pose_comparison_iter001.npz`
- `gt_comparison_iter001.npz`

RECOVAR vs GT at iter 2:
- angular mean: `3.386003°`
- view-direction mean: `2.750653°`
- in-plane mean: `1.583801°`
- translation mean: `0.454050 px`

RELION vs GT at iter 2:
- angular mean: `15.788637°`
- view-direction mean: `10.241913°`
- in-plane mean: `10.179044°`
- translation mean: `0.638228 px`

RECOVAR vs RELION at iter 2:
- angular mean: `14.984153°`
- view-direction mean: `9.496651°`
- in-plane mean: `10.092593°`
- translation mean: `0.506481 px`

Merged map vs GT:
- RECOVAR regularized corr: `0.757285`
- RELION corr: `0.771748`
- RECOVAR unregularized corr: `0.959141`
- RECOVAR regularized FSC<0.5 shell: `40`
- RELION FSC<0.5 shell: `31`
- RECOVAR regularized FSC<0.143 shell: `41`
- RELION FSC<0.143 shell: `42`
- RECOVAR unregularized FSC<0.5 shell: `40`
- RECOVAR unregularized FSC<0.143 shell: `41`

Interpretation:
- On this adaptive stress run, FSC is roughly tied while real-space
  correlation is more sensitive to regularization.
- The remaining gap on this dataset is not a pose-collapse and is not a
  gross tau2-scaling failure.

## 5k convergence replay status

Job:
- Slurm job `7219478`

Command shape:
- `scripts/run_multi_iter_parity.py --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star --iter 3 --max_iter 11 --max_healpix_order 4 --output_dir _agent_scratch/multi_iter_11_full_v1_childfix_gt`

Current status when this note was written:
- running on Slurm
- output directory already contains `it001_*` and `it002_*` intermediate
  artifacts, including local-search selector dumps

Next readout to collect after completion:
- final FSC shells vs GT
- final RECOVAR vs RELION pose metrics
- final RECOVAR vs GT pose metrics
- per-iteration trajectory to confirm stability through convergence
