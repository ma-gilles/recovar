# Spike Full-Atom State Sweeps

For a fresh-clone, student-facing reproduction guide, start with
`STUDENT_RUNBOOK.md`. It shows how to set a per-user scratch root, build the
pixi environment, run a smoke job, run the dataset-size sweep, postprocess
metrics, and optionally launch ModelAngelo.

Scratch root for the current B100 full-atom spike dataset-size experiments:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp
```

## State 50 Baseline

Purpose: original dataset-size sweep targeting state 50.

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516
```

Completed source datasets with `07_compute_state` and all 50 kernel estimates:

```text
n00010000
n00030000
n00100000
n00300000
n01000000
n03000000
```

Important outputs:

```text
plots/dataset_size_global_fsc_candidate50_reslog_semilogy.png
plots/dataset_size_modelangelo_rmsd_inverse_A_images_semilogy_y_by_image_order.png
plots/focus_moving_same_element_heavy_rmsd_atoms_state0050.tar.gz
modelangelo_dataset_size_relion_adhoc_m100_20260516/moving_region_all_atom_same_element_rmsd.csv
```

## State 25 Reuse Sweep

Purpose: reuse the state-50 datasets/pipelines but infer target state 25.

State-25 output root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517
```

Source root reused:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516
```

Submitted jobs:

```text
8359908  spike-fa-s25-cs  compute_state array for 10k, 30k, 100k, 300k, 1M, 3M
8359960  ma-fixed-b-ds    ModelAngelo fixed-B array, afterok:8359908
8360062  score-s25-ma     moving-region RMSD scoring, afterok:8359960
```

Per-dataset output shape:

```text
<state25-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/
  03_dataset -> symlink to source dataset
  04_ground_truth -> symlink to source GT volumes
  05_masks -> symlink to source masks
  06_pipeline -> symlink to source oracle pipeline
  target_latent_point_state0025.txt
  07_compute_state/
```

ModelAngelo/scoring output:

```text
<state25-root>/modelangelo_dataset_size_relion_adhoc_m100_state0025_20260517/
  runs_all.tsv
  postprocess/
  modelangelo/
  moving_region_all_atom_same_element_rmsd_state0025.csv
```

Submit command:

```bash
sbatch scripts/experiments/spike_fullatom_state_sweeps/submit_state25_compute_state_reuse.sbatch
```

## State 0 Reuse Sweep

Purpose: reuse the same datasets/pipelines but infer target state 0. The
compute-state target volume is `gt_vol0000.mrc`. The PDB morph files are
one-indexed, so the dependent ModelAngelo scoring job uses `morph_001.pdb` as
the state-0 structural reference.

State-0 output root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0000_reuse_20260517
```

Submitted jobs:

```text
8360617  spike-fa-s00-cs  compute_state array for 10k, 30k, 100k, 300k, 1M, 3M
8360620  ma-s00-fixed-b   ModelAngelo fixed-B array, afterok:8360617
8360623  score-s00-ma     moving-region RMSD scoring, afterok:8360620
```

ModelAngelo/scoring output:

```text
<state0-root>/modelangelo_dataset_size_relion_adhoc_m100_state0000_20260517/
  runs_all.tsv
  postprocess/
  modelangelo/
  moving_region_all_atom_same_element_rmsd_state0000.csv
```

Submit command:

```bash
STATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0000_reuse_20260517
sbatch \
  --job-name=spike-fa-s00-cs \
  --export=ALL,TARGET_STATE=0,STATE_ROOT="$STATE_ROOT" \
  scripts/experiments/spike_fullatom_state_sweeps/submit_state25_compute_state_reuse.sbatch
```

## Noise 100 / B80 Dataset-Size Sweep

Purpose: repeat the full-atom state-50 RECOVAR dataset-size sweep with much
noisier images and a slightly sharper rendered ground-truth stack. This uses
the same `spike_consistency` driver/worktree as the B100/noise10 full sweep so
the directory layout and compute-state settings are comparable.

Settings:

```text
target state: 50
grid size: 256
voxel size: 1.25 A
render B-factor: 80 A^2
compute_state B-factor: 0
noise_level: 100
zdim: 4
kernel regression: standard
n-bins: 50
saved estimates: all 50 candidates
image counts: 10k, 30k, 100k, 300k, 1M
```

Output root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518
```

Shared rendered B80 raw volumes:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_shared_20260518/01_raw_volumes
```

Per-dataset output shape:

```text
<noise100-b80-root>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/
  01_raw_volumes -> symlink/reuse of shared B80 volumes
  03_dataset
  04_ground_truth
  05_masks
  06_pipeline
  07_compute_state
```

Submit command:

```bash
sbatch scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
```

Submitted jobs:

```text
8374692  spike-fa-n100-b80  failed immediately; submit wrapper forgot to create SHARED_RAW before find
8374757  spike-fa-n100-b80  completed RECOVAR full sweep for 10k, 30k, 100k, 300k, 1M
```

Initial FSC0.5 results, using the unfiltered compute_state map against GT
inside the supplied mask:

```text
n_images    FSC0.5 resolution
10,000      10.22 A
30,000       8.36 A
100,000      6.86 A
300,000      6.41 A
1,000,000    5.25 A
```

Combined plots and CSV:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/plots/noise100_b80_combined_fsc_vs_gt_fsc05.png
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/plots/noise100_b80_fsc05_resolution_vs_n.png
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/plots/noise100_b80_fsc05_summary.csv
```

Mean-subtracted residual metrics:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_metrics.csv
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc_curves_fsc05.png
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_resolution_vs_n_fsc05.png
```

These compare `(estimate - mean(GT states))` against
`(gt_vol0050 - mean(GT states))` inside `focus_mask_moving.mrc`, using FSC 0.5.
The local metric uses sampled 12 A radius balls at 12 A spacing in the focus
mask.

```text
n_images    residual FSC0.5    residual locres median    residual locres p90
10,000      320.00 A           26.25 A                  26.25 A
30,000      173.80 A           26.25 A                  26.25 A
100,000     320.00 A           26.25 A                  26.25 A
300,000     230.95 A           26.25 A                  26.25 A
1,000,000   320.00 A           15.93 A                  26.25 A
```

Low-frequency-ignored residual FSC plots:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc_curves_fsc05_ignore_below_0p05.png
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc05_resolution_vs_n_minfreq_sweep.png
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc05_minfreq_sweep.csv
```

Using `min_freq = 0.05 1/A` (ignore signal coarser than 20 A), only the 1M
run has a real FSC0.5 crossing:

```text
n_images    residual FSC0.5 after cutoff
10,000      20.00 A, already below 0.5 at cutoff
30,000      20.00 A, already below 0.5 at cutoff
100,000     20.00 A, already below 0.5 at cutoff
300,000     20.00 A, already below 0.5 at cutoff
1,000,000    9.52 A, real crossing
```

ModelAngelo fixed-B scoring:

```text
8398721  ma-n100-b80-m80  ModelAngelo/RELION fixed B -80 array
8398729  score-ma-n100-b80 dependency scorer, cancelled because 10k/30k/100k ModelAngelo tasks failed
```

ModelAngelo produced no C-alpha atoms for 10k, 30k, or 100k. The 300k and 1M
runs produced scoreable raw CIFs:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/modelangelo_dataset_size_relion_adhoc_m80_20260518/moving_region_all_atom_same_element_rmsd.csv
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/modelangelo_dataset_size_relion_adhoc_m80_20260518/plots/rmsd_summary_best_available.csv
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/modelangelo_dataset_size_relion_adhoc_m80_20260518/plots/modelangelo_moving_region_rmsd_vs_n.png
```

```text
n_images    best available ModelAngelo moving-region RMSD
10,000      no C-alpha atoms predicted
30,000      no C-alpha atoms predicted
100,000     no C-alpha atoms predicted
300,000     9.81 A raw CIF, p99 17.97 A
1,000,000   5.32 A raw CIF, p99 12.28 A
```

## Noise 30 / B80 Dataset-Size Sweep

Purpose: repeat the same state-50 full-atom RECOVAR dataset-size experiment as
the noise100/B80 sweep, but with `noise_level = 30`.

Settings:

```text
target state: 50
grid size: 256
voxel size: 1.25 A
render B-factor: 80 A^2
compute_state B-factor: 0
noise_level: 30
zdim: 4
kernel regression: standard
n-bins: 50
saved estimates: all 50 candidates
image counts: 10k, 30k, 100k, 300k, 1M
```

Output root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518
```

Shared rendered B80 raw volumes:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_b80_shared_20260518/01_raw_volumes
```

Submit command:

```bash
ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518
SHARED=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_b80_shared_20260518
sbatch \
  --job-name=spike-fa-n30-b80 \
  --export=ALL,NOISE_LEVEL=30.0,RENDER_BFACTOR=80,BASE_ROOT="$ROOT",SHARED_ROOT="$SHARED" \
  scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
```

Submitted jobs:

```text
8400584  spike-fa-n30-b80    completed RECOVAR dataset-size array
8400724  post-n30-b80        completed postprocess for FSC/locres plots
8400725  ma-n30-b80-m80      completed with 10k/30k failures, 100k/300k/1M successes
8400726  score-ma-n30-b80    completed ModelAngelo scorer/plotter
```

Postprocess outputs:

```text
<root>/plots/noise30_b80_combined_fsc_vs_gt_fsc05.png
<root>/plots/noise30_b80_fsc05_resolution_vs_n.png
<root>/plots/noise30_b80_fsc05_summary.csv
<root>/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc_curves_fsc05.png
<root>/mean_subtracted_metrics_20260518_r12/mean_subtracted_resolution_vs_n_fsc05.png
<root>/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc_curves_fsc05_ignore_below_0p05.png
<root>/mean_subtracted_metrics_20260518_r12/mean_subtracted_fsc05_resolution_vs_n_minfreq_sweep.png
<root>/modelangelo_dataset_size_relion_adhoc_m80_20260518/plots/modelangelo_moving_region_rmsd_vs_n.png
```

Initial masked FSC0.5 results, using the unfiltered compute_state map against
GT inside the supplied mask:

```text
n_images    FSC0.5 resolution
10,000       9.13 A
30,000       7.17 A
100,000      5.97 A
300,000      5.20 A
1,000,000    4.39 A
```

Mean-subtracted residual metrics inside `focus_mask_moving.mrc`:

```text
n_images    residual FSC0.5    residual locres median    residual locres p90
10,000      320.00 A           26.25 A                  26.25 A
30,000      320.00 A           26.25 A                  26.25 A
100,000     320.00 A           25.85 A                  26.25 A
300,000     320.00 A           26.25 A                  26.25 A
1,000,000   320.00 A           10.46 A                  26.25 A
```

Using `min_freq = 0.05 1/A` (ignore signal coarser than 20 A), the residual
FSC0.5 crossings are:

```text
n_images    residual FSC0.5 after cutoff
10,000      20.00 A, already below 0.5 at cutoff
30,000      20.00 A, already below 0.5 at cutoff
100,000      8.44 A
300,000     20.00 A, already below 0.5 at cutoff
1,000,000    6.92 A
```

ModelAngelo fixed-B scoring (`ADHOC_BFAC=-80`) failed at 10k and 30k. Best
available raw-CIF moving-region same-element RMSD:

```text
n_images    RMSD      p90       p99
100,000     1.88 A    2.94 A    5.35 A
300,000     1.45 A    2.32 A    4.96 A
1,000,000   1.22 A    1.97 A    4.07 A
```

3M/10M extension submitted May 18:

```bash
ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518
SHARED=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_b80_shared_20260518
sbatch \
  --array=0-1%2 \
  --job-name=spike-fa-n30-b80-big \
  --export=ALL,NOISE_LEVEL=30.0,RENDER_BFACTOR=80,BASE_ROOT="$ROOT",SHARED_ROOT="$SHARED",N_IMAGES_VALUES_STR="3000000 10000000" \
  scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
```

Submitted jobs and final status:

```text
8428659_0  spike-fa-n30-b80-big  3M source RECOVAR task, completed in 11:02:54
8428659_1  spike-fa-n30-b80-big  10M source RECOVAR task, failed after 20:09:42
8428676    post-n30-b80-big      dependency-never-satisfied because 10M source failed
8428677    ma-n30-b80-big        dependency-never-satisfied because 10M source failed
8428678    score-ma-n30-big      blocked because ModelAngelo dependency never ran
```

The 10M failure happened during compute_state with a missing staged lazy-cache
MRC under `/scratch/gpfs/GILLES/mg6942/tmp/spike-fa-noise30.0-b80-8428659-1/recovar_cache/`.

Early 3M ModelAngelo pass:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/modelangelo_early_3m_relion_adhoc_m80_20260519
```

This was run locally on `della-mol` A100 GPU 1 because the A100 Slurm jobs
were pending while the 10M RECOVAR job was still running. RELION postprocess
reported 3.60 A for the 3M half-map FSC. Moving-region same-element RMSD
against `morph_050.pdb`:

```text
model_kind    RMSD      median    p90      p99      atoms
pruned        2.68 A    0.64 A    4.66 A   8.71 A   15,925
raw           1.04 A    0.50 A    1.46 A   3.87 A   22,058
```

## Noise 30 / B80 State 0 And State 25 Reuse Sweeps

Purpose: reuse the noise30/B80 state-50 source datasets and oracle pipelines,
but infer states 0 and 25. This avoids regenerating images.

Source root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518
```

State roots:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0000_reuse_20260519
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0025_reuse_20260519
```

Submitted jobs:

```text
8456994  spike-fa-n30-s00   compute_state array for 10k, 30k, 100k, 300k, 1M, 3M
8457026  post-n30-s00       dependent FSC/locres postprocess for 10k through 3M
8457027  ma-n30-s00         dependent ModelAngelo array for 10k through 3M
8457028  score-ma-n30-s00   dependent ModelAngelo scorer
8457044  spike-n30-s00-10m  10M compute_state, afterok on source 10M job 8428659
8457045  post-n30-s00-10m   dependent FSC/locres postprocess including 10M
8457046  ma-n30-s00-10m     dependent ModelAngelo for 10M
8457047  score-s00-10m      dependent ModelAngelo scorer including 10M

8456995  spike-fa-n30-s25   compute_state array for 10k, 30k, 100k, 300k, 1M, 3M
8457029  post-n30-s25       dependent FSC/locres postprocess for 10k through 3M
8457030  ma-n30-s25         dependent ModelAngelo array for 10k through 3M
8457031  score-ma-n30-s25   dependent ModelAngelo scorer
8457048  spike-n30-s25-10m  10M compute_state, afterok on source 10M job 8428659
8457049  post-n30-s25-10m   dependent FSC/locres postprocess including 10M
8457050  ma-n30-s25-10m     dependent ModelAngelo for 10M
8457051  score-s25-10m      dependent ModelAngelo scorer including 10M
```

The 10M state-0/state-25 jobs are blocked because the noise30/B80 source 10M
task failed.

ModelAngelo scoring uses the same moving-region residue set as the state-50
scoring. References are `morph_001.pdb` for state 0 because morph PDBs are
one-indexed, and `morph_025.pdb` for state 25.

## Noise 100 / B80 State 0 And State 25 Reuse Sweeps

Purpose: repeat the noise30 state reuse experiment on the completed noise100
B80 source datasets. The source root has 10k through 1M only.

Source root:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518
```

State roots:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0000_reuse_20260520
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0025_reuse_20260520
```

Submitted jobs:

```text
8506487  spike-fa-n100-s00  compute_state array for 10k, 30k, 100k, 300k, 1M
8506488  post-n100-s00      dependent FSC/locres postprocess
8506489  ma-n100-s00        dependent ModelAngelo array
8506490  score-ma-n100-s00  dependent ModelAngelo scorer

8506491  spike-fa-n100-s25  compute_state array for 10k, 30k, 100k, 300k, 1M
8506492  post-n100-s25      dependent FSC/locres postprocess
8506493  ma-n100-s25        dependent ModelAngelo array
8506494  score-ma-n100-s25  dependent ModelAngelo scorer
```

3M extension submitted May 20:

```text
8507272  spike-fa-n100-b80-3m    state50 source RECOVAR 3M
8507273  post-n100-b80-3m        state50 postprocess after 3M source
8507274  ma-n100-s50-3m          state50 ModelAngelo after 3M source
8507275  score-ma-n100-s50-3m    state50 ModelAngelo scorer

8507276  spike-n100-s00-3m       state0 compute_state reuse after 3M source
8507277  post-n100-s00-3m        state0 postprocess
8507279  ma-n100-s00-3m          state0 ModelAngelo
8507280  score-ma-n100-s00-3m    state0 ModelAngelo scorer

8507282  spike-n100-s25-3m       state25 compute_state reuse after 3M source
8507283  post-n100-s25-3m        state25 postprocess
8507284  ma-n100-s25-3m          state25 ModelAngelo
8507285  score-ma-n100-s25-3m    state25 ModelAngelo scorer
```

The 3M ModelAngelo run lists are:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_parallel_20260518/modelangelo_dataset_size_relion_adhoc_m80_20260518/runs_3m.tsv
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0000_reuse_20260520/modelangelo_dataset_size_relion_adhoc_m80_20260520/runs_3m.tsv
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise100_b80_state0025_reuse_20260520/modelangelo_dataset_size_relion_adhoc_m80_20260520/runs_3m.tsv
```

ModelAngelo scoring uses the same moving-region residue set as the state-50
scoring. References are `morph_001.pdb` for state 0 and `morph_025.pdb` for
state 25.
