# Handoff: corrected cryoDRGN and CryoSPARC 3DFlex sweep

This is the runbook for repeating the method sweep on a new dataset using the
pipeline choices that worked in the spike full-atom noise10/B100 debugging
round. The main corrections were sign handling, cryoDRGN latent dimension, the
CryoSPARC consensus source, and mean-latent/mean-embedding evaluation.

## Non-negotiable choices

- Run a small sanity point first, usually 100k images, before launching the full
  sweep.
- Use sign-flipped particles consistently.
  - cryoDRGN: pass `--uninvert-data`.
  - CryoSPARC import_particles: set `sign=1`.
  - Do not use the older CryoSPARC `sign=-1` default from the generic helper.
- cryoDRGN primary setting is `zdim=1`.
  - Do not launch `zdim=8` unless it is explicitly a comparison branch.
  - The previous corrected-sign zdim1 branch looked better than zdim8.
- For 3DFlex, do not use a RECOVAR mean map as the consensus map.
  - Import sign-flipped particles.
  - Run CryoSPARC Homogeneous Reconstruction Only.
  - Use that homogeneous reconstruction volume as the 3DFlex consensus input.
- Do not import or use the loose simulation mask as the 3DFlex mask.
  - Preferred branch: create a dataset-specific solvent mask from the
    homogeneous reconstruction, resampled to the 3DFlex prep box when needed,
    then connect that mask to mesh prep.
  - If the threshold is uncertain, also run a no-external-mask/default-mask
    branch, but keep it labeled separately.
- 3DFlex primary setting is `K=1`.
  - `K=2` can be added as a comparison branch after the K=1 pipeline is healthy.
- Evaluate methods at the mean embedding/latent coordinate of the images in
  each ground-truth state.
  - cryoDRGN: mean of `z.<epoch>.pkl` vectors for each state, then decode.
  - 3DFlex: mean of CryoSPARC 3DFlex latent coordinates for each state, then
    run `flex_generate` with those coordinates.
- Use the same mask convention across RECOVAR, cryoDRGN, and 3DFlex comparisons.
  Report the exact mask path and whether it was soft or binarized.

## Paths and dataset contract

Set these for the new sweep:

```bash
WORKDIR=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar
SOURCE_ROOT=/path/to/source_sweep_root
BENCH_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/<new_method_sweep_name>
IMAGE_COUNTS="100000 300000 1000000 3000000"   # adapt as needed
STATES="0,25,50"                               # adapt if the dataset differs
VOXEL_SIZE=1.25                                # adapt if the dataset differs
```

Expected source layout for each image count:

```text
$SOURCE_ROOT/n########/runs/n########_seed0000/
  03_dataset/particles.256.mrcs
  03_dataset/particles.star
  03_dataset/poses.pkl
  03_dataset/ctf.pkl
  03_dataset/state_assignment.npy
  04_ground_truth/gt_vol0000.mrc
  04_ground_truth/gt_vol0025.mrc
  04_ground_truth/gt_vol0050.mrc
  05_masks/focus_mask_moving.mrc
  05_masks/volume_mask_union.mrc
```

For broad-mask RECOVAR-style plots, use the dataset's broad soft mask if one
exists. In the prior noise10 run, this was:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc
```

Do not silently reuse that mask on a different box, voxel size, or molecule.

## Environment rules

Use pixi for RECOVAR-side scripts:

```bash
cd "$WORKDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY="$WORKDIR/.pixi/envs/default/bin/python"
```

Use the cryoDRGN module for cryoDRGN:

```bash
set +u
module purge
module load cryodrgn/3.4.3
set -u
```

Use the CryoSPARC tools module for CryoSPARC scripts:

```bash
set +u
module load cryosparc-tools/5.0.3
set -u
```

The `set +u` guard matters because the module system can reference `PS1` and
fail under `set -u`.

Use the module Python for scripts that import `cryodrgn` or `cryosparc.tools`.
Use pixi Python for RECOVAR-side plotting/scoring scripts that do not import
those packages.

## cryoDRGN training

Reference submitter:

```text
scripts/experiments/spike_fullatom_method_benchmark/submit_cryodrgn_noise10_b100_zdim1_correct_sign_single_size.sbatch
```

That script already applies the corrected choices:

- `--zdim 1`
- `--uninvert-data`
- `--lazy`
- `--seed 0`
- `--num-epochs 20`
- `--num-workers 4`
- `--max-threads $SLURM_CPUS_PER_TASK`
- optional `--multigpu` when multiple GPUs are visible and `USE_MULTIGPU=1`

The training command shape is:

```bash
cryodrgn train_vae "$DATASET/particles.256.mrcs" \
  --poses "$DATASET/poses.pkl" \
  --ctf "$DATASET/ctf.pkl" \
  --zdim 1 \
  --lazy \
  --seed 0 \
  --num-epochs 20 \
  --num-workers 4 \
  --max-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --uninvert-data \
  -o "$OUTDIR"
```

For cryoDRGN, keep the multiprocessing temp path short. The prior broken jobs
failed with `OSError: AF_UNIX path too long` before producing checkpoints.
Use `/scratch/gpfs/CRYOEM/gilleslab/tmp` as the temp root, but keep the suffix
short:

```bash
export DRGN_TMP_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp
export TMPDIR="$DRGN_TMP_ROOT/drgn_${SLURM_JOB_ID:-manual}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
```

Do not put cryoDRGN `TMPDIR` under a deeply nested bench-root path.

Submit examples:

```bash
cd "$WORKDIR"
mkdir -p "$BENCH_ROOT/slurm"

for n in $IMAGE_COUNTS; do
  N_IMAGES="$n" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BENCH_ROOT="$BENCH_ROOT" \
  WORKDIR="$WORKDIR" \
  DRGN_TMP_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp \
  USE_MULTIGPU=1 \
  sbatch \
    --job-name "drgn-z1-n$n" \
    --output "$BENCH_ROOT/slurm/%x-%j.out" \
    --error "$BENCH_ROOT/slurm/%x-%j.err" \
    scripts/experiments/spike_fullatom_method_benchmark/submit_cryodrgn_noise10_b100_zdim1_correct_sign_single_size.sbatch
done
```

For the very large image counts, request multiple GPUs only if the cluster can
start them in a reasonable time:

```bash
N_IMAGES=3000000 USE_MULTIGPU=1 sbatch --gres=gpu:a100:4 --time=120:00:00 ...
```

The script only adds `--multigpu` when more than one GPU is actually visible.

Monitor:

```bash
squeue -u "$USER" -o "%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R"
tail -n 80 "$BENCH_ROOT/slurm/<job>.out"
tail -n 80 "$BENCH_ROOT/n########/cryodrgn/zdim1/run.log"
ls -lh "$BENCH_ROOT/n########/cryodrgn/zdim1"/weights.*.pkl
```

Healthy cryoDRGN runs produce `weights.<epoch>.pkl` and `z.<epoch>.pkl` after
each epoch.

## cryoDRGN evaluation

Decode and score the mean embedding for each state, not arbitrary latent
positions. Reuse the existing helpers:

```text
scripts/experiments/spike_fullatom_method_benchmark/evaluation_compute_mean_embeddings.py
scripts/experiments/spike_fullatom_method_benchmark/evaluation_decode_mean_embeddings.py
scripts/experiments/spike_fullatom_method_benchmark/evaluation_score_decoded_volumes.py
scripts/experiments/spike_fullatom_method_benchmark/plot_cryodrgn_sweep_with_nearest_gt.py
```

Example for final epoch 19:

```bash
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
PIXI_PY="$WORKDIR/.pixi/envs/default/bin/python"

for n in $IMAGE_COUNTS; do
  label="$(printf 'n%08d' "$n")"
  source_run="$SOURCE_ROOT/$label/runs/${label}_seed0000"
  run_root="$BENCH_ROOT/$label"
  eval_root="$BENCH_ROOT/evaluation_size_sweep_zdim1_correct_sign/$label"

  "$PIXI_PY" scripts/experiments/spike_fullatom_method_benchmark/evaluation_compute_mean_embeddings.py \
    --source-run "$source_run" \
    --bench-root "$run_root" \
    --out-root "$eval_root" \
    --methods cryodrgn \
    --labels "$STATES" \
    --cryodrgn-epoch 19

  set +u
  module load cryodrgn/3.4.3
  set -u
  python scripts/experiments/spike_fullatom_method_benchmark/evaluation_decode_mean_embeddings.py \
    --bench-root "$run_root" \
    --evaluation-root "$eval_root" \
    --methods cryodrgn \
    --apix "$VOXEL_SIZE" \
    --device 0 \
    --overwrite
done
```

For RECOVAR-style comparison curves against nearest-GT averages:

```bash
"$PIXI_PY" scripts/experiments/spike_fullatom_method_benchmark/plot_cryodrgn_sweep_with_nearest_gt.py \
  --root "$BENCH_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --evaluation-root "$BENCH_ROOT/evaluation_size_sweep_zdim1_correct_sign" \
  --out-dir "$BENCH_ROOT/evaluation_size_sweep_zdim1_correct_sign/cryodrgn_curves_with_nearest100_gt_broadmask_soft_epoch019" \
  --sweep-axis image_count \
  --image-counts "$(echo "$IMAGE_COUNTS" | tr ' ' ',')" \
  --epoch 19 \
  --states "$STATES" \
  --nearest-count 100 \
  --mask /path/to/dataset_broad_mask.mrc \
  --gt-average-half-widths 5,10,20
```

For an epoch sweep at one image count, use the same script with
`--sweep-axis epoch --image-counts <n> --epochs 0-19`.

## CryoSPARC 3DFlex pipeline

Reference scripts:

```text
scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_100k.py
scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_1m_j397_mask_k1.py
scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_generate_mean_latents.py
scripts/experiments/spike_fullatom_method_benchmark/score_cryosparc_3dflex_mean_latents.py
```

The 1M J397 script shows the corrected job graph, but do not hardcode its
`J409` mask for a new dataset unless intentionally testing that exact old mask.
For a new dataset, create a new mask from that dataset's homogeneous
reconstruction and record its CryoSPARC job ID in the manifest.

Correct job graph for each image count:

1. `import_particles`
   - `particle_meta_path = $SOURCE_RUN/03_dataset/particles.star`
   - `particle_blob_path = $SOURCE_RUN/03_dataset`
   - `ignore_pose = 0`
   - `ignore_splits = 1`
   - `sign = 1`
2. `homo_reconstruct`
   - Connect particles from import.
   - This is only to get the consensus map for 3DFlex and to inspect sign.
3. `volume_tools` mask creation from the homogeneous reconstruction
   - Make a solvent mask from the homogeneous volume.
   - Resample/crop to the 3DFlex prep box if needed.
   - Threshold the mask so it is tight enough around the density.
   - Record the threshold, box, and output job ID.
4. `flex_prep`
   - Connect imported particles and homogeneous reconstruction volume.
   - Use `bin_size_pix = 128` unless there is a specific reason to change it.
5. `flex_meshprep`
   - Connect `volume` from `flex_prep`.
   - Connect the dataset-specific mask if using the mask branch.
   - Parameters that worked:
     - `mask_in_lowpass_A = 10.0`
     - `mask_in_threshold_level = 0.5`
     - `mask_dilate_A = 2.0`
     - `mask_pad_A = 5.0`
     - `tetra_num_cells = 20`
6. `flex_train`
   - Primary: `flex_K = 1`.
   - Optional comparison: `flex_K = 2`.
7. `flex_highres`
   - Connect `flex_model` from train and particles from prep.
8. `flex_generate`
   - Generate mean-latent maps for each GT state.
   - Also generate a default motion series along latent coordinates, following
     the CryoSPARC 3DFlex tutorial, for visual inspection.

CryoSPARC settings:

```text
Project default: P587 unless the new dataset uses a new project.
Import lane: 24hrs
GPU lane: 48hrs-a100 for large runs
Credentials notebook: /home/mg6942/recovar/20231229_3dflex.ipynb
```

3DFlex does not expose a safe multi-GPU path for these jobs in our tested
workflow. Do not assume requesting multiple GPUs will speed it up. The
CryoSPARC job specs observed for homogeneous reconstruction, flex_train, and
flex_highres used one GPU.

Save a JSON manifest under:

```text
$BENCH_ROOT/cryosparc_3dflex/
```

The manifest must include at least:

- source run
- image count
- CryoSPARC project/workspace
- import sign
- consensus source
- mask mode and mask job ID
- K values
- all job IDs
- lanes
- submission time

## 3DFlex mean-latent evaluation

After `flex_train` and `flex_highres` finish, evaluate at mean latent
coordinates for each GT state:

1. Locate the latest `Jxxx_latents_*.cs` from the 3DFlex train job.
2. Load `$SOURCE_RUN/03_dataset/state_assignment.npy`.
3. Compute the mean latent vector for each state.
4. Create a CryoSPARC external job containing those latent coordinates.
5. Run `flex_generate` connected to:
   - `flex_model` from the train job
   - `volume_flex` from the high-res reconstruction job
   - the external mean-latent table
6. Score the generated maps against the matching GT maps with the agreed mask.

Reference helper:

```bash
set +u
module load cryosparc-tools/5.0.3
set -u
python scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_generate_mean_latents.py \
  --source-run "$SOURCE_RUN/n00100000/runs/n00100000_seed0000" \
  --bench-root "$BENCH_ROOT" \
  --submit
```

For a new sweep, update that helper or write a small wrapper so it reads the
train/highres job IDs from each image-count manifest instead of using old
hardcoded jobs.

Then score:

```bash
"$PIXI_PY" scripts/experiments/spike_fullatom_method_benchmark/score_cryosparc_3dflex_mean_latents.py \
  --source-run "$source_run" \
  --bench-root "$BENCH_ROOT" \
  --manifest "$BENCH_ROOT/cryosparc_3dflex/<manifest>.json"
```

If broad-mask curves are required, extend the scorer or reuse the same
shell-metric code path as the cryoDRGN/RECOVAR plotting scripts. Report both
the focus/global/broad mask choice and whether the mask was soft or binarized.

## Monitoring checks

cryoDRGN:

- Check Slurm output and `run.log`.
- Confirm `weights.<epoch>.pkl` and `z.<epoch>.pkl` are appearing.
- Decode one early map once the first useful checkpoint exists and inspect it
  before trusting a large sweep.
- If there is `OSError: AF_UNIX path too long`, stop that branch and fix
  `TMPDIR`.

CryoSPARC:

- After import plus homogeneous reconstruction, inspect the homogeneous map.
  Wrong sign means the import sign is wrong.
- CryoSPARC jobs that show `launched` may be waiting in the CryoSPARC Slurm
  lane. Check the CryoSPARC UI and, if possible, the Slurm queue for the
  cryoSPARC runner user/account.
- Do not cancel old branches unless the user explicitly asks. Submit corrected
  branches with clear names and manifests.

## Required report at the end

Report:

- dataset/source root and output root
- exact image counts and states
- cryoDRGN job IDs, commands, output directories, and checkpoint status
- 3DFlex project/workspace/job IDs and manifest paths
- sign settings used by each method
- 3DFlex consensus source and mask source
- mean-embedding/mean-latent generation status
- metrics CSV paths and plot paths
- any failed jobs and the key error lines
- exact commands needed to reproduce evaluation

## Copy-paste prompt for another agent

```text
You are working in /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar on Princeton Della.

Goal: run a corrected cryoDRGN plus CryoSPARC 3DFlex method sweep on a new
dataset. Start with 100k images as a sanity check, then scale to the requested
image counts only after sign, checkpoints, homogeneous reconstruction, and one
decoded/generated map look correct.

Read and follow this runbook first:
scripts/experiments/spike_fullatom_method_benchmark/HANDOFF_CORRECTED_CRYODRGN_3DFLEX_SWEEP.md

Set:
SOURCE_ROOT=<new source sweep root>
BENCH_ROOT=<new output root under /scratch/gpfs/CRYOEM/gilleslab/tmp or /scratch/gpfs/GILLES/mg6942>
IMAGE_COUNTS=<space-separated counts>
STATES=0,25,50 unless the dataset defines different GT states
VOXEL_SIZE=<dataset voxel size>

Expected per-count input layout:
$SOURCE_ROOT/n########/runs/n########_seed0000/03_dataset/particles.256.mrcs
$SOURCE_ROOT/n########/runs/n########_seed0000/03_dataset/particles.star
$SOURCE_ROOT/n########/runs/n########_seed0000/03_dataset/poses.pkl
$SOURCE_ROOT/n########/runs/n########_seed0000/03_dataset/ctf.pkl
$SOURCE_ROOT/n########/runs/n########_seed0000/03_dataset/state_assignment.npy
$SOURCE_ROOT/n########/runs/n########_seed0000/04_ground_truth/gt_vol0000.mrc
$SOURCE_ROOT/n########/runs/n########_seed0000/04_ground_truth/gt_vol0025.mrc
$SOURCE_ROOT/n########/runs/n########_seed0000/04_ground_truth/gt_vol0050.mrc

cryoDRGN:
- Use module cryodrgn/3.4.3, not pixi Python.
- Train zdim=1 only as the primary branch.
- Use corrected sign: pass --uninvert-data.
- Use lazy loading, seed 0, 20 epochs, 4 workers, max threads equal Slurm CPUs.
- Keep TMPDIR short, rooted at /scratch/gpfs/CRYOEM/gilleslab/tmp, e.g.
  /scratch/gpfs/CRYOEM/gilleslab/tmp/drgn_${SLURM_JOB_ID}.
- Submit with scripts/experiments/spike_fullatom_method_benchmark/submit_cryodrgn_noise10_b100_zdim1_correct_sign_single_size.sbatch, overriding SOURCE_ROOT, BENCH_ROOT, and N_IMAGES.
- Use USE_MULTIGPU=1 and request multiple GPUs only for very large image counts if that will not block scheduling too badly.
- Monitor for weights.<epoch>.pkl and z.<epoch>.pkl. If there are no checkpoints and AF_UNIX path errors appear, fix TMPDIR and resubmit.
- Decode a volume from an early/final checkpoint to visually verify sign before trusting the whole sweep.

cryoDRGN evaluation:
- Compute mean embeddings per GT state from z.<epoch>.pkl and state_assignment.npy.
- Decode the mean embeddings.
- Produce RECOVAR-style FSC/error curves with plot_cryodrgn_sweep_with_nearest_gt.py, using nearest-count 100 and the agreed soft mask.
- For final comparison, use epoch 19. Also make an epoch sweep for the largest completed count if requested.

3DFlex:
- Use module cryosparc-tools/5.0.3. If using set -u, wrap module load with set +u / set -u.
- Use CryoSPARC credentials from /home/mg6942/recovar/20231229_3dflex.ipynb unless told otherwise.
- Import particles with sign=1. Do not use sign=-1.
- Run homogeneous reconstruction only after import and inspect it; this is the consensus map for 3DFlex.
- Do not use the RECOVAR mean map as consensus.
- Do not use the loose simulation mask as the 3DFlex mask.
- Create a dataset-specific solvent mask from the homogeneous reconstruction, resample/crop to the 3DFlex prep box if necessary, and connect that mask to flex_meshprep. If threshold/mask quality is uncertain, also run a separate no-external-mask/default branch and label it clearly.
- Run flex_prep with bin_size_pix=128, then flex_meshprep, flex_train K=1, and flex_highres K=1. K=2 is optional after K=1 is healthy.
- Save a manifest JSON under $BENCH_ROOT/cryosparc_3dflex with all job IDs, sign, consensus source, mask source, lanes, and parameters.
- 3DFlex should be treated as single-GPU in this workflow; do not assume multi-GPU speedup.

3DFlex evaluation:
- From the completed flex_train job, load the latest latent .cs file.
- Compute the mean latent coordinate of images in each GT state using state_assignment.npy.
- Run flex_generate at those mean latent coordinates using the train flex_model and highres volume_flex.
- Also generate motion along latent coordinates as in the CryoSPARC 3DFlex tutorial for visual inspection.
- Score mean-latent generated maps vs GT states using the same mask convention as the cryoDRGN/RECOVAR plots. Save CSVs and plots.

At the end report:
- all Slurm job IDs and CryoSPARC job IDs
- output roots and manifest paths
- exact train/import/flex commands or parameters
- checkpoint and manifest existence
- sign settings and mask sources
- metric CSVs and plot paths
- failed jobs with key log excerpts
- commands to reproduce the evaluation
```
