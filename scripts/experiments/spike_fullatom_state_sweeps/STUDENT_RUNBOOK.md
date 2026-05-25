# Student Quick Guide: Full-Atom Spike Experiments On Della

This is the short path for a student on Della to clone the branch, run the
experiment, and download the results. Do not run from `mg6942`'s home or
scratch directories.

## What Each Step Does

| step | purpose |
|---|---|
| 1. clone | downloads RECOVAR and checks out this experiment branch |
| 2. setup | creates the pixi environment and builds the CUDA backprojector |
| 3. inputs | checks that the shared spike PDBs and masks are readable |
| 4. smoke run | runs one 10k-image job to catch setup/path problems |
| 5. full run | runs the dataset-size sweep and saves all kernel estimates |
| 6. postprocess | makes FSC/resolution/mean-subtracted metric plots |
| 7. download | copies plots, CSVs, masks, and volumes to a local machine |

## 1. Clone

```bash
export RECOVAR_STUDENT_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student_$(date +%Y%m%d)
mkdir -p "$RECOVAR_STUDENT_ROOT"/{clone,slurmo,tmp,pixi_home,rattler_cache}

cd "$RECOVAR_STUDENT_ROOT/clone"
git clone git@github.com:ma-gilles/recovar.git
cd recovar
git checkout codex/kernel-bandwidth-student-clean
```

Use `https://github.com/ma-gilles/recovar.git` instead if GitHub SSH is not
configured.

## 2. Set Up

```bash
cd "$RECOVAR_STUDENT_ROOT/clone/recovar"

export AGENT_ID=student_setup_$(date +%Y%m%d_%H%M%S)
export TMPDIR="$RECOVAR_STUDENT_ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$RECOVAR_STUDENT_ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$RECOVAR_STUDENT_ROOT/rattler_cache/$AGENT_ID"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

pixi install
pixi run install-recovar

PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed

module load cudatoolkit/12.8 || true
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
module unload cudatoolkit/12.8 || true
```

Check that imports come from this clone:

```bash
pixi run smoke-import-recovar
"$PIXI_PY" - <<'PY'
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
print("recovar", pathlib.Path(recovar.__file__).resolve())
print("jax", pathlib.Path(jax.__file__).resolve())
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
PY
```

## 3. Check Inputs

```bash
export PDB_DIR=/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs
export DEFAULT_MASK=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc

ls -lh "$PDB_DIR"/morph_001.pdb "$PDB_DIR"/morph_050.pdb "$PDB_DIR"/morph_100.pdb
ls -lh "$DEFAULT_MASK"
```

If a path is not readable, copy that file somewhere readable and change the
corresponding variable.

## 4. Run A 10k Smoke Job

```bash
cd "$RECOVAR_STUDENT_ROOT/clone/recovar"

export SMOKE_ROOT="$RECOVAR_STUDENT_ROOT/spike_smoke_noise100_b80"
export SMOKE_SHARED="$RECOVAR_STUDENT_ROOT/spike_smoke_noise100_b80_shared"

sbatch \
  --array=0-0 \
  --output="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.out" \
  --error="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.err" \
  --export=ALL,WORKDIR="$PWD",SCRATCH_ROOT="$RECOVAR_STUDENT_ROOT",BASE_ROOT="$SMOKE_ROOT",SHARED_ROOT="$SMOKE_SHARED",PDB_DIR="$PDB_DIR",MASK="$DEFAULT_MASK",N_IMAGES_VALUES_STR="10000",NOISE_LEVEL=100.0,RENDER_BFACTOR=80 \
  scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f "$RECOVAR_STUDENT_ROOT"/slurmo/spike-fa-n100-b80-*.out
```

Smoke success means this file exists:

```bash
ls -lh "$SMOKE_ROOT/n00010000/runs/n00010000_seed0000/07_compute_state/state000_unfil.mrc"
```

## 5. Run The Full Sweep

```bash
export FULL_ROOT="$RECOVAR_STUDENT_ROOT/spike_fullatom_consistency_grid256_noise100_b80"
export FULL_SHARED="$RECOVAR_STUDENT_ROOT/spike_fullatom_consistency_grid256_noise100_b80_shared"
export N_IMAGES_VALUES_STR="10000 30000 100000 300000 1000000"

sbatch \
  --array=0-4%3 \
  --output="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.out" \
  --error="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.err" \
  --export=ALL,WORKDIR="$PWD",SCRATCH_ROOT="$RECOVAR_STUDENT_ROOT",BASE_ROOT="$FULL_ROOT",SHARED_ROOT="$FULL_SHARED",PDB_DIR="$PDB_DIR",MASK="$DEFAULT_MASK",N_IMAGES_VALUES_STR="$N_IMAGES_VALUES_STR",NOISE_LEVEL=100.0,RENDER_BFACTOR=80 \
  scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
```

For 3M images, use:

```bash
export N_IMAGES_VALUES_STR="10000 30000 100000 300000 1000000 3000000"
# Submit with --array=0-5%2.
```

## 6. Postprocess

```bash
sbatch \
  --output="$RECOVAR_STUDENT_ROOT/slurmo/%x-%j.out" \
  --error="$RECOVAR_STUDENT_ROOT/slurmo/%x-%j.err" \
  --export=ALL,WORKDIR="$PWD",SCRATCH_ROOT="$RECOVAR_STUDENT_ROOT",ROOT="$FULL_ROOT",LABEL="$(basename "$FULL_ROOT")",TARGET_STATE=50,MASK="$DEFAULT_MASK" \
  scripts/experiments/spike_fullatom_state_sweeps/postprocess_fullatom_dataset_size.sbatch
```

Look at:

```bash
ls "$FULL_ROOT/plots"
ls "$FULL_ROOT/mean_subtracted_metrics_20260518_r12"
```

## 7. Download Results

Run this from a local machine. Replace `della` and `REMOTE_FULL_ROOT` as needed.

```bash
REMOTE_FULL_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/<student_user>/<run_root>

rsync -avP --include='*/' \
  --include='*.png' --include='*.pdf' --include='*.csv' --include='*.mrc' \
  --exclude='*' \
  della:"$REMOTE_FULL_ROOT/" ./recovar_spike_results/
```

Only the compute_state volumes for 100k images:

```bash
rsync -avP \
  della:"$REMOTE_FULL_ROOT/n00100000/runs/n00100000_seed0000/07_compute_state/" \
  ./n00100000_compute_state/
```

## Optional

State 0 or state 25 reuse:
`scripts/experiments/spike_fullatom_state_sweeps/submit_state25_compute_state_reuse.sbatch`

ModelAngelo scoring:
`scripts/experiments/modelangelo_spike_scoring/README.md`

Method benchmarks:
`scripts/experiments/spike_fullatom_method_benchmark/README.md`

## Common Problems

| symptom | fix |
|---|---|
| `recovar.__file__` points outside the clone | rerun setup with `PYTHONNOUSERSITE=1` and no conda env |
| JAX sees no GPU in Slurm | check the queue, partition, and `nvidia-smi` in the log |
| CUDA library is missing | rerun `PYTHON="$(pixi run which python)" make -C recovar/cuda clean all` |
| shared input is unreadable | copy it to a readable path and override the env var |
