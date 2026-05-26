# Student Spike Experiment: Minimal Instructions

This is intentionally script-first. The student should not copy a page of
`sbatch` commands by hand.

## 1. Clone And Set Up

From a login node:

```bash
STUDENT_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student
mkdir -p "$STUDENT_ROOT/clone"
git clone --branch codex/kernel-bandwidth-student-clean \
  git@github.com:ma-gilles/recovar.git "$STUDENT_ROOT/clone/recovar"

"$STUDENT_ROOT/clone/recovar/scripts/experiments/spike_fullatom_state_sweeps/download_student_spike_experiment.sh" \
  "$STUDENT_ROOT"
```

The setup script uses that fresh clone, builds pixi, builds the CUDA
backprojector, checks imports, and writes:

```text
/scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student/student_spike_env.sh
```

## 2. Run

```bash
source /scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student/student_spike_env.sh

$RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh smoke
$RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh full
$RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh postprocess
```

What those do:

| command | does |
|---|---|
| `smoke` | one 10k-image test run |
| `full` | 10k, 30k, 100k, 300k, 1M image sweep |
| `postprocess` | FSC/resolution/mean-subtracted plots |
| `plot100k` | shell metrics for the completed 100k compute_state map |

Logs go here:

```text
$RECOVAR_STUDENT_ROOT/slurmo/
```

Results go here:

```text
$RECOVAR_STUDENT_ROOT/spike_fullatom_consistency_grid256_noise100_b80/
```

The run script prints the exact output root, Slurm log directory, and
per-size run directories before it submits jobs.

## 3. Plot One Compute-State Result

After the 100k run exists:

```bash
$RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh plot100k
```

`plot100k` uses the standard full-sweep location by default. If the run was
launched into a custom root, point it at the exact run directory:

```bash
PLOT_RUN_DIR=/path/to/n00100000_seed0000 \
  $RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh plot100k
```

This writes:

```text
<run-dir>/plots/compute_state_shell_metrics/shell_metrics.png
<run-dir>/plots/compute_state_shell_metrics/shell_metrics.csv
<run-dir>/plots/compute_state_shell_metrics/summary.csv
```

The plot shows masked FSC vs GT, relative Fourier error per shell, log-scale
relative error, and cumulative relative error.

## 4. Optional: Copy Results Locally

Run from a local machine:

```bash
REMOTE_FULL_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/<student_user>/spike_fullatom_student/spike_fullatom_consistency_grid256_noise100_b80

rsync -avP --include='*/' \
  --include='*.png' --include='*.pdf' --include='*.csv' --include='*.mrc' \
  --exclude='*' \
  della:"$REMOTE_FULL_ROOT/" ./recovar_spike_results/
```

Only 100k compute-state volumes:

```bash
rsync -avP \
  della:"$REMOTE_FULL_ROOT/n00100000/runs/n00100000_seed0000/07_compute_state/" \
  ./n00100000_compute_state/
```

## Hardcoded Inputs

The scripts default to:

```text
PDB_DIR=/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs
MASK=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc
noise_level=100
render_bfactor=80
target_state=50
pipeline=oracle basis/embedding path (`--use-oracle-pipeline`)
```

Override with environment variables only if one of those paths is unreadable.
