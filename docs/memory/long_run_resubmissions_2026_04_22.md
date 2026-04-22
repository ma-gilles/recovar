# Long-Run Resubmissions: 2026-04-22

## Purpose

Resubmit the long RECOVAR-vs-RELION quality runs that were not usable as
final evidence:

- 11-iteration 5k replay on the current speed-optimized branch state
- 20k benchmark at noise `1.0`
- 20k benchmark at noise `0.1`

This note is the single audit trail for the reruns: old failure mode,
new command/job metadata, and final outcome.

## Failure Modes Being Fixed

### 11-iteration 5k replay

- Old job: `7219478`
- Old log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-5k11-v1-7219478.out`
- Problem:
  timed out on the old local-search path and later iterations expanded
  into pathological multi-million-rotation unions, so it is not
  representative of the current branch state.

### 20k benchmark jobs

- Old job (`noise=1.0`): `7227985`
- Old log (`noise=1.0`):
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-bench-7227985.out`
- Old job (`noise=0.1`): `7228047`
- Old log (`noise=0.1`):
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-n0p1-7228047.out`
- Problem:
  `scripts/run_comparison.py` defaulted to `DATA_DIR/relion_ref`, while
  the benchmark harness writes RELION output under
  `DATA_DIR/relion_ref_benchmark`. The comparison phase therefore failed
  with a missing `run_it001_data.star` even though RELION output itself
  existed.

## Code Fix Used For Resubmission

- `scripts/run_comparison.py`
  - added `--relion_ref_dir`
  - autodetects `relion_ref_benchmark` when `relion_ref` is absent
- `scripts/run_relion_parity_benchmark_slurm.sh`
  - now passes `--relion_ref_dir "$RELION_REF_DIR"` explicitly
  - default `REPO_DIR` resolves from the script location so the current
    checkout is used by default

## Resubmissions

### 11-iteration 5k replay

- Git commit:
  `e08d92da`
- Command / script:
  `sbatch /scratch/gpfs/GILLES/mg6942/tmp/codex_resubmit_jobs_20260422_085035_21094/parity_5k11_e08d92da.sbatch`
- Job id:
  `7240804`
- Output dir:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/multi_iter_11_full_e08d92da_grouped4096_gt`
- Log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-5k11-v2-7240804.out`
- Status:
  `RUNNING`
- Live note:
  as of 2026-04-22 09:18 ET, the job is running on `della-h20g5`
  (`squeue` elapsed `26:07`). The log still contains the old
  `pip uninstall -y recovar` traceback near the top, but the job is
  past bootstrap and actively running RECOVAR local-search EM.
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`

### 20k benchmark, noise `1.0`

- Git commit:
  `e08d92da`
- Command / script:
  `sbatch --job-name=parity-20k-bench-v2 --output=/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-bench-v2-%j.out --export=ALL,REPO_DIR=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar,DATA_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_20k_benchmark,OUTPUT_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_20k_benchmark/comparison_results_v2,OUR_RESULTS_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_20k_benchmark/our_results_relion_v2,N_IMAGES=20000,GRID_SIZE=128,NOISE_LEVEL=1.0,MAX_ITER=10 scripts/run_relion_parity_benchmark_slurm.sh`
- Job id:
  `7240805`
- Data dir:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_20k_benchmark`
- Log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-bench-v2-7240805.out`
- Status:
  `RUNNING`
- Live note:
  as of 2026-04-22 09:18 ET, the job is running on `della-h20g1`
  (`squeue` elapsed `26:07`). The log still contains the old
  `pip uninstall -y recovar` traceback near the top, but the job is
  past bootstrap and actively running RECOVAR EM.
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`

### 20k benchmark, noise `0.1`

- Git commit:
  `e08d92da`
- Command / script:
  `sbatch --job-name=parity-20k-n0p1-v2 --output=/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-n0p1-v2-%j.out --export=ALL,REPO_DIR=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar,DATA_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise0p1_20k_benchmark,OUTPUT_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise0p1_20k_benchmark/comparison_results_v2,OUR_RESULTS_DIR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise0p1_20k_benchmark/our_results_relion_v2,N_IMAGES=20000,GRID_SIZE=128,NOISE_LEVEL=0.1,MAX_ITER=15 scripts/run_relion_parity_benchmark_slurm.sh`
- Job id:
  `7240806`
- Data dir:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise0p1_20k_benchmark`
- Log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-n0p1-v2-7240806.out`
- Status:
  `RUNNING`
- Live note:
  as of 2026-04-22 09:18 ET, the job is running on `della-h20g2`
  (`squeue` elapsed `26:07`) and is already through RECOVAR EM with no
  bootstrap traceback at the top of the log.
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`
