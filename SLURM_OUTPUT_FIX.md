# SLURM Output File Management - Fix Documentation

## Issues Identified

### Issue 1: Invalid `--output-dir` Flag
The script was using `--output-dir="$SCRIPT_DIR/scripts/output"` as a crun flag, but this flag doesn't exist in crun. This caused errors:
```
/var/spool/slurm/d/job1163005/slurm_script: line 29: --output-dir=/home/scratch.dleshchev_other/heterogeneity_dev/scripts/output: No such file or directory
```

### Issue 2: crun Overrides SLURM Directives
The `#SBATCH --output=` directive in the batch script is ignored by crun. When using `crun -b`, crun controls where output files are placed (always in the current working directory), regardless of SLURM directives in the batch script.

## Solution Implemented

### 1. Removed Invalid Flags
Removed the non-existent `--output-dir` flags from both `submit_job()` and `submit_multinode_job()` functions.

### 2. Accept crun Behavior
Accept that crun places output files in the project root directory (`slurm-<jobid>.out`).

### 3. Added Output Organization Utility
Added a new action `organize-outputs` that moves all `slurm-*.out` files from the root directory to `scripts/output/`.

## Usage Workflow

### Submit a job:
```bash
./crun_recovar_workload.sh test-2gpu
```

This will:
- Submit the job via crun
- Create output file as `slurm-<jobid>.out` in the project root
- Display the job ID and monitoring command

### Organize output files:
```bash
./crun_recovar_workload.sh organize-outputs
```

This will:
- Find all `slurm-*.out` files in the project root
- Move them to `scripts/output/`
- Report how many files were moved

## Example Session

```bash
# Submit test job
$ ./crun_recovar_workload.sh test-2gpu
========================================
Submitting job: Test 2 GPUs
Task command: pixi run test-2gpu
Number of GPUs: 2
Time limit: 00:30:00
========================================
Generated batch script: /home/scratch.dleshchev_other/heterogeneity_dev/scripts/job_scripts/recovar_batch_964998.sh

Job ID: 1163243
Output file: slurm-1163243.out

To monitor: tail -f slurm-1163243.out
To organize outputs later, run: ./crun_recovar_workload.sh organize-outputs
Job submitted successfully!

# Monitor the job
$ tail -f slurm-1163243.out

# After job completion (or anytime), organize outputs
$ ./crun_recovar_workload.sh organize-outputs
========================================
Organizing output files
========================================
Moving 5 files to scripts/output/
  Moving slurm-1163243.out
  Moving slurm-1163216.out
  ...
Done! All output files moved to scripts/output/
```

## Why Not Use sbatch Directly?

We initially tried using `sbatch` directly instead of `crun -b` to respect the `#SBATCH --output` directive. However, this failed because:

1. `crun` is a wrapper that automatically handles:
   - GPU selection and allocation
   - Partition selection based on GPU requirements
   - Node constraint matching
   - Resource queries

2. Using `sbatch` directly requires manually specifying all these parameters, which are complex and cluster-specific.

Therefore, the pragmatic solution is to use `crun` and organize outputs post-submission.

## Files Modified

- `crun_recovar_workload.sh`:
  - Removed `--output-dir` flags from crun commands (lines 112-113, 138-139)
  - Enhanced `submit_job()` to capture and display job ID
  - Added `organize-outputs` action (lines 184-199, 290-302)
  - Updated help text to include the new utility

## Test Results

✅ Job submission works without errors
✅ Output files are created in project root
✅ `organize-outputs` successfully moves files to `scripts/output/`
✅ Test job `test-2gpu` (Job ID: 1163243) runs successfully
✅ No more "No such file or directory" errors

## Date
February 4, 2026
