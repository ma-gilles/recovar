#!/bin/bash
# Generic SLURM submit template for recovar.
#
# Variables (Jinja2 syntax):
#   {{ slurm_directives }}  — pre-built #SBATCH lines (job_name, partition,
#                             account, GPU spec, ntasks, cpus, mem, time,
#                             output, plus anything from raw_directives).
#                             Use this if you want recovar to compose the
#                             #SBATCH block; or replace it with your own
#                             literal #SBATCH lines for full control.
#   {{ tmpdir_block }}      — SLURM_TMPDIR-aware tmp setup.
#   {{ cleanup_block }}     — trap that removes anything recovar created.
#   {{ extra_exports }}     — env vars passed via the executor.
#   {{ pixi_bin_dir }}      — quoted path to the recovar interpreter's bin
#                             dir (so the `recovar` shebang resolves).
#   {{ command }}           — the recovar invocation, already shlex.join'd.

{{ slurm_directives }}

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
{{ tmpdir_block }}
export PATH={{ pixi_bin_dir }}:$PATH
{{ extra_exports }}

# ── Cleanup trap ──
{{ cleanup_block }}

# ── Site customizations go here ──
# module load cuda/12.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate myenv

# ── Run the actual command ──
{{ command }}
EXIT_CODE=$?

exit $EXIT_CODE
