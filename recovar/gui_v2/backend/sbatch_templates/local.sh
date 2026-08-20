#!/bin/bash
# Local-execution template for recovar.
#
# When using the local executor (no SLURM), this template is informational
# only — the local executor doesn't actually invoke `sbatch`. It can still
# be used by `POST /api/jobs/preview-sbatch` so users can see the command
# that will run.

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
{{ extra_exports }}

# ── Run the actual command ──
{{ command }}
exit $?
