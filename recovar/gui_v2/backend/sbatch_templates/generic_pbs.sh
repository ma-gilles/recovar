#!/bin/bash
# Generic PBS / Torque submit template for recovar.
#
# This template is for sites that use PBS-style schedulers. The structured
# slurm_directives block is omitted because PBS uses #PBS, not #SBATCH;
# instead, fill in the directives literally below. recovar's executor
# is SLURM-only today, but you can use this template via
# `recovar gui --executor local` plus a separate qsub wrapper.

#PBS -N recovar
#PBS -q {{ partition or 'default' }}
{% if account %}#PBS -A {{ account }}{% endif %}
{% if gpus %}#PBS -l select=1:ncpus={{ cpus }}:mem={{ memory }}:ngpus={{ gpus }}{% else %}#PBS -l select=1:ncpus={{ cpus }}:mem={{ memory }}{% endif %}
#PBS -l walltime={{ time }}
#PBS -o {{ output_path }}
#PBS -j oe

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# Most PBS sites expose $TMPDIR per-job; only fall back if absent.
if [ -z "${TMPDIR:-}" ]; then
    export TMPDIR="$(mktemp -d -t recovar-XXXXXX)"
fi
mkdir -p "$TMPDIR"

export PATH={{ pixi_bin_dir }}:$PATH
{{ extra_exports }}

# ── Run the actual command ──
{{ command }}
exit $?
