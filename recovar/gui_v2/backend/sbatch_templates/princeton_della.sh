#!/bin/bash
# Princeton Della template — provided as a worked example of a non-trivial
# site config. Other sites should clone this, change the partition/account/
# scratch path, and point their per-project recovar.toml at it via
# `template_path = "/abs/path/to/your_template.sh"`.

{{ slurm_directives }}

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
{{ tmpdir_block }}
export PATH={{ pixi_bin_dir }}:$PATH
{{ extra_exports }}

# ── Cleanup trap ──
{{ cleanup_block }}

# ── Run the actual command ──
{{ command }}
EXIT_CODE=$?

exit $EXIT_CODE
