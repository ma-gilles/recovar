#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

backend="${EM_FAST_GUARD_BACKEND:-cpu}"
if [[ "$backend" == "gpu" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "EM_FAST_GUARD_BACKEND=gpu requested, but nvidia-smi is not available" >&2
    exit 1
  fi
  nvidia-smi
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
else
  export JAX_PLATFORMS=cpu
  unset JAX_PLATFORM_NAME
fi

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.pixi/envs/default/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

"$PYTHON_BIN" - <<'PY'
import pathlib
import importlib
import sys

import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
pixi_env = (repo / ".pixi" / "envs" / "default").resolve()
assert str(recovar_file).startswith(str(repo) + "/"), recovar_file
assert str(jax_file).startswith(str(pixi_env) + "/"), (jax_file, pixi_env)
for helper in (
    "relion_replay", "relion_normalization", "projector_preparation",
    "score_outputs", "local_batch_planning", "k_class_results", "scoring_policy", "helpers.resolution", "helpers.bpref_diagnostics",
    "helpers.significant_samples", "helpers.coarse_score_diagnostics", "helpers.sparse_bucket_arrays", "helpers.compact_candidates", "helpers.pass2_diagnostics", "helpers.norm_scale_diagnostics", "helpers.relion_ctf", "helpers.scale_groups", "helpers.normalization_inputs",
    "helpers.vdam_replay", "fixed_capacity_local", "local_layout", "local_projection_cache", "local_timing",
):
    importlib.import_module(f"recovar.em.dense_single_volume.{helper}")
execution_modules = (
    "iteration_loop", "half_scoring", "k_class", "em_engine", "local_em_engine", "local_big_jit",
    "helpers.significance", "helpers.sparse_pass2_bucketed",
)
loaded = [name for name in execution_modules
          if f"recovar.em.dense_single_volume.{name}" in sys.modules]
assert not loaded, f"EM helper imports must not load execution modules: {loaded}"
print(f"provenance_ok recovar={recovar_file} jax={jax_file}")
print("helper_import_boundary_ok")
PY

tests=(
  tests/unit/test_em_fast_guardrail.py
  tests/unit/test_relion_replay_state.py
  tests/unit/test_healpix_order_oracle.py
  tests/unit/test_resolution_scheduling.py
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_pass1_matches_dense_primitives_for_modes
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_mstep_matches_dense_primitives_and_adjoint
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_masks_padded_image_rows
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_matches_dense_engine_on_single_image_local_grid
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_windowed_with_pre_shifts_matches_dense_engine
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_default_path_matches_debug_split_path
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_big_jit_bucket_matches_debug_split
)

exec "$PYTHON_BIN" -m pytest "${tests[@]}" -q "$@"
