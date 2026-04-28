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

import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
assert str(recovar_file).startswith(str(repo) + "/"), recovar_file
assert ".pixi/envs/default/" in str(jax_file), jax_file
print(f"provenance_ok recovar={recovar_file} jax={jax_file}")
PY

tests=(
  tests/unit/test_em_fast_guardrail.py
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_pass1_matches_dense_primitives_for_modes
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_mstep_matches_dense_primitives_and_adjoint
  tests/unit/test_dense_big_jit.py::test_dense_big_jit_masks_padded_image_rows
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_matches_dense_engine_on_single_image_local_grid
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_windowed_with_pre_shifts_matches_dense_engine
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_default_fused_path_matches_materialized_split
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_forced_native_half_preprocess_matches_legacy
  tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_big_jit_bucket_matches_legacy
)

exec "$PYTHON_BIN" -m pytest "${tests[@]}" -q "$@"
