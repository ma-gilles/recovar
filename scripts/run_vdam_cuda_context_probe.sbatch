#!/usr/bin/env bash
#SBATCH --job-name=vdam-cuda-context
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cuda-context-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${PROBE_EXECUTABLE:?set the immutable native probe executable}"
: "${PROBE_LIBRARY:?set the immutable probe shared object}"
: "${EXPECTED_PROBE_EXECUTABLE_SHA256:?pin the native probe executable}"
: "${EXPECTED_PROBE_LIBRARY_SHA256:?pin the probe shared object}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
PROVENANCE=${OUTPUT_ROOT}/provenance
ANALYSIS=${OUTPUT_ROOT}/analysis

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -x "${PROBE_EXECUTABLE}"
test -s "${PROBE_LIBRARY}"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${PROBE_EXECUTABLE}" | awk '{print $1}')" = "${EXPECTED_PROBE_EXECUTABLE_SHA256}"
test "$(sha256sum "${PROBE_LIBRARY}" | awk '{print $1}')" = "${EXPECTED_PROBE_LIBRARY_SHA256}"

mkdir -p "${PROVENANCE}" "${ANALYSIS}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
export PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
export RATTLER_CACHE_DIR=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
if [[ -n "${TARGET_GPU_UUID}" && "${gpu_uuid}" != "${TARGET_GPU_UUID}" ]]; then
  echo "wrong physical GPU: expected ${TARGET_GPU_UUID}, got ${gpu_uuid}" >&2
  exit 75
fi

"${PROBE_EXECUTABLE}" > "${ANALYSIS}/native_runtime_context.json"

REPO_ROOT="${REPO_ROOT}" PROBE_LIBRARY="${PROBE_LIBRARY}" "${PIXI_PY}" - <<'PY' \
  > "${ANALYSIS}/jax_runtime_context.json"
import ctypes
import json
import os
import pathlib
import sys

repo = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo))
import recovar
import jax

assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
devices = jax.devices("gpu")
assert len(devices) == 1
library = ctypes.CDLL(os.environ["PROBE_LIBRARY"])
probe = library.recovar_cuda_runtime_context_probe_json
probe.restype = ctypes.c_char_p
report = json.loads(probe().decode("utf-8"))
report["jax_device"] = str(devices[0])
report["recovar_file"] = str(pathlib.Path(recovar.__file__).resolve())
report["jax_file"] = str(pathlib.Path(jax.__file__).resolve())
print(json.dumps(report, sort_keys=True))
PY

NATIVE_REPORT="${ANALYSIS}/native_runtime_context.json" \
JAX_REPORT="${ANALYSIS}/jax_runtime_context.json" \
GPU_UUID="${gpu_uuid}" JOB_ID="${SLURM_JOB_ID}" \
"${PIXI_PY}" - <<'PY' > "${ANALYSIS}/context_comparison.json"
import json
import os
from pathlib import Path

native = json.loads(Path(os.environ["NATIVE_REPORT"]).read_text())
jax = json.loads(Path(os.environ["JAX_REPORT"]).read_text())
fields = (
    "runtime_device_flags",
    "primary_context_flags",
    "primary_context_active",
    "stream_flags",
    "stream_priority",
    "least_stream_priority",
    "greatest_stream_priority",
)
matches = {field: native[field] == jax[field] for field in fields}
report = {
    "schema": "recovar.vdam_cuda_context_comparison.v1",
    "status": "complete",
    "job_id": os.environ["JOB_ID"],
    "gpu_uuid": os.environ["GPU_UUID"],
    "native": native,
    "jax": jax,
    "field_matches": matches,
    "all_observable_context_fields_match": all(matches.values()),
}
print(json.dumps(report, indent=2, sort_keys=True))
PY

sha256sum "${PROBE_EXECUTABLE}" "${PROBE_LIBRARY}" \
  "${ANALYSIS}/native_runtime_context.json" \
  "${ANALYSIS}/jax_runtime_context.json" \
  "${ANALYSIS}/context_comparison.json" > "${PROVENANCE}/sha256.txt"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/recovar_head.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/recovar_status.txt"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
