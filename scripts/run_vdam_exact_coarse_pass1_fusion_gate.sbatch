#!/usr/bin/env bash
#SBATCH --job-name=vdam-coarse-pass1-fuse
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-coarse-pass1-fuse-%j.out

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?pin the committed exact-pass1-fusion implementation}"
: "${RECOVAR_CUDA_LIB:?pin a qualified H100 RECOVAR CUDA binary}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified CUDA binary digest}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_exact_coarse_pass1_fusion_gate_${SLURM_JOB_ID}}

test -x "${PIXI_PY}"
test -f "${RECOVAR_CUDA_LIB}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing to reuse gate output: ${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}" /scratch/gpfs/GILLES/mg6942/slurmo
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"

unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV RECOVAR_DISABLE_CUDA
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export TMPDIR=${OUTPUT_ROOT}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar
from recovar import cuda_backproject

repo = pathlib.Path(__import__("os").environ["PYTHONPATH"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM exact coarse pass-1 fusion GPU/CUDA-FFI gate passed", jax.devices())
PY

"${PIXI_PY}" -m pytest -vv --run-gpu \
    "${REPO_ROOT}/tests/unit/test_cuda_relion_fine_diff2.py::test_fused_relion_coarse_pass1_matches_sequential_bookkeeping_bitwise"

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
printf '%s\n' "${EXPECTED_REPO_HEAD}" > "${OUTPUT_ROOT}/repo_head.txt"
sha256sum "${RECOVAR_CUDA_LIB}" > "${OUTPUT_ROOT}/cuda_sha256.txt"
printf '{"schema":"recovar.vdam_exact_coarse_pass1_fusion_gate.v1","job_id":"%s","repo_head":"%s","cuda_sha256":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${EXPECTED_CUDA_SHA256}" \
    > "${OUTPUT_ROOT}/completion.json"
touch "${OUTPUT_ROOT}/COMPLETED"
