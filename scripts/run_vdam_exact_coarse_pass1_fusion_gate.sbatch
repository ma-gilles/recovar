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

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_exact_coarse_pass1_fusion_gate_${SLURM_JOB_ID}}
NVCC_BIN=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
CUDA_ARCH_FLAGS=${VDAM_CUDA_ARCH_FLAGS:--gencode arch=compute_90,code=sm_90}
RECOVAR_CUDA_LIB=${OUTPUT_ROOT}/libcuda_backproject_h100.so

test -x "${PIXI_PY}"
test -x "${NVCC_BIN}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
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
export RECOVAR_CUDA_LIB
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export TMPDIR=${OUTPUT_ROOT}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

"${NVCC_BIN}" --version > "${OUTPUT_ROOT}/nvcc_version.txt"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${OUTPUT_ROOT}/cuda_arch_flags.txt"
make -B -C "${REPO_ROOT}/recovar/cuda" \
    PYTHON="${PIXI_PY}" \
    NVCC="${NVCC_BIN}" \
    CUDA_ARCH="${CUDA_ARCH_FLAGS}" \
    LIB="${RECOVAR_CUDA_LIB}"

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
sha256sum "${OUTPUT_ROOT}/nvcc_version.txt" "${OUTPUT_ROOT}/cuda_arch_flags.txt" \
    > "${OUTPUT_ROOT}/toolchain_sha256.txt"
CUDA_SHA256=$(sha256sum "${RECOVAR_CUDA_LIB}" | awk '{print $1}')
printf '{"schema":"recovar.vdam_exact_coarse_pass1_fusion_gate.v1","job_id":"%s","repo_head":"%s","cuda_sha256":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${CUDA_SHA256}" \
    > "${OUTPUT_ROOT}/completion.json"
touch "${OUTPUT_ROOT}/COMPLETED"
