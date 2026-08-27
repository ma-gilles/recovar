#!/usr/bin/env bash
#SBATCH --job-name=vdam-block-replay
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-block-replay-%j.out

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?pin the committed replay implementation}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
NVCC_BIN=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_block_replay_gate_${SLURM_JOB_ID}}
CUDA_BINARY=${OUTPUT_ROOT}/libcuda_backproject.so

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
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export TMPDIR=${OUTPUT_ROOT}/tmp
mkdir -p "${TMPDIR}"

make -B -C "${REPO_ROOT}/recovar/cuda" \
    PYTHON="${PIXI_PY}" \
    NVCC="${NVCC_BIN}" \
    CUDA_ARCH="-gencode arch=compute_90,code=sm_90" \
    LIB="${CUDA_BINARY}"

"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar
from recovar import cuda_backproject

repo = pathlib.Path(__import__("os").environ["PYTHONPATH"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert jax.default_backend() == "gpu"
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM captured-block replay GPU/CUDA-FFI gate passed", jax.devices())
PY

"${PIXI_PY}" -m pytest -q \
    "${REPO_ROOT}/tests/unit/test_vdam_block_start_replay.py" \
    "${REPO_ROOT}/tests/unit/test_build_vdam_worker_schedule.py" \
    "${REPO_ROOT}/tests/unit/initial_model/test_run_vdam_candidate_against_native_panel.py" \
    "${REPO_ROOT}/tests/unit/test_cuda_relion_translation.py::test_relion_vdam_fused_source_uses_native_separate_accumulator_storage" \
    "${REPO_ROOT}/tests/unit/test_cuda_relion_translation.py::test_relion_vdam_mstep_fused_projector_x_half_validates_projector_shape" \
    "${REPO_ROOT}/tests/unit/test_cuda_relion_translation.py::test_relion_vdam_captured_rotation_order_matches_reverse_replay"

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
sha256sum "${CUDA_BINARY}" | tee "${OUTPUT_ROOT}/cuda_sha256.txt"
printf '{"schema":"recovar.vdam_captured_block_start_gate.v1","job_id":"%s","repo_head":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" > "${OUTPUT_ROOT}/completion.json"
touch "${OUTPUT_ROOT}/COMPLETED"
