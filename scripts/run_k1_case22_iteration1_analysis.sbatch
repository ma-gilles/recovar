#!/usr/bin/env bash
#SBATCH --job-name=k1c22-it1-analyze
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/analyze-recovery-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/analyze-recovery-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:05:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it1_first_boundary_20260810T0945ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default/bin/python
PIXI_ENV=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case22_iteration1_analysis.sbatch
SELECTION=${ROOT}/provenance/SELECTION_IT1_CS056.json

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${SELECTION}" | awk '{print $1}')" = 8d5a32dea7a2cde8d0b43f90b8955a533773ced5e38c74a62c1bb4ca73817939
test "$(sha256sum "${REPO}/scripts/analyze_k1_bpref_factor_boundary.py" | awk '{print $1}')" = c57a0aa241e1c654cf10e695ff40b75d4a0ecf5a720d437d087d7b591577cc34
test "$(find "${ROOT}/capture" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq 3
test "$(find "${ROOT}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq 3

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_bpref_factor_boundary \
    --factor-directory "${ROOT}/capture" \
    --recovar-capture-directory "${ROOT}/recovar/pass2" \
    --selection-json "${SELECTION}" \
    --capture-inertness-json "${ROOT}/analysis/RELION_IT1_FACTOR_CAPTURE_INERTNESS.json" \
    --output-json "${ROOT}/analysis/CASE22_IT1_BPREF_FACTOR_BOUNDARY.json" \
    > "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stderr"

"${PYTHON}" - "${ROOT}/analysis/CASE22_IT1_BPREF_FACTOR_BOUNDARY.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["physical_iteration"] == 1
assert report["current_size"] == 56
assert report["particle_count"] == 3
assert all(row["support_exact"] for row in report["particles"])
PY

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    sha256sum "${SELECTION}" "${ROOT}/analysis/CASE22_IT1_BPREF_FACTOR_BOUNDARY.json"
} > "${ROOT}/provenance/ANALYSIS_RECOVERY_${SLURM_JOB_ID}.txt"
touch "${ROOT}/ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
