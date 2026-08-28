#!/usr/bin/env bash
#SBATCH --job-name=k1-final-recon-audit
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RECOVAR_DUMP:?RECOVAR_DUMP is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${NATIVE_JOB_ID:?NATIVE_JOB_ID is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO_ROOT}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/analyze_k1_joined_final_reconstruction_stages.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_reconstruction_stage_boundary.py
NATIVE_STAGES=${NATIVE_RUN_ROOT}/reconstruct
NATIVE_PREFIX=${NATIVE_RUN_ROOT}/bpref/mstep_it016_rank1_half0_c0_pre_reconstruct
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance
REPORT=${ANALYSIS}/K1_JOINED_FINAL_RECONSTRUCTION_STAGES.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_RUN_ROOT}/SUCCESS_${NATIVE_JOB_ID}"
test -s "${RECOVAR_DUMP}"
for suffix in bpref_data bpref_weight tau2; do
    test -s "${NATIVE_PREFIX}_${suffix}.bin"
done
for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${NATIVE_STAGES}" -maxdepth 1 -type f -name "reconstruct_rank01_pid*_call0000_${stage}.bin" | wc -l)" -eq 1
done
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -z "$(find "${RUN_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE ! -name 'slurm-*.out' -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${ANALYSIS}" "${PROVENANCE}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_ROOT=/usr/local/cuda-12.8
export PATH=${CUDA_ROOT}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible H100}"

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
print(jax.devices())
PY

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
git status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${ANALYZER}" "${LAUNCHER}" "${RECOVAR_DUMP}" \
    "${NATIVE_PREFIX}_bpref_data.bin" "${NATIVE_PREFIX}_bpref_weight.bin" \
    "${NATIVE_PREFIX}_tau2.bin" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"
find "${NATIVE_STAGES}" -maxdepth 1 -type f -name 'reconstruct_rank01_pid*_call0000_*' \
    -print0 | sort -z | xargs -0 sha256sum >> "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

"${PYTHON}" -m scripts.analyze_k1_reconstruction_stage_boundary \
    --recovar-accumulator "${RECOVAR_DUMP}" \
    --native-stage-dir "${NATIVE_STAGES}" \
    --native-call-index 0 \
    --native-tau-provenance live-pre-reconstruction \
    --joined --halves 1 \
    --relion-bpref-prefix "${NATIVE_PREFIX}" \
    --output-json "${REPORT}" \
    > "${ANALYSIS}/K1_JOINED_FINAL_RECONSTRUCTION_STAGES.stdout.json"

"${PYTHON}" - "${REPORT}" "${ANALYSIS}/summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
stages = report["halves"]["1"]["stages"]
ordered = (
    "bpref_numerator_decentered_sign_aligned",
    "fweight_decentered",
    "wiener_denominator_after_floor",
    "fconv_divided_sign_aligned",
    "volume_before_gridding_sign_aligned",
    "volume_after_gridding_sign_aligned",
)
summary = {name: stages[name]["relative_l2"] for name in ordered}
assert all(math.isfinite(float(value)) for value in summary.values())
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

sha256sum "${REPORT}" "${ANALYSIS}/summary.json" \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
