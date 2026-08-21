#!/usr/bin/env bash
#SBATCH --job-name=k1-native-ref-score
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT}"
: "${RUNTIME_ROOT:?set RUNTIME_ROOT}"
: "${EXPECTED_REPO_HEAD:?set EXPECTED_REPO_HEAD}"
: "${EXPECTED_REPO_DIFF_SHA256:?set EXPECTED_REPO_DIFF_SHA256}"
: "${EXPECTED_CUDA_LIB_SHA256:?set EXPECTED_CUDA_LIB_SHA256}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${K1_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so}
COMPONENTS=${K1_NATIVE_REF_COMPONENTS:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_two_projection_20260811T1339ET/capture/part24_stack12.p1-v2.bin}
OPERANDS=${K1_NATIVE_REF_OPERANDS:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_two_projection_20260811T1339ET/capture/part24_stack12.p1-op-v2.bin}
RECOVAR=${K1_NATIVE_REF_RECOVAR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_expf_ffi_capture_20260811T1640ET/score/significance_orig000011_it002_cs056.npz}
NATIVE_PARENT=${K1_NATIVE_REF_NATIVE_PARENT-}
RECOVAR_PARENT=${K1_NATIVE_REF_RECOVAR_PARENT-}
HIGH_SHELL_CUTOFF=${K1_NATIVE_REF_HIGH_SHELL_CUTOFF-}
HIGH_SHELL_RETAINED_FRACTION=${K1_NATIVE_REF_HIGH_SHELL_RETAINED_FRACTION-}
UNIFORM_PIXEL_WEIGHT_FACTOR=${K1_NATIVE_REF_UNIFORM_PIXEL_WEIGHT_FACTOR-}
NOISE_REPORT_JSON=${K1_NATIVE_REF_NOISE_REPORT_JSON-}
NOISE_SHELL_MAX=${K1_NATIVE_REF_NOISE_SHELL_MAX-}
if [[ -n "${NATIVE_PARENT}" || -n "${RECOVAR_PARENT}" ]]; then
    test -n "${NATIVE_PARENT}" -a -n "${RECOVAR_PARENT}"
fi
if [[ -n "${HIGH_SHELL_CUTOFF}" || -n "${HIGH_SHELL_RETAINED_FRACTION}" ]]; then
    test -n "${HIGH_SHELL_CUTOFF}" -a -n "${HIGH_SHELL_RETAINED_FRACTION}"
fi

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test -s "${COMPONENTS}" -a -s "${OPERANDS}" -a -s "${RECOVAR}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}" \
    "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance"

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
export RECOVAR_CUDA_LIB=${CUDA_LIB}

COMMAND=(
    "${PYTHON}" -m scripts.analyze_k1_native_reference_score_counterfactual
    --components "${COMPONENTS}"
    --operands "${OPERANDS}"
    --recovar "${RECOVAR}"
    --output-json "${RUN_ROOT}/analysis/native_reference_score_counterfactual.json"
)
if [[ -n "${NATIVE_PARENT}" ]]; then
    COMMAND+=(--native-parent "${NATIVE_PARENT}" --recovar-parent "${RECOVAR_PARENT}")
fi
if [[ -n "${HIGH_SHELL_CUTOFF}" ]]; then
    COMMAND+=(
        --high-shell-cutoff "${HIGH_SHELL_CUTOFF}"
        --high-shell-retained-fraction "${HIGH_SHELL_RETAINED_FRACTION}"
    )
fi
if [[ -n "${UNIFORM_PIXEL_WEIGHT_FACTOR}" ]]; then
    COMMAND+=(--uniform-pixel-weight-factor "${UNIFORM_PIXEL_WEIGHT_FACTOR}")
fi
if [[ -n "${NOISE_REPORT_JSON}" ]]; then
    COMMAND+=(--noise-report-json "${NOISE_REPORT_JSON}")
    if [[ -n "${NOISE_SHELL_MAX}" ]]; then
        COMMAND+=(--noise-shell-max "${NOISE_SHELL_MAX}")
    fi
fi
cd "${REPO}"
"${COMMAND[@]}"
sha256sum "${RUN_ROOT}/analysis/native_reference_score_counterfactual.json" \
    > "${RUN_ROOT}/provenance/artifacts_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
