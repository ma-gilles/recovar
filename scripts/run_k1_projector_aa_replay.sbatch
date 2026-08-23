#!/usr/bin/env bash
#SBATCH --job-name=k1-projector-aa
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail
: "${RUN_ROOT:?set RUN_ROOT}"
: "${RUNTIME_ROOT:?set RUNTIME_ROOT}"
: "${EXPECTED_LAUNCHER_SHA256:?set EXPECTED_LAUNCHER_SHA256}"
: "${EXPECTED_ANALYZER_SHA256:?set EXPECTED_ANALYZER_SHA256}"
: "${EXPECTED_REPO_HEAD:?set EXPECTED_REPO_HEAD}"
: "${EXPECTED_REPO_DIFF_SHA256:?set EXPECTED_REPO_DIFF_SHA256}"
: "${EXPECTED_CUDA_SHA256:?set EXPECTED_CUDA_SHA256}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_projector_aa_replay.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_projector_aa_replay.py
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
RECOVAR_CANDIDATES=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_scale_aa_features_orig1096_it2_20260812T1945ET/pass2/scale_aa_chunked_orig001096_half1_cs060.npz
RECOVAR_PROJECTOR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_projector_orig1096_it2_20260812T1955ET/projector/iter001_half0_relion_projector_half.npz
NATIVE_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_scale_aa_candidates_it2_20260812T1937ET/capture
NATIVE_COMPONENTS=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_terms_native_retry_it2_20260812T1820ET/noise/sigma2_noise_components.tsv

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -s "${RECOVAR_CANDIDATES}" -a -s "${RECOVAR_PROJECTOR}" -a -s "${NATIVE_COMPONENTS}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
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
export RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP=1

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

OUTPUT=${RUN_ROOT}/analysis/projector_aa_replay.json
start=$(date +%s)
"${PYTHON}" -m scripts.analyze_k1_projector_aa_replay \
    --recovar-candidates "${RECOVAR_CANDIDATES}" \
    --recovar-projector "${RECOVAR_PROJECTOR}" \
    --native-directory "${NATIVE_DIR}" \
    --native-components "${NATIVE_COMPONENTS}" \
    --output-json "${OUTPUT}"
printf '{"job_id":"%s","wall_s":%d,"chunk_size":2048}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" \
    > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${OUTPUT}" \
    > "${RUN_ROOT}/provenance/static_and_output_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
