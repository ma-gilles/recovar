#!/usr/bin/env bash
#SBATCH --job-name=k1-native-recon
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${FT_Y:?FT_Y is required}"
: "${FT_CTF:?FT_CTF is required}"
: "${TAU2_PRE_RECONSTRUCTION:?TAU2_PRE_RECONSTRUCTION is required}"
: "${REFINEMENT_RESULTS:?REFINEMENT_RESULTS is required}"
: "${RECOVAR_MAP:?RECOVAR_MAP is required}"
: "${RELION_MAP:?RELION_MAP is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/audit_k1_native_bpref_reconstruct.sbatch
ANALYZER=${REPO}/scripts/audit_k1_native_bpref_reconstruct.py
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/analysis/native_bpref_reconstruct}
ITERATION_INDEX=${ITERATION_INDEX:-0}
PARTICLE_DIAMETER_ANG=${PARTICLE_DIAMETER_ANG:-200}
VOXEL_SIZE=${VOXEL_SIZE:-2.125}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
for artifact in "${FT_Y}" "${FT_CTF}" "${TAU2_PRE_RECONSTRUCTION}" \
  "${REFINEMENT_RESULTS}" "${RECOVAR_MAP}" "${RELION_MAP}"; do
  test -s "${artifact}"
done
test ! -e "${OUTPUT_DIR}/AUDIT.json"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache" \
  "${RUNTIME_ROOT}/pycache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0 RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
  > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
"${PYTHON}" -c "import pathlib,recovar,jax; repo=pathlib.Path('${REPO}').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; from recovar.relion_bind import _relion_bind_core"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${FT_Y}" "${FT_CTF}" \
  "${TAU2_PRE_RECONSTRUCTION}" \
  "${REFINEMENT_RESULTS}" "${RECOVAR_MAP}" "${RELION_MAP}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git -C "${REPO}" status --short --branch \
  > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"

command=("${PYTHON}" -m scripts.audit_k1_native_bpref_reconstruct
  --ft-y "${FT_Y}" --ft-ctf "${FT_CTF}"
  --tau2-pre-reconstruction "${TAU2_PRE_RECONSTRUCTION}"
  --refinement-results "${REFINEMENT_RESULTS}"
  --recovar-map "${RECOVAR_MAP}" --relion-map "${RELION_MAP}"
  --output-dir "${OUTPUT_DIR}" --iteration-index "${ITERATION_INDEX}"
  --particle-diameter-ang "${PARTICLE_DIAMETER_ANG}"
  --voxel-size "${VOXEL_SIZE}")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"

cd "${REPO}"
start_epoch=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/analysis/run_${SLURM_JOB_ID}.stdout" \
  2> "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
wall_s=$(( $(date +%s) - start_epoch ))
test -s "${OUTPUT_DIR}/AUDIT.json"
sha256sum "${OUTPUT_DIR}/AUDIT.json" "${OUTPUT_DIR}/shellwise_fsc.npz" \
  > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "${wall_s}" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
