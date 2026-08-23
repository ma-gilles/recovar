#!/usr/bin/env bash
#SBATCH --job-name=k1c04-preprocess
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:45:00

set -euo pipefail

: "${K1_PREPROCESS_ROOT:?set the disposable capture root}"
: "${K1_PREPROCESS_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${K1_PREPROCESS_ROOT}
RUNTIME=${K1_PREPROCESS_RUNTIME}
PART_ID=${K1_PREPROCESS_PART_ID:-80015}
STACK_ONE_BASED=${K1_PREPROCESS_STACK_ONE_BASED:-82010}
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_exact_state_build_20260821T1530ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
CONTROL=${CASE}/relion_ref
CONTINUE=${CONTROL}/run_it001_optimiser.star
BINARY=${BUILD_ROOT}/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case04_native_preprocess_target.sbatch

for value_name in PART_ID STACK_ONE_BASED; do
    value=${!value_name}
    if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = 46968272c586f5780a382034df72d894d65169bfc1d36fafc034a99e11ddd4aa
test "$(sha256sum "${CONTINUE}" | awk '{print $1}')" = 881de25817c531ecc6df6879171d942152e27f8cfebfaf3db29c2052b0463b91
test -z "$(find "${ROOT}/capture" "${ROOT}/relion" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RELION_P1_PREPROCESS_CAPTURE_DIR=${ROOT}/capture
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=2
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=1073741824
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=${PART_ID}
export RELION_PREITER_PARTICLE_ORDER_SEED=1705
export RELION_PREITER_PARTICLE_ORDER_ITER=2
export RELION_SAMPLING_PERTURBATION_OVERRIDE=0.18417978286743164
export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=2

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${BINARY}" "${CONTINUE}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --continue "${CONTINUE}" --o "${ROOT}/relion/run"
    --iter 2 --auto_iter_max 2 --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${CASE}/data"
    "${command[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

"${PYTHON}" -m scripts.validate_relion_preprocess_capture \
    "${ROOT}/capture" \
    --expected-particles 1 \
    --expected-part-ids "${PART_ID}" \
    --expected-stack-indices "${STACK_ONE_BASED}" \
    --expected-iteration 2 \
    --output "${ROOT}/analysis/PREPROCESS_CAPTURE_VALIDATION.json"

find "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"iteration":2,"part_id":%d,"stack_index":%d,"capture":"seven_stage_preprocess"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${PART_ID}" "${STACK_ONE_BASED}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
