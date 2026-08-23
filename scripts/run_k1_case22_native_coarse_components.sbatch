#!/usr/bin/env bash
#SBATCH --job-name=k1c22-components
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:20:00

set -euo pipefail
: "${K1_CASE22_COMPONENT_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_COMPONENT_RUNTIME:?set the disposable runtime root}"
: "${K1_CASE22_COMPONENT_SOURCE:?set the pinned RELION source checkout}"
: "${K1_CASE22_COMPONENT_BUILD:?set the pinned RELION build directory}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_HEAD:?pin the diagnostic RELION commit}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the capture executable}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the capture-only source diff}"

ROOT=${K1_CASE22_COMPONENT_ROOT}
RUNTIME=${K1_CASE22_COMPONENT_RUNTIME}
SOURCE=${K1_CASE22_COMPONENT_SOURCE}
BUILD=${K1_CASE22_COMPONENT_BUILD}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_case22_native_coarse_components.sbatch
RELION_BINARY=${BUILD}/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion
TARGET_ITERATION=${K1_CASE22_COMPONENT_ITERATION:-2}
TARGET_PART_ID=${K1_CASE22_COMPONENT_PART_ID:-229}
TARGET_STACK_ONE_BASED=${K1_CASE22_COMPONENT_STACK_ONE_BASED:-1204}
ROTATION_KEYS=${K1_CASE22_COMPONENT_ROTATION_KEYS:-16709,36064}
EXPECTED_FOLLOWERS=${K1_CASE22_COMPONENT_EXPECTED_FOLLOWERS:-1}
NOISE_DUMP=${K1_CASE22_COMPONENT_NOISE_DUMP:-0}

for numeric_value in "${TARGET_ITERATION}" "${TARGET_PART_ID}" \
    "${TARGET_STACK_ONE_BASED}" "${EXPECTED_FOLLOWERS}"; do
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "component capture selectors must be non-negative integers" >&2
        exit 2
    fi
done
if (( TARGET_ITERATION < 1 || TARGET_STACK_ONE_BASED < 1 || EXPECTED_FOLLOWERS < 1 )); then
    echo "iteration, stack index, and expected followers must be positive" >&2
    exit 2
fi
case "${NOISE_DUMP}" in
    0|1) ;;
    *) echo "K1_CASE22_COMPONENT_NOISE_DUMP must be 0 or 1" >&2; exit 2 ;;
esac

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6
test -z "$(find "${CAPTURE}" -mindepth 1 -print -quit)"
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${ROOT}/provenance"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible GPU}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_P1_CAPTURE_DIR=${CAPTURE}
export RELION_P1_CAPTURE_SCHEMA=2
export RELION_P1_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=${EXPECTED_FOLLOWERS}
export RELION_P1_CAPTURE_MAX_BYTES=100000000
export RELION_P1_CAPTURE_PART_IDS=${TARGET_PART_ID}
export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=${ROTATION_KEYS}
if [[ "${NOISE_DUMP}" = 1 ]]; then
    export RELION_DUMP_SIGMA2_NOISE_DIR=${ROOT}/noise
    mkdir -p "${RELION_DUMP_SIGMA2_NOISE_DIR}"
fi

command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max "${TARGET_ITERATION}"
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"

components=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK_ONE_BASED}.p1-v2.bin
operands=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK_ONE_BASED}.p1-op-v2.bin
printf -v iteration_label '%03d' "${TARGET_ITERATION}"
test -s "${components}"
test -s "${operands}"
test -s "${RELION_OUT}/run_it${iteration_label}_data.star"
if [[ "${NOISE_DUMP}" = 1 ]]; then
    test -s "${RELION_DUMP_SIGMA2_NOISE_DIR}/sigma2_noise_raw.tsv"
    test -s "${RELION_DUMP_SIGMA2_NOISE_DIR}/sigma2_noise_components.tsv"
    test -s "${RELION_DUMP_SIGMA2_NOISE_DIR}/sigma2_noise_support.tsv"
fi
sha256sum "${LAUNCHER}" "${RELION_BINARY}" \
    "${components}" "${operands}" \
    > "${ROOT}/provenance/static_and_capture_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
