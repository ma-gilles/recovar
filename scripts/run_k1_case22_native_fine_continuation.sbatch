#!/usr/bin/env bash
#SBATCH --job-name=k1c22-fine-cont
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:20:00

set -euo pipefail
: "${K1_CASE22_FINE_CONT_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_FINE_CONT_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the capture binary}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the capture source diff}"
: "${EXPECTED_CONTINUE_SHA256:?pin the incoming optimiser boundary}"

ROOT=${K1_CASE22_FINE_CONT_ROOT}
RUNTIME=${K1_CASE22_FINE_CONT_RUNTIME}
TARGET_STACK_ONE_BASED=${K1_CASE22_TARGET_STACK_ONE_BASED:-262}
TARGET_ITERATION=${K1_CASE22_TARGET_ITERATION:-3}
if (( TARGET_STACK_ONE_BASED < 1 || TARGET_ITERATION != 3 )); then
    echo "this focused continuation requires a positive stack and physical iteration 3" >&2
    exit 2
fi

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case22_native_fine_continuation.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_uuidfix_565f4363_20260717_004300/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/relion_ref
CONTINUE=${K1_CASE22_NATIVE_CONTINUE:-${CONTROL}/run_it002_optimiser.star}
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${CONTINUE}" | awk '{print $1}')" = "${EXPECTED_CONTINUE_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" diff | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${ROOT}/provenance" "${ROOT}/analysis"

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

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_BPRE_CAPTURE_DIR=${CAPTURE}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=1
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${TARGET_STACK_ONE_BASED}
export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --continue "${CONTINUE}" --o "${RELION_OUT}/run"
    --auto_iter_max "${TARGET_ITERATION}" --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${CONTINUE}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start ))

fine_score=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin')
factor=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin')
test "$(printf '%s\n' "${fine_score}" | wc -l)" -eq 1
test "$(printf '%s\n' "${factor}" | wc -l)" -eq 1
test -s "${RELION_OUT}/run_it003_data.star"
"${PYTHON}" - <<'PY' "${fine_score}" "${factor}" "${TARGET_STACK_ONE_BASED}"
import sys

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import load_fine_score_capture

fine = load_fine_score_capture(sys.argv[1])
factor = load_factor_capture(sys.argv[2])
expected_stack = int(sys.argv[3])
assert fine.stack_index == factor.stack_index == expected_stack
assert fine.header[4] == factor.header[9] == 3
assert factor.geometry_only
PY
! rg -qi 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${RELION_OUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":3,"stack_index_one_based":%d,"continued_from":"%s","geometry_only":true}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_STACK_ONE_BASED}" "${CONTINUE}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${CAPTURE}" "${RELION_OUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
