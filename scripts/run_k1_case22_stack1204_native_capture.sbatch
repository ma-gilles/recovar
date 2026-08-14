#!/usr/bin/env bash
#SBATCH --job-name=k1c22-s1204-rel
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
: "${K1_CASE22_STACK1204_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_STACK1204_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the capture binary}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the inert capture source diff}"

ROOT=${K1_CASE22_STACK1204_ROOT}
RUNTIME=${K1_CASE22_STACK1204_RUNTIME}
TARGET_STACK_ONE_BASED=${K1_CASE22_TARGET_STACK_ONE_BASED:-1204}
TARGET_STACKS_ONE_BASED=${K1_CASE22_TARGET_STACKS_ONE_BASED:-${TARGET_STACK_ONE_BASED}}
TARGET_STACKS_ONE_BASED=${TARGET_STACKS_ONE_BASED//:/,}
TARGET_ITERATION=${K1_CASE22_TARGET_ITERATION:-2}
MAX_ITER=${K1_CASE22_MAX_ITER:-${TARGET_ITERATION}}
GEOMETRY_ONLY=${K1_CASE22_BPRE_GEOMETRY_ONLY:-0}
CAPTURE_FINE_OPERAND=${K1_CASE22_CAPTURE_FINE_OPERAND:-0}
FINE_OPERAND_ROTATION_LOCAL=${K1_CASE22_FINE_OPERAND_ROTATION_LOCAL:-0}
FINE_OPERAND_TRANSLATION=${K1_CASE22_FINE_OPERAND_TRANSLATION:-0}
if (( TARGET_STACK_ONE_BASED < 1 || TARGET_ITERATION < 1 || MAX_ITER < TARGET_ITERATION )); then
    echo "target stack/iteration must be positive and max_iter must reach the target" >&2
    exit 2
fi
IFS=',' read -r -a target_stack_values <<< "${TARGET_STACKS_ONE_BASED}"
for target_stack in "${target_stack_values[@]}"; do
    if [[ ! "${target_stack}" =~ ^[0-9]+$ ]] || (( target_stack < 1 )); then
        echo "target stacks must be comma-separated positive integers" >&2
        exit 2
    fi
done
TARGET_COUNT=${#target_stack_values[@]}
if (( TARGET_COUNT < 1 )) || [[ "$(printf '%s\n' "${target_stack_values[@]}" | sort -u | wc -l)" -ne "${TARGET_COUNT}" ]]; then
    echo "target stacks must be nonempty and unique" >&2
    exit 2
fi
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case22_stack1204_native_capture.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
CONTROL=${K1_CASE22_NATIVE_CONTROL:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion}
CAPTURE=${ROOT}/native/capture
RELION_OUT=${ROOT}/native/relion

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" diff | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
for directory in "${CAPTURE}" "${RELION_OUT}"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${ROOT}/native/provenance" "${ROOT}/native/analysis"

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
nvidia-smi -q > "${ROOT}/native/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_BPRE_CAPTURE_DIR=${CAPTURE}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=${TARGET_COUNT}
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=${TARGET_COUNT}
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${TARGET_STACKS_ONE_BASED}
if [[ "${GEOMETRY_ONLY}" = 1 ]]; then
    export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1
fi
if [[ "${CAPTURE_FINE_OPERAND}" = 1 ]]; then
    export RELION_FINE_OPERAND_CAPTURE=1
    export RELION_FINE_OPERAND_CAPTURE_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL}
    export RELION_FINE_OPERAND_CAPTURE_TRANSLATIONS=${FINE_OPERAND_TRANSLATION}
    export RELION_FINE_OPERAND_CAPTURE_EXPECTED_CANDIDATES=1
    export RELION_FINE_OPERAND_CAPTURE_MAX_BYTES=1000000
fi

command=(
    mpirun -n 3 "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max "${MAX_ITER}"
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${command[@]}" > "${ROOT}/native/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/native/provenance/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" > "${ROOT}/native/provenance/walltime_${SLURM_JOB_ID}.json"

test "$(find "${CAPTURE}" -maxdepth 1 -name '*.fine-score-v1.bin' | wc -l)" -eq "${TARGET_COUNT}"
test "$(find "${CAPTURE}" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq "${TARGET_COUNT}"
if [[ "${CAPTURE_FINE_OPERAND}" = 1 ]]; then
    test "$(find "${CAPTURE}" -maxdepth 1 -name '*.fine-operand-v1.bin' | wc -l)" -eq "${TARGET_COUNT}"
fi
printf -v target_iteration_label '%03d' "${TARGET_ITERATION}"
test -s "${RELION_OUT}/run_it${target_iteration_label}_data.star"

"${PYTHON}" - <<'PY' "${CAPTURE}" "${TARGET_STACKS_ONE_BASED}" "${TARGET_ITERATION}"
from pathlib import Path
import sys

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import load_fine_score_capture

capture = Path(sys.argv[1])
expected_stacks = {int(value) for value in sys.argv[2].split(",")}
expected_iteration = int(sys.argv[3])
fine = [load_fine_score_capture(path) for path in sorted(capture.glob("*.fine-score-v1.bin"))]
factor = [load_factor_capture(path) for path in sorted(capture.glob("*.bpre-v2.bin"))]
assert {item.stack_index for item in fine} == expected_stacks
assert {item.stack_index for item in factor} == expected_stacks
assert {item.header[4] for item in fine} == {expected_iteration}
assert {item.header[9] for item in factor} == {expected_iteration}
PY

cd "${AUDIT_REPO}"
set +e
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" \
    --iterations $(seq 1 "${TARGET_ITERATION}") \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/native/analysis/RELION_CAPTURE_INERTNESS_FULL.json" \
    --output-markdown "${ROOT}/native/analysis/RELION_CAPTURE_INERTNESS_FULL.md"
audit_status=$?
set -e
printf '{"audit_exit_status":%d,"capture_particle_count":%d,"target_stacks_one_based":"%s"}\n' \
    "${audit_status}" "${TARGET_COUNT}" "${TARGET_STACKS_ONE_BASED}" \
    > "${ROOT}/native/analysis/CAPTURE_AUDIT_STATUS.json"

find "${CAPTURE}" "${RELION_OUT}" "${ROOT}/native/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${ROOT}/native/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
if (( audit_status == 0 )); then
    touch "${ROOT}/native/SUCCESS_${SLURM_JOB_ID}"
else
    touch "${ROOT}/native/CAPTURE_COMPLETE_AUDIT_REJECTED_${SLURM_JOB_ID}"
fi
