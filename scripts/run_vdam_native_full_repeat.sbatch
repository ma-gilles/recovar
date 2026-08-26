#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-full
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-native-full-%j.out

set -euo pipefail

: "${REFERENCE_ROOT:?set the completed paired VDAM baseline root}"
: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${EXPECTED_RELION_SHA256:?pin the stock RELION executable}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
REFERENCE_COMMAND=${REFERENCE_ROOT}/relion/relion_command.json
REFERENCE_RELION=${REFERENCE_ROOT}/relion
DATA_DIR=${REFERENCE_ROOT}/data
RELION_DIR=${OUTPUT_ROOT}/relion
ANALYSIS_DIR=${OUTPUT_ROOT}/analysis
PROVENANCE_DIR=${OUTPUT_ROOT}/provenance
CAPTURE_DIR=${VDAM_NATIVE_CAPTURE_DIR:-}
CAPTURE_ITERATION=${VDAM_NATIVE_CAPTURE_ITERATION:-1}
CAPTURE_PART_ID=${VDAM_NATIVE_CAPTURE_PART_ID:-0}
CAPTURE_STACK_INDEX=${VDAM_NATIVE_CAPTURE_STACK_INDEX:-}
CAPTURE_PERTURBATION=${VDAM_NATIVE_CAPTURE_PERTURBATION:-}
CAPTURE_ONLY=${VDAM_NATIVE_CAPTURE_ONLY:-0}
RELION_SKIP_MODULE=${VDAM_RELION_SKIP_MODULE:-0}
TARGET_GPU_UUID=${VDAM_TARGET_GPU_UUID:-}

case "${CAPTURE_ONLY}" in 0|1) ;; *) echo "VDAM_NATIVE_CAPTURE_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "${RELION_SKIP_MODULE}" in 0|1) ;; *) echo "VDAM_RELION_SKIP_MODULE must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${CAPTURE_ONLY}" == 1 ]]; then
    test -n "${CAPTURE_DIR}"
fi

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
test -s "${REFERENCE_COMMAND}"
test -s "${REFERENCE_RELION}/run_it200_model.star"
test -s "${DATA_DIR}/particles.star"
test -x "${PIXI_PY}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
mkdir -p "${RELION_DIR}" "${ANALYSIS_DIR}" "${PROVENANCE_DIR}"
if [[ -n "${CAPTURE_DIR}" ]]; then
    mkdir -p "${CAPTURE_DIR}"
    test -z "$(find "${CAPTURE_DIR}" -mindepth 1 -print -quit)"
fi

mapfile -d '' -t RELION_COMMAND < <("${PIXI_PY}" - "${REFERENCE_COMMAND}" "${RELION_DIR}/run" <<'PY'
import json
import sys
from pathlib import Path

command = [str(value) for value in json.loads(Path(sys.argv[1]).read_text())["argv"]]
output_count = 0
for index, token in enumerate(command):
    if token == "--o":
        command[index + 1] = sys.argv[2]
        output_count += 1
if output_count != 1:
    raise SystemExit(f"expected exactly one RELION --o flag, found {output_count}")
if command[command.index("--iter") + 1] != "200":
    raise SystemExit("native full repeat requires the sealed 200-iteration command")
for token in command:
    sys.stdout.buffer.write(token.encode() + b"\0")
PY
)
RELION_BINARY=${RELION_COMMAND[0]}
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

if [[ -n "${CAPTURE_DIR}" ]]; then
    export RELION_ACC_DUMP_DIR=${CAPTURE_DIR}
    export RELION_ACC_DUMP_PART_ID=${CAPTURE_PART_ID}
    export RELION_ACC_DUMP_ITER=${CAPTURE_ITERATION}
    export RELION_ACC_DUMP_VERBOSE=1
    if [[ -n "${CAPTURE_STACK_INDEX}" ]]; then
        export RELION_ACC_DUMP_STACK_INDEX=${CAPTURE_STACK_INDEX}
    fi
    if [[ -n "${CAPTURE_PERTURBATION}" ]]; then
        export RELION_FORCE_SAMPLING_PERTURB=${CAPTURE_PERTURBATION}
    fi
fi

ORIGINAL_PATH=${PATH}
ORIGINAL_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
if [[ "${RELION_SKIP_MODULE}" == 0 ]]; then
    source /etc/profile.d/modules.sh
    set +u
    module purge
    module load relion/5.0.0/gcc-11.5.0
    set -u
fi
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/repo_head.txt"
sha256sum "${RELION_BINARY}" "${REFERENCE_COMMAND}" "${DATA_DIR}/particles.star" \
    > "${PROVENANCE_DIR}/static_inputs.sha256"
nvidia-smi -q > "${PROVENANCE_DIR}/nvidia_smi.txt"
printf '%q ' "${RELION_COMMAND[@]}" > "${PROVENANCE_DIR}/relion_command.sh"
printf '\n' >> "${PROVENANCE_DIR}/relion_command.sh"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
case "${gpu_uuid_before}" in GPU-*) ;; *) echo "invalid GPU UUID: ${gpu_uuid_before}" >&2; exit 2 ;; esac
if [[ -n "${TARGET_GPU_UUID}" ]] && [[ "${gpu_uuid_before}" != "${TARGET_GPU_UUID}" ]]; then
    echo "VDAM_TARGET_GPU_MISS expected=${TARGET_GPU_UUID} observed=${gpu_uuid_before}" >&2
    exit 75
fi
start_epoch=$(date +%s)
(
    cd "${DATA_DIR}"
    /usr/bin/time -v srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
        "${RELION_COMMAND[@]}"
) > "${RELION_DIR}/runner.stdout" 2> "${RELION_DIR}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))
gpu_uuid_after=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
test "${gpu_uuid_before}" = "${gpu_uuid_after}"
test -s "${RELION_DIR}/run_it200_model.star"
if [[ -n "${CAPTURE_DIR}" ]]; then
    test -s "${CAPTURE_DIR}/Pmax.bin"
    test -s "${CAPTURE_DIR}/pass0_coarse_raw_diff2.bin"
    test -s "${CAPTURE_DIR}/pass1_exp_Mweight_posterior.bin"
fi
touch "${OUTPUT_ROOT}/SCIENCE_COMPLETED"
export PATH=${ORIGINAL_PATH}
export LD_LIBRARY_PATH=${ORIGINAL_LD_LIBRARY_PATH}
unset MPI_ROOT CUDA_HOME

if [[ "${CAPTURE_ONLY}" == 1 ]]; then
    printf '{"job_id":"%s","wall_s":%d,"gpu_uuid":"%s","capture_iteration":%d,"capture_part_id":%d,"capture_stack_index":"%s"}\n' \
        "${SLURM_JOB_ID}" "${wall_s}" "${gpu_uuid_before}" "${CAPTURE_ITERATION}" \
        "${CAPTURE_PART_ID}" "${CAPTURE_STACK_INDEX}" \
        > "${PROVENANCE_DIR}/capture_completion.json"
    touch "${OUTPUT_ROOT}/CAPTURE_COMPLETED"
    exit 0
fi

map_args=()
particle_iterations=()
for ((iteration = 0; iteration <= 200; iteration++)); do
    map_args+=(--checkpoint "${iteration}")
    if (( iteration > 0 )); then
        particle_iterations+=("${iteration}")
    fi
done
set +e
"${PIXI_PY}" -m scripts.audit_vdam_kclass_trajectory \
    --candidate-dir "${RELION_DIR}" --reference-dir "${REFERENCE_RELION}" --K 1 \
    "${map_args[@]}" --minimum-fsc-auc 0.999 --minimum-assignment-accuracy 0.999 \
    --output-json "${ANALYSIS_DIR}/native_repeat_maps.json" \
    --output-shells-npz "${ANALYSIS_DIR}/native_repeat_shells.npz"
map_status=$?
"${PIXI_PY}" -m scripts.audit_vdam_particle_state_trajectory \
    --recovar-dir "${RELION_DIR}" --relion-dir "${REFERENCE_RELION}" \
    --iterations "${particle_iterations[@]}" \
    --output "${ANALYSIS_DIR}/native_repeat_particles.json"
particle_status=$?
set -e

printf '{"job_id":"%s","wall_s":%d,"gpu_uuid":"%s","map_status":%d,"particle_status":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${gpu_uuid_before}" "${map_status}" "${particle_status}" \
    > "${PROVENANCE_DIR}/completion.json"
sha256sum "${ANALYSIS_DIR}"/* > "${PROVENANCE_DIR}/analysis.sha256"
if (( map_status != 0 || particle_status != 0 )); then
    touch "${OUTPUT_ROOT}/AUDIT_FAILED"
    exit 1
fi
touch "${OUTPUT_ROOT}/COMPLETED"
