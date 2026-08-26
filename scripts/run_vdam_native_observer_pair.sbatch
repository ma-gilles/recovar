#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-observer
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:10:00

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${DATA_DIR:?set the frozen materialized VDAM fixture directory}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_RELION_SHA256:?pin the patched RELION binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
RELION_BINARY=${VDAM_RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
TARGET_ITERATION=${TARGET_ITERATION:-2}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
TARGET_PART_ID=${TARGET_PART_ID:-66}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1058}
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
PROVENANCE=${OUTPUT_ROOT}/provenance
ANALYSIS=${OUTPUT_ROOT}/analysis

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -x "${RELION_BINARY}"
test -x "${PIXI_PY}"
test "${TARGET_ITERATION}" -ge 1
test "${NR_ITER_SCHEDULE}" -ge "${TARGET_ITERATION}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_SHA256}"
mkdir -p "${PROVENANCE}" "${ANALYSIS}"
test -z "$(find "${PROVENANCE}" "${ANALYSIS}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
mkdir -p "${TMPDIR}"

source /etc/profile.d/modules.sh
set +u
module purge
set +e
module load relion/5.0.0/gcc-11.5.0
module_status=$?
set -e
if [[ "${module_status}" -ne 0 ]]; then
  echo "RELION module returned ${module_status}; using the pinned runtime overlay" >&2
fi
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
case "${gpu_uuid_before}" in GPU-*) ;; *) echo "invalid GPU UUID: ${gpu_uuid_before}" >&2; exit 2 ;; esac
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"

run_variant() {
  local variant=$1
  local observer=$2
  local output=${OUTPUT_ROOT}/${variant}
  local capture=${OUTPUT_ROOT}/${variant}_capture
  mkdir -p "${output}" "${capture}"
  test -z "$(find "${output}" "${capture}" -mindepth 1 -print -quit)"

  unset RELION_ACC_DUMP_DIR RELION_ACC_DUMP_PART_ID RELION_ACC_DUMP_ITER RELION_ACC_DUMP_VERBOSE
  if [[ "${observer}" == 1 ]]; then
    export RELION_ACC_DUMP_DIR=${capture}
    export RELION_ACC_DUMP_PART_ID=${TARGET_PART_ID}
    export RELION_ACC_DUMP_ITER=${TARGET_ITERATION}
    export RELION_ACC_DUMP_VERBOSE=0
  fi

  local command=(
    srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "${RELION_BINARY}"
    --o "${output}/run" --iter "${NR_ITER_SCHEDULE}" --grad --denovo_3dref
    --grad_write_iter 1 --i particles.star --ctf --K 1 --sym C1
    --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 3 --pad 1
    --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6
    --offset_step 2 --auto_sampling --tau2_fudge 4 --random_seed 0 --j 8 --gpu 0
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE}/${variant}_command.sh"
  printf '\n' >> "${PROVENANCE}/${variant}_command.sh"

  set +e
  (
    cd "${DATA_DIR}"
    "${command[@]}"
  ) > "${output}/runner.stdout" 2> "${output}/runner.stderr" &
  local relion_pid=$!
  set -e
  local target_tag
  target_tag=$(printf '%03d' "${TARGET_ITERATION}")
  local target_complete=0
  for _poll_index in $(seq 1 300); do
    if [[ -s "${output}/run_it${target_tag}_sampling.star" ]]; then
      target_complete=1
      break
    fi
    if ! kill -0 "${relion_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  test "${target_complete}" -eq 1
  local active_step
  active_step=$(
    squeue --steps -j "${SLURM_JOB_ID}" -h -o '%i' \
      | awk -v job="${SLURM_JOB_ID}" '$0 ~ "^" job "\\.[0-9]+$" {step=$0} END {print step}'
  )
  if [[ -n "${active_step}" ]]; then
    scancel "${active_step}"
  else
    kill "${relion_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${relion_pid}"
  set -e

  test -s "${output}/run_it${target_tag}_data.star"
  test -s "${output}/run_it${target_tag}_class001.mrc"
  if [[ "${observer}" == 1 ]]; then
    test -s "${capture}/img0_part${TARGET_PART_ID}_storeWavg_sorted_weights.bin"
  else
    test -z "$(find "${capture}" -mindepth 1 -print -quit)"
  fi
}

run_variant observer_off_a 0
run_variant observer_on 1
run_variant observer_off_b 0
unset RELION_ACC_DUMP_DIR RELION_ACC_DUMP_PART_ID RELION_ACC_DUMP_ITER RELION_ACC_DUMP_VERBOSE

target_tag=$(printf '%03d' "${TARGET_ITERATION}")
"${PIXI_PY}" - "${OUTPUT_ROOT}" "${target_tag}" "${TARGET_STACK_INDEX}" \
  "${gpu_uuid_before}" "${ANALYSIS}/observer_pair.json" <<'PY'
import json
import sys
from pathlib import Path

import starfile

root = Path(sys.argv[1])
iteration_tag = sys.argv[2]
stack_index = int(sys.argv[3])
gpu_uuid = sys.argv[4]
output = Path(sys.argv[5])
variants = {}
for name in ("observer_off_a", "observer_on", "observer_off_b"):
    value = starfile.read(root / name / f"run_it{iteration_tag}_data.star")
    particles = value["particles"] if isinstance(value, dict) else value
    row = particles[
        particles["rlnImageName"].astype(str).str.startswith(f"{stack_index}@")
    ]
    if len(row) != 1:
        raise SystemExit(f"{name}: target stack identity is not unique")
    row = row.iloc[0]
    variants[name] = {
        "image_name": str(row["rlnImageName"]),
        "pmax": float(row["rlnMaxValueProbDistribution"]),
        "pose_deg": [
            float(row["rlnAngleRot"]),
            float(row["rlnAngleTilt"]),
            float(row["rlnAnglePsi"]),
        ],
        "translation_angstrom": [
            float(row["rlnOriginXAngst"]),
            float(row["rlnOriginYAngst"]),
        ],
    }
report = {
    "schema": "recovar.vdam_native_observer_pair.v1",
    "physical_gpu_uuid": gpu_uuid,
    "variants": variants,
    "observer_off_repeat_pmax_absolute_error": abs(
        variants["observer_off_a"]["pmax"] - variants["observer_off_b"]["pmax"]
    ),
    "observer_on_minus_off_a_pmax": (
        variants["observer_on"]["pmax"] - variants["observer_off_a"]["pmax"]
    ),
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

gpu_uuid_after=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after}" = "${gpu_uuid_before}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
sha256sum "${REPO_ROOT}/scripts/run_vdam_native_observer_pair.sbatch" \
  "${DATA_DIR}/particles.star" "${RELION_BINARY}" > "${PROVENANCE}/static_inputs.sha256"
printf '{"job_id":"%s","git_head":"%s","physical_gpu_uuid":"%s","target_iteration":%d,"nr_iter_schedule":%d,"target_part_id":%d,"target_stack_index":%d}\n' \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${gpu_uuid_before}" "${TARGET_ITERATION}" \
  "${NR_ITER_SCHEDULE}" "${TARGET_PART_ID}" "${TARGET_STACK_INDEX}" > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
