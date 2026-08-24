#!/usr/bin/env bash
#SBATCH --job-name=vdam-first-boundary
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-first-boundary-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_OPTIMISER:?set the exact incoming full-schedule optimiser.star}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
TARGET_ITERATION=${TARGET_ITERATION:-33}
# These are deliberately different identity spaces for the sealed gf01
# boundary: RECOVAR selects the zero-based source STAR row, while RELION's
# capture selects its shuffled internal particle id and verifies the 1-based
# stack index independently.
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-1002}
TARGET_RELION_PART_ID=${TARGET_RELION_PART_ID:-6}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1003}
TARGET_RELION_PERTURBATION=${TARGET_RELION_PERTURBATION:-0.322510}
CAPTURE_LOCAL_SCORE=${CAPTURE_LOCAL_SCORE:-0}
case "${CAPTURE_LOCAL_SCORE}" in 0|1) ;; *) echo "CAPTURE_LOCAL_SCORE must be 0 or 1" >&2; exit 2 ;; esac
CAPTURE_NATIVE_REPLAY=${CAPTURE_NATIVE_REPLAY:-1}
case "${CAPTURE_NATIVE_REPLAY}" in 0|1) ;; *) echo "CAPTURE_NATIVE_REPLAY must be 0 or 1" >&2; exit 2 ;; esac
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
REFERENCE_RELION_DIR=${REFERENCE_RELION_DIR:-$(dirname "${RELION_OPTIMISER}")}
NATIVE_ROOT=${OUTPUT_ROOT}/native_continuation
RECOVAR_ROOT=${OUTPUT_ROOT}/recovar_prefix
RECOVAR_CAPTURE=${OUTPUT_ROOT}/recovar_fused
RECOVAR_SCORE_CAPTURE=${OUTPUT_ROOT}/recovar_score
PROVENANCE=${OUTPUT_ROOT}/provenance
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
RELION_BIND_BINARY=${REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
iteration_tag=$(printf '%03d' "${TARGET_ITERATION}")
test -s "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star"
test -s "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_sampling.star"
test -x "${RELION_BINARY}"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
mkdir -p "${NATIVE_ROOT}" "${RECOVAR_ROOT}" "${RECOVAR_CAPTURE}" "${RECOVAR_SCORE_CAPTURE}" "${PROVENANCE}"
test -z "$(find "${NATIVE_ROOT}" "${RECOVAR_ROOT}" "${RECOVAR_CAPTURE}" "${RECOVAR_SCORE_CAPTURE}" "${PROVENANCE}" -mindepth 1 -print -quit)"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
case "${gpu_uuid_before}" in GPU-*) ;; *) echo "invalid GPU UUID: ${gpu_uuid_before}" >&2; exit 2 ;; esac

if [[ "${CAPTURE_NATIVE_REPLAY}" == 1 ]]; then
  touch "${NATIVE_ROOT}/SAFE_TO_DELETE"
  VDAM_RELION_CONT_ROOT=${NATIVE_ROOT} \
  VDAM_RELION_CONT_INPUT_DIR=${DATA_DIR} \
  VDAM_RELION_CONT_OPTIMISER=${RELION_OPTIMISER} \
  VDAM_RELION_CONT_ITERATION=${TARGET_ITERATION} \
  VDAM_RELION_CONT_PART_ID=${TARGET_RELION_PART_ID} \
  VDAM_RELION_CONT_STACK_INDEX=${TARGET_STACK_INDEX} \
  VDAM_RELION_CONT_CAPTURE=1 \
  VDAM_RELION_CONT_PERTURBATION=${TARGET_RELION_PERTURBATION} \
  VDAM_RELION_BINARY=${RELION_BINARY} \
  bash "${REPO_ROOT}/scripts/run_vdam_relion_continuation_capture.sbatch"

  rg -q "_rlnSamplingPerturbInstance[[:space:]]+${TARGET_RELION_PERTURBATION}" \
    "${NATIVE_ROOT}/relion/run_it${iteration_tag}_sampling.star"
  "${PIXI_PY}" "${REPO_ROOT}/scripts/audit_vdam_particle_state_trajectory.py" \
    --recovar-dir "${REFERENCE_RELION_DIR}" --relion-dir "${NATIVE_ROOT}/relion" \
    --iterations "${TARGET_ITERATION}" \
    --output "${PROVENANCE}/native_replay_state_audit.json"
  "${PIXI_PY}" - "${PROVENANCE}/native_replay_state_audit.json" \
    "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star" \
    "${NATIVE_ROOT}/relion/run_it${iteration_tag}_data.star" \
    "${TARGET_STACK_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

import starfile

report = json.loads(Path(sys.argv[1]).read_text())
row = report["iterations"][0]
if int(row["divergent_particle_count"]) != 0:
    raise SystemExit("forced native continuation did not reproduce the sealed particle states")


def particles(path):
    value = starfile.read(path)
    return value["particles"] if isinstance(value, dict) else value


reference = particles(sys.argv[2])
replay = particles(sys.argv[3])
image_prefix = f"{int(sys.argv[4])}@"
name_column = "rlnImageName"
pmax_column = "rlnMaxValueProbDistribution"
reference_row = reference[reference[name_column].astype(str).str.startswith(image_prefix)]
replay_row = replay[replay[name_column].astype(str).str.startswith(image_prefix)]
if len(reference_row) != 1 or len(replay_row) != 1:
    raise SystemExit("target stack identity is not unique in native replay audit")
reference_pmax = float(reference_row.iloc[0][pmax_column])
replay_pmax = float(replay_row.iloc[0][pmax_column])
target_error = abs(replay_pmax - reference_pmax)
report["target_pmax"] = {
    "image_name": str(reference_row.iloc[0][name_column]),
    "reference": reference_pmax,
    "replay": replay_pmax,
    "absolute_error": target_error,
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if target_error > 5.0e-4:
    raise SystemExit("forced native continuation target Pmax differs from the sealed trajectory")
PY
else
  test -z "$(find "${NATIVE_ROOT}" -mindepth 1 -print -quit)"
fi

gpu_uuid_after_relion=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_relion}" = "${gpu_uuid_before}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export JAX_COMPILATION_CACHE_DIR=${OUTPUT_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR=${RECOVAR_CAPTURE}
export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=${TARGET_ORIGINAL_INDEX}
export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION=${TARGET_ITERATION}
unset RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY
if [[ "${CAPTURE_LOCAL_SCORE}" == 1 ]]; then
  export RECOVAR_LOCAL_SCORE_DUMP_DIR=${RECOVAR_SCORE_CAPTURE}
  export RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES=${TARGET_ORIGINAL_INDEX}
  export RECOVAR_LOCAL_SCORE_DUMP_ITERATION=${TARGET_ITERATION}
  export RECOVAR_LOCAL_SCORE_DUMP_OPERANDS=1
else
  unset RECOVAR_LOCAL_SCORE_DUMP_DIR RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES
  unset RECOVAR_LOCAL_SCORE_DUMP_ITERATION RECOVAR_LOCAL_SCORE_DUMP_OPERANDS
fi
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

COMMAND=(
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_DIR}/particles.star"
  --o "${RECOVAR_ROOT}/run"
  --nr_iter 200
  --grad_write_iter 1
  --K 1
  --tau2_fudge 4
  --sym C1
  --do_run_C1 1
  --particle_diameter 200
  --random_seed 0
  --healpix_order 1
  --oversampling 1
  --offset_range 6
  --offset_step 2
  --padding_factor 1
  --image_batch_size 500
  --datadir "${DATA_DIR}"
  --gpu 0
  --require_custom_cuda
  --diagnostic_stop_after_iteration "${TARGET_ITERATION}"
)
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE}/recovar_command.sh"
printf '\n' >> "${PROVENANCE}/recovar_command.sh"
sha256sum "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch" \
  "${REPO_ROOT}/scripts/run_vdam_relion_continuation_capture.sbatch" \
  "${DATA_DIR}/particles.star" "${RELION_OPTIMISER}" \
  "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star" \
  "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_sampling.star" "${CUDA_BINARY}" \
  "${RELION_BIND_BINARY}" "${RELION_BINARY}" > "${PROVENANCE}/static_inputs.sha256"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"

start_epoch=$(date +%s)
(
  cd "${REPO_ROOT}"
  "${COMMAND[@]}"
) > "${RECOVAR_ROOT}/runner.stdout" 2> "${RECOVAR_ROOT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${RECOVAR_ROOT}/run_it${iteration_tag}_class001.mrc"
test -s "${RECOVAR_ROOT}/run_it${iteration_tag}_data.star"
test -s "${RECOVAR_ROOT}/run_it${iteration_tag}_model.star"
test -s "${RECOVAR_ROOT}/run_it${iteration_tag}_recovar_meta.json"
test "$(find "${RECOVAR_CAPTURE}" -maxdepth 1 -type f -name 'local_fused_posterior_*.npz' | wc -l)" -eq 1
if [[ "${CAPTURE_LOCAL_SCORE}" == 1 ]]; then
  test "$(find "${RECOVAR_SCORE_CAPTURE}" -maxdepth 1 -type f -name 'local_score_*.npz' | wc -l)" -eq 1
else
  test -z "$(find "${RECOVAR_SCORE_CAPTURE}" -mindepth 1 -print -quit)"
fi
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
gpu_uuid_after_recovar=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_recovar}" = "${gpu_uuid_before}"
printf '{"job_id":"%s","git_head":"%s","physical_gpu_uuid":"%s","nr_iter_schedule":200,"stopped_after_iteration":%d,"target_original_index":%d,"target_relion_part_id":%d,"target_stack_index":%d,"capture_local_score":%s,"capture_native_replay":%s,"recovar_wall_s":%d}\n' \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${gpu_uuid_before}" "${TARGET_ITERATION}" \
  "${TARGET_ORIGINAL_INDEX}" "${TARGET_RELION_PART_ID}" "${TARGET_STACK_INDEX}" \
  "${CAPTURE_LOCAL_SCORE}" "${CAPTURE_NATIVE_REPLAY}" "${wall_s}" \
  > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
