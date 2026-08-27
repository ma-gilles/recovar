#!/usr/bin/env bash
#SBATCH --job-name=vdam-first-boundary
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
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
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
VDAM_SCORECARD=${VDAM_SCORECARD:-}
CASE_ID=${CASE_ID:-}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
# These are deliberately different identity spaces for the sealed gf01
# boundary: RECOVAR selects the zero-based source STAR row, while RELION's
# capture selects its shuffled internal particle id and verifies the 1-based
# stack index independently.
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-1002}
TARGET_RELION_PART_ID=${TARGET_RELION_PART_ID:-6}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1003}
TARGET_RELION_PERTURBATION=${TARGET_RELION_PERTURBATION:-}
TARGET_RELION_PERTURBATION_SOURCE=${TARGET_RELION_PERTURBATION_SOURCE:-}
CAPTURE_LOCAL_SCORE=${CAPTURE_LOCAL_SCORE:-0}
case "${CAPTURE_LOCAL_SCORE}" in 0|1) ;; *) echo "CAPTURE_LOCAL_SCORE must be 0 or 1" >&2; exit 2 ;; esac
CAPTURE_FUSED_SCORES=${CAPTURE_FUSED_SCORES:-0}
case "${CAPTURE_FUSED_SCORES}" in 0|1) ;; *) echo "CAPTURE_FUSED_SCORES must be 0 or 1" >&2; exit 2 ;; esac
CAPTURE_COARSE_SCORE=${CAPTURE_COARSE_SCORE:-0}
case "${CAPTURE_COARSE_SCORE}" in 0|1) ;; *) echo "CAPTURE_COARSE_SCORE must be 0 or 1" >&2; exit 2 ;; esac
CAPTURE_NATIVE_REPLAY=${CAPTURE_NATIVE_REPLAY:-1}
case "${CAPTURE_NATIVE_REPLAY}" in 0|1) ;; *) echo "CAPTURE_NATIVE_REPLAY must be 0 or 1" >&2; exit 2 ;; esac
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
REFERENCE_RELION_DIR=${REFERENCE_RELION_DIR:-$(dirname "${RELION_OPTIMISER}")}
NATIVE_ROOT=${OUTPUT_ROOT}/native_continuation
RECOVAR_ROOT=${OUTPUT_ROOT}/recovar_prefix
RECOVAR_CAPTURE=${OUTPUT_ROOT}/recovar_fused
RECOVAR_SCORE_CAPTURE=${OUTPUT_ROOT}/recovar_score
RECOVAR_COARSE_CAPTURE=${OUTPUT_ROOT}/recovar_coarse
PROVENANCE=${OUTPUT_ROOT}/provenance
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_SOURCE_BINARY=${RECOVAR_CUDA_LIB_OVERRIDE:-${REPO_ROOT}/recovar/cuda/libcuda_backproject.so}
CUDA_BINARY=${OUTPUT_ROOT}/runtime/native/libcuda_backproject.so
RELION_BIND_BINARY=${REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
test "${TARGET_ITERATION}" -ge 1
test "${NR_ITER_SCHEDULE}" -ge "${TARGET_ITERATION}"
if [[ -n "${VDAM_SCORECARD}" || -n "${CASE_ID}" ]]; then
  test -n "${VDAM_SCORECARD}"
  test -n "${CASE_ID}"
  test -s "${VDAM_SCORECARD}"
fi
iteration_tag=$(printf '%03d' "${TARGET_ITERATION}")
test -s "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star"
test -s "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_sampling.star"
if [[ -z "${TARGET_RELION_PERTURBATION}" ]]; then
  if [[ -z "${TARGET_RELION_PERTURBATION_SOURCE}" ]]; then
    TARGET_RELION_PERTURBATION_SOURCE=$(dirname "${REFERENCE_RELION_DIR}")/recovar/run_it${iteration_tag}_recovar_meta.json
  fi
  test -s "${TARGET_RELION_PERTURBATION_SOURCE}"
  TARGET_RELION_PERTURBATION=$("${PIXI_PY}" - "${TARGET_RELION_PERTURBATION_SOURCE}" <<'PY'
import json
import math
import sys
from pathlib import Path

value = float(json.loads(Path(sys.argv[1]).read_text())["random_perturbation"])
if not math.isfinite(value) or not -0.5 <= value <= 0.5:
    raise SystemExit(f"invalid full-precision perturbation: {value!r}")
print(repr(value))
PY
  )
else
  TARGET_RELION_PERTURBATION_SOURCE=${TARGET_RELION_PERTURBATION_SOURCE:-environment}
fi
test -x "${RELION_BINARY}"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${CUDA_SOURCE_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
case "${gpu_uuid_before}" in GPU-*) ;; *) echo "invalid GPU UUID: ${gpu_uuid_before}" >&2; exit 2 ;; esac
if [[ -n "${TARGET_GPU_UUID}" ]] && [[ "${gpu_uuid_before}" != "${TARGET_GPU_UUID}" ]]; then
  echo "VDAM_TARGET_GPU_MISS expected=${TARGET_GPU_UUID} observed=${gpu_uuid_before}" >&2
  exit 75
fi
mkdir -p "${NATIVE_ROOT}" "${RECOVAR_ROOT}" "${RECOVAR_CAPTURE}" "${RECOVAR_SCORE_CAPTURE}" "${RECOVAR_COARSE_CAPTURE}" "${PROVENANCE}"
test -z "$(find "${NATIVE_ROOT}" "${RECOVAR_ROOT}" "${RECOVAR_CAPTURE}" "${RECOVAR_SCORE_CAPTURE}" "${RECOVAR_COARSE_CAPTURE}" "${PROVENANCE}" -mindepth 1 -print -quit)"
mkdir -p "$(dirname "${CUDA_BINARY}")"
cp --reflink=auto "${CUDA_SOURCE_BINARY}" "${CUDA_BINARY}"
chmod 0755 "${CUDA_BINARY}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

(
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
  export PYTHONNOUSERSITE=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export RECOVAR_CUDA_LIB=${CUDA_BINARY}
  "${PIXI_PY}" - <<'PY'
import jax
import recovar.cuda_backproject as cuda_backproject

assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM boundary GPU/CUDA-FFI gate passed", jax.devices())
PY
)
if [[ "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" != "${EXPECTED_CUDA_SHA256}" ]]; then
  echo "VDAM boundary CUDA library changed during import" >&2
  exit 2
fi

if [[ "${CAPTURE_NATIVE_REPLAY}" == 1 ]]; then
  touch "${NATIVE_ROOT}/SAFE_TO_DELETE"
  VDAM_RELION_CONT_ROOT=${NATIVE_ROOT} \
  VDAM_RELION_CONT_INPUT_DIR=${DATA_DIR} \
  VDAM_RELION_CONT_OPTIMISER=${RELION_OPTIMISER} \
  VDAM_RELION_CONT_ITERATION=${TARGET_ITERATION} \
  VDAM_RELION_CONT_NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE} \
  VDAM_RELION_CONT_PART_ID=${TARGET_RELION_PART_ID} \
  VDAM_RELION_CONT_STACK_INDEX=${TARGET_STACK_INDEX} \
  VDAM_RELION_CONT_CAPTURE=1 \
  VDAM_RELION_CONT_PERTURBATION=${TARGET_RELION_PERTURBATION} \
  VDAM_RELION_BINARY=${RELION_BINARY} \
  bash "${REPO_ROOT}/scripts/run_vdam_relion_continuation_capture.sbatch"

  "${PIXI_PY}" - \
    "${NATIVE_ROOT}/relion/run_it${iteration_tag}_sampling.star" \
    "${TARGET_RELION_PERTURBATION}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r"_rlnSamplingPerturbInstance\s+([-+0-9.eE]+)", text)
if match is None:
    raise SystemExit("continued sampling STAR lacks the perturbation field")
observed = float(match.group(1))
expected = float(sys.argv[2])
# MetaDataTable::getValueToString uses %12.5f for negative doubles and
# %12.6f for non-negative doubles.  Verify that exact serialized value while
# retaining the full-precision perturbation in the live RELION process.
serialized_expected = float(format(expected, ".5f" if expected < 0.0 else ".6f"))
if observed != serialized_expected:
    raise SystemExit(
        "serialized perturbation differs from RELION's representation of the "
        "forced full-precision value: "
        f"observed={observed:.9g} expected={serialized_expected:.9g} "
        f"live={expected:.17g}"
    )
PY
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.audit_vdam_particle_state_trajectory \
      --recovar-dir "${REFERENCE_RELION_DIR}" --relion-dir "${NATIVE_ROOT}/relion" \
      --iterations "${TARGET_ITERATION}" \
      --output "${PROVENANCE}/native_replay_state_audit.json"
  )
  "${PIXI_PY}" - "${PROVENANCE}/native_replay_state_audit.json" \
    "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star" \
    "${NATIVE_ROOT}/relion/run_it${iteration_tag}_data.star" \
    "${TARGET_STACK_INDEX}" "${NR_ITER_SCHEDULE}" <<'PY'
import json
import sys
from pathlib import Path

import starfile

report = json.loads(Path(sys.argv[1]).read_text())
# The continuation must retain the autonomous 200-iteration schedule because
# RELION derives its VDAM subset annealing from the total iteration count.
# Only the requested iteration is audited and captured below.
report["native_replay_full_schedule"] = True
report["native_replay_nr_iter_schedule"] = int(sys.argv[5])


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
if [[ "${CAPTURE_FUSED_SCORES}" == 1 ]]; then
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES=1
else
  unset RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES
fi
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
if [[ "${CAPTURE_COARSE_SCORE}" == 1 ]]; then
  export RECOVAR_SIGNIFICANCE_DUMP_DIR=${RECOVAR_COARSE_CAPTURE}
  export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDEX}
  export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=${TARGET_ITERATION}
else
  unset RECOVAR_SIGNIFICANCE_DUMP_DIR RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES
  unset RECOVAR_SIGNIFICANCE_DUMP_ITERATION
fi
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

if [[ -n "${VDAM_SCORECARD}" ]]; then
  command_json=${PROVENANCE}/recovar_command.json
  "${PIXI_PY}" - "${VDAM_SCORECARD}" "${CASE_ID}" "${DATA_DIR}" \
    "${RECOVAR_ROOT}/run" "${TARGET_ITERATION}" "${NR_ITER_SCHEDULE}" \
    "${command_json}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.run_vdam_relion_parity_case import build_recovar_command

scorecard_path = Path(sys.argv[1])
case_id = sys.argv[2]
fixture_dir = Path(sys.argv[3])
output_prefix = Path(sys.argv[4])
target_iteration = int(sys.argv[5])
nr_iter_schedule = int(sys.argv[6])
output_json = Path(sys.argv[7])
scorecard = json.loads(scorecard_path.read_text())
matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
if len(matches) != 1:
    raise SystemExit(
        f"expected exactly one scorecard definition for {case_id}, found {len(matches)}"
    )
definition = matches[0]["definition"]
if int(definition["nr_classes"]) != 1:
    raise SystemExit("first-state boundary capture currently requires K=1")
if int(definition["nr_iter"]) != nr_iter_schedule:
    raise SystemExit(
        "scorecard nr_iter differs from NR_ITER_SCHEDULE: "
        f"{definition['nr_iter']} != {nr_iter_schedule}"
    )
input_name = (
    "particles_relion_identity_ctf.star"
    if (fixture_dir / "particles_relion_identity_ctf.star").is_file()
    else "particles.star"
)
command = build_recovar_command(
    input_star=fixture_dir / input_name,
    output_prefix=output_prefix,
    fixture_dir=fixture_dir,
    definition=definition,
    image_batch_size=500,
)
command.extend(["--diagnostic_stop_after_iteration", str(target_iteration)])
output_json.write_text(json.dumps({"argv": command}, indent=2, sort_keys=True) + "\n")
PY
  mapfile -d '' -t COMMAND < <("${PIXI_PY}" - "${command_json}" <<'PY'
import json
import sys
from pathlib import Path

for token in json.loads(Path(sys.argv[1]).read_text())["argv"]:
    sys.stdout.buffer.write(str(token).encode() + b"\0")
PY
  )
else
  COMMAND=(
    "${PIXI_PY}" -m scripts.run_ab_initio
    --i "${DATA_DIR}/particles.star"
    --o "${RECOVAR_ROOT}/run"
    --nr_iter "${NR_ITER_SCHEDULE}"
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
fi
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE}/recovar_command.sh"
printf '\n' >> "${PROVENANCE}/recovar_command.sh"
STATIC_INPUTS=(
  "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"
  "${REPO_ROOT}/scripts/run_vdam_relion_continuation_capture.sbatch"
  "${DATA_DIR}/particles.star"
  "${RELION_OPTIMISER}"
  "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_data.star"
  "${REFERENCE_RELION_DIR}/run_it${iteration_tag}_sampling.star"
  "${CUDA_SOURCE_BINARY}"
  "${CUDA_BINARY}"
  "${RELION_BIND_BINARY}"
  "${RELION_BINARY}"
)
if [[ -n "${VDAM_SCORECARD}" ]]; then
  STATIC_INPUTS+=("${VDAM_SCORECARD}")
fi
if [[ "${TARGET_RELION_PERTURBATION_SOURCE}" != environment ]]; then
  STATIC_INPUTS+=("${TARGET_RELION_PERTURBATION_SOURCE}")
fi
sha256sum "${STATIC_INPUTS[@]}" > "${PROVENANCE}/static_inputs.sha256"
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
if [[ "${CAPTURE_FUSED_SCORES}" == 1 ]]; then
  "${PIXI_PY}" - "${RECOVAR_CAPTURE}" <<'PY'
import sys
from pathlib import Path

import numpy as np

paths = list(Path(sys.argv[1]).glob("local_fused_posterior_*.npz"))
if len(paths) != 1:
    raise SystemExit("expected exactly one fused-posterior capture")
with np.load(paths[0], allow_pickle=False) as payload:
    required = {
        "pass2_scores_raw",
        "pass2_scores_total",
        "rotation_log_prior",
        "translation_log_prior",
    }
    missing = required - set(payload.files)
    if missing:
        raise SystemExit(f"fused score capture is missing fields: {sorted(missing)}")
    if payload["pass2_scores_total"].shape != payload["posterior"].shape:
        raise SystemExit("fused score/posterior capture shapes differ")
PY
fi
if [[ "${CAPTURE_LOCAL_SCORE}" == 1 ]]; then
  test "$(find "${RECOVAR_SCORE_CAPTURE}" -maxdepth 1 -type f -name 'local_score_*.npz' | wc -l)" -eq 1
else
  test -z "$(find "${RECOVAR_SCORE_CAPTURE}" -mindepth 1 -print -quit)"
fi
if [[ "${CAPTURE_COARSE_SCORE}" == 1 ]]; then
  test "$(find "${RECOVAR_COARSE_CAPTURE}" -maxdepth 1 -type f -name 'significance_*.npz' | wc -l)" -eq 1
else
  test -z "$(find "${RECOVAR_COARSE_CAPTURE}" -mindepth 1 -print -quit)"
fi
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
gpu_uuid_after_recovar=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_recovar}" = "${gpu_uuid_before}"
if [[ "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" != "${EXPECTED_CUDA_SHA256}" ]]; then
  echo "VDAM boundary CUDA library changed during science" >&2
  exit 2
fi
test "$(sha256sum "${CUDA_SOURCE_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
"${PIXI_PY}" - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${gpu_uuid_before}" "${NR_ITER_SCHEDULE}" \
  "${TARGET_ITERATION}" "${TARGET_ORIGINAL_INDEX}" "${TARGET_RELION_PART_ID}" \
  "${TARGET_STACK_INDEX}" "${CAPTURE_LOCAL_SCORE}" "${CAPTURE_FUSED_SCORES}" \
  "${CAPTURE_COARSE_SCORE}" "${CAPTURE_NATIVE_REPLAY}" "${wall_s}" "${CASE_ID}" \
  "${VDAM_SCORECARD}" "${TARGET_RELION_PERTURBATION}" \
  "${TARGET_RELION_PERTURBATION_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "physical_gpu_uuid": sys.argv[4],
    "nr_iter_schedule": int(sys.argv[5]),
    "stopped_after_iteration": int(sys.argv[6]),
    "target_original_index": int(sys.argv[7]),
    "target_relion_part_id": int(sys.argv[8]),
    "target_stack_index": int(sys.argv[9]),
    "capture_local_score": bool(int(sys.argv[10])),
    "capture_fused_scores": bool(int(sys.argv[11])),
    "capture_coarse_score": bool(int(sys.argv[12])),
    "capture_native_replay": bool(int(sys.argv[13])),
    "recovar_wall_s": int(sys.argv[14]),
    "case_id": sys.argv[15] or None,
    "scorecard": sys.argv[16] or None,
    "target_relion_perturbation": float(sys.argv[17]),
    "target_relion_perturbation_source": sys.argv[18],
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
