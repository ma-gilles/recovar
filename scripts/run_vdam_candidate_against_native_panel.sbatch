#!/usr/bin/env bash
#SBATCH --job-name=vdam-candidate-panel
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-candidate-panel-%j.out

set -euo pipefail

: "${CASE_ID:?set the frozen VDAM case id}"
: "${OUTPUT_ROOT:?set a new disposable candidate output root}"
: "${PANEL_ROOT:?set the completed native-repeat panel root}"
: "${FIXTURE_DIR:?set the frozen materialized fixture directory}"
: "${TARGET_GPU_UUID:?pin the physical GPU used by the native panel}"
: "${EXPECTED_REPO_HEAD:?pin the candidate source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified CUDA library}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_SCORECARD=${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_gui_default_full_suite_v1.json}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
RECOVAR_CUDA_LIB_PATH=${RECOVAR_CUDA_LIB_OVERRIDE:-${REPO_ROOT}/recovar/cuda/libcuda_backproject.so}
CANDIDATE_ROOT=${OUTPUT_ROOT}/${CASE_ID}
RECOVAR_DIR=${CANDIDATE_ROOT}/recovar
PROVENANCE_DIR=${CANDIDATE_ROOT}/provenance
ANALYSIS_DIR=${CANDIDATE_ROOT}/analysis/native_envelope

test -x "${PIXI_PY}"
test -s "${VDAM_SCORECARD}"
test -s "${RECOVAR_CUDA_LIB_PATH}"
test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test -s "${FIXTURE_DIR}/fixture_materialization.json"
test -s "${FIXTURE_DIR}/particles.star"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB_PATH}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

shopt -s nullglob
RELION_BIND_MATCHES=("${REPO_ROOT}"/recovar/relion_bind/_relion_bind_core*.so)
shopt -u nullglob
if (( ${#RELION_BIND_MATCHES[@]} != 1 )); then
    echo "expected exactly one local RELION binding" >&2
    exit 2
fi
test "$(sha256sum "${RELION_BIND_MATCHES[0]}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"

PANEL_GPU_UUID=$("${PIXI_PY}" - "${PANEL_ROOT}" "${CASE_ID}" "${VDAM_REPEAT_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id = sys.argv[2]
count = int(sys.argv[3])
values = {
    json.loads(
        (root / f"repeat-{index:02d}" / case_id / "paired_gpu_uuid.json").read_text()
    )["physical_gpu_uuid"]
    for index in range(1, count + 1)
}
if len(values) != 1:
    raise SystemExit(f"native panel spans physical GPUs: {sorted(values)}")
print(values.pop())
PY
)
test "${PANEL_GPU_UUID}" = "${TARGET_GPU_UUID}"

# A native envelope is only calibrated for the physical GPU that produced it.
# Exit before creating output when Slurm assigns another H100 so the job can be
# safely resubmitted without leaving a partial candidate behind.
mapfile -t VISIBLE_GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if (( ${#VISIBLE_GPU_UUIDS[@]} != 1 )); then
    echo "expected exactly one visible physical GPU, found ${#VISIBLE_GPU_UUIDS[@]}" >&2
    exit 2
fi
VISIBLE_GPU_UUID=${VISIBLE_GPU_UUIDS[0]//[[:space:]]/}
if [[ "${VISIBLE_GPU_UUID}" != "${TARGET_GPU_UUID}" ]]; then
    echo "VDAM_TARGET_GPU_MISS expected=${TARGET_GPU_UUID} observed=${VISIBLE_GPU_UUID}" >&2
    exit 75
fi

if [[ -d "${CANDIDATE_ROOT}" ]] && [[ -n "$(find "${CANDIDATE_ROOT}" -mindepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty candidate root: ${CANDIDATE_ROOT}" >&2
    exit 2
fi
mkdir -p "${RECOVAR_DIR}" "${PROVENANCE_DIR}" "${ANALYSIS_DIR}"
touch "${CANDIDATE_ROOT}/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${RECOVAR_CUDA_LIB_PATH}
export JAX_COMPILATION_CACHE_DIR=${OUTPUT_ROOT}/jax_compilation_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.01
export TMPDIR=${CANDIDATE_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${CANDIDATE_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${CANDIDATE_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core

repo = pathlib.Path(__import__("os").environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
binding_path = pathlib.Path(_relion_bind_core.__file__).resolve()
assert binding_path.is_file()
assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM candidate-only GPU/CUDA-FFI gate passed", jax.devices(), binding_path)
PY

"${PIXI_PY}" - "${VDAM_SCORECARD}" "${CASE_ID}" "${FIXTURE_DIR}" \
    "${RECOVAR_DIR}/run" "${RECOVAR_DIR}/recovar_command.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.run_vdam_relion_parity_case import build_recovar_command

scorecard = json.loads(Path(sys.argv[1]).read_text())
case_id = sys.argv[2]
fixture_dir = Path(sys.argv[3])
output_prefix = Path(sys.argv[4])
output_json = Path(sys.argv[5])
matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one scorecard row for {case_id}, found {len(matches)}")
definition = matches[0]["definition"]
if int(definition["nr_classes"]) != 1 or int(definition["nr_iter"]) != 200:
    raise SystemExit("candidate-panel qualification requires a frozen K=1, 200-iteration case")
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
output_json.write_text(json.dumps({"argv": command}, indent=2, sort_keys=True) + "\n")
PY
mapfile -d '' -t RECOVAR_COMMAND < <("${PIXI_PY}" - "${RECOVAR_DIR}/recovar_command.json" <<'PY'
import json
import sys
from pathlib import Path

for token in json.loads(Path(sys.argv[1]).read_text())["argv"]:
    sys.stdout.buffer.write(str(token).encode() + b"\0")
PY
)

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/repo_head.txt"
sha256sum "${RECOVAR_CUDA_LIB}" "${RELION_BIND_MATCHES[0]}" "${VDAM_SCORECARD}" \
    "${FIXTURE_DIR}/particles.star" "${FIXTURE_DIR}/fixture_materialization.json" \
    > "${PROVENANCE_DIR}/input_sha256.txt"
nvidia-smi -q > "${PROVENANCE_DIR}/nvidia_smi.txt"
printf '%q ' "${RECOVAR_COMMAND[@]}" > "${PROVENANCE_DIR}/recovar_command.sh"
printf '\n' >> "${PROVENANCE_DIR}/recovar_command.sh"
env | LC_ALL=C sort > "${PROVENANCE_DIR}/runtime_environment.txt"

start_epoch=$(date +%s)
/usr/bin/time -v srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=none \
    "${RECOVAR_COMMAND[@]}" > "${RECOVAR_DIR}/runner.stdout" 2> "${RECOVAR_DIR}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))
test -s "${RECOVAR_DIR}/run_it200_model.star"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "${TARGET_GPU_UUID}"
printf '{"schema":"recovar.vdam_candidate_panel_science.v1","job_id":"%s","wall_s":%d,"gpu_uuid":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_GPU_UUID}" > "${PROVENANCE_DIR}/completion.json"
touch "${CANDIDATE_ROOT}/SCIENCE_COMPLETED"

PIXEL_SIZE=$("${PIXI_PY}" - "${FIXTURE_DIR}/particles.star" <<'PY'
import sys
from pathlib import Path

import starfile

value = starfile.read(Path(sys.argv[1]))
if not isinstance(value, dict) or value.get("optics") is None:
    raise SystemExit("fixture STAR has no optics block")
optics = value["optics"]
matches = [column for column in optics.columns if str(column).lstrip("_") == "rlnImagePixelSize"]
if len(matches) != 1:
    raise SystemExit(f"expected one rlnImagePixelSize column, found {matches}")
values = sorted(set(float(item) for item in optics[matches[0]].tolist()))
if len(values) != 1 or values[0] <= 0:
    raise SystemExit(f"expected one positive pixel size, found {values}")
print(values[0])
PY
)
export CANDIDATE_ROOT PANEL_ROOT FIXTURE_DIR VDAM_SCORECARD CASE_ID PIXEL_SIZE
export VDAM_REPEAT_COUNT REPO_ROOT EXPECTED_REPO_HEAD PIXI_PY_OVERRIDE="${PIXI_PY}"
export ANALYSIS_DIR
bash "${REPO_ROOT}/scripts/run_vdam_candidate_envelope_audits.sbatch"
sha256sum "${ANALYSIS_DIR}"/* > "${PROVENANCE_DIR}/analysis_sha256.txt"
touch "${CANDIDATE_ROOT}/COMPLETED"
