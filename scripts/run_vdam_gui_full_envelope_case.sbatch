#!/usr/bin/env bash
#SBATCH --job-name=vdam-gui-envelope
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --array=2-22%4
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-gui-envelope-%A_%a.out

set -euo pipefail

: "${OUTPUT_ROOT:?set the disposable full-envelope output root}"
: "${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_SCORECARD=${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_gui_default_full_suite_v1.json}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
RELION_REFINE=${RELION_REFINE:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
EXPECTED_REPO_HEAD=${EXPECTED_REPO_HEAD:-}

if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
        (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 999 )); then
    echo "VDAM envelope task must be an integer in [1, 999]" >&2
    exit 2
fi
if [[ ! "${VDAM_REPEAT_COUNT}" =~ ^[0-9]+$ ]] || (( VDAM_REPEAT_COUNT < 2 )); then
    echo "VDAM_REPEAT_COUNT must be at least two" >&2
    exit 2
fi
test -x "${PIXI_PY}"
test -x "${RELION_REFINE}"
test -s "${VDAM_SCORECARD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
if [[ -n "${EXPECTED_REPO_HEAD}" ]]; then
    test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
fi

CASE_ID=$(printf 'vdam-gf%02d' "${SLURM_ARRAY_TASK_ID}")
case_match_count=$("${PIXI_PY}" - "${VDAM_SCORECARD}" "${CASE_ID}" <<'PY'
import json
import sys
from pathlib import Path

scorecard = json.loads(Path(sys.argv[1]).read_text())
print(sum(case.get("id") == sys.argv[2] for case in scorecard.get("cases", [])))
PY
)
if [[ "${case_match_count}" != 1 ]]; then
    echo "VDAM scorecard must contain exactly one ${CASE_ID}; found ${case_match_count}" >&2
    exit 2
fi
PANEL_ROOT=${OUTPUT_ROOT}/${CASE_ID}
if [[ -d "${PANEL_ROOT}" ]] && [[ -n "$(find "${PANEL_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty panel root: ${PANEL_ROOT}" >&2
    exit 2
fi
mkdir -p "${PANEL_ROOT}"
touch "${PANEL_ROOT}/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_CUDA_LIB=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
export JAX_COMPILATION_CACHE_DIR=${OUTPUT_ROOT}/jax_compilation_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.01
export TMPDIR=${PANEL_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${PANEL_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${PANEL_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

repeat_one_root=${PANEL_ROOT}/repeat-01
set +e
"${PIXI_PY}" -m scripts.run_vdam_relion_parity_case \
    --case-id "${CASE_ID}" --output-root "${repeat_one_root}" \
    --repo "${REPO_ROOT}" --scorecard "${VDAM_SCORECARD}" \
    --relion-refine "${RELION_REFINE}" --threads 8
paired_status=$?
set -e
candidate_root=${repeat_one_root}/${CASE_ID}
test -s "${candidate_root}/trajectory_audit.json"
test -s "${candidate_root}/particle_state_trajectory_audit.json"
test -s "${candidate_root}/run_provenance.json"
test -s "${candidate_root}/paired_gpu_uuid.json"
if (( paired_status != 0 )); then
    "${PIXI_PY}" - "${candidate_root}/trajectory_audit.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("schema") != "recovar.vdam_relion_fsc_trajectory_audit.v1":
    raise SystemExit("paired run failed before writing valid trajectory evidence")
print("paired strict result retained:", report.get("result"))
PY
fi

for repeat_index in $(seq 2 "${VDAM_REPEAT_COUNT}"); do
    repeat_root=${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")
    "${PIXI_PY}" -m scripts.run_vdam_relion_native_repeat \
        --case-id "${CASE_ID}" --candidate-root "${candidate_root}" \
        --output-root "${repeat_root}" --repo "${REPO_ROOT}" \
        --scorecard "${VDAM_SCORECARD}" --relion-refine "${RELION_REFINE}" \
        --threads 8
done

"${PIXI_PY}" - "${PANEL_ROOT}" "${CASE_ID}" "${VDAM_REPEAT_COUNT}" \
    "$(git -C "${REPO_ROOT}" rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id = sys.argv[2]
repeat_count = int(sys.argv[3])
source_head = sys.argv[4]
candidate = root / "repeat-01" / case_id
gpu = json.loads((candidate / "paired_gpu_uuid.json").read_text())["physical_gpu_uuid"]
repeats = []
for index in range(1, repeat_count + 1):
    repeat = root / f"repeat-{index:02d}" / case_id
    for name in (
        "trajectory_audit.json",
        "particle_state_trajectory_audit.json",
        "run_provenance.json",
        "paired_gpu_uuid.json",
    ):
        if not (repeat / name).is_file():
            raise SystemExit(f"repeat {index} is missing {name}")
    observed = json.loads((repeat / "paired_gpu_uuid.json").read_text())
    if set(observed.get(key) for key in (
        "physical_gpu_uuid", "relion_gpu_uuid", "recovar_gpu_uuid"
    )) != {gpu}:
        raise SystemExit(f"repeat {index} changed physical GPU")
    repeats.append(str(repeat))
report = {
    "schema": "recovar.vdam_gui_full_envelope_science.v1",
    "status": "science_complete",
    "case_id": case_id,
    "source_head": source_head,
    "physical_gpu_uuid": gpu,
    "candidate_root": str(candidate),
    "repeat_count": repeat_count,
    "repeat_roots": repeats,
    "slurm_job_id": str(__import__("os").environ.get("SLURM_JOB_ID", "")),
}
(root / "science_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
touch "${PANEL_ROOT}/SCIENCE_COMPLETED"
