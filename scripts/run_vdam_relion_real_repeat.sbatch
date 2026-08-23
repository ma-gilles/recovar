#!/usr/bin/env bash
#SBATCH --job-name=vdam-real-repeat
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-real-repeat-%j.out

set -euo pipefail

: "${REFERENCE_ROOT:?submit with REFERENCE_ROOT pointing to a completed paired run}"
: "${OUTPUT_ROOT:?submit with a fresh OUTPUT_ROOT}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SCORECARD="${VDAM_REAL_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_real_data_suite_v1.json}"
PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
REFERENCE_COMMAND="${REFERENCE_ROOT}/relion/relion_command.json"
REFERENCE_PROVENANCE="${REFERENCE_ROOT}/run_provenance.json"

cd "${REPO_ROOT}"
test -x "${PIXI_PY}"
test -s "${REFERENCE_COMMAND}"
test -s "${REFERENCE_PROVENANCE}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse repeat output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/relion" /scratch/gpfs/GILLES/mg6942/slurmo
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"

eval "$("${PIXI_PY}" - "${REFERENCE_PROVENANCE}" "${SCORECARD}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text())
scorecard = json.loads(Path(sys.argv[2]).read_text())
case_id = provenance["case_id"]
case = next(row for row in scorecard["cases"] if row["id"] == case_id)
print(f"CASE_ID={shlex.quote(case_id)}")
print(f"DATA_DIR={shlex.quote(case['data_dir'])}")
print(f"INPUT_STAR={shlex.quote(case['input_star'])}")
print("ITERATIONS=" + shlex.quote(" ".join(str(value) for value in scorecard["acceptance_contract"]["required_checkpoints"] if value)))
PY
)"

mapfile -d '' -t RELION_CMD < <("${PIXI_PY}" - "${REFERENCE_COMMAND}" <<'PY'
import json
import sys
from pathlib import Path

for token in json.loads(Path(sys.argv[1]).read_text())["argv"]:
    sys.stdout.buffer.write(str(token).encode() + b"\0")
PY
)

output_flag_count=0
input_flag_count=0
for ((index = 0; index < ${#RELION_CMD[@]}; index++)); do
  if [[ "${RELION_CMD[index]}" == "--o" ]]; then
    ((output_flag_count += 1))
    RELION_CMD[index + 1]="${OUTPUT_ROOT}/relion/run"
  elif [[ "${RELION_CMD[index]}" == "--i" ]]; then
    ((input_flag_count += 1))
    if [[ "${RELION_CMD[index + 1]}" != "${INPUT_STAR}" ]]; then
      echo "reference RELION input differs from frozen scorecard" >&2
      exit 2
    fi
  fi
done
if (( output_flag_count != 1 || input_flag_count != 1 )); then
  echo "reference RELION command must contain exactly one --o and --i" >&2
  exit 2
fi

"${PIXI_PY}" - "${OUTPUT_ROOT}" "${REFERENCE_ROOT}" "${CASE_ID}" "${RELION_CMD[@]}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
reference_root = Path(sys.argv[2]).resolve()
command = sys.argv[4:]
binary = Path(command[0]).resolve(strict=True)
(root / "relion" / "relion_command.json").write_text(
    json.dumps({"argv": command}, indent=2, sort_keys=True) + "\n"
)
report = {
    "schema": "recovar.vdam_relion_real_repeat.v1",
    "case_id": sys.argv[3],
    "reference_root": str(reference_root),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "relion_executable": str(binary),
    "relion_executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
}
(root / "repeat_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

GPU_UUID_BEFORE="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
SECONDS=0
(cd "${DATA_DIR}" && "${RELION_CMD[@]}") >"${OUTPUT_ROOT}/relion/relion.log" 2>&1
RELION_WALL_S="${SECONDS}"
GPU_UUID_AFTER="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
test "${GPU_UUID_BEFORE}" = "${GPU_UUID_AFTER}"

read -r -a ITERATION_ARRAY <<<"${ITERATIONS}"
"${PIXI_PY}" scripts/audit_vdam_particle_state_trajectory.py \
  --recovar-dir "${REFERENCE_ROOT}/relion" \
  --relion-dir "${OUTPUT_ROOT}/relion" \
  --iterations "${ITERATION_ARRAY[@]}" \
  --output "${OUTPUT_ROOT}/relion_repeat_particle_state.json"

"${PIXI_PY}" - "${OUTPUT_ROOT}/repeat_provenance.json" "${GPU_UUID_BEFORE}" "${RELION_WALL_S}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report["physical_gpu_uuid"] = sys.argv[2]
report["relion_wall_s"] = int(sys.argv[3])
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

echo "VDAM RELION real-data repeat complete: ${OUTPUT_ROOT}"
