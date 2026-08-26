#!/usr/bin/env bash
#SBATCH --job-name=k1c04-pmax-native
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_WRAPPER_SHA256:?EXPECTED_WRAPPER_SHA256 is required}"
: "${EXPECTED_BASE_LAUNCHER_SHA256:?EXPECTED_BASE_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
WRAPPER=${REPO}/scripts/run_k1_case04_pmax_top1_native_capture.sbatch
BASE=${REPO}/scripts/run_k1_case04_native_fine_continuation.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
PANEL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_exactlib_567b22d4ea_20260825T0837ET/analysis/control_vs_exactlib_iter2/candidate_top24.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${PANEL}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${EXPECTED_WRAPPER_SHA256}"
test "$(sha256sum "${BASE}" | awk '{print $1}')" = "${EXPECTED_BASE_LAUNCHER_SHA256}"

mkdir -p "${RUN_ROOT}/provenance"
TARGET_STACK_ONE_BASED=$("${PYTHON}" - "${PANEL}" "${RUN_ROOT}/provenance/target_row.json" <<'PY'
import json
import sys
from pathlib import Path

panel_path = Path(sys.argv[1])
output = Path(sys.argv[2])
panel = json.loads(panel_path.read_text())
row = panel["rows"][0]
source_row = int(row["source_row_zero_based"])
output.write_text(json.dumps({
    "schema": "recovar.em.k1_case04_pmax_top1_native_target.v1",
    "panel": str(panel_path.resolve()),
    "row": row,
    "stack_index_one_based": source_row + 1,
}, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(source_row + 1)
PY
)
test "${TARGET_STACK_ONE_BASED}" -gt 0

export RUN_ROOT RUNTIME_ROOT TARGET_STACK_ONE_BASED
export CAPTURE_COARSE=1 CAPTURE_PPREF=1
export EXPECTED_REPO_HEAD=567b22d4ea851589d2d96fbbcad4cad440de93f2
export EXPECTED_LAUNCHER_SHA256=${EXPECTED_BASE_LAUNCHER_SHA256}

exec bash "${BASE}"
