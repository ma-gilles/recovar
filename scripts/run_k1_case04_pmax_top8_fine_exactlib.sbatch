#!/usr/bin/env bash
#SBATCH --job-name=k1c04-pmax-fine8
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_WRAPPER_SHA256:?EXPECTED_WRAPPER_SHA256 is required}"
: "${EXPECTED_BASE_LAUNCHER_SHA256:?EXPECTED_BASE_LAUNCHER_SHA256 is required}"
: "${EXPECTED_REPO_DIFF_SHA256:?EXPECTED_REPO_DIFF_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
WRAPPER=${REPO}/scripts/run_k1_case04_pmax_top8_fine_exactlib.sbatch
BASE=${REPO}/scripts/run_k1_existing_relion_prefix.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
CASE_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
PANEL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_exactlib_567b22d4ea_20260825T0837ET/analysis/control_vs_exactlib_iter2/candidate_top24.json
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${PANEL}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${EXPECTED_WRAPPER_SHA256}"
test "$(sha256sum "${BASE}" | awk '{print $1}')" = "${EXPECTED_BASE_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

mkdir -p "${RUN_ROOT}/provenance"
TARGET_SOURCE_ROWS=$("${PYTHON}" - "${PANEL}" "${RUN_ROOT}/provenance/target_rows.json" <<'PY'
import json
import sys
from pathlib import Path

panel_path = Path(sys.argv[1])
output = Path(sys.argv[2])
panel = json.loads(panel_path.read_text())
rows = panel["rows"][:8]
source_rows = [int(row["source_row_zero_based"]) for row in rows]
assert len(source_rows) == 8 and len(set(source_rows)) == 8
output.write_text(json.dumps({
    "schema": "recovar.em.k1_case04_pmax_top8_fine_targets.v1",
    "panel": str(panel_path.resolve()),
    "rows": rows,
}, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(",".join(str(row) for row in source_rows))
PY
)
test -n "${TARGET_SOURCE_ROWS}"

export REPO PYTHON CUDA_LIB RELION_BIND
export RUN_ROOT RUNTIME_ROOT CASE_SOURCE
export CASE_NAME=4_high_noise_100k_g256_white_noise3_bf80
export SEED=1704 MAX_ITER=2 TARGET_SOURCE_ROWS TARGET_CURRENT_SIZE=100 TARGET_ITERATION=2
export PASSIVE_SIGNIFICANCE_DUMP=0 PASSIVE_STOP_AFTER_TARGET=0 STOP_AFTER_TARGET=1
export ENABLE_EXACT_BPREF=1 ENABLE_LIVE_INITIAL_NOISE=0 ENABLE_NATIVE_TEXTURE=0
export SPECTRUM_NORM=1 SOFTMASK_REDUCTION=control
export EXPECTED_HEAD=567b22d4ea851589d2d96fbbcad4cad440de93f2
export EXPECTED_LAUNCHER_SHA256=${EXPECTED_BASE_LAUNCHER_SHA256}
export EXPECTED_REPO_DIFF_SHA256

exec bash "${BASE}"
