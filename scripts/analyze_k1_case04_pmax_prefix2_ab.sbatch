#!/usr/bin/env bash
#SBATCH --job-name=k1-pmax-prefix2-ab
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_STATE_AB_SHA256:?EXPECTED_STATE_AB_SHA256 is required}"
: "${EXPECTED_PANEL_SHA256:?EXPECTED_PANEL_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_567b22d4ea_20260825T0835ET
CANDIDATE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_exactlib_567b22d4ea_20260825T0837ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case04_pmax_prefix2_exactnative_exactlib_20260825T0837ET
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
OUT=${CANDIDATE}/analysis/control_vs_exactlib_iter2
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case04_pmax_prefix2_ab.sbatch
STATE_AB=${REPO}/scripts/analyze_k1_parity_state_ab.py
PANEL_BUILDER=${REPO}/scripts/build_k1_pmax_capture_panel.py

test -f "${CONTROL}/SAFE_TO_DELETE" -a -f "${CANDIDATE}/SAFE_TO_DELETE"
test -f "${RUNTIME}/SAFE_TO_DELETE"
test -s "${CONTROL}/parity/iter_002.npz" -a -s "${CANDIDATE}/parity/iter_002.npz"
test -s "${CONTROL}/analysis/prefix_it002/particle_state_arrays.npz"
test -s "${CANDIDATE}/analysis/prefix_it002/particle_state_arrays.npz"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${STATE_AB}" | awk '{print $1}')" = "${EXPECTED_STATE_AB_SHA256}"
test "$(sha256sum "${PANEL_BUILDER}" | awk '{print $1}')" = "${EXPECTED_PANEL_SHA256}"
test ! -e "${OUT}"

mkdir -p "${OUT}" "${RUNTIME}/pmax_ab_tmp" "${RUNTIME}/pmax_ab_pixi_home" \
    "${RUNTIME}/pmax_ab_rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/pmax_ab_tmp
export PIXI_HOME=${RUNTIME}/pmax_ab_pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/pmax_ab_rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.build_k1_pmax_capture_panel \
    --parity-dump "${CONTROL}/parity/iter_002.npz" \
    --input-star "${CASE}/data/particles.star" \
    --relion-star "${CASE}/relion_ref/run_it002_data.star" \
    --top-n 24 --output-json "${OUT}/control_top24.json" >/dev/null
"${PYTHON}" -m scripts.build_k1_pmax_capture_panel \
    --parity-dump "${CANDIDATE}/parity/iter_002.npz" \
    --input-star "${CASE}/data/particles.star" \
    --relion-star "${CASE}/relion_ref/run_it002_data.star" \
    --top-n 24 --output-json "${OUT}/candidate_top24.json" >/dev/null

"${PYTHON}" -m scripts.analyze_k1_autonomous_boundary_movement \
    --baseline-particles "${CONTROL}/analysis/prefix_it002/particle_state_arrays.npz" \
    --candidate-particles "${CANDIDATE}/analysis/prefix_it002/particle_state_arrays.npz" \
    --baseline-fsc "${CONTROL}/analysis/prefix_it002/fsc.json" \
    --candidate-fsc "${CANDIDATE}/analysis/prefix_it002/fsc.json" \
    --particle-iteration 2 --relion-iteration 2 \
    --scope "case 4 iteration-2 older-library control versus exact-interface library" \
    --output-json "${OUT}/population_movement.json" \
    --output-markdown "${OUT}/population_movement.md" >/dev/null

CONTROL_DUMP=${CONTROL}/parity/iter_002.npz \
CANDIDATE_DUMP=${CANDIDATE}/parity/iter_002.npz \
PANEL=${OUT}/candidate_top24.json \
OUTPUT=${OUT}/candidate_top24_state_ab.json \
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from scripts.analyze_k1_parity_state_ab import analyze


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


control = Path(os.environ["CONTROL_DUMP"])
candidate = Path(os.environ["CANDIDATE_DUMP"])
panel_path = Path(os.environ["PANEL"])
output = Path(os.environ["OUTPUT"])
panel = json.loads(panel_path.read_text())
rows = []
for target in panel["rows"]:
    report = analyze(
        control_path=control,
        candidate_path=candidate,
        half=int(target["half"]),
        original_index=int(target["source_row_zero_based"]),
    )
    rows.append(
        {
            "rank_by_absolute_pmax_error": int(target["rank_by_absolute_pmax_error"]),
            "image_name": target["image_name"],
            "relion_pmax": target["relion_pmax"],
            "recovar_pmax": target["recovar_pmax"],
            "state_ab": report,
        }
    )
payload = {
    "schema": "recovar.em.k1_case04_pmax_prefix2_panel_ab.v1",
    "status": "complete",
    "rows": rows,
    "artifacts": {
        "panel": str(panel_path.resolve()),
        "panel_sha256": sha256(panel_path),
        "control_dump": str(control.resolve()),
        "control_dump_sha256": sha256(control),
        "candidate_dump": str(candidate.resolve()),
        "candidate_dump_sha256": sha256(candidate),
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
PY

sha256sum "${LAUNCHER}" "${STATE_AB}" "${PANEL_BUILDER}" \
    "${CONTROL}/parity/iter_002.npz" "${CANDIDATE}/parity/iter_002.npz" \
    "${OUT}/control_top24.json" "${OUT}/candidate_top24.json" \
    "${OUT}/population_movement.json" "${OUT}/candidate_top24_state_ab.json" \
    > "${OUT}/manifest.sha256"
touch "${CANDIDATE}/PREFIX2_AB_SUCCESS_${SLURM_JOB_ID}"
