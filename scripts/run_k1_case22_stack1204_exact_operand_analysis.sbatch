#!/usr/bin/env bash
#SBATCH --job-name=k1c22-op-analyze
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00

set -euo pipefail
: "${K1_CASE22_OPERAND_ANALYSIS_ROOT:?set the disposable analysis root}"
: "${K1_CASE22_OPERAND_ANALYSIS_RUNTIME:?set the disposable runtime root}"
: "${K1_CASE22_OPERAND_CAPTURE_A:?set the rotation-145621 capture root}"
: "${K1_CASE22_OPERAND_CAPTURE_B:?set the rotation-145623 capture root}"
: "${K1_CASE22_RECOVAR_DUMP:?set the uninterrupted RECOVAR pass-2 dump}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${K1_CASE22_OPERAND_ANALYSIS_ROOT}
RUNTIME=${K1_CASE22_OPERAND_ANALYSIS_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case22_stack1204_exact_operand_analysis.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_exact_ppref_operand_tuple.py
RECOVAR=${K1_CASE22_RECOVAR_DUMP}
CAPTURE_A=${K1_CASE22_OPERAND_CAPTURE_A}
CAPTURE_B=${K1_CASE22_OPERAND_CAPTURE_B}
RECOVAR_ROW_A=${K1_CASE22_OPERAND_RECOVAR_ROW_A:-50293}
RECOVAR_ROW_B=${K1_CASE22_OPERAND_RECOVAR_ROW_B:-50295}
LABEL_A=${K1_CASE22_OPERAND_LABEL_A:-rank1_native_r145621_t80_recovar_r50293_t80}
LABEL_B=${K1_CASE22_OPERAND_LABEL_B:-rank2_native_r145623_t82_recovar_r50295_t82}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -f "${CAPTURE_A}"/SUCCESS_* -a -f "${CAPTURE_B}"/SUCCESS_*
test -s "${RECOVAR}"
test -x "${PYTHON}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${ROOT}/analysis" "${ROOT}/provenance"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
"${PYTHON}" - <<'PY'
import jax
assert jax.default_backend() == "gpu", jax.devices()
print(jax.devices())
PY

run_one() {
    local capture_root=$1
    local recovar_rotation_row=$2
    local label=$3
    local ppref=${capture_root}/capture/ppref_iter002_rank001_model000.bin
    local operand fine_score
    operand=$(find "${capture_root}/capture" -maxdepth 1 -name '*_stack1204_class1.fine-operand-v1.bin')
    fine_score=$(find "${capture_root}/capture" -maxdepth 1 -name '*_stack1204_class1.fine-score-v1.bin')
    test -s "${ppref}" -a -s "${operand}" -a -s "${fine_score}"
    "${PYTHON}" "${ANALYZER}" \
        --ppref "${ppref}" \
        --capture "${operand}" \
        --recovar "${RECOVAR}" \
        --recovar-rotation-row "${recovar_rotation_row}" \
        --fine-score "${fine_score}" \
        --physical-image-size 128 \
        --output-json "${ROOT}/analysis/${label}.json" \
        > "${ROOT}/analysis/${label}.stdout"
}

run_one "${CAPTURE_A}" "${RECOVAR_ROW_A}" "${LABEL_A}"
run_one "${CAPTURE_B}" "${RECOVAR_ROW_B}" "${LABEL_B}"

"${PYTHON}" - <<'PY' "${ROOT}/analysis"
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
reports = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*.json"))]
boundary_a, boundary_b = (report["posterior_boundary"] for report in reports)
pairwise = {}
for field in (
    "raw_diff2",
    "orientation_log_prior",
    "translation_log_prior",
    "combined_preexponent",
    "normalized_posterior",
):
    native_gap = boundary_b["native"][field] - boundary_a["native"][field]
    recovar_gap = boundary_b["recovar"][field] - boundary_a["recovar"][field]
    pairwise[field] = {
        "native_b_minus_a": native_gap,
        "recovar_b_minus_a": recovar_gap,
        "exact_equal": native_gap == recovar_gap,
        "delta_recovar_minus_native": recovar_gap - native_gap,
    }
summary = {
    "schema": "recovar.em.k1_case22_stack1204_exact_operand_pair.v1",
    "status": "complete",
    "candidates": [
        {
            "identity": report["identity"],
            "first_exact_unequal_boundary": report["first_exact_unequal_boundary"],
            "stage_metrics": report["stage_metrics"],
            "raw_scores": report["raw_scores"],
            "posterior_boundary": report["posterior_boundary"],
        }
        for report in reports
    ],
    "candidate_b_minus_a": pairwise,
}
(directory / "K1_CASE22_STACK1204_EXACT_OPERAND_PAIR.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

sha256sum "${LAUNCHER}" "${ANALYZER}" "${RECOVAR}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
find "${ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/analysis_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
