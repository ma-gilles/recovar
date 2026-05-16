#!/usr/bin/env bash
# Chained K-class parity replay: runs N single-step replays
# (prev=i, target=i+1) and writes a unified summary.json with
# per-iter mean_corr, per-class corrs, |ΔPmax|.
#
# Usage:
#   qparity_kclass_chain.sh <K> <relion_dir> <data_star> <n_iters> <out_dir> <af> <firstiter_cc>

set -euo pipefail

K="$1"
RELION_DIR="$2"
DATA_STAR="$3"
N_ITERS="$4"
OUT_DIR="$5"
AF="$6"
FIRSTITER_CC="$7"

REPO_ROOT=/home/mg6942/myscratch/recovar_wt_qparity_20260502_183314_5307
mkdir -p "${OUT_DIR}"

BASE_ARGS=(--adaptive-2pass --significance-adaptive-fraction "${AF}")

PER_ITER_JSON="${OUT_DIR}/per_iter.json"
echo "[" > "${PER_ITER_JSON}"
SEP=""

for i in $(seq 0 $(($N_ITERS - 1))); do
  TARGET=$((i + 1))
  STEP_DIR="${OUT_DIR}/step_${i}_to_${TARGET}"
  mkdir -p "${STEP_DIR}"
  STEP_ARGS=("${BASE_ARGS[@]}")
  # RELION's --firstiter_cc winner-take-all only applies to iter 1 (step 0->1).
  # For step 1->2 onwards RELION uses soft Gaussian scoring, so recovar must too.
  if [[ "${FIRSTITER_CC}" == "yes" && "${i}" -eq 0 ]]; then
    STEP_ARGS+=(--winner-take-all-mstep)
  fi
  echo "=== chained step ${i} -> ${TARGET} (extra: ${STEP_ARGS[*]}) ==="
  pixi run python "${REPO_ROOT}/scripts/run_k_class_parity.py" \
    --relion-dir "${RELION_DIR}" \
    --data-star "${DATA_STAR}" \
    --prev-iter "${i}" \
    --target-iter "${TARGET}" \
    --output-dir "${STEP_DIR}" \
    "${STEP_ARGS[@]}"
  STEP_SUMMARY="${STEP_DIR}/summary.json"
  if [[ ! -f "${STEP_SUMMARY}" ]]; then
    echo "ERROR: missing ${STEP_SUMMARY}" >&2
    exit 1
  fi
  echo "${SEP}" >> "${PER_ITER_JSON}"
  python -c "import json,sys; d=json.load(open('${STEP_SUMMARY}')); out={'iter':${TARGET},'mean_corr':d['best_permutation']['mean_corr'],'per_class_corr':d['best_permutation']['map_correlations'],'pmax_abs_mean':d['pmax']['abs_mean'],'pmax_abs_max':d['pmax']['abs_max'],'class_assignment_accuracy':d['class_assignment_accuracy_after_permutation']}; sys.stdout.write(json.dumps(out, indent=2))" >> "${PER_ITER_JSON}"
  SEP=","
done
echo "]" >> "${PER_ITER_JSON}"
echo "Wrote ${PER_ITER_JSON}"
