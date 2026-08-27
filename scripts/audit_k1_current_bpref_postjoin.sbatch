#!/usr/bin/env bash
#SBATCH --job-name=k1-bpref-postjoin
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RECOVAR_INTERMEDIATES:?RECOVAR_INTERMEDIATES is required}"
: "${NATIVE_ROOT:?NATIVE_ROOT is required}"
: "${NATIVE_REPEAT_ROOT:?NATIVE_REPEAT_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_current_bpref_postjoin.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_half1_raw_accumulator.py
ANALYSIS=${RUN_ROOT}/analysis
ARCHIVE=${RUN_ROOT}/recovar_bpref_postjoin_it001.npz

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
for field in Ft_y_0 Ft_y_1 Ft_ctf_0 Ft_ctf_1 fsc; do
    test -s "${RECOVAR_INTERMEDIATES}/it000_${field}.npy"
done
for root in "${NATIVE_ROOT}" "${NATIVE_REPEAT_ROOT}"; do
    for half_rank in 1:1 2:2; do
        half=${half_rank%%:*}
        rank=${half_rank##*:}
        test -s "${root}/mstep_it001_rank${rank}_half${half}_c0_post_lowres_join_bpref_data.bin"
        test -s "${root}/mstep_it001_rank${rank}_half${half}_c0_post_lowres_join_bpref_weight.bin"
    done
done
test ! -e "${ARCHIVE}" -a ! -e "${ANALYSIS}"

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache

cd "${REPO}"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
"${PYTHON}" - "${RECOVAR_INTERMEDIATES}" "${ARCHIVE}" <<'PY'
import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
output = Path(sys.argv[2])
arrays = {
    field: np.load(source / f"it000_{field}.npy", allow_pickle=False)
    for field in ("Ft_y_0", "Ft_y_1", "Ft_ctf_0", "Ft_ctf_1")
}
shape = round(arrays["Ft_y_0"].size ** (1.0 / 3.0))
assert shape**3 == arrays["Ft_y_0"].size
np.savez(
    output,
    schema=np.asarray("recovar-bpref-accum-v2"),
    run_id=np.asarray("case10-current-b39-terminal-it1"),
    iteration=np.asarray(1, dtype=np.int32),
    current_size=np.asarray(56, dtype=np.int32),
    padding_factor=np.asarray(2, dtype=np.int32),
    grid_size=np.asarray(384, dtype=np.int32),
    voxel_size=np.asarray(1.4166666666666667, dtype=np.float64),
    volume_shape=np.asarray([384, 384, 384], dtype=np.int32),
    mstep_accumulator_shape=np.asarray([shape, shape, shape], dtype=np.int32),
    **arrays,
)
PY

native_data_h1=${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_data.bin
native_weight_h1=${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_weight.bin
native_data_h2=${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_data.bin
native_weight_h2=${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_weight.bin
repeat_data_h1=${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_data.bin
repeat_weight_h1=${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_weight.bin
repeat_data_h2=${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_data.bin
repeat_weight_h2=${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_weight.bin

"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${ARCHIVE}" \
    --native-data "${native_data_h1}" --native-weight "${native_weight_h1}" \
    --native-repeat-data "${repeat_data_h1}" --native-repeat-weight "${repeat_weight_h1}" \
    --native-data-half2 "${native_data_h2}" --native-weight-half2 "${native_weight_h2}" \
    --recovar-fsc "${RECOVAR_INTERMEDIATES}/it000_fsc.npy" \
    --native-stage post_lowres_join --half 1 \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${ANALYSIS}/POSTJOIN_HALF1.json"

"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${ARCHIVE}" \
    --native-data "${native_data_h2}" --native-weight "${native_weight_h2}" \
    --native-repeat-data "${repeat_data_h2}" --native-repeat-weight "${repeat_weight_h2}" \
    --native-stage post_lowres_join --half 2 \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${ANALYSIS}/POSTJOIN_HALF2.json"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${ARCHIVE}" \
    "${ANALYSIS}/POSTJOIN_HALF1.json" "${ANALYSIS}/POSTJOIN_HALF2.json" \
    "${RECOVAR_INTERMEDIATES}"/it000_{Ft_y_0,Ft_y_1,Ft_ctf_0,Ft_ctf_1,fsc}.npy \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
sha256sum "${NATIVE_ROOT}"/*.bin "${NATIVE_REPEAT_ROOT}"/*.bin \
    > "${RUN_ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
