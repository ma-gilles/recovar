#!/usr/bin/env bash
#SBATCH --job-name=k1c10-bpref-cmp
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
RECOVAR_ROOT=${RECOVAR_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_bpref_boundary_output_scale_4a3a34bd7_20260824T0400ET/recovar}
RECOVAR_REPEAT_ROOT=${RECOVAR_REPEAT_ROOT:-}
NATIVE_ROOT=${NATIVE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_native_it1_bpref_accum_20260824T0415ET/bpref_accum}
NATIVE_REPEAT_ROOT=${NATIVE_REPEAT_ROOT:-}
PYTHON=${REPO}/.pixi/envs/default/bin/python
OUT=${RUN_ROOT}/analysis

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${RECOVAR_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz"
test -s "${RECOVAR_ROOT}/accum/recovar_bpref_accum_it001.npz"
if [[ -n "${RECOVAR_REPEAT_ROOT}" ]]; then
    test -s "${RECOVAR_REPEAT_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz"
    test -s "${RECOVAR_REPEAT_ROOT}/accum/recovar_bpref_accum_it001.npz"
fi
for half_rank in 1:1 2:2; do
    half=${half_rank%%:*}
    rank=${half_rank##*:}
    for stage in pre post; do
        for kind in data weight; do
            test -s "${NATIVE_ROOT}/mstep_it001_rank${rank}_half${half}_c0_${stage}_lowres_join_bpref_${kind}.bin"
        done
    done
done
if [[ -n "${NATIVE_REPEAT_ROOT}" ]]; then
    for half_rank in 1:1 2:2; do
        half=${half_rank%%:*}
        rank=${half_rank##*:}
        for stage in pre post; do
            for kind in data weight; do
                test -s "${NATIVE_REPEAT_ROOT}/mstep_it001_rank${rank}_half${half}_c0_${stage}_lowres_join_bpref_${kind}.bin"
            done
        done
    done
fi

mkdir -p "${OUT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache"
test -z "$(find "${OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache

cd "${REPO}"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${RECOVAR_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz" \
    "${RECOVAR_ROOT}/accum/recovar_bpref_accum_it001.npz" \
    "${NATIVE_ROOT}"/*.bin \
    > "${RUN_ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"
if [[ -n "${RECOVAR_REPEAT_ROOT}" ]]; then
    sha256sum "${RECOVAR_REPEAT_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz" \
        "${RECOVAR_REPEAT_ROOT}/accum/recovar_bpref_accum_it001.npz" \
        >> "${RUN_ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"
fi
if [[ -n "${NATIVE_REPEAT_ROOT}" ]]; then
    sha256sum "${NATIVE_REPEAT_ROOT}"/*.bin \
        >> "${RUN_ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"
fi

pre_h1_repeat_args=()
pre_h2_repeat_args=()
post_h1_repeat_args=()
post_h2_repeat_args=()
recovar_pre_repeat_args=()
recovar_post_repeat_args=()
if [[ -n "${RECOVAR_REPEAT_ROOT}" ]]; then
    recovar_pre_repeat_args=(
        --recovar-repeat "${RECOVAR_REPEAT_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz"
    )
    recovar_post_repeat_args=(
        --recovar-repeat "${RECOVAR_REPEAT_ROOT}/accum/recovar_bpref_accum_it001.npz"
    )
fi
if [[ -n "${NATIVE_REPEAT_ROOT}" ]]; then
    pre_h1_repeat_args=(
        --native-repeat-data "${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_pre_lowres_join_bpref_data.bin"
        --native-repeat-weight "${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_pre_lowres_join_bpref_weight.bin"
    )
    pre_h2_repeat_args=(
        --native-repeat-data "${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_data.bin"
        --native-repeat-weight "${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_weight.bin"
    )
    post_h1_repeat_args=(
        --native-repeat-data "${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_data.bin"
        --native-repeat-weight "${NATIVE_REPEAT_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_weight.bin"
    )
    post_h2_repeat_args=(
        --native-repeat-data "${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_data.bin"
        --native-repeat-weight "${NATIVE_REPEAT_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_weight.bin"
    )
fi

"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${RECOVAR_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz" \
    --native-data "${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_pre_lowres_join_bpref_data.bin" \
    --native-weight "${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_pre_lowres_join_bpref_weight.bin" \
    --native-data-half2 "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_data.bin" \
    --native-weight-half2 "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_weight.bin" \
    --native-stage pre_lowres_join --half 1 \
    "${recovar_pre_repeat_args[@]}" \
    "${pre_h1_repeat_args[@]}" \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${OUT}/PREJOIN_HALF1.json"
"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${RECOVAR_ROOT}/prejoin/recovar_bpref_prejoin_it001.npz" \
    --native-data "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_data.bin" \
    --native-weight "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_pre_lowres_join_bpref_weight.bin" \
    --native-stage pre_lowres_join --half 2 \
    "${recovar_pre_repeat_args[@]}" \
    "${pre_h2_repeat_args[@]}" \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${OUT}/PREJOIN_HALF2.json"
"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${RECOVAR_ROOT}/accum/recovar_bpref_accum_it001.npz" \
    --native-data "${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_data.bin" \
    --native-weight "${NATIVE_ROOT}/mstep_it001_rank1_half1_c0_post_lowres_join_bpref_weight.bin" \
    --native-data-half2 "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_data.bin" \
    --native-weight-half2 "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_weight.bin" \
    --recovar-fsc "${RECOVAR_ROOT}/output/intermediates/it000_fsc.npy" \
    --native-stage post_lowres_join --half 1 \
    "${recovar_post_repeat_args[@]}" \
    "${post_h1_repeat_args[@]}" \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${OUT}/POSTJOIN_HALF1.json"
"${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
    --recovar-bpref "${RECOVAR_ROOT}/accum/recovar_bpref_accum_it001.npz" \
    --native-data "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_data.bin" \
    --native-weight "${NATIVE_ROOT}/mstep_it001_rank2_half2_c0_post_lowres_join_bpref_weight.bin" \
    --native-stage post_lowres_join --half 2 \
    "${recovar_post_repeat_args[@]}" \
    "${post_h2_repeat_args[@]}" \
    --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${OUT}/POSTJOIN_HALF2.json"

sha256sum "${OUT}"/*.json > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
