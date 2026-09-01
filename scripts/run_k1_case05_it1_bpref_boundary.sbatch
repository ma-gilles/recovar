#!/usr/bin/env bash
#SBATCH --job-name=k1c05-it1-bpref
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_boundary_retry_20260822T1515ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_boundary_retry_20260822T1515ET/logs/run-%j.err
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

: "${EXPECTED_REPO_HEAD:?submission must pin repository HEAD}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?submission must pin the RECOVAR source diff}"
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin this launcher}"
: "${EXPECTED_ANALYZER_SHA256:?submission must pin the boundary analyzer}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_boundary_retry_20260822T1515ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case05_it1_bpref_boundary_retry_20260822T1515ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/5_very_high_noise_100k_g256_white_noise10_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
RELION_EXE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_operand_capture_clean_20260810T1330ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
LAUNCHER=${REPO}/scripts/run_k1_case05_it1_bpref_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_half1_raw_accumulator.py
NATIVE_OUT=${ROOT}/native/relion
NATIVE_DUMP=${ROOT}/native/raw_bpref
RECOVAR_OUT=${ROOT}/recovar/output
RECOVAR_ACCUM=${ROOT}/recovar/accum
RECOVAR_PREJOIN=${ROOT}/recovar/prejoin
ANALYSIS=${ROOT}/analysis
PROVENANCE=${ROOT}/provenance

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${RELION_EXE}" | awk '{print $1}')" = 7b989db3b9c1c67ba9752e82803e982b892ffc647ecf5536ae35a9f243cfe72d
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = b5178b902aafaa945bf7271421edab28bd75f0291484006833e0da0ebc91494b
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 078363915af39e644c0540d6e1ad020e41d94c3c7ee298e28cf39d4788cd384c
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = fbb7bbcba5ad804a6d4d78bac87f024f3a876c82f21f9397c5278961b30eda47
test "$(sha256sum "${CONTROL}/run_it000_data.star" | awk '{print $1}')" = 8f7ae3f0b9bb0a5422ea24f71ed28511ccf1900d2d9309cd9e6f5b195d4a6811
test "$(sha256sum "${CONTROL}/run_it000_optimiser.star" | awk '{print $1}')" = 770ac542f4c8ddccfeb80af4b2e1504b17a47ba2c60c09aae3fa0e55b394b067
test "$(stat -Lc %s "${DATA}/particles.256.mrcs")" = 26214401024
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test -z "$(find "${NATIVE_OUT}" "${NATIVE_DUMP}" "${RECOVAR_OUT}" "${RECOVAR_ACCUM}" "${RECOVAR_PREJOIN}" "${ANALYSIS}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${PROVENANCE}/nvidia_smi_${SLURM_JOB_ID}.txt"

# Passive native observer: stop after the first physical iteration and retain
# the post-low-resolution-join numerator and denominator before reconstruction.
export RECOVAR_MSTEP_DUMP_DIR=${NATIVE_DUMP}
export RECOVAR_MSTEP_DUMP_MAX_CALLS=1
export RECOVAR_MSTEP_DUMP_RAW=1
native_command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_EXE}"
    --i particles.star --ref reference_init_relion.mrc --o "${NATIVE_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1705 --auto_iter_max 1
    --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${native_command[@]}" > "${PROVENANCE}/native_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/native_command_${SLURM_JOB_ID}.sh"
native_start=$(date +%s)
(
    cd "${DATA}"
    "${native_command[@]}"
) > "${NATIVE_OUT}/runner.stdout" 2> "${NATIVE_OUT}/runner.stderr"
native_wall_s=$(( $(date +%s) - native_start ))

test -s "${NATIVE_OUT}/run_it001_data.star"
for rank in 01 02; do
    test -s "${NATIVE_DUMP}/raw_bpref_data_rank${rank}_call0000.bin"
    test -s "${NATIVE_DUMP}/raw_bpref_weight_rank${rank}_call0000.bin"
    test -s "${NATIVE_DUMP}/downsampled_avg_rank${rank}_call0000.txt"
done
test -s "${NATIVE_DUMP}/downsampled_fsc_rank01_call0000.txt"
! rg -q 'CUDA_ERROR_OUT_OF_MEMORY|Out of memory|Traceback' "${NATIVE_OUT}/runner.stderr"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${NATIVE_OUT}" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ANALYSIS}/RELION_RAW_BPREF_CAPTURE_INERTNESS.json" \
    --output-markdown "${ANALYSIS}/RELION_RAW_BPREF_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ANALYSIS}/RELION_RAW_BPREF_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

unset RECOVAR_MSTEP_DUMP_DIR RECOVAR_MSTEP_DUMP_MAX_CALLS RECOVAR_MSTEP_DUMP_RAW
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${RECOVAR_ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${RECOVAR_PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case05-it1-selected-treatment-same-h100
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py \
    > "${PROVENANCE}/recovar_source_diff_${SLURM_JOB_ID}.patch"
sha256sum "${RELION_EXE}" "${CUDA_LIB}" "${RELION_BIND_LIB}" "${LAUNCHER}" \
    "${ANALYZER}" "${DATA}/particles.star" "${DATA}/reference_init_relion.mrc" \
    "${CONTROL}/run_it000_data.star" "${CONTROL}/run_it000_optimiser.star" \
    > "${PROVENANCE}/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(recovar.__file__)
print(jax.__file__)
print(jax.devices())
PY

recovar_command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${RECOVAR_OUT}" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1705 --perturb_seed 1705
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${RECOVAR_OUT}/benchmark_ledger.json"
    --timing_dir "${RECOVAR_OUT}/timing"
    --save_intermediates_dir "${RECOVAR_OUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${CONTROL}/run_it000_data.star"
    --relion_optimiser "${CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${CONTROL}")
printf '%q ' "${recovar_command[@]}" > "${PROVENANCE}/recovar_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/recovar_command_${SLURM_JOB_ID}.sh"
recovar_start=$(date +%s)
"${recovar_command[@]}" > "${RECOVAR_OUT}/run.log" 2>&1
recovar_wall_s=$(( $(date +%s) - recovar_start ))

rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1706' "${RECOVAR_OUT}/run.log"
rg -q 'preserving fresh RELION physical BPref particle order' "${RECOVAR_OUT}/run.log"
test -s "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz"
test -s "${RECOVAR_PREJOIN}/recovar_bpref_prejoin_it001.npz"
test -s "${RECOVAR_OUT}/intermediates/it000_fsc.npy"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${RECOVAR_OUT}/run.log"

"${PYTHON}" "${ANALYZER}" \
    --recovar-bpref "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz" \
    --native-data "${NATIVE_DUMP}/raw_bpref_data_rank01_call0000.bin" \
    --native-weight "${NATIVE_DUMP}/raw_bpref_weight_rank01_call0000.bin" \
    --native-data-half2 "${NATIVE_DUMP}/raw_bpref_data_rank02_call0000.bin" \
    --native-weight-half2 "${NATIVE_DUMP}/raw_bpref_weight_rank02_call0000.bin" \
    --recovar-fsc "${RECOVAR_OUT}/intermediates/it000_fsc.npy" \
    --half 1 --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${ANALYSIS}/K1_CASE05_IT1_BPREF_HALF1_AND_FSC.json"
"${PYTHON}" "${ANALYZER}" \
    --recovar-bpref "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz" \
    --native-data "${NATIVE_DUMP}/raw_bpref_data_rank02_call0000.bin" \
    --native-weight "${NATIVE_DUMP}/raw_bpref_weight_rank02_call0000.bin" \
    --half 2 --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
    --output-json "${ANALYSIS}/K1_CASE05_IT1_BPREF_HALF2.json"

"${PYTHON}" - \
    "${ANALYSIS}/K1_CASE05_IT1_BPREF_HALF1_AND_FSC.json" \
    "${ANALYSIS}/K1_CASE05_IT1_BPREF_HALF2.json" \
    "${ANALYSIS}/K1_CASE05_IT1_BPREF_SUMMARY.json" <<'PY'
import json
import sys

half1 = json.load(open(sys.argv[1]))
half2 = json.load(open(sys.argv[2]))
summary = {
    "schema": "recovar.em.k1_case05_it1_bpref_summary.v1",
    "first_unequal_boundary": "post_lowres_join_bpref_accumulator",
    "half1": {
        "raw_numerator_relative_l2": half1["raw_accumulator"]["numerator"]["relative_l2"],
        "raw_denominator_relative_l2": half1["raw_accumulator"]["denominator"]["relative_l2"],
        "downsampled_numerator_relative_l2": half1["recovar_vs_native"]["numerator"]["relative_l2"],
        "downsampled_denominator_relative_l2": half1["recovar_vs_native"]["denominator"]["relative_l2"],
    },
    "half2": {
        "raw_numerator_relative_l2": half2["raw_accumulator"]["numerator"]["relative_l2"],
        "raw_denominator_relative_l2": half2["raw_accumulator"]["denominator"]["relative_l2"],
        "downsampled_numerator_relative_l2": half2["recovar_vs_native"]["numerator"]["relative_l2"],
        "downsampled_denominator_relative_l2": half2["recovar_vs_native"]["denominator"]["relative_l2"],
    },
    "two_half_fsc_relative_l2": half1["two_half_fsc"]["metrics"]["relative_l2"],
    "two_half_fsc_replay_vs_production_relative_l2": half1["two_half_fsc"]["recovar_replay_vs_production"]["relative_l2"],
}
with open(sys.argv[3], "w") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

find "${NATIVE_OUT}" "${NATIVE_DUMP}" "${RECOVAR_OUT}" "${RECOVAR_ACCUM}" \
    "${RECOVAR_PREJOIN}" "${ANALYSIS}" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","native_wall_s":%d,"recovar_wall_s":%d,"physical_iteration":1}\n' \
    "${SLURM_JOB_ID}" "${native_wall_s}" "${recovar_wall_s}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
