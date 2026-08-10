#!/usr/bin/env bash
#SBATCH --job-name=k1c22-it1-boundary
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:30:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it1_first_boundary_20260810T0945ET
DOCS_REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
FACTOR_REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_bpref_factor_capture_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default/bin/python
PIXI_ENV=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default
LAUNCHER=${DOCS_REPO}/scripts/run_k1_case22_iteration1_boundary.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
SEALED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_control/relion
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
SELECTION=${ROOT}/provenance/SELECTION.json

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha 1739fb94d8067210a4094d858cb74780268a106a2484ace00d55fb1a1af66b0f "${RELION_BINARY}"
require_sha 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569 "${CUDA_LIB}"
require_sha 8b5dcd9d5147707c6c04c192eee4bd7c3374e26c696ab4a8c8ce78c89c451604 "${SELECTION}"
require_sha 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6 "${DATA}/particles.star"
require_sha 804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9 "${DATA}/particles.128.mrcs"
test "$(git -C "${FACTOR_REPO}" rev-parse HEAD)" = 0dbd53b779d4c57226417f011d71a8f68ee98a0a
test "$(git -C "${FACTOR_REPO}" rev-parse HEAD^{tree})" = a95422aa073bddde31f730c90abe353513acf0ff
test -z "$(git -C "${FACTOR_REPO}" status --porcelain=v1)"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = 3111173edb150f98bc2446efc54229ff736b40a1
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1)"
for directory in "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/recovar/output" "${ROOT}/recovar/pass2" "${ROOT}/analysis"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
RELION_LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${RELION_LD_LIBRARY_PATH}
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

export RELION_BPRE_CAPTURE_DIR=${ROOT}/capture
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=1
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=3
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=3
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=252,591,2124

{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "factor_repo_head=$(git -C "${FACTOR_REPO}" rev-parse HEAD)"
    echo "physical_iteration=1"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

relion_command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${ROOT}/relion/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 1 --pool 3
    --gpu 0:0:0 --j 4
)
printf '%q ' "${relion_command[@]}" > "${ROOT}/provenance/command_relion_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_relion_${SLURM_JOB_ID}.sh"
relion_start=$(date +%s)
(
    cd "${DATA}"
    "${relion_command[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
printf '{"wall_s":%d}\n' "$(( $(date +%s) - relion_start ))" > "${ROOT}/provenance/walltime_relion_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/capture" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq 3
test -s "${ROOT}/relion/run_it001_data.star" -a -s "${ROOT}/relion/run_it001_half1_model.star"

cd "${DOCS_REPO}"
"${PYTHON}" scripts/validate_relion_bpref_factor_capture.py \
    "${ROOT}/capture" --selection-json "${SELECTION}" \
    --output-json "${ROOT}/analysis/RELION_IT1_FACTOR_VALIDATION.json" \
    > "${ROOT}/logs/validate_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/validate_factor_${SLURM_JOB_ID}.stderr"

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${SEALED_CONTROL}" --candidate-dir "${ROOT}/relion" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_IT1_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_IT1_CAPTURE_INERTNESS.md" \
    > "${ROOT}/logs/capture_inertness_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/capture_inertness_${SLURM_JOB_ID}.stderr"

"${PYTHON}" - "${ROOT}/analysis/RELION_IT1_CAPTURE_INERTNESS.json" \
    "${ROOT}/analysis/RELION_IT1_FACTOR_CAPTURE_INERTNESS.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
report = json.loads(source.read_text())
row = report["iterations"][0]
minimum_auc = min(product["fsc_auc"] for product in row["maps"].values())
qualified = bool(
    row["iteration"] == 1
    and row["map_stable"]
    and row["selected_topology_exact"]
    and minimum_auc >= 0.999999
)
result = {
    "schema": "recovar.em.k1_bpref_factor_capture_inertness.v1",
    "status": "qualified" if qualified else "rejected",
    "qualified": qualified,
    "case_id": 22,
    "physical_iteration": 1,
    "target_stack_indices_one_based": [252, 591, 2124],
    "minimum_signed_normalized_non_dc_fsc_auc": minimum_auc,
    "selected_topology_exact": bool(row["selected_topology_exact"]),
    "full_inertness_report": str(source.resolve()),
    "metric_policy": "signed shellwise FSC/FSC-AUC; exact topology; no correlation",
}
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
if not qualified:
    raise SystemExit("iteration-1 capture inertness did not qualify")
PY

set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${RELION_LD_LIBRARY_PATH}
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*) unset "${variable_name}" ;; esac
done < <(env)
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/cuda_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${FACTOR_REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_PASS2_DUMP_DIR=${ROOT}/recovar/pass2
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=251,590,2123
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=56
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${FACTOR_REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject
root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

recovar_start=$(date +%s)
"${PYTHON}" -m scripts.run_full_refinement \
    --data_dir "${DATA}" --output "${ROOT}/recovar/output" --max_iter 1 \
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0 \
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187 \
    --rotation_block_size 8192 --seed 1722 --perturb_seed 1722 \
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1 \
    --firstiter_cc --apply-initial-lowpass \
    --benchmark_ledger_json "${ROOT}/recovar/output/benchmark_ledger.json" \
    --timing_dir "${ROOT}/recovar/output/timing" \
    --save_intermediates_dir "${ROOT}/recovar/output/intermediates" \
    --save_intermediates_skip_unregularized \
    --relion_half_sets "${SEALED_CONTROL}/run_it000_data.star" \
    --relion_optimiser "${SEALED_CONTROL}/run_it000_optimiser.star" \
    --relion_init_dir "${SEALED_CONTROL}" \
    > "${ROOT}/recovar/output/run.log" 2>&1
printf '{"wall_s":%d}\n' "$(( $(date +%s) - recovar_start ))" > "${ROOT}/provenance/walltime_recovar_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq 3
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/recovar/output/run.log"

cd "${DOCS_REPO}"
"${PYTHON}" scripts/analyze_k1_bpref_factor_boundary.py \
    --factor-directory "${ROOT}/capture" \
    --recovar-capture-directory "${ROOT}/recovar/pass2" \
    --selection-json "${SELECTION}" \
    --capture-inertness-json "${ROOT}/analysis/RELION_IT1_FACTOR_CAPTURE_INERTNESS.json" \
    --output-json "${ROOT}/analysis/CASE22_IT1_BPREF_FACTOR_BOUNDARY.json" \
    > "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stderr"

find "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/recovar" "${ROOT}/analysis" \
    -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
