#!/usr/bin/env bash
#SBATCH --job-name=k1c26-preproc
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_preprocess_20260810T2141ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_preprocess_20260810T2141ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"

ROOT=${RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_preprocess_20260810T2141ET}
RUNTIME=${RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_native_preprocess_20260810T2141ET}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_preprocess_capture.sbatch
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case22_operand_capture_build_ac12ca86_20260729T234000ET
RELION_BINARY=${BUILD_ROOT}/build_cuda126_mpi416/bin/relion_refine_mpi
BUILD_PROVENANCE=${BUILD_ROOT}/provenance/BUILD_PROVENANCE.md
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
CAPTURE_DIR=${ROOT}/capture
RELION_OUT=${ROOT}/relion
CAPTURE_PART_IDS=${CAPTURE_PART_IDS:-634}
CAPTURE_STACK_INDICES=${CAPTURE_STACK_INDICES:-67}
CAPTURE_EXPECTED_PARTICLES=${CAPTURE_EXPECTED_PARTICLES:-1}
CAPTURE_MAX_PARTICLES_PER_RANK=${CAPTURE_MAX_PARTICLES_PER_RANK:-1}
CAPTURE_MAX_BYTES=${CAPTURE_MAX_BYTES:-1000000}
CAPTURE_ITER=${CAPTURE_ITER:-2}

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha 982c15cfcdce94823c471228edef47839fa7d239ccac166c4ba66c829cd1f6ba "${RELION_BINARY}"
require_sha 18ed78327726a52f419a830d018427968d5afb5ece4c00319ace47ffbb7fa9fd "${BUILD_PROVENANCE}"
require_sha 775da1ea8fb8bad8d69ac296f150adbe21750d67b8832db8aaf699048665f673 "${REPO}/scripts/validate_relion_preprocess_capture.py"
require_sha 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139 "${DATA}/particles.star"
require_sha 4cd77e4ddeb731f719adb0cfd661e4f00221e07d08e15e1a5dc229737d9c848c "${DATA}/particles.128.mrcs"
require_sha 7aafc4f04e482f7410ff37686ba46db14578334ca77909f04bff30dbba2789d1 "${DATA}/reference_init_relion.mrc"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -z "$(find "${CAPTURE_DIR}" -mindepth 1 -print -quit)"
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
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
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RELION_P1_PREPROCESS_CAPTURE_DIR=${CAPTURE_DIR}
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=${CAPTURE_ITER}
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=${CAPTURE_EXPECTED_PARTICLES}
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=${CAPTURE_MAX_PARTICLES_PER_RANK}
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=${CAPTURE_MAX_BYTES}
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=${CAPTURE_PART_IDS}

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "capture_part_ids=${CAPTURE_PART_IDS}"
    echo "capture_stack_indices=${CAPTURE_STACK_INDICES}"
    echo "capture_expected_particles=${CAPTURE_EXPECTED_PARTICLES}"
    echo "capture_iteration=${CAPTURE_ITER}"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${BUILD_PROVENANCE}" \
        "${REPO}/scripts/validate_relion_preprocess_capture.py" \
        "${DATA}/particles.star" "${DATA}/particles.128.mrcs" \
        "${DATA}/reference_init_relion.mrc"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.preprocess-v1.bin' | wc -l)" -eq "${CAPTURE_EXPECTED_PARTICLES}"
test -z "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"
"${PYTHON}" "${REPO}/scripts/validate_relion_preprocess_capture.py" \
    "${CAPTURE_DIR}" --expected-particles "${CAPTURE_EXPECTED_PARTICLES}" \
    --expected-part-ids "${CAPTURE_PART_IDS}" \
    --expected-stack-indices "${CAPTURE_STACK_INDICES}" --expected-iteration "${CAPTURE_ITER}" \
    --output "${ROOT}/analysis/RELION_PREPROCESS_CAPTURE_V1.json"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - <<'PY' "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json"
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

sha256sum "${CAPTURE_DIR}"/*.preprocess-v1.bin \
    "${ROOT}/analysis/RELION_PREPROCESS_CAPTURE_V1.json" \
    "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"particle_count":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_EXPECTED_PARTICLES}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
