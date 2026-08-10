#!/usr/bin/env bash
#SBATCH --job-name=k1c22-fine-op
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET/logs/run-%j.err
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
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin the launcher hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
LAUNCHER=${REPO}/scripts/run_k1_case22_fine_operand_capture.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
SEALED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CAPTURE_DIR=${ROOT}/capture
RELION_OUT=${ROOT}/relion

EXPECTED_REPO_HEAD=ea8f2b21bd3041c71ac747a9d4b7aee96882a61e
EXPECTED_AUDIT_HEAD=187a9f60cdea19363d49d46099b052f23288f05a
EXPECTED_RELION_HEAD=f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
EXPECTED_RELION_TREE=3111173edb150f98bc2446efc54229ff736b40a1
EXPECTED_RELION_BINARY_SHA256=1739fb94d8067210a4094d858cb74780268a106a2484ace00d55fb1a1af66b0f
CAPTURE_STACKS=252,591,2124

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha "${EXPECTED_RELION_BINARY_SHA256}" "${RELION_BINARY}"
require_sha 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6 "${DATA}/particles.star"
require_sha 804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9 "${DATA}/particles.128.mrcs"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = "${EXPECTED_AUDIT_HEAD}"
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = "${EXPECTED_RELION_TREE}"
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1)"
test -x "${PYTHON}"
test -z "$(find "${CAPTURE_DIR}" -mindepth 1 -print -quit)"
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"

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
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -eq 1

export RELION_BPRE_CAPTURE_DIR=${CAPTURE_DIR}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=2
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=3
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=3
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${CAPTURE_STACKS}
export RELION_FINE_OPERAND_CAPTURE=1
export RELION_FINE_OPERAND_CAPTURE_ROTATION_LOCAL=0
export RELION_FINE_OPERAND_CAPTURE_TRANSLATIONS=56
export RELION_FINE_OPERAND_CAPTURE_EXPECTED_CANDIDATES=1
export RELION_FINE_OPERAND_CAPTURE_MAX_BYTES=1000000

{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "relion_head=${EXPECTED_RELION_HEAD}"
    echo "capture_stacks=${CAPTURE_STACKS}"
    echo "capture_rotation_local=0"
    echo "capture_translation=56"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" \
    "${REPO}/scripts/validate_relion_fine_operand_capture.py" \
    "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_start_${SLURM_JOB_ID}.txt"

command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star
    --ref reference_init_relion.mrc
    --o "${RELION_OUT}/run"
    --auto_refine
    --split_random_halves
    --particle_diameter 200
    --ini_high 30
    --firstiter_cc
    --ctf
    --flatten_solvent
    --zero_mask
    --low_resol_join_halves 40
    --norm
    --scale
    --healpix_order 3
    --offset_range 3.0
    --offset_step 1.0
    --oversampling 1
    --pad 2
    --random_seed 1722
    --auto_iter_max 2
    --pool 3
    --gpu 0:0:0
    --j 4
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
end_epoch=$(date +%s)
printf '{"wall_s":%d,"gpu_uuid":"%s"}\n' "$((end_epoch - start_epoch))" "${GPU_UUID}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"

for suffix in data.star optimiser.star sampling.star half1_model.star half2_model.star half1_class001.mrc half2_class001.mrc
do
    test -s "${RELION_OUT}/run_it002_${suffix}"
done
test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.fine-operand-v1.bin' | wc -l)" -eq 3
test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq 3
test -z "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"

for stack in 252 591 2124; do
    capture=$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name "*_stack${stack}_class1.fine-operand-v1.bin")
    test -n "${capture}"
    "${PYTHON}" "${REPO}/scripts/validate_relion_fine_operand_capture.py" \
        "${capture}" \
        --expected-stack "${stack}" \
        --expected-class 1 \
        --expected-rotation-local 0 \
        --expected-translations 56 \
        --output "${ROOT}/analysis/FINE_OPERAND_VALIDATION_STACK${stack}.json" \
        > "${ROOT}/logs/validate_stack${stack}_${SLURM_JOB_ID}.stdout" \
        2> "${ROOT}/logs/validate_stack${stack}_${SLURM_JOB_ID}.stderr"
done

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${SEALED_CONTROL}" \
    --candidate-dir "${RELION_OUT}" \
    --iterations 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_FINE_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_FINE_CAPTURE_INERTNESS.md" \
    > "${ROOT}/logs/capture_inertness_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/capture_inertness_${SLURM_JOB_ID}.stderr"

"${PYTHON}" - "${ROOT}/analysis/RELION_FINE_CAPTURE_INERTNESS.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
iteration = report["iterations"][0]
assert iteration["iteration"] == 2
assert iteration["map_stable"]
assert iteration["selected_topology_exact"]
assert min(item["fsc_auc"] for item in iteration["maps"].values()) >= 0.999999
PY

find "${CAPTURE_DIR}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/capture_artifacts_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_end_${SLURM_JOB_ID}.txt"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
