#!/usr/bin/env bash
#SBATCH --job-name=k1-native-verbose
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail
trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${CONTROL_DIR:?CONTROL_DIR is required}"
: "${TARGET_SOURCE_ROW:?TARGET_SOURCE_ROW is required}"
: "${RANDOM_SEED:?RANDOM_SEED is required}"
: "${TARGET_ITERATION:?TARGET_ITERATION is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_PARTICLES_SHA256:?EXPECTED_PARTICLES_SHA256 is required}"
: "${EXPECTED_REFERENCE_SHA256:?EXPECTED_REFERENCE_SHA256 is required}"
: "${EXPECTED_RELION_BINARY_SHA256:?EXPECTED_RELION_BINARY_SHA256 is required}"

case "${TARGET_SOURCE_ROW}" in
  ''|*[!0-9]*) echo "TARGET_SOURCE_ROW must be a non-negative integer" >&2; exit 2 ;;
esac
case "${RANDOM_SEED}" in
  ''|*[!0-9]*) echo "RANDOM_SEED must be a non-negative integer" >&2; exit 2 ;;
esac
case "${TARGET_ITERATION}" in
  ''|*[!0-9]*) echo "TARGET_ITERATION must be a positive integer" >&2; exit 2 ;;
esac
test "${TARGET_ITERATION}" -gt 0
TARGET_STACK_INDEX=$((TARGET_SOURCE_ROW + 1))

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_native_fresh_verbose_score_capture.sbatch
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi}
AUDIT_REPO=${AUDIT_REPO:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout}
CAPTURE=${RUN_ROOT}/capture
RELION_OUT=${RUN_ROOT}/relion

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -s "${DATA_DIR}/particles.star" -a -s "${DATA_DIR}/reference_init_relion.mrc"
test -s "${CONTROL_DIR}/run_it001_data.star"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')" = "${EXPECTED_PARTICLES_SHA256}"
test "$(sha256sum "${DATA_DIR}/reference_init_relion.mrc" | awk '{print $1}')" = "${EXPECTED_REFERENCE_SHA256}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"

mkdir -p "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" \
  "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in
    RECOVAR_*|RELION_ACC_DUMP_*|RELION_BPRE_*|RELION_FINE_*|RELION_PPREF_*)
      unset "${variable_name}"
      ;;
  esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_ACC_DUMP_DIR=${CAPTURE}
export RELION_ACC_DUMP_STACK_INDEX=${TARGET_STACK_INDEX}
export RELION_ACC_DUMP_ITER=${TARGET_ITERATION}
export RELION_ACC_DUMP_VERBOSE=1
export RELION_ACC_DUMP_CC_COMPONENTS=1
export RELION_ACC_DUMP_PASS1_DIFF2=1

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200
  --ini_high 30 --firstiter_cc --ctf --flatten_solvent --zero_mask
  --low_resol_join_halves 40 --norm --scale --healpix_order 3
  --offset_range 3.0 --offset_step 1.0 --oversampling 1 --pad 2
  --random_seed "${RANDOM_SEED}" --auto_iter_max "${TARGET_ITERATION}"
  --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA_DIR}/particles.star" \
  "${DATA_DIR}/reference_init_relion.mrc" \
  "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
(
  cd "${DATA_DIR}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

printf -v iteration_label '%03d' "${TARGET_ITERATION}"
test -s "${RELION_OUT}/run_it${iteration_label}_data.star"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_corr_img.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_Fimg_corrected_real.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_Fimg_corrected_imag.bin' | wc -l)" -eq 1
test -s "${CAPTURE}/pass1_class0_fine_ref_real.bin"
test -s "${CAPTURE}/pass1_class0_fine_ref_imag.bin"
test -s "${CAPTURE}/pass1_class0_fine_shifted_real.bin"
test -s "${CAPTURE}/pass1_class0_fine_shifted_imag.bin"
test -s "${CAPTURE}/pass1_exp_Mweight_raw_preprior.bin"

mapfile -t audit_iterations < <(seq 1 "${TARGET_ITERATION}")
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL_DIR}" --candidate-dir "${RELION_OUT}" \
  --iterations "${audit_iterations[@]}" --map-fsc-auc-threshold 0.999999 \
  --output-json "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
  --output-markdown "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"target_iteration":%d,"source_row_zero_based":%d,"stack_index_one_based":%d}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_ITERATION}" "${TARGET_SOURCE_ROW}" \
  "${TARGET_STACK_INDEX}" > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
