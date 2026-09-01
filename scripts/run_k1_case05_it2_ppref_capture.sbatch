#!/usr/bin/env bash
#SBATCH --job-name=k1c05-it2-ppref
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

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${BUILD_ROOT:?BUILD_ROOT is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_RELION_BINARY_SHA256:?EXPECTED_RELION_BINARY_SHA256 is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
RELION_EXE=${RELION_BINARY}
AUDIT_REPO=${AUDIT_REPO:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_case05_it2_ppref_capture.sbatch
DATA_DIR=${DATA_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/5_very_high_noise_100k_g256_white_noise10_bf80/data}
CONTROL_DIR=${CONTROL_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/5_very_high_noise_100k_g256_white_noise10_bf80/relion_ref}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717}
CAPTURE=${RUN_ROOT}/ppref
RELION_OUT=${RUN_ROOT}/relion

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${BUILD_ROOT}/SAFE_TO_DELETE"
test -f "${BUILD_ROOT}/SUCCESS_12784135"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_EXE}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')" = 078363915af39e644c0540d6e1ad020e41d94c3c7ee298e28cf39d4788cd384c
test "$(sha256sum "${DATA_DIR}/reference_init_relion.mrc" | awk '{print $1}')" = fbb7bbcba5ad804a6d4d78bac87f024f3a876c82f21f9397c5278961b30eda47
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
mkdir -p "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" \
  "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
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

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_PPREF_CAPTURE_DIR=${CAPTURE}
export RELION_PPREF_CAPTURE_ITER=2
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${RELION_EXE}" "${LAUNCHER}" "${DATA_DIR}/particles.star" \
  "${DATA_DIR}/reference_init_relion.mrc" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_EXE}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed 1705 --auto_iter_max 2
  --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
  cd "${DATA_DIR}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'ppref_iter002_*.bin' | wc -l)" -eq 2
test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"
cd "${AUDIT_REPO}"
"${PYTHON}" scripts/audit_relion_control_reproducibility.py \
  --control-dir "${CONTROL_DIR}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${RUN_ROOT}/analysis/RELION_PPREF_CAPTURE_INERTNESS.json" \
  --output-markdown "${RUN_ROOT}/analysis/RELION_PPREF_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${RUN_ROOT}/analysis/RELION_PPREF_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"iterations":[1,2]}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
