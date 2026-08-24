#!/usr/bin/env bash
#SBATCH --job-name=k1-c10-fresh-fine-pixels
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
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

TARGET_SOURCE_ROW=${K1_NATIVE_TARGET_SOURCE_ROW:-12334}
case "${TARGET_SOURCE_ROW}" in
  ''|*[!0-9]*) echo "K1_NATIVE_TARGET_SOURCE_ROW must be a non-negative integer" >&2; exit 2 ;;
esac
TARGET_STACK_INDEX=$((TARGET_SOURCE_ROW + 1))

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
CONTROL=${CASE}/relion_ref
BINARY=/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case10_native_fresh_verbose_fine.sbatch
CAPTURE=${RUN_ROOT}/capture
OUT=${RUN_ROOT}/relion

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = a9a961340af621d1cd581ccea2e96274f978b53213f9606c9f0e30d271903b8c
test "$(sha256sum "${CASE}/data/particles.star" | awk '{print $1}')" = 3be82df3627e0d6c3c12d1eb42e1ab0c7feb09117bb66a38cc53c0621e9f8436
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test -z "$(find "${CAPTURE}" "${OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
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
export RELION_ACC_DUMP_ITER=2
export RELION_ACC_DUMP_VERBOSE=1
export RELION_ACC_DUMP_CC_COMPONENTS=1
export RELION_ACC_DUMP_PASS1_DIFF2=1

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200
  --ini_high 30 --firstiter_cc --ctf --flatten_solvent --zero_mask
  --low_resol_join_halves 40 --norm --scale --healpix_order 3
  --offset_range 3.0 --offset_step 1.0 --oversampling 1 --pad 2
  --random_seed 1710 --auto_iter_max 2 --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${BINARY}" "${CASE}/data/particles.star" \
  "${CASE}/data/reference_init_relion.mrc" \
  "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
(
  cd "${CASE}/data"
  "${command[@]}"
) > "${OUT}/runner.stdout" 2> "${OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${OUT}/run_it001_data.star" -a -s "${OUT}/run_it002_data.star"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_corr_img.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_Fimg_corrected_real.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name 'pass1_img*_Fimg_corrected_imag.bin' | wc -l)" -eq 1
test -s "${CAPTURE}/pass1_class0_fine_ref_real.bin"
test -s "${CAPTURE}/pass1_class0_fine_ref_imag.bin"
test -s "${CAPTURE}/pass1_class0_fine_shifted_real.bin"
test -s "${CAPTURE}/pass1_class0_fine_shifted_imag.bin"
test -s "${CAPTURE}/pass1_exp_Mweight_raw_preprior.bin"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL}" --candidate-dir "${OUT}" --iterations 1 2 \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
  --output-markdown "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${OUT}" "${RUN_ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"fresh_iterations":[1,2],"source_row_zero_based":%d,"stack_index_one_based":%d,"binary_sha256":"a9a961340af621d1cd581ccea2e96274f978b53213f9606c9f0e30d271903b8c"}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_SOURCE_ROW}" "${TARGET_STACK_INDEX}" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
