#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-native-fresh-it3-raw
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RECOVAR_K1_NATIVE_FRESH_IT3_RUN_ROOT:?set the empty run root}"
: "${RECOVAR_K1_NATIVE_FRESH_IT3_RUNTIME_ROOT:?set the empty runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin the launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin the repository head}"

ROOT=${RECOVAR_K1_NATIVE_FRESH_IT3_RUN_ROOT}
RUNTIME=${RECOVAR_K1_NATIVE_FRESH_IT3_RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_operand_capture_clean_20260810T1330ET/build/bin/relion_refine_mpi
LAUNCHER=${REPO}/scripts/run_k1_case22_native_fresh_it3_raw_accum.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = 7b989db3b9c1c67ba9752e82803e982b892ffc647ecf5536ae35a9f243cfe72d
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(find "${ROOT}/dump" "${ROOT}/relion" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0 CUDA_VISIBLE_DEVICES=0

# A fresh process reaches physical iteration 3 at dump call 2.  Capturing all
# three calls makes that chronology explicit and prevents a serialized
# continuation from being mistaken for the uninterrupted native trajectory.
export RECOVAR_MSTEP_DUMP_DIR=${ROOT}/dump
export RECOVAR_MSTEP_DUMP_MAX_CALLS=3
export RECOVAR_MSTEP_DUMP_RAW=1

COMMAND=(
  srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${ROOT}/relion/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 3 --pool 3
  --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${BINARY}" "${DATA}/particles.star" \
  "${DATA}/reference_init_relion.mrc" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start=$(date +%s)
(
  cd "${DATA}"
  "${COMMAND[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
wall_s=$(( $(date +%s) - start ))

test -s "${ROOT}/relion/run_it003_data.star"
for rank in 01 02; do
  for call in 0000 0001 0002; do
    test -s "${ROOT}/dump/raw_bpref_data_rank${rank}_call${call}.bin"
    test -s "${ROOT}/dump/raw_bpref_weight_rank${rank}_call${call}.bin"
  done
done
grep -q 'Auto-refine: Iteration= 3' "${ROOT}/relion/runner.stdout"
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${ROOT}/relion/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":3,"dump_call":2,"gpu_model":"H100","raw_bpref_capture":true,"fresh_uninterrupted_auto_refine":true}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${ROOT}/dump" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ROOT}/provenance/dump_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
