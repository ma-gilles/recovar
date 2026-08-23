#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-bpm-v2
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_rotation_mass_full_20260810T1730ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_rotation_mass_full_20260810T1730ET/logs/run-%j.err
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
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher}"
ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_rotation_mass_full_20260810T1730ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_native_rotation_mass_full_20260810T1730ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_rotation_mass_build_20260810T1610ET/build/bin/relion_refine_mpi
CONTINUE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion/run_it001_optimiser.star
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
LAUNCHER=${REPO}/scripts/run_k1_case22_native_rotation_mass_all.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = 2cfb54501ed139d42bde78b7ec0dd6460b13b6f63e4d5732c9bfe8c2f4193e01
test "$(sha256sum "${CONTINUE}" | awk '{print $1}')" = 9aebfe257244d5871d496838b6527c9d63e2e77fe0522d5246d2509d4634d25c
test -z "$(find "${ROOT}/capture" "${ROOT}/relion" -mindepth 1 -print -quit)"

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

export RELION_PREITER_PARTICLE_ORDER_SEED=1723
export RELION_PREITER_PARTICLE_ORDER_ITER=2
export RELION_SAMPLING_PERTURBATION_OVERRIDE=0.409489929676
export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=2
export RELION_BPM_CAPTURE_DIR=${ROOT}/capture
export RELION_BPM_CAPTURE_SCHEMA=1
export RELION_BPM_CAPTURE_COMPACT_ROTATION_MASS=1
export RELION_BPM_CAPTURE_ITER=2
export RELION_BPM_CAPTURE_CLASS=1
export RELION_BPM_CAPTURE_EXPECTED_PARTICLES=3000
export RELION_BPM_CAPTURE_MAX_PARTICLES_PER_RANK=1600
export RELION_BPM_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPM_CAPTURE_MAX_BYTES=50000000000

COMMAND=(
  srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
  --continue "${CONTINUE}" --o "${ROOT}/relion/run"
  --auto_iter_max 2 --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${BINARY}" "${CONTINUE}" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start=$(date +%s)
(
  cd "${DATA}"
  "${COMMAND[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
wall_s=$(( $(date +%s) - start ))

test -s "${ROOT}/relion/run_it002_data.star"
grep -q '\[RELION_PREITER_PARTICLE_ORDER\] iteration 2 effective_seed 1723 particles 3000' "${ROOT}/relion/runner.stderr"
grep -q '\[RELION_SAMPLING_PERTURBATION_OVERRIDE\] iter 2 requested 0.409489929676 applied' "${ROOT}/relion/runner.stdout"
"${PYTHON}" -m scripts.validate_relion_bpref_rotation_mass \
  "${ROOT}/capture" --expected-particles 3000 \
  --output-json "${ROOT}/analysis/RELION_BPREF_ROTATION_MASS_ALL.json"
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${ROOT}/relion/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"particle_count":3000,"passive_rotation_mass_capture":true}\n' "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
