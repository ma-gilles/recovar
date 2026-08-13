#!/usr/bin/env bash
#SBATCH --job-name=k1c22-native-aa
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:20:00

set -euo pipefail
: "${K1_CASE22_NATIVE_AA_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_NATIVE_AA_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_HEAD:?pin the diagnostic RELION commit}"
: "${EXPECTED_RELION_DIFF_SHA256:?pin the diagnostic RELION diff}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the diagnostic executable}"

ROOT=${K1_CASE22_NATIVE_AA_ROOT}
RUNTIME=${K1_CASE22_NATIVE_AA_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_case22_native_scale_aa_pixels.sbatch
RELION_SOURCE=/scratch/gpfs/GILLES/mg6942/relion
RELION_BINARY=${RELION_SOURCE}/build_patched/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION_OUT=${ROOT}/relion
CAPTURE=${ROOT}/capture

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_DIFF_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(strings "${RELION_BINARY}" | grep -c RELION_DUMP_SIGMA2_AA_PART_ID)" -eq 1
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"
test -z "$(find "${CAPTURE}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${ROOT}/provenance"

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
export RELION_DUMP_SIGMA2_NOISE_DIR=${CAPTURE}
export RELION_DUMP_SIGMA2_AA_PART_ID=109
export RELION_DUMP_SIGMA2_AA_ITER=2

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"max_iter":2,"part_id":109}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"

test -s "${RELION_OUT}/run_it002_data.star"
test -s "${CAPTURE}/scale_aa_pixels.tsv"
awk -F'\t' '
    $1 != "acc_scale_pixel" { exit 1 }
    $2 != "iter=2" { exit 1 }
    $3 != "part_id=109" { exit 1 }
    $4 != "halfset=1" { exit 1 }
    END { if (NR == 0) exit 1 }
' "${CAPTURE}/scale_aa_pixels.tsv"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${CAPTURE}/scale_aa_pixels.tsv" \
    > "${ROOT}/provenance/static_and_capture_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
