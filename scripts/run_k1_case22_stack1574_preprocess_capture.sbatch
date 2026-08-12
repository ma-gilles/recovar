#!/usr/bin/env bash
#SBATCH --job-name=k1c22-preprocess
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00

set -euo pipefail
: "${K1_CASE22_PREPROCESS_ROOT:?set the disposable capture root}"
: "${K1_CASE22_PREPROCESS_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the passive capture binary}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the passive capture source diff}"

ROOT=${K1_CASE22_PREPROCESS_ROOT}
RUNTIME=${K1_CASE22_PREPROCESS_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case22_stack1574_preprocess_capture.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 9e73ee7de74f1ab550bbc55b4dcf2b8b2101f5df1228788e84a6def91b846fb6
test "$(sha256sum "${DATA}/particles.128.mrcs" | awk '{print $1}')" = 804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = 3a351b6fdaa631213713628952604aafe2f2b393ff6d30f121ef9fbf139e90b9
for directory in "${CAPTURE}" "${RELION_OUT}"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done

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
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${ROOT}/provenance" "${ROOT}/analysis"

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
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_P1_PREPROCESS_CAPTURE_DIR=${CAPTURE}
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=2
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=2000000
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=639

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"iteration":2,"part_id":639,"stack_index_one_based":1574}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"

test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.preprocess-v1.bin' | wc -l)" -eq 1
test -s "${CAPTURE}/part639_stack1574.preprocess-v1.bin"
test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"
test -s "${RELION_OUT}/run_it002_data.star"
"${PYTHON}" "${REPO}/scripts/validate_relion_preprocess_capture.py" \
    "${CAPTURE}" --expected-particles 1 --expected-part-ids 639 \
    --expected-stack-indices 1574 --expected-iteration 2 \
    --output "${ROOT}/analysis/RELION_PREPROCESS_CAPTURE_V1.json"

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"

find "${CAPTURE}" "${RELION_OUT}" "${ROOT}/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
