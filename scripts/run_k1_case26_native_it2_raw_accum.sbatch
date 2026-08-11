#!/usr/bin/env bash
#SBATCH --job-name=k1c26-native-i2-acc
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_raw_accum_20260811T0120ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_raw_accum_20260811T0120ET/logs/run-%j.err
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

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_raw_accum_20260811T0120ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_native_it2_raw_accum_20260811T0120ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_it2_raw_accum.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
RELION_OUT=${ROOT}/relion
DUMP=${ROOT}/bpref_accum
RECOVAR_PREJOIN=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_fused_firstiter_prefix2_20260811T0140ET/prejoin/recovar_bpref_prejoin_it002.npz

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h | sha256sum | awk '{print $1}')" = 09e0de1c559d97b8c400adbe06b99fea46d7dba74845480ece6470e21a462e67
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 3fe0a3a89696f7e58a35883a33a2fb3e766a750cb164b29398f1e6ba9d34b60e
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139
test "$(sha256sum "${DATA}/particles.128.mrcs" | awk '{print $1}')" = 4cd77e4ddeb731f719adb0cfd661e4f00221e07d08e15e1a5dc229737d9c848c
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = 7aafc4f04e482f7410ff37686ba46db14578334ca77909f04bff30dbba2789d1
test -x "${PYTHON}" -a -x "${RELION_BINARY}" -a -s "${RECOVAR_PREJOIN}"
test -z "$(find "${RELION_OUT}" "${DUMP}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
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
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RELION_DUMP_DIR=${DUMP}
export RELION_DUMP_BPREF=1
export RELION_DUMP_ITER=2

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

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
{
    echo "job_id=${SLURM_JOB_ID}"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "physical_iteration=2"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${RECOVAR_PREJOIN}"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${RELION_OUT}/run_it002_data.star"
for half_rank in 1:1 2:2; do
    half=${half_rank%%:*}
    rank=${half_rank##*:}
    prefix=${DUMP}/mstep_it002_rank${rank}_half${half}_c0_pre_lowres_join_bpref
    test -s "${prefix}_data.bin" -a -s "${prefix}_weight.bin" -a -s "${prefix}_meta.txt"
    "${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
        --recovar-prejoin "${RECOVAR_PREJOIN}" \
        --native-data "${prefix}_data.bin" \
        --native-weight "${prefix}_weight.bin" \
        --half "${half}" --expected-local-iteration 2 --physical-iteration 2 \
        --output-json "${ROOT}/analysis/CASE26_IT2_RAW_ACCUM_HALF${half}.json"
done
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${RELION_OUT}/runner.stderr"

find "${DUMP}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/dump_${SLURM_JOB_ID}.sha256"
sha256sum "${ROOT}/analysis/CASE26_IT2_RAW_ACCUM_HALF"{1,2}".json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"gpu_uuid":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${GPU_UUID}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
