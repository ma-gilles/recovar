#!/usr/bin/env bash
#SBATCH --job-name=vdam-mstep-fullschedule
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-mstep-fullschedule-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${DATA_DIR:?set the frozen materialized VDAM fixture directory}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${VDAM_RELION_BINARY:?set the isolated patched non-MPI RELION binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
TARGET_ITERATION=${TARGET_ITERATION:-2}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
PART_ID=${VDAM_PREPROCESS_PART_ID:-0}
RELION_SOURCE=${VDAM_RELION_SOURCE:-/scratch/gpfs/GILLES/mg6942/relion}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
RELION_BIND_BINARY=${REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
RELION_OUTPUT=${OUTPUT_ROOT}/relion
RECOVAR_OUTPUT=${OUTPUT_ROOT}/recovar
NATIVE_MSTEP=${OUTPUT_ROOT}/native_mstep
RECOVAR_MSTEP=${OUTPUT_ROOT}/recovar_mstep
NATIVE_CAPTURE=${OUTPUT_ROOT}/native_capture
PROVENANCE=${OUTPUT_ROOT}/provenance
ANALYSIS=${OUTPUT_ROOT}/analysis

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -x "${PIXI_PY}"
test "${TARGET_ITERATION}" -ge 1
test "${TARGET_ITERATION}" -le "${NR_ITER_SCHEDULE}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
mkdir -p "${RELION_OUTPUT}" "${RECOVAR_OUTPUT}" "${NATIVE_MSTEP}" \
  "${RECOVAR_MSTEP}" "${NATIVE_CAPTURE}" "${PROVENANCE}" "${ANALYSIS}"
for directory in "${RELION_OUTPUT}" "${RECOVAR_OUTPUT}" "${NATIVE_MSTEP}" \
  "${RECOVAR_MSTEP}" "${NATIVE_CAPTURE}" "${PROVENANCE}" "${ANALYSIS}"; do
  test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${OUTPUT_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"

source /etc/profile.d/modules.sh
set +u
module purge
set +e
module load relion/5.0.0/gcc-11.5.0
module_status=$?
set -e
if [[ "${module_status}" -ne 0 ]]; then
  echo "RELION module returned ${module_status}; using the pinned runtime overlay" >&2
fi
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_DEBUG_DUMP_DIR=${NATIVE_MSTEP}
export RECOVAR_DEBUG_DUMP_MSTEP_ITER=1
export RELION_ACC_DUMP_DIR=${NATIVE_CAPTURE}
export RELION_ACC_DUMP_PART_ID=${PART_ID}
export RELION_ACC_DUMP_ITER=${TARGET_ITERATION}
export RELION_ACC_DUMP_VERBOSE=0

RELION_COMMAND=(
  srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "${VDAM_RELION_BINARY}"
  --o "${RELION_OUTPUT}/run" --iter "${NR_ITER_SCHEDULE}" --grad --denovo_3dref
  --grad_write_iter 1 --i particles.star --ctf --K 1 --sym C1
  --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 3 --pad 1
  --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6.0
  --offset_step 2.0 --auto_sampling --tau2_fudge 4.0 --random_seed 0 --j 8 --gpu 0
)
printf '%q ' "${RELION_COMMAND[@]}" > "${PROVENANCE}/relion_command.sh"
printf '\n' >> "${PROVENANCE}/relion_command.sh"

set +e
(
  cd "${DATA_DIR}"
  "${RELION_COMMAND[@]}"
) > "${RELION_OUTPUT}/runner.stdout" 2> "${RELION_OUTPUT}/runner.stderr" &
relion_pid=$!
set -e
target_tag=$(printf '%03d' "${TARGET_ITERATION}")
target_complete=0
for _poll_index in $(seq 1 1200); do
  if [[ -s "${RELION_OUTPUT}/run_it${target_tag}_sampling.star" ]]; then
    target_complete=1
    break
  fi
  if ! kill -0 "${relion_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
test "${target_complete}" -eq 1
active_step=$(
  squeue --steps -j "${SLURM_JOB_ID}" -h -o '%i' \
    | awk -v job="${SLURM_JOB_ID}" '$0 ~ "^" job "\\.[0-9]+$" {step=$0} END {print step}'
)
if [[ -n "${active_step}" ]]; then
  scancel "${active_step}"
else
  kill "${relion_pid}" 2>/dev/null || true
fi
set +e
wait "${relion_pid}"
set -e
touch "${OUTPUT_ROOT}/NATIVE_STOPPED_AFTER_IT${target_tag}"

for required in \
  pipe_it1_c0_bp_data_pre_reweight.bin pipe_it1_c0_bp_weight.bin \
  pipe_it1_c0_bp_data_h_pre_reweight.bin pipe_it1_c0_bp_weight_h.bin \
  mstep_it1_c0_iref_before.bin mstep_it1_c0_iref_after.bin; do
  test -s "${NATIVE_MSTEP}/${required}"
done

gpu_uuid_after_relion=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_relion}" = "${gpu_uuid_before}"

unset RECOVAR_DEBUG_DUMP_DIR RECOVAR_DEBUG_DUMP_MSTEP_ITER
unset RELION_ACC_DUMP_DIR RELION_ACC_DUMP_PART_ID RELION_ACC_DUMP_ITER RELION_ACC_DUMP_VERBOSE
export RECOVAR_MSTEP_DUMP_DIR=${RECOVAR_MSTEP}
export RECOVAR_MSTEP_DUMP_ITER=1

RECOVAR_COMMAND=(
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_DIR}/particles.star"
  --o "${RECOVAR_OUTPUT}/run"
  --nr_iter "${NR_ITER_SCHEDULE}"
  --grad_write_iter 1
  --K 1
  --tau2_fudge 4
  --sym C1
  --do_run_C1 1
  --particle_diameter 200
  --random_seed 0
  --healpix_order 1
  --oversampling 1
  --offset_range 6
  --offset_step 2
  --padding_factor 1
  --image_batch_size 500
  --datadir "${DATA_DIR}"
  --gpu 0
  --require_custom_cuda
  --diagnostic_stop_after_iteration "${TARGET_ITERATION}"
)
printf '%q ' "${RECOVAR_COMMAND[@]}" > "${PROVENANCE}/recovar_command.sh"
printf '\n' >> "${PROVENANCE}/recovar_command.sh"
(
  cd "${REPO_ROOT}"
  "${RECOVAR_COMMAND[@]}"
) > "${RECOVAR_OUTPUT}/runner.stdout" 2> "${RECOVAR_OUTPUT}/runner.stderr"

test -s "${RECOVAR_OUTPUT}/run_it${target_tag}_class001.mrc"
test -s "${RECOVAR_MSTEP}/accum_h0_data.npy"
test -s "${RECOVAR_MSTEP}/iref_out_relion_frame.npy"
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.analyze_vdam_mstep_boundary \
    --native-directory "${NATIVE_MSTEP}" \
    --recovar-directory "${RECOVAR_MSTEP}" \
    --iteration 1 \
    --output-json "${ANALYSIS}/mstep_boundary.json"
)

gpu_uuid_after_recovar=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_recovar}" = "${gpu_uuid_before}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
git -C "${RELION_SOURCE}" rev-parse HEAD > "${PROVENANCE}/relion_head.txt"
git -C "${RELION_SOURCE}" diff --binary | sha256sum > "${PROVENANCE}/relion_diff.sha256"
sha256sum "${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_boundary.sbatch" \
  "${REPO_ROOT}/scripts/analyze_vdam_mstep_boundary.py" "${DATA_DIR}/particles.star" \
  "${CUDA_BINARY}" "${RELION_BIND_BINARY}" "${VDAM_RELION_BINARY}" \
  > "${PROVENANCE}/static_inputs.sha256"
printf '{"job_id":"%s","git_head":"%s","physical_gpu_uuid":"%s","nr_iter_schedule":%d,"stopped_after_iteration":%d}\n' \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${gpu_uuid_before}" \
  "${NR_ITER_SCHEDULE}" "${TARGET_ITERATION}" > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
