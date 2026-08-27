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
RELION_THREADS=${RELION_THREADS:-8}
RANDOM_SEED=${RANDOM_SEED:-0}
WORKER_TRACE_REPLAY=${RELION_VDAM_WORKER_TRACE_REPLAY:-0}
WORKER_TRACE_PARTICLES=${RELION_VDAM_WORKER_TRACE_PARTICLES:-200}
WORKER_TRACE_DATASET_PARTICLES=${RELION_VDAM_WORKER_TRACE_DATASET_PARTICLES:-3000}
WORKER_TRACE_POOL_SIZE=${RELION_VDAM_WORKER_TRACE_POOL_SIZE:-24}
NATIVE_SECOND_MOMENT_REPLAY=${VDAM_NATIVE_SECOND_MOMENT_REPLAY:-0}
PART_ID=${VDAM_PREPROCESS_PART_ID:-0}
BPREF_CONTRIBUTION_HALF=${BPREF_CONTRIBUTION_HALF:-}
BPREF_CONTRIBUTION_CAPTURE_HALF=${BPREF_CONTRIBUTION_CAPTURE_HALF:-${BPREF_CONTRIBUTION_HALF}}
BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP=${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP:-}
BPREF_CONTRIBUTION_CURRENT_SIZE=${BPREF_CONTRIBUTION_CURRENT_SIZE:-}
BPREF_CONTRIBUTION_ORIGINAL_INDICES=${BPREF_CONTRIBUTION_ORIGINAL_INDICES:-}
RELION_SOURCE=${VDAM_RELION_SOURCE:-/scratch/gpfs/GILLES/mg6942/relion}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_SOURCE_BINARY=${RECOVAR_CUDA_LIB_SOURCE:-${REPO_ROOT}/recovar/cuda/libcuda_backproject.so}
CUDA_BINARY=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/cuda/libcuda_backproject.so
RELION_BIND_BINARY=${REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
RELION_OUTPUT=${OUTPUT_ROOT}/relion
RECOVAR_OUTPUT=${OUTPUT_ROOT}/recovar
NATIVE_MSTEP=${OUTPUT_ROOT}/native_mstep
RECOVAR_MSTEP=${OUTPUT_ROOT}/recovar_mstep
NATIVE_CAPTURE=${OUTPUT_ROOT}/native_capture
PROVENANCE=${OUTPUT_ROOT}/provenance
ANALYSIS=${OUTPUT_ROOT}/analysis
CONTRIBUTIONS=${OUTPUT_ROOT}/recovar_bpref_contributions
WORKER_TRACE=${PROVENANCE}/relion_worker_trace_it001.tsv
WORKER_SCHEDULE=${ANALYSIS}/relion_worker_schedule_it001.npz

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -x "${PIXI_PY}"
test "${TARGET_ITERATION}" -ge 1
test "${TARGET_ITERATION}" -le "${NR_ITER_SCHEDULE}"
test "${RELION_THREADS}" -ge 1
test "${RELION_THREADS}" -le "${SLURM_CPUS_PER_TASK:-8}"
test "${WORKER_TRACE_REPLAY}" = 0 -o "${WORKER_TRACE_REPLAY}" = 1
test "${NATIVE_SECOND_MOMENT_REPLAY}" = 0 -o "${NATIVE_SECOND_MOMENT_REPLAY}" = 1
if [[ "${WORKER_TRACE_REPLAY}" = 1 ]]; then
  test "${TARGET_ITERATION}" -eq 1
  test "${RELION_THREADS}" -eq 8
  test "${WORKER_TRACE_PARTICLES}" -ge 1
  test "${WORKER_TRACE_DATASET_PARTICLES}" -ge "${WORKER_TRACE_PARTICLES}"
  test "${WORKER_TRACE_POOL_SIZE}" -ge 1
fi
if [[ -n "${BPREF_CONTRIBUTION_HALF}" ]]; then
  test "${BPREF_CONTRIBUTION_HALF}" = 1 -o "${BPREF_CONTRIBUTION_HALF}" = 2
  test "${BPREF_CONTRIBUTION_CAPTURE_HALF}" = 1 -o "${BPREF_CONTRIBUTION_CAPTURE_HALF}" = 2
  if [[ -n "${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP}" ]]; then
    test "${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP}" -eq "$((BPREF_CONTRIBUTION_HALF - 1))"
  fi
  if [[ -n "${BPREF_CONTRIBUTION_ORIGINAL_INDICES}" ]]; then
    if [[ -n "$(printf '%s' "${BPREF_CONTRIBUTION_ORIGINAL_INDICES}" | tr -d '0-9,;[:space:]')" ]]; then
      echo "BPREF_CONTRIBUTION_ORIGINAL_INDICES contains invalid characters" >&2
      exit 2
    fi
  fi
  test -n "${BPREF_CONTRIBUTION_CURRENT_SIZE}"
  test "${BPREF_CONTRIBUTION_CURRENT_SIZE}" -ge 1
fi
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test -s "${CUDA_SOURCE_BINARY}"
test "$(sha256sum "${CUDA_SOURCE_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
mkdir -p "${RELION_OUTPUT}" "${RECOVAR_OUTPUT}" "${NATIVE_MSTEP}" \
  "${RECOVAR_MSTEP}" "${NATIVE_CAPTURE}" "${PROVENANCE}" "${ANALYSIS}" \
  "${CONTRIBUTIONS}"
for directory in "${RELION_OUTPUT}" "${RECOVAR_OUTPUT}" "${NATIVE_MSTEP}" \
  "${RECOVAR_MSTEP}" "${NATIVE_CAPTURE}" "${PROVENANCE}" "${ANALYSIS}" \
  "${CONTRIBUTIONS}"; do
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
mkdir -p "$(dirname "${CUDA_BINARY}")"
cp --no-preserve=mode,ownership,timestamps "${CUDA_SOURCE_BINARY}" "${CUDA_BINARY}"
chmod u+rw "${CUDA_BINARY}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
ORIGINAL_PATH=${PATH}
ORIGINAL_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

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
export RECOVAR_DEBUG_DUMP_MSTEP_ITER=${TARGET_ITERATION}
export RELION_ACC_DUMP_DIR=${NATIVE_CAPTURE}
export RELION_ACC_DUMP_PART_ID=${PART_ID}
export RELION_ACC_DUMP_ITER=${TARGET_ITERATION}
export RELION_ACC_DUMP_VERBOSE=0
if [[ "${WORKER_TRACE_REPLAY}" = 1 ]]; then
  export RELION_VDAM_WORKER_LOG=${WORKER_TRACE}
  export RELION_VDAM_WORKER_TRACE_ITER=1
fi

RELION_COMMAND=(
  srun --ntasks=1 --cpus-per-task="${RELION_THREADS}" --cpu-bind=none "${VDAM_RELION_BINARY}"
  --o "${RELION_OUTPUT}/run" --iter "${NR_ITER_SCHEDULE}" --grad --denovo_3dref
  --grad_write_iter 1 --i particles.star --ctf --K 1 --sym C1
  --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 3 --pad 1
  --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6.0
  --offset_step 2.0 --auto_sampling --tau2_fudge 4.0 --random_seed "${RANDOM_SEED}" --j "${RELION_THREADS}" --gpu 0
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

if [[ "${WORKER_TRACE_REPLAY}" = 1 ]]; then
  test -s "${WORKER_TRACE}"
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.build_vdam_worker_schedule \
      --trace "${WORKER_TRACE}" --iteration 1 \
      --n-particles "${WORKER_TRACE_PARTICLES}" \
      --dataset-particles "${WORKER_TRACE_DATASET_PARTICLES}" \
      --n-threads "${RELION_THREADS}" --pool-size "${WORKER_TRACE_POOL_SIZE}" \
      --output-npz "${WORKER_SCHEDULE}" \
      --output-json "${ANALYSIS}/relion_worker_schedule_it001.json"
  )
  test -s "${WORKER_SCHEDULE}"
  sha256sum "${WORKER_TRACE}" "${WORKER_SCHEDULE}" \
    "${ANALYSIS}/relion_worker_schedule_it001.json" \
    > "${PROVENANCE}/relion_worker_schedule.sha256"
fi

for required in \
  pipe_it${TARGET_ITERATION}_c0_bp_data_pre_reweight.bin \
  pipe_it${TARGET_ITERATION}_c0_bp_weight.bin \
  pipe_it${TARGET_ITERATION}_c0_bp_data_h_pre_reweight.bin \
  pipe_it${TARGET_ITERATION}_c0_bp_weight_h.bin \
  mstep_it${TARGET_ITERATION}_c0_iref_before.bin \
  mstep_it${TARGET_ITERATION}_c0_iref_after.bin; do
  test -s "${NATIVE_MSTEP}/${required}"
done
NATIVE_SECOND_MOMENT_REPLAY_BIN=${NATIVE_MSTEP}/pipe_it${TARGET_ITERATION}_c0_Igrad2_post.bin
if [[ "${NATIVE_SECOND_MOMENT_REPLAY}" = 1 ]]; then
  test -s "${NATIVE_SECOND_MOMENT_REPLAY_BIN}"
  sha256sum "${NATIVE_SECOND_MOMENT_REPLAY_BIN}" \
    > "${PROVENANCE}/native_second_moment_replay.sha256"
fi

gpu_uuid_after_relion=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_relion}" = "${gpu_uuid_before}"
test "$(sha256sum "${CUDA_SOURCE_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

set +u
module purge
set -u
export PATH=${ORIGINAL_PATH}
if [[ -n "${ORIGINAL_LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH=${ORIGINAL_LD_LIBRARY_PATH}
else
  unset LD_LIBRARY_PATH
fi
unset MPI_ROOT CUDA_HOME OMPI_MCA_orte_tmpdir_base OMPI_MCA_shmem_mmap_enable_nfs_warning
unset RECOVAR_DEBUG_DUMP_DIR RECOVAR_DEBUG_DUMP_MSTEP_ITER
unset RELION_ACC_DUMP_DIR RELION_ACC_DUMP_PART_ID RELION_ACC_DUMP_ITER RELION_ACC_DUMP_VERBOSE
unset RELION_VDAM_WORKER_LOG RELION_VDAM_WORKER_TRACE_ITER
export RECOVAR_MSTEP_DUMP_DIR=${RECOVAR_MSTEP}
export RECOVAR_MSTEP_DUMP_ITER=${TARGET_ITERATION}
if [[ "${NATIVE_SECOND_MOMENT_REPLAY}" = 1 ]]; then
  export RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_BIN=${NATIVE_SECOND_MOMENT_REPLAY_BIN}
  export RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER=${TARGET_ITERATION}
fi
if [[ "${WORKER_TRACE_REPLAY}" = 1 ]]; then
  export RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ=${WORKER_SCHEDULE}
fi
if [[ -n "${BPREF_CONTRIBUTION_HALF}" ]]; then
  mapfile -t contribution_metadata < <(
    "${PIXI_PY}" - "${DATA_DIR}/particles.star" \
      "${PROVENANCE}/bpref_image_names.npy" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import starfile

star_path, output_path, data_directory = map(Path, sys.argv[1:])
star = starfile.read(star_path)
particles = star["particles"] if isinstance(star, dict) else star
image_names = np.asarray(particles["rlnImageName"].astype(str).tolist())
if image_names.ndim != 1 or image_names.size == 0:
    raise SystemExit("VDAM BPref capture requires nonempty rlnImageName metadata")
stack_paths = []
image_numbers = []
for image_name in image_names.tolist():
    image_number, separator, stack_name = image_name.partition("@")
    if separator != "@" or not stack_name or int(image_number) <= 0:
        raise SystemExit(f"invalid rlnImageName: {image_name}")
    image_numbers.append(int(image_number))
    stack_path = Path(stack_name)
    if not stack_path.is_absolute():
        stack_path = Path(data_directory) / stack_path
    stack_paths.append(stack_path.resolve())
if len(set(stack_paths)) != 1 or not stack_paths[0].is_file():
    raise SystemExit("VDAM BPref capture requires one resolved particle stack")
resolved_image_names = np.asarray(
    [f"{image_number}@{stack_path}" for image_number, stack_path in zip(image_numbers, stack_paths)]
)
np.save(output_path, resolved_image_names)
print(stack_paths[0])
print(image_names.size)
PY
  )
  test "${#contribution_metadata[@]}" -eq 2
  contribution_stack=${contribution_metadata[0]}
  contribution_particle_count=${contribution_metadata[1]}
  test "${contribution_particle_count}" -ge 1
  export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTIONS}
  export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=${TARGET_ITERATION}
  export RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF=${BPREF_CONTRIBUTION_CAPTURE_HALF}
  export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=${BPREF_CONTRIBUTION_CURRENT_SIZE}
  if [[ -n "${BPREF_CONTRIBUTION_ORIGINAL_INDICES}" ]]; then
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${BPREF_CONTRIBUTION_ORIGINAL_INDICES}
  fi
  export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=vdam-gf01-it${TARGET_ITERATION}-capture-half${BPREF_CONTRIBUTION_CAPTURE_HALF}
  export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${PROVENANCE}/bpref_image_names.npy
  export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256
  RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=$(sha256sum "${contribution_stack}" | awk '{print $1}')
fi

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
  --random_seed "${RANDOM_SEED}"
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
if [[ -n "${BPREF_CONTRIBUTION_HALF}" ]]; then
  test "$(find "${CONTRIBUTIONS}" -maxdepth 1 -type f \
    -name "bpref_contribution_rows_it${target_tag}_h*.npz" | wc -l)" -ge 1
fi
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.analyze_vdam_mstep_boundary \
    --native-directory "${NATIVE_MSTEP}" \
    --recovar-directory "${RECOVAR_MSTEP}" \
    --iteration "${TARGET_ITERATION}" \
    --output-json "${ANALYSIS}/mstep_boundary.json"
)
if [[ -n "${BPREF_CONTRIBUTION_HALF}" ]]; then
  mapfile -t contribution_paths < <(
    find "${CONTRIBUTIONS}" -maxdepth 1 -type f \
      -name "bpref_contribution_rows_it${target_tag}_h*.npz" | sort
  )
  test "${#contribution_paths[@]}" -ge 1
  BPREF_ANALYZER_GROUP_ARGS=()
  if [[ -n "${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP}" ]]; then
    BPREF_ANALYZER_GROUP_ARGS=(
      --reconstruction-group "${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP}"
    )
  fi
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.analyze_vdam_bpref_accumulator_boundary \
      "${contribution_paths[@]}" \
      --native-directory "${NATIVE_MSTEP}" \
      --recovar-directory "${RECOVAR_MSTEP}" \
      --half "${BPREF_CONTRIBUTION_HALF}" \
      "${BPREF_ANALYZER_GROUP_ARGS[@]}" \
      --output-json "${ANALYSIS}/bpref_accumulator_boundary.json"
  )
  test -s "${ANALYSIS}/bpref_accumulator_boundary.json"
fi

gpu_uuid_after_recovar=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "${gpu_uuid_after_recovar}" = "${gpu_uuid_before}"
test "$(sha256sum "${CUDA_SOURCE_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
git -C "${RELION_SOURCE}" rev-parse HEAD > "${PROVENANCE}/relion_head.txt"
git -C "${RELION_SOURCE}" diff --binary | sha256sum > "${PROVENANCE}/relion_diff.sha256"
sha256sum "${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_boundary.sbatch" \
  "${REPO_ROOT}/scripts/analyze_vdam_mstep_boundary.py" \
  "${REPO_ROOT}/scripts/analyze_vdam_bpref_accumulator_boundary.py" \
  "${DATA_DIR}/particles.star" \
  "${CUDA_SOURCE_BINARY}" "${CUDA_BINARY}" "${RELION_BIND_BINARY}" "${VDAM_RELION_BINARY}" \
  > "${PROVENANCE}/static_inputs.sha256"
printf '{"job_id":"%s","git_head":"%s","physical_gpu_uuid":"%s","nr_iter_schedule":%d,"stopped_after_iteration":%d,"relion_threads":%d,"random_seed":%d,"worker_trace_replay":%d,"native_second_moment_replay":%d,"bpref_contribution_half":"%s","bpref_contribution_capture_half":"%s","bpref_contribution_reconstruction_group":"%s","bpref_contribution_current_size":"%s","bpref_contribution_original_indices":"%s"}\n' \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${gpu_uuid_before}" \
  "${NR_ITER_SCHEDULE}" "${TARGET_ITERATION}" "${RELION_THREADS}" "${RANDOM_SEED}" "${WORKER_TRACE_REPLAY}" "${NATIVE_SECOND_MOMENT_REPLAY}" "${BPREF_CONTRIBUTION_HALF}" \
  "${BPREF_CONTRIBUTION_CAPTURE_HALF}" "${BPREF_CONTRIBUTION_RECONSTRUCTION_GROUP}" \
  "${BPREF_CONTRIBUTION_CURRENT_SIZE}" "${BPREF_CONTRIBUTION_ORIGINAL_INDICES}" \
  > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
