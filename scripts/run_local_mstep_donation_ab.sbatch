#!/usr/bin/env bash
#SBATCH --job-name=local-donate-ab
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/local-donate-ab-%j.out

set -euo pipefail

: "${LOCAL_MSTEP_DONATION_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${VDAM_LATE_PROFILE_CHECKPOINT:?set the frozen GF46 run_it180 optimiser}"
: "${VDAM_LATE_PROFILE_DATA_STAR:?set the frozen GF46 run_it180 data STAR}"
: "${VDAM_LATE_PROFILE_DATA_DIR:?set the frozen GF46 particle data directory}"
: "${EXPECTED_REPO_HEAD:?pin the reviewed source head}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the reviewed source manifest}"
: "${EXPECTED_NODE_NAME:?pin the H100 node}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${RELION_BIND_BINARY_OVERRIDE:?set the qualified RELION binding binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the RELION binding binary}"
: "${EXPECTED_FOCUSED_TEST_COUNT:?pin the exact focused pytest count}"

ROOT=${LOCAL_MSTEP_DONATION_ROOT}
CHECKPOINT=${VDAM_LATE_PROFILE_CHECKPOINT}
DATA_STAR=${VDAM_LATE_PROFILE_DATA_STAR}
DATA_DIR=${VDAM_LATE_PROFILE_DATA_DIR}
PARTICLE_STACK=${VDAM_LATE_PROFILE_PARTICLE_STACK:-${DATA_DIR}/particles.128.mrcs}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_RUNTIME_TOOLKIT=${CUDA_RUNTIME_TOOLKIT:-/usr/local/cuda-13.1}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
MPI_ROOT=${MPI_ROOT:-/usr/local/openmpi/cuda-12.6/4.1.6/gcc}
CUSPARSE_LIBRARY=${CUSPARSE_LIBRARY:-${CUDA_BUILD_TOOLKIT}/lib64/libcusparse.so.12}
CHECKPOINT_ITERATION=${CHECKPOINT_ITERATION:-180}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
PROFILED_ITERATION=$((CHECKPOINT_ITERATION + 1))
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
RESULTS=${ROOT}/results
RUNS=${ROOT}/runs
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
ANALYSIS=${RESULTS}/donation_ab_analysis.json
FOCUSED_TEST=tests/unit/initial_model/test_local_mstep_donation_ab.py
JOB_TMP_ROOT=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
JOB_PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
JOB_RATTLER_CACHE=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/em/dense_single_volume/local_big_jit.py
  recovar/em/dense_single_volume/local_em_engine.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/run_local_mstep_donation_ab.py
  scripts/analyze_local_mstep_donation_ab.py
  scripts/run_local_mstep_donation_ab.sbatch
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/initial_model/test_local_mstep_donation_ab.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="local M-step donation A/B failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  trap - ERR
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

write_source_manifest() {
  (
    cd "${REPO_ROOT}"
    sha256sum "${SOURCE_FILES[@]}"
  )
}

test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test -d "${CUDA_RUNTIME_TOOLKIT}/lib64"
test -s "${CUSPARSE_LIBRARY}"
test -s "${CHECKPOINT}"
test -s "${DATA_STAR}"
test -d "${DATA_DIR}"
test -s "${PARTICLE_STACK}"
test -s "${RELION_BIND_BINARY_OVERRIDE}"
[[ "${EXPECTED_FOCUSED_TEST_COUNT}" =~ ^[1-9][0-9]*$ ]]
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${EXPECTED_NODE_NAME}"
test "${PROFILED_ITERATION}" -lt "${NR_ITER_SCHEDULE}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi --id="${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${RESULTS}" "${RUNS}"
touch "${ROOT}/SAFE_TO_DELETE"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

checkpoint_prefix=${CHECKPOINT%_optimiser.star}
STATIC_INPUTS=(
  "${CHECKPOINT}"
  "${checkpoint_prefix}_model.star"
  "${checkpoint_prefix}_data.star"
  "${checkpoint_prefix}_sampling.star"
  "${checkpoint_prefix}_class001.mrc"
  "${checkpoint_prefix}_1moment001.mrc"
  "${checkpoint_prefix}_1moment002.mrc"
  "${checkpoint_prefix}_2moment001.mrc"
  "${DATA_STAR}"
  "${PARTICLE_STACK}"
  "${RELION_BIND_BINARY_OVERRIDE}"
  "${CUSPARSE_LIBRARY}"
)
for path in "${STATIC_INPUTS[@]}"; do
  test -s "${path}"
done
sha256sum "${STATIC_INPUTS[@]}" > "${PROVENANCE}/static_inputs.sha256"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${CUDA_BUILD_TOOLKIT}/bin/nvcc" \
  CUDA_ARCH='-gencode arch=compute_90,code=sm_90' \
  LIB="${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
cuda_sha256=$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')
printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"
cp --reflink=auto "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_DIR}/"
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_BINARY_OVERRIDE}")
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${JOB_TMP_ROOT}
export PIXI_HOME=${JOB_PIXI_HOME}
export RATTLER_CACHE_DIR=${JOB_RATTLER_CACHE}
export PATH=${CUDA_RUNTIME_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_BUILD_TOOLKIT}/lib64:${CUDA_BUILD_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_RUNTIME_TOOLKIT}/lib64:${CUDA_RUNTIME_TOOLKIT}/targets/x86_64-linux/lib
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${TMPDIR}/SAFE_TO_DELETE" "${PIXI_HOME}/SAFE_TO_DELETE" "${RATTLER_CACHE_DIR}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - <<'PY'
import os
import pathlib

import jax
import recovar
from recovar import cuda_backproject
from recovar.relion_bind import _relion_bind_core as relion_bind_core

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
print("local donation A/B GPU/runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" -m pytest -vv --run-gpu \
    --junitxml="${RESULTS}/pytest.junit.xml" "${FOCUSED_TEST}"
) > "${RESULTS}/pytest.stdout" 2> "${RESULTS}/pytest.stderr"
"${PIXI_PY}" - "${RESULTS}/pytest.junit.xml" "${EXPECTED_FOCUSED_TEST_COUNT}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = root.findall(".//testcase")
assert len(cases) == int(sys.argv[2]), (len(cases), sys.argv[2])
for case in cases:
    assert case.find("failure") is None, ET.tostring(case, encoding="unicode")
    assert case.find("error") is None, ET.tostring(case, encoding="unicode")
    assert case.find("skipped") is None, ET.tostring(case, encoding="unicode")
PY

printf 'ordinal\trepeat\tarm\n' > "${PROVENANCE}/execution_order.tsv"

run_arm() {
  local ordinal=$1
  local repeat=$2
  local arm=$3
  local run_dir=${RUNS}/repeat-${repeat}/${arm}
  local output_dir=${run_dir}/result
  local sample_file=${run_dir}/nvidia_smi.tsv
  local cache_dir=${RUNTIME}/jax_cache/${arm}
  mkdir -p "${run_dir}" "${cache_dir}"
  touch "${run_dir}/SAFE_TO_DELETE" "${cache_dir}/SAFE_TO_DELETE"
  printf '%s\t%s\t%s\n' "${ordinal}" "${repeat}" "${arm}" \
    >> "${PROVENANCE}/execution_order.tsv"
  printf 'unix_ns\tmemory_mib\n' > "${sample_file}"
  (
    cd "${REPO_ROOT}"
    env \
      LD_PRELOAD="${CUSPARSE_LIBRARY}" \
      JAX_COMPILATION_CACHE_DIR="${cache_dir}" \
      "${PIXI_PY}" -m scripts.run_local_mstep_donation_ab \
        --arm "${arm}" \
        --checkpoint-optimiser "${CHECKPOINT}" \
        --input-star "${DATA_STAR}" \
        --data-dir "${DATA_DIR}" \
        --output-root "${output_dir}" \
        --checkpoint-iteration "${CHECKPOINT_ITERATION}" \
        --nr-iter "${NR_ITER_SCHEDULE}" \
        --random-seed 29 \
        --image-batch-size 500 \
        --exact-local-bucket-radix 4 \
        --expected-repo-head "${EXPECTED_REPO_HEAD}" \
        --expected-gpu-uuid "${selected_gpu_uuid}"
  ) > "${run_dir}/runner.stdout" 2> "${run_dir}/runner.stderr" &
  local runner_pid=$!
  while kill -0 "${runner_pid}" 2>/dev/null; do
    local timestamp_ns
    local memory_mib
    timestamp_ns=$(date +%s%N)
    memory_mib=$(nvidia-smi --id="${selected_gpu_uuid}" \
      --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p' | tr -d '[:space:]')
    if [[ "${memory_mib}" =~ ^[0-9]+$ ]]; then
      printf '%s\t%s\n' "${timestamp_ns}" "${memory_mib}" >> "${sample_file}"
    fi
    sleep 0.05
  done
  wait "${runner_pid}"
  test -s "${output_dir}/donation_arm_summary.json"
  test -s "${output_dir}/TIMED_BEGIN"
  test -s "${output_dir}/TIMED_END"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
}

# Cross order on one physical H100; every entry is a fresh Python process.
run_arm 1 01 donated
run_arm 2 01 control
run_arm 3 02 control
run_arm 4 02 donated
run_arm 5 03 donated
run_arm 6 03 control

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.analyze_local_mstep_donation_ab \
    --root "${ROOT}" \
    --output "${ANALYSIS}" \
    --expected-repo-head "${EXPECTED_REPO_HEAD}" \
    --expected-gpu-uuid "${selected_gpu_uuid}"
) > "${RESULTS}/analysis.stdout" 2> "${RESULTS}/analysis.stderr"
test -s "${ANALYSIS}"
"${PIXI_PY}" - "${ANALYSIS}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "recovar.local_mstep_donation_ab_analysis.v1"
assert payload["passed"] is True
assert payload["same_source_exact_science_outputs"] is True
assert payload["same_declared_discrete_science_metadata"] is True
assert payload["arithmetic_changed"] is False
assert payload["production_xhalf_adjoint_program_observed"] is True
assert payload["alias_contract_passed"] is True
assert payload["default_promotion_allowed"] is False
PY

test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${cuda_sha256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
analysis_sha256=$(sha256sum "${ANALYSIS}" | awk '{print $1}')
junit_sha256=$(sha256sum "${RESULTS}/pytest.junit.xml" | awk '{print $1}')
"${PIXI_PY}" - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" "${source_manifest_sha256}" \
  "${cuda_sha256}" "${analysis_sha256}" "${junit_sha256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.local_mstep_donation_ab_slurm.v1",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "gpu_uuid": sys.argv[4],
    "source_manifest_sha256": sys.argv[5],
    "cuda_sha256": sys.argv[6],
    "analysis_sha256": sys.argv[7],
    "junit_sha256": sys.argv[8],
    "execution_order": [
        "donated-01", "control-01", "control-02",
        "donated-02", "donated-03", "control-03",
    ],
    "fresh_process_per_arm": True,
    "same_physical_h100": True,
    "default_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
touch "${ROOT}/COMPLETED"
