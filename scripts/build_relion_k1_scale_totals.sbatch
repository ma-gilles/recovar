#!/usr/bin/env bash
#SBATCH --job-name=relion-k1-scale-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_SOURCE_HEAD:?EXPECTED_SOURCE_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
SOURCE=${ROOT}/source
BUILD=${ROOT}/build_mpi_gcc
LAUNCHER=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820/scripts/build_relion_k1_scale_totals.sbatch

require_equal() {
    local label=$1 actual=$2 expected=$3
    if [[ "${actual}" != "${expected}" ]]; then
        printf 'provenance mismatch: %s actual=%q expected=%q\n' "${label}" "${actual}" "${expected}" >&2
        exit 2
    fi
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE" || {
    printf 'missing SAFE_TO_DELETE marker under %s or %s\n' "${ROOT}" "${RUNTIME}" >&2
    exit 2
}
require_equal source_head "$(git -C "${SOURCE}" rev-parse HEAD)" "${EXPECTED_SOURCE_HEAD}"
require_equal source_diff "$(git -C "${SOURCE}" diff --binary | sha256sum | awk '{print $1}')" "${EXPECTED_SOURCE_DIFF_SHA256}"
require_equal launcher "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" "${EXPECTED_LAUNCHER_SHA256}"
require_equal cuda_arch "$(cmake -LA -N "${BUILD}" | awk -F= '/^CUDA_ARCH:/{print $2}')" 90
require_equal build_type "$(cmake -LA -N "${BUILD}" | awk -F= '/^CMAKE_BUILD_TYPE:/{print $2}')" Release

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/gcc/4.1.6 cudatoolkit/12.6
set -u
export CPATH=/usr/local/cuda-12.6/include:/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include${CPATH:+:${CPATH}}

git -C "${SOURCE}" status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff --binary > "${ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
cmake -LA -N "${BUILD}" > "${ROOT}/provenance/cmake_cache_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"

cmake --build "${BUILD}" --target refine_mpi --parallel "${SLURM_CPUS_PER_TASK}" \
    > "${ROOT}/build_${SLURM_JOB_ID}.log" 2>&1

BINARY=${BUILD}/bin/relion_refine_mpi
test -x "${BINARY}"
test "$(strings "${BINARY}" | rg -c RELION_DUMP_K1_SCALE_TOTALS_DIR)" -eq 1
sha256sum "${BINARY}" > "${ROOT}/provenance/binary_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_BUILD_${SLURM_JOB_ID}"
