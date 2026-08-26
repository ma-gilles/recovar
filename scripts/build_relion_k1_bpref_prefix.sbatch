#!/usr/bin/env bash
#SBATCH --job-name=k1-bpref-prefix-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_SOURCE_HEAD:?EXPECTED_SOURCE_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

SOURCE=${RUN_ROOT}/source
BUILD=${RUN_ROOT}/build_mpi_path_wrapper
LAUNCHER=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820/scripts/build_relion_k1_bpref_prefix.sbatch
PROVENANCE=${RUN_ROOT}/provenance

require_equal() {
    local label=$1 actual=$2 expected=$3
    if [[ "${actual}" != "${expected}" ]]; then
        printf 'provenance mismatch: %s actual=%q expected=%q\n' "${label}" "${actual}" "${expected}" >&2
        exit 2
    fi
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
require_equal source_head "$(git -C "${SOURCE}" rev-parse HEAD)" "${EXPECTED_SOURCE_HEAD}"
require_equal source_diff "$(git -C "${SOURCE}" diff --binary | sha256sum | awk '{print $1}')" "${EXPECTED_SOURCE_DIFF_SHA256}"
require_equal launcher "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" "${EXPECTED_LAUNCHER_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PROVENANCE}"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/gcc/4.1.6 cudatoolkit/12.6
set -u

git -C "${SOURCE}" status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff --binary > "${PROVENANCE}/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
command -v mpicc > "${PROVENANCE}/mpicc_${SLURM_JOB_ID}.txt"
command -v mpicxx > "${PROVENANCE}/mpicxx_${SLURM_JOB_ID}.txt"

cmake -S "${SOURCE}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${RUN_ROOT}/install" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCUDA=ON -DCUDA_ARCH=90 -DGUI=OFF -DBUILD_TESTS=OFF \
    -DFORCE_OWN_FFTW=ON \
    -DOWN_FFTW_INCLUDES=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include \
    -DOWN_FFTW_DOUBLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3.so \
    -DOWN_FFTW_SINGLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3f.so \
    > "${RUN_ROOT}/configure_${SLURM_JOB_ID}.log" 2>&1
cmake --build "${BUILD}" --target refine_mpi --parallel "${SLURM_CPUS_PER_TASK}" \
    > "${RUN_ROOT}/build_${SLURM_JOB_ID}.log" 2>&1

BINARY=${BUILD}/bin/relion_refine_mpi
test -x "${BINARY}"
test "$(strings "${BINARY}" | rg -c RELION_DUMP_K1_BPREF_PREFIX_DIR)" -eq 1
cmake -LA -N "${BUILD}" > "${PROVENANCE}/cmake_cache_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" > "${PROVENANCE}/binary_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_BUILD_${SLURM_JOB_ID}"
