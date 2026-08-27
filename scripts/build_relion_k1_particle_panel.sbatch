#!/usr/bin/env bash
#SBATCH --job-name=k1-particle-panel-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00

set -euo pipefail

failed_command=""
trap 'failed_command=${BASH_COMMAND}; printf "BUILD_SETUP_FAILED line=%s command=%q\n" "${LINENO}" "${failed_command}" >&2' ERR

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${BUILD_ROOT:?BUILD_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${MPI_DEV_ROOT:?MPI_DEV_ROOT is required}"
: "${CUDA_DEV_OVERLAY:?CUDA_DEV_OVERLAY is required}"
: "${EXPECTED_SOURCE_HEAD:?EXPECTED_SOURCE_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_MPI_WRAPPERS_SHA256:?EXPECTED_MPI_WRAPPERS_SHA256 is required}"
: "${EXPECTED_MPI_DEV_MANIFEST_SHA256:?EXPECTED_MPI_DEV_MANIFEST_SHA256 is required}"
: "${EXPECTED_CUDA_DEV_LIBS_SHA256:?EXPECTED_CUDA_DEV_LIBS_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
LAUNCHER=${REPO}/scripts/build_relion_k1_particle_panel.sbatch
CC_WRAPPER=${REPO}/scripts/relion_mpi_cc_wrapper.sh
CXX_WRAPPER=${REPO}/scripts/relion_mpi_cxx_wrapper.sh
BUILD=${BUILD_ROOT}/build
PROVENANCE=${BUILD_ROOT}/provenance

test -f "${SOURCE_ROOT}/SAFE_TO_DELETE"
test -f "${BUILD_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${MPI_DEV_ROOT}/SAFE_TO_DELETE"
test -d "${CUDA_DEV_OVERLAY}/cuda_include"
test -d "${CUDA_DEV_OVERLAY}/lib"
test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${EXPECTED_SOURCE_HEAD}"
test "$(git -C "${SOURCE_ROOT}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CC_WRAPPER}" "${CXX_WRAPPER}" | awk '{print $1}' | sha256sum | awk '{print $1}')" = "${EXPECTED_MPI_WRAPPERS_SHA256}"
test "$(sha256sum "${MPI_DEV_ROOT}/files.sha256" | awk '{print $1}')" = "${EXPECTED_MPI_DEV_MANIFEST_SHA256}"
sha256sum --quiet --check "${MPI_DEV_ROOT}/files.sha256"
test "$(sha256sum "${CUDA_DEV_OVERLAY}/lib/libcufft.so" "${CUDA_DEV_OVERLAY}/lib/libcurand.so" | awk '{print $1}' | sha256sum | awk '{print $1}')" = "${EXPECTED_CUDA_DEV_LIBS_SHA256}"
test -z "$(
    find "${BUILD_ROOT}" -mindepth 1 \
        ! -name SAFE_TO_DELETE \
        ! -name 'slurm-*.out' \
        -print -quit
)"
printf 'BUILD_SETUP_OK stage=immutable_inputs\n'

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
export PYTHONNOUSERSITE=1
export RELION_MPI_ROOT=${MPI_DEV_ROOT}
export CPATH=${CUDA_DEV_OVERLAY}/cuda_include:${CUDA_DEV_OVERLAY}/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=${CUDA_DEV_OVERLAY}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PROVENANCE}"
printf 'BUILD_SETUP_OK stage=runtime_roots\n'

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 cudatoolkit/12.6
set -u
printf 'BUILD_SETUP_OK stage=modules\n'
NVCC=$(command -v nvcc)
test -x "${CC_WRAPPER}"
test -x "${CXX_WRAPPER}"
test -r "${RELION_MPI_ROOT}/include/mpi.h"
test -r "${RELION_MPI_ROOT}/lib64/libmpi.so"
test -r "${RELION_MPI_ROOT}/lib64/libmpi_cxx.so"
test -x "${NVCC}"
printf 'BUILD_SETUP_OK stage=toolchain_paths\n'

git -C "${SOURCE_ROOT}" status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git -C "${SOURCE_ROOT}" diff --binary > "${PROVENANCE}/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
sha256sum "${CC_WRAPPER}" "${CXX_WRAPPER}" \
    > "${PROVENANCE}/mpi_wrappers_${SLURM_JOB_ID}.sha256"
"${NVCC}" --version > "${PROVENANCE}/nvcc_${SLURM_JOB_ID}.txt"
printf 'BUILD_SETUP_OK stage=toolchain_provenance\n'

cmake -S "${SOURCE_ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_ROOT}/install" \
    -DCMAKE_C_COMPILER="${CC_WRAPPER}" \
    -DCMAKE_CXX_COMPILER="${CXX_WRAPPER}" \
    -DCUDA=ON -DCUDA_ARCH=90 -DGUI=OFF -DBUILD_TESTS=OFF \
    -DCUDA_cufft_LIBRARY="${CUDA_DEV_OVERLAY}/lib/libcufft.so" \
    -DCUDA_curand_LIBRARY="${CUDA_DEV_OVERLAY}/lib/libcurand.so" \
    -DFETCH_WEIGHTS=OFF \
    -DFORCE_OWN_FFTW=ON \
    -DOWN_FFTW_INCLUDES=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include \
    -DOWN_FFTW_DOUBLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3.so \
    -DOWN_FFTW_SINGLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3f.so \
    > "${BUILD_ROOT}/configure_${SLURM_JOB_ID}.log" 2>&1
printf 'BUILD_SETUP_OK stage=configure\n'
cmake --build "${BUILD}" --target refine_mpi --parallel "${SLURM_CPUS_PER_TASK}" \
    > "${BUILD_ROOT}/build_${SLURM_JOB_ID}.log" 2>&1
printf 'BUILD_SETUP_OK stage=compile\n'

BINARY=${BUILD}/bin/relion_refine_mpi
test -x "${BINARY}"
test "$(strings "${BINARY}" | rg -c RELION_ACC_DUMP_PART_IDS)" -ge 1
test "$(strings "${BINARY}" | rg -c RELION_ACC_DUMP_BPREF_STATE)" -ge 1
test "$(strings "${BINARY}" | rg -c RELION_RESTORE_FRESH_PHYSICAL_ORDER)" -ge 1
cmake -LA -N "${BUILD}" > "${PROVENANCE}/cmake_cache_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" > "${PROVENANCE}/binary_${SLURM_JOB_ID}.sha256"
touch "${BUILD_ROOT}/SUCCESS_BUILD_${SLURM_JOB_ID}"
