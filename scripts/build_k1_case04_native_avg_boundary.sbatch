#!/usr/bin/env bash
#SBATCH --job-name=k1c04-avg-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${K1_NATIVE_AVG_BUILD_ROOT:?set the disposable build root}"
: "${K1_NATIVE_AVG_BUILD_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${K1_NATIVE_AVG_BUILD_ROOT}
RUNTIME=${K1_NATIVE_AVG_BUILD_RUNTIME}
SOURCE=${ROOT}/source
BUILD=${ROOT}/build
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
LAUNCHER=${REPO}/scripts/build_k1_case04_native_avg_boundary.sbatch
OVERLAY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
MPI_INCLUDE=${OVERLAY}/openmpi_dev_overlay/include
GPU_LIB=${OVERLAY}/system_dev_overlay/lib
CUDA_INCLUDE=${OVERLAY}/system_dev_overlay/cuda_include
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
FFTW=/scratch/gpfs/GILLES/mg6942/relion/external/fftw

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${SOURCE}" diff HEAD --binary | sha256sum | awk '{print $1}')" = 80a6036450dc3b20f2548aae91a67b52a93608fd56837563f041860ed1777326
test "$(sha256sum "${SOURCE}/src/ml_optimiser.cpp" | awk '{print $1}')" = f020e099e26d7c6e0381dc6928312bf46cd29a22f4ab5ba69aad09c20e5c137b

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${BUILD}" \
  "${ROOT}/logs" "${ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u

sha256sum "${LAUNCHER}" "${SOURCE}/src/ml_optimiser.cpp" \
  > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git -C "${SOURCE}" diff HEAD --binary \
  > "${ROOT}/provenance/relion_source_${SLURM_JOB_ID}.patch"

cmake -S "${SOURCE}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_PATH="${GPU_LIB}" \
  -DCMAKE_CXX_FLAGS="-I${CUDA_INCLUDE}" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDA_cufft_LIBRARY:FILEPATH="${GPU_LIB}/libcufft.so" \
  -DCUDA_curand_LIBRARY:FILEPATH="${GPU_LIB}/libcurand.so" \
  -DMPI_C_COMPILER_INCLUDE_DIRS="${MPI_INCLUDE}" \
  -DMPI_CXX_COMPILER_INCLUDE_DIRS="${MPI_INCLUDE}" \
  -DMPI_C_HEADER_DIR="${MPI_INCLUDE}" \
  -DMPI_CXX_HEADER_DIR="${MPI_INCLUDE}" \
  -DMPI_C_LIB_NAMES=mpi \
  -DMPI_CXX_LIB_NAMES='mpi_cxx;mpi' \
  -DMPI_mpi_LIBRARY="${MPI_ROOT}/lib64/libmpi.so" \
  -DMPI_mpi_cxx_LIBRARY="${MPI_ROOT}/lib64/libmpi_cxx.so" \
  -DOWN_FFTW_INCLUDES:PATH="${FFTW}/include" \
  -DOWN_FFTW_DOUBLE:FILEPATH="${FFTW}/lib/libfftw3.so" \
  -DOWN_FFTW_SINGLE:FILEPATH="${FFTW}/lib/libfftw3f.so" \
  -DFETCH_WEIGHTS=OFF -DCUDA_ARCH=80 -DGUI=OFF -DFORCE_OWN_FFTW=OFF
cmake --build "${BUILD}" --target refine_mpi -j8

test -x "${BUILD}/bin/relion_refine_mpi"
sha256sum "${BUILD}/bin/relion_refine_mpi" "${BUILD}/lib/librelion_gpu_util.a" \
  > "${ROOT}/provenance/build_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/BUILD_SUCCESS_${SLURM_JOB_ID}"
