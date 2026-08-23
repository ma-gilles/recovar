#!/usr/bin/env bash
#SBATCH --job-name=k1-ppref-iso
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_RELION_SOURCE_HEAD:?EXPECTED_RELION_SOURCE_HEAD is required}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?EXPECTED_RELION_SOURCE_DIFF_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
SOURCE=${RUN_ROOT}/source
BUILD=${BUILD_OVERRIDE:-${RUN_ROOT}/build}
LAUNCHER=${REPO}/scripts/build_relion_k1_ppref_capture_isolated.sbatch
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_SOURCE_HEAD}"
test "$(git -C "${SOURCE}" diff --name-only)" = src/acc/cuda/cuda_ml_optimiser.cu
test "$(git -C "${SOURCE}" diff -- src/acc/cuda/cuda_ml_optimiser.cu | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test -z "$(find "${BUILD}" -mindepth 1 -print -quit 2>/dev/null || true)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${BUILD}" \
  "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
MPI_WRAPPER_ROOT=${RELION_RUNTIME}/openmpi_dev_overlay
CUDA_HOME=/usr/local/cuda-12.6
MPI_INCLUDE=${RELION_RUNTIME}/openmpi_dev_overlay/include
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export CPATH=${RELION_RUNTIME}/system_dev_overlay/cuda_include${CPATH:+:${CPATH}}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff --binary > "${RUN_ROOT}/provenance/relion_source.patch"

cmake -S "${SOURCE}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release -DGUI=OFF -DCUDA=ON -DCUDA_ARCH=80 -DFETCH_WEIGHTS=OFF \
  -DCUDA_cufft_LIBRARY=${RELION_RUNTIME}/system_dev_overlay/lib/libcufft.so \
  -DCUDA_curand_LIBRARY=${RELION_RUNTIME}/system_dev_overlay/lib/libcurand.so \
  -DMPI_C_COMPILER=${MPI_WRAPPER_ROOT}/bin/mpicc \
  -DMPI_CXX_COMPILER=${MPI_WRAPPER_ROOT}/bin/mpicxx \
  -DMPI_C_COMPILER_INCLUDE_DIRS=${MPI_INCLUDE} \
  -DMPI_CXX_COMPILER_INCLUDE_DIRS=${MPI_INCLUDE} \
  -DMPI_C_HEADER_DIR=${MPI_INCLUDE} \
  -DMPI_CXX_HEADER_DIR=${MPI_INCLUDE} \
  -DMPI_C_LIB_NAMES=mpi \
  '-DMPI_CXX_LIB_NAMES=mpi_cxx;mpi' \
  -DMPI_mpi_LIBRARY=${MPI_ROOT}/lib64/libmpi.so \
  -DMPI_mpi_cxx_LIBRARY=${MPI_ROOT}/lib64/libmpi_cxx.so \
  -DFORCE_OWN_FFTW=OFF \
  -DOWN_FFTW_INCLUDES=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include \
  -DOWN_FFTW_DOUBLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3.so \
  -DOWN_FFTW_SINGLE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib/libfftw3f.so \
  > "${RUN_ROOT}/logs/configure.stdout" 2> "${RUN_ROOT}/logs/configure.stderr"
cmake --build "${BUILD}" --target refine_mpi -- -j4 \
  > "${RUN_ROOT}/logs/build.stdout" 2> "${RUN_ROOT}/logs/build.stderr"

BINARY=${BUILD}/bin/relion_refine_mpi
test -x "${BINARY}"
test "$(strings "${BINARY}" | grep -c RELION_PPREF_CAPTURE_DIR)" -eq 1
test "$(strings "${BINARY}" | grep -c 'ppref_iter%03d_rank%03d_model%03d.bin')" -eq 1
test "$(strings "${BINARY}" | grep -c RELION_SCORING_NOISE_CAPTURE)" -eq 1
sha256sum "${BINARY}" "${LAUNCHER}" "${RUN_ROOT}/provenance/relion_source.patch" \
  > "${RUN_ROOT}/provenance/build_outputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
