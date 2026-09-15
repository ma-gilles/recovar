#!/usr/bin/env bash
#SBATCH --job-name=vdam-worker-trace-build
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-worker-trace-build-%j.out

set -euo pipefail

: "${RELION_SOURCE:?set the isolated RELION worker-trace source worktree}"
: "${RELION_BUILD_ROOT:?set a disposable build/provenance root}"
: "${EXPECTED_RELION_BASE:?pin the uninstrumented RELION source commit}"
: "${EXPECTED_RELION_HEAD:?pin the worker-trace source commit}"

test -f "${RELION_BUILD_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${RELION_SOURCE}" merge-base "${EXPECTED_RELION_BASE}" HEAD)" = \
    "${EXPECTED_RELION_BASE}"
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1 --untracked-files=no)"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

BUILD=${RELION_BUILD_ROOT}/build
PROVENANCE=${RELION_BUILD_ROOT}/provenance
MPI_WRAPPER_ROOT=/scratch/gpfs/GILLES/mg6942/relion_runtime_build/openmpi_cuda126_gcc
MPI_LIBRARY_ROOT=${MPI_WRAPPER_ROOT}/lib64
CUDA_MATH_TARGET=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux
CUDA_HEADER_OVERLAY=/scratch/gpfs/GILLES/mg6942/relion_runtime_build/cuda126_headers/include
export CUDA_LIB_PATH=${CUDA_MATH_TARGET}
export CUDA_INC_PATH=${CUDA_HEADER_OVERLAY}
export FFTW_INCLUDE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include
export FFTW_LIB=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib
mkdir -p "${BUILD}" "${PROVENANCE}"

git -C "${RELION_SOURCE}" show --stat --oneline HEAD \
    > "${PROVENANCE}/source_head.txt"
git -C "${RELION_SOURCE}" diff "${EXPECTED_RELION_BASE}"..HEAD -- \
    src/ml_optimiser.cpp src/ml_optimiser.h \
    src/acc/cuda/cuda_ml_optimiser.cu \
    > "${PROVENANCE}/worker_trace.patch"

/usr/bin/cmake --fresh -S "${RELION_SOURCE}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_C_FLAGS:STRING="-I${MPI_WRAPPER_ROOT}/include -I${CUDA_HEADER_OVERLAY}" \
    -DCMAKE_CXX_FLAGS:STRING="-I${MPI_WRAPPER_ROOT}/include -I${CUDA_HEADER_OVERLAY}" \
    -DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,-rpath,${MPI_LIBRARY_ROOT}" \
    -DCMAKE_LIBRARY_PATH:STRING="${CUDA_MATH_TARGET}/lib" \
    -DCUDA_TOOLKIT_INCLUDE:PATH="${CUDA_HEADER_OVERLAY}" \
    -DCUDA_NVCC_FLAGS:STRING="-I${CUDA_HEADER_OVERLAY}" \
    -DMPI_C_COMPILER:FILEPATH="$(command -v gcc)" \
    -DMPI_CXX_COMPILER:FILEPATH="$(command -v g++)" \
    -DMPI_C_COMPILER_INCLUDE_DIRS:STRING="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_CXX_COMPILER_INCLUDE_DIRS:STRING="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_C_INCLUDE_DIRS:STRING="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_CXX_INCLUDE_DIRS:STRING="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_C_HEADER_DIR:PATH="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_CXX_HEADER_DIR:PATH="${MPI_WRAPPER_ROOT}/include" \
    -DMPI_C_LIB_NAMES:STRING=mpi \
    -DMPI_CXX_LIB_NAMES:STRING='mpi_cxx;mpi' \
    -DMPI_mpi_LIBRARY:FILEPATH="${MPI_LIBRARY_ROOT}/libmpi.so" \
    -DMPI_mpi_cxx_LIBRARY:FILEPATH="${MPI_LIBRARY_ROOT}/libmpi_cxx.so" \
    -DCUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    -DCUDA_cufft_LIBRARY:FILEPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib/libcufft.so.11.3.0.4 \
    -DCUDA_curand_LIBRARY:FILEPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib/libcurand.so.10.3.7.77 \
    -DCUDA_ARCH=80 -DGUI=OFF -DBUILD_TESTS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
/usr/bin/cmake --build "${BUILD}" --target refine --parallel "${SLURM_CPUS_PER_TASK}"

BINARY=${BUILD}/bin/relion_refine
test -x "${BINARY}"
strings "${BINARY}" | grep -Fx RELION_VDAM_WORKER_LOG_SCHEMA_V2 \
    > "${PROVENANCE}/schema_marker.txt"
sha256sum "${BINARY}" "${PROVENANCE}/worker_trace.patch" \
    > "${PROVENANCE}/build.sha256"
printf '{"job_id":"%s","source_base":"%s","source_head":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_RELION_BASE}" "${EXPECTED_RELION_HEAD}" \
    > "${PROVENANCE}/build_job.json"
