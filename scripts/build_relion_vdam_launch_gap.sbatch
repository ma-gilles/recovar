#!/usr/bin/env bash
#SBATCH --job-name=vdam-launch-gap-build
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-launch-gap-build-%j.out

set -euo pipefail

: "${RELION_SOURCE:?set the isolated launch-gap RELION source worktree}"
: "${RELION_BUILD_ROOT:?set a new SAFE_TO_DELETE build root}"
: "${EXPECTED_RELION_HEAD:?pin the instrumented RELION source commit}"

test -f "${RELION_BUILD_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${RELION_BUILD_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1 --untracked-files=no)"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

BUILD=${RELION_BUILD_ROOT}/build
PROVENANCE=${RELION_BUILD_ROOT}/provenance
MPI_ROOT=/scratch/gpfs/GILLES/mg6942/relion_runtime_build/openmpi_cuda126_gcc
MPI_LIB=${MPI_ROOT}/lib64
CUDA_MATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux
CUDA_HEADERS=/scratch/gpfs/GILLES/mg6942/relion_runtime_build/cuda126_headers/include
export CUDA_LIB_PATH=${CUDA_MATH}
export CUDA_INC_PATH=${CUDA_HEADERS}
export FFTW_INCLUDE=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/include
export FFTW_LIB=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib
mkdir -p "${BUILD}" "${PROVENANCE}"

git -C "${RELION_SOURCE}" show --stat --oneline HEAD > "${PROVENANCE}/source_head.txt"
git -C "${RELION_SOURCE}" show --format=fuller --binary HEAD -- \
    src/acc/acc_ml_optimiser_impl.h > "${PROVENANCE}/launch_gap.patch"

/usr/bin/cmake --fresh -S "${RELION_SOURCE}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_C_FLAGS:STRING="-I${MPI_ROOT}/include -I${CUDA_HEADERS}" \
    -DCMAKE_CXX_FLAGS:STRING="-I${MPI_ROOT}/include -I${CUDA_HEADERS}" \
    -DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,-rpath,${MPI_LIB}" \
    -DCMAKE_LIBRARY_PATH:STRING="${CUDA_MATH}/lib" \
    -DCUDA_TOOLKIT_INCLUDE:PATH="${CUDA_HEADERS}" \
    -DCUDA_NVCC_FLAGS:STRING="-I${CUDA_HEADERS}" \
    -DMPI_C_COMPILER:FILEPATH="$(command -v gcc)" \
    -DMPI_CXX_COMPILER:FILEPATH="$(command -v g++)" \
    -DMPI_C_COMPILER_INCLUDE_DIRS:STRING="${MPI_ROOT}/include" \
    -DMPI_CXX_COMPILER_INCLUDE_DIRS:STRING="${MPI_ROOT}/include" \
    -DMPI_C_INCLUDE_DIRS:STRING="${MPI_ROOT}/include" \
    -DMPI_CXX_INCLUDE_DIRS:STRING="${MPI_ROOT}/include" \
    -DMPI_C_HEADER_DIR:PATH="${MPI_ROOT}/include" \
    -DMPI_CXX_HEADER_DIR:PATH="${MPI_ROOT}/include" \
    -DMPI_C_LIB_NAMES:STRING=mpi \
    -DMPI_CXX_LIB_NAMES:STRING='mpi_cxx;mpi' \
    -DMPI_mpi_LIBRARY:FILEPATH="${MPI_LIB}/libmpi.so" \
    -DMPI_mpi_cxx_LIBRARY:FILEPATH="${MPI_LIB}/libmpi_cxx.so" \
    -DCUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    -DCUDA_cufft_LIBRARY:FILEPATH=${CUDA_MATH}/lib/libcufft.so.11.3.0.4 \
    -DCUDA_curand_LIBRARY:FILEPATH=${CUDA_MATH}/lib/libcurand.so.10.3.7.77 \
    -DCUDA_ARCH=80 -DGUI=OFF -DBUILD_TESTS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
/usr/bin/cmake --build "${BUILD}" --target refine --parallel "${SLURM_CPUS_PER_TASK}"

BINARY=${BUILD}/bin/relion_refine
test -x "${BINARY}"
strings "${BINARY}" | grep -F wavg_to_bpref_host_gap_ns \
    > "${PROVENANCE}/schema_marker.txt"
sha256sum "${BINARY}" "${PROVENANCE}/launch_gap.patch" \
    > "${PROVENANCE}/build.sha256"
printf '{"job_id":"%s","source_head":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_RELION_HEAD}" \
    > "${PROVENANCE}/build_job.json"

