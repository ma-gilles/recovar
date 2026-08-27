#!/usr/bin/env bash
set -euo pipefail

mpi_root=${RELION_MPI_ROOT:-/usr/local/openmpi/cuda-12.6/4.1.6/gcc}
exec /usr/bin/gcc \
    -I"${mpi_root}/include" \
    "$@" \
    -L"${mpi_root}/lib64" \
    -Wl,-rpath,"${mpi_root}/lib64" \
    -lmpi
