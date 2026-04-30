cmake_minimum_required(VERSION 3.15)
project(relion_bind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── RELION source tree (header-only dependency for now) ──────────────
set(RELION_SRC_DIR "/scratch/gpfs/GILLES/mg6942/relion/src"
    CACHE PATH "Path to RELION source (the 'src' directory)")

if(NOT EXISTS "${RELION_SRC_DIR}/projector.h")
    message(FATAL_ERROR "RELION_SRC_DIR=${RELION_SRC_DIR} does not contain projector.h")
endif()

# ── pybind11 ─────────────────────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)

# ── FFTW3 (header needed by RELION includes, library for FFT ops) ───
# Search pixi env first, then system
set(PIXI_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../.pixi/envs/default")
find_path(FFTW3_INCLUDE_DIR fftw3.h
    HINTS "${PIXI_PREFIX}/include/fftw" "${PIXI_PREFIX}/include"
    PATHS /usr/include /usr/local/include)

find_library(FFTW3_LIBRARY NAMES fftw3
    HINTS "${PIXI_PREFIX}/lib"
    PATHS /usr/lib64 /usr/lib /usr/local/lib)

if(NOT FFTW3_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find fftw3.h — set FFTW3_INCLUDE_DIR")
endif()
message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")
if(FFTW3_LIBRARY)
    message(STATUS "FFTW3 library: ${FFTW3_LIBRARY}")
else()
    message(STATUS "FFTW3 library: not found (link will be header-only)")
endif()

# ── CPU-only: suppress CUDA/HIP guards ──────────────────────────────
add_compile_definitions(
    # Suppress RELION's GPU code paths
    # Note: leave RELION_SINGLE_PRECISION undefined → RFLOAT=double
)

# ── RELION source files ─────────────────────────────────────────────
# Minimal set for Projector + BackProjector + supporting types.
# We compile these .cpp files directly (NOT linking against relion_lib).
set(RELION_SOURCES
    ${RELION_SRC_DIR}/projector.cpp
    ${RELION_SRC_DIR}/backprojector.cpp
    ${RELION_SRC_DIR}/fftw.cpp
    ${RELION_SRC_DIR}/euler.cpp
    ${RELION_SRC_DIR}/tabfuncs.cpp
    ${RELION_SRC_DIR}/funcs.cpp
    ${RELION_SRC_DIR}/args.cpp
    ${RELION_SRC_DIR}/error.cpp
    ${RELION_SRC_DIR}/filename.cpp
    ${RELION_SRC_DIR}/strings.cpp
    ${RELION_SRC_DIR}/metadata_table.cpp
    ${RELION_SRC_DIR}/metadata_container.cpp
    ${RELION_SRC_DIR}/symmetries.cpp
    ${RELION_SRC_DIR}/mask.cpp
    ${RELION_SRC_DIR}/complex.cpp
    ${RELION_SRC_DIR}/CPlot2D.cpp
    ${RELION_SRC_DIR}/time.cpp
    ${RELION_SRC_DIR}/pipeline_control.cpp
    ${RELION_SRC_DIR}/ctf.cpp
    ${RELION_SRC_DIR}/transformations.cpp
    ${RELION_SRC_DIR}/matrix1d.cpp
    ${RELION_SRC_DIR}/matrix2d.cpp
    ${RELION_SRC_DIR}/numerical_recipes.cpp
    ${RELION_SRC_DIR}/healpix_sampling.cpp
    ${RELION_SRC_DIR}/Healpix_2.15a/healpix_base.cc
    ${RELION_SRC_DIR}/Healpix_2.15a/cxxutils.cc
)

# ── The pybind11 module ─────────────────────────────────────────────
pybind11_add_module(_relion_bind_core
    ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/projector_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ctf_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fftw_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backprojector_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/estep_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/initialmodel_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/obs_model_stub.cpp
    ${RELION_SOURCES}
)

target_include_directories(_relion_bind_core PRIVATE
    # RELION headers use #include "src/xxx.h" — so include the parent
    "${RELION_SRC_DIR}/.."
    "${RELION_SRC_DIR}/Healpix_2.15a"
    "${FFTW3_INCLUDE_DIR}"
)

# Suppress CUDA/HIP via compiler flags (in case macros.h checks these)
target_compile_options(_relion_bind_core PRIVATE
    -U_CUDA_ENABLED -U_HIP_ENABLED
    -Wno-sign-compare -Wno-unused-variable -Wno-unused-parameter
)

if(FFTW3_LIBRARY)
    target_link_libraries(_relion_bind_core PRIVATE ${FFTW3_LIBRARY})
endif()

# Link system FFTW if .so symlink missing but .so.3 exists
if(NOT FFTW3_LIBRARY AND EXISTS "/usr/lib64/libfftw3.so.3")
    target_link_libraries(_relion_bind_core PRIVATE "/usr/lib64/libfftw3.so.3")
endif()

# Install into the relion_bind package directory
install(TARGETS _relion_bind_core DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
