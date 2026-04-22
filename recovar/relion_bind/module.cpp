/**
 * RELION pybind11 bindings — main module definition.
 *
 * Compiles a minimal subset of RELION's C++ code (CPU-only, double precision)
 * and exposes subfunctions to Python for exact parity testing against recovar.
 *
 * Build: cmake -B build -S . && cmake --build build
 * Usage: from recovar.relion_bind._relion_bind_core import ...
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations — each .cpp file registers its own submodule
void init_projector_bindings(py::module_ &m);
void init_ctf_bindings(py::module_ &m);
void init_fftw_bindings(py::module_ &m);
void init_sampling_bindings(py::module_ &m);
void init_backprojector_bindings(py::module_ &m);
void init_estep_bindings(py::module_ &m);

PYBIND11_MODULE(_relion_bind_core, m) {
    m.doc() = "RELION C++ subfunctions exposed to Python for parity testing";

    init_projector_bindings(m);
    init_ctf_bindings(m);
    init_fftw_bindings(m);
    init_sampling_bindings(m);
    init_backprojector_bindings(m);
    init_estep_bindings(m);
}
