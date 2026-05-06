/**
 * Projector bindings: griddingCorrect, project, computeFourierTransformMap.
 *
 * Phase 1: griddingCorrect (build smoke test).
 * Phase 2b: project() + computeFourierTransformMap().
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <src/projector.h>

namespace py = pybind11;

// ── M1: griddingCorrect ─────────────────────────────────────────────

static py::array_t<double> gridding_correct(
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_in,
    int ori_size,
    int padding_factor,
    int interpolator
) {
    auto buf = vol_in.request();
    if (buf.ndim != 3)
        throw std::runtime_error("vol_in must be 3D");

    long N = buf.shape[0];
    if (buf.shape[1] != N || buf.shape[2] != N)
        throw std::runtime_error("vol_in must be cubic (N, N, N)");

    py::array_t<double> result({N, N, N});
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, buf.ptr, N * N * N * sizeof(double));

    MultidimArray<RFLOAT> vol(N, N, N);
    std::memcpy(vol.data, rbuf.ptr, N * N * N * sizeof(RFLOAT));

    Projector proj(ori_size, interpolator, (RFLOAT)padding_factor, 10, 3);
    proj.griddingCorrect(vol);

    std::memcpy(rbuf.ptr, vol.data, N * N * N * sizeof(RFLOAT));
    return result;
}


// ── E1+E2: computeFourierTransformMap + project ─────────────────────

/**
 * Initialize a RELION Projector from a real-space volume and return the
 * projector-centered Fourier data for inspection.
 *
 * This calls Projector::computeFourierTransformMap() which:
 * 1. Optionally applies gridding correction
 * 2. Pads the volume by padding_factor
 * 3. FFTs and applies CenterFFTbySign
 * 4. Copies into the projector-centered storage (x≥0, yz-centered)
 */
static py::tuple compute_fourier_transform_map(
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_in,
    int ori_size,
    int padding_factor,
    int interpolator,
    int current_size,
    bool do_gridding,
    int data_dim
) {
    auto buf = vol_in.request();
    if (buf.ndim != 3)
        throw std::runtime_error("vol_in must be 3D");

    long N = buf.shape[0];

    // Copy into MultidimArray
    MultidimArray<RFLOAT> vol(N, N, N);
    std::memcpy(vol.data, (double*)buf.ptr, N * N * N * sizeof(RFLOAT));

    // Create projector
    Projector proj(ori_size, interpolator, (RFLOAT)padding_factor, 10, data_dim);

    // Compute power spectrum (required output parameter)
    MultidimArray<RFLOAT> power_spectrum;

    proj.computeFourierTransformMap(vol, power_spectrum, current_size, 1,
                                    do_gridding);

    // Extract projector data as numpy array
    // proj.data is MultidimArray<Complex> with shape (pad_size, pad_size, pad_size/2+1)
    long pz = ZSIZE(proj.data);
    long py_ = YSIZE(proj.data);
    long px = XSIZE(proj.data);

    // Complex → 2 doubles per element
    py::array_t<std::complex<double>> proj_data({pz, py_, px});
    auto proj_buf = proj_data.request();
    std::memcpy(proj_buf.ptr, proj.data.data,
                pz * py_ * px * sizeof(std::complex<double>));

    // Power spectrum
    long ps_size = XSIZE(power_spectrum);
    py::array_t<double> ps({ps_size});
    if (ps_size > 0) {
        auto ps_buf = ps.request();
        std::memcpy(ps_buf.ptr, power_spectrum.data, ps_size * sizeof(double));
    }

    return py::make_tuple(proj_data, ps,
                          (int)proj.ori_size,
                          (int)proj.padding_factor,
                          (int)proj.r_max,
                          (int)proj.r_min_nn,
                          (int)proj.interpolator);
}


/**
 * Project a volume (via its Projector) to get a 2D Fourier slice.
 *
 * Takes a real-space volume and a 3×3 rotation matrix, initializes
 * the Projector, and returns the 2D Fourier slice.
 *
 * The rotation matrix A should be in RELION convention (row-major 3×3).
 * Internally, RELION computes Ainv = A.inv() * padding_factor, then:
 *   xp = Ainv[0,0]*x + Ainv[0,1]*y
 *   yp = Ainv[1,0]*x + Ainv[1,1]*y
 *   zp = Ainv[2,0]*x + Ainv[2,1]*y
 * and trilinearly interpolates from projector data at (xp, yp, zp).
 */
static py::array_t<std::complex<double>> project_volume(
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotation_matrix,
    int ori_size,
    int padding_factor,
    int interpolator,
    int current_size,
    bool do_gridding,
    int data_dim,
    bool current_size_output
) {
    auto vol_buf = vol_in.request();
    if (vol_buf.ndim != 3)
        throw std::runtime_error("vol_in must be 3D");

    auto rot_buf = rotation_matrix.request();
    if (rot_buf.ndim != 2 || rot_buf.shape[0] != 3 || rot_buf.shape[1] != 3)
        throw std::runtime_error("rotation_matrix must be (3, 3)");

    long N = vol_buf.shape[0];

    // Copy volume into MultidimArray
    MultidimArray<RFLOAT> vol(N, N, N);
    std::memcpy(vol.data, (double*)vol_buf.ptr, N * N * N * sizeof(RFLOAT));

    // Create projector and initialize
    Projector proj(ori_size, interpolator, (RFLOAT)padding_factor, 10, data_dim);
    MultidimArray<RFLOAT> power_spectrum;
    proj.computeFourierTransformMap(vol, power_spectrum, current_size, 1,
                                    do_gridding);

    // Copy rotation matrix into RELION's Matrix2D
    Matrix2D<RFLOAT> A(3, 3);
    double* rot_ptr = (double*)rot_buf.ptr;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            MAT_ELEM(A, i, j) = rot_ptr[i * 3 + j];

    // Pre-allocate output (RELION project() uses DIRECT indexing, no Xmipp origin needed)
    MultidimArray<Complex> img_out;
    int output_size = current_size_output && current_size > 0 ? current_size : ori_size;
    img_out.resize(output_size, output_size / 2 + 1);
    img_out.initZeros();

    // Project
    proj.project(img_out, A);

    // Copy to numpy
    long out_y = YSIZE(img_out);
    long out_x = XSIZE(img_out);

    py::array_t<std::complex<double>> result({out_y, out_x});
    auto res_buf = result.request();
    std::memcpy(res_buf.ptr, img_out.data,
                out_y * out_x * sizeof(std::complex<double>));

    return result;
}


void init_projector_bindings(py::module_ &m) {
    // M1: gridding correction
    m.def("gridding_correct", &gridding_correct,
          py::arg("vol_in"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 1,
          py::arg("interpolator") = 1,
          R"doc(
Apply RELION's Projector::griddingCorrect to a real-space volume.

Divides each voxel by sinc²(r / (N * pf)) for trilinear interpolation,
or sinc(r / (N * pf)) for nearest-neighbour.

Parameters
----------
vol_in : ndarray, shape (N, N, N), float64
    Real-space volume.
ori_size : int
    RELION's ori_size (original box size before padding).
padding_factor : int
    Padding factor (1 or 2).
interpolator : int
    1 = TRILINEAR, 0 = NEAREST_NEIGHBOUR.

Returns
-------
ndarray, shape (N, N, N), float64
    Gridding-corrected volume.
)doc");

    // E1: computeFourierTransformMap
    m.def("compute_fourier_transform_map", &compute_fourier_transform_map,
          py::arg("vol_in"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 2,
          py::arg("interpolator") = 1,
          py::arg("current_size") = -1,
          py::arg("do_gridding") = true,
          py::arg("data_dim") = 2,
          R"doc(
Initialize RELION Projector from a real-space volume.

Returns (proj_data, power_spectrum, ori_size, padding_factor, r_max,
         r_min_nn, interpolator).

proj_data is the projector-centered Fourier storage:
shape (pad_size, pad_size, pad_size//2+1) complex128.
)doc");

    // E2: project
    m.def("project_volume", &project_volume,
          py::arg("vol_in"),
          py::arg("rotation_matrix"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 2,
          py::arg("interpolator") = 1,
          py::arg("current_size") = -1,
          py::arg("do_gridding") = true,
          py::arg("data_dim") = 2,
          py::arg("current_size_output") = false,
          R"doc(
Project a volume using RELION's Projector::project.

Parameters
----------
vol_in : ndarray (N, N, N) float64
    Real-space volume in RELION convention.
rotation_matrix : ndarray (3, 3) float64
    Rotation matrix in RELION convention.
ori_size : int
    Original box size.
padding_factor : int
    Padding factor (1 or 2).
interpolator : int
    1 = TRILINEAR.
current_size : int
    Resolution limit (-1 = full).
do_gridding : bool
    Apply gridding correction before FFT.
data_dim : int
    Projection data dimensionality. Use 2 for SPA 3D-reference-to-2D-image
    scoring; RELION changes projector normalisation based on this value.
current_size_output : bool
    If true, allocate the output Fourier image as
    (current_size, current_size//2+1), matching RELION E-step buffers.

Returns
-------
ndarray (ori_size, ori_size//2+1) complex128
    2D Fourier slice (half-complex, FFTW layout after CenterFFTbySign).
)doc");

    m.attr("NEAREST_NEIGHBOUR") = py::int_(NEAREST_NEIGHBOUR);
    m.attr("TRILINEAR") = py::int_(TRILINEAR);
}
