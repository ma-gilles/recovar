/**
 * FFTW utility bindings: windowFourierTransform, shiftImageInFourierTransform.
 *
 * Phase 2 (P3): windowFourierTransform — RELION's current_size cropping.
 * Phase 3 (E3): shiftImageInFourierTransformWithTabSincos (future).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <src/fftw.h>

namespace py = pybind11;

/**
 * Crop/pad a 2D FFTW half-transform to a new dimension.
 *
 * This is RELION's windowFourierTransform for 2D images.
 * Input: (oriydim, orixdim/2+1) complex half-transform.
 * Output: (newdim, newdim/2+1) complex half-transform.
 */
static py::array_t<std::complex<double>> window_fourier_transform_2d(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> ft_in,
    int newdim
) {
    auto buf = ft_in.request();
    if (buf.ndim != 2)
        throw std::runtime_error("ft_in must be 2D");

    long ny_in = buf.shape[0];
    long nx_in = buf.shape[1];

    // Copy into MultidimArray<Complex>
    MultidimArray<Complex> in_arr(ny_in, nx_in);
    std::memcpy(in_arr.data, buf.ptr, ny_in * nx_in * sizeof(Complex));

    MultidimArray<Complex> out_arr;
    windowFourierTransform(in_arr, out_arr, newdim);

    long ny_out = YSIZE(out_arr);
    long nx_out = XSIZE(out_arr);

    py::array_t<std::complex<double>> result({ny_out, nx_out});
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, out_arr.data,
                ny_out * nx_out * sizeof(std::complex<double>));
    return result;
}

/**
 * Crop/pad a 3D FFTW half-transform to a new dimension.
 */
static py::array_t<std::complex<double>> window_fourier_transform_3d(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> ft_in,
    int newdim
) {
    auto buf = ft_in.request();
    if (buf.ndim != 3)
        throw std::runtime_error("ft_in must be 3D");

    long nz_in = buf.shape[0];
    long ny_in = buf.shape[1];
    long nx_in = buf.shape[2];

    MultidimArray<Complex> in_arr(nz_in, ny_in, nx_in);
    std::memcpy(in_arr.data, buf.ptr,
                nz_in * ny_in * nx_in * sizeof(Complex));

    MultidimArray<Complex> out_arr;
    windowFourierTransform(in_arr, out_arr, newdim);

    long nz_out = ZSIZE(out_arr);
    long ny_out = YSIZE(out_arr);
    long nx_out = XSIZE(out_arr);

    py::array_t<std::complex<double>> result({nz_out, ny_out, nx_out});
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, out_arr.data,
                nz_out * ny_out * nx_out * sizeof(std::complex<double>));
    return result;
}


/**
 * Shift a 2D image in Fourier space + optional window to newdim.
 *
 * This matches RELION's shiftImageInFourierTransformWithTabSincos
 * but uses std::sin/cos instead of tabulated lookups.
 *
 * Shifts are in pixels (not normalized). RELION divides by -oridim
 * internally, and uses FFTW frequency indices (ip, jp) for the dot product.
 */
static py::array_t<std::complex<double>> shift_image_in_fourier_transform_2d(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> ft_in,
    double oridim,
    long newdim,
    double xshift,
    double yshift
) {
    auto buf = ft_in.request();
    if (buf.ndim != 2)
        throw std::runtime_error("ft_in must be 2D");

    long ny_in = buf.shape[0];
    long nx_in = buf.shape[1];

    // Use RELION's actual function via MultidimArray
    MultidimArray<Complex> in_arr(ny_in, nx_in);
    std::memcpy(in_arr.data, buf.ptr, ny_in * nx_in * sizeof(Complex));

    MultidimArray<Complex> out_arr;

    // Use shiftImageInFourierTransform (non-tabulated, no window)
    // But that version has a bug with raw indices. Use the manual approach
    // matching the tabulated version's logic exactly.
    long newhdim = newdim / 2 + 1;
    out_arr.initZeros(newdim, newhdim);

    double xs = xshift / (-oridim);
    double ys = yshift / (-oridim);
    double twopi = 2.0 * M_PI;

    // FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D logic:
    // i goes 0..newdim-1, j goes 0..newhdim-1
    // ip = (i < newhdim) ? i : (i - newdim)   [FFTW y-frequency]
    // jp = j                                    [FFTW x-frequency, non-negative]
    for (long i = 0; i < newdim; i++) {
        long ip = (i < newhdim) ? i : (i - newdim);
        for (long j = 0; j < newhdim; j++) {
            long jp = j;
            double dotp = twopi * (jp * xs + ip * ys);
            double a = cos(dotp);
            double b = sin(dotp);
            double c = DIRECT_A2D_ELEM(in_arr, i, j).real;
            double d = DIRECT_A2D_ELEM(in_arr, i, j).imag;
            double ac = a * c;
            double bd = b * d;
            double ab_cd = (a + b) * (c + d);
            DIRECT_A2D_ELEM(out_arr, i, j) = Complex(ac - bd, ab_cd - ac - bd);
        }
    }

    py::array_t<std::complex<double>> result({newdim, newhdim});
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, out_arr.data,
                newdim * newhdim * sizeof(std::complex<double>));
    return result;
}


void init_fftw_bindings(py::module_ &m) {
    m.def("window_fourier_transform_2d", &window_fourier_transform_2d,
          py::arg("ft_in"),
          py::arg("newdim"),
          R"doc(
Crop/pad a 2D FFTW half-transform to newdim.

Input shape: (oriydim, orixdim/2+1), complex128.
Output shape: (newdim, newdim/2+1), complex128.

This is RELION's windowFourierTransform used for current_size cropping.
)doc");

    m.def("window_fourier_transform_3d", &window_fourier_transform_3d,
          py::arg("ft_in"),
          py::arg("newdim"),
          R"doc(
Crop/pad a 3D FFTW half-transform to newdim.

Input shape: (nz, ny, nx), complex128.
Output shape: (newdim, newdim, newdim/2+1), complex128.
)doc");

    m.def("shift_image_in_fourier_transform_2d",
          &shift_image_in_fourier_transform_2d,
          py::arg("ft_in"),
          py::arg("oridim"),
          py::arg("newdim"),
          py::arg("xshift"),
          py::arg("yshift"),
          R"doc(
Shift a 2D image in Fourier space, optionally cropping to newdim.

Matches RELION's shiftImageInFourierTransformWithTabSincos.
Shifts are in pixels (RELION divides by -oridim internally).

Input shape: (oriydim, orixdim/2+1), complex128.
Output shape: (newdim, newdim/2+1), complex128.
)doc");
}
