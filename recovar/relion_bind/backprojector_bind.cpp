/**
 * BackProjector bindings for parity testing.
 *
 * Phase 4: M-step backprojection + reconstruction + FSC.
 *
 * Binds:
 *   - backproject_images: accumulate weighted 2D images into 3D Fourier
 *   - reconstruct_volume: Wiener solve → real-space volume
 *   - compute_fsc: gold-standard FSC between two half-maps
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <src/backprojector.h>
#include <src/fftw.h>

namespace py = pybind11;


/**
 * Backproject a set of 2D Fourier images into 3D, then reconstruct.
 *
 * Takes N images (FFTW half-complex), N rotation matrices, optional
 * per-pixel CTF weights, and regularization tau2 per shell.
 *
 * Returns the reconstructed real-space volume (ori_size^3).
 */
static py::array_t<double> backproject_and_reconstruct(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> images,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau2,
    int ori_size,
    int padding_factor,
    int interpolator,
    bool do_map,
    int max_iter_preweight,
    RFLOAT tau2_fudge,
    bool skip_gridding
) {
    auto img_buf = images.request();
    auto rot_buf = rotations.request();
    auto wt_buf = weights.request();
    auto tau_buf = tau2.request();

    if (img_buf.ndim != 3)
        throw std::runtime_error("images must be (N, ori_size, ori_size/2+1)");
    if (rot_buf.ndim != 3 || rot_buf.shape[1] != 3 || rot_buf.shape[2] != 3)
        throw std::runtime_error("rotations must be (N, 3, 3)");

    long n_images = img_buf.shape[0];
    long ny = img_buf.shape[1];
    long nx = img_buf.shape[2];

    if (rot_buf.shape[0] != n_images)
        throw std::runtime_error("rotations count must match images count");

    bool has_weights = (wt_buf.ndim == 3 && wt_buf.shape[0] == n_images);

    // Create BackProjector (C1 symmetry)
    BackProjector bp(ori_size, 3, "C1", interpolator, (float)padding_factor,
                     10, 0, 1.9, 15, 2, skip_gridding);
    bp.initZeros(-1);

    // Accumulate images
    for (long i = 0; i < n_images; i++) {
        MultidimArray<Complex> img_arr(ny, nx);
        std::complex<double>* src = (std::complex<double>*)img_buf.ptr + i * ny * nx;
        std::memcpy(img_arr.data, src, ny * nx * sizeof(Complex));

        Matrix2D<RFLOAT> A(3, 3);
        double* rot_ptr = (double*)rot_buf.ptr + i * 9;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                MAT_ELEM(A, r, c) = rot_ptr[r * 3 + c];

        if (has_weights) {
            MultidimArray<RFLOAT> wt_arr(ny, nx);
            double* wt_ptr = (double*)wt_buf.ptr + i * ny * nx;
            std::memcpy(wt_arr.data, wt_ptr, ny * nx * sizeof(RFLOAT));
            bp.set2DFourierTransform(img_arr, A, &wt_arr);
        } else {
            bp.set2DFourierTransform(img_arr, A);
        }
    }

    // Prepare tau2 as MultidimArray
    long n_shells = tau_buf.shape[0];
    MultidimArray<RFLOAT> tau2_arr(n_shells);
    double* tau_ptr = (double*)tau_buf.ptr;
    for (long s = 0; s < n_shells; s++)
        DIRECT_A1D_ELEM(tau2_arr, s) = tau_ptr[s];

    // Reconstruct
    MultidimArray<RFLOAT> vol_out;
    bp.reconstruct(vol_out, max_iter_preweight, do_map, tau2_arr,
                   tau2_fudge, 1.0, -1, false);

    // Copy to numpy
    long nz = ZSIZE(vol_out);
    long out_ny = YSIZE(vol_out);
    long out_nx = XSIZE(vol_out);
    py::array_t<double> result({nz, out_ny, out_nx});
    auto res_buf = result.request();
    std::memcpy(res_buf.ptr, vol_out.data,
                nz * out_ny * out_nx * sizeof(double));

    return result;
}


/**
 * Compute FSC between two sets of backprojected images.
 *
 * Each set is backprojected separately, then getDownsampledAverage is
 * called on each, and calculateDownSampledFourierShellCorrelation computes
 * the FSC.
 *
 * Returns per-shell FSC values.
 */
static py::array_t<double> compute_fsc_from_halfsets(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> images_h1,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotations_h1,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> images_h2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotations_h2,
    int ori_size,
    int padding_factor,
    int interpolator
) {
    auto build_bp = [&](
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>& imgs,
        py::array_t<double, py::array::c_style | py::array::forcecast>& rots
    ) -> BackProjector {
        auto img_buf = imgs.request();
        auto rot_buf = rots.request();
        long n = img_buf.shape[0];
        long ny = img_buf.shape[1];
        long nx = img_buf.shape[2];

        BackProjector bp(ori_size, 3, "C1", interpolator, (float)padding_factor,
                         10, 0, 1.9, 15, 2, false);
        bp.initZeros(-1);

        for (long i = 0; i < n; i++) {
            MultidimArray<Complex> img_arr(ny, nx);
            std::complex<double>* src = (std::complex<double>*)img_buf.ptr + i * ny * nx;
            std::memcpy(img_arr.data, src, ny * nx * sizeof(Complex));

            Matrix2D<RFLOAT> A(3, 3);
            double* rot_ptr = (double*)rot_buf.ptr + i * 9;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    MAT_ELEM(A, r, c) = rot_ptr[r * 3 + c];

            bp.set2DFourierTransform(img_arr, A);
        }
        return bp;
    };

    BackProjector bp1 = build_bp(images_h1, rotations_h1);
    BackProjector bp2 = build_bp(images_h2, rotations_h2);

    MultidimArray<Complex> avg1, avg2;
    bp1.getDownsampledAverage(avg1);
    bp2.getDownsampledAverage(avg2);

    MultidimArray<RFLOAT> fsc;
    bp1.calculateDownSampledFourierShellCorrelation(avg1, avg2, fsc);

    long n_shells = XSIZE(fsc);
    py::array_t<double> result(n_shells);
    auto res_buf = result.request();
    std::memcpy(res_buf.ptr, fsc.data, n_shells * sizeof(double));

    return result;
}


/**
 * Raw backprojection only (no reconstruction).
 *
 * Returns the accumulated (data, weight) arrays from BackProjector.
 * Useful for inspecting intermediate state before Wiener solve.
 */
static std::tuple<py::array_t<std::complex<double>>, py::array_t<double>>
get_backprojector_data(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> images,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    int ori_size,
    int padding_factor,
    int interpolator
) {
    auto img_buf = images.request();
    auto rot_buf = rotations.request();
    auto wt_buf = weights.request();

    long n_images = img_buf.shape[0];
    long ny = img_buf.shape[1];
    long nx = img_buf.shape[2];

    bool has_weights = (wt_buf.ndim == 3 && wt_buf.shape[0] == n_images);

    BackProjector bp(ori_size, 3, "C1", interpolator, (float)padding_factor,
                     10, 0, 1.9, 15, 2, false);
    bp.initZeros(-1);

    for (long i = 0; i < n_images; i++) {
        MultidimArray<Complex> img_arr(ny, nx);
        std::complex<double>* src = (std::complex<double>*)img_buf.ptr + i * ny * nx;
        std::memcpy(img_arr.data, src, ny * nx * sizeof(Complex));

        Matrix2D<RFLOAT> A(3, 3);
        double* rot_ptr = (double*)rot_buf.ptr + i * 9;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                MAT_ELEM(A, r, c) = rot_ptr[r * 3 + c];

        if (has_weights) {
            MultidimArray<RFLOAT> wt_arr(ny, nx);
            double* wt_ptr = (double*)wt_buf.ptr + i * ny * nx;
            std::memcpy(wt_arr.data, wt_ptr, ny * nx * sizeof(RFLOAT));
            bp.set2DFourierTransform(img_arr, A, &wt_arr);
        } else {
            bp.set2DFourierTransform(img_arr, A);
        }
    }

    // Extract data array
    long dz = ZSIZE(bp.data);
    long dy = YSIZE(bp.data);
    long dx = XSIZE(bp.data);
    py::array_t<std::complex<double>> data_out({dz, dy, dx});
    std::memcpy(data_out.request().ptr, bp.data.data,
                dz * dy * dx * sizeof(std::complex<double>));

    // Extract weight array
    long wz = ZSIZE(bp.weight);
    long wy = YSIZE(bp.weight);
    long wx = XSIZE(bp.weight);
    py::array_t<double> weight_out({wz, wy, wx});
    std::memcpy(weight_out.request().ptr, bp.weight.data,
                wz * wy * wx * sizeof(double));

    return std::make_tuple(data_out, weight_out);
}


void init_backprojector_bindings(py::module_ &m) {
    m.def("backproject_and_reconstruct", &backproject_and_reconstruct,
          py::arg("images"),
          py::arg("rotations"),
          py::arg("weights"),
          py::arg("tau2"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 2,
          py::arg("interpolator") = TRILINEAR,
          py::arg("do_map") = true,
          py::arg("max_iter_preweight") = 10,
          py::arg("tau2_fudge") = 1.0,
          py::arg("skip_gridding") = false,
          R"doc(
Backproject 2D Fourier images and reconstruct a 3D volume.
images: (N, ori_size, ori_size/2+1) complex
rotations: (N, 3, 3) rotation matrices (RELION convention)
weights: (N, ori_size, ori_size/2+1) or empty — per-pixel weights (CTF²/σ²)
tau2: per-shell regularization values
Returns: (ori_size, ori_size, ori_size) real-space volume
)doc");

    m.def("compute_fsc_from_halfsets", &compute_fsc_from_halfsets,
          py::arg("images_h1"),
          py::arg("rotations_h1"),
          py::arg("images_h2"),
          py::arg("rotations_h2"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 2,
          py::arg("interpolator") = TRILINEAR,
          R"doc(
Compute FSC between two half-set backprojections.
Returns per-shell FSC values.
)doc");

    m.def("get_backprojector_data", &get_backprojector_data,
          py::arg("images"),
          py::arg("rotations"),
          py::arg("weights"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 2,
          py::arg("interpolator") = TRILINEAR,
          R"doc(
Raw backprojection: returns (data, weight) arrays before Wiener solve.
data: complex (pf*ori_size, pf*ori_size, pf*ori_size/2+1) projector-centered
weight: real, same shape
)doc");
}
