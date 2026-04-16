/**
 * E-step standalone bindings for parity testing.
 *
 * These are faithful reimplementations of the inner-loop math from
 * ml_optimiser.cpp (which we cannot compile due to dependency surface).
 * Each function matches the RELION formula line-by-line.
 *
 * Binds:
 *   E5: convert_squared_differences_to_weights (ml_optimiser.cpp:7704)
 *   E7: compute_weighted_noise (ml_optimiser.cpp:8241, noise part)
 *   M5: find_current_resolution (ml_optimiser.cpp:5579)
 *   M9: update_noise_estimate (ml_optimiser.cpp:5246)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <src/macros.h>

namespace py = pybind11;


/**
 * E5: Convert squared differences to posterior weights.
 *
 * Reimplements convertAllSquaredDifferencesToWeights (ml_optimiser.cpp:7995-8007).
 * For each (orientation, translation):
 *   weight = pdf_orientation * pdf_offset * exp(-(diff2 - min_diff2))
 * Then normalize: posterior = weight / sum(weights)
 *
 * The overflow guard at diff2 > 700 (double) matches RELION exactly.
 */
static py::array_t<double> convert_squared_differences_to_weights(
    py::array_t<double, py::array::c_style | py::array::forcecast> diff2_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> orientation_prior_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset_prior_in,
    double min_diff2
) {
    auto diff2_buf = diff2_in.request();
    auto orient_buf = orientation_prior_in.request();
    auto offset_buf = offset_prior_in.request();

    if (diff2_buf.ndim != 2)
        throw std::runtime_error("diff2 must be (n_orient, n_trans)");

    long n_orient = diff2_buf.shape[0];
    long n_trans = diff2_buf.shape[1];

    if (orient_buf.shape[0] != n_orient)
        throw std::runtime_error("orientation_prior size must match n_orient");
    if (offset_buf.shape[0] != n_trans)
        throw std::runtime_error("offset_prior size must match n_trans");

    double* diff2 = (double*)diff2_buf.ptr;
    double* orient_prior = (double*)orient_buf.ptr;
    double* offset_prior = (double*)offset_buf.ptr;

    // Normalize priors by their means (ml_optimiser.cpp:7808-7871)
    double orient_mean = 0.0;
    for (long i = 0; i < n_orient; i++)
        orient_mean += orient_prior[i];
    orient_mean /= (double)n_orient;

    double offset_mean = 0.0;
    for (long j = 0; j < n_trans; j++)
        offset_mean += offset_prior[j];
    offset_mean /= (double)n_trans;

    py::array_t<double> result({n_orient, n_trans});
    double* out = (double*)result.request().ptr;

    double sum_weight = 0.0;

    for (long i = 0; i < n_orient; i++) {
        double pdf_orient = (orient_mean > 0.0)
            ? orient_prior[i] / orient_mean : 1.0;

        for (long j = 0; j < n_trans; j++) {
            double pdf_offset = (offset_mean > 0.0)
                ? offset_prior[j] / offset_mean : 1.0;

            double w = pdf_orient * pdf_offset;
            double d = diff2[i * n_trans + j] - min_diff2;

            // Overflow guard: ml_optimiser.cpp:7998-8002
            if (d > 700.0)
                w = 0.0;
            else
                w *= std::exp(-d);

            out[i * n_trans + j] = w;
            sum_weight += w;
        }
    }

    // Normalize to posteriors
    if (sum_weight > 0.0) {
        double inv_sum = 1.0 / sum_weight;
        for (long k = 0; k < n_orient * n_trans; k++)
            out[k] *= inv_sum;
    }

    return result;
}


/**
 * E7: Compute weighted noise residual per shell.
 *
 * Reimplements the noise part of storeWeightedSums (ml_optimiser.cpp:8628-8652).
 * For each pixel: sigma2_noise[shell] += weight * |ctf*proj - img|^2
 *
 * Takes pre-computed projections, images, CTF, and per-pixel weights.
 * shell_indices maps each (iy, ix) pixel to its frequency shell.
 */
static py::array_t<double> compute_weighted_noise(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> images_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> projections_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> ctf_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixel_weights_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> shell_indices_in,
    int n_shells
) {
    auto img_buf = images_in.request();
    auto proj_buf = projections_in.request();
    auto ctf_buf = ctf_in.request();
    auto pw_buf = pixel_weights_in.request();
    auto si_buf = shell_indices_in.request();

    if (img_buf.ndim != 3)
        throw std::runtime_error("images must be (N, ny, nx)");

    long n_images = img_buf.shape[0];
    long ny = img_buf.shape[1];
    long nx = img_buf.shape[2];
    long npix = ny * nx;

    std::complex<double>* imgs = (std::complex<double>*)img_buf.ptr;
    std::complex<double>* projs = (std::complex<double>*)proj_buf.ptr;
    double* ctf = (double*)ctf_buf.ptr;
    double* pw = (double*)pw_buf.ptr;
    int* shells = (int*)si_buf.ptr;

    py::array_t<double> result(n_shells);
    double* sigma2 = (double*)result.request().ptr;
    std::fill(sigma2, sigma2 + n_shells, 0.0);

    for (long i = 0; i < n_images; i++) {
        for (long p = 0; p < npix; p++) {
            int s = shells[p];
            if (s < 0 || s >= n_shells) continue;

            std::complex<double> img_val = imgs[i * npix + p];
            std::complex<double> proj_val = projs[i * npix + p];
            double c = ctf[i * npix + p];
            double w = pw[i * npix + p];

            double diff_real = c * proj_val.real() - img_val.real();
            double diff_imag = c * proj_val.imag() - img_val.imag();
            sigma2[s] += w * (diff_real * diff_real + diff_imag * diff_imag);
        }
    }

    return result;
}


/**
 * M5: Find current resolution from data_vs_prior array.
 *
 * Reimplements updateCurrentResolution (ml_optimiser.cpp:5579-5620).
 * Returns the last shell index where data_vs_prior >= 1.0.
 */
static int find_current_resolution(
    py::array_t<double, py::array::c_style | py::array::forcecast> data_vs_prior_in,
    int minres_shell
) {
    auto buf = data_vs_prior_in.request();
    long n = buf.shape[0];
    double* dvp = (double*)buf.ptr;

    int maxres = 0;
    for (long ires = 1; ires < n; ires++) {
        if (dvp[ires] < 1.0)
            break;
        maxres = (int)ires;
    }

    if (maxres < minres_shell)
        maxres = minres_shell;

    return maxres;
}


/**
 * M9: Update noise estimate per shell.
 *
 * Reimplements the noise update in maximizationOtherParameters
 * (ml_optimiser.cpp:5246-5286).
 *
 * sigma2_noise[n] = wsum_sigma2_noise[n] / (2 * sumw * Npix_per_shell[n])
 */
static py::array_t<double> update_noise_estimate(
    py::array_t<double, py::array::c_style | py::array::forcecast> wsum_sigma2_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> npix_per_shell_in,
    double sum_weight
) {
    auto ws_buf = wsum_sigma2_in.request();
    auto np_buf = npix_per_shell_in.request();

    long n = ws_buf.shape[0];
    double* wsum = (double*)ws_buf.ptr;
    double* npix = (double*)np_buf.ptr;

    py::array_t<double> result(n);
    double* out = (double*)result.request().ptr;

    for (long i = 0; i < n; i++) {
        double denom = 2.0 * sum_weight * npix[i];
        if (denom > 0.0) {
            out[i] = wsum[i] / denom;
        } else {
            out[i] = 0.0;
        }
        // Clamp to minimum (ml_optimiser.cpp:5278)
        if (out[i] < 1e-15)
            out[i] = 1e-15;
        // Fill holes from previous shell (ml_optimiser.cpp:5282)
        if (out[i] < 1e-14 && i > 0)
            out[i] = out[i - 1];
    }

    return result;
}


void init_estep_bindings(py::module_ &m) {
    m.def("convert_squared_differences_to_weights",
          &convert_squared_differences_to_weights,
          py::arg("diff2"),
          py::arg("orientation_prior"),
          py::arg("offset_prior"),
          py::arg("min_diff2"),
          R"doc(
E5: Convert squared differences to posterior weights.
diff2: (n_orient, n_trans) squared difference scores
orientation_prior: (n_orient,) prior on each orientation
offset_prior: (n_trans,) prior on each translation
min_diff2: minimum diff2 for numerical stability
Returns: (n_orient, n_trans) normalized posterior weights
)doc");

    m.def("compute_weighted_noise", &compute_weighted_noise,
          py::arg("images"),
          py::arg("projections"),
          py::arg("ctf"),
          py::arg("pixel_weights"),
          py::arg("shell_indices"),
          py::arg("n_shells"),
          R"doc(
E7: Compute weighted noise residual per Fourier shell.
images: (N, ny, nx) complex Fourier images
projections: (N, ny, nx) complex projected references
ctf: (N, ny, nx) CTF values
pixel_weights: (N, ny, nx) posterior weights per pixel
shell_indices: (ny, nx) int mapping pixel to shell index
n_shells: number of output shells
Returns: (n_shells,) accumulated weighted squared residuals
)doc");

    m.def("find_current_resolution", &find_current_resolution,
          py::arg("data_vs_prior"),
          py::arg("minres_shell") = 0,
          R"doc(
M5: Find highest shell where data_vs_prior >= 1.0.
Returns shell index (int).
)doc");

    m.def("update_noise_estimate", &update_noise_estimate,
          py::arg("wsum_sigma2"),
          py::arg("npix_per_shell"),
          py::arg("sum_weight"),
          R"doc(
M9: Update noise estimate from weighted residuals.
sigma2[n] = wsum_sigma2[n] / (2 * sum_weight * npix_per_shell[n])
Returns: (n_shells,) noise variance per shell.
)doc");
}
