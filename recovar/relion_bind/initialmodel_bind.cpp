/**
 * InitialModel / VDAM bindings for RELION parity.
 *
 * Two surfaces are exposed:
 *
 *  1. VDAM moment primitives — the `BackProjector` members that drive the
 *     gradient update (`reweightGrad`, `getFristMoment`, `getSecondMoment`,
 *     `applyMomenta`, `reconstructGrad`). These are routed through the real
 *     RELION classes.
 *
 *  2. VDAM scheduler free functions — `updateSubsetSize`, `updateStepSize`,
 *     `updateTau2Fudge`, `randomiseParticlesOrder`. These are re-expressed
 *     as free C++ functions whose body is copy-verbatim from
 *     `ml_optimiser.cpp` / `exp_model.cpp` so they compile the same C++
 *     code without requiring a full `MlOptimiser` construction. The Python
 *     schedule implementations at `recovar/em/initial_model/schedules.py`
 *     are validated against these bindings in
 *     `tests/unit/test_relion_bind/test_initialmodel_bind.py`.
 *
 * Usage:
 *   from recovar.relion_bind._relion_bind_core import (
 *       vdam_reweight_grad, vdam_first_moment, vdam_second_moment,
 *       vdam_apply_momenta, vdam_reconstruct_grad,
 *       vdam_compute_subset_size, vdam_compute_stepsize,
 *       vdam_compute_tau2_fudge, vdam_randomise_particles_order,
 *   )
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <src/backprojector.h>
#include <src/fftw.h>
#include <src/funcs.h>       // for init_random_generator / rnd_unif

namespace py = pybind11;


// ===========================================================================
//  Helpers
// ===========================================================================

static BackProjector make_empty_backprojector(
    int ori_size,
    int padding_factor,
    const std::string &sym,
    int interpolator
) {
    BackProjector bp(ori_size, 3, sym, interpolator, (float)padding_factor,
                     10, 0, 1.9, 15, 2, false);
    bp.initZeros(-1);
    return bp;
}


static MultidimArray<Complex> numpy_to_complex_3d(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr
) {
    auto buf = arr.request();
    if (buf.ndim != 3)
        throw std::runtime_error("expected 3D complex array (k, i, j)");
    long kdim = buf.shape[0];
    long idim = buf.shape[1];
    long jdim = buf.shape[2];
    MultidimArray<Complex> out(kdim, idim, jdim);
    std::memcpy(out.data, buf.ptr, kdim * idim * jdim * sizeof(Complex));
    return out;
}


static py::array_t<std::complex<double>> complex_3d_to_numpy(
    const MultidimArray<Complex> &arr
) {
    long kdim = ZSIZE(arr);
    long idim = YSIZE(arr);
    long jdim = XSIZE(arr);
    py::array_t<std::complex<double>> out({kdim, idim, jdim});
    std::memcpy(out.request().ptr, arr.data,
                kdim * idim * jdim * sizeof(Complex));
    return out;
}


static py::array_t<double> real_1d_to_numpy(const MultidimArray<RFLOAT> &arr) {
    long n = XSIZE(arr);
    py::array_t<double> out(n);
    std::memcpy(out.request().ptr, arr.data, n * sizeof(RFLOAT));
    return out;
}


// ===========================================================================
//  VDAM moment primitives (pass-through to RELION's BackProjector)
// ===========================================================================


/**
 * Wrap `BackProjector::reweightGrad`.
 *
 * Needs (data, weight) buffers from a previous backprojection accumulation.
 * We reconstruct a minimal BackProjector and inject the buffers, then call
 * reweightGrad and return the updated data.
 *
 * Source: backprojector.cpp:1933.
 */
static py::array_t<std::complex<double>> vdam_reweight_grad(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> weight_in,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.data = numpy_to_complex_3d(data_in);
    // weight is real
    auto wt_buf = weight_in.request();
    bp.weight.resize(wt_buf.shape[0], wt_buf.shape[1], wt_buf.shape[2]);
    std::memcpy(bp.weight.data, wt_buf.ptr,
                wt_buf.shape[0] * wt_buf.shape[1] * wt_buf.shape[2] * sizeof(RFLOAT));
    if (r_max > 0)
        bp.r_max = r_max;

    bp.reweightGrad();

    return complex_3d_to_numpy(bp.data);
}


/**
 * Wrap `BackProjector::getFristMoment`.
 *
 * Source: backprojector.cpp:1943.
 */
static py::array_t<std::complex<double>> vdam_first_moment(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mom_in,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max,
    double lambda
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.data = numpy_to_complex_3d(data_in);
    if (r_max > 0)
        bp.r_max = r_max;

    MultidimArray<Complex> mom = numpy_to_complex_3d(mom_in);

    bp.getFristMoment(mom, (RFLOAT)lambda);
    return complex_3d_to_numpy(mom);
}


/**
 * Wrap `BackProjector::getSecondMoment`.
 *
 * Source: backprojector.cpp:1975.
 */
static py::array_t<std::complex<double>> vdam_second_moment(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_other_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mom_in,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max,
    double lambda
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.data = numpy_to_complex_3d(data_in);
    if (r_max > 0)
        bp.r_max = r_max;

    MultidimArray<Complex> data_other = numpy_to_complex_3d(data_other_in);
    MultidimArray<Complex> mom = numpy_to_complex_3d(mom_in);

    bp.getSecondMoment(mom, data_other, (RFLOAT)lambda);
    return complex_3d_to_numpy(mom);
}


/**
 * Wrap `BackProjector::applyMomenta`.
 *
 * Source: backprojector.cpp:2000.
 */
static py::tuple vdam_apply_momenta(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mom1_h1_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mom1_h2_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mom2_in,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.data = numpy_to_complex_3d(data_in);
    if (r_max > 0)
        bp.r_max = r_max;

    MultidimArray<Complex> mom1_h1 = numpy_to_complex_3d(mom1_h1_in);
    MultidimArray<Complex> mom1_h2 = numpy_to_complex_3d(mom1_h2_in);
    MultidimArray<Complex> mom2 = numpy_to_complex_3d(mom2_in);

    bp.applyMomenta(mom1_h1, mom1_h2, mom2);

    return py::make_tuple(
        complex_3d_to_numpy(bp.data),
        real_1d_to_numpy(bp.mom1_noise_power)
    );
}


/**
 * Wrap `BackProjector::reconstructGrad`.
 *
 * Source: backprojector.cpp:2054.
 */
static py::array_t<double> vdam_reconstruct_grad(
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> weight_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> fsc_in,
    double grad_stepsize,
    double tau2_fudge,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max,
    double min_resol_shell,
    bool use_fsc
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.data = numpy_to_complex_3d(data_in);
    auto wt_buf = weight_in.request();
    bp.weight.resize(wt_buf.shape[0], wt_buf.shape[1], wt_buf.shape[2]);
    std::memcpy(bp.weight.data, wt_buf.ptr,
                wt_buf.shape[0] * wt_buf.shape[1] * wt_buf.shape[2] * sizeof(RFLOAT));
    if (r_max > 0)
        bp.r_max = r_max;

    auto vol_buf = vol_in.request();
    long nz = vol_buf.shape[0];
    long ny = vol_buf.shape[1];
    long nx = vol_buf.shape[2];
    MultidimArray<RFLOAT> vol(nz, ny, nx);
    std::memcpy(vol.data, vol_buf.ptr, nz * ny * nx * sizeof(RFLOAT));

    MultidimArray<RFLOAT> fsc;
    auto fsc_buf = fsc_in.request();
    long n_shells = fsc_buf.shape[0];
    fsc.resize(n_shells);
    std::memcpy(fsc.data, fsc_buf.ptr, n_shells * sizeof(RFLOAT));

    bp.reconstructGrad(vol, fsc, (RFLOAT)grad_stepsize, (RFLOAT)tau2_fudge,
                       (RFLOAT)min_resol_shell, use_fsc, false);

    py::array_t<double> out({(long)ZSIZE(vol), (long)YSIZE(vol), (long)XSIZE(vol)});
    std::memcpy(out.request().ptr, vol.data,
                ZSIZE(vol) * YSIZE(vol) * XSIZE(vol) * sizeof(RFLOAT));
    return out;
}


// ===========================================================================
//  VDAM scheduler free functions (copy-verbatim from ml_optimiser.cpp)
// ===========================================================================


/**
 * Replicates MlOptimiser::updateSubsetSize for `gradient_refine && !do_auto_refine`.
 *
 * Source: ml_optimiser.cpp:10238-10271 copy-verbatim minus the auto-refine
 * branch and the `myverb` logging. Returning `-1` means "all particles".
 */
static long vdam_compute_subset_size(
    int iter,
    int nr_iter,
    int grad_ini_iter,
    int grad_inbetween_iter,
    long grad_ini_subset_size,
    long grad_fin_subset_size,
    long nr_particles,
    int grad_em_iters,
    bool do_grad,
    bool has_converged,
    bool grad_has_converged,
    int nr_classes,
    bool do_split_random_halves,
    int grad_suspended_local_searches_iter
) {
    long subset_size;

    if (iter < grad_ini_iter) {
        subset_size = grad_ini_subset_size;
    } else if (iter < grad_ini_iter + grad_inbetween_iter) {
        subset_size = grad_ini_subset_size +
                      ROUND((RFLOAT(iter - grad_ini_iter) / RFLOAT(grad_inbetween_iter)) *
                            (grad_fin_subset_size - grad_ini_subset_size));
    } else {
        subset_size = grad_fin_subset_size;
    }

    long effective = nr_particles;
    if (do_split_random_halves)
        effective = std::floor(RFLOAT(nr_particles) / 2.0);

    if (!do_grad ||
        nr_iter - iter < grad_em_iters ||
        (nr_iter == iter && nr_classes > 1) ||
        subset_size >= effective ||
        grad_suspended_local_searches_iter > 0 ||
        has_converged || grad_has_converged)
    {
        subset_size = -1;
    }

    return subset_size;
}


/**
 * Replicates MlOptimiser::updateStepSize.
 *
 * Source: ml_optimiser.cpp:10278-10325 copy-verbatim.
 */
static double vdam_compute_stepsize(
    int iter,
    int grad_ini_iter,
    int grad_inbetween_iter,
    bool is_3d_model,
    int ref_dim,
    double grad_stepsize_in,
    std::string grad_stepsize_scheme_in
) {
    RFLOAT _stepsize = (RFLOAT)grad_stepsize_in;
    std::string _scheme = grad_stepsize_scheme_in;

    if (_stepsize <= 0) {
        if (ref_dim == 3 && !is_3d_model) _stepsize = 0.3;
        else if (ref_dim == 3 && is_3d_model) _stepsize = 0.5;
        else _stepsize = 0.3;
    }

    if (_scheme.empty()) {
        if (ref_dim == 3 && !is_3d_model) _scheme = "plain";
        else if (ref_dim == 3 && is_3d_model) _scheme = std::to_string(0.9 / _stepsize) + "-step";
        else _scheme = std::to_string(0.9 / _stepsize) + "-step";
    }

    if (_scheme == "plain")
        return _stepsize;

    if (_scheme.find("-step") != std::string::npos) {
        float inflate = textToFloat(_scheme.substr(0, _scheme.find("-step")));
        if (inflate <= 0.)
            throw std::runtime_error("Invalid inflate value for grad_stepsize_scheme");
        RFLOAT x = (RFLOAT)iter;
        RFLOAT a = grad_inbetween_iter / 2.0;
        RFLOAT b = (RFLOAT)grad_ini_iter;
        RFLOAT scale = 1. / (std::pow(10.0, (x - b - a / 2.) / (a / 4.)) + 1.);
        return (_stepsize * inflate) * scale + _stepsize * (1 - scale);
    }

    throw std::runtime_error("Invalid value for grad_stepsize_scheme");
}


/**
 * Replicates MlOptimiser::updateTau2Fudge.
 *
 * Source: ml_optimiser.cpp:10327-10379 copy-verbatim.
 */
static double vdam_compute_tau2_fudge(
    int iter,
    int grad_ini_iter,
    int grad_inbetween_iter,
    bool is_3d_model,
    int ref_dim,
    bool do_auto_refine,
    double tau2_fudge_arg,
    std::string tau2_fudge_scheme_in
) {
    RFLOAT _fudge = (RFLOAT)tau2_fudge_arg;
    std::string _scheme = tau2_fudge_scheme_in;

    if (_fudge <= 0) {
        if (do_auto_refine) _fudge = 1;
        else {
            if (ref_dim == 3 && !is_3d_model) _fudge = 4;
            else if (ref_dim == 3 && is_3d_model) _fudge = 4;
            else _fudge = 4;
        }
    }

    if (_scheme.empty()) {
        if (ref_dim == 3 && !is_3d_model) _scheme = "plain";
        else if (ref_dim == 3 && is_3d_model) _scheme = std::to_string(_fudge / 1.) + "-step";
        else _scheme = std::to_string(_fudge / 1.) + "-step";
    }

    if (_scheme == "plain")
        return _fudge;

    if (_scheme.find("-step") != std::string::npos) {
        float deflate = textToFloat(_scheme.substr(0, _scheme.find("-step")));
        if (deflate <= 0.)
            throw std::runtime_error("Invalid deflate value for tau2_fudge_scheme");
        RFLOAT x = (RFLOAT)iter;
        RFLOAT a = grad_inbetween_iter / 4.0;
        RFLOAT b = (RFLOAT)grad_ini_iter;
        RFLOAT scale = 1. / (std::pow(10.0, (x - b - a / 2.) / (a / 4.)) + 1.);
        return (_fudge / deflate) * scale + _fudge * (1 - scale);
    }

    throw std::runtime_error("Invalid value for tau2_fudge_scheme");
}


/**
 * Return the first `n_draws` values of RELION's `rnd_unif()` after
 * `init_random_generator(seed)`. Useful for reproducing per-particle
 * random orientation draws during the InitialModel bootstrap
 * (ml_optimiser.cpp:3132-3145).
 */
static py::array_t<double> vdam_rnd_unif_sequence(
    int seed,
    long n_draws
) {
    init_random_generator(seed);
    py::array_t<double> out((py::ssize_t)n_draws);
    double* p = (double*)out.request().ptr;
    for (long i = 0; i < n_draws; i++)
        p[i] = (double)rnd_unif();
    return out;
}


/**
 * Replicates Experiment::randomiseParticlesOrder's non-halves Fisher-Yates
 * path, using RELION's own Mersenne-Twister + ran1 pair (via `rnd_unif`).
 *
 * Returns the shuffled particle ids.
 */
static py::array_t<long> vdam_randomise_particles_order(
    long nr_particles,
    int seed
) {
    std::vector<long> order(nr_particles);
    for (long i = 0; i < nr_particles; i++)
        order[i] = i;

    if (nr_particles <= 1) {
        py::array_t<long> out((py::ssize_t)nr_particles);
        std::memcpy(out.request().ptr, order.data(), nr_particles * sizeof(long));
        return out;
    }

    init_random_generator(seed);
    // Fisher-Yates from the tail, matching exp_model.cpp
    for (long i = nr_particles - 1; i > 0; i--) {
        RFLOAT u = rnd_unif();
        long j = (long)(u * i + 0.5);
        if (j < 0) j = 0;
        if (j > i) j = i;
        if (j != i) {
            long tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    py::array_t<long> out((py::ssize_t)nr_particles);
    std::memcpy(out.request().ptr, order.data(), nr_particles * sizeof(long));
    return out;
}


// ===========================================================================
//  Binding registration
// ===========================================================================


void init_initialmodel_bindings(py::module_ &m) {
    // ----- Moment primitives -----

    m.def("vdam_reweight_grad", &vdam_reweight_grad,
          py::arg("data"), py::arg("weight"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          R"doc(
BackProjector::reweightGrad — divide accumulator data by weight
(max-1-gated). Returns the updated data array.
)doc");

    m.def("vdam_first_moment", &vdam_first_moment,
          py::arg("data"), py::arg("mom"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          py::arg("lambda") = 0.9,
          R"doc(
BackProjector::getFristMoment — exponential-moving-average first moment.
If mom.sum() == 0, copy data into mom (initialisation). Else:
  mom <- lambda*mom + (1-lambda)*data.
)doc");

    m.def("vdam_second_moment", &vdam_second_moment,
          py::arg("data"), py::arg("data_other"), py::arg("mom"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          py::arg("lambda") = 0.9,
          R"doc(
BackProjector::getSecondMoment — normalised-difference second moment.
mom.real <- lambda*mom.real + (1-lambda)*(|data_other-data|^2/|data_other+data|^2)/4
mom.imag <- 0
)doc");

    m.def("vdam_apply_momenta", &vdam_apply_momenta,
          py::arg("data"), py::arg("mom1_h1"), py::arg("mom1_h2"), py::arg("mom2"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          R"doc(
BackProjector::applyMomenta — combine momenta + compute noise-power.
Returns (updated_data, mom1_noise_power).
)doc");

    m.def("vdam_reconstruct_grad", &vdam_reconstruct_grad,
          py::arg("vol"), py::arg("data"), py::arg("weight"), py::arg("fsc"),
          py::arg("grad_stepsize"), py::arg("tau2_fudge"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          py::arg("min_resol_shell") = 0.0, py::arg("use_fsc") = false,
          R"doc(
BackProjector::reconstructGrad — apply the gradient update to vol.
Returns the updated real-space volume.
)doc");

    // ----- Scheduler free functions -----

    m.def("vdam_compute_subset_size", &vdam_compute_subset_size,
          py::arg("iter"), py::arg("nr_iter"),
          py::arg("grad_ini_iter"), py::arg("grad_inbetween_iter"),
          py::arg("grad_ini_subset_size"), py::arg("grad_fin_subset_size"),
          py::arg("nr_particles"), py::arg("grad_em_iters") = 0,
          py::arg("do_grad") = true,
          py::arg("has_converged") = false, py::arg("grad_has_converged") = false,
          py::arg("nr_classes") = 1, py::arg("do_split_random_halves") = false,
          py::arg("grad_suspended_local_searches_iter") = 0,
          R"doc(
MlOptimiser::updateSubsetSize, gradient_refine && !do_auto_refine branch.
Returns -1 when subset should span all particles.
)doc");

    m.def("vdam_compute_stepsize", &vdam_compute_stepsize,
          py::arg("iter"), py::arg("grad_ini_iter"), py::arg("grad_inbetween_iter"),
          py::arg("is_3d_model"), py::arg("ref_dim"),
          py::arg("grad_stepsize") = -1.0,
          py::arg("grad_stepsize_scheme") = std::string(""),
          "MlOptimiser::updateStepSize copy-verbatim.");

    m.def("vdam_compute_tau2_fudge", &vdam_compute_tau2_fudge,
          py::arg("iter"), py::arg("grad_ini_iter"), py::arg("grad_inbetween_iter"),
          py::arg("is_3d_model"), py::arg("ref_dim"),
          py::arg("do_auto_refine") = false,
          py::arg("tau2_fudge_arg") = -1.0,
          py::arg("tau2_fudge_scheme") = std::string(""),
          "MlOptimiser::updateTau2Fudge copy-verbatim.");

    m.def("vdam_randomise_particles_order", &vdam_randomise_particles_order,
          py::arg("nr_particles"), py::arg("seed"),
          "Experiment::randomiseParticlesOrder (non-halves) via RELION rnd_unif.");

    m.def("vdam_rnd_unif_sequence", &vdam_rnd_unif_sequence,
          py::arg("seed"), py::arg("n_draws"),
          "Return the first n_draws of rnd_unif() after init_random_generator(seed).");
}
