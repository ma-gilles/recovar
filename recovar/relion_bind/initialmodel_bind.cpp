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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include <src/backprojector.h>
#include <src/ctf.h>
#include <src/euler.h>
#include <src/fftw.h>
#include <src/funcs.h>       // for init_random_generator / rnd_unif
#include <src/gradient_optimisation.h>  // for SomGraph::make_blobs_3d
#include <src/mask.h>        // for softMaskOutsideMap
#include <src/projector.h>

namespace py = pybind11;


// ===========================================================================
//  Helpers
// ===========================================================================

static BackProjector make_empty_backprojector(
    int ori_size,
    int padding_factor,
    const std::string &sym,
    int interpolator,
    bool skip_gridding = false
) {
    BackProjector bp(ori_size, 3, sym, interpolator, (float)padding_factor,
                     10, 0, 1.9, 15, 2, skip_gridding);
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
    // RELION BackProjector data uses centered origin (setXmippOrigin) with
    // xinit=0 (half-complex axis starts at DC). Without this the
    // FOR_ALL_ELEMENTS_IN_ARRAY3D iteration ranges are wrong and fsc
    // counter/prev_power are computed over the wrong shells.
    out.setXmippOrigin();
    out.xinit = 0;
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


static py::array_t<double> vdam_projector_power_spectrum(
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
    if (buf.shape[0] != ori_size || buf.shape[1] != ori_size || buf.shape[2] != ori_size)
        throw std::runtime_error("vol_in shape must equal (ori_size, ori_size, ori_size)");

    MultidimArray<RFLOAT> vol(ori_size, ori_size, ori_size);
    std::memcpy(vol.data, buf.ptr,
                (size_t)ori_size * (size_t)ori_size * (size_t)ori_size * sizeof(RFLOAT));

    Projector projector(ori_size, interpolator, (RFLOAT)padding_factor, 10, data_dim);
    MultidimArray<RFLOAT> power_spectrum;
    projector.computeFourierTransformMap(
        vol,
        power_spectrum,
        current_size,
        1,
        do_gridding,
        true
    );
    return real_1d_to_numpy(power_spectrum);
}


static MultidimArray<RFLOAT> numpy_to_real_3d(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    auto buf = arr.request();
    if (buf.ndim != 3)
        throw std::runtime_error("expected 3D real array (k, i, j)");
    long kdim = buf.shape[0];
    long idim = buf.shape[1];
    long jdim = buf.shape[2];
    MultidimArray<RFLOAT> out(kdim, idim, jdim);
    std::memcpy(out.data, buf.ptr, kdim * idim * jdim * sizeof(RFLOAT));
    out.setXmippOrigin();
    out.xinit = 0;
    return out;
}


static void initial_low_pass_filter_reference(
    MultidimArray<RFLOAT> &vol,
    int ori_size,
    double pixel_size,
    double ini_high_ang
) {
    if (ini_high_ang <= 0.0)
        return;

    RFLOAT radius = (RFLOAT)ori_size * (RFLOAT)pixel_size / (RFLOAT)ini_high_ang;
    const RFLOAT width_fmask_edge = (RFLOAT)2.0;  // ml_optimiser.h: WIDTH_FMASK_EDGE
    radius -= width_fmask_edge / (RFLOAT)2.0;
    RFLOAT radius_p = radius + width_fmask_edge;

    FourierTransformer transformer;
    MultidimArray<Complex> Faux;
    transformer.FourierTransform(vol, Faux);
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux) {
        RFLOAT r = sqrt((RFLOAT)(kp * kp + ip * ip + jp * jp));
        if (r < radius) {
            continue;
        } else if (r > radius_p) {
            DIRECT_A3D_ELEM(Faux, k, i, j) = 0.;
        } else {
            DIRECT_A3D_ELEM(Faux, k, i, j) *=
                (RFLOAT)0.5 - (RFLOAT)0.5 * cos(PI * (radius_p - r) / width_fmask_edge);
        }
    }
    transformer.inverseFourierTransform(Faux, vol);
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
    // weight must share data's Xmipp origin; RELION's A3D_ELEM(weight, k, i, j)
    // inside reweightGrad uses centered indexing and reads out of bounds
    // without this.
    auto wt_buf = weight_in.request();
    bp.weight.resize(wt_buf.shape[0], wt_buf.shape[1], wt_buf.shape[2]);
    std::memcpy(bp.weight.data, wt_buf.ptr,
                wt_buf.shape[0] * wt_buf.shape[1] * wt_buf.shape[2] * sizeof(RFLOAT));
    bp.weight.setXmippOrigin();
    bp.weight.xinit = 0;
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
    bool use_fsc,
    bool skip_gridding,
    py::object mom1_noise_power_in
) {
    // MATCH RELION iter-1 M-step: wsum_model.BPref is built with
    // skip_gridding=true (MlWsumModel::initialise passes the optimiser's
    // skip_gridding through, and the optimiser's default is true for
    // gradient refinement per ml_optimiser.cpp).
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator, skip_gridding);
    bp.data = numpy_to_complex_3d(data_in);
    auto wt_buf = weight_in.request();
    bp.weight.resize(wt_buf.shape[0], wt_buf.shape[1], wt_buf.shape[2]);
    std::memcpy(bp.weight.data, wt_buf.ptr,
                wt_buf.shape[0] * wt_buf.shape[1] * wt_buf.shape[2] * sizeof(RFLOAT));
    // Match RELION's centered-origin layout for weight as well.
    bp.weight.setXmippOrigin();
    bp.weight.xinit = 0;
    if (r_max > 0) {
        bp.r_max = r_max;
        // Also reset pad_size to match: pad_size = 2*(round(pf*r_max)+1)+1
        bp.pad_size = 2 * ((int)(padding_factor * r_max + 0.5) + 1) + 1;
    }

    // Populate mom1_noise_power if caller provided it. This is the
    // per-shell FSC-weighting signal that applyMomenta normally writes
    // when do_half is true; if absent, reconstructGrad's `use_fsc=false`
    // branch falls to fsc_estimate=1 and tau2_fudge=1 which is NOT what
    // RELION produces when pseudo_halfsets is active.
    if (!mom1_noise_power_in.is_none()) {
        auto m1 = mom1_noise_power_in.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto m1_buf = m1.request();
        long n = m1_buf.shape[m1_buf.ndim - 1];
        if (m1_buf.ndim > 1) n = m1_buf.size;  // accept (1,1,N) or (N,)
        bp.mom1_noise_power.resize(n);
        std::memcpy(bp.mom1_noise_power.data, m1_buf.ptr, n * sizeof(RFLOAT));
    }

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


/**
 * Wrap `BackProjector::updateSSNRarrays` for an already-accumulated BPref.
 *
 * Source: ml_optimiser.cpp::maximization calls updateSSNRarrays on
 * wsum_model.BPref[iclass] after maximizationGradientParameters has already
 * run reweightGrad/getFristMoment/getSecondMoment/applyMomenta.  The VDAM
 * native path therefore must use the BPref weight buffer directly rather than
 * rebuilding a BackProjector from images and rotations.
 */
static py::tuple vdam_update_ssnr_arrays_from_bpref(
    py::array_t<double, py::array::c_style | py::array::forcecast> weight_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> fsc_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau2_in,
    double tau2_fudge,
    int ori_size,
    int padding_factor,
    int interpolator,
    int r_max,
    bool update_tau2_with_fsc,
    bool is_whole_instead_of_half,
    bool correct_tau2_by_avgctf2
) {
    BackProjector bp = make_empty_backprojector(ori_size, padding_factor, "C1", interpolator);
    bp.weight = numpy_to_real_3d(weight_in);
    if (r_max > 0)
        bp.r_max = r_max;

    long n_shells = ori_size / 2 + 1;

    MultidimArray<RFLOAT> fsc(n_shells);
    fsc.initZeros();
    auto fsc_buf = fsc_in.request();
    double* fsc_ptr = (double*)fsc_buf.ptr;
    long fsc_n = fsc_buf.size;
    for (long s = 0; s < n_shells && s < fsc_n; s++)
        DIRECT_A1D_ELEM(fsc, s) = (RFLOAT)fsc_ptr[s];

    MultidimArray<RFLOAT> tau2(n_shells);
    tau2.initZeros();
    auto tau2_buf = tau2_in.request();
    double* tau2_ptr = (double*)tau2_buf.ptr;
    long tau2_n = tau2_buf.size;
    for (long s = 0; s < n_shells && s < tau2_n; s++)
        DIRECT_A1D_ELEM(tau2, s) = (RFLOAT)tau2_ptr[s];

    MultidimArray<RFLOAT> sigma2(n_shells);
    MultidimArray<RFLOAT> data_vs_prior(n_shells);
    MultidimArray<RFLOAT> fourier_coverage(n_shells);
    MultidimArray<RFLOAT> avgctf2(n_shells);
    avgctf2.initConstant(1.0);

    bp.updateSSNRarrays(
        (RFLOAT)tau2_fudge,
        tau2,
        sigma2,
        data_vs_prior,
        fourier_coverage,
        fsc,
        avgctf2,
        update_tau2_with_fsc,
        is_whole_instead_of_half,
        correct_tau2_by_avgctf2
    );

    return py::make_tuple(
        real_1d_to_numpy(tau2),
        real_1d_to_numpy(sigma2),
        real_1d_to_numpy(data_vs_prior),
        real_1d_to_numpy(fourier_coverage)
    );
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
 * Runs the entire RELION InitialModel bootstrap loop in C++ using real
 * RELION classes — softMaskOutsideMap, FourierTransformer, CenterFFTbySign,
 * windowFourierTransform, CTF::getFftwImage, BackProjector::set2DFourierTransform,
 * BackProjector::reconstruct — exactly mirroring
 * ml_optimiser.cpp::calculateSumOfPowerSpectraAndAverageImage
 * (fn_ref == "None" branch, lines 3127-3205) + the subsequent
 * BPref[k].reconstruct call at line 3265.
 *
 * Returns Iref of shape `(nr_classes, ori_size, ori_size, ori_size)` in
 * FFTW-natural layout (origin at array index 0, i.e. BEFORE fftshift).
 * Caller applies `np.fft.fftshift` to put the origin at the array
 * centre (recovar convention).
 */
static py::array_t<double> vdam_bootstrap_iref(
    py::array_t<double, py::array::c_style | py::array::forcecast> images,  // (N, H, W) real
    py::array_t<double, py::array::c_style | py::array::forcecast> defU,    // (N,)
    py::array_t<double, py::array::c_style | py::array::forcecast> defV,    // (N,)
    py::array_t<double, py::array::c_style | py::array::forcecast> defAngle,// (N,) degrees
    py::array_t<double, py::array::c_style | py::array::forcecast> phase_shift, // (N,) degrees
    double voltage,       // kV
    double Cs,            // mm
    double Q0,            // amplitude contrast
    double pixel_size,    // Å
    int ori_size,
    int nr_classes,
    double particle_diameter_ang,
    double width_mask_edge_px,
    bool do_zero_mask,
    bool do_ctf_correction,
    int random_seed,
    int padding_factor,   // 1 for InitialModel GUI, 2 for auto-refine
    int interpolator,     // TRILINEAR
    int current_size,     // 1/getResolution(ROUND(0.07*ori_size)); -1 => no windowing
    long minimum_nr_particles  // cap per optics group (RELION default 1000 for 2D)
) {
    auto img_buf = images.request();
    if (img_buf.ndim != 3)
        throw std::runtime_error("images must be (N, H, W)");
    long N = img_buf.shape[0];
    long H = img_buf.shape[1];
    long W = img_buf.shape[2];
    if (H != ori_size || W != ori_size)
        throw std::runtime_error("image H, W must equal ori_size");

    const double* du = (const double*)defU.request().ptr;
    const double* dv = (const double*)defV.request().ptr;
    const double* da = (const double*)defAngle.request().ptr;
    const double* dp = (const double*)phase_shift.request().ptr;

    RFLOAT radius_px = particle_diameter_ang / (2.0 * pixel_size);

    // One BackProjector per class (matches wsum_model.BPref[iclass]).
    // RELION's bootstrap path: MlWsumModel uses skip_gridding=true and the
    // current_size passed to initZeros is the pixel-resolution equivalent
    // of ini_high (for 64px/8.5A/ini_high=136A this is 4). r_max ends up
    // = 2 -> pad_size = 7.
    std::vector<BackProjector> bps;
    bps.reserve(nr_classes);
    bool skip_gridding_bootstrap = true;  // MATCH RELION: wsum_model.BPref is skip_gridding
    for (int k = 0; k < nr_classes; k++) {
        bps.emplace_back(ori_size, 3, "C1", interpolator, (float)padding_factor,
                         10, 0, 1.9, 15, 2, skip_gridding_bootstrap);
        bps.back().initZeros(current_size);
    }

    FourierTransformer transformer;

    // RELION's bootstrap caps at minimum_nr_particles_sigma2_noise per
    // optics group (ml_optimiser.cpp:2898 + :2716). For 2D non-tomo the
    // default is 1000. Iterating ALL particles over-accumulates by
    // N/1000 at N=5000, inflating BPref data/weight by 5x.
    long todo = minimum_nr_particles;
    if (todo < nr_classes * 5) todo = nr_classes * 5;  // fn_ref == None floor
    if (todo > N) todo = N;

    for (long part_id_sorted = 0; part_id_sorted < todo; part_id_sorted++) {
        // 1. Per-particle RNG reset.
        init_random_generator(random_seed + (int)part_id_sorted);

        // 2-4. Random Euler draws
        RFLOAT rot  = rnd_unif() * 360.0;
        RFLOAT tilt = rnd_unif() * 180.0;
        RFLOAT psi  = rnd_unif() * 360.0;

        Matrix2D<RFLOAT> A;
        Euler_angles2matrix(rot, tilt, psi, A, false);

        int iclass = (int)(part_id_sorted % nr_classes);

        // Load image into MultidimArray
        Image<RFLOAT> img;
        img().initZeros(ori_size, ori_size);
        const double* row = (const double*)img_buf.ptr + part_id_sorted * H * W;
        for (long i = 0; i < H * W; i++)
            img.data.data[i] = (RFLOAT)row[i];
        img().setXmippOrigin();

        // 7. Soft-mask
        if (do_zero_mask) {
            softMaskOutsideMap(img(), radius_px, width_mask_edge_px, NULL);
        }

        // 8. FFT (RELION uses normalize=true forward, dividing by N^d)
        MultidimArray<Complex> Faux;
        transformer.FourierTransform(img(), Faux, false);

        // 9-10. RELION windows first, then applies CenterFFTbySign to the
        // windowed transform. The order matters for odd bootstrap sizes such
        // as current_size=9.
        MultidimArray<Complex> Fimg;
        windowFourierTransform(Faux, Fimg, current_size > 0 ? current_size : ori_size);
        CenterFFTbySign(Fimg);

        // 11. Compute CTF
        MultidimArray<RFLOAT> Fctf;
        Fctf.resize(Fimg);
        Fctf.initConstant(1.0);
        if (do_ctf_correction) {
            CTF ctf;
            ctf.setValues(du[part_id_sorted], dv[part_id_sorted], da[part_id_sorted],
                          voltage, Cs, Q0, 0.0 /* Bfac */, 1.0 /* scale */,
                          dp[part_id_sorted]);
            ctf.getFftwImage(Fctf, ori_size, ori_size, pixel_size,
                             false,  // ctf_phase_flipped
                             false,  // only_flip_phases
                             false,  // intact_ctf_first_peak
                             true,   // do_damping
                             false); // do_ctf_padding

            // ---- RECOVAR DEBUG: dump first 3 particles' Fimg+Fctf+A ----
            {
                const char* dbg_dir = getenv("RECOVAR_DEBUG_DUMP_DIR_OURS");
                if (dbg_dir != NULL && part_id_sorted < 3) {
                    char p[1024]; FILE* f;
                    snprintf(p, sizeof(p), "%s/p%ld_Fimg_preCTF.bin", dbg_dir, part_id_sorted);
                    f = fopen(p, "wb");
                    if (f) { long nz=1, ny=YSIZE(Fimg), nx=XSIZE(Fimg);
                      fwrite(&nz,sizeof(long),1,f); fwrite(&ny,sizeof(long),1,f); fwrite(&nx,sizeof(long),1,f);
                      fwrite(Fimg.data, sizeof(Complex), ny*nx, f); fclose(f); }
                    snprintf(p, sizeof(p), "%s/p%ld_Fctf.bin", dbg_dir, part_id_sorted);
                    f = fopen(p, "wb");
                    if (f) { long nz=1, ny=YSIZE(Fctf), nx=XSIZE(Fctf);
                      fwrite(&nz,sizeof(long),1,f); fwrite(&ny,sizeof(long),1,f); fwrite(&nx,sizeof(long),1,f);
                      fwrite(Fctf.data, sizeof(RFLOAT), ny*nx, f); fclose(f); }
                    snprintf(p, sizeof(p), "%s/p%ld_A_euler.txt", dbg_dir, part_id_sorted);
                    f = fopen(p, "w");
                    if (f) {
                        for (int ri=0; ri<3; ri++)
                            for (int ci=0; ci<3; ci++)
                                fprintf(f, "A[%d][%d]=%.12e\n", ri, ci, (double)MAT_ELEM(A, ri, ci));
                        fclose(f);
                    }
                }
            }

            // 12. Fimg *= Fctf; Fctf² = Fctf * Fctf (weight)
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg) {
                DIRECT_MULTIDIM_ELEM(Fimg, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
                DIRECT_MULTIDIM_ELEM(Fctf, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
            }
        }

        // 13. Backproject into class iclass
        bps[iclass].set2DFourierTransform(Fimg, A, &Fctf);
    }

    // Reconstruct per-class at do_map=false
    int n_shells = ori_size / 2 + 1;
    MultidimArray<RFLOAT> dummy_tau2(n_shells);
    dummy_tau2.initZeros();

    py::array_t<double> out({(long)nr_classes, (long)ori_size, (long)ori_size, (long)ori_size});
    double* out_ptr = (double*)out.request().ptr;
    std::memset(out_ptr, 0, nr_classes * ori_size * ori_size * ori_size * sizeof(double));

    for (int k = 0; k < nr_classes; k++) {
        // ---- RECOVAR DEBUG DUMP: our BP accumulator before reconstruct ----
        {
            const char* dbg_dir = getenv("RECOVAR_DEBUG_DUMP_DIR");
            if (dbg_dir != NULL) {
                char path[1024]; FILE* f;
                snprintf(path, sizeof(path), "%s/our_bpref_c%d_data_before.bin", dbg_dir, k);
                f = fopen(path, "wb");
                if (f) {
                    long nz = ZSIZE(bps[k].data), ny = YSIZE(bps[k].data), nx = XSIZE(bps[k].data);
                    fwrite(&nz, sizeof(long), 1, f);
                    fwrite(&ny, sizeof(long), 1, f);
                    fwrite(&nx, sizeof(long), 1, f);
                    fwrite(bps[k].data.data, sizeof(Complex), nz * ny * nx, f);
                    fclose(f);
                }
                snprintf(path, sizeof(path), "%s/our_bpref_c%d_weight_before.bin", dbg_dir, k);
                f = fopen(path, "wb");
                if (f) {
                    long nz = ZSIZE(bps[k].weight), ny = YSIZE(bps[k].weight), nx = XSIZE(bps[k].weight);
                    fwrite(&nz, sizeof(long), 1, f);
                    fwrite(&ny, sizeof(long), 1, f);
                    fwrite(&nx, sizeof(long), 1, f);
                    fwrite(bps[k].weight.data, sizeof(RFLOAT), nz * ny * nx, f);
                    fclose(f);
                }
            }
        }
        MultidimArray<RFLOAT> vol;
        bps[k].reconstruct(vol, 10, false /* do_map */, dummy_tau2,
                           1.0 /* tau2_fudge */, 1.0 /* normalise */, -1 /* minres_map */,
                           false /* printTimes */);
        // vol is shape (ori, ori, ori) — copy into output slab
        if (ZSIZE(vol) != (long)ori_size || YSIZE(vol) != (long)ori_size
            || XSIZE(vol) != (long)ori_size) {
            throw std::runtime_error("reconstructed volume shape != ori_size");
        }
        std::memcpy(
            out_ptr + (long)k * ori_size * ori_size * ori_size,
            vol.data,
            (long)ori_size * ori_size * ori_size * sizeof(double)
        );
    }

    return out;
}


/**
 * Apply RELION's post-bootstrap InitialModel reference processing.
 *
 * This mirrors ml_optimiser.cpp:2940-2980 after the raw random-orientation
 * bootstrap reconstruction:
 *   1. initialLowPassFilterReferences()
 *   2. SomGraph::make_blobs_3d(pos), SomGraph::make_blobs_3d(neg)
 *   3. Iref = pos - neg / 2, rescaled to the original standard deviation
 *   4. initialLowPassFilterReferences()
 *   5. softMaskOutsideMap(Iref, diameter/2, width_mask_edge)
 *
 * The function intentionally does not seed rand(). In RELION, make_blobs_3d
 * consumes the C rand() state left by the bootstrap loop's final
 * init_random_generator(random_seed + part_id) + Euler draws.
 */
static py::array_t<double> vdam_postprocess_initial_iref(
    py::array_t<double, py::array::c_style | py::array::forcecast> iref_in,
    double pixel_size,
    double ini_high_ang,
    double particle_diameter_ang,
    double width_mask_edge_px,
    bool do_init_blobs,
    bool is_helical_segment
) {
    auto buf = iref_in.request();
    if (buf.ndim != 4)
        throw std::runtime_error("iref must have shape (K, Z, Y, X)");
    long K = buf.shape[0];
    long zdim = buf.shape[1];
    long ydim = buf.shape[2];
    long xdim = buf.shape[3];
    if (zdim != ydim || ydim != xdim)
        throw std::runtime_error("iref volumes must be cubic");

    int ori_size = (int)xdim;
    double diameter_px = particle_diameter_ang / pixel_size;
    const double* in_ptr = (const double*)buf.ptr;
    py::array_t<double> out({K, zdim, ydim, xdim});
    double* out_ptr = (double*)out.request().ptr;
    long nvox = zdim * ydim * xdim;

    for (long kclass = 0; kclass < K; kclass++) {
        MultidimArray<RFLOAT> vol(zdim, ydim, xdim);
        for (long n = 0; n < nvox; n++)
            DIRECT_MULTIDIM_ELEM(vol, n) = (RFLOAT)in_ptr[kclass * nvox + n];
        vol.setXmippOrigin();

        initial_low_pass_filter_reference(vol, ori_size, pixel_size, ini_high_ang);

        if (do_init_blobs) {
            MultidimArray<RFLOAT> blobs_pos(vol), blobs_neg(vol);
            SomGraph::make_blobs_3d(blobs_pos, vol, 40, (RFLOAT)diameter_px, is_helical_segment);
            SomGraph::make_blobs_3d(blobs_neg, vol, 40, (RFLOAT)diameter_px, is_helical_segment);
            RFLOAT old_std = SomGraph::std(vol);

            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol) {
                DIRECT_MULTIDIM_ELEM(vol, n) =
                    DIRECT_MULTIDIM_ELEM(blobs_pos, n) - DIRECT_MULTIDIM_ELEM(blobs_neg, n) / (RFLOAT)2.0;
            }

            RFLOAT new_std = SomGraph::std(vol);
            if (new_std > 0.) {
                vol *= old_std / new_std;
            }
        }

        if (do_init_blobs) {
            initial_low_pass_filter_reference(vol, ori_size, pixel_size, ini_high_ang);
            softMaskOutsideMap(vol, (RFLOAT)(diameter_px / 2.0), (RFLOAT)width_mask_edge_px, NULL);
        }

        for (long n = 0; n < nvox; n++)
            out_ptr[kclass * nvox + n] = (double)DIRECT_MULTIDIM_ELEM(vol, n);
    }

    return out;
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
 * RELION InitialModel expected angular/translation accuracy estimator.
 *
 * This is the SPA 3D-reference-to-2D-image subset loop from
 * MlOptimiser::calculateExpectedAngularErrors, expressed as a stateless helper
 * so native InitialModel can feed updateAngularSampling with the same units and
 * thresholds as RELION without constructing a full MlOptimiser.
 */
static py::dict vdam_expected_angular_errors(
    py::array_t<double, py::array::c_style | py::array::forcecast> references,
    py::array_t<double, py::array::c_style | py::array::forcecast> eulers_deg,
    py::array_t<long, py::array::c_style | py::array::forcecast> particle_ids,
    py::array_t<int, py::array::c_style | py::array::forcecast> class_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> pdf_class,
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma2_noise,
    py::array_t<double, py::array::c_style | py::array::forcecast> defU,
    py::array_t<double, py::array::c_style | py::array::forcecast> defV,
    py::array_t<double, py::array::c_style | py::array::forcecast> defAngle,
    py::array_t<double, py::array::c_style | py::array::forcecast> phase_shift,
    double voltage,
    double Cs,
    double Q0,
    double pixel_size,
    int ori_size,
    int current_image_size,
    int padding_factor,
    int interpolator,
    double sigma2_fudge,
    int random_seed,
    bool do_ctf_correction,
    bool do_ctf_padding
) {
    auto refs_buf = references.request();
    auto euler_buf = eulers_deg.request();
    auto particle_buf = particle_ids.request();
    auto class_buf = class_ids.request();
    auto pdf_buf = pdf_class.request();
    auto sigma_buf = sigma2_noise.request();
    auto defU_buf = defU.request();
    auto defV_buf = defV.request();
    auto defA_buf = defAngle.request();
    auto phase_buf = phase_shift.request();

    if (refs_buf.ndim != 4)
        throw std::runtime_error("references must have shape (K, N, N, N)");
    const long K = refs_buf.shape[0];
    if (refs_buf.shape[1] != ori_size || refs_buf.shape[2] != ori_size || refs_buf.shape[3] != ori_size)
        throw std::runtime_error("reference volume shape does not match ori_size");
    if (euler_buf.ndim != 2 || euler_buf.shape[1] != 3)
        throw std::runtime_error("eulers_deg must have shape (n_trials, 3)");
    const long n_trials = euler_buf.shape[0];
    if (particle_buf.ndim != 1 || particle_buf.shape[0] != n_trials)
        throw std::runtime_error("particle_ids must have shape (n_trials,)");
    if (class_buf.ndim != 1 || class_buf.shape[0] != n_trials)
        throw std::runtime_error("class_ids must have shape (n_trials,)");
    if (pdf_buf.ndim != 1 || pdf_buf.shape[0] != K)
        throw std::runtime_error("pdf_class must have shape (K,)");
    if (sigma_buf.ndim != 1)
        throw std::runtime_error("sigma2_noise must be a 1D shell spectrum");
    const long n_particles = defU_buf.shape[0];
    if (defU_buf.ndim != 1 || defV_buf.ndim != 1 || defA_buf.ndim != 1 || phase_buf.ndim != 1)
        throw std::runtime_error("CTF parameter arrays must be 1D");
    if (defV_buf.shape[0] != n_particles || defA_buf.shape[0] != n_particles || phase_buf.shape[0] != n_particles)
        throw std::runtime_error("CTF parameter arrays must have matching lengths");
    if (current_image_size <= 0 || current_image_size > ori_size || current_image_size % 2 != 0)
        throw std::runtime_error("current_image_size must be a positive even size <= ori_size");
    if (pixel_size <= 0.0)
        throw std::runtime_error("pixel_size must be positive");
    if (sigma2_fudge <= 0.0)
        throw std::runtime_error("sigma2_fudge must be positive");

    const double *refs_ptr = static_cast<double*>(refs_buf.ptr);
    const double *eulers_ptr = static_cast<double*>(euler_buf.ptr);
    const long *particle_ptr = static_cast<long*>(particle_buf.ptr);
    const int *class_ptr = static_cast<int*>(class_buf.ptr);
    const double *pdf_ptr = static_cast<double*>(pdf_buf.ptr);
    const double *sigma_ptr = static_cast<double*>(sigma_buf.ptr);
    const double *defU_ptr = static_cast<double*>(defU_buf.ptr);
    const double *defV_ptr = static_cast<double*>(defV_buf.ptr);
    const double *defA_ptr = static_cast<double*>(defA_buf.ptr);
    const double *phase_ptr = static_cast<double*>(phase_buf.ptr);

    std::vector<Projector> projectors;
    projectors.reserve((size_t)K);
    const long nvox = (long)ori_size * ori_size * ori_size;
    for (long k = 0; k < K; k++) {
        MultidimArray<RFLOAT> vol(ori_size, ori_size, ori_size);
        std::memcpy(vol.data, refs_ptr + k * nvox, nvox * sizeof(RFLOAT));
        Projector projector(ori_size, interpolator, (RFLOAT)padding_factor, 10, 2);
        MultidimArray<RFLOAT> power_spectrum;
        projector.computeFourierTransformMap(vol, power_spectrum, current_image_size, 1, true);
        projectors.push_back(projector);
    }

    std::vector<double> acc_rot_class((size_t)K, 999.0);
    std::vector<double> acc_trans_class((size_t)K, 999.0);
    std::vector<long> class_counts((size_t)K, 0);
    double acc_rot = 999.0;
    double acc_trans = 999.0;
    const double pvalue = 4.60517;

    for (long k = 0; k < K; k++) {
        if (pdf_ptr[k] < 0.01)
            continue;

        double rot_sum = 0.0;
        double trans_sum = 0.0;
        long count = 0;

        for (long trial = 0; trial < n_trials; trial++) {
            if (class_ptr[trial] != k)
                continue;
            const long part_id = particle_ptr[trial];
            if (part_id < 0 || part_id >= n_particles)
                throw std::runtime_error("particle_ids contains an entry outside the CTF parameter arrays");

            CTF ctf;
            ctf.setValues(
                defU_ptr[part_id],
                defV_ptr[part_id],
                defA_ptr[part_id],
                voltage,
                Cs,
                Q0,
                0.0,
                1.0,
                phase_ptr[part_id],
                -1.0
            );
            MultidimArray<RFLOAT> Fctf(current_image_size, current_image_size / 2 + 1);
            Fctf.initConstant(1.0);
            if (do_ctf_correction) {
                ctf.getFftwImage(
                    Fctf,
                    ori_size,
                    ori_size,
                    pixel_size,
                    false,
                    false,
                    false,
                    true,
                    do_ctf_padding
                );
            }

            for (int imode = 0; imode < 2; imode++) {
                double ang_error = 0.0;
                double sh_error = 0.0;
                double my_snr = 0.0;

                while (my_snr <= pvalue) {
                    double ang_step;
                    if (ang_error < 0.2)
                        ang_step = 0.05;
                    else if (ang_error < 1.0)
                        ang_step = 0.1;
                    else if (ang_error < 2.0)
                        ang_step = 0.2;
                    else if (ang_error < 5.0)
                        ang_step = 0.5;
                    else if (ang_error < 10.0)
                        ang_step = 1.0;
                    else if (ang_error < 20.0)
                        ang_step = 2.0;
                    else
                        ang_step = 5.0;

                    double sh_step;
                    if (sh_error < 1.0)
                        sh_step = 0.1;
                    else if (sh_error < 2.0)
                        sh_step = 0.2;
                    else if (sh_error < 5.0)
                        sh_step = 0.5;
                    else if (sh_error < 10.0)
                        sh_step = 1.0;
                    else
                        sh_step = 2.0;

                    ang_error += ang_step;
                    sh_error += sh_step;
                    if ((imode == 0 && ang_error > 30.0) || (imode == 1 && sh_error > 10.0))
                        break;

                    init_random_generator(random_seed + (int)part_id);

                    const double rot1 = eulers_ptr[trial * 3 + 0];
                    const double tilt1 = eulers_ptr[trial * 3 + 1];
                    const double psi1 = eulers_ptr[trial * 3 + 2];
                    double rot2 = rot1;
                    double tilt2 = tilt1;
                    double psi2 = psi1;
                    double xshift = 0.0;
                    double yshift = 0.0;

                    if (imode == 0) {
                        const double ran = rnd_unif();
                        if (ran < 0.3333)
                            rot2 = rot1 + ang_error;
                        else if (ran < 0.6667)
                            tilt2 = tilt1 + ang_error;
                        else
                            psi2 = psi1 + ang_error;
                    } else {
                        const double ran = rnd_unif();
                        if (ran < 0.5)
                            xshift = sh_error;
                        else
                            yshift = sh_error;
                    }

                    MultidimArray<Complex> F1(current_image_size, current_image_size / 2 + 1);
                    MultidimArray<Complex> F2(current_image_size, current_image_size / 2 + 1);
                    F1.initZeros();
                    F2.initZeros();
                    Matrix2D<RFLOAT> A1(3, 3), A2(3, 3);
                    Euler_angles2matrix(rot1, tilt1, psi1, A1, false);
                    projectors[(size_t)k].get2DFourierTransform(F1, A1);

                    if (imode == 0) {
                        Euler_angles2matrix(rot2, tilt2, psi2, A2, false);
                        projectors[(size_t)k].get2DFourierTransform(F2, A2);
                    } else {
                        shiftImageInFourierTransform(F1, F2, (RFLOAT)ori_size, (RFLOAT)(-xshift), (RFLOAT)(-yshift), (RFLOAT)0.0);
                    }

                    if (do_ctf_correction) {
                        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1) {
                            DIRECT_MULTIDIM_ELEM(F1, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
                            DIRECT_MULTIDIM_ELEM(F2, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
                        }
                    }

                    my_snr = 0.0;
                    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1) {
                        const long idx = n;
                        const long ix = idx % XSIZE(F1);
                        const long iy_linear = idx / XSIZE(F1);
                        const long iy = (iy_linear < current_image_size / 2 + 1)
                            ? iy_linear
                            : (iy_linear - current_image_size);
                        const int ires = ROUND(std::sqrt((double)(iy * iy + ix * ix)));
                        if (ires > 0 && ires < sigma_buf.shape[0]) {
                            const double sigma = sigma_ptr[ires];
                            if (sigma > 0.0) {
                                const Complex diff = DIRECT_MULTIDIM_ELEM(F1, n) - DIRECT_MULTIDIM_ELEM(F2, n);
                                my_snr += norm(diff) / (2.0 * sigma2_fudge * sigma);
                            }
                        }
                    }
                }

                if (imode == 0)
                    rot_sum += ang_error;
                else
                    trans_sum += pixel_size * sh_error;
            }
            count++;
        }

        if (count > 0) {
            acc_rot_class[(size_t)k] = rot_sum / (double)count;
            acc_trans_class[(size_t)k] = trans_sum / (double)count;
            class_counts[(size_t)k] = count;
            acc_rot = std::min(acc_rot, acc_rot_class[(size_t)k]);
            acc_trans = std::min(acc_trans, acc_trans_class[(size_t)k]);
        }
    }

    py::array_t<double> rot_out((py::ssize_t)K);
    py::array_t<double> trans_out((py::ssize_t)K);
    py::array_t<long> counts_out((py::ssize_t)K);
    std::memcpy(rot_out.request().ptr, acc_rot_class.data(), K * sizeof(double));
    std::memcpy(trans_out.request().ptr, acc_trans_class.data(), K * sizeof(double));
    std::memcpy(counts_out.request().ptr, class_counts.data(), K * sizeof(long));

    py::dict out;
    out["acc_rot"] = acc_rot;
    out["acc_trans"] = acc_trans;
    out["acc_rot_class"] = rot_out;
    out["acc_trans_class"] = trans_out;
    out["class_counts"] = counts_out;
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

    // RELION's exp_model.cpp uses std::shuffle(sorted_idx, std::mt19937(seed))
    // for non-halves randomisation. That consumes RNG state differently from
    // rnd_unif() Fisher-Yates, so use the same standard-library path here for
    // byte-exact VDAM subset parity.
    std::mt19937 rng((unsigned int)seed);
    std::shuffle(order.begin(), order.end(), rng);

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
          py::arg("skip_gridding") = true,
          py::arg("mom1_noise_power") = py::none(),
          R"doc(
BackProjector::reconstructGrad — apply the gradient update to vol.
Returns the updated real-space volume.

mom1_noise_power: optional per-shell noise power. When provided (as
produced by applyMomenta in the pseudo-halfsets pipeline) reconstructGrad
uses the SNR-weighted fsc_estimate path; otherwise it defaults to
fsc_estimate=1.
)doc");

    m.def("vdam_update_ssnr_arrays_from_bpref", &vdam_update_ssnr_arrays_from_bpref,
          py::arg("weight"), py::arg("fsc"), py::arg("tau2"),
          py::arg("tau2_fudge"),
          py::arg("ori_size"), py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR, py::arg("r_max") = -1,
          py::arg("update_tau2_with_fsc") = false,
          py::arg("is_whole_instead_of_half") = false,
          py::arg("correct_tau2_by_avgctf2") = false,
          R"doc(
BackProjector::updateSSNRarrays for an already-accumulated VDAM BPref.
Returns (tau2, sigma2, data_vs_prior, fourier_coverage).
)doc");

    m.def("vdam_projector_power_spectrum", &vdam_projector_power_spectrum,
          py::arg("vol_in"),
          py::arg("ori_size"),
          py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR,
          py::arg("current_size") = -1,
          py::arg("do_gridding") = true,
          py::arg("data_dim") = 2,
          R"doc(
Projector::computeFourierTransformMap power_spectrum for InitialModel.

This mirrors MlModel::setFourierTransformMaps(!fix_tau), which refreshes
tau2_class from the current real-space reference at E-step setup.
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

    m.def("vdam_expected_angular_errors", &vdam_expected_angular_errors,
          py::arg("references"),
          py::arg("eulers_deg"),
          py::arg("particle_ids"),
          py::arg("class_ids"),
          py::arg("pdf_class"),
          py::arg("sigma2_noise"),
          py::arg("defU"), py::arg("defV"), py::arg("defAngle"),
          py::arg("phase_shift"),
          py::arg("voltage"), py::arg("Cs"), py::arg("Q0"),
          py::arg("pixel_size"), py::arg("ori_size"),
          py::arg("current_image_size"),
          py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR,
          py::arg("sigma2_fudge") = 1.0,
          py::arg("random_seed") = 0,
          py::arg("do_ctf_correction") = true,
          py::arg("do_ctf_padding") = false,
          R"doc(
SPA 3D InitialModel accuracy estimator from
MlOptimiser::calculateExpectedAngularErrors. Returns acc_rot/acc_trans plus
per-class arrays.
)doc");

    m.def("vdam_bootstrap_iref", &vdam_bootstrap_iref,
          py::arg("images"),
          py::arg("defU"), py::arg("defV"), py::arg("defAngle"),
          py::arg("phase_shift"),
          py::arg("voltage"), py::arg("Cs"), py::arg("Q0"),
          py::arg("pixel_size"), py::arg("ori_size"),
          py::arg("nr_classes"),
          py::arg("particle_diameter_ang"), py::arg("width_mask_edge_px"),
          py::arg("do_zero_mask"), py::arg("do_ctf_correction"),
          py::arg("random_seed"),
          py::arg("padding_factor") = 1,
          py::arg("interpolator") = TRILINEAR,
          py::arg("current_size") = -1,
          py::arg("minimum_nr_particles") = 1000,
          R"doc(
Run the RELION InitialModel bootstrap (ml_optimiser.cpp:3127-3205 +
reconstruct at :3265) end-to-end in C++. Returns the reconstructed
Iref of shape (nr_classes, ori, ori, ori). Caller applies fftshift to put the
origin at the array centre.
)doc");

    m.def("vdam_postprocess_initial_iref", &vdam_postprocess_initial_iref,
          py::arg("iref"),
          py::arg("pixel_size"),
          py::arg("ini_high_ang"),
          py::arg("particle_diameter_ang"),
          py::arg("width_mask_edge_px"),
          py::arg("do_init_blobs") = true,
          py::arg("is_helical_segment") = false,
          R"doc(
Apply RELION's post-bootstrap InitialModel reference processing
(initialLowPassFilterReferences + SomGraph::make_blobs_3d + second
low-pass + softMaskOutsideMap). Input/output are RELION-frame real-space
volumes of shape (K, ori, ori, ori). The function intentionally preserves
the existing C rand() state, matching RELION's bootstrap-to-blob sequence.
)doc");
}
