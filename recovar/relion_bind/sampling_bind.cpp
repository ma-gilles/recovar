/**
 * HealpixSampling bindings for parity testing.
 *
 * Phase 5: S1 (orientations), S2 (translations), S3 (perturbation).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <src/healpix_sampling.h>
#include <src/euler.h>
#include <src/symmetries.h>

namespace py = pybind11;


static HealpixSampling make_3d_sampling(
    int healpix_order,
    double psi_step,
    const std::string& symmetry
) {
    if (healpix_order < 0)
        throw std::runtime_error("healpix_order must be nonnegative");

    SymList symmetry_list;
    int point_group = 0;
    int point_group_order = 0;
    if (!symmetry_list.isSymmetryGroup(symmetry, point_group, point_group_order))
        throw std::runtime_error("unrecognized RELION point-group symmetry: " + symmetry);
    if (point_group == pg_I5 || point_group == pg_I5H)
        throw std::runtime_error("RELION recognizes I5/I5H but does not implement them");

    HealpixSampling sampling;
    sampling.clear();
    sampling.is_3D = true;
    sampling.isRelax = false;
    sampling.fn_sym = symmetry;
    sampling.healpix_order = healpix_order;
    sampling.limit_tilt = 90.0;
    sampling.psi_step = psi_step;
    if (sampling.psi_step < 0.0)
        sampling.psi_step = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));
    sampling.healpix_base.Set(healpix_order, NEST);
    sampling.initialiseSymMats(
        sampling.fn_sym,
        sampling.pgGroup,
        sampling.pgOrder,
        sampling.R_repository,
        sampling.L_repository);
    sampling.setOrientations(healpix_order, sampling.psi_step);
    return sampling;
}


static py::dict get_symmetry_operators(const std::string& symmetry) {
    SymList symmetry_list;
    int point_group = 0;
    int point_group_order = 0;
    if (!symmetry_list.isSymmetryGroup(symmetry, point_group, point_group_order))
        throw std::runtime_error("unrecognized RELION point-group symmetry: " + symmetry);
    if (point_group == pg_I5 || point_group == pg_I5H)
        throw std::runtime_error("RELION recognizes I5/I5H but does not implement them");
    // C1 has only the implicit identity operator.  Avoid asking SymList to
    // locate a symmetry-definition file for the default case: RELION's
    // symmetry-file search depends on the process environment and C1
    // refinement has historically been portable without that dependency.
    const bool is_identity_group = point_group == pg_CN && point_group_order == 1;
    if (!is_identity_group)
        symmetry_list.read_sym_file(symmetry);

    const long count = is_identity_group ? 1 : static_cast<long>(symmetry_list.SymsNo()) + 1;
    py::array_t<double> left({count, (long)3, (long)3});
    py::array_t<double> right({count, (long)3, (long)3});
    auto left_view = left.mutable_unchecked<3>();
    auto right_view = right.mutable_unchecked<3>();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double value = row == col ? 1.0 : 0.0;
            left_view(0, row, col) = value;
            right_view(0, row, col) = value;
        }
    }

    Matrix2D<RFLOAT> L(4, 4), R(4, 4);
    for (long index = 1; index < count; ++index) {
        symmetry_list.get_matrices(static_cast<int>(index - 1), L, R);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                left_view(index, row, col) = static_cast<double>(L(row, col));
                right_view(index, row, col) = static_cast<double>(R(row, col));
            }
        }
    }

    py::dict result;
    result["left"] = std::move(left);
    result["right"] = std::move(right);
    result["point_group"] = point_group;
    result["point_group_order"] = point_group_order;
    return result;
}


static py::dict get_healpix_sampling_metadata(
    int healpix_order,
    double psi_step,
    const std::string& symmetry
) {
    HealpixSampling sampling = make_3d_sampling(healpix_order, psi_step, symmetry);
    const long direction_count = static_cast<long>(sampling.rot_angles.size());
    const long psi_count = static_cast<long>(sampling.psi_angles.size());
    py::array_t<int64_t> directions_ipix({direction_count});
    py::array_t<double> rot({direction_count});
    py::array_t<double> tilt({direction_count});
    py::array_t<double> psi({psi_count});
    auto ipix_view = directions_ipix.mutable_unchecked<1>();
    auto rot_view = rot.mutable_unchecked<1>();
    auto tilt_view = tilt.mutable_unchecked<1>();
    auto psi_view = psi.mutable_unchecked<1>();
    for (long index = 0; index < direction_count; ++index) {
        ipix_view(index) = static_cast<int64_t>(sampling.directions_ipix[index]);
        rot_view(index) = static_cast<double>(sampling.rot_angles[index]);
        tilt_view(index) = static_cast<double>(sampling.tilt_angles[index]);
    }
    for (long index = 0; index < psi_count; ++index)
        psi_view(index) = static_cast<double>(sampling.psi_angles[index]);

    py::dict result;
    result["directions_ipix"] = std::move(directions_ipix);
    result["rot"] = std::move(rot);
    result["tilt"] = std::move(tilt);
    result["psi"] = std::move(psi);
    result["point_group"] = sampling.pgGroup;
    result["point_group_order"] = sampling.pgOrder;
    return result;
}


/**
 * Get the coarse HEALPix direction grid (rot, tilt) for C1 symmetry.
 * Returns (n_directions, 2) array of [rot, tilt] in degrees.
 *
 * Bypasses HealpixSampling::initialise to avoid SymList::read_sym_file
 * which may fail when the working directory lacks RELION's symmetry files.
 * For C1 symmetry, removeSymmetryEquivalentPoints is a no-op anyway.
 */
static py::array_t<double> get_healpix_directions(
    int healpix_order,
    const std::string& symmetry
) {
    if (symmetry != "C1" && symmetry != "c1") {
        HealpixSampling sampling = make_3d_sampling(healpix_order, -1.0, symmetry);
        const long count = static_cast<long>(sampling.rot_angles.size());
        py::array_t<double> result({count, (long)2});
        auto r = result.mutable_unchecked<2>();
        for (long index = 0; index < count; ++index) {
            r(index, 0) = sampling.rot_angles[index];
            r(index, 1) = sampling.tilt_angles[index];
        }
        return result;
    }

    Healpix_Base hpx(healpix_order, NEST);
    long npix = hpx.Npix();

    py::array_t<double> result({npix, (long)2});
    auto r = result.mutable_unchecked<2>();
    for (long ipix = 0; ipix < npix; ipix++) {
        double zz, phi;
        hpx.pix2ang_z_phi(ipix, zz, phi);
        double rot = RAD2DEG(phi);
        double tilt = ACOSD(zz);
        // checkDirection: rot in [-180,180], tilt in [0,180]
        if (rot > 180.0) rot -= 360.0;
        r(ipix, 0) = rot;
        r(ipix, 1) = tilt;
    }
    return result;
}


/**
 * Get full coarse grid: (rot, tilt, psi) for given healpix_order + psi_step.
 * Returns (n_dir * n_psi, 3) array of [rot, tilt, psi] in degrees.
 */
static py::array_t<double> get_coarse_orientations(
    int healpix_order,
    double psi_step,
    const std::string& symmetry
) {
    if (symmetry == "C1" || symmetry == "c1") {
        // Keep the pre-symmetry C1 implementation byte-for-byte.  This path
        // is used by strict numerical-parity gates and must not inherit any
        // changed floating-point evaluation order from HealpixSampling.
        Healpix_Base hpx(healpix_order, NEST);
        const long npix = hpx.Npix();
        if (psi_step < 0)
            psi_step = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));
        const int nr_psi = CEIL(360.0 / psi_step);
        psi_step = 360.0 / static_cast<double>(nr_psi);

        py::array_t<double> result({npix * nr_psi, (long)3});
        auto r = result.mutable_unchecked<2>();
        long idx = 0;
        for (long ipix = 0; ipix < npix; ++ipix) {
            double zz, phi;
            hpx.pix2ang_z_phi(ipix, zz, phi);
            double rot = RAD2DEG(phi);
            const double tilt = ACOSD(zz);
            if (rot > 180.0)
                rot -= 360.0;
            for (int ipsi = 0; ipsi < nr_psi; ++ipsi) {
                r(idx, 0) = rot;
                r(idx, 1) = tilt;
                r(idx, 2) = ipsi * psi_step;
                ++idx;
            }
        }
        return result;
    }

    HealpixSampling sampling = make_3d_sampling(healpix_order, psi_step, symmetry);
    const long npix = static_cast<long>(sampling.rot_angles.size());
    const int nr_psi = static_cast<int>(sampling.psi_angles.size());

    long n_total = npix * nr_psi;
    py::array_t<double> result({n_total, (long)3});
    auto r = result.mutable_unchecked<2>();
    long idx = 0;
    for (long idir = 0; idir < npix; idir++) {
        for (int ipsi = 0; ipsi < nr_psi; ipsi++) {
            r(idx, 0) = sampling.rot_angles[idir];
            r(idx, 1) = sampling.tilt_angles[idir];
            r(idx, 2) = sampling.psi_angles[ipsi];
            idx++;
        }
    }
    return result;
}


static bool is_c1_symmetry(const std::string& symmetry) {
    return symmetry == "C1" || symmetry == "c1";
}


/**
 * Build the symmetry-reduced sampling used by getOrientations.
 *
 * Construction enumerates and reduces the whole point-group grid, which costs
 * milliseconds per call at HEALPix order 3-4 for I1. Batched callers build it
 * once and reuse it for every row.
 */
static HealpixSampling make_symmetric_oversampling(
    int healpix_order,
    double random_perturbation,
    const std::string& symmetry
) {
    HealpixSampling sampling = make_3d_sampling(healpix_order, -1.0, symmetry);
    sampling.random_perturbation = random_perturbation;
    return sampling;
}


/**
 * Append RELION getOrientations rows for one symmetry-reduced (idir, ipsi).
 *
 * getOrientations only reads the sampling (relion/src/healpix_sampling.cpp),
 * so one prebuilt sampling serves any number of rows with identical results.
 */
static void append_symmetric_oversampled_orientations(
    HealpixSampling& sampling,
    int oversampling_order,
    long idir,
    long ipsi,
    std::vector<double> &my_rot,
    std::vector<double> &my_tilt,
    std::vector<double> &my_psi
) {
    if (idir < 0 || idir >= static_cast<long>(sampling.rot_angles.size()))
        throw std::runtime_error("idir out of range");
    if (ipsi < 0 || ipsi >= static_cast<long>(sampling.psi_angles.size()))
        throw std::runtime_error("ipsi out of range");
    std::vector<RFLOAT> rot, tilt, psi;
    std::vector<int> pointer_dir_nonzeroprior, pointer_psi_nonzeroprior;
    std::vector<RFLOAT> directions_prior, psi_prior;
    sampling.getOrientations(
        idir,
        ipsi,
        oversampling_order,
        rot,
        tilt,
        psi,
        pointer_dir_nonzeroprior,
        directions_prior,
        pointer_psi_nonzeroprior,
        psi_prior);
    my_rot.insert(my_rot.end(), rot.begin(), rot.end());
    my_tilt.insert(my_tilt.end(), tilt.begin(), tilt.end());
    my_psi.insert(my_psi.end(), psi.begin(), psi.end());
}


/**
 * Get oversampled orientations for a given (idir, ipsi) pair.
 * Returns (n_oversampled, 3) array of [rot, tilt, psi] in degrees.
 *
 * Directly implements RELION's getOrientations logic for C1 symmetry.
 * idir indexes into HEALPix NEST pixels, ipsi indexes into psi grid.
 */
static void append_oversampled_orientations(
    int healpix_order,
    int oversampling_order,
    long idir,
    long ipsi,
    double random_perturbation,
    std::vector<double> &my_rot,
    std::vector<double> &my_tilt,
    std::vector<double> &my_psi,
    const std::string& symmetry
) {
    if (!is_c1_symmetry(symmetry)) {
        HealpixSampling sampling = make_symmetric_oversampling(
            healpix_order, random_perturbation, symmetry);
        append_symmetric_oversampled_orientations(
            sampling, oversampling_order, idir, ipsi, my_rot, my_tilt, my_psi);
        return;
    }

    Healpix_Base hpx_coarse(healpix_order, NEST);

    double psi_step = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));
    int nr_psi = CEIL(360.0 / psi_step);
    psi_step = 360.0 / (double)nr_psi;
    double psi_center = ipsi * psi_step;

    const size_t first_orientation = my_rot.size();

    if (oversampling_order == 0) {
        double zz, phi;
        hpx_coarse.pix2ang_z_phi(idir, zz, phi);
        double rot = RAD2DEG(phi);
        double tilt = ACOSD(zz);
        if (rot > 180.0) rot -= 360.0;
        my_rot.push_back(rot);
        my_tilt.push_back(tilt);
        my_psi.push_back(psi_center);
    } else {
        Healpix_Base hpx_fine(oversampling_order + healpix_order, NEST);
        int fact = hpx_fine.Nside() / hpx_coarse.Nside();
        int x, y, face;
        hpx_coarse.nest2xyf(idir, x, y, face);

        int nr_psi_over = ROUND(std::pow(2., oversampling_order));

        for (int j = fact * y; j < fact * (y + 1); ++j) {
            for (int i = fact * x; i < fact * (x + 1); ++i) {
                long overpix = hpx_fine.xyf2nest(i, j, face);
                double zz, phi;
                hpx_fine.pix2ang_z_phi(overpix, zz, phi);
                double rot = RAD2DEG(phi);
                double tilt = ACOSD(zz);
                if (rot > 180.0) rot -= 360.0;

                for (int ipsi_over = 0; ipsi_over < nr_psi_over; ipsi_over++) {
                    double overpsi = psi_center - 0.5 * psi_step
                                     + (0.5 + ipsi_over) * psi_step / nr_psi_over;
                    my_rot.push_back(rot);
                    my_tilt.push_back(tilt);
                    my_psi.push_back(overpsi);
                }
            }
        }
    }

    // Apply perturbation (RELION getOrientations lines 1909-1934)
    if (std::abs(random_perturbation) > 0.) {
        double angular_sampling = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));
        double myperturb = random_perturbation * angular_sampling;
        for (size_t iover = first_orientation; iover < my_rot.size(); iover++) {
            Matrix2D<RFLOAT> A(3,3), R(3,3);
            Euler_angles2matrix(my_rot[iover], my_tilt[iover], my_psi[iover], A);
            Euler_angles2matrix(myperturb, myperturb, myperturb, R);
            A = A * R;
            Euler_matrix2angles(A, my_rot[iover], my_tilt[iover], my_psi[iover]);
        }
    }

}


static py::array_t<double> get_oversampled_orientations(
    int healpix_order,
    int oversampling_order,
    long idir,
    long ipsi,
    double random_perturbation,
    const std::string& symmetry
) {
    std::vector<double> my_rot, my_tilt, my_psi;
    append_oversampled_orientations(
        healpix_order, oversampling_order, idir, ipsi, random_perturbation,
        my_rot, my_tilt, my_psi, symmetry);
    long n = my_rot.size();
    py::array_t<double> result({n, (long)3});
    auto r = result.mutable_unchecked<2>();
    for (long i = 0; i < n; i++) {
        r(i, 0) = my_rot[i];
        r(i, 1) = my_tilt[i];
        r(i, 2) = my_psi[i];
    }
    return result;
}


/** Generate oversampled Euler rows for many coarse samples in input order. */
static py::array_t<double> get_oversampled_orientations_batch(
    int healpix_order,
    int oversampling_order,
    py::array_t<long, py::array::c_style | py::array::forcecast> idirs,
    py::array_t<long, py::array::c_style | py::array::forcecast> ipsis,
    double random_perturbation,
    const std::string& symmetry
) {
    auto idir_values = idirs.unchecked<1>();
    auto ipsi_values = ipsis.unchecked<1>();
    if (idir_values.shape(0) != ipsi_values.shape(0))
        throw std::runtime_error("idirs and ipsis must have the same length");

    std::vector<double> my_rot, my_tilt, my_psi;
    const size_t children_per_parent =
        oversampling_order == 0 ? 1 : (size_t)std::pow(8., oversampling_order);
    my_rot.reserve(idir_values.shape(0) * children_per_parent);
    my_tilt.reserve(idir_values.shape(0) * children_per_parent);
    my_psi.reserve(idir_values.shape(0) * children_per_parent);
    if (!is_c1_symmetry(symmetry) && idir_values.shape(0) > 0) {
        HealpixSampling sampling = make_symmetric_oversampling(
            healpix_order, random_perturbation, symmetry);
        for (py::ssize_t i = 0; i < idir_values.shape(0); i++)
            append_symmetric_oversampled_orientations(
                sampling, oversampling_order, idir_values(i), ipsi_values(i),
                my_rot, my_tilt, my_psi);
    } else {
        for (py::ssize_t i = 0; i < idir_values.shape(0); i++)
            append_oversampled_orientations(
                healpix_order, oversampling_order, idir_values(i), ipsi_values(i),
                random_perturbation, my_rot, my_tilt, my_psi, symmetry);
    }

    const py::ssize_t count = (py::ssize_t)my_rot.size();
    py::array_t<double> result({count, (py::ssize_t)3});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < count; i++) {
        output(i, 0) = my_rot[i];
        output(i, 1) = my_tilt[i];
        output(i, 2) = my_psi[i];
    }
    return result;
}


/**
 * Get the coarse translation grid (in Angstroms).
 * Returns (n_trans, 2) array of [x, y] offsets in Angstroms.
 *
 * Directly implements RELION's setTranslations for 2D non-helical SPA.
 */
static py::array_t<double> get_coarse_translations(
    double offset_range,
    double offset_step
) {
    int maxp = CEIL(offset_range / offset_step);
    std::vector<double> tx, ty;
    for (int ix = -maxp; ix <= maxp; ix++) {
        double xoff = ix * offset_step;
        for (int iy = -maxp; iy <= maxp; iy++) {
            double yoff = iy * offset_step;
            if (xoff * xoff + yoff * yoff < offset_range * offset_range + 0.001) {
                tx.push_back(xoff);
                ty.push_back(yoff);
            }
        }
    }
    long n = tx.size();
    py::array_t<double> result({n, (long)2});
    auto r = result.mutable_unchecked<2>();
    for (long i = 0; i < n; i++) {
        r(i, 0) = tx[i];
        r(i, 1) = ty[i];
    }
    return result;
}


/**
 * Get oversampled translations for a given coarse translation index.
 * Returns (n_oversampled, 2) array of [x, y] in pixels.
 *
 * Directly implements RELION's getTranslationsInPixel for 2D non-helical SPA.
 */
static py::array_t<double> get_oversampled_translations(
    double offset_range,
    double offset_step,
    long itrans,
    int oversampling_order,
    double pixel_size,
    double random_perturbation
) {
    // Build coarse grid first to get the itrans-th translation
    int maxp = CEIL(offset_range / offset_step);
    std::vector<double> coarse_x, coarse_y;
    for (int ix = -maxp; ix <= maxp; ix++) {
        double xoff = ix * offset_step;
        for (int iy = -maxp; iy <= maxp; iy++) {
            double yoff = iy * offset_step;
            if (xoff * xoff + yoff * yoff < offset_range * offset_range + 0.001) {
                coarse_x.push_back(xoff);
                coarse_y.push_back(yoff);
            }
        }
    }
    if (itrans < 0 || itrans >= (long)coarse_x.size())
        throw std::runtime_error("itrans out of range");

    std::vector<double> tx, ty;
    if (oversampling_order == 0) {
        tx.push_back(coarse_x[itrans] / pixel_size);
        ty.push_back(coarse_y[itrans] / pixel_size);
    } else {
        int nr_over = ROUND(std::pow(2., oversampling_order));
        for (int iox = 0; iox < nr_over; iox++) {
            double over_xoff = coarse_x[itrans] - 0.5 * offset_step
                               + (0.5 + iox) * offset_step / nr_over;
            for (int ioy = 0; ioy < nr_over; ioy++) {
                double over_yoff = coarse_y[itrans] - 0.5 * offset_step
                                   + (0.5 + ioy) * offset_step / nr_over;
                tx.push_back(over_xoff / pixel_size);
                ty.push_back(over_yoff / pixel_size);
            }
        }
    }

    // Apply perturbation
    if (std::abs(random_perturbation) > 0.) {
        double myperturb = random_perturbation * offset_step / pixel_size;
        for (size_t i = 0; i < tx.size(); i++) {
            tx[i] += myperturb;
            ty[i] += myperturb;
        }
    }

    long n = tx.size();
    py::array_t<double> result({n, (long)2});
    auto r = result.mutable_unchecked<2>();
    for (long i = 0; i < n; i++) {
        r(i, 0) = tx[i];
        r(i, 1) = ty[i];
    }
    return result;
}


/**
 * Convert Euler angles to rotation matrix using RELION's convention.
 * Input: (rot, tilt, psi) in degrees.
 * Returns: (3, 3) rotation matrix.
 */
static py::array_t<double> euler_angles_to_matrix(double rot, double tilt, double psi) {
    Matrix2D<RFLOAT> A(3, 3);
    Euler_angles2matrix(rot, tilt, psi, A);

    py::array_t<double> result({(long)3, (long)3});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r(i, j) = A(i, j);
    return result;
}


/**
 * Convert an (N, 3) array of Euler angles to the inverse matrices produced by
 * RELION's host generateEulerMatrices(..., inverse=true) path.
 *
 * This deliberately executes Euler_angles2matrix and Matrix2D::inv in C++.
 * Reimplementing the same formulas with NumPy is mathematically equivalent,
 * but libm/vectorized-trig rounding can differ by a few double-precision ulps;
 * those ulps affect RELION's strict radius cutoff at the outer Fourier shell.
 */
static py::array_t<double> euler_angles_to_inverse_matrices(
    py::array_t<double, py::array::c_style | py::array::forcecast> angles
) {
    auto input = angles.unchecked<2>();
    if (input.shape(1) != 3)
        throw std::runtime_error("angles must have shape (N,3)");

    const py::ssize_t count = input.shape(0);
    py::array_t<double> result({count, (py::ssize_t)3, (py::ssize_t)3});
    auto output = result.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < count; i++) {
        Matrix2D<RFLOAT> A(3, 3);
        Euler_angles2matrix(input(i, 0), input(i, 1), input(i, 2), A);
        A = A.inv();
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                output(i, row, col) = A(row, col);
    }
    return result;
}


/**
 * Convert rotation matrix to Euler angles using RELION's convention.
 * Input: (3, 3) rotation matrix.
 * Returns: (rot, tilt, psi) tuple in degrees.
 */
static std::tuple<double, double, double> matrix_to_euler_angles(
    py::array_t<double, py::array::c_style | py::array::forcecast> mat
) {
    auto buf = mat.request();
    if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3)
        throw std::runtime_error("mat must be (3,3)");

    Matrix2D<RFLOAT> A(3, 3);
    double *ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A(i, j) = ptr[i * 3 + j];

    RFLOAT rot, tilt, psi;
    Euler_matrix2angles(A, rot, tilt, psi);
    return std::make_tuple(rot, tilt, psi);
}


/**
 * Get angular sampling in degrees for given healpix_order + adaptive_oversampling.
 */
static double get_angular_sampling(int healpix_order, int adaptive_oversampling) {
    return 360.0 / (6 * ROUND(std::pow(2., healpix_order + adaptive_oversampling)));
}


void init_sampling_bindings(py::module_ &m) {
    m.def("get_healpix_directions", &get_healpix_directions,
          py::arg("healpix_order"),
          py::arg("symmetry") = "C1",
          R"doc(
Get HEALPix direction grid (rot, tilt) for C1 symmetry.
Returns (n_directions, 2) with [rot, tilt] in degrees.
)doc");

    m.def("get_coarse_orientations", &get_coarse_orientations,
          py::arg("healpix_order"),
          py::arg("psi_step") = -1.0,
          py::arg("symmetry") = "C1",
          R"doc(
Get full coarse orientation grid (rot, tilt, psi) for a RELION symmetry.
psi_step < 0 uses RELION's default: 360 / (6 * 2^order).
Returns (n_total, 3) with [rot, tilt, psi] in degrees.
)doc");

    m.def("get_oversampled_orientations", &get_oversampled_orientations,
          py::arg("healpix_order"),
          py::arg("oversampling_order"),
          py::arg("idir"),
          py::arg("ipsi"),
          py::arg("random_perturbation") = 0.0,
          py::arg("symmetry") = "C1",
          R"doc(
Get oversampled orientations for a coarse (idir, ipsi) pair.
Returns (n_oversampled, 3) with [rot, tilt, psi] in degrees.
)doc");

    m.def("get_oversampled_orientations_batch", &get_oversampled_orientations_batch,
          py::arg("healpix_order"),
          py::arg("oversampling_order"),
          py::arg("idirs"),
          py::arg("ipsis"),
          py::arg("random_perturbation") = 0.0,
          py::arg("symmetry") = "C1",
          "Get oversampled Euler rows for arrays of coarse direction/psi IDs.");

    m.def("get_coarse_translations", &get_coarse_translations,
          py::arg("offset_range"),
          py::arg("offset_step"),
          R"doc(
Get coarse translation grid in Angstroms.
Returns (n_trans, 2) with [x, y] offsets.
)doc");

    m.def("get_oversampled_translations", &get_oversampled_translations,
          py::arg("offset_range"),
          py::arg("offset_step"),
          py::arg("itrans"),
          py::arg("oversampling_order"),
          py::arg("pixel_size"),
          py::arg("random_perturbation") = 0.0,
          R"doc(
Get oversampled translations for a coarse translation index.
Returns (n_oversampled, 2) with [x, y] in pixels.
)doc");

    m.def("euler_angles_to_matrix", &euler_angles_to_matrix,
          py::arg("rot"), py::arg("tilt"), py::arg("psi"),
          "Convert (rot, tilt, psi) degrees → (3,3) rotation matrix (RELION convention).");

    m.def("euler_angles_to_inverse_matrices", &euler_angles_to_inverse_matrices,
          py::arg("angles"),
          "Convert (N,3) Euler angles to RELION host inverse matrices.");

    m.def("matrix_to_euler_angles", &matrix_to_euler_angles,
          py::arg("mat"),
          "Convert (3,3) rotation matrix → (rot, tilt, psi) degrees (RELION convention).");

    m.def("get_angular_sampling", &get_angular_sampling,
          py::arg("healpix_order"),
          py::arg("adaptive_oversampling") = 0,
          "Angular sampling step in degrees for given order + oversampling.");

    m.def("get_symmetry_operators", &get_symmetry_operators,
          py::arg("symmetry"),
          R"doc(
Return RELION's ordered point-group operators, including identity first.

The result contains ``left`` and ``right`` arrays with shape ``(n, 3, 3)``
for RELION's ``E' = L E R`` convention, plus the integer point-group code and
order. RELION's internal ``SymList`` omits identity; this binding prepends it.
)doc");

    m.def("get_healpix_sampling_metadata", &get_healpix_sampling_metadata,
          py::arg("healpix_order"),
          py::arg("psi_step") = -1.0,
          py::arg("symmetry") = "C1",
          R"doc(
Return RELION's symmetry-reduced coarse sampling axes and source indices.

``directions_ipix`` contains each retained direction's original NEST HEALPix
pixel. ``rot`` and ``tilt`` index those retained ASU rows; ``psi`` is the
independent in-plane axis.
)doc");
}
