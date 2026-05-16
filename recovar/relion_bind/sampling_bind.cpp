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

namespace py = pybind11;


/**
 * Get the coarse HEALPix direction grid (rot, tilt) for C1 symmetry.
 * Returns (n_directions, 2) array of [rot, tilt] in degrees.
 *
 * Bypasses HealpixSampling::initialise to avoid SymList::read_sym_file
 * which may fail when the working directory lacks RELION's symmetry files.
 * For C1 symmetry, removeSymmetryEquivalentPoints is a no-op anyway.
 */
static py::array_t<double> get_healpix_directions(int healpix_order) {
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
static py::array_t<double> get_coarse_orientations(int healpix_order, double psi_step) {
    Healpix_Base hpx(healpix_order, NEST);
    long npix = hpx.Npix();

    if (psi_step < 0)
        psi_step = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));

    int nr_psi = CEIL(360.0 / psi_step);
    psi_step = 360.0 / (double)nr_psi;

    long n_total = npix * nr_psi;
    py::array_t<double> result({n_total, (long)3});
    auto r = result.mutable_unchecked<2>();
    long idx = 0;
    for (long ipix = 0; ipix < npix; ipix++) {
        double zz, phi;
        hpx.pix2ang_z_phi(ipix, zz, phi);
        double rot = RAD2DEG(phi);
        double tilt = ACOSD(zz);
        if (rot > 180.0) rot -= 360.0;
        for (int ipsi = 0; ipsi < nr_psi; ipsi++) {
            r(idx, 0) = rot;
            r(idx, 1) = tilt;
            r(idx, 2) = ipsi * psi_step;
            idx++;
        }
    }
    return result;
}


/**
 * Get oversampled orientations for a given (idir, ipsi) pair.
 * Returns (n_oversampled, 3) array of [rot, tilt, psi] in degrees.
 *
 * Directly implements RELION's getOrientations logic for C1 symmetry.
 * idir indexes into HEALPix NEST pixels, ipsi indexes into psi grid.
 */
static py::array_t<double> get_oversampled_orientations(
    int healpix_order,
    int oversampling_order,
    long idir,
    long ipsi,
    double random_perturbation
) {
    Healpix_Base hpx_coarse(healpix_order, NEST);

    double psi_step = 360.0 / (6 * ROUND(std::pow(2., healpix_order)));
    int nr_psi = CEIL(360.0 / psi_step);
    psi_step = 360.0 / (double)nr_psi;
    double psi_center = ipsi * psi_step;

    std::vector<double> my_rot, my_tilt, my_psi;

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
        for (size_t iover = 0; iover < my_rot.size(); iover++) {
            Matrix2D<RFLOAT> A(3,3), R(3,3);
            Euler_angles2matrix(my_rot[iover], my_tilt[iover], my_psi[iover], A);
            Euler_angles2matrix(myperturb, myperturb, myperturb, R);
            A = A * R;
            Euler_matrix2angles(A, my_rot[iover], my_tilt[iover], my_psi[iover]);
        }
    }

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
          R"doc(
Get HEALPix direction grid (rot, tilt) for C1 symmetry.
Returns (n_directions, 2) with [rot, tilt] in degrees.
)doc");

    m.def("get_coarse_orientations", &get_coarse_orientations,
          py::arg("healpix_order"),
          py::arg("psi_step") = -1.0,
          R"doc(
Get full coarse orientation grid (rot, tilt, psi) for C1 symmetry.
psi_step < 0 uses RELION's default: 360 / (6 * 2^order).
Returns (n_total, 3) with [rot, tilt, psi] in degrees.
)doc");

    m.def("get_oversampled_orientations", &get_oversampled_orientations,
          py::arg("healpix_order"),
          py::arg("oversampling_order"),
          py::arg("idir"),
          py::arg("ipsi"),
          py::arg("random_perturbation") = 0.0,
          R"doc(
Get oversampled orientations for a coarse (idir, ipsi) pair.
Returns (n_oversampled, 3) with [rot, tilt, psi] in degrees.
)doc");

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

    m.def("matrix_to_euler_angles", &matrix_to_euler_angles,
          py::arg("mat"),
          "Convert (3,3) rotation matrix → (rot, tilt, psi) degrees (RELION convention).");

    m.def("get_angular_sampling", &get_angular_sampling,
          py::arg("healpix_order"),
          py::arg("adaptive_oversampling") = 0,
          "Angular sampling step in degrees for given order + oversampling.");
}
