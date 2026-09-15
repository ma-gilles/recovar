/**
 * CTF bindings: getFftwImage with and without do_ctf_padding.
 *
 * Phase 2 (P1): CTF::getFftwImage — compare RELION's 2×-padded CTF
 * against recovar's direct CTF computation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <src/ctf.h>

namespace py = pybind11;

/**
 * Compute a CTF image using RELION's CTF::getFftwImage.
 *
 * Returns an FFTW half-transform of shape (oriydim, orixdim/2+1).
 *
 * Parameters match CTF::setValues + getFftwImage:
 *   defU, defV     : defocus in angstroms (positive = underfocused)
 *   defAng         : defocus angle in degrees
 *   voltage        : accelerating voltage in kV
 *   Cs             : spherical aberration in mm
 *   Q0             : amplitude contrast (0.07 cryo, 0.2 stain)
 *   Bfac           : B-factor (0 for no damping)
 *   angpix         : pixel size in angstroms
 *   orixdim        : box size in x (pixels, full real-space)
 *   oriydim        : box size in y (pixels, full real-space)
 *   do_ctf_padding : if true, compute in 2× box then downsample
 *   do_abs         : return |CTF| instead of signed CTF
 *   do_damping     : apply B-factor damping
 */
static py::array_t<double> get_ctf_image(
    double defU,
    double defV,
    double defAng,
    double voltage,
    double Cs,
    double Q0,
    double Bfac,
    double angpix,
    int orixdim,
    int oriydim,
    bool do_ctf_padding,
    bool do_abs,
    bool do_damping,
    double phase_shift,
    double scale
) {
    CTF ctf;
    ctf.setValues(defU, defV, defAng, voltage, Cs, Q0, Bfac, scale,
                  phase_shift, /*dose=*/-1.0);

    MultidimArray<RFLOAT> result(oriydim, orixdim / 2 + 1);

    ctf.getFftwImage(result, orixdim, oriydim, angpix,
                     do_abs,
                     /*do_only_flip_phases=*/false,
                     /*do_intact_until_first_peak=*/false,
                     do_damping,
                     do_ctf_padding,
                     /*do_intact_after_first_peak=*/false);

    long ny = YSIZE(result);
    long nx = XSIZE(result);
    py::array_t<double> out({ny, nx});
    auto buf = out.request();
    std::memcpy(buf.ptr, result.data, ny * nx * sizeof(double));
    return out;
}

/**
 * Compute the raw CTF value at a single frequency.
 *
 * Useful for unit tests: getCTF(x, y) where x, y are in 1/Å.
 */
static double get_ctf_value(
    double defU,
    double defV,
    double defAng,
    double voltage,
    double Cs,
    double Q0,
    double Bfac,
    double angpix,
    double freq_x,
    double freq_y,
    double phase_shift,
    double scale
) {
    CTF ctf;
    ctf.setValues(defU, defV, defAng, voltage, Cs, Q0, Bfac, scale,
                  phase_shift, /*dose=*/-1.0);
    return ctf.getCTF(freq_x, freq_y,
                      /*do_abs=*/false,
                      /*do_only_flip_phases=*/false,
                      /*do_intact_until_first_peak=*/false,
                      /*do_damping=*/(Bfac != 0.0));
}


void init_ctf_bindings(py::module_ &m) {
    m.def("get_ctf_image", &get_ctf_image,
          py::arg("defU"),
          py::arg("defV"),
          py::arg("defAng"),
          py::arg("voltage"),
          py::arg("Cs"),
          py::arg("Q0"),
          py::arg("Bfac") = 0.0,
          py::arg("angpix") = 1.0,
          py::arg("orixdim") = 128,
          py::arg("oriydim") = 128,
          py::arg("do_ctf_padding") = false,
          py::arg("do_abs") = false,
          py::arg("do_damping") = false,
          py::arg("phase_shift") = 0.0,
          py::arg("scale") = 1.0,
          R"doc(
Compute CTF image using RELION's CTF::getFftwImage.

Returns FFTW half-transform, shape (oriydim, orixdim/2+1).

Parameters
----------
defU, defV : float
    Defocus in angstroms (positive = underfocused).
defAng : float
    Defocus angle in degrees.
voltage : float
    Accelerating voltage in kV.
Cs : float
    Spherical aberration in mm.
Q0 : float
    Amplitude contrast.
Bfac : float
    B-factor (0 = no damping).
angpix : float
    Pixel size in angstroms.
orixdim, oriydim : int
    Box size in pixels.
do_ctf_padding : bool
    If True, compute in 2× box then downsample (RELION default in GUI).
do_abs : bool
    Return |CTF| instead of signed CTF.
do_damping : bool
    Apply B-factor damping.
phase_shift : float
    Phase plate shift in degrees.
scale : float
    CTF scale factor.
)doc");

    m.def("get_ctf_value", &get_ctf_value,
          py::arg("defU"),
          py::arg("defV"),
          py::arg("defAng"),
          py::arg("voltage"),
          py::arg("Cs"),
          py::arg("Q0"),
          py::arg("Bfac") = 0.0,
          py::arg("angpix") = 1.0,
          py::arg("freq_x") = 0.0,
          py::arg("freq_y") = 0.0,
          py::arg("phase_shift") = 0.0,
          py::arg("scale") = 1.0,
          R"doc(
Compute a single CTF value at frequency (freq_x, freq_y) in 1/Å.
)doc");
}
