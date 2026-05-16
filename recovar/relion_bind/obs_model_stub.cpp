/**
 * Minimal stubs for ObservationModel methods referenced by ctf.cpp.
 *
 * In our bindings, CTF::obsModel is always NULL (we use CTF::setValues,
 * not CTF::setValuesByGroup). These functions are never called at runtime,
 * but the linker needs their symbols because ctf.cpp references them.
 */

#include <src/jaz/single_particle/obs_model.h>
#include <src/error.h>

// Every method just errors if somehow called.
#define STUB_BODY REPORT_ERROR("ObservationModel stub: should never be called in binding mode")

RFLOAT ObservationModel::getPixelSize(int opticsGroup) const { STUB_BODY; return 0; }
int ObservationModel::getBoxSize(int opticsGroup) const { STUB_BODY; return 0; }
bool ObservationModel::getCtfPremultiplied(int opticsGroup) const { STUB_BODY; return false; }
Matrix2D<RFLOAT> ObservationModel::getMagMatrix(int opticsGroup) const { STUB_BODY; return Matrix2D<RFLOAT>(); }
const BufferedImage<RFLOAT>& ObservationModel::getGammaOffset(int opticsGroup, int s) {
    STUB_BODY;
    static BufferedImage<RFLOAT> dummy;
    return dummy;
}

// ObservationModel constructor/destructor (header may require them)
ObservationModel::ObservationModel() {}
ObservationModel::ObservationModel(const MetaDataTable& opticsMdt, bool do_die_upon_error)
    : opticsMdt(opticsMdt) {}
