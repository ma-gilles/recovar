import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core_ctf as core_ctf

pytestmark = pytest.mark.unit


def test_core_reexports_ctf_api():
    assert core.CTFParamIndex is core_ctf.CTFParamIndex
    assert core.evaluate_ctf is core_ctf.evaluate_ctf
    assert core.cryodrgn_CTF is core_ctf.cryodrgn_CTF


def test_evaluate_ctf_shape():
    freqs = np.array([[0.0, 0.0], [0.1, -0.2]], dtype=np.float32)
    out = np.asarray(
        core_ctf.evaluate_ctf(
            freqs,
            dfu=15000.0,
            dfv=15000.0,
            dfang=0.0,
            volt=300.0,
            cs=2.7,
            w=0.1,
            phase_shift=0.0,
            bfactor=0.0,
        )
    )
    assert out.shape == (2,)


def test_critical_exposure_decreases_with_frequency():
    freq = np.array([0.05, 0.10, 0.20], dtype=np.float32)
    out = np.asarray(core_ctf.critical_exposure(freq, voltage=300.0))
    assert np.all(np.diff(out) < 0)


def test_ctfparams_basic_operations():
    params = core_ctf.CTFParams.create_standard_params(n_images=2, contrast=1.0)
    assert params.shape == (2, len(core_ctf.CTFParamIndex))
    assert params.validate() is True

    params.set_param(core_ctf.CTFParamIndex.VOLT, np.array([200.0, 300.0]))
    np.testing.assert_array_equal(params.get_param(core_ctf.CTFParamIndex.VOLT), np.array([200.0, 300.0]))

    params.add_tilt_series_params(dose_values=np.array([1.0, 2.0]), tilt_angles=np.array([3.0, 6.0]))
    np.testing.assert_array_equal(params.get_param(core_ctf.CTFParamIndex.DOSE), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(params.get_param(core_ctf.CTFParamIndex.TILT_ANGLE), np.array([3.0, 6.0]))


def test_ctfparams_validation_errors():
    params = core_ctf.CTFParams(np.zeros((1, len(core_ctf.CTFParamIndex))))
    with pytest.raises(ValueError):
        params.validate()

    bad = core_ctf.CTFParams.create_standard_params(n_images=1, volt=-1.0)
    with pytest.raises(ValueError):
        bad.validate()


def test_get_dose_filters_shape_for_non_square_image():
    out = np.asarray(
        core_ctf.get_dose_filters(
            Apix=1.0,
            image_shape=(2, 3),
            cumulative_dose=np.array([0.0, 1.0], dtype=np.float32),
            tilt_angles=np.array([0.0, 30.0], dtype=np.float32),
            voltage=300.0,
        )
    )
    assert out.shape == (2, 6)


def test_evaluate_ctf_wrapper_tilt_series_uses_voltage(monkeypatch):
    captured = {}

    def fake_dose_filters(Apix, image_shape, dose_per_tilt, angle_per_tilt, tilt_numbers, voltage):
        captured["voltage"] = float(voltage)
        return np.ones((len(tilt_numbers), image_shape[0] * image_shape[1]), dtype=np.float32)

    monkeypatch.setattr(core_ctf, "get_dose_filters_from_tilt_number", fake_dose_filters)
    monkeypatch.setattr(
        core_ctf,
        "cryodrgn_CTF",
        lambda CTF_params, image_shape, voxel_size: np.ones(
            (CTF_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32
        ),
    )

    ctf_params = np.zeros((2, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([1.0, 2.0])
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    core_ctf.evaluate_ctf_wrapper_tilt_series(
        ctf_params,
        image_shape=(2, 2),
        voxel_size=1.0,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    assert captured["voltage"] == pytest.approx(300.0)


def test_evaluate_ctf_wrapper_antialiasing_shape():
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0
    out = np.asarray(core_ctf.evaluate_ctf_wrapper(ctf_params, image_shape=(4, 4), voxel_size=1.0, antialiasing=True))
    assert out.shape == (1, 16)
