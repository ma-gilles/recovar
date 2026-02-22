import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core_ctf as core_ctf
import recovar.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def test_core_reexports_ctf_api():
    assert core.CTFParamIndex is core_ctf.CTFParamIndex
    assert core.evaluate_ctf is core_ctf.evaluate_ctf
    assert core.cryodrgn_CTF is core_ctf.cryodrgn_CTF
    assert core.cryodrgn_CTF_half is core_ctf.cryodrgn_CTF_half


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
        captured["voltage"] = voltage
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
    # Call the un-jitted Python function so monkeypatched side effects are concrete.
    core_ctf.evaluate_ctf_wrapper_tilt_series.__wrapped__(
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


def test_evaluate_ctf_wrapper_tilt_series_v2_shape():
    ctf_params = np.zeros((3, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.TILT_ANGLE] = np.array([0.0, 3.0, 6.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0

    out = np.asarray(core_ctf.evaluate_ctf_wrapper_tilt_series_v2(ctf_params, image_shape=(4, 4), voxel_size=1.0))
    assert out.shape == (3, 16)
    assert np.isfinite(out).all()


def _make_standard_ctf_params(n_images):
    ctf_params = np.zeros((n_images, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DFU] = 15000.0
    ctf_params[:, core_ctf.CTFParamIndex.DFV] = 17000.0
    ctf_params[:, core_ctf.CTFParamIndex.DFANG] = 20.0
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.PHASE_SHIFT] = 5.0
    ctf_params[:, core_ctf.CTFParamIndex.BFACTOR] = 10.0
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.arange(n_images, dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.TILT_ANGLE] = 3.0 * np.arange(n_images, dtype=np.float32)
    return ctf_params


@pytest.mark.parametrize("image_shape", [(4, 8), (6, 10)])
def test_cryodrgn_ctf_half_matches_full_mapping(image_shape):
    ctf_params = _make_standard_ctf_params(3)[:, :9]
    full = np.asarray(core_ctf.cryodrgn_CTF(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half = np.asarray(core_ctf.cryodrgn_CTF_half(ctf_params, image_shape=image_shape, voxel_size=1.0))
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("antialiasing", [False, True])
def test_evaluate_ctf_wrapper_half_matches_full_mapping(antialiasing):
    ctf_params = _make_standard_ctf_params(2)[:, :9]
    image_shape = (4, 8)
    full = np.asarray(
        core_ctf.evaluate_ctf_wrapper(
            ctf_params, image_shape=image_shape, voxel_size=1.0, antialiasing=antialiasing
        )
    )
    half = np.asarray(
        core_ctf.evaluate_ctf_wrapper_half(
            ctf_params, image_shape=image_shape, voxel_size=1.0, antialiasing=antialiasing
        )
    )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


def test_tilt_series_half_wrappers_match_full_mapping():
    ctf_params = _make_standard_ctf_params(4)
    image_shape = (4, 8)

    full = np.asarray(
        core_ctf.evaluate_ctf_wrapper_tilt_series(
            ctf_params,
            image_shape=image_shape,
            voxel_size=1.0,
            dose_per_tilt=2.9,
            angle_per_tilt=3.0,
        )
    )
    half = np.asarray(
        core_ctf.evaluate_ctf_wrapper_tilt_series_half(
            ctf_params,
            image_shape=image_shape,
            voxel_size=1.0,
            dose_per_tilt=2.9,
            angle_per_tilt=3.0,
        )
    )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)

    full_v2 = np.asarray(core_ctf.evaluate_ctf_wrapper_tilt_series_v2(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half_v2 = np.asarray(core_ctf.evaluate_ctf_wrapper_tilt_series_v2_half(ctf_params, image_shape=image_shape, voxel_size=1.0))
    expected_v2 = np.asarray(fourier_transform_utils.full_image_to_half_image(full_v2, image_shape))
    np.testing.assert_allclose(half_v2, expected_v2, atol=1e-6, rtol=1e-6)


def test_get_cryo_et_ctf_fun_half_matches_tilt_series_half_wrapper():
    ctf_params = _make_standard_ctf_params(3)
    image_shape = (4, 8)
    ctf_fun = core_ctf.get_cryo_ET_CTF_fun_half(dose_per_tilt=2.9, angle_per_tilt=3.0)
    out_fun = np.asarray(ctf_fun(ctf_params, image_shape, 1.0))
    out_ref = np.asarray(
        core_ctf.evaluate_ctf_wrapper_tilt_series_half(
            ctf_params,
            image_shape=image_shape,
            voxel_size=1.0,
            dose_per_tilt=2.9,
            angle_per_tilt=3.0,
        )
    )
    np.testing.assert_allclose(out_fun, out_ref, atol=1e-6, rtol=1e-6)


def test_ctf_half_pointwise_application_matches_full_mapping():
    rng = np.random.default_rng(41)
    image_shape = (6, 10)
    n_images = 3
    ctf_params = _make_standard_ctf_params(n_images)[:, :9]

    ctf_full = np.asarray(core_ctf.evaluate_ctf_wrapper(ctf_params, image_shape=image_shape, voxel_size=1.0))
    ctf_half = np.asarray(core_ctf.evaluate_ctf_wrapper_half(ctf_params, image_shape=image_shape, voxel_size=1.0))

    real_images = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    images_full = np.asarray(fourier_transform_utils.get_dft2(real_images)).reshape(n_images, -1)
    images_half = np.asarray(fourier_transform_utils.full_image_to_half_image(images_full, image_shape))

    mapped_full_product = np.asarray(
        fourier_transform_utils.full_image_to_half_image(images_full * ctf_full, image_shape)
    )
    half_product = images_half * ctf_half
    np.testing.assert_allclose(half_product, mapped_full_product, atol=1e-6, rtol=1e-6)


def test_get_cryo_et_ctf_fun_matches_wrapper_call():
    ctf_params = np.zeros((2, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([0.0, 1.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0

    ctf_fun = core_ctf.get_cryo_ET_CTF_fun(dose_per_tilt=2.9, angle_per_tilt=3.0)
    out_fun = np.asarray(ctf_fun(ctf_params, (4, 4), 1.0))
    out_ref = np.asarray(
        core_ctf.evaluate_ctf_wrapper_tilt_series(
            ctf_params,
            image_shape=(4, 4),
            voxel_size=1.0,
            dose_per_tilt=2.9,
            angle_per_tilt=3.0,
        )
    )
    np.testing.assert_allclose(out_fun, out_ref, atol=1e-6, rtol=1e-6)
