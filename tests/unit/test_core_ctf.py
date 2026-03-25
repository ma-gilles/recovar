import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core.ctf as core_ctf
import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def test_core_reexports_ctf_api():
    assert core.CTFParamIndex is core_ctf.CTFParamIndex
    assert core.evaluate_ctf is core_ctf.evaluate_ctf
    assert core.CTFEvaluator is core_ctf.CTFEvaluator


def test_evaluate_ctf_shape():
    freqs = np.array([[0.0, 0.0], [0.1, -0.2]], dtype=np.float32)
    # Single-image CTF params: [DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR, CONTRAST]
    ctf_params = np.array([[15000.0, 15000.0, 0.0, 300.0, 2.7, 0.1, 0.0, 0.0, 1.0]], dtype=np.float32)
    out = np.asarray(core_ctf.evaluate_ctf(freqs, ctf_params))
    assert out.shape == (1, 2)


def test_critical_exposure_decreases_with_frequency():
    freq = np.array([0.05, 0.10, 0.20], dtype=np.float32)
    out = np.asarray(core_ctf.critical_exposure(freq, voltage=300.0))
    assert np.all(np.diff(out) < 0)


def test_ctf_evaluator_spa_mode():
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    ctf_params = np.zeros((2, 9), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0
    out = np.asarray(evaluator(ctf_params, image_shape=(4, 4), voxel_size=1.0))
    assert out.shape == (2, 16)

    out_half = np.asarray(evaluator(ctf_params, image_shape=(4, 4), voxel_size=1.0, half_image=True))
    assert out_half.shape == (2, 4 * (4 // 2 + 1))


def test_ctf_evaluator_cryo_et_mode():
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)
    ctf_params = np.zeros((2, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([1.0, 2.0])
    out = np.asarray(evaluator(ctf_params, image_shape=(4, 4), voxel_size=1.0))
    assert out.shape == (2, 16)


def test_ctf_evaluator_tilt_series_mode():
    evaluator = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    assert evaluator.dose_per_tilt == 2.9
    assert evaluator.angle_per_tilt == 3.0


def test_as_ctf_evaluator_wraps_callable():
    fn = lambda params, shape, vs, **kw: np.ones((params.shape[0], shape[0] * shape[1]))
    wrapped = core_ctf.as_ctf_evaluator(fn)
    assert isinstance(wrapped, core_ctf._LegacyCTFAdapter)
    # Wrapping an evaluator should return it unchanged
    assert core_ctf.as_ctf_evaluator(wrapped) is wrapped
    # CTFEvaluator should pass through
    ev = core_ctf.CTFEvaluator()
    assert core_ctf.as_ctf_evaluator(ev) is ev


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


def test_tilt_series_ctf_uses_voltage(monkeypatch):
    captured = {}

    def fake_dose_filters(Apix, image_shape, dose_per_tilt, angle_per_tilt, tilt_numbers, voltage, *, half_image=False):
        captured["voltage"] = voltage
        return np.ones((len(tilt_numbers), image_shape[0] * image_shape[1]), dtype=np.float32)

    monkeypatch.setattr(core_ctf, "get_dose_filters_from_tilt_number", fake_dose_filters)
    monkeypatch.setattr(
        core_ctf,
        "_compute_spa_ctf",
        lambda CTF_params, image_shape, voxel_size, **kw: np.ones(
            (CTF_params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float32
        ),
    )

    ctf_params = np.zeros((2, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([1.0, 2.0])
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    # Call the un-jitted Python function so monkeypatched side effects are concrete.
    core_ctf._compute_tilt_series_ctf.__wrapped__(
        ctf_params,
        image_shape=(2, 2),
        voxel_size=1.0,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    assert captured["voltage"] == pytest.approx(300.0)


def test_spa_antialiased_evaluator_shape():
    ctf_params = np.zeros((1, 9), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA_ANTIALIASED)
    out = np.asarray(evaluator(ctf_params, image_shape=(4, 4), voxel_size=1.0))
    assert out.shape == (1, 16)


def test_cryo_et_evaluator_shape():
    ctf_params = np.zeros((3, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.TILT_ANGLE] = np.array([0.0, 3.0, 6.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0

    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)
    out = np.asarray(evaluator(ctf_params, image_shape=(4, 4), voxel_size=1.0))
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
def test_spa_ctf_half_matches_full_mapping(image_shape):
    ctf_params = _make_standard_ctf_params(3)[:, :9]
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    full = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0, half_image=True))
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", [core_ctf.CTFMode.SPA, core_ctf.CTFMode.SPA_ANTIALIASED])
def test_spa_modes_half_matches_full_mapping(mode):
    ctf_params = _make_standard_ctf_params(2)[:, :9]
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=mode)
    full = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0, half_image=True))
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


def test_tilt_series_half_matches_full_mapping():
    ctf_params = _make_standard_ctf_params(4)
    image_shape = (4, 8)

    evaluator_ts = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    full = np.asarray(evaluator_ts(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half = np.asarray(evaluator_ts(ctf_params, image_shape=image_shape, voxel_size=1.0, half_image=True))
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)

    evaluator_et = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)
    full_v2 = np.asarray(evaluator_et(ctf_params, image_shape=image_shape, voxel_size=1.0))
    half_v2 = np.asarray(evaluator_et(ctf_params, image_shape=image_shape, voxel_size=1.0, half_image=True))
    expected_v2 = np.asarray(fourier_transform_utils.full_image_to_half_image(full_v2, image_shape))
    np.testing.assert_allclose(half_v2, expected_v2, atol=1e-6, rtol=1e-6)


def test_tilt_series_evaluator_half_matches_internal_function():
    ctf_params = _make_standard_ctf_params(3)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    out_eval = np.asarray(evaluator(ctf_params, image_shape, 1.0, half_image=True))
    out_ref = np.asarray(
        core_ctf._compute_tilt_series_ctf(
            ctf_params,
            image_shape,
            1.0,
            2.9,
            3.0,
            half_image=True,
        )
    )
    np.testing.assert_allclose(out_eval, out_ref, atol=1e-6, rtol=1e-6)


def test_ctf_half_pointwise_application_matches_full_mapping():
    rng = np.random.default_rng(41)
    image_shape = (6, 10)
    n_images = 3
    ctf_params = _make_standard_ctf_params(n_images)[:, :9]

    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    ctf_full = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    ctf_half = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0, half_image=True))

    real_images = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    images_full = np.asarray(fourier_transform_utils.get_dft2(real_images)).reshape(n_images, -1)
    images_half = np.asarray(fourier_transform_utils.full_image_to_half_image(images_full, image_shape))

    mapped_full_product = np.asarray(
        fourier_transform_utils.full_image_to_half_image(images_full * ctf_full, image_shape)
    )
    half_product = images_half * ctf_half
    np.testing.assert_allclose(half_product, mapped_full_product, atol=1e-6, rtol=1e-6)


def test_tilt_series_evaluator_matches_internal_function():
    ctf_params = np.zeros((2, 11), dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.DOSE] = np.array([0.0, 1.0], dtype=np.float32)
    ctf_params[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    ctf_params[:, core_ctf.CTFParamIndex.CS] = 2.7
    ctf_params[:, core_ctf.CTFParamIndex.W] = 0.1
    ctf_params[:, core_ctf.CTFParamIndex.CONTRAST] = 1.0

    evaluator = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    out_eval = np.asarray(evaluator(ctf_params, (4, 4), 1.0))
    out_ref = np.asarray(
        core_ctf._compute_tilt_series_ctf(
            ctf_params,
            image_shape=(4, 4),
            voxel_size=1.0,
            dose_per_tilt=2.9,
            angle_per_tilt=3.0,
        )
    )
    np.testing.assert_allclose(out_eval, out_ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------
import jax


@pytest.mark.gpu
def test_evaluate_ctf_on_gpu(gpu_device):
    freqs = np.array([[0.0, 0.0], [0.1, -0.2], [0.05, 0.15]], dtype=np.float32)
    ctf_params = np.array([[15000.0, 15000.0, 0.0, 300.0, 2.7, 0.1, 0.0, 0.0, 1.0]], dtype=np.float32)
    cpu_out = np.asarray(core_ctf.evaluate_ctf(freqs, ctf_params))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_ctf.evaluate_ctf(jax.device_put(freqs), jax.device_put(ctf_params)))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_batch_evaluate_ctf_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(4)[:, :9]
    psi = np.array([[0.0, 0.0], [0.1, -0.2], [0.05, 0.15]], dtype=np.float32)
    cpu_out = np.asarray(core_ctf.evaluate_ctf(psi, ctf_params))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_ctf.evaluate_ctf(jax.device_put(psi), jax.device_put(ctf_params)))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_spa_evaluator_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(3)[:, :9].astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    cpu_out = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_spa_evaluator_half_matches_full_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(3)[:, :9].astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    with jax.default_device(gpu_device):
        full = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
        half = np.asarray(
            evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0, half_image=True)
        )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [core_ctf.CTFMode.SPA, core_ctf.CTFMode.SPA_ANTIALIASED])
def test_spa_modes_evaluator_on_gpu(gpu_device, mode):
    ctf_params = _make_standard_ctf_params(2)[:, :9].astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=mode)
    cpu_out = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_spa_evaluator_half_matches_full_mapping_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(2)[:, :9].astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    with jax.default_device(gpu_device):
        full = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
        half = np.asarray(
            evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0, half_image=True)
        )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_cryo_et_evaluator_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(4).astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)
    cpu_out = np.asarray(evaluator(ctf_params, image_shape=image_shape, voxel_size=1.0))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_tilt_series_evaluator_half_matches_full_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(4).astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    with jax.default_device(gpu_device):
        full = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
        half = np.asarray(
            evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0, half_image=True)
        )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_cryo_et_evaluator_half_matches_full_on_gpu(gpu_device):
    ctf_params = _make_standard_ctf_params(4).astype(np.float32)
    image_shape = (4, 8)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)
    with jax.default_device(gpu_device):
        full = np.asarray(evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0))
        half = np.asarray(
            evaluator(jax.device_put(ctf_params), image_shape=image_shape, voxel_size=1.0, half_image=True)
        )
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_get_dose_filters_on_gpu(gpu_device):
    cpu_out = np.asarray(
        core_ctf.get_dose_filters(
            Apix=1.0,
            image_shape=(4, 8),
            cumulative_dose=np.array([0.0, 1.0, 3.0], dtype=np.float32),
            tilt_angles=np.array([0.0, 15.0, 30.0], dtype=np.float32),
            voltage=300.0,
        )
    )
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_ctf.get_dose_filters(
                Apix=1.0,
                image_shape=(4, 8),
                cumulative_dose=jax.device_put(np.array([0.0, 1.0, 3.0], dtype=np.float32)),
                tilt_angles=jax.device_put(np.array([0.0, 15.0, 30.0], dtype=np.float32)),
                voltage=300.0,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)
