"""Real native-oracle checks of the pure FP64 device VDAM transaction."""

from copy import deepcopy

import numpy as np
import pytest
from helpers.vdam import relative_metrics

from recovar.em.relion.relion_vdam_mstep import relion_vdam_m_step_device, relion_vdam_m_step_host

pytestmark = pytest.mark.unit




def _case(size=8, padding=1, radius=2, pseudo=True, moments="populated"):
    rng = np.random.default_rng(29)
    m = padding * size
    shape = (2 * (padding * radius + 1) + 1,) * 2
    shape += (shape[0] // 2 + 1,)
    moment_shape = (m, m, m // 2 + 1)
    d0, d1 = [rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2)]
    weights = [rng.uniform(0.0, 3.0, shape) for _ in range(2)]
    # Cover exact max(1,w) edges, including zero, in actual active support.
    for weight in weights:
        weight[shape[0] // 2, shape[0] // 2, :3] = [0.0, 1.0, np.nextafter(1.0, 2.0)]
    first = [rng.normal(size=moment_shape) + 1j * rng.normal(size=moment_shape) for _ in range(2)]
    if moments == "zero":
        first = [np.zeros(moment_shape, np.complex128) for _ in range(2)]
    elif moments == "imaginary":
        first = [1j * np.ones(moment_shape, np.complex128) for _ in range(2)]
    elif moments == "cancelling":
        first = [np.zeros(moment_shape, np.complex128) for _ in range(2)]
        for value in first:
            value.ravel()[:4] = [1e16, 1, -1e16, 0]
            value.imag[:] = 2.0
    second = rng.uniform(0.1, 2.0, moment_shape) + 1j * rng.uniform(0.1, 1.0, moment_shape)
    tau = rng.uniform(0.1, 2.0, size // 2 + 1)
    tau[0] = 0.0
    return dict(
        reference_relion=rng.normal(size=(size,) * 3),
        data_h0=d0,
        weight_h0=weights[0],
        data_h1=d1 if pseudo else None,
        weight_h1=weights[1] if pseudo else None,
        mom1_h0=first[0],
        mom1_h1=first[1] if pseudo else None,
        mom2=second,
        fsc_ssnr=rng.uniform(0.1, 0.9, size // 2 + 1),
        fsc_reconstruct=rng.uniform(0.1, 0.9, size // 2 + 1),
        tau2=tau,
        grad_stepsize=0.3,
        tau2_fudge=4.0,
        ori_size=size,
        padding_factor=padding,
        interpolator=1,
        r_max=radius,
        min_resol_shell=3.0,
    )


def _native(bind, case):
    args = case.copy()
    args["vol_relion"] = args.pop("reference_relion")
    return bind.vdam_m_step_transaction(**args)


@pytest.fixture(scope="module")
def bind():
    from recovar.relion_bind import _relion_bind_core

    assert hasattr(_relion_bind_core, "vdam_m_step_transaction")
    assert hasattr(_relion_bind_core, "vdam_first_moment_initializes"), "build native branch certificate"
    return _relion_bind_core


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize("moments", ["zero", "populated", "imaginary", "cancelling"])
def test_complete_device_transaction_native_fp64(bind, size, padding, full, pseudo, moments):
    case = _case(size, padding, size // (2 if full else 4), pseudo, moments)
    _assert_case(bind, case)


def _assert_case(bind, case):
    size, padding = case["ori_size"], case["padding_factor"]
    pseudo = case["data_h1"] is not None
    before = deepcopy(case)
    expected = _native(bind, case)
    actual = relion_vdam_m_step_host(**case)
    assert actual.keys() == expected.keys()
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None
            continue
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].shape == expected[key].shape
        metrics = relative_metrics(expected[key], actual[key])
        # Existing native-projector FP64 contract; never widened for this port.
        assert np.all(metrics < 1e-12), (key, metrics)
    np.testing.assert_array_equal(actual["tau2"], case["tau2"])
    for key in case:
        if isinstance(case[key], np.ndarray):
            np.testing.assert_array_equal(case[key], before[key], err_msg=key)
    m = padding * size
    coord = np.arange(m) - m // 2
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + np.arange(m // 2 + 1)[None, None, :] ** 2
    outside = r2 >= (padding * case["r_max"]) ** 2
    np.testing.assert_array_equal(actual["mom1_h0"][outside], case["mom1_h0"][outside])
    np.testing.assert_array_equal(actual["mom2"][outside], case["mom2"][outside])
    if pseudo:
        np.testing.assert_array_equal(actual["mom2"].imag[~outside], 0.0)
    else:
        np.testing.assert_array_equal(actual["mom2"], case["mom2"])
        np.testing.assert_array_equal(actual["mom1_noise_power"], 0.0)


def test_native_branch_certificate_real_only_and_serial_order(bind):
    values = np.zeros((8, 8, 5), np.complex128)
    values.imag[:] = 1.0
    assert bind.vdam_first_moment_initializes(values)
    values.ravel()[:4] = [1e16, 1, -1e16, 0]
    assert bind.vdam_first_moment_initializes(values)
    values.ravel()[:4] = [1e16, -1e16, 1, 0]
    assert not bind.vdam_first_moment_initializes(values)
    assert bind.vdam_first_moment_initializes(values[:, :, ::-1])


def test_dynamic_radius_reuses_device_executable(bind):
    relion_vdam_m_step_device.clear_cache()
    for radius in (1, 2, 8, 0):
        case = _case(size=16, radius=radius or 8)
        case["r_max"] = radius
        result = relion_vdam_m_step_host(**case)
        assert np.all(np.isfinite(result["iref"]))
    assert relion_vdam_m_step_device._cache_size() == 1


@pytest.mark.parametrize("kind", ["tiny_sigma", "negative_tau", "nan_sigma", "nan_tau"])
def test_native_ssnr_error_branches(bind, kind):
    case = _case(size=16)
    if kind == "tiny_sigma":
        case["weight_h0"].fill(1e-30)
    elif kind == "nan_sigma":
        case["weight_h0"].fill(np.nan)
    elif kind == "nan_tau":
        case["tau2"][0] = np.nan
    else:
        case["tau2"][0] = -1.0
    with pytest.raises(Exception):
        _native(bind, case)
    with pytest.raises(ValueError, match="native SSNR rejects"):
        relion_vdam_m_step_host(**case)


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize("moments", ["zero", "populated", "imaginary", "cancelling"])
def test_physical_hermitian_transaction_native_fp64(bind, size, padding, full, pseudo, moments):
    case = _case(size, padding, size // (2 if full else 4), pseudo, moments)
    # Half-volume inputs require Hermitian symmetry on the x=0 plane. Generate
    # that physical contract explicitly. The arbitrary-input panel also remains;
    # its FFT8 cases now verify the original native capability fallback.
    for key in ("data_h0", "weight_h0", "data_h1", "weight_h1", "mom1_h0", "mom1_h1", "mom2"):
        value = case[key]
        if value is None:
            continue
        n = value.shape[0]
        indices = (2 * (n // 2) - np.arange(n)) % n
        value[:, :, 0] = (value[:, :, 0] + value[indices[:, None], indices[None, :], 0].conj()) / 2.0
    _assert_case(bind, case)


@pytest.mark.parametrize("size,padding,expected_backend", [(8, 1, "native"), (8, 2, "device"), (16, 1, "device")])
def test_host_backend_capability_attestation(bind, monkeypatch, size, padding, expected_backend):
    from recovar.em.relion import relion_vdam_mstep as helper

    case = _case(size=size, padding=padding, radius=size // 4)
    expected = _native(bind, case)
    calls = []
    native_transaction = bind.vdam_m_step_transaction
    device_transaction = helper.relion_vdam_m_step_device

    def native(*args, **kwargs):
        calls.append("native")
        assert expected_backend == "native"
        return native_transaction(*args, **kwargs)

    def device(*args, **kwargs):
        calls.append("device")
        assert expected_backend == "device"
        return device_transaction(*args, **kwargs)

    monkeypatch.setattr(bind, "vdam_m_step_transaction", native)
    monkeypatch.setattr(helper, "relion_vdam_m_step_device", device)
    if expected_backend == "native":

        def forbidden_certificate(*args, **kwargs):
            raise AssertionError("native fallback must not prepare device inputs")

        monkeypatch.setattr(bind, "vdam_first_moment_initializes", forbidden_certificate)
    actual = helper.relion_vdam_m_step_host(**case)
    assert calls == [expected_backend]
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None
        elif expected_backend == "native":
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert np.all(relative_metrics(actual[key], expected[key]) < 1e-12)


def test_pure_device_rejects_unsupported_small_fft():
    with pytest.raises(ValueError, match="FFT grid >=16"):
        relion_vdam_m_step_device(
            *([None] * 10),
            0.3,
            4.0,
            2,
            False,
            False,
            ori_size=8,
            padding_factor=1,
            pseudo_halfsets=True,
        )
