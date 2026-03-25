"""Equivalence tests proving CTFEvaluator produces identical output to the old
internal functions for every mode, shape, and half-image combination.

These tests were written as part of the CTF backward-compat removal to verify
that no mode was inadvertently broken.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core.ctf as core_ctf
import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: realistic CTF parameters (not zeros)
# ---------------------------------------------------------------------------


def _make_realistic_ctf_params(n, n_cols=11):
    """Return ``(n, n_cols)`` CTF params with realistic, non-zero values."""
    rng = np.random.default_rng(42)
    p = np.zeros((n, n_cols), dtype=np.float32)
    p[:, core_ctf.CTFParamIndex.DFU] = rng.uniform(8000, 25000, size=n)
    p[:, core_ctf.CTFParamIndex.DFV] = p[:, 0] + rng.uniform(-2000, 2000, size=n)
    p[:, core_ctf.CTFParamIndex.DFANG] = rng.uniform(0, 180, size=n)
    p[:, core_ctf.CTFParamIndex.VOLT] = 300.0
    p[:, core_ctf.CTFParamIndex.CS] = 2.7
    p[:, core_ctf.CTFParamIndex.W] = 0.07
    p[:, core_ctf.CTFParamIndex.PHASE_SHIFT] = rng.uniform(0, 10, size=n)
    p[:, core_ctf.CTFParamIndex.BFACTOR] = rng.uniform(0, 50, size=n)
    p[:, core_ctf.CTFParamIndex.CONTRAST] = rng.uniform(0.8, 1.2, size=n)
    if n_cols > 9:
        p[:, core_ctf.CTFParamIndex.DOSE] = rng.uniform(0.5, 5.0, size=n)
    if n_cols > 10:
        p[:, core_ctf.CTFParamIndex.TILT_ANGLE] = rng.uniform(0, 60, size=n)
    return p


# ---------------------------------------------------------------------------
# Phase 1 equivalence tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_shape", [(8, 8), (6, 10)])
@pytest.mark.parametrize("half_image", [False, True])
def test_spa_mode_bitwise_identical(image_shape, half_image):
    """CTFEvaluator(SPA) must be bitwise identical to _compute_spa_ctf."""
    params = _make_realistic_ctf_params(4, n_cols=9)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)

    out_eval = np.asarray(evaluator(params, image_shape, 1.5, half_image=half_image))
    out_ref = np.asarray(core_ctf._compute_spa_ctf(params, image_shape, 1.5, half_image=half_image))

    np.testing.assert_array_equal(out_eval, out_ref)


@pytest.mark.parametrize("image_shape", [(8, 8), (6, 10)])
@pytest.mark.parametrize("half_image", [False, True])
def test_spa_antialiased_mode_bitwise_identical(image_shape, half_image):
    """CTFEvaluator(SPA_ANTIALIASED) must be bitwise identical to _compute_spa_ctf_antialiased."""
    params = _make_realistic_ctf_params(3, n_cols=9)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA_ANTIALIASED)

    out_eval = np.asarray(evaluator(params, image_shape, 1.5, half_image=half_image))
    out_ref = np.asarray(core_ctf._compute_spa_ctf_antialiased(params, image_shape, 1.5, half_image=half_image))

    np.testing.assert_array_equal(out_eval, out_ref)


@pytest.mark.parametrize("image_shape", [(8, 8), (6, 10)])
@pytest.mark.parametrize("half_image", [False, True])
def test_tilt_series_mode_bitwise_identical(image_shape, half_image):
    """CTFEvaluator(TILT_SERIES) must be bitwise identical to _compute_tilt_series_ctf."""
    params = _make_realistic_ctf_params(5, n_cols=11)
    dose_per_tilt = 2.9
    angle_per_tilt = 3.0
    evaluator = core_ctf.CTFEvaluator(
        mode=core_ctf.CTFMode.TILT_SERIES,
        dose_per_tilt=dose_per_tilt,
        angle_per_tilt=angle_per_tilt,
    )

    out_eval = np.asarray(evaluator(params, image_shape, 1.5, half_image=half_image))
    out_ref = np.asarray(
        core_ctf._compute_tilt_series_ctf(
            params,
            image_shape,
            1.5,
            dose_per_tilt,
            angle_per_tilt,
            half_image=half_image,
        )
    )

    np.testing.assert_array_equal(out_eval, out_ref)


@pytest.mark.parametrize("image_shape", [(8, 8), (6, 10)])
@pytest.mark.parametrize("half_image", [False, True])
def test_cryo_et_mode_bitwise_identical(image_shape, half_image):
    """CTFEvaluator(CRYO_ET) must be bitwise identical to _compute_cryo_et_ctf."""
    params = _make_realistic_ctf_params(4, n_cols=11)
    evaluator = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.CRYO_ET)

    out_eval = np.asarray(evaluator(params, image_shape, 1.5, half_image=half_image))
    out_ref = np.asarray(core_ctf._compute_cryo_et_ctf(params, image_shape, 1.5, half_image=half_image))

    np.testing.assert_array_equal(out_eval, out_ref)


@pytest.mark.parametrize(
    "mode,n_cols",
    [
        (core_ctf.CTFMode.SPA, 9),
        (core_ctf.CTFMode.SPA_ANTIALIASED, 9),
        (core_ctf.CTFMode.TILT_SERIES, 11),
        (core_ctf.CTFMode.CRYO_ET, 11),
    ],
)
def test_ctf_evaluator_output_dtype_is_float32(mode, n_cols):
    """All modes return float32 — proves the old CTF_fun() dtype cast was redundant."""
    params = _make_realistic_ctf_params(3, n_cols=n_cols)
    kwargs = {}
    if mode == core_ctf.CTFMode.TILT_SERIES:
        kwargs = dict(dose_per_tilt=2.9, angle_per_tilt=3.0)
    evaluator = core_ctf.CTFEvaluator(mode=mode, **kwargs)
    out = evaluator(params, (8, 8), 1.5)
    assert out.dtype == np.float32


@pytest.mark.parametrize("image_shape", [(8, 8), (6, 10)])
def test_compute_ctf_and_compute_ctf_half_consistent(image_shape):
    """config.compute_ctf() full spectrum matches config.compute_ctf_half() via known mapping."""
    from recovar.core.configs import ForwardModelConfig

    params = _make_realistic_ctf_params(3, n_cols=9)
    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=(8, 8, 8),
        grid_size=8,
        voxel_size=1.5,
        padding=0,
        disc_type="linear_interp",
        ctf=core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA),
    )

    full = np.asarray(config.compute_ctf(params))
    half = np.asarray(config.compute_ctf_half(params))
    expected_half = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected_half, atol=1e-6, rtol=1e-6)


def test_compute_ctf_at_shape_matches_direct_call():
    """config.compute_ctf_at_shape(params, upsampled_shape) == config.ctf(params, upsampled_shape, vs)."""
    from recovar.core.configs import ForwardModelConfig

    params = _make_realistic_ctf_params(2, n_cols=9)
    config = ForwardModelConfig(
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        grid_size=8,
        voxel_size=1.5,
        padding=0,
        disc_type="linear_interp",
        ctf=core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA),
    )

    upsampled_shape = (16, 16)
    out_method = np.asarray(config.compute_ctf_at_shape(params, upsampled_shape))
    out_direct = np.asarray(config.ctf(params, upsampled_shape, config.voxel_size))
    np.testing.assert_array_equal(out_method, out_direct)


def test_as_ctf_evaluator_identity_and_wrapping():
    """Identity for CTFEvaluator, wraps callables correctly."""
    ev = core_ctf.CTFEvaluator(mode=core_ctf.CTFMode.SPA)
    assert core_ctf.as_ctf_evaluator(ev) is ev

    adapter = core_ctf._LegacyCTFAdapter(lambda p, s, v, **kw: np.ones(10))
    assert core_ctf.as_ctf_evaluator(adapter) is adapter

    fn = lambda p, s, v, **kw: np.ones(10)
    wrapped = core_ctf.as_ctf_evaluator(fn)
    assert isinstance(wrapped, core_ctf._LegacyCTFAdapter)
    assert core_ctf.as_ctf_evaluator(wrapped) is wrapped
