"""Typed M precision routing and publication; not full-trajectory qualification."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers.vdam import numpy_rnd_unif_factory

from recovar.em.vdam import driver, iteration_loop, m_step, mstep_single_class, native_options
from recovar.em.vdam.init import initialise_denovo_state
from scripts import run_ab_initio

pytestmark = pytest.mark.unit

F32_STATE = {
    "Iref": np.float32,
    "Igrad1": np.complex64,
    "Igrad2": np.complex64,
    "sigma2_class": np.float32,
    "data_vs_prior_class": np.float32,
    "fourier_coverage_class": np.float32,
}
REPLAYS = [
    mstep_single_class.VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV,
    mstep_single_class.VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV,
    mstep_single_class.VDAM_NATIVE_BPREF_DATA_REPLAY_ENV,
    mstep_single_class.VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV,
    mstep_single_class.VDAM_NATIVE_IREF_INPUT_REPLAY_ENV,
    "RECOVAR_MSTEP_DUMP_DIR",
]


@pytest.fixture(autouse=True)
def clean_diagnostics(monkeypatch):
    for name in REPLAYS + [driver.INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV]:
        monkeypatch.delenv(name, raising=False)


def _state(K=1):
    state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=K, nr_iter=2, n_directions=3, pseudo_halfsets=True)
    state.tau2_class[:] = np.nextafter(1.0, 2.0)
    return state


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_cli_forwards_explicit_precision_and_keeps_legacy_default(monkeypatch, dtype):
    calls = []
    monkeypatch.setattr(
        driver,
        "run_native_initial_model",
        lambda opts: calls.append(opts) or SimpleNamespace(final_mrc="a.mrc", final_model_star="a.star"),
    )
    argv = ["--i", "missing.star"]
    if dtype == "float32":
        argv += ["--mstep-compute-dtype", dtype, "--mstep-backend", "jax"]
    assert run_ab_initio.main(argv) == 0
    assert calls[0].mstep_compute_dtype == dtype
    assert calls[0].mstep_backend == ("jax" if dtype == "float32" else "native")


def test_cli_rejects_native_float32_before_driver(monkeypatch):
    monkeypatch.setattr(driver, "run_native_initial_model", lambda *_: pytest.fail("driver called"))
    with pytest.raises(SystemExit):
        run_ab_initio.main(["--i", "missing.star", "--mstep-compute-dtype", "float32"])


@pytest.mark.parametrize("K", [1, 4])
def test_one_time_conversion_only_changes_m_owned_state(K):
    state = _state(K)
    state.Iref[:] = np.nextafter(1.0, 2.0)
    before = {
        f.name: getattr(state, f.name).tobytes()
        for f in fields(state)
        if isinstance(getattr(state, f.name), np.ndarray)
    }
    assert driver._prepare_mstep_state_precision(state, "float64") is state
    converted = driver._prepare_mstep_state_precision(state, "float32")
    for f in fields(state):
        original, actual = getattr(state, f.name), getattr(converted, f.name)
        if f.name in F32_STATE:
            assert actual.dtype == np.dtype(F32_STATE[f.name])
            np.testing.assert_array_equal(actual, original.astype(F32_STATE[f.name]))
            assert not np.shares_memory(actual, original)
        else:
            assert actual is original
        if isinstance(original, np.ndarray):
            assert original.tobytes() == before[f.name]
    assert converted.tau2_class.dtype == np.float64
    assert converted.tau2_class[0, 0] != float(np.float32(converted.tau2_class[0, 0]))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_driver_converts_before_initial_artifact_and_forwards_loop(monkeypatch, tmp_path, dtype):
    state = _state()
    dataset = SimpleNamespace(n_images=20, voxel_size=1.0, tilt_series_flag=False)
    monkeypatch.setattr(driver, "read_star", lambda _: (pd.DataFrame(index=range(20)), None))
    monkeypatch.setattr(driver, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(driver, "maybe_cache_raw_image_loaders", lambda _: None)
    monkeypatch.setattr(driver, "_configure_relion_image_mask", lambda *a: None)
    monkeypatch.setattr(driver, "_native_optics_state", lambda *a: None)
    monkeypatch.setattr(driver, "_particle_state_from_star", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_initial_sampling_state", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(driver, "_build_sampling_plan", lambda *a, **k: SimpleNamespace(rotations=None))
    monkeypatch.setattr(driver, "_initial_state_from_particles", lambda *a, **k: (state, np.zeros(20, int)))
    monkeypatch.setattr(driver, "_native_expectation_step", lambda *a, **k: None)
    observed = []
    monkeypatch.setattr(
        driver,
        "_write_iteration_artifacts",
        lambda _, current, iteration, *a, **k: observed.append((iteration, current)),
    )

    class AtLoop(Exception):
        pass

    def at_loop(current, **kwargs):
        assert len(observed) == 1 and observed[0] == (0, current)
        assert kwargs.get("mstep_compute_dtype", "float64") == dtype
        expected = np.float32 if dtype == "float32" else np.float64
        assert current.Iref.dtype == expected
        assert current.tau2_class is state.tau2_class
        masked = kwargs["post_mstep_update"](current, 1, {})
        assert masked.Iref.dtype == expected
        raise AtLoop

    monkeypatch.setattr(driver, "run_vdam_iterations", at_loop)
    with pytest.raises(AtLoop):
        driver.run_native_initial_model(
            native_options.NativeInitialModelOptions(
                fn_img="missing.star",
                outputname=str(tmp_path / "run"),
                nr_iter=2,
                mstep_backend="jax",
                mstep_compute_dtype=dtype,
            )
        )


@pytest.mark.parametrize("dtype,backend", [("float32", "native"), ("float16", "jax")])
def test_driver_invalid_precision_rejected_before_io(dtype, backend):
    with pytest.raises(ValueError, match="M-step|mstep_compute_dtype"):
        driver.run_native_initial_model(
            native_options.NativeInitialModelOptions(fn_img="missing.star", mstep_backend=backend, mstep_compute_dtype=dtype)
        )


def test_driver_reference_replay_rejected_before_io(monkeypatch):
    monkeypatch.setenv(driver.INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, "missing_{iteration}.mrc")
    with pytest.raises(ValueError, match="reference replay"):
        driver.run_native_initial_model(
            native_options.NativeInitialModelOptions(fn_img="missing.star", mstep_backend="jax", mstep_compute_dtype="float32")
        )


def _accum(state, k=0):
    return m_step.VdamAccumulator(np.zeros((19, 19, 10), np.complex128), np.ones((19, 19, 10), np.float64), k, 0)


def _call(state, **kwargs):
    return m_step.vdam_m_step_single_class(
        state,
        k=state.K - 1,
        accum_h0=_accum(state),
        accum_h1=_accum(state),
        grad_current_stepsize=0.0,
        tau2_fudge_factor=4.0,
        mstep_backend="jax",
        mstep_compute_dtype="float32",
        **kwargs,
    )


@pytest.mark.parametrize("env", REPLAYS)
def test_float32_rejects_diagnostics_before_replay_consumption(monkeypatch, env):
    monkeypatch.setenv(env, "missing")
    monkeypatch.setattr(
        mstep_single_class, "_maybe_replay_native_bpref_accumulators", lambda *a, **k: pytest.fail("consumed replay")
    )
    with pytest.raises(ValueError, match=env):
        _call(driver._prepare_mstep_state_precision(_state(), "float32"))


def test_float32_rejects_primitive_route_before_overrides(monkeypatch):
    monkeypatch.setattr(
        mstep_single_class, "_maybe_replay_native_bpref_accumulators", lambda *a, **k: pytest.fail("consumed replay")
    )
    with pytest.raises(ValueError, match="transaction"):
        _call(driver._prepare_mstep_state_precision(_state(), "float32"), use_native_transaction=False)


@pytest.mark.parametrize("field", list(F32_STATE))
def test_float32_rejects_mixed_publication_state_before_execution(monkeypatch, field):
    state = driver._prepare_mstep_state_precision(_state(), "float32")
    value = getattr(state, field)
    setattr(state, field, value.astype(np.complex128 if np.iscomplexobj(value) else np.float64))
    monkeypatch.setattr(mstep_single_class, "_get_bindings", lambda: pytest.fail("entered transaction"))
    with pytest.raises(ValueError, match=field):
        _call(state)


def test_real_host_device_transaction_publishes_f32_and_preserves_k4_other_slots(monkeypatch):
    from recovar.em.relion import relion_vdam_mstep as helper

    state = driver._prepare_mstep_state_precision(_state(4), "float32")
    state.Iref[:] = np.arange(1, 5, dtype=np.float32)[:, None, None, None]
    before = {name: getattr(state, name).copy() for name in F32_STATE}
    # Exact-zero first moments have an unambiguous native initialization certificate.
    bind = SimpleNamespace(
        vdam_m_step_transaction=lambda *a: pytest.fail("native executed"), vdam_first_moment_initializes=lambda *_: True
    )
    monkeypatch.setattr(mstep_single_class, "_get_bindings", lambda: bind)
    from recovar.relion_bind import _relion_bind_core

    monkeypatch.setattr(_relion_bind_core, "vdam_first_moment_initializes", bind.vdam_first_moment_initializes)
    actual = _call(state)
    for name, dtype in F32_STATE.items():
        assert getattr(actual, name).dtype == np.dtype(dtype)
        slots = [0, 1, 2, 4, 5, 6] if name == "Igrad1" else [0, 1, 2]
        assert getattr(actual, name)[slots].tobytes() == before[name][slots].tobytes()
        assert getattr(state, name).tobytes() == before[name].tobytes()
    np.testing.assert_array_equal(actual.Iref[3], state.Iref[3])
    assert actual.tau2_class.tobytes() == state.tau2_class.tobytes()
    assert actual.tau2_class.dtype == np.float64
    assert helper.relion_vdam_m_step_host is not None


@pytest.mark.parametrize("bad_field", ["iref", "mom1_h0", "sigma2", "tau2"])
def test_publication_rejects_wrong_result_dtype_or_rounded_prior(bad_field):
    state = driver._prepare_mstep_state_precision(_state(), "float32")
    result = dict(
        iref=state.Iref[0],
        mom1_h0=state.Igrad1[0],
        mom1_h1=state.Igrad1[1],
        mom2=state.Igrad2[0],
        tau2=state.tau2_class[0],
        sigma2=state.sigma2_class[0],
        data_vs_prior=state.data_vs_prior_class[0],
        fourier_coverage=state.fourier_coverage_class[0],
    )
    result[bad_field] = result[bad_field].astype(
        np.float32 if bad_field == "tau2" else np.complex128 if "mom" in bad_field else np.float64
    )
    with pytest.raises(ValueError, match="output|authoritative tau2"):
        mstep_single_class._run_m_step_transaction(
            lambda *a: result,
            state,
            0,
            _accum(state),
            _accum(state),
            grad_current_stepsize=0,
            tau2_fudge_factor=4,
            padding_factor=1,
            r_max=8,
            min_resol_shell=1,
            mstep_compute_dtype="float32",
        )


def test_actual_loop_forwards_f32_to_m_without_changing_authoritative_state(monkeypatch):
    state = driver._prepare_mstep_state_precision(_state(), "float32")
    calls = []

    def step(current, **kwargs):
        assert kwargs["mstep_compute_dtype"] == "float32"
        assert current.Iref.dtype == np.float32
        calls.append(current.iter)
        return current

    monkeypatch.setattr(m_step, "vdam_m_step_single_class", step)
    iteration_loop.run_vdam_iterations(
        state,
        nr_particles=20,
        optics_group_by_particle=[0] * 20,
        grad_ini_subset_size=10,
        grad_fin_subset_size=10,
        tau2_fudge_arg=4.0,
        grad_em_iters=0,
        random_seed=29,
        rnd_unif_factory=numpy_rnd_unif_factory,
        expectation_step=lambda current, ids, halves: (
            [_accum(current), _accum(current)],
            {"max_posterior_per_image": np.ones(len(ids))},
        ),
        refresh_tau2_from_projector=False,
        mstep_backend="jax",
        mstep_compute_dtype="float32",
    )
    assert calls == [1, 2]


def test_solvent_route_uses_explicit_f32_product_and_preserves_default():
    state = driver._prepare_mstep_state_precision(_state(), "float32")
    state.Iref[:] = np.float32(1.0000001192092896)
    mask = np.full((16,) * 3, 1.0000000596046446, dtype=np.float64)
    default = iteration_loop.relion_solvent_flatten_state(state, mask=mask)
    actual = iteration_loop.relion_solvent_flatten_state(state, mask=mask, compute_dtype="float32")
    np.testing.assert_array_equal(default.Iref, (state.Iref * mask).astype(np.float32))
    np.testing.assert_array_equal(actual.Iref, state.Iref * mask.astype(np.float32))
    assert actual.Iref.dtype == np.float32 and actual.tau2_class is state.tau2_class
    assert np.any(default.Iref != actual.Iref)
    with pytest.raises(ValueError, match="float32 state.Iref"):
        iteration_loop.relion_solvent_flatten_state(_state(), mask=mask, compute_dtype="float32")


@pytest.mark.parametrize("missing", ["vdam_m_step_transaction", "vdam_first_moment_initializes"])
def test_float32_missing_native_certificate_capability_never_falls_back(monkeypatch, missing):
    bindings = {
        "vdam_m_step_transaction": lambda *a: pytest.fail("native executed"),
        "vdam_first_moment_initializes": lambda *a: True,
    }
    del bindings[missing]
    monkeypatch.setattr(mstep_single_class, "_get_bindings", lambda: SimpleNamespace(**bindings))
    with pytest.raises(RuntimeError, match="requires"):
        _call(driver._prepare_mstep_state_precision(_state(), "float32"))
