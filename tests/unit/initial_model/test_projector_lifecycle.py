"""Exact projector/spectrum reuse and the refresh-to-E-step lifetime."""
from dataclasses import replace

import numpy as np
import pytest

from recovar.em.initial_model import dense_adapter as adapter
from recovar.em.initial_model import driver, initialise_denovo_state
from recovar.em.initial_model import iteration_loop as loop
from recovar.em.initial_model.subset import numpy_rnd_unif_factory
from recovar.utils.helpers import recovar_volume_to_relion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("classes,current_size,padding", [(1, 8, 1), (2, 12, 1), (1, 16, 2)])
def test_shared_projector_matches_both_original_native_calls(
    classes, current_size, padding, monkeypatch, tmp_path
):
    from recovar.relion_bind import _relion_bind_core as bind

    state = initialise_denovo_state(
        ori_size=16, pixel_size=1.0, K=classes, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    state.Iref = np.random.default_rng(17).normal(size=state.Iref.shape)
    state.current_size = current_size
    references_before = state.Iref.copy()
    native_projector = bind.compute_fourier_transform_map
    builds = []

    def counted_projector(*args):
        builds.append(1)
        return native_projector(*args)

    monkeypatch.setattr(bind, "compute_fourier_transform_map", counted_projector)
    monkeypatch.setenv(adapter._RELION_PROJECTOR_DUMP_DIR_ENV, str(tmp_path))
    inputs, power = adapter.prepare_relion_projector_class_inputs_and_power(
        state, padding_factor=padding
    )
    assert len(builds) == classes
    with np.load(tmp_path / "iter000_relion_projector_half.npz") as dumped:
        np.testing.assert_array_equal(dumped["projector_half"], inputs[2])
        assert int(dumped["current_size"]) == current_size
        assert int(dumped["padding_factor"]) == padding
    expected_half = []
    expected_power = []
    for ref in state.Iref:
        native = np.ascontiguousarray(recovar_volume_to_relion(ref))
        half, _, _, _, radius, _, _ = bind.compute_fourier_transform_map(
            native, 16, padding, 1, current_size, True, 2
        )
        expected_half.append(np.asarray(half, dtype=np.complex64))
        expected_power.append(bind.vdam_projector_power_spectrum(
            native, 16, padding, 1, current_size, True, 2
        ))
        assert inputs[3] == radius
    np.testing.assert_array_equal(inputs[2], np.asarray(expected_half))
    np.testing.assert_array_equal(power, np.asarray(expected_power))
    old_inputs = adapter.prepare_relion_projector_class_inputs(state, padding_factor=padding)
    for actual, expected in zip(inputs, old_inputs):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state.Iref, references_before)


@pytest.mark.parametrize("backend", ["native", "jax"])
def test_context_builds_once_and_consumes_once(monkeypatch, backend):
    state = initialise_denovo_state(
        ori_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    calls = []

    def prepare(current, *, padding_factor, interpolator, projector_setup_backend):
        assert projector_setup_backend == backend
        calls.append((current.iter, current.Iref.copy()))
        return (None, None, current.Iref.copy(), 4), np.full((1, 5), current.iter)

    monkeypatch.setattr(driver, "prepare_relion_projector_class_inputs_and_power", prepare)
    ctx = driver._IterationProjectorContext(projector_setup_backend=backend)
    for iteration in (1, 2):
        state = replace(state, iter=iteration, Iref=state.Iref + 1)
        before = state.tau2_class.copy()
        refreshed = ctx.refresh(state, padding_factor=1, interpolator=1)
        np.testing.assert_array_equal(state.tau2_class, before)
        np.testing.assert_array_equal(refreshed.tau2_class, np.full((1, 5), iteration))
        inputs = ctx.take(refreshed, padding_factor=1)
        np.testing.assert_array_equal(inputs[2], state.Iref)
        assert ctx.take(refreshed, padding_factor=1) is None
        assert ctx.reference is None
    assert len(calls) == 2
    assert not np.array_equal(calls[0][1], calls[1][1])


@pytest.mark.parametrize("change", ["reference", "iteration", "geometry", "padding"])
def test_context_rejects_stale_handoff_and_clears(monkeypatch, change):
    state = initialise_denovo_state(
        ori_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    monkeypatch.setattr(driver, "prepare_relion_projector_class_inputs_and_power",
                        lambda *a, **k: ((None, None, None, 4), np.ones((1, 5))))
    ctx = driver._IterationProjectorContext()
    current = ctx.refresh(state, padding_factor=1, interpolator=1)
    kwargs = {"padding_factor": 1}
    if change == "reference":
        current = replace(current, Iref=current.Iref.copy())
    elif change == "iteration":
        current = replace(current, iter=1)
    elif change == "geometry":
        current = replace(current, current_size=6)
    else:
        kwargs["padding_factor"] = 2
    with pytest.raises(ValueError, match="reference or geometry"):
        ctx.take(current, **kwargs)
    assert ctx.prepared is ctx.reference is ctx.geometry is None


@pytest.mark.parametrize("refresh_enabled", [False, True])
@pytest.mark.parametrize("mstep_backend", ["native", "jax"])
def test_loop_callback_is_once_before_estep_and_respects_disabled(monkeypatch, refresh_enabled, mstep_backend):
    state = initialise_denovo_state(
        ori_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    events = []

    def refresh(current, *, padding_factor, interpolator):
        events.append((current.iter, "refresh", current.current_size))
        assert padding_factor == interpolator == 1
        return current

    def estep(current, ids, halves):
        events.append((current.iter, "estep", current.current_size))
        return [], {"max_posterior_per_image": np.ones(len(ids))}

    def mstep(current, **kwargs):
        assert kwargs["mstep_backend"] == mstep_backend
        return current

    monkeypatch.setattr(loop, "vdam_m_step", mstep)
    loop.run_vdam_iterations(
        state, nr_particles=20, optics_group_by_particle=[0] * 20,
        grad_ini_subset_size=10, grad_fin_subset_size=10, tau2_fudge_arg=4.0,
        grad_em_iters=0, random_seed=29, rnd_unif_factory=numpy_rnd_unif_factory,
        expectation_step=estep, refresh_tau2_from_projector=refresh_enabled,
        projector_refresh_fn=refresh, mstep_backend=mstep_backend,
    )
    expected = ["refresh", "estep"] * 2 if refresh_enabled else ["estep"] * 2
    assert [event[1] for event in events] == expected
