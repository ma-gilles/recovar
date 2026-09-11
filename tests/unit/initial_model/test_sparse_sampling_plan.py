"""Sparse plans retain trial geometry without constructing unused fine matrices."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.initial_model import driver, initialise_denovo_state, native_options, native_sampling

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("order,oversampling", [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)])
@pytest.mark.parametrize("perturbation", [0.0, -0.375])
def test_deferred_plan_preserves_geometry(monkeypatch, order, oversampling, perturbation):
    opts = native_options.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=order,
        oversampling=oversampling,
        random_perturbation=perturbation,
    )
    dense = native_sampling._build_sampling_plan(opts, iteration=3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unused full fine grid was materialized")

    monkeypatch.setattr(driver.sampling, "get_oversampled_relion_hidden_rotation_grid_from_samples", forbidden)
    sparse = native_sampling._build_sampling_plan(opts, iteration=3, defer_fine_rotations=True)
    assert sparse.n_rotations == dense.n_rotations == len(dense.rotations)
    if oversampling:
        assert sparse.rotations is None
    else:
        np.testing.assert_array_equal(sparse.rotations, dense.rotations)
    for field in fields(dense):
        if field.name != "rotations":
            np.testing.assert_array_equal(getattr(sparse, field.name), getattr(dense, field.name))


def test_deferred_plan_cannot_enter_dense_execution(monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_SPARSE_PASS2", "1")
    opts = native_options.NativeInitialModelOptions(fn_img="particles.star", healpix_order=0)
    plan = native_sampling._build_sampling_plan(opts, defer_fine_rotations=True)
    with pytest.raises(ValueError, match="Deferred fine rotations require sparse"):
        driver._dense_estep_config(
            SimpleNamespace(voxel_size=1.0, n_images=2),
            opts,
            np.ones(33, dtype=np.float32),
            plan,
            np.zeros((2, 2), dtype=np.float32),
        )


@pytest.mark.parametrize("selector", ["0", "1", "invalid"])
def test_expectation_deferred_plan_routing_and_metadata(monkeypatch, selector):
    monkeypatch.setenv("RECOVAR_VDAM_DEFER_SPARSE_ROTATIONS", selector)
    monkeypatch.delenv("RECOVAR_DISABLE_SPARSE_PASS2", raising=False)
    opts = native_options.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=0,
        oversampling=1,
        random_perturbation=0.25,
    )
    dense = native_sampling._build_sampling_plan(opts, iteration=3)
    expected_count = len(dense.rotations)
    calls = []

    def fake_run(dataset, state, config, *, particle_ids, halfset_ids):
        calls.append(config)
        if selector == "1":
            assert config.rotations is None
        else:
            np.testing.assert_array_equal(config.rotations, dense.rotations)
        np.testing.assert_array_equal(config.translations, dense.translations)
        assert config.engine_kwargs["sparse_pass2"]
        return SimpleNamespace(accumulators=[], meta={})

    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run)
    if selector == "1":

        def forbidden(*args, **kwargs):
            raise AssertionError("Sparse E-step built a full fine grid")

        monkeypatch.setattr(driver.sampling, "get_oversampled_relion_hidden_rotation_grid_from_samples", forbidden)
    dataset = SimpleNamespace(voxel_size=1.0, n_images=2)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=3, n_directions=3)
    state.iter = 3
    expectation = driver._native_expectation_step(
        dataset,
        opts,
        np.ones(33, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    args = (state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))
    if selector == "invalid":
        with pytest.raises(ValueError, match="must be 0 or 1"):
            expectation(*args)
        assert not calls
    else:
        _, meta = expectation(*args)
        assert len(calls) == 1
        assert meta["n_rotations"] == expected_count
        assert meta["n_translations"] == len(dense.translations)
