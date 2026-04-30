"""InitialModel dense K-class E-step adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.dense_adapter import (
    DenseInitialModelEstepConfig,
    class_log_priors_from_state,
    run_dense_initial_model_estep,
    split_pseudo_halfset_particle_ids,
)

pytestmark = pytest.mark.unit


class _Dataset:
    n_images = 4


def _fake_result(n_classes: int, n: int):
    return SimpleNamespace(
        Ft_y=[np.full(n**3, k + 1, dtype=np.complex64) for k in range(n_classes)],
        Ft_ctf=[np.full(n**3, (k + 1) * 2, dtype=np.float32) for k in range(n_classes)],
        class_posterior_sums=np.arange(n_classes, dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        stats=SimpleNamespace(max_posterior_per_image=np.asarray([0.25, 0.75], dtype=np.float32)),
    )


def test_split_pseudo_halfset_particle_ids_sorts_by_micrograph_name():
    h0, h1 = split_pseudo_halfset_particle_ids(
        5,
        micrograph_names=np.asarray(["b", "a", "b", "a", "c"]),
    )
    np.testing.assert_array_equal(h0, np.asarray([1, 0, 4]))
    np.testing.assert_array_equal(h1, np.asarray([3, 2]))


def test_class_log_priors_from_state_normalizes_weights():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.pdf_class = np.asarray([2.0, 1.0])
    np.testing.assert_allclose(class_log_priors_from_state(state), np.log([2.0 / 3.0, 1.0 / 3.0]))


def test_dense_initial_model_estep_calls_joint_k_class_once_per_halfset(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(dataset, means, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs):
        calls.append(
            {
                "means_shape": np.asarray(means).shape,
                "image_indices": np.asarray(kwargs["image_indices"]).copy(),
                "class_log_priors": np.asarray(kwargs["class_log_priors"]).copy(),
                "current_size": kwargs["current_size"],
            }
        )
        return _fake_result(n_classes=2, n=8)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.current_size = 8
    state.pdf_class = np.asarray([0.75, 0.25])
    config = DenseInitialModelEstepConfig(
        means=np.zeros((2, 8**3), dtype=np.complex64),
        mean_variance=np.ones((2, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert len(calls) == 2
    assert calls[0]["means_shape"] == (2, 8**3)
    np.testing.assert_array_equal(calls[0]["image_indices"], [0, 2])
    np.testing.assert_array_equal(calls[1]["image_indices"], [1, 3])
    np.testing.assert_allclose(calls[0]["class_log_priors"], np.log([0.75, 0.25]))
    assert calls[0]["current_size"] == 8

    assert len(result.accumulators) == 4
    assert [(a.halfset_idx, a.class_idx) for a in result.accumulators] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for accum in result.accumulators:
        assert accum.data.shape == (8, 8, 5)
        assert accum.weight.shape == (8, 8, 5)
    np.testing.assert_allclose(result.accumulators[0].data, 1.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 4.0)
    assert result.meta["halfset_ids"] == (0, 1)


def test_dense_initial_model_estep_handles_empty_halfset(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(*args, **kwargs):
        calls.append(kwargs["image_indices"])
        return _fake_result(n_classes=1, n=8)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 2]),
        halfset_ids=np.asarray([0, 0], dtype=np.int8),
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, 2])
    assert len(result.accumulators) == 2
    np.testing.assert_allclose(result.accumulators[1].data, 0.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 0.0)


def test_dense_initial_model_estep_rejects_sparse_pass2():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        engine_kwargs={"sparse_pass2": True},
    )

    with pytest.raises(NotImplementedError, match="sparse_pass2 must be disabled"):
        run_dense_initial_model_estep(_Dataset(), state, config)
