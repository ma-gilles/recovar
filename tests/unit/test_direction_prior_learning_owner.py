"""Learned direction-prior updates have one owner; history formats their snapshots.

``mean_helpers.update_learned_direction_priors`` mutates the caller-owned
per-half prior lists; ``RefinementHistory.record_direction_prior`` and
``record_rotation_posterior`` own the float64 snapshot copies.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from recovar.em.helpers.iteration_history import RefinementHistory
from recovar.em.helpers.orientation_priors import collapse_rotation_posterior_to_direction_prior
from recovar.em.refinement import mean_helpers
from recovar.em.sampling import rotation_grid_size

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)


def _lists():
    return [None, None], [None, None], [None, None], [None, None]


def _update(rotation_posterior_per_half, lists, **overrides):
    global_prior, global_order, class_prior, class_order = lists
    kwargs = dict(
        rotation_posterior_per_half=rotation_posterior_per_half,
        class_rotation_posterior_per_half=[None, None],
        global_direction_prior_per_half=global_prior,
        global_direction_prior_order_per_half=global_order,
        class_direction_prior_per_half=class_prior,
        class_direction_prior_order_per_half=class_order,
        k_class_enabled=False,
        n_classes=1,
        use_local=False,
        k1_direction_prior_order=ORDER,
        k1_direction_prior_size=N_ROT,
        current_healpix_order=ORDER,
        exhaustive_grid_size=N_ROT,
        n_effective_rotations=N_ROT,
        dtype=np.float32,
        log=logging.getLogger(__name__),
    )
    kwargs.update(overrides)
    return mean_helpers.update_learned_direction_priors(**kwargs)


def test_k1_learns_one_prior_per_half_at_the_scoring_order():
    rng = np.random.default_rng(0)
    posteriors = [rng.uniform(0.0, 1.0, N_ROT).astype(np.float32) for _ in range(2)]
    lists = _lists()
    assert _update(posteriors, lists) is None
    global_prior, global_order, class_prior, class_order = lists
    for k in range(2):
        expected = collapse_rotation_posterior_to_direction_prior(
            np.asarray(posteriors[k], dtype=np.float64), ORDER, dtype=np.float32
        )
        assert global_prior[k].dtype == np.float32
        assert global_prior[k].tobytes() == expected.tobytes()
        assert global_order[k] == ORDER
    assert class_prior == [None, None] and class_order == [None, None]


def test_k1_size_mismatch_leaves_priors_untouched():
    posteriors = [np.ones(N_ROT, dtype=np.float32), np.ones(N_ROT + 1, dtype=np.float32)]
    lists = _lists()
    _update(posteriors, lists)
    assert lists == ([None, None], [None, None], [None, None], [None, None])


def test_k1_skips_a_half_whose_prior_cannot_form_a_log_prior(monkeypatch, caplog):
    calls = []

    def failing_log_prior(direction_prior, healpix_order):
        calls.append(int(healpix_order))
        if len(calls) == 1:
            raise ValueError("bad support")

    monkeypatch.setattr(mean_helpers, "make_relion_direction_log_prior", failing_log_prior)
    posteriors = [np.ones(N_ROT, dtype=np.float32), np.ones(N_ROT, dtype=np.float32)]
    lists = _lists()
    with caplog.at_level(logging.WARNING, logger=__name__):
        _update(posteriors, lists)
    global_prior, global_order, _, _ = lists
    assert calls == [ORDER, ORDER]
    assert global_prior[0] is None and global_order[0] is None
    assert global_prior[1] is not None and global_order[1] == ORDER
    assert "Skipping K=1 direction prior update for half-1 at healpix_order=1: bad support" in caplog.text


def test_kclass_combines_halves_on_the_exhaustive_grid_with_independent_copies():
    rng = np.random.default_rng(1)
    n_classes = 2
    class_posteriors = [rng.uniform(0.0, 1.0, (n_classes, N_ROT)) for _ in range(2)]
    lists = _lists()
    _update(
        [np.ones(N_ROT), np.ones(N_ROT)],
        lists,
        class_rotation_posterior_per_half=class_posteriors,
        k_class_enabled=True,
        n_classes=n_classes,
    )
    global_prior, global_order, class_prior, class_order = lists
    expected = mean_helpers._combined_class_direction_prior_from_halves(
        class_posteriors, n_classes, ORDER, dtype=np.float32
    )
    assert global_prior == [None, None] and global_order == [None, None]
    for k in range(2):
        assert class_prior[k].shape == expected.shape and class_prior[k].dtype == np.float32
        assert class_prior[k].tobytes() == expected.tobytes()
        assert class_order[k] == ORDER
    assert class_prior[0] is not class_prior[1]
    assert not np.shares_memory(class_prior[0], class_prior[1])


@pytest.mark.parametrize(
    "override",
    [dict(use_local=True), dict(n_effective_rotations=N_ROT + 3), dict(class_rotation_posterior_per_half=[None, np.ones((2, N_ROT))])],
)
def test_kclass_update_requires_global_scoring_on_the_exhaustive_grid(override):
    lists = _lists()
    kwargs = dict(class_rotation_posterior_per_half=[np.ones((2, N_ROT)), np.ones((2, N_ROT))], k_class_enabled=True, n_classes=2)
    kwargs.update(override)
    _update([np.ones(N_ROT), np.ones(N_ROT)], lists, **kwargs)
    assert lists == ([None, None], [None, None], [None, None], [None, None])


def test_history_records_float64_copies_of_k1_priors_and_none_for_missing():
    prior = np.asarray([0.25, 0.75], dtype=np.float32)
    history = RefinementHistory()
    history.record_direction_prior([None, None], [prior, None], k_class_enabled=False)
    stored = history.direction_prior_trajectory_per_half[0]
    assert stored[1] is None
    assert stored[0].dtype == np.float64 and stored[0].tolist() == [0.25, 0.75]
    assert not np.shares_memory(stored[0], prior)


def test_history_records_class_zero_of_each_half_for_kclass():
    class_prior = np.asarray([[0.1, 0.9], [0.6, 0.4]], dtype=np.float64)
    history = RefinementHistory()
    history.record_direction_prior([class_prior, class_prior.copy()], [np.ones(2), np.ones(2)], k_class_enabled=True)
    stored = history.direction_prior_trajectory_per_half[0]
    assert [s.tolist() for s in stored] == [[0.1, 0.9], [0.1, 0.9]]
    assert all(s.dtype == np.float64 and not np.shares_memory(s, class_prior) for s in stored)


def test_history_records_float64_rotation_posterior_copies():
    posterior = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    history = RefinementHistory()
    history.record_rotation_posterior([posterior, None])
    stored = history.rotation_posterior_trajectory_per_half[0]
    assert stored[1] is None
    assert stored[0].dtype == np.float64 and stored[0].tolist() == [1.0, 2.0, 3.0]
    assert not np.shares_memory(stored[0], posterior)
