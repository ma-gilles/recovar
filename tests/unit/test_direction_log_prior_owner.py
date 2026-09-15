"""RELION ``pdf_direction`` scoring rules have one owner used by both controller passes.

Rules mirrored from ``ml_optimiser.cpp``: the direction prior multiplies
orientation weights only in ``NOPRIOR`` (global) mode; local searches use the
explicit direction/psi priors; a sampling-order change resets the prior to an
even distribution; RELION keeps one prior per class and copies class 0 to
every class when seeding; each half scores with its own model.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from recovar.em.helpers import orientation_priors as op
from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)
N_PIX = N_ROT // rotation_grid_n_in_planes(ORDER)


def _priors(**overrides):
    kwargs = dict(
        use_local=False,
        scoring_healpix_order=ORDER,
        k_class_enabled=False,
        n_classes=1,
        class_direction_prior=None,
        class_direction_prior_order=None,
        global_direction_prior=None,
        global_direction_prior_order=None,
        sealed_sampling_state=None,
        dtype=np.float32,
        log=logging.getLogger(__name__),
        half_index=0,
    )
    kwargs.update(overrides)
    return op.relion_direction_log_priors_for_half(**kwargs)


def _prior(seed):
    values = np.random.default_rng(seed).uniform(0.1, 1.0, N_PIX)
    return (values / values.sum()).astype(np.float32)


def _none(result):
    assert result.rotation_log_prior is None and result.class_rotation_log_prior is None


def test_local_search_uses_no_direction_prior_even_when_one_is_learned():
    _none(_priors(use_local=True, global_direction_prior=_prior(0), global_direction_prior_order=ORDER))
    _none(
        _priors(
            use_local=True,
            k_class_enabled=True,
            n_classes=2,
            class_direction_prior=np.stack([_prior(1), _prior(2)]),
            class_direction_prior_order=ORDER,
        )
    )


def test_k1_prior_at_the_scoring_order_expands_onto_the_canonical_grid(caplog):
    prior = _prior(3)
    with caplog.at_level(logging.INFO, logger=__name__):
        result = _priors(global_direction_prior=prior, global_direction_prior_order=ORDER)
    expected = op.make_relion_direction_log_prior(prior, ORDER, dtype=np.float32)
    assert result.class_rotation_log_prior is None
    assert result.rotation_log_prior.dtype == np.float32
    assert result.rotation_log_prior.tobytes() == expected.tobytes()
    assert "Using learned global direction prior half-1" in caplog.text


@pytest.mark.parametrize("stale_order", [ORDER + 1, None])
def test_k1_prior_at_another_order_is_uniform(stale_order):
    _none(_priors(global_direction_prior=_prior(4), global_direction_prior_order=stale_order))
    _none(_priors())


def test_sealed_sampling_expands_onto_the_captured_direction_rows():
    prior = _prior(5)
    sealed = {"directions_ipix": np.asarray([7, 3, 11]), "psi_angles_deg": np.asarray([0.0, 180.0])}
    result = _priors(global_direction_prior=prior, global_direction_prior_order=ORDER, sealed_sampling_state=sealed)
    expected = op._sealed_direction_log_prior(prior, sealed, dtype=np.float32)
    assert result.rotation_log_prior.shape == (6,)
    assert result.rotation_log_prior.tobytes() == expected.tobytes()


def test_kclass_uses_per_class_priors_at_the_scoring_order(caplog):
    class_prior = np.stack([_prior(6), _prior(7)])
    with caplog.at_level(logging.INFO, logger=__name__):
        result = _priors(k_class_enabled=True, n_classes=2, class_direction_prior=class_prior, class_direction_prior_order=ORDER)
    assert result.rotation_log_prior is None
    expected = np.stack([op.make_relion_direction_log_prior(class_prior[c], ORDER, dtype=np.float32) for c in range(2)])
    assert result.class_rotation_log_prior.shape == (2, N_ROT)
    assert result.class_rotation_log_prior.tobytes() == expected.tobytes()
    assert "Using learned per-class global direction prior half-1: 2 classes" in caplog.text


@pytest.mark.parametrize("class_prior_order", [None, ORDER + 1])
def test_kclass_shared_prior_is_copied_to_every_class(class_prior_order, caplog):
    """RELION seeds K references by copying pdf_direction[0]; a stale class prior does not block it."""

    shared = _prior(8)
    class_prior = None if class_prior_order is None else np.stack([_prior(9), _prior(10)])
    with caplog.at_level(logging.INFO, logger=__name__):
        result = _priors(
            k_class_enabled=True,
            n_classes=2,
            class_direction_prior=class_prior,
            class_direction_prior_order=class_prior_order,
            global_direction_prior=shared,
            global_direction_prior_order=ORDER,
        )
    expected_row = op.make_relion_direction_log_prior(shared, ORDER, dtype=np.float32)
    assert result.rotation_log_prior is None
    assert result.class_rotation_log_prior.shape == (2, N_ROT)
    for row in result.class_rotation_log_prior:
        assert row.tobytes() == expected_row.tobytes()
    assert "Using shared global direction prior half-1: 2 classes" in caplog.text


def test_kclass_without_a_matching_prior_is_uniform():
    _none(_priors(k_class_enabled=True, n_classes=2))
    _none(
        _priors(
            k_class_enabled=True,
            n_classes=2,
            class_direction_prior=np.stack([_prior(11), _prior(12)]),
            class_direction_prior_order=ORDER + 1,
            global_direction_prior=_prior(13),
            global_direction_prior_order=ORDER + 1,
        )
    )


def test_half_index_only_labels_the_log(caplog):
    with caplog.at_level(logging.INFO, logger=__name__):
        _priors(global_direction_prior=_prior(14), global_direction_prior_order=ORDER, half_index=1)
    assert "half-2" in caplog.text
