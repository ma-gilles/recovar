"""Snapshot direction-prior initialization has one owner in ``orientation_priors``."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import orientation_priors as op

pytestmark = pytest.mark.unit

N_PIX = 768  # HEALPix order 3


def _init(prior, **overrides):
    kwargs = dict(k_class_enabled=False, n_classes=1, dtype=np.float32, log=logging.getLogger(__name__))
    kwargs.update(overrides)
    return op.initial_direction_priors_from_snapshot(prior, **kwargs)


def test_no_snapshot_prior_leaves_every_half_unset():
    assert _init(None) == ([None, None], [None, None], [None, None], [None, None])
    assert _init(None, k_class_enabled=True, n_classes=3) == ([None, None], [None, None], [None, None], [None, None])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_k1_vector_is_shared_by_both_halves_with_inferred_order(dtype, caplog):
    prior = np.linspace(0.0, 1.0, N_PIX)
    with caplog.at_level(logging.INFO, logger=__name__):
        global_prior, global_order, class_prior, class_order = _init(prior, dtype=dtype)
    assert class_prior == [None, None] and class_order == [None, None]
    assert global_order == [3, 3]
    for half in global_prior:
        assert half.dtype == dtype and half.shape == (N_PIX,)
        assert half.tolist() == prior.astype(dtype).tolist()
    assert global_prior[0] is not global_prior[1]
    assert "loaded init direction prior half-1: 768 directions" in caplog.text
    assert "1 zero-probability" in caplog.text


def test_k1_per_half_list_keeps_a_missing_half_unset():
    prior = np.linspace(0.1, 1.0, N_PIX)
    global_prior, global_order, _, _ = _init([None, prior])
    assert global_prior[0] is None and global_order[0] is None
    assert global_prior[1].shape == (N_PIX,) and global_order[1] == 3


def test_kclass_snapshot_yields_per_class_priors_per_half(caplog):
    shared = np.stack([np.linspace(0.0, 1.0, N_PIX), np.linspace(1.0, 2.0, N_PIX)])
    with caplog.at_level(logging.INFO, logger=__name__):
        global_prior, global_order, class_prior, class_order = _init(shared, k_class_enabled=True, n_classes=2)
    assert global_prior == [None, None] and global_order == [None, None]
    assert class_order == [3, 3]
    for half in class_prior:
        assert half.dtype == np.float32 and half.shape == (2, N_PIX)
    assert "loaded init class direction priors half-1: 2 classes, 768 directions" in caplog.text


def test_kclass_per_half_snapshot_keeps_halves_distinct():
    per_half = np.stack([np.stack([np.linspace(0.0, 1.0, N_PIX)] * 2), np.stack([np.linspace(2.0, 3.0, N_PIX)] * 2)])
    _, _, class_prior, class_order = _init(per_half, k_class_enabled=True, n_classes=2)
    assert class_order == [3, 3]
    expected = op.normalize_class_direction_prior_per_half(per_half, 2, dtype=np.float32)
    for half, want in zip(class_prior, expected):
        assert half.dtype == np.float32 and half.tobytes() == np.asarray(want, dtype=np.float32).tobytes()
    assert class_prior[0].tobytes() != class_prior[1].tobytes()


def test_unrecognized_prior_length_is_rejected():
    with pytest.raises(ValueError, match="Cannot infer healpix order"):
        _init(np.ones(100))
