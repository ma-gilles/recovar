"""The class-segmented route must publish the engine's authoritative joint Pmax.

Measured motivation (job 14028280, original image 2468, iteration 1): the segmented
engine produced joint Pmax 0.8218123316764832 in its own score coordinates, while
``_assemble_result`` rebuilt it as ``max(best) - logsumexp(class_log_evidence)`` from
operands already quantized to the float32 grid at absolute magnitude ~16455, where the
ULP is 1.953125e-3 instead of 1.907e-6, and published 0.8230574131011963.

These tests pin the repair and the three behaviours it must not disturb.
"""
from __future__ import annotations

import numpy as np
import pytest

from recovar.em.classification import k_class_results
from recovar.em.classification.k_class_results import _assemble_result
from recovar.em.helpers.types import make_relion_stats

# Real operands captured at row 20 of group 1, iteration 1, job 14028280.
LARGE_OFFSET_CLASS_LOG_EVIDENCE = np.array(
    [[-16462.1015625], [-np.inf], [-16459.5390625], [-16455.275390625]], dtype=np.float64
)
LARGE_OFFSET_CLASS_BEST = np.array(
    [-16462.64, -np.inf, -16460.092, -16455.455078125], dtype=np.float32
)
ENGINE_JOINT_PMAX = np.float32(0.8218123316764832)
REBUILT_JOINT_PMAX = 0.8230574131011963


def _stats(best, *, log_evidence=-16455.26, max_posterior=0.0, n=1):
    return make_relion_stats(
        log_evidence_per_image=np.full(n, log_evidence, dtype=np.float32),
        best_log_score_per_image=np.asarray(best, dtype=np.float32).reshape(n),
        max_posterior_per_image=np.full(n, max_posterior, dtype=np.float32),
        rotation_posterior_sums=np.zeros((n, 1), dtype=np.float64),
    )


def _assemble(**overrides):
    kwargs = dict(
        class_log_evidence=LARGE_OFFSET_CLASS_LOG_EVIDENCE,
        new_means=None,
        Ft_y=[np.zeros(1, dtype=np.complex64) for _ in range(4)],
        Ft_ctf=[np.zeros(1, dtype=np.float32) for _ in range(4)],
        per_class_hard_assignments=np.zeros((4, 1), dtype=np.int64),
        per_class_stats=tuple(_stats(b) for b in LARGE_OFFSET_CLASS_BEST),
        noise_stats=None,
    )
    kwargs.update(overrides)
    return _assemble_result(**kwargs)


def test_large_offset_rebuild_is_the_defect_this_repair_removes():
    """Without the engine value the rebuild lands ~1.2e-3 away, as measured."""
    rebuilt = float(_assemble().stats.max_posterior_per_image[0])
    assert rebuilt == pytest.approx(REBUILT_JOINT_PMAX, abs=1e-12)
    assert abs(rebuilt - float(ENGINE_JOINT_PMAX)) > 1e-3


def test_segmented_route_publishes_the_engine_joint_pmax():
    """The repair: the supplied engine value is carried through unchanged."""
    published = _assemble(
        joint_max_posterior_per_image=np.array([ENGINE_JOINT_PMAX], dtype=np.float32),
    ).stats.max_posterior_per_image
    assert float(published[0]) == pytest.approx(float(ENGINE_JOINT_PMAX), abs=0.0)
    assert float(published[0]) != pytest.approx(REBUILT_JOINT_PMAX, abs=1e-9)


def test_best_and_log_evidence_are_not_touched_by_the_repair():
    """Only Pmax changes; the other published fields keep the generic arithmetic."""
    without = _assemble().stats
    with_engine = _assemble(
        joint_max_posterior_per_image=np.array([ENGINE_JOINT_PMAX], dtype=np.float32),
    ).stats
    assert np.array_equal(
        np.asarray(without.best_log_score_per_image), np.asarray(with_engine.best_log_score_per_image)
    )
    assert np.array_equal(
        np.asarray(without.log_evidence_per_image), np.asarray(with_engine.log_evidence_per_image)
    )


def test_responsibilities_are_not_touched_by_the_repair():
    without = _assemble()
    with_engine = _assemble(
        joint_max_posterior_per_image=np.array([ENGINE_JOINT_PMAX], dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray(without.class_responsibilities), np.asarray(with_engine.class_responsibilities)
    )


def test_independent_class_route_still_rebuilds():
    """The per-class route supplies no engine joint value and must keep the rebuild.

    Its classes are scored in separate calls, so no single call owns the joint
    normalization and there is no authoritative joint posterior to carry.
    """
    assert float(_assemble().stats.max_posterior_per_image[0]) == pytest.approx(
        REBUILT_JOINT_PMAX, abs=1e-12
    )


def test_firstiter_winner_take_all_still_wins_over_the_engine_value():
    published = _assemble(
        joint_max_posterior_per_image=np.array([ENGINE_JOINT_PMAX], dtype=np.float32),
        firstiter_winner_take_all=True,
    ).stats.max_posterior_per_image
    assert float(published[0]) == 1.0


def test_k1_branch_is_unchanged():
    """K=1 keeps taking its own branch even if an engine value is supplied."""
    single = _stats(LARGE_OFFSET_CLASS_BEST[3], max_posterior=0.5)
    result = _assemble_result(
        class_log_evidence=LARGE_OFFSET_CLASS_LOG_EVIDENCE[3:4],
        new_means=None,
        Ft_y=[np.zeros(1, dtype=np.complex64)],
        Ft_ctf=[np.zeros(1, dtype=np.float32)],
        per_class_hard_assignments=np.zeros((1, 1), dtype=np.int64),
        per_class_stats=(single,),
        noise_stats=None,
        joint_max_posterior_per_image=np.array([ENGINE_JOINT_PMAX], dtype=np.float32),
    )
    assert float(result.stats.max_posterior_per_image[0]) == pytest.approx(0.5, abs=0.0)


def test_branch_order_is_firstiter_then_k1_then_segmented_then_generic():
    """Guard the ordering the three tests above depend on."""
    import inspect

    src = inspect.getsource(k_class_results._assemble_result)
    order = [
        src.index("if firstiter_winner_take_all:"),
        src.index("elif len(per_class_stats) == 1:"),
        src.index("elif joint_max_posterior_per_image is not None:"),
    ]
    assert order == sorted(order)
