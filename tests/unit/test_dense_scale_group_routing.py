"""Dense scoring routes RELION scale groups through the engine that accumulates them.

RELION's ``storeWeightedSums`` accumulates the group-scale ``XA``/``AA`` sums
and the norm-correction residuals in every expectation pass regardless of
oversampling; only RECOVAR's adaptive/sparse engine does, so scale groups
select it even at oversampling 0 for K=1 and for K-class scoring.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from recovar.em.dense_single_volume import half_scoring

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("oversampling", [0, 1, 2])
def test_scale_groups_always_select_the_adaptive_engine(oversampling):
    assert half_scoring._dense_uses_adaptive_engine(oversampling, np.zeros(4, dtype=np.int32)) is True


def test_without_scale_groups_only_positive_oversampling_selects_it():
    assert half_scoring._dense_uses_adaptive_engine(0, None) is False
    assert half_scoring._dense_uses_adaptive_engine(1, None) is True
    assert half_scoring._dense_uses_adaptive_engine(np.int64(0), None) is False


def _routed_block(source, start, end):
    block = source[source.index(start) :]
    return block[: block.index(end)]


def test_k1_dense_scorer_uses_the_routing_rule_and_no_longer_rejects_scale_groups():
    source = inspect.getsource(half_scoring._score_half_dense)
    assert "if _dense_uses_adaptive_engine(state.adaptive_oversampling, group_ids_k):" in source
    assert "does not accumulate group XA/AA statistics" not in source
    # Oversampling 0 with scale groups is RELION's single pass on the current grid.
    routed = _routed_block(source, "if _dense_uses_adaptive_engine(", "if relion_firstiter_cc_this_iter:")
    assert "firstiter_coarse_current_size = cs_for_engine" in routed
    assert "firstiter_fine_current_size = cs_for_engine" in routed


def test_k_class_dense_scorer_routes_scale_groups_at_oversampling_zero():
    source = inspect.getsource(half_scoring._score_half_dense)
    gate = "elif _dense_uses_adaptive_engine(state.adaptive_oversampling, group_ids_k):"
    assert source.count(gate) == 1
    routed = _routed_block(source, gate, "build_adaptive_pass2_grids(")
    assert "firstiter_coarse_current_size = cs_for_engine" in routed
    assert "firstiter_fine_current_size = cs_for_engine" in routed
    assert "elif firstiter_coarse_current_size is not None and int(state.adaptive_oversampling) > 0:" not in source


def test_k_class_positive_oversampling_never_drops_to_the_direct_engine():
    """RELION keeps two passes under adaptive oversampling even when coarse_size == current_size."""
    source = inspect.getsource(half_scoring._score_half_dense)
    gate = "elif _dense_uses_adaptive_engine(state.adaptive_oversampling, group_ids_k):"
    assert "firstiter_coarse_current_size is not None" not in _routed_block(source, "if k_class_enabled:", gate)
    assert "or firstiter_coarse_current_size is not None" not in source
    assert half_scoring._dense_uses_adaptive_engine(1, None) is True
