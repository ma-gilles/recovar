"""The fused local score passes derive their reconstruction support through ``_support_from_local_probs``."""

from __future__ import annotations

import inspect

import pytest

from recovar.em.local import local_score_pass

pytestmark = pytest.mark.unit


def test_fused_abs2_on_demand_pass_uses_the_support_owner():
    source = inspect.getsource(local_score_pass._fused_score_normalize_support_abs2_on_demand_impl)
    assert source.count("_support_from_local_probs(") == 1
    assert "_compute_reconstruction_support_full_sort_jit(" not in source
    assert "compute_reconstruction_support_from_threshold(" not in source


def test_support_owner_is_the_only_inline_support_computation():
    module_source = inspect.getsource(local_score_pass)
    assert module_source.count("= _compute_reconstruction_support_full_sort_jit(") == 1
    assert module_source.count("= compute_reconstruction_support_from_threshold(") == 1
