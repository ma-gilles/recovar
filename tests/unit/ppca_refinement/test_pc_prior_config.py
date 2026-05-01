"""Phase 1 (M0) test: PCPriorConfig defaults, immutability, serialization.

Defaults match §7 of recovar/em/ppca_refinement/CLAUDE.md and the project
plan in docs/math/ppca_refine_plan_2026_05_01.md. If anyone changes these
defaults without updating the docs, this test catches it.
"""

from __future__ import annotations

import dataclasses

import pytest

from recovar.ppca import PCPriorConfig

pytestmark = pytest.mark.unit


def test_pc_prior_config_defaults_match_documented_contract():
    cfg = PCPriorConfig()
    assert cfg.latent_prior_mode == "identity"
    assert cfg.pc_prior_mode == "hybrid_shell"
    assert cfg.prior_scale == 1.0
    assert cfg.variance_floor == 1e-8
    assert cfg.use_q_total_for_division is True
    assert cfg.smooth_shell_prior is True
    assert cfg.prior_freeze_iters == 3
    assert cfg.recompute_once_after_iter == 5
    assert cfg.allow_every_iter_prior_update is False


def test_pc_prior_config_to_dict_is_json_serializable():
    import json

    cfg = PCPriorConfig()
    d = cfg.to_dict()
    # Round-trip via JSON to confirm primitive values only.
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    assert d == d2
    # Must contain all dataclass fields.
    expected_keys = {f.name for f in dataclasses.fields(PCPriorConfig)}
    assert set(d.keys()) == expected_keys


def test_pc_prior_config_is_frozen():
    cfg = PCPriorConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.prior_scale = 2.0  # type: ignore[misc]


def test_pc_prior_config_recompute_can_be_disabled():
    cfg = PCPriorConfig(recompute_once_after_iter=None)
    assert cfg.recompute_once_after_iter is None
    assert cfg.to_dict()["recompute_once_after_iter"] is None


def test_em_accepts_pc_prior_config_kwarg_without_error():
    """EM must accept ``pc_prior_config=None`` (default) and an explicit
    ``PCPriorConfig`` instance without raising at the signature level. Real
    EM behavior with the kwarg is exercised by larger integration tests in
    later milestones.
    """
    import inspect

    from recovar.ppca import EM

    sig = inspect.signature(EM)
    assert "pc_prior_config" in sig.parameters
    assert sig.parameters["pc_prior_config"].default is None
