"""Unit tests for the CryoBench PPCA diagnostic script."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")

from scripts.ppca_abinitio.run_cryobench import build_anneal_schedule, resolve_n_burnin

pytestmark = [pytest.mark.unit]


def test_resolve_n_burnin_defaults_by_external_mode():
    assert resolve_n_burnin("discrete_volumes", None) == 0
    assert resolve_n_burnin("gaussian_pc", None) == 10


def test_resolve_n_burnin_respects_explicit_override():
    assert resolve_n_burnin("discrete_volumes", 3) == 3
    assert resolve_n_burnin("gaussian_pc", 0) == 0


def test_resolve_n_burnin_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        resolve_n_burnin("discrete_volumes", -1)


def test_anneal_schedule_none_is_flat():
    s = build_anneal_schedule("none", anneal_iters=30, total_iters=30)
    assert s == [1.0] * 30


def test_anneal_schedule_log1000_starts_at_1000_ends_at_1():
    s = build_anneal_schedule("log1000", anneal_iters=30, total_iters=30)
    assert len(s) == 30
    assert math.isclose(s[0], 1000.0, rel_tol=1e-9)
    assert math.isclose(s[-1], 1.0, rel_tol=1e-9)
    # Strictly monotonically decreasing
    assert all(s[i] > s[i + 1] for i in range(len(s) - 1))


def test_anneal_schedule_log100_starts_at_100_ends_at_1():
    s = build_anneal_schedule("log100", anneal_iters=10, total_iters=10)
    assert math.isclose(s[0], 100.0, rel_tol=1e-9)
    assert math.isclose(s[-1], 1.0, rel_tol=1e-9)


def test_anneal_schedule_linear50_starts_at_50_ends_at_1():
    s = build_anneal_schedule("linear50", anneal_iters=5, total_iters=5)
    assert math.isclose(s[0], 50.0, rel_tol=1e-9)
    assert math.isclose(s[-1], 1.0, rel_tol=1e-9)


def test_anneal_schedule_tail_pads_with_ones():
    s = build_anneal_schedule("log1000", anneal_iters=10, total_iters=30)
    assert len(s) == 30
    # First 10 are the annealing ramp, last 20 are all 1.0
    assert s[9] == 1.0
    assert s[10:] == [1.0] * 20
    assert s[0] > s[9]


def test_anneal_schedule_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown anneal kind"):
        build_anneal_schedule("cosine", anneal_iters=30, total_iters=30)
