"""Tests for `recovar.em.ppca_abinitio.metrics`.

Pins the score-stage primary metric (`true_state_mass`), top-1
accuracy, true-state rank, the bootstrap CI helper, and the
`score_diagnostic_one_seed` aggregator that builds the Stage 0B
exit-criterion result object.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.em.ppca_abinitio.metrics import (
    BootstrapCI,
    bootstrap_ci_mean,
    per_image_true_state_mass,
    per_image_true_state_rank,
    score_diagnostic_one_seed,
    top1_acc,
    true_state_mass,
    true_state_rank,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Hand-built log_resp tensors with known answers
# ---------------------------------------------------------------------------


def _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass):
    """Construct a `(n_img, n_rot, n_trans)` log_resp where each
    image has mass `true_mass` on `(true_r[i], true_t[i])` and the
    remainder spread uniformly over the other entries."""
    log_resp = np.full((n_img, n_rot, n_trans), -np.inf, dtype=np.float64)
    for i in range(n_img):
        n_other = n_rot * n_trans - 1
        other = (1.0 - true_mass) / n_other
        log_resp[i, :, :] = np.log(other)
        log_resp[i, true_r[i], true_t[i]] = np.log(true_mass)
    return log_resp


def test_per_image_true_state_mass_known_value():
    n_img, n_rot, n_trans = 5, 3, 2
    true_r = np.array([0, 1, 2, 0, 1])
    true_t = np.array([0, 1, 0, 1, 0])
    log_resp = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.7)

    masses = per_image_true_state_mass(log_resp, true_r, true_t)
    np.testing.assert_allclose(masses, 0.7 * np.ones(n_img), rtol=1e-12)


def test_true_state_mass_average():
    n_img, n_rot, n_trans = 4, 4, 1
    true_r = np.array([0, 1, 2, 3])
    true_t = np.zeros(4, dtype=np.int64)
    log_resp = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.4)
    assert true_state_mass(log_resp, true_r, true_t) == pytest.approx(0.4, rel=1e-12)


def test_top1_acc_when_truth_is_argmax():
    n_img, n_rot, n_trans = 6, 5, 3
    rng = np.random.default_rng(0)
    true_r = rng.integers(0, n_rot, size=n_img)
    true_t = rng.integers(0, n_trans, size=n_img)
    log_resp = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.6)
    assert top1_acc(log_resp, true_r, true_t) == pytest.approx(1.0)


def test_top1_acc_when_truth_is_below_max():
    n_img, n_rot, n_trans = 4, 3, 2
    rng = np.random.default_rng(1)
    true_r = rng.integers(0, n_rot, size=n_img)
    true_t = rng.integers(0, n_trans, size=n_img)
    # mass 0.1 on truth, the other 5 cells get 0.18 each — argmax is not the truth
    log_resp = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.1)
    assert top1_acc(log_resp, true_r, true_t) == pytest.approx(0.0)


def test_true_state_rank_argmax_means_rank_zero():
    n_img, n_rot, n_trans = 4, 3, 2
    rng = np.random.default_rng(2)
    true_r = rng.integers(0, n_rot, size=n_img)
    true_t = rng.integers(0, n_trans, size=n_img)
    log_resp = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.6)
    ranks = per_image_true_state_rank(log_resp, true_r, true_t)
    np.testing.assert_array_equal(ranks, np.zeros(n_img))


def test_true_state_rank_uniform_spreads_to_n_minus_one():
    """If all entries have equal mass, every entry has the same
    log_resp, so the count of strictly greater entries is 0."""
    n_img, n_rot, n_trans = 3, 4, 2
    log_resp = np.full((n_img, n_rot, n_trans), -np.log(8), dtype=np.float64)
    true_r = np.zeros(n_img, dtype=np.int64)
    true_t = np.zeros(n_img, dtype=np.int64)
    assert true_state_rank(log_resp, true_r, true_t) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_constant_data_has_zero_width():
    """Bootstrapping a constant array gives mean=constant and
    zero-width CI."""
    vals = np.full(50, 3.7)
    ci = bootstrap_ci_mean(vals, n_bootstrap=200, seed=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.mean == pytest.approx(3.7)
    assert ci.ci_low == pytest.approx(3.7)
    assert ci.ci_high == pytest.approx(3.7)
    assert ci.level == 0.95


def test_bootstrap_ci_normal_data_covers_true_mean():
    rng = np.random.default_rng(0)
    n = 200
    true_mean = 1.0
    vals = rng.standard_normal(n) + true_mean
    ci = bootstrap_ci_mean(vals, n_bootstrap=500, seed=42)
    # Standard error ≈ 1/sqrt(200) ≈ 0.07; CI width ≈ 4*SE ≈ 0.28
    assert ci.ci_low <= true_mean <= ci.ci_high
    assert (ci.ci_high - ci.ci_low) < 0.5


def test_bootstrap_ci_excludes_zero_when_mean_is_far_from_zero():
    """Pin the form of the Stage 0B exit criterion: a clearly
    positive mean should yield a CI that excludes zero."""
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(200) * 0.1 + 0.5  # tight around 0.5
    ci = bootstrap_ci_mean(vals, n_bootstrap=500, seed=0)
    assert ci.ci_low > 0
    assert ci.ci_high > 0


# ---------------------------------------------------------------------------
# Stage 0B aggregator
# ---------------------------------------------------------------------------


def test_score_diagnostic_aggregates_per_image_metrics():
    n_img, n_rot, n_trans = 20, 4, 2
    rng = np.random.default_rng(0)
    true_r = rng.integers(0, n_rot, size=n_img)
    true_t = rng.integers(0, n_trans, size=n_img)
    log_resp_h = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.3)
    log_resp_p = _make_known_log_resp(n_img, n_rot, n_trans, true_r, true_t, true_mass=0.6)
    val_idx = np.arange(10, 20)  # last 10 are val

    diag = score_diagnostic_one_seed(
        log_resp_h,
        log_resp_p,
        true_r,
        true_t,
        val_idx,
        family="B",
        seed=7,
        n_bootstrap=200,
    )
    assert diag.family == "B"
    assert diag.seed == 7
    assert diag.n_val == 10
    assert diag.homog_true_state_mass.mean == pytest.approx(0.3, rel=1e-12)
    assert diag.ppca_true_state_mass.mean == pytest.approx(0.6, rel=1e-12)
    assert diag.delta_true_state_mass.mean == pytest.approx(0.3, rel=1e-12)
    # Constant per-image values → CI width is zero
    assert diag.delta_true_state_mass.ci_low == pytest.approx(0.3, rel=1e-12)
    assert diag.delta_true_state_mass.ci_high == pytest.approx(0.3, rel=1e-12)
