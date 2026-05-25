"""Phase-5 unit tests for the EM-compatible BnB pruning rules.

Covers:
- Margin pruning preserves the top candidate by upper score.
- Orientation/shift caps fire when survivor count exceeds the fraction and
  do NOT fire when the count is below it.
- Floor on min_orientations / min_shifts prevents over-pruning on noisy data.
- Omitted-mass upper bound is conservative (>= true posterior mass dropped).
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.options import BranchBoundOptions
from recovar.em.dense_single_volume.bnb.pruning import (
    apply_orientation_cap,
    apply_shift_cap,
    compute_omitted_mass_upper,
    prune_by_score_margin,
    prune_by_tail_mass_and_caps,
)


def _rand_scores(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32) * 2.0


def test_margin_pruning_keeps_top_candidate():
    """The best candidate per image must always survive margin pruning."""
    scores = _rand_scores(42, (3, 5, 4))
    sample_mask = np.ones((3, 5, 4), dtype=bool)
    kept = prune_by_score_margin(sample_mask, scores, tau=1.0)
    for i in range(3):
        flat = scores[i].reshape(-1)
        best = np.argmax(flat)
        assert kept[i].reshape(-1)[best], f"top candidate dropped for image {i}"


def test_margin_pruning_drops_far_candidates():
    """Candidates more than tau below the max must be dropped."""
    scores = np.zeros((1, 4, 3), dtype=np.float32)
    scores[0, 1, 1] = 10.0  # top
    scores[0, 0, 0] = 9.0   # within tau=2
    scores[0, 2, 2] = 5.0   # too far
    mask = np.ones((1, 4, 3), dtype=bool)
    kept = prune_by_score_margin(mask, scores, tau=2.0)
    assert kept[0, 1, 1]
    assert kept[0, 0, 0]
    assert not kept[0, 2, 2]


def test_orientation_cap_fires_when_needed():
    """Cap fires for an image with more than ``fraction*n_rot`` survivors."""
    scores = _rand_scores(1, (2, 10, 3))
    mask = np.ones((2, 10, 3), dtype=bool)
    new_mask, cap_applied = apply_orientation_cap(
        mask, scores, fraction=0.3, floor=0,
    )
    # ceil(0.3 * 10) = 3 rotations max.
    n_rot_kept = new_mask.any(axis=2).sum(axis=1)
    assert np.all(n_rot_kept == 3)
    assert np.all(cap_applied)


def test_orientation_cap_does_not_fire_when_below_threshold():
    """If only a few rotations are active, the cap should not fire."""
    scores = _rand_scores(2, (1, 10, 3))
    mask = np.zeros((1, 10, 3), dtype=bool)
    mask[0, :2, :] = True  # only 2 rotations active
    new_mask, cap_applied = apply_orientation_cap(
        mask, scores, fraction=0.5, floor=0,
    )
    # max allowed = ceil(0.5 * 10) = 5 >= 2 active, no cap firing.
    assert not cap_applied[0]
    np.testing.assert_array_equal(new_mask, mask)


def test_floor_protects_against_overpruning():
    """``min_orientations_per_image`` keeps rotations even when fraction=0."""
    scores = _rand_scores(3, (1, 10, 3))
    mask = np.ones((1, 10, 3), dtype=bool)
    new_mask, _ = apply_orientation_cap(mask, scores, fraction=0.0, floor=4)
    n_rot_kept = new_mask.any(axis=2).sum(axis=1)
    assert n_rot_kept[0] == 4, "floor must keep at least 4 rotations"


def test_omitted_mass_upper_zero_when_nothing_pruned():
    """If pre == post, omitted mass = 0."""
    scores = _rand_scores(4, (3, 4, 5))
    mask = np.ones((3, 4, 5), dtype=bool)
    rho = compute_omitted_mass_upper(mask, mask, scores)
    np.testing.assert_allclose(rho, 0.0, atol=1e-7)


def test_omitted_mass_upper_one_when_all_pruned():
    """If kept set is empty, omitted mass = 1.0 sentinel."""
    scores = _rand_scores(5, (2, 3, 3))
    pre = np.ones((2, 3, 3), dtype=bool)
    post = np.zeros((2, 3, 3), dtype=bool)
    rho = compute_omitted_mass_upper(pre, post, scores)
    np.testing.assert_allclose(rho, 1.0)


def test_omitted_mass_upper_conservative():
    """Pruning a low-score candidate yields a small mass; pruning the top
    yields a large mass."""
    scores = np.zeros((1, 3, 3), dtype=np.float32)
    scores[0, 0, 0] = 5.0   # top
    scores[0, 0, 1] = 4.5
    scores[0, 1, 0] = 0.0   # tail
    pre = np.zeros((1, 3, 3), dtype=bool)
    pre[0, 0, 0] = pre[0, 0, 1] = pre[0, 1, 0] = True
    # Prune just the tail; kept = top two.
    post = pre.copy()
    post[0, 1, 0] = False
    rho_tail = compute_omitted_mass_upper(pre, post, scores)
    # Prune the top; kept = the second and tail.
    post2 = pre.copy()
    post2[0, 0, 0] = False
    rho_top = compute_omitted_mass_upper(pre, post2, scores)
    assert rho_tail[0] < rho_top[0]
    # Sanity: rho_tail should be tiny, rho_top should be near 0.5+.
    assert rho_tail[0] < 0.05
    assert rho_top[0] > 0.5


def test_prune_with_strict_options_keeps_only_top():
    """Very small tail tol + very low caps -> single best candidate per image."""
    scores = _rand_scores(6, (2, 5, 4))
    mask = np.ones((2, 5, 4), dtype=bool)
    opts = BranchBoundOptions(
        posterior_tail_tol=1e-300,         # tau -> ~690, only exact ties
        max_orientation_fraction=0.0,      # cap to floor
        max_shift_fraction=0.0,
        min_orientations_per_image=1,
        min_shifts_per_image=1,
        min_joint_candidates_per_image=1,
        max_joint_candidates_per_image=1,
    )
    new_mask, rho, cap_applied = prune_by_tail_mass_and_caps(mask, scores, opts)
    # Exactly one (rot, shift) survives per image.
    n_kept = new_mask.reshape(2, -1).sum(axis=1)
    assert np.all(n_kept == 1)
    # And it must be the per-image argmax.
    for i in range(2):
        flat = scores[i].reshape(-1)
        best = np.argmax(flat)
        assert new_mask[i].reshape(-1)[best], f"image {i} kept the wrong candidate"


def test_prune_no_pruning_keeps_everything():
    """Loose options -> no pruning."""
    scores = _rand_scores(7, (2, 5, 4))
    mask = np.ones((2, 5, 4), dtype=bool)
    opts = BranchBoundOptions(
        posterior_tail_tol=1.0,              # tau = 0 — keep within 0 of best
        max_orientation_fraction=1.0,
        max_shift_fraction=1.0,
        min_orientations_per_image=5,
        min_shifts_per_image=4,
        min_joint_candidates_per_image=20,
        max_joint_candidates_per_image=20,
    )
    new_mask, _, cap_applied = prune_by_tail_mass_and_caps(mask, scores, opts)
    assert np.all(new_mask)
    assert not np.any(cap_applied)
