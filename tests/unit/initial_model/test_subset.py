"""Phase-1 subset/ordering tests (pure Python, no RELION binding).

The RELION-generator binding lands in Phase 2. Here we cover:

  - `randomise_particles_order` is a bijection (permutation) and reproduces
    Fisher-Yates semantics against a deterministic `rnd_unif` stream.
  - `select_vdam_subset` stable-sorts the prefix by optics group and emits
    RELION BPref pseudo-halfset ids of length `subset_size`.
  - `assign_pseudo_halfsets_for_particle_ids` produces `part_id % 2` ids.
  - `pseudo_halfsets_active` matches RELION's activation logic
    (ml_optimiser.cpp:1920).
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model.subset import (
    assign_pseudo_halfsets,
    assign_pseudo_halfsets_for_particle_ids,
    numpy_rnd_unif_factory,
    pseudo_halfsets_active,
    randomise_particles_order,
    select_vdam_subset,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fisher-Yates shuffle
# ---------------------------------------------------------------------------


class TestRandomiseOrder:
    def test_shuffle_is_permutation(self):
        rnd = numpy_rnd_unif_factory(seed=42)
        order = randomise_particles_order(500, rnd)
        assert order.shape == (500,)
        assert order.dtype == np.int64
        assert set(int(x) for x in order) == set(range(500))

    def test_shuffle_is_deterministic_given_same_rnd(self):
        order_a = randomise_particles_order(500, numpy_rnd_unif_factory(seed=42))
        order_b = randomise_particles_order(500, numpy_rnd_unif_factory(seed=42))
        np.testing.assert_array_equal(order_a, order_b)

    def test_different_seeds_yield_different_orders(self):
        a = randomise_particles_order(500, numpy_rnd_unif_factory(seed=42))
        b = randomise_particles_order(500, numpy_rnd_unif_factory(seed=43))
        assert not np.array_equal(a, b)

    def test_empty(self):
        rnd = numpy_rnd_unif_factory(seed=0)
        out = randomise_particles_order(0, rnd)
        assert out.shape == (0,)
        assert out.dtype == np.int64

    def test_size_one_is_identity(self):
        rnd = numpy_rnd_unif_factory(seed=0)
        out = randomise_particles_order(1, rnd)
        assert out.tolist() == [0]

    def test_manual_rnd_sequence_produces_known_shuffle(self):
        """Fisher-Yates from the tail: for i=4..1, j=int(u*i + 0.5) then swap.

        With u = 0 at every draw, j = 0 for all i. Walking from i=4 down to
        1 with j=0 each time:
          start: [0, 1, 2, 3, 4]
          i=4, j=0 -> [4, 1, 2, 3, 0]
          i=3, j=0 -> [3, 1, 2, 4, 0]
          i=2, j=0 -> [2, 1, 3, 4, 0]
          i=1, j=0 -> [1, 2, 3, 4, 0]
        """
        calls = {"n": 0}

        def rnd(_call_idx: int) -> float:
            calls["n"] += 1
            return 0.0

        order = randomise_particles_order(5, rnd)
        assert order.tolist() == [1, 2, 3, 4, 0]
        assert calls["n"] == 4  # n - 1 = 4 swaps

    def test_manual_rnd_sequence_u_equals_1_puts_last_first(self):
        """With u = 1 - eps at every draw, j = i at each step -> no swaps,
        except rounding 1*i gets clamped to i so the algorithm short-circuits.

        RELION uses `int(u*i + 0.5)` with clamping; u=1.0 -> j=i+0 (clamped) ->
        swap with self, no-op.
        """
        order = randomise_particles_order(5, lambda _i: 1.0)
        assert order.tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Subset prefix + stable-sort + halfset alternation
# ---------------------------------------------------------------------------


class TestSelectVdamSubset:
    def test_prefix_length(self):
        shuffled = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
        # All same optics group
        og = [0] * 10
        plan = select_vdam_subset(shuffled, subset_size=4, optics_group_by_particle=og, pseudo_halfsets=True)
        assert plan.particle_ids.shape == (4,)
        assert plan.halfset_ids.shape == (4,)

    def test_stable_sort_by_optics_group(self):
        # Shuffled prefix picks particles from two optics groups
        # Particle 0, 2, 4 are in group 0; particle 1, 3, 5 in group 1
        shuffled = np.array([5, 0, 3, 4, 1, 2], dtype=np.int64)
        og = [0, 1, 0, 1, 0, 1]  # particle i -> optics group
        plan = select_vdam_subset(shuffled, subset_size=6, optics_group_by_particle=og, pseudo_halfsets=True)
        # After stable-sort by optics group, group 0 particles come first in
        # their original (shuffled) relative order, then group 1:
        #   group 0 (in shuffled order): 0, 4, 2  -> [0, 4, 2]
        #   group 1 (in shuffled order): 5, 3, 1  -> [5, 3, 1]
        assert plan.particle_ids.tolist() == [0, 4, 2, 5, 3, 1]

    def test_halfsets_follow_global_particle_id_parity(self):
        shuffled = np.array([5, 0, 3, 4, 1, 2], dtype=np.int64)
        og = [0] * 6
        plan = select_vdam_subset(shuffled, subset_size=6, optics_group_by_particle=og, pseudo_halfsets=True)
        assert plan.particle_ids.tolist() == [5, 0, 3, 4, 1, 2]
        assert plan.halfset_ids.tolist() == [1, 0, 1, 0, 1, 0]

    def test_halfsets_can_follow_relion_internal_ids_after_external_sort(self):
        shuffled = np.array([5, 0, 3, 4, 1, 2], dtype=np.int64)
        internal_ids = np.arange(6, dtype=np.int64)
        og = [0, 1, 0, 1, 0, 1]
        plan = select_vdam_subset(
            shuffled,
            subset_size=6,
            optics_group_by_particle=og,
            pseudo_halfsets=True,
            halfset_particle_ids=internal_ids,
        )
        assert plan.particle_ids.tolist() == [0, 4, 2, 5, 3, 1]
        assert plan.halfset_ids.tolist() == [1, 1, 1, 0, 0, 0]

    def test_halfsets_all_zero_when_not_pseudo(self):
        shuffled = np.array([0, 1, 2, 3], dtype=np.int64)
        og = [0] * 4
        plan = select_vdam_subset(shuffled, subset_size=4, optics_group_by_particle=og, pseudo_halfsets=False)
        assert plan.halfset_ids.tolist() == [0, 0, 0, 0]

    def test_subset_size_out_of_range(self):
        shuffled = np.arange(10, dtype=np.int64)
        og = [0] * 10
        with pytest.raises(ValueError):
            select_vdam_subset(shuffled, subset_size=-1, optics_group_by_particle=og, pseudo_halfsets=True)
        with pytest.raises(ValueError):
            select_vdam_subset(shuffled, subset_size=11, optics_group_by_particle=og, pseudo_halfsets=True)

    def test_full_pipeline_on_fixture_size_500(self):
        """Integration-ish: shuffle 500 particles, take a 200-prefix, sort,
        halfset. All invariants hold.
        """
        rnd = numpy_rnd_unif_factory(seed=123)
        order = randomise_particles_order(500, rnd)
        # Mock optics groups: first 300 particles in group 0, rest in group 1
        og = [0 if i < 300 else 1 for i in range(500)]
        plan = select_vdam_subset(order, subset_size=200, optics_group_by_particle=og, pseudo_halfsets=True)

        assert plan.particle_ids.shape == (200,)
        assert plan.halfset_ids.shape == (200,)

        # Every particle id is valid and unique
        assert plan.particle_ids.min() >= 0
        assert plan.particle_ids.max() < 500
        assert np.unique(plan.particle_ids).size == 200

        # Stable-sort by optics group: all group-0 particles come before
        # any group-1 particle
        og_keys = np.array([og[int(p)] for p in plan.particle_ids])
        boundaries = np.where(np.diff(og_keys) != 0)[0]
        assert boundaries.size <= 1, "more than one group-boundary - stable sort broken"

        # RELION's BPref pseudo-halfsets route by global particle id parity.
        np.testing.assert_array_equal(plan.halfset_ids, plan.particle_ids % 2)


# ---------------------------------------------------------------------------
# Pseudo-halfset activation
# ---------------------------------------------------------------------------


class TestPseudoHalfsetsActive:
    def test_gui_initial_model_path(self):
        # ml_optimiser.cpp:1920 with --grad, no --split_random_halves
        assert pseudo_halfsets_active(gradient_refine=True, do_split_random_halves=False) is True

    def test_auto_refine_path(self):
        # ml_optimiser.cpp:1920 with --auto_refine (which adds --split_random_halves)
        # grad off, real halves on
        assert pseudo_halfsets_active(gradient_refine=False, do_split_random_halves=True) is False

    def test_em_no_halves(self):
        assert pseudo_halfsets_active(gradient_refine=False, do_split_random_halves=False) is False

    def test_grad_with_split_halves_is_not_pseudo(self):
        # Odd combination but RELION's logic still says no
        assert pseudo_halfsets_active(gradient_refine=True, do_split_random_halves=True) is False


class TestAssignPseudoHalfsets:
    def test_even_length(self):
        assert assign_pseudo_halfsets(4).tolist() == [0, 1, 0, 1]

    def test_odd_length(self):
        assert assign_pseudo_halfsets(5).tolist() == [0, 1, 0, 1, 0]

    def test_zero(self):
        assert assign_pseudo_halfsets(0).tolist() == []

    def test_dtype(self):
        out = assign_pseudo_halfsets(3)
        assert out.dtype == np.int8


class TestAssignPseudoHalfsetsForParticleIds:
    def test_global_particle_id_parity(self):
        ids = np.asarray([5, 0, 3, 4, 1, 2], dtype=np.int64)
        assert assign_pseudo_halfsets_for_particle_ids(ids).tolist() == [1, 0, 1, 0, 1, 0]

    def test_empty(self):
        out = assign_pseudo_halfsets_for_particle_ids(np.asarray([], dtype=np.int64))
        assert out.tolist() == []
        assert out.dtype == np.int8
