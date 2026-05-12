"""VDAM particle subset selection: random shuffle, prefix, stable-sort, halfset id.

Mirrors RELION's ``randomiseParticlesOrder`` → first ``subset_size`` → stable-sort by
optics group (ml_optimiser.cpp:4907) → ``part_id % 2`` pseudo-halfset assignment
(:10349). The native parity path calls the C++ binding; the Python helpers below
are a deterministic fallback and unit-testable subset primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

RndUnifFn = Callable[[int], float]
"""``rnd_unif(call_idx)`` returning the next U(0,1). Tests use a NumPy PRNG;
production wraps RELION's ``init_random_generator(seed) + ran1()``."""


@dataclass(frozen=True)
class SubsetPlan:
    """Per-iteration plan from ``select_vdam_subset``: ``particle_ids`` (int64, shuffled+sorted) and ``halfset_ids`` (int8 0/1, or zeros)."""

    particle_ids: np.ndarray
    halfset_ids: np.ndarray


def randomise_particles_order(nr_particles: int, rnd_unif: RndUnifFn) -> np.ndarray:
    """Fisher-Yates fallback (not bit-exact to RELION; use ``vdam_randomise_particles_order`` for parity)."""
    if nr_particles <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.arange(nr_particles, dtype=np.int64)
    for call_idx, i in enumerate(range(nr_particles - 1, 0, -1)):
        j = max(0, min(i, int(rnd_unif(call_idx) * i + 0.5)))  # ROUND = floor(x+0.5)
        if j != i:
            order[i], order[j] = order[j], order[i]
    return order


def _stable_sort_by_optics_group(particle_ids: np.ndarray, optics_group_by_particle: Sequence[int]) -> np.ndarray:
    """Stable-sort by optics-group; matches C++ ``std::stable_sort`` on integer keys."""
    if particle_ids.size == 0:
        return particle_ids
    keys = np.asarray([optics_group_by_particle[int(p)] for p in particle_ids], dtype=np.int64)
    return particle_ids[np.argsort(keys, kind="stable")]


def pseudo_halfsets_active(gradient_refine: bool, do_split_random_halves: bool) -> bool:
    """ml_optimiser.cpp:1920 ``grad_pseudo_halfsets = do_grad && !do_split_random_halves`` (always True for GUI InitialModel)."""
    return gradient_refine and not do_split_random_halves


def assign_pseudo_halfsets(n: int) -> np.ndarray:
    """Alternating 0/1 halfset ids; production routing uses ``assign_pseudo_halfsets_for_particle_ids``."""
    return (np.arange(max(0, n), dtype=np.int64) % 2).astype(np.int8)


def assign_pseudo_halfsets_for_particle_ids(particle_ids: np.ndarray) -> np.ndarray:
    """RELION BPref pseudo-halfset ids: ``global part_id % 2``."""
    return (np.asarray(particle_ids, dtype=np.int64) % 2).astype(np.int8, copy=False)


def select_vdam_subset(
    shuffled_particle_ids: np.ndarray,
    subset_size: int,
    optics_group_by_particle: Sequence[int],
    pseudo_halfsets: bool,
) -> SubsetPlan:
    """Per-iteration plan: prefix-N of shuffle, stable-sort by optics group, BPref halfset assignment.

    Caller must resolve ``subset_size=-1`` to ``nr_particles`` first.
    """
    if subset_size < 0 or subset_size > shuffled_particle_ids.size:
        raise ValueError(
            f"subset_size={subset_size} out of range for nr_particles={shuffled_particle_ids.size}; resolve -1 first"
        )
    sorted_prefix = _stable_sort_by_optics_group(shuffled_particle_ids[:subset_size], optics_group_by_particle)
    halfsets = (
        assign_pseudo_halfsets_for_particle_ids(sorted_prefix)
        if pseudo_halfsets
        else np.zeros(sorted_prefix.size, dtype=np.int8)
    )
    return SubsetPlan(particle_ids=sorted_prefix, halfset_ids=halfsets)


def numpy_rnd_unif_factory(seed: int) -> RndUnifFn:
    """Deterministic NumPy-backed ``rnd_unif`` for tests (not bit-exact to RELION)."""
    rng = np.random.default_rng(seed)

    def _rnd(_call_idx: int) -> float:
        return float(rng.random())

    return _rnd
