"""VDAM particle subset selection: random order, subset prefix, halfset id.

Mirrors the RELION 5.0 sequence:

1. `Experiment::randomiseParticlesOrder(random_seed + iter, false, false)`
   shuffles the full particle list using a Mersenne-Twister seeded with
   `random_seed + iter`. (RELION source: exp_model.cpp.)
2. Take the first `subset_size` particles of the shuffled list.
3. Stable-sort the selected prefix by optics-group id
   (ml_optimiser.cpp:4907) so particles from the same optics group are
   processed together.
4. For `grad_pseudo_halfsets=true`, alternate halfset ids along the
   stable-sorted prefix (ml_optimiser.cpp:5139).

RELION's shuffle is not a standard `std::shuffle`; it uses its own
`init_random_generator` + `ran1` pair (`rnd_unif`). We reproduce the
sequence via an explicit Fisher-Yates over the RELION generator, exposed
separately via `relion_bind` in Phase 2 for parity testing. Phase 1 only
exposes the deterministic *algorithm*, not the generator; the generator
binding is wired in via `rnd_unif_sequence` argument so tests can feed
either NumPy randoms (for standalone tests) or a RELION-generator stream
(for Phase 2 parity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

RndUnifFn = Callable[[int], float]
"""Callable that returns the next `rnd_unif(0, 1)` draw.

In pure-Python tests this can be a NumPy PRNG wrapper; in Phase 2 it will
be a RELION binding that mirrors `init_random_generator(seed) + ran1()`.
The `int` argument is an opaque call counter useful for debugging but not
used by the algorithm itself.
"""


@dataclass(frozen=True)
class SubsetPlan:
    """The per-iteration subset plan produced by `select_vdam_subset`.

    `particle_ids` are the global particle indices after shuffle + stable-
    sort by optics group. `halfset_ids` is 0/1 for each particle when
    `pseudo_halfsets_active`, else all zeros.
    """

    particle_ids: np.ndarray  # int64, shape (subset_size,)
    halfset_ids: np.ndarray  # int8, shape (subset_size,)


# ---------------------------------------------------------------------------
# Fisher-Yates shuffle via a RELION-compatible rnd_unif source
# ---------------------------------------------------------------------------


def randomise_particles_order(
    nr_particles: int,
    rnd_unif: RndUnifFn,
) -> np.ndarray:
    """Reproduce `Experiment::randomiseParticlesOrder` (non-halves path).

    RELION's implementation is Fisher-Yates with the swap index drawn from
    `ROUND(rnd_unif() * (n-1))` for the last element `n` at each step
    (i.e. classic in-place shuffle, iterating from the end toward the
    start).

    Returns the shuffled list of global particle indices as a NumPy int64
    array.
    """
    if nr_particles <= 0:
        return np.zeros(0, dtype=np.int64)

    order = np.arange(nr_particles, dtype=np.int64)
    # Fisher-Yates: for i from n-1 down to 1, swap order[i] with order[j]
    # where j = ROUND(rnd_unif() * i). Matches RELION's MT-based shuffle.
    call_idx = 0
    for i in range(nr_particles - 1, 0, -1):
        u = rnd_unif(call_idx)
        call_idx += 1
        # ROUND(x) = floor(x + 0.5); RELION uses this for the shuffle index
        j = int(u * i + 0.5)
        if j < 0:
            j = 0
        elif j > i:
            j = i
        if j != i:
            tmp = order[i]
            order[i] = order[j]
            order[j] = tmp
    return order


# ---------------------------------------------------------------------------
# Subset prefix + stable-sort by optics group
# ---------------------------------------------------------------------------


def _stable_sort_by_optics_group(
    particle_ids: np.ndarray,
    optics_group_by_particle: Sequence[int],
) -> np.ndarray:
    """Stable-sort `particle_ids` by optics-group id.

    Python's `sorted(..., key=...)` is stable; NumPy's `np.argsort(kind='stable')`
    (i.e. mergesort) matches C++ `std::stable_sort` byte-for-byte on
    integer keys.
    """
    if particle_ids.size == 0:
        return particle_ids
    keys = np.asarray([optics_group_by_particle[int(p)] for p in particle_ids], dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    return particle_ids[order]


def pseudo_halfsets_active(gradient_refine: bool, do_split_random_halves: bool) -> bool:
    """Mirror ml_optimiser.cpp:1920 `grad_pseudo_halfsets = do_grad && !do_split_random_halves`.

    For the GUI InitialModel path `gradient_refine=True`,
    `do_split_random_halves=False`, so this is always True.
    """
    return gradient_refine and not do_split_random_halves


def assign_pseudo_halfsets(n: int) -> np.ndarray:
    """Assign halfset ids 0/1 along a stable-sorted prefix.

    RELION alternates ids across consecutive particles in the subset
    (ml_optimiser.cpp:5139). With a stable-sorted-by-optics-group prefix
    this means each optics group gets particles split roughly evenly
    between the two pseudo-halves.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int8)
    return (np.arange(n, dtype=np.int64) % 2).astype(np.int8)


def select_vdam_subset(
    shuffled_particle_ids: np.ndarray,
    subset_size: int,
    optics_group_by_particle: Sequence[int],
    pseudo_halfsets: bool,
) -> SubsetPlan:
    """Produce a per-iteration `SubsetPlan`.

    `shuffled_particle_ids` is the full-dataset shuffle (output of
    `randomise_particles_order`). `subset_size` is the already-resolved
    value from `compute_subset_size` (caller is responsible for translating
    `-1` into `nr_particles`).

    RELION source: ml_optimiser.cpp:4907 (stable-sort) + 5139 (halfset
    alternation).
    """
    if subset_size < 0 or subset_size > shuffled_particle_ids.size:
        raise ValueError(
            f"subset_size={subset_size} out of range for "
            f"nr_particles={shuffled_particle_ids.size}; caller must resolve -1 first"
        )

    prefix = shuffled_particle_ids[:subset_size]
    sorted_prefix = _stable_sort_by_optics_group(prefix, optics_group_by_particle)

    if pseudo_halfsets:
        halfsets = assign_pseudo_halfsets(sorted_prefix.size)
    else:
        halfsets = np.zeros(sorted_prefix.size, dtype=np.int8)

    return SubsetPlan(particle_ids=sorted_prefix, halfset_ids=halfsets)


# ---------------------------------------------------------------------------
# A NumPy-backed rnd_unif for tests
# ---------------------------------------------------------------------------


def numpy_rnd_unif_factory(seed: int) -> RndUnifFn:
    """Build a deterministic `rnd_unif` source backed by NumPy's PCG64.

    This is NOT bit-identical to RELION's MT19937 + ran1 pair and cannot
    be used for cross-implementation parity. It exists so Phase-1
    standalone tests can exercise the shuffle/subset logic before the
    Phase-2 RELION-generator binding lands.
    """
    rng = np.random.default_rng(seed)

    def _rnd(_call_idx: int) -> float:
        return float(rng.random())

    return _rnd
