"""VDAM particle subset selection: random order, subset prefix, halfset id.

Mirrors the RELION 5.0 sequence:

1. RELION starts from its current `sorted_idx` particle list, which InitialModel
   initialises in stable micrograph-name order. If `random_seed == 0`, RELION
   leaves this order unchanged. Otherwise
   `Experiment::randomiseParticlesOrder(random_seed + iter, false, false)`
   shuffles that list using a Mersenne-Twister seeded with `random_seed + iter`.
   (RELION source: exp_model.cpp.)
2. Take the first `subset_size` particles of the shuffled list.
3. Stable-sort the selected prefix by optics-group id
   (ml_optimiser.cpp:4907) so particles from the same optics group are
   processed together.
4. For `grad_pseudo_halfsets=true`, route each particle to pseudo-halfset
   `part_id % 2`, matching the BPref accumulation offset in
   `storeWeightedSums` (ml_optimiser.cpp:10349).

RELION's non-halves shuffle uses `std::shuffle(sorted_idx, mt19937(seed))`.
The native parity path calls the C++ binding directly for byte-exact subset
selection. The pure-Python Fisher-Yates helper below remains as a deterministic
fallback and as a small unit-testable subset-selection primitive.
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
# Deterministic Fisher-Yates fallback
# ---------------------------------------------------------------------------


def randomise_particles_order(
    nr_particles: int,
    rnd_unif: RndUnifFn,
) -> np.ndarray:
    """Fallback shuffle used only when the RELION binding is unavailable.

    The native parity path uses `vdam_randomise_particles_order`, which calls
    RELION's `std::shuffle` path. This helper keeps standalone tests and
    binding-free environments deterministic but is not bit-exact to RELION.

    Returns the shuffled list of global particle indices as a NumPy int64
    array.
    """
    if nr_particles <= 0:
        return np.zeros(0, dtype=np.int64)

    order = np.arange(nr_particles, dtype=np.int64)
    # Fisher-Yates: for i from n-1 down to 1, swap order[i] with order[j].
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
    """Assign legacy alternating halfset ids for standalone callers.

    InitialModel reconstruction routing should use
    :func:`assign_pseudo_halfsets_for_particle_ids` instead.  RELION routes
    BPref pseudo-halfsets by global ``part_id % 2`` at accumulation time.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int8)
    return (np.arange(n, dtype=np.int64) % 2).astype(np.int8)


def assign_pseudo_halfsets_for_particle_ids(particle_ids: np.ndarray) -> np.ndarray:
    """Return RELION BPref pseudo-halfset ids for global particle indices."""

    ids = np.asarray(particle_ids, dtype=np.int64)
    return (ids % 2).astype(np.int8, copy=False)


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

    RELION source: ml_optimiser.cpp:4907 (stable-sort) and
    ml_optimiser.cpp:10349 (BPRef pseudo-halfset offset uses ``part_id % 2``).
    """
    if subset_size < 0 or subset_size > shuffled_particle_ids.size:
        raise ValueError(
            f"subset_size={subset_size} out of range for "
            f"nr_particles={shuffled_particle_ids.size}; caller must resolve -1 first"
        )

    prefix = shuffled_particle_ids[:subset_size]
    sorted_prefix = _stable_sort_by_optics_group(prefix, optics_group_by_particle)

    if pseudo_halfsets:
        halfsets = assign_pseudo_halfsets_for_particle_ids(sorted_prefix)
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
