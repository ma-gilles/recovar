"""Per-iteration particle subset selection for the InitialModel schedule.

RELION's SGD/VDAM subset draw (``MlOptimiser::iterate`` subset handling) and
the order restoration used by continuation runs. ``iteration_loop`` draws the
subset for every iteration through these owners.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from recovar.em.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    VdamPhaseLengths,
    compute_phase_lengths,
    compute_subset_size,
)
from recovar.em.vdam.state import InitialModelState
from recovar.em.vdam.subset import RndUnifFn, pseudo_halfsets_active, randomise_particles_order, select_vdam_subset


def _resolve_phase_lengths(
    nr_iter: int,
    grad_ini_frac: float,
    grad_fin_frac: float,
    phase_lengths: VdamPhaseLengths | None,
) -> VdamPhaseLengths:
    if phase_lengths is None:
        return compute_phase_lengths(nr_iter, grad_ini_frac, grad_fin_frac)
    if not isinstance(phase_lengths, VdamPhaseLengths):
        raise TypeError("phase_lengths must be a VdamPhaseLengths instance")
    values = (
        int(phase_lengths.grad_ini_iter),
        int(phase_lengths.grad_inbetween_iter),
        int(phase_lengths.grad_fin_iter),
    )
    if any(value < 0 for value in values) or sum(values) != int(nr_iter):
        raise ValueError("phase_lengths must be non-negative and sum to nr_iter")
    return phase_lengths


def select_subset_for_iter(
    state: InitialModelState,
    iter: int,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    rnd_unif_factory: Callable[[int], RndUnifFn],
    random_seed: int,
    do_grad: bool,
    particle_order: Sequence[int] | None = None,
) -> InitialModelState:
    """Select RELION's per-iteration VDAM subset.

    RELION skips particle-order randomization entirely when ``random_seed`` is
    zero. Otherwise it shuffles RELION's current ``sorted_idx`` particle list
    with seed ``random_seed + iter``, takes the first ``subset_size`` (or all
    particles if ``-1``), stable-sorts by optics group, and assigns pseudo-
    halfset ids.
    """
    stored_order = state.sorted_particle_ids
    stored_part_ids = state.sorted_particle_part_ids
    if stored_order is not None or stored_part_ids is not None:
        if stored_order is None or stored_part_ids is None:
            raise ValueError("stored RELION particle order is incomplete")
        base_order = np.asarray(stored_order, dtype=np.int64)
        base_halfset_ids = np.asarray(stored_part_ids, dtype=np.int64)
        if base_order.shape != (int(nr_particles),) or base_halfset_ids.shape != base_order.shape:
            raise ValueError(
                "stored RELION particle order must match nr_particles: "
                f"{base_order.shape}, {base_halfset_ids.shape} != ({int(nr_particles)},)",
            )
    elif particle_order is None:
        base_order = np.arange(int(nr_particles), dtype=np.int64)
        base_halfset_ids = np.arange(int(nr_particles), dtype=np.int64)
    else:
        base_order = np.asarray(particle_order, dtype=np.int64)
        if base_order.shape != (int(nr_particles),):
            raise ValueError(f"particle_order must have shape ({int(nr_particles)},), got {base_order.shape}")
        if (
            np.unique(base_order).size != int(nr_particles)
            or np.any(base_order < 0)
            or np.any(base_order >= nr_particles)
        ):
            raise ValueError("particle_order must be a permutation of particle ids [0, nr_particles)")
        # RELION's InitialModel pseudo-halfset routing uses Experiment's
        # internal ``part_id``, i.e. the position in its read-order particle
        # table, not RECOVAR's original input-table row:
        # ``iproj_offset = (part_id % 2) * nr_classes`` in storeWeightedSums.
        # ``particle_order`` maps those internal positions to RECOVAR dataset
        # rows, so parity must travel with the positions through shuffling.
        base_halfset_ids = np.arange(int(nr_particles), dtype=np.int64)

    subset_size = state.subset_size if state.subset_size != -1 else nr_particles
    doing_subset = 0 < int(subset_size) < int(nr_particles)
    first_randomisation = stored_order is None
    if int(random_seed) == 0 or (not first_randomisation and not doing_subset):
        shuffled = base_order.copy()
        shuffled_halfset_ids = base_halfset_ids.copy()
    else:
        # C++ binding does std::shuffle byte-exact vs RELION; Python is a fallback.
        try:
            from recovar.relion_bind import _relion_bind_core as _bind

            permutation = np.asarray(
                _bind.vdam_randomise_particles_order(int(nr_particles), int(random_seed + iter)), dtype=np.int64
            )
        except (ImportError, AttributeError):
            permutation = randomise_particles_order(nr_particles, rnd_unif_factory(random_seed + iter))
        shuffled = base_order[permutation]
        shuffled_halfset_ids = base_halfset_ids[permutation]

    # `-1` (all particles) still needs to be translated via select_vdam_subset
    pseudo = do_grad and pseudo_halfsets_active(gradient_refine=True, do_split_random_halves=False)
    plan = select_vdam_subset(
        shuffled_particle_ids=shuffled,
        subset_size=subset_size,
        optics_group_by_particle=optics_group_by_particle,
        pseudo_halfsets=pseudo,
        halfset_particle_ids=shuffled_halfset_ids,
    )
    # Persist the sorted prefix and untouched tail for the next iteration's shuffle.
    shuffled[:subset_size] = plan.particle_ids
    shuffled_halfset_ids[:subset_size] = plan.part_ids
    new_state = replace(state)
    new_state.subset_particle_ids = plan.particle_ids
    new_state.subset_halfset_ids = plan.halfset_ids
    new_state.sorted_particle_ids = shuffled
    new_state.sorted_particle_part_ids = shuffled_halfset_ids
    new_state.pseudo_halfsets = pseudo
    return new_state


def restore_subset_order_for_continuation(
    state: InitialModelState,
    *,
    through_iteration: int,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    random_seed: int,
    rnd_unif_factory: Callable[[int], RndUnifFn],
    particle_order: Sequence[int] | None = None,
    grad_ini_frac: float = 0.3,
    grad_fin_frac: float = 0.2,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
    phase_lengths: VdamPhaseLengths | None = None,
) -> InitialModelState:
    """Rebuild RELION's transient ``sorted_idx`` at a restart boundary.

    InitialModel mutates ``Experiment::sorted_idx`` after every deterministic
    shuffle and stable optics-group sort, but RELION does not serialize that
    vector in an optimiser checkpoint.  A bounded diagnostic continuation
    must therefore replay the inexpensive ordering chronology from iteration
    one; starting the next shuffle from the input order selects a different
    particle subset even when every scientific checkpoint array is exact.

    The replay is valid while convergence has not occurred because subset
    scheduling is then a pure function of the command and iteration.  The
    onset iteration of convergence is not serialized, so fail closed instead
    of guessing when either convergence flag is already set.
    """

    through_iteration = int(through_iteration)
    nr_particles = int(nr_particles)
    if through_iteration < 0 or through_iteration > int(state.nr_iter):
        raise ValueError("continuation iteration must be between 0 and nr_iter")
    if int(state.iter) != through_iteration:
        raise ValueError(
            "continuation state iteration does not match the requested order replay"
        )
    if nr_particles <= 0:
        raise ValueError("continuation particle count must be positive")
    if state.sorted_particle_ids is not None or state.sorted_particle_part_ids is not None:
        raise ValueError("continuation state already contains a serialized particle order")
    if through_iteration and (state.has_converged or state.grad_has_converged):
        raise ValueError(
            "cannot reconstruct particle order after an unrecorded convergence boundary"
        )

    phase_lengths = _resolve_phase_lengths(
        int(state.nr_iter),
        float(grad_ini_frac),
        float(grad_fin_frac),
        phase_lengths,
    )
    order_state = replace(
        state,
        subset_particle_ids=None,
        subset_halfset_ids=None,
        sorted_particle_ids=None,
        sorted_particle_part_ids=None,
    )
    for iteration in range(1, through_iteration + 1):
        subset_size = compute_subset_size(
            iter=iteration,
            phase_lengths=phase_lengths,
            grad_ini_subset_size=int(grad_ini_subset_size),
            grad_fin_subset_size=int(grad_fin_subset_size),
            nr_particles=nr_particles,
            nr_iter=int(state.nr_iter),
            grad_em_iters=int(grad_em_iters),
            has_converged=False,
            grad_has_converged=False,
            nr_classes=int(state.K),
        )
        order_state = replace(order_state, subset_size=int(subset_size))
        do_grad = (int(state.nr_iter) - iteration) >= int(grad_em_iters)
        order_state = select_subset_for_iter(
            order_state,
            iter=iteration,
            nr_particles=nr_particles,
            optics_group_by_particle=optics_group_by_particle,
            rnd_unif_factory=rnd_unif_factory,
            random_seed=int(random_seed),
            do_grad=do_grad,
            particle_order=particle_order,
        )

    if through_iteration and int(order_state.subset_size) != int(state.subset_size):
        raise ValueError(
            "replayed continuation subset size differs from the native checkpoint"
        )
    restored = replace(state)
    restored.subset_particle_ids = order_state.subset_particle_ids
    restored.subset_halfset_ids = order_state.subset_halfset_ids
    restored.sorted_particle_ids = order_state.sorted_particle_ids
    restored.sorted_particle_part_ids = order_state.sorted_particle_part_ids
    restored.pseudo_halfsets = order_state.pseudo_halfsets
    return restored
