"""VDAM iteration loop `run_vdam_iterations`.

Mirrors `MlOptimiser::iterate` (ml_optimiser.cpp:3458-3550) for the
gradient-refine branch:

  for iter in 1 .. nr_iter:
      schedule_update(state, iter)        # stepsize, tau2_fudge, subset
      do_grad = ...                       # drop grad at the EM tail
      pseudo_halfsets = do_grad
      select_subset_for_iter(state, ...)  # shuffle + prefix + stable-sort
      update_current_resolution(state)    # FSC-driven from iter 2
      expectation_step(state, ...)        # E-step adapter -> posteriors
      maximisation_step(state, ...)       # VDAM M-step
      write_iter_artifacts(state, iter)

The E-step adapter (`expectation_step`) is a callback supplied by the
caller because it requires dense-path kernels + real particle data. This
module is the pure orchestrator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from .m_step import vdam_m_step
from .schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_MU,
    VdamPhaseLengths,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)
from .state import InitialModelState
from .subset import (
    RndUnifFn,
    pseudo_halfsets_active,
    randomise_particles_order,
    select_vdam_subset,
)

# Callback signatures
ExpectationStepFn = Callable[
    # (state, particle_ids, halfset_ids) -> (posterior_accumulators, posterior_meta)
    [InitialModelState, np.ndarray, np.ndarray],
    tuple,  # (List[VdamAccumulator], dict)
]
"""E-step callback. Must return `(accumulators, meta)` where accumulators
holds 2K entries when pseudo_halfsets is on (halfset-0 first, then
halfset-1) and meta is a free-form dict written into per-iter STAR output
(Pmax, nr_significant, best_class, best_euler, best_trans).
"""

IterArtifactSink = Callable[[InitialModelState, int, dict], None]


def _posterior_sums_from_meta(meta: dict, key: str) -> np.ndarray | None:
    value = meta.get(key)
    if value is not None:
        return np.asarray(value, dtype=np.float64)

    prefix = "halfset_"
    suffix = f"_{key}"
    values = [
        np.asarray(meta[name], dtype=np.float64)
        for name in sorted(meta)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if not values:
        return None
    total = values[0].copy()
    for value in values[1:]:
        total += value
    return total


def update_probabilities_from_estep_meta(
    state: InitialModelState,
    meta: dict,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """Update class and direction priors from E-step posterior masses.

    Mirrors ``MlOptimiser::maximizationOtherParameters`` for ``pdf_class``
    and ``pdf_direction``. Subset gradient iterations use momentum ``mu``;
    all-particle or EM-tail iterations use ``my_mu = 0`` and replace priors by
    the current weighted sums.
    """
    class_sums = _posterior_sums_from_meta(meta, "class_posterior_sums")
    if class_sums is None:
        return state
    class_sums = np.asarray(class_sums, dtype=np.float64)
    if class_sums.shape != (state.K,):
        raise ValueError(f"class_posterior_sums must have shape ({state.K},), got {class_sums.shape}")
    if not np.all(np.isfinite(class_sums)) or np.any(class_sums < 0.0):
        raise ValueError("class_posterior_sums must be non-negative and finite")
    sum_weight = float(np.sum(class_sums))
    if sum_weight <= 0.0:
        return state

    my_mu = float(mu) if do_grad and state.subset_size != -1 else 0.0
    if my_mu < 0.0 or my_mu > 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")

    new_state = replace(state)
    new_pdf_class = np.asarray(state.pdf_class, dtype=np.float64) * my_mu
    new_pdf_class += (1.0 - my_mu) * class_sums / sum_weight
    pdf_class_sum = float(np.sum(new_pdf_class))
    if pdf_class_sum > 0.0:
        new_pdf_class /= pdf_class_sum
    new_state.pdf_class = new_pdf_class

    direction_sums = _posterior_sums_from_meta(meta, "class_direction_posterior_sums")
    if direction_sums is not None and state.pdf_direction is not None:
        direction_sums = np.asarray(direction_sums, dtype=np.float64)
        expected = np.asarray(state.pdf_direction).shape
        if direction_sums.shape != expected:
            raise ValueError(f"class_direction_posterior_sums must have shape {expected}, got {direction_sums.shape}")
        if not np.all(np.isfinite(direction_sums)) or np.any(direction_sums < 0.0):
            raise ValueError("class_direction_posterior_sums must be non-negative and finite")
        new_pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64) * my_mu
        new_pdf_direction += (1.0 - my_mu) * direction_sums / sum_weight
        new_state.pdf_direction = new_pdf_direction

    return new_state


def default_schedule_update(
    state: InitialModelState,
    iter: int,
    phase_lengths: VdamPhaseLengths,
    *,
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    nr_particles: int,
    tau2_fudge_arg: float,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
) -> InitialModelState:
    """Apply the three VDAM schedules to `state.iter=iter`."""
    subset_size = compute_subset_size(
        iter=iter,
        phase_lengths=phase_lengths,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        nr_particles=nr_particles,
        nr_iter=state.nr_iter,
        grad_em_iters=grad_em_iters,
        has_converged=state.has_converged,
        grad_has_converged=state.grad_has_converged,
        nr_classes=state.K,
    )
    stepsize = compute_stepsize(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
    )
    tau2_fudge = compute_tau2_fudge(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
        tau2_fudge_arg=tau2_fudge_arg,
    )
    new_state = replace(state)
    new_state.iter = iter
    new_state.subset_size = subset_size
    new_state.grad_current_stepsize = stepsize
    new_state.tau2_fudge_factor = tau2_fudge
    return new_state


def select_subset_for_iter(
    state: InitialModelState,
    iter: int,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    rnd_unif_factory: Callable[[int], RndUnifFn],
    random_seed: int,
    do_grad: bool,
) -> InitialModelState:
    """Shuffle the full particle list with seed `random_seed + iter`, take
    the first `subset_size` (or all particles if -1), stable-sort by optics
    group, and alternate halfset ids.

    `rnd_unif_factory` is a callable mapping a seed to an `RndUnifFn`; we
    pass `random_seed + iter` to match RELION's
    `randomiseParticlesOrder(random_seed + iter, ...)`.
    """
    # RELION's exp_model.cpp:451 uses `std::shuffle(sorted_idx, std::mt19937(seed))`
    # for non-halves randomisation. The Python rnd_unif Fisher-Yates does NOT
    # match std::shuffle byte-for-byte, so we route through the C++ binding
    # `vdam_randomise_particles_order` which calls std::shuffle directly.
    # Falls back to the Python implementation if the binding is unavailable.
    try:
        from recovar.relion_bind import _relion_bind_core as _bind

        shuffled = np.asarray(
            _bind.vdam_randomise_particles_order(int(nr_particles), int(random_seed + iter)), dtype=np.int64
        )
    except (ImportError, AttributeError):
        rnd = rnd_unif_factory(random_seed + iter)
        shuffled = randomise_particles_order(nr_particles, rnd)
    subset_size = state.subset_size if state.subset_size != -1 else nr_particles
    # `-1` (all particles) still needs to be translated via select_vdam_subset
    pseudo = do_grad and pseudo_halfsets_active(gradient_refine=True, do_split_random_halves=False)
    plan = select_vdam_subset(
        shuffled_particle_ids=shuffled,
        subset_size=subset_size,
        optics_group_by_particle=optics_group_by_particle,
        pseudo_halfsets=pseudo,
    )
    new_state = replace(state)
    new_state.subset_particle_ids = plan.particle_ids
    new_state.subset_halfset_ids = plan.halfset_ids
    new_state.pseudo_halfsets = pseudo
    return new_state


def run_vdam_iterations(
    state: InitialModelState,
    *,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    tau2_fudge_arg: float,
    grad_em_iters: int,
    random_seed: int,
    rnd_unif_factory: Callable[[int], RndUnifFn],
    expectation_step: ExpectationStepFn,
    iter_artifact_sink: IterArtifactSink = lambda *args, **kw: None,
    grad_ini_frac: float = 0.3,
    grad_fin_frac: float = 0.2,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """Run the full VDAM iteration loop.

    `state` must already be the output of `initialise_denovo_state(...)`
    with `sigma2_noise` filled (via `seed_noise_from_mavg`).
    """
    phase_lengths = compute_phase_lengths(state.nr_iter, grad_ini_frac, grad_fin_frac)
    current = state

    for it in range(1, state.nr_iter + 1):
        do_grad = ((state.nr_iter - it) >= grad_em_iters) and not current.has_converged

        current = default_schedule_update(
            current,
            iter=it,
            phase_lengths=phase_lengths,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            nr_particles=nr_particles,
            tau2_fudge_arg=tau2_fudge_arg,
            grad_em_iters=grad_em_iters,
        )

        current = select_subset_for_iter(
            current,
            iter=it,
            nr_particles=nr_particles,
            optics_group_by_particle=optics_group_by_particle,
            rnd_unif_factory=rnd_unif_factory,
            random_seed=random_seed,
            do_grad=do_grad,
        )

        # E-step: caller-supplied closure over the data loader + dense kernels
        accumulators, meta = expectation_step(
            current,
            current.subset_particle_ids,
            current.subset_halfset_ids,
        )

        # M-step
        current = vdam_m_step(
            current,
            accumulators=accumulators,
            grad_current_stepsize=current.grad_current_stepsize,
            tau2_fudge_factor=current.tau2_fudge_factor,
        )
        current = update_probabilities_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)

        iter_artifact_sink(current, it, meta)

    return current
