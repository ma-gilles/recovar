"""VDAM M-step: gradient moment update + reference reconstruction.

Pipeline per class k:

  1. Raw accumulation of (data_k, weight_k) from the posterior-weighted
     backprojection of particles in this halfset (handled in Phase 4 by
     the E-step adapter / dense-path backprojector).

  2. `reweightGrad(data_k, weight_k)` — divides by max(1, weight) to
     normalise accumulator (backprojector.cpp:1933).

  3. `getFristMoment(Igrad1[h*K + k], reweighted_data_k, mu)` — EMA
     first moment update per halfset slot (backprojector.cpp:1943).

  4. `getSecondMoment(Igrad2[k], data_h0, data_h1, mu)` — normalised
     difference second moment (backprojector.cpp:1975).

  5. `applyMomenta(Igrad1[h0_k], Igrad1[h1_k], Igrad2[k])` — combine
     momenta + derive mom1_noise_power (backprojector.cpp:2000).

  6. `reconstructGrad(Iref[k], fsc_halves_class[k],
                      grad_current_stepsize, tau2_fudge_factor,
                      grad_min_resol_shell, use_fsc=False)` —
     apply the gradient update to Iref (backprojector.cpp:2054).

All six primitives are routed through the RELION C++ bindings added in
Phase 2 so the M-step is bit-identical to the RELION implementation from
the first commit. A later-phase pure-JAX port is possible but out of scope
for parity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .state import InitialModelState, half_slot_index


def _get_bindings():
    """Import the RELION binding module; raise a clear error if unbuilt."""
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "VDAM M-step requires the RELION bindings. Run:\n  pixi run python recovar/relion_bind/build.py"
        ) from e
    return m


@dataclass
class VdamAccumulator:
    """Per-class raw backprojection accumulator.

    The E-step adapter produces one `VdamAccumulator` per `(class, halfset)`
    pair (so `2K` total when `pseudo_halfsets` is active). `data` and
    `weight` have the padded Fourier shape `(N_pad, N_pad, N_pad // 2 + 1)`
    at `padding_factor=1`.
    """

    data: np.ndarray  # complex128, shape (Nz_pad, Ny_pad, Nx_pad_half)
    weight: np.ndarray  # float64, same shape
    class_idx: int
    halfset_idx: int


def _r_max_from_state(state: InitialModelState, padding_factor: int) -> int:
    """`r_max = current_size / 2` at padding_factor=1; RELION's
    `initZeros(-1)` computes `r_max = ori_size / (padding_factor * 2)` so
    we pass the current-size equivalent explicitly.

    Source: backprojector.cpp::initZeros and ml_optimiser.cpp:5846.
    """
    return state.current_size // 2


def vdam_m_step_single_class(
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: Optional[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float = 0.0,
    padding_factor: int = 1,
) -> InitialModelState:
    """Run the VDAM M-step for a single class, returning an updated state.

    `accum_h1` is None when `pseudo_halfsets` is off.

    For pseudo-halfsets: the FSC / noise-power estimate comes from the
    difference of the two halfset data arrays inside `applyMomenta`; we
    then feed `use_fsc=False` so `reconstructGrad` derives the FSC
    internally from `mom1_noise_power`.

    This is deliberately per-class rather than vectorised over K so the
    RELION binding calls match the per-class loop shape RELION uses.
    """
    if not (0 <= k < state.K):
        raise ValueError(f"class index {k} out of range")
    if state.pseudo_halfsets and accum_h1 is None:
        raise ValueError("pseudo_halfsets=True requires accum_h1")
    if not state.pseudo_halfsets and accum_h1 is not None:
        raise ValueError("pseudo_halfsets=False must have accum_h1=None")

    bind = _get_bindings()
    ori_size = state.ori_size
    r_max = _r_max_from_state(state, padding_factor)
    # RELION uses different EMA constants for the two moments:
    #   getFristMoment default lambda = 0.9   (backprojector.h:335)
    #   getSecondMoment default lambda = 0.999 (backprojector.h:343)
    mu_first = 0.9
    mu_second = 0.999

    # Step 2. reweightGrad per halfset
    data_h0 = np.asarray(bind.vdam_reweight_grad(accum_h0.data, accum_h0.weight, ori_size, padding_factor, 1, r_max))
    if state.pseudo_halfsets:
        data_h1 = np.asarray(
            bind.vdam_reweight_grad(accum_h1.data, accum_h1.weight, ori_size, padding_factor, 1, r_max)
        )
    else:
        data_h1 = None

    # Step 3. getFristMoment per halfset
    slot_h0 = half_slot_index(k, 0, state.K, state.pseudo_halfsets)
    new_Igrad1 = state.Igrad1.copy()

    new_Igrad1[slot_h0] = np.asarray(
        bind.vdam_first_moment(
            data_h0,
            state.Igrad1[slot_h0],
            ori_size,
            padding_factor,
            1,
            r_max,
            **{"lambda": mu_first},
        )
    )
    if state.pseudo_halfsets:
        slot_h1 = half_slot_index(k, 1, state.K, state.pseudo_halfsets)
        new_Igrad1[slot_h1] = np.asarray(
            bind.vdam_first_moment(
                data_h1,
                state.Igrad1[slot_h1],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_first},
            )
        )

    # Step 4. getSecondMoment (uses both halfset accumulators)
    new_Igrad2 = state.Igrad2.copy()
    if state.pseudo_halfsets:
        new_Igrad2[k] = np.asarray(
            bind.vdam_second_moment(
                data_h0,
                data_h1,
                state.Igrad2[k],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_second},
            )
        )

    # Step 5. applyMomenta — returns (post_apply_data, mom1_noise_power)
    # For the non-halfset case, mom1_noise_power stays empty; pass data twice
    # so RELION's `do_half = false` branch triggers.
    m1_h0 = new_Igrad1[slot_h0]
    if state.pseudo_halfsets:
        m1_h1 = new_Igrad1[slot_h1]
    else:
        m1_h1 = m1_h0  # RELION checks nzyxdim mismatch to decide; pass-through

    post_data_tuple = bind.vdam_apply_momenta(
        data_h0,  # any one of the halfset datas — reconstructGrad reads
        # data from PPref anyway, so this argument is used for
        # shape only (internal BackProjector copy)
        m1_h0,
        m1_h1,
        new_Igrad2[k],
        ori_size,
        padding_factor,
        1,
        r_max,
    )
    # We want the noise power for reconstructGrad's internal FSC estimate
    _post_data, mom1_noise_power = post_data_tuple
    mom1_noise_power = np.asarray(mom1_noise_power)
    _post_data = np.asarray(_post_data)

    # Step 6. reconstructGrad updates Iref[k].
    # Pass weight from the h0 accumulator and the noise-power spectrum emitted
    # by applyMomenta. This is the RELION pseudo-halfset path; omitting
    # mom1_noise_power silently falls back to the wrong FSC/tau weighting.
    #
    # Frame: state.Iref[k] is in recovar's real-space frame (negated &
    # axis-flipped relative to RELION), but vdam_reconstruct_grad runs
    # inside RELION's BackProjector — its FFT must see the same volume
    # RELION's reconstructGrad would. Convert recovar↔RELION on the
    # boundary so the gradient update is computed in the same frame as
    # the accumulators (which the iter-1 BPref test verifies are RELION
    # frame at machine precision).
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    iref_relion_in = recovar_volume_to_relion(np.asarray(state.Iref[k]))
    new_Iref = state.Iref.copy()
    new_Iref[k] = relion_volume_to_recovar(
        np.asarray(
            bind.vdam_reconstruct_grad(
                iref_relion_in,
                _post_data,
                accum_h0.weight,
                state.fsc_halves_class[k],
                grad_current_stepsize,
                tau2_fudge_factor,
                ori_size,
                padding_factor,
                1,
                r_max,
                grad_min_resol_shell,
                False,
                True,
                mom1_noise_power,
            )
        )
    )

    new_state = replace(state)
    new_state.Iref = new_Iref
    new_state.Igrad1 = new_Igrad1
    new_state.Igrad2 = new_Igrad2
    # mom1_noise_power per class -> could be exposed via state if Phase 4
    # needs it for scoring; left internal for now.
    return new_state


def vdam_m_step(
    state: InitialModelState,
    accumulators: list[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float = 0.0,
    padding_factor: int = 1,
) -> InitialModelState:
    """Run the full VDAM M-step across all K classes.

    `accumulators` must hold `2K` entries when pseudo_halfsets is active
    (halfset 0 of each class first, then halfset 1), else `K` entries.
    The ordering mirrors RELION's BackProjector indexing.
    """
    K = state.K
    if state.pseudo_halfsets:
        if len(accumulators) != 2 * K:
            raise ValueError(f"expected 2K={2 * K} accumulators, got {len(accumulators)}")
    else:
        if len(accumulators) != K:
            raise ValueError(f"expected K={K} accumulators, got {len(accumulators)}")

    out = state
    for k in range(K):
        if state.pseudo_halfsets:
            accum_h0 = accumulators[k]
            accum_h1 = accumulators[K + k]
        else:
            accum_h0 = accumulators[k]
            accum_h1 = None
        out = vdam_m_step_single_class(
            out,
            k=k,
            accum_h0=accum_h0,
            accum_h1=accum_h1,
            grad_current_stepsize=grad_current_stepsize,
            tau2_fudge_factor=tau2_fudge_factor,
            grad_min_resol_shell=grad_min_resol_shell,
            padding_factor=padding_factor,
        )
    return out
