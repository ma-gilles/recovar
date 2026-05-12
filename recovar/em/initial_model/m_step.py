"""VDAM M-step: gradient moment update + reference reconstruction.

Routes each step through the RELION C++ binding (backprojector.cpp:1933-2054)
for bit-identical parity: reweightGrad, getFristMoment, getSecondMoment,
applyMomenta, updateSSNRarrays, reconstructGrad.
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
    """VDAM M-step for one class (per-class loop matches RELION's binding shape).

    Pseudo-halfsets: FSC/noise-power is derived from the halfset-data difference
    in ``applyMomenta``; ``reconstructGrad`` then uses ``mom1_noise_power``.
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
    # backprojector.h:335/343 EMA defaults
    mu_first, mu_second = 0.9, 0.999

    import os as _os

    _dump_dir = _os.environ.get("RECOVAR_MSTEP_DUMP_DIR")
    _do_dump = _dump_dir is not None and int(getattr(state, "iter", 0)) == int(
        _os.environ.get("RECOVAR_MSTEP_DUMP_ITER", "1")
    )
    _dump_prefix = f"c{k}_" if state.K > 1 else ""

    def _dump(name, arr):
        if not _do_dump:
            return
        from pathlib import Path as _Path

        _Path(_dump_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{_dump_dir}/{_dump_prefix}{name}.npy", np.asarray(arr))

    _dump("accum_h0_data", accum_h0.data)
    _dump("accum_h0_weight", accum_h0.weight)
    if state.pseudo_halfsets:
        _dump("accum_h1_data", accum_h1.data)
        _dump("accum_h1_weight", accum_h1.weight)
    _dump("iref_in", state.Iref[k])
    _dump("Igrad1_in_h0", state.Igrad1[half_slot_index(k, 0, state.K, state.pseudo_halfsets)])
    if state.pseudo_halfsets:
        _dump("Igrad1_in_h1", state.Igrad1[half_slot_index(k, 1, state.K, state.pseudo_halfsets)])
    _dump("Igrad2_in", state.Igrad2[k])
    _dump("fsc_halves_in", state.fsc_halves_class[k])

    # Step 2. reweightGrad per halfset
    data_h0 = np.asarray(bind.vdam_reweight_grad(accum_h0.data, accum_h0.weight, ori_size, padding_factor, 1, r_max))
    if state.pseudo_halfsets:
        data_h1 = np.asarray(
            bind.vdam_reweight_grad(accum_h1.data, accum_h1.weight, ori_size, padding_factor, 1, r_max)
        )
    else:
        data_h1 = None
    _dump("data_h0_post_reweight", data_h0)
    if data_h1 is not None:
        _dump("data_h1_post_reweight", data_h1)

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
    _dump("m1_h0_post", new_Igrad1[slot_h0])
    if state.pseudo_halfsets:
        _dump("m1_h1_post", new_Igrad1[slot_h1])

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
        _dump("m2_post", new_Igrad2[k])

    # Step 5. applyMomenta. Non-halfset: pass m1 twice to trigger do_half=false.
    m1_h0 = new_Igrad1[slot_h0]
    m1_h1 = new_Igrad1[slot_h1] if state.pseudo_halfsets else m1_h0
    _post_data, mom1_noise_power = bind.vdam_apply_momenta(
        data_h0, m1_h0, m1_h1, new_Igrad2[k], ori_size, padding_factor, 1, r_max
    )
    _post_data = np.asarray(_post_data)
    mom1_noise_power = np.asarray(mom1_noise_power)
    _dump("post_apply_data", _post_data)
    _dump("mom1_noise_power", mom1_noise_power)

    # Step 6. updateSSNRarrays (update_tau2_with_fsc=false in gradient mode);
    # drives updateCurrentResolution for the next expectation step.
    new_tau2_class = state.tau2_class.copy()
    new_sigma2_class = state.sigma2_class.copy()
    new_fourier_coverage_class = state.fourier_coverage_class.copy()
    new_data_vs_prior_class = state.data_vs_prior_class.copy()
    fsc_for_ssnr = np.asarray(state.fsc_halves_class[0], dtype=np.float64)
    # K=1: avg(h0,h1) matches RELION's unified BPref weight (HEALPix 1→2 at iter-10).
    # K>1: h0/h1 per-class accumulators are asymmetric, so averaging regresses K=4 CC.
    if accum_h1 is not None and state.K == 1:
        weight_for_ssnr = 0.5 * (accum_h0.weight + accum_h1.weight)
    else:
        weight_for_ssnr = accum_h0.weight
    tau2, sigma2, data_vs_prior, fourier_coverage = bind.vdam_update_ssnr_arrays_from_bpref(
        weight_for_ssnr,
        fsc_for_ssnr,
        state.tau2_class[k],
        tau2_fudge_factor,
        ori_size,
        padding_factor,
        1,
        r_max,
        False,
        False,
        False,
    )
    new_tau2_class[k] = np.asarray(tau2, dtype=np.float64)
    new_sigma2_class[k] = np.asarray(sigma2, dtype=np.float64)
    new_data_vs_prior_class[k] = np.asarray(data_vs_prior, dtype=np.float64)
    new_fourier_coverage_class[k] = np.asarray(fourier_coverage, dtype=np.float64)
    _dump("tau2_post_ssnr", new_tau2_class[k])
    _dump("sigma2_post_ssnr", new_sigma2_class[k])
    _dump("data_vs_prior_post_ssnr", new_data_vs_prior_class[k])
    _dump("fourier_coverage_post_ssnr", new_fourier_coverage_class[k])

    # Step 7. reconstructGrad updates Iref[k] (RELION pseudo-halfset path —
    # mom1_noise_power required for correct FSC/tau weighting). Convert
    # recovar↔RELION at the boundary so the gradient is in the same frame as the accumulators.
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    iref_relion_in = recovar_volume_to_relion(np.asarray(state.Iref[k]))
    _dump("iref_relion_in", iref_relion_in)
    effective_stepsize = float(grad_current_stepsize) * (
        1.0 - np.exp(-float(3 * state.K + 10) * float(np.asarray(state.pdf_class)[k]))
    )
    _dump("effective_stepsize", np.asarray([effective_stepsize], dtype=np.float64))
    new_Iref = state.Iref.copy()
    new_Iref[k] = relion_volume_to_recovar(
        np.asarray(
            bind.vdam_reconstruct_grad(
                iref_relion_in,
                _post_data,
                accum_h0.weight,
                state.fsc_halves_class[k],
                effective_stepsize,
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
    _dump("iref_out_recovar_frame", new_Iref[k])
    _dump("iref_out_relion_frame", recovar_volume_to_relion(new_Iref[k]))

    new_state = replace(state)
    new_state.Iref = new_Iref
    new_state.Igrad1 = new_Igrad1
    new_state.Igrad2 = new_Igrad2
    new_state.tau2_class = new_tau2_class
    new_state.sigma2_class = new_sigma2_class
    new_state.data_vs_prior_class = new_data_vs_prior_class
    new_state.fourier_coverage_class = new_fourier_coverage_class
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
    """Full VDAM M-step over K classes.

    ``accumulators`` holds ``2K`` entries when ``pseudo_halfsets`` is active
    (halfset 0 of each class first, then halfset 1), else ``K``.
    """
    K = state.K
    expected = 2 * K if state.pseudo_halfsets else K
    if len(accumulators) != expected:
        raise ValueError(f"expected {expected} accumulators, got {len(accumulators)}")

    out = state
    for k in range(K):
        out = vdam_m_step_single_class(
            out,
            k=k,
            accum_h0=accumulators[k],
            accum_h1=accumulators[K + k] if state.pseudo_halfsets else None,
            grad_current_stepsize=grad_current_stepsize,
            tau2_fudge_factor=tau2_fudge_factor,
            grad_min_resol_shell=grad_min_resol_shell,
            padding_factor=padding_factor,
        )
    return out
