"""VDAM M-step: gradient moment update + reference reconstruction.

Defaults to the RELION C++ moment/reconstruction transaction. The opt-in JAX
backend retains the same state layout and native diagnostic boundaries.
"""

from __future__ import annotations

from typing import Literal

from recovar.em.vdam.mstep_accumulator import VdamAccumulator
from recovar.em.vdam.mstep_single_class import vdam_m_step_single_class
from recovar.em.vdam.state import InitialModelState

# Numerical state owned by the M transaction; authoritative priors stay separate.


def vdam_m_step(
    state: InitialModelState,
    accumulators: list[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float | None = None,
    padding_factor: int = 1,
    use_native_transaction: bool = True,
    mstep_backend: Literal["native", "jax"] = "native",
    mstep_compute_dtype: Literal["float32", "float64"] = "float64",
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
            use_native_transaction=use_native_transaction,
            mstep_backend=mstep_backend,
            mstep_compute_dtype=mstep_compute_dtype,
        )
    return out
