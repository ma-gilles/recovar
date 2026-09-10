"""Fixed-order segment sums for opt-in run-to-run bitwise repeatability.

``array.at[ids].add(values)`` with repeated ids lowers to GPU atomics whose
association order depends on the scheduler, so float sums differ at the ulp
level between otherwise identical runs.  Under
``RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1`` the EM engines route their
duplicate-index accumulations (per-shell noise/power binning, per-group scale
terms, per-image residual totals) through :func:`fixed_order_segment_sum`,
a masked reduction whose order is fixed by XLA.  Same operands and dtype; only
the association order changes.  Diagnostic opt-in, see
``docs/development/em_status.md`` (determinism).
"""

from __future__ import annotations

import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

DETERMINISTIC_REDUCTIONS_ENV = "RECOVAR_EM_DETERMINISTIC_REDUCTIONS"


def deterministic_reductions_enabled() -> bool:
    """Read the opt-in at trace time; unset or ``0`` keeps the atomic scatters."""

    return parse_env_binary_flag(DETERMINISTIC_REDUCTIONS_ENV)


def fixed_order_segment_sum(values, segment_ids, n_segments: int):
    """Sum ``values`` (..., n) into ``n_segments`` bins keyed by ``segment_ids`` (n,).

    Ids outside ``[0, n_segments)`` are dropped, matching a sentinel-bin scatter.
    Every bin is a masked reduction over the value axis, so the result does not
    depend on GPU scheduling.  Output shape is ``(..., n_segments)`` in the value dtype.
    """

    n_segments = int(n_segments)
    values = jnp.asarray(values)
    segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
    if segment_ids.ndim != 1 or values.shape[-1:] != segment_ids.shape:
        raise ValueError(
            f"segment ids {segment_ids.shape} must index the last axis of values {values.shape}"
        )
    bin_ids = jnp.arange(n_segments, dtype=jnp.int32)
    member = segment_ids[None, :] == bin_ids[:, None]
    zero = jnp.zeros((), dtype=values.dtype)
    return jnp.sum(jnp.where(member, values[..., None, :], zero), axis=-1)


def add_segment_sum(accumulator, segment_ids, values):
    """``accumulator.at[segment_ids].add(values)``, fixed-order under the opt-in."""

    if not deterministic_reductions_enabled():
        return accumulator.at[segment_ids].add(values)
    return accumulator + fixed_order_segment_sum(
        jnp.asarray(values, dtype=accumulator.dtype), segment_ids, accumulator.shape[0]
    )
