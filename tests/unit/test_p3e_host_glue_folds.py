"""P3-E: the merged head's remaining EM-only host glue, folded into programs.

Ticket: ``em_parity_tickets_20260918/P3E_merged_head_census_and_glue.md``.

Every fold here is pure data movement or host arithmetic, so every assertion
is bitwise against the expression it replaces rather than against a tolerance.
The eager dispatch counts come from the same
``EvalTrace.process_primitive`` hook T19's census uses, so the "no eager
dispatch" claims are the claims the census measures, on CPU.

Covered here:

* two per-chunk host scalars reach the device without an eager
  ``convert_element_type``, and the M-step carry's real-part dtype is NumPy
  promotion rather than a 0-d device allocation read for its dtype;
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2 import resident_pass2 as rp

pytestmark = pytest.mark.unit


class _DispatchCounter:
    """Count eager JAX primitive dispatches, the way the T19 census does."""

    def __init__(self):
        self.count = 0
        self.by_primitive: dict[str, int] = {}

    def __enter__(self):
        from jax._src import core

        self._core = core
        self._original = core.EvalTrace.process_primitive
        counter = self

        def process_primitive(trace, primitive, args, params):
            counter.count += 1
            counter.by_primitive[primitive.name] = (
                counter.by_primitive.get(primitive.name, 0) + 1
            )
            return counter._original(trace, primitive, args, params)

        core.EvalTrace.process_primitive = process_primitive
        return self

    def __exit__(self, *exc):
        self._core.EvalTrace.process_primitive = self._original
        return False


def _same(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right, equal_nan=True)
    )


# ------------------------------------------------------------ host scalars ---


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
def test_scalar_operand_matches_jnp_asarray_bitwise(dtype):
    value = np.int32(4231)
    folded = rp._scalar_operand(value, dtype)
    loose = jnp.asarray(value, dtype=dtype)
    assert folded.dtype == loose.dtype
    assert folded.shape == loose.shape == ()
    assert folded.weak_type == loose.weak_type
    assert int(folded) == int(loose) == 4231


def test_scalar_operand_issues_no_eager_dispatch():
    value = np.int32(17)
    rp._scalar_operand(value, jnp.int32)  # warm any first-call machinery
    with _DispatchCounter() as folded:
        rp._scalar_operand(value, jnp.int32)
    with _DispatchCounter() as loose:
        jnp.asarray(value, dtype=jnp.int32)
    assert folded.count == 0, folded.by_primitive
    assert loose.count > 0, "the expression this replaces did dispatch"


# -------------------------------------------------------- carry real dtype ---


@pytest.mark.parametrize(
    "cross_dtype", [jnp.complex64, jnp.complex128, jnp.float32, jnp.float64]
)
def test_real_part_dtype_is_numpy_promotion(cross_dtype):
    """The host form gives the dtype the device form gave, for every operand."""

    assert (
        np.zeros((), dtype=cross_dtype).real.dtype
        == jnp.zeros((), dtype=cross_dtype).real.dtype
    )


