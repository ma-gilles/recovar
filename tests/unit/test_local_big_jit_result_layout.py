"""Named local results preserve compiled leaves and carry-buffer ownership."""

import jax
import pytest

from recovar.em.local import local_big_jit as bucket


@pytest.mark.parametrize("route", ["score", "deferred", "source_vdam", "mstep"])
@pytest.mark.parametrize("debug_count", [0, 2, 9])
def test_named_result_preserves_leaf_order_and_carry_identity(route, debug_count):
    values = [object() for _ in range(41)]
    core = bucket._LocalBigJitCore(*values[:22])
    payload = {}
    count = 0
    if route == "deferred":
        count = 10
        payload["deferred_mstep"] = bucket._LocalDeferredMstep(*values[22:32])
    elif route == "source_vdam":
        count = 6
        payload["source_vdam"] = bucket._LocalSourceVdam(*values[22:28])
    elif route == "mstep":
        count = 2
        payload["mstep_tensors"] = bucket._LocalMstepTensors(*values[22:24])
    if debug_count:
        payload["debug"] = bucket._LocalBigJitDebug(*values[22 + count:22 + count + debug_count])
    result = bucket._LocalBigJitResult(core, **payload)
    expected = values[:22 + count + debug_count]
    assert jax.tree_util.tree_leaves(result) == expected
    carry, local = bucket._split_local_big_jit_carry(result)
    assert list(carry) == values[:8] + values[9:11]
    assert jax.tree_util.tree_leaves(local) == [values[8], *expected[11:]]
    assert all(value not in jax.tree_util.tree_leaves(local) for value in carry)
    restored = bucket._reconstruct_fixed_capacity_score_only_result(carry, local)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(result)
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(restored), expected, strict=True))


def test_reconstruct_rejects_incomplete_carry():
    with pytest.raises(ValueError, match="ten carry values"):
        bucket._reconstruct_fixed_capacity_score_only_result((), None)
