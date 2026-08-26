import numpy as np
import pytest

from scripts.audit_k1_native_bpref_reconstruct import (
    _infer_odd_cube,
    _public_full_to_relion_x_half,
)


@pytest.mark.unit
def test_infer_odd_cube_rejects_non_odd_cube():
    assert _infer_odd_cube(5**3) == 5
    with pytest.raises(ValueError, match="expected odd cube"):
        _infer_odd_cube(4**3)
    with pytest.raises(ValueError, match="expected odd cube"):
        _infer_odd_cube(126)


@pytest.mark.unit
def test_public_full_to_relion_x_half_inverts_axis_order_and_packs_positive_x():
    side = 5
    public = np.arange(side**3, dtype=np.float32).reshape(side, side, side)
    packed = _public_full_to_relion_x_half(public.reshape(-1), side)

    expected = public.transpose(2, 1, 0)[:, :, side // 2 :]
    np.testing.assert_array_equal(packed, expected)
    assert packed.flags.c_contiguous
