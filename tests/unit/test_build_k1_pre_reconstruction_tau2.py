import pytest

from scripts.build_k1_pre_reconstruction_tau2 import _infer_odd_cube


@pytest.mark.unit
def test_infer_odd_cube_requires_odd_full_cube():
    assert _infer_odd_cube(115**3) == (115, 115, 115)
    with pytest.raises(ValueError, match="odd full-cube"):
        _infer_odd_cube(114**3)
    with pytest.raises(ValueError, match="odd full-cube"):
        _infer_odd_cube(115**3 - 1)
