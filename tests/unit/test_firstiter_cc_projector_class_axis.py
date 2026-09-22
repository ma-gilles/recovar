"""The coarse K-class scorer accepts K=1 single-slab RELION projectors (final-Q contract)."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _slab(shape, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def test_k1_single_slab_gains_class_axis_before_upload():
    """The firstiter-CC probe passes its host-compacted (z, y, x_half) PPref for K=1.

    Consolidation 41128dcd0 kept the 4-D-only check, so every real-data K=1 run
    failed in iteration 1 (e.g. 'got (79, 79, 40)').
    """
    from recovar.em.scoring.significance import _class_stacked_coarse_relion_projector

    slab = _slab((23, 23, 12))
    result = _class_stacked_coarse_relion_projector(slab, 1, use_float64_scoring=False, use_float64_projections=False)
    assert result.shape == (1, 23, 23, 12)
    assert result.dtype == np.complex64
    np.testing.assert_array_equal(np.asarray(result)[0], slab)


def test_class_stacked_projector_passes_through_and_rejects_mismatches():
    from recovar.em.scoring.significance import _class_stacked_coarse_relion_projector

    stacked = _slab((2, 9, 9, 6))
    result = _class_stacked_coarse_relion_projector(
        stacked, 2, use_float64_scoring=False, use_float64_projections=False
    )
    np.testing.assert_array_equal(np.asarray(result), stacked)
    with pytest.raises(ValueError, match="relion_projector_half must have shape"):
        _class_stacked_coarse_relion_projector(stacked[0], 2, use_float64_scoring=False, use_float64_projections=False)
    with pytest.raises(ValueError, match="relion_projector_half must have shape"):
        _class_stacked_coarse_relion_projector(stacked, 1, use_float64_scoring=False, use_float64_projections=False)
