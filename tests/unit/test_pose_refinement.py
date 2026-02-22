import numpy as np
import pytest

from recovar import pose_refinement

pytestmark = pytest.mark.unit


def test_compute_log_likelihoods_placeholder_smoke():
    rotations = np.zeros((2, 3, 3), dtype=np.float32)
    translations = np.zeros((2, 2), dtype=np.float32)
    assert pose_refinement.compute_log_likelihoods(rotations, translations) is None
