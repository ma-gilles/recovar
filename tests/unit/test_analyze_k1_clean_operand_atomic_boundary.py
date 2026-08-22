import numpy as np

from scripts.analyze_k1_coarse_operand_boundary_v3 import _atomic_add_log_score_values


def test_atomic_outcomes_preserve_native_initial_term_and_sign():
    # Four active lanes for one translation. Different atomic orders have two
    # legal binary32 results at this cancellation boundary.
    lanes = np.asarray([1.0e8, 1.0, -1.0e8, 1.0], dtype=np.float32)

    actual = _atomic_add_log_score_values(
        lanes,
        translation_count=1,
        translation=0,
        initial_diff2=np.float32(0.25),
    )

    assert actual.dtype == np.float32
    assert actual.size >= 2
    assert np.all(actual <= 0.0)
