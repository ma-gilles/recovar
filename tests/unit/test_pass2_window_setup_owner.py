"""Both bucketed pass-2 entry points resolve their window, precision and fine translation prior through one owner."""

import inspect

import numpy as np

from recovar.em.sparse_pass2 import sparse_pass2_bucketed as sp


def test_window_setup_owner_rejects_exact_gaussian_below_the_image_size_without_half_spectrum():
    import pytest

    with pytest.raises(NotImplementedError, match="half_spectrum_scoring=True"):
        sp._pass2_window_setup((16, 16), current_size=8, reconstruction_current_size=None, half_spectrum_scoring=False, square_window=False,
                               relion_firstiter_score_mode="gaussian", use_exact_relion_gaussian=True, use_float64_scoring=False)
    setup = sp._pass2_window_setup((16, 16), current_size=8, reconstruction_current_size=12, half_spectrum_scoring=True, square_window=False,
                                   relion_firstiter_score_mode="normalized_cc", use_exact_relion_gaussian=True, use_float64_scoring=True)
    assert setup.mstep_current_size == 12 and setup.n_half == 16 * 9
    assert setup.window_spec_kwargs == {"score_square": True, "score_include_dc": True}
    assert setup.precision_policy.use_float64_scoring is True


def test_fine_translation_prior_owner_returns_none_without_a_prior():
    assert sp._fine_translation_prior_2d(None, None, n_images=2, n_fine_trans=3, dtype=np.float32) is None
    out = sp._fine_translation_prior_2d(np.zeros((2, 2)), np.array([0, 0, 1]), n_images=2, n_fine_trans=3, dtype=np.float32)
    assert out.shape == (2, 3) and out.dtype == np.float32


def test_both_entry_points_use_the_owners():
    for fn in (sp.compute_pass2_stats_sparse_bucketed, sp.compute_k_class_pass2_stats_sparse_fused):
        src = inspect.getsource(fn)
        assert src.count("= _pass2_window_setup(") == 1 and src.count("_fine_translation_prior_2d(") == 1
        assert "make_fourier_window_spec(" not in src and "expand_fine_translation_prior(" not in src
