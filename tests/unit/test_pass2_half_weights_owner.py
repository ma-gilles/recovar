"""Both bucketed pass-2 entry points build their scoring half-image weights through one owner."""

import inspect

import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sp


def test_both_entry_points_use_the_owner():
    for fn in (sp.compute_pass2_stats_sparse_bucketed, sp.compute_k_class_pass2_stats_sparse_fused):
        src = inspect.getsource(fn)
        assert src.count("= _pass2_half_weights(") == 1 and "make_scoring_half_image_weights(" not in src


def test_owner_casts_and_rewindows_in_double(monkeypatch):
    monkeypatch.setattr(sp, "make_scoring_half_image_weights", lambda shape, *, relion_half_sum, exclude_relion_redundant_x0: jnp.ones((4,), dtype=jnp.float32) * (2.0 if exclude_relion_redundant_x0 else 1.0))

    class Window:
        @staticmethod
        def score_values(w):
            return w[:2]

    hw, hww = sp._pass2_half_weights((2, 2), Window(), half_spectrum_scoring=True, relion_firstiter_score_mode="gaussian", use_float64_scoring=True)
    assert hw.dtype == jnp.float64 and hww.dtype == jnp.float64 and hww.shape == (2,) and float(hw[0]) == 2.0
    hw32, _ = sp._pass2_half_weights((2, 2), Window(), half_spectrum_scoring=True, relion_firstiter_score_mode="normalized_cc", use_float64_scoring=False)
    assert hw32.dtype == jnp.float32 and float(hw32[0]) == 1.0
