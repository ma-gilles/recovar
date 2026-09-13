"""Both bucketed pass-2 entry points resolve their RELION fine-scoring flags through one owner."""

import inspect

from recovar.em.sparse_pass2 import sparse_pass2_bucketed as sp


def test_both_entry_points_use_the_owner():
    for fn in (sp.compute_pass2_stats_sparse_bucketed, sp.compute_k_class_pass2_stats_sparse_fused):
        src = inspect.getsource(fn)
        assert src.count("= _pass2_relion_flags(") == 1
        assert "use_exact_relion_gaussian = bool(" not in src and "use_relion_f32_fine_posterior = bool(" not in src


def test_owner_rules(monkeypatch):
    from recovar.em.sparse_pass2 import sparse_pass2_window

    monkeypatch.setattr(sparse_pass2_window, "parse_env_flag", lambda name, default=False: False)
    monkeypatch.setattr(sparse_pass2_window, "relion_x_half_f32_fine_posterior_enabled", lambda: True)
    exact, ffi, f32 = sp._pass2_relion_flags(relion_exact_fine_gaussian=True, relion_firstiter_score_mode="normalized_cc", relion_fine_diff2_fused_ffi=False, relion_f32_fine_posterior=False)
    assert (exact, ffi, f32) == (False, False, True)
    exact, ffi, f32 = sp._pass2_relion_flags(relion_exact_fine_gaussian=True, relion_firstiter_score_mode="gaussian", relion_fine_diff2_fused_ffi=True, relion_f32_fine_posterior=False)
    assert (exact, ffi, f32) == (True, True, True)
