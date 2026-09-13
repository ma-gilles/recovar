"""The sparse pass-2 scoring window, RELION flags and per-pass budgets have their own owner."""

import inspect

from recovar.em.sparse_pass2 import sparse_pass2_bucketed, sparse_pass2_window


def test_owner_holds_the_window_and_the_pass2_module_routes_to_it():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for name in ("_sparse_pass2_window_setup", "_pass2_window_setup", "_pass2_relion_flags", "_pass2_half_weights", "_pass2_projection_budget", "_fine_translation_prior_2d", "_shared_k_class_noise_variance", "subtract_projected_reference_from_sparse_mstep_sums"):
        assert inspect.getmodule(getattr(sparse_pass2_window, name)) is sparse_pass2_window and f"\ndef {name}(" not in pass2_src
    assert sparse_pass2_bucketed._pass2_window_setup is sparse_pass2_window._pass2_window_setup
    assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(sparse_pass2_window)
