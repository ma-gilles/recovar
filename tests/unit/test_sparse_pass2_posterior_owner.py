"""The sparse pass-2 posterior normalization has its own owner."""

import inspect

from recovar.em.sparse_pass2 import sparse_pass2_bucketed, sparse_pass2_posterior


def test_owner_holds_the_posteriors_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for name in ("_normalize_pass2_bucket_with_log_z", "_winner_take_all_bucket_probs", "_relion_f32_fine_posterior", "_relion_pass2_reconstruction_probs_for_mstep", "_relion_joint_winner_take_all_masks"):
        assert inspect.getmodule(getattr(sparse_pass2_posterior, name)) is sparse_pass2_posterior and f"\ndef {name}(" not in pass2_src
    assert sparse_pass2_bucketed._relion_pass2_reconstruction_probs_for_mstep is sparse_pass2_posterior._relion_pass2_reconstruction_probs_for_mstep
    assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(sparse_pass2_posterior)
