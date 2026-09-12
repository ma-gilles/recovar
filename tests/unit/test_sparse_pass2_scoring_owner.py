"""The sparse pass-2 scoring kernels have their own owner."""

import inspect

from recovar.em.sparse_pass2 import sparse_pass2_bucketed, sparse_pass2_scoring


def test_owner_holds_the_scorers_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for name in ("_gaussian_algebraic_score_terms", "_score_pass2_bucket_gaussian_algebraic", "_relion_cuda_fine_diff2_sum", "_relion_cuda_fine_diff2_to_scores", "_score_pass2_bucket_relion_gpu_diff2", "_score_pass2_bucket_normalized_cc", "_relion_powerclass_operands", "_relion_powerclass_noise_terms", "_score_pass2_pairs_relion_gpu_diff2"):
        assert inspect.getmodule(getattr(sparse_pass2_scoring, name)) is sparse_pass2_scoring and f"\ndef {name}(" not in pass2_src
    assert sparse_pass2_bucketed._score_pass2_bucket_relion_gpu_diff2 is sparse_pass2_scoring._score_pass2_bucket_relion_gpu_diff2
    assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(sparse_pass2_scoring)
