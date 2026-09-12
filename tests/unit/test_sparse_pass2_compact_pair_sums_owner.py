"""The sparse pass-2 compact-pair weighted sums and active-row selection have their own owner."""

import inspect

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed, sparse_pass2_compact_pair_sums, sparse_pass2_noise_blocks


def test_owner_holds_the_sums_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for name in ("_compact_pair_weighted_rotation_and_image_sums", "_compact_pair_weighted_sums_and_noise_native", "_compact_pair_dense_probs_and_reductions", "_compute_active_noise_rows_chunked", "_rectangular_active_weighted_sums"):
        assert inspect.getmodule(getattr(sparse_pass2_compact_pair_sums, name)) is sparse_pass2_compact_pair_sums and f"\ndef {name}(" not in pass2_src
    assert sparse_pass2_bucketed._compact_pair_weighted_rotation_and_image_sums is sparse_pass2_compact_pair_sums._compact_pair_weighted_rotation_and_image_sums
    for name in ("_compute_noise_block_chunked", "_compute_noise_block_and_norm_residual_chunked"):
        assert inspect.getmodule(getattr(sparse_pass2_noise_blocks, name)) is sparse_pass2_noise_blocks and f"\ndef {name}(" not in pass2_src
    for mod in (sparse_pass2_compact_pair_sums, sparse_pass2_noise_blocks):
        assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(mod)
    assert "sparse_pass2_compact_pair_sums import" not in inspect.getsource(sparse_pass2_noise_blocks)
