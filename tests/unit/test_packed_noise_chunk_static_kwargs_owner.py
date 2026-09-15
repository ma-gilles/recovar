"""Both packed-noise chunk accumulations in the local engine share one binding of their loop-invariant keywords."""

import inspect

from recovar.em.local import local_em_engine


def test_packed_noise_chunk_calls_share_the_static_binding():
    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert src.count("packed_noise_chunk_static_kwargs = dict(") == 1
    assert src.count("**packed_noise_chunk_static_kwargs,") == 2
    bind = src.index("packed_noise_chunk_static_kwargs = dict(")
    assert src.index("if proj_for_noise is None:", bind) < src.index("_accumulate_packed_noise_chunk(", bind)
    segment = src[bind : src.index("_block_until_ready(block_noise_shells, block_norm_residual)", bind)]
    for key in ("bucket_group_ids=bucket_group_ids", "packed_ctf_probs=packed_ctf_probs", "n_shells=n_shells"):
        assert segment.count(key) == 1, key
