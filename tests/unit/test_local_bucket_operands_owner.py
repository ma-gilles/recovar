"""The local engine converts a bucket's translation prior, rotation mask and sample mask once per bucket."""

import inspect

from recovar.em.dense_single_volume import local_em_engine


def test_bucket_operands_are_converted_once_before_the_fused_score_chain():
    src = inspect.getsource(local_em_engine.run_local_em_exact)
    start = src.index("bucket_translation_log_prior = jnp.asarray(bucket.translation_log_prior)")
    chain = src[start : src.index("normalize_t0 = time.time()", start)]
    assert chain.count("jnp.asarray(bucket.translation_log_prior)") == 1
    assert chain.count("jnp.asarray(bucket.local_rotation_mask)") == 1
    assert chain.count("jnp.asarray(bucket.local_sample_mask)") == 1
    assert chain.count("bucket_local_sample_mask,") == 6 and chain.count("bucket_translation_log_prior,") == 6
    assert start < src.index("used_fused_score_mstep = True")
    # the deferred reconstruction mask reuses the same bound array
    assert src.count("reconstruction_rotation_mask = bucket_local_rotation_mask") == 1
