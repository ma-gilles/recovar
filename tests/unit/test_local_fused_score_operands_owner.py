"""The four fused-score variants in the local engine receive one bound operand tuple."""

import inspect

from recovar.em.local import local_em_engine


def test_fused_score_variants_share_the_operand_binding():
    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert src.count("fused_score_operands = (") == 1 and src.count("*fused_score_operands,") == 4
    bind = src.index("fused_score_operands = (")
    chain = src[bind : src.index("normalize_t0 = time.time()", bind)]
    assert chain.count("half_weights_windowed if use_window else half_weights") == 2  # the binding and the plain weighted scorer
    assert src.index("used_fused_score_mstep = True") < bind < src.index("if can_use_fused_score_mstep and score_only", bind)
