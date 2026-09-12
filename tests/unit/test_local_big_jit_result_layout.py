"""The 22-value big-JIT core layout has one named owner on both sides of the JIT boundary."""

import inspect

from recovar.em.local import local_big_jit, local_em_engine

CORE_FIELDS = (
    "Ft_y",
    "Ft_ctf",
    "noise_wsum",
    "noise_img_power",
    "noise_a2",
    "noise_xa",
    "noise_scale_xa",
    "noise_scale_aa",
    "bucket_norm_correction",
    "noise_sigma2_offset",
    "noise_sumw",
    "batch_norm",
    "log_Z",
    "best_log_score",
    "best_argmax",
    "max_posterior",
    "probs_sum_t",
    "reconstruction_probs_sum_t",
    "n_significant_samples",
    "reconstruction_sample_mask",
    "reconstruction_rotation_mask",
    "reconstruction_row_count",
)


def test_core_layout_names_and_fixed_capacity_carry_window():
    assert local_big_jit._LocalBigJitCore._fields == CORE_FIELDS
    # The fixed-capacity scan carries slots 7..17 of this layout; the layout owner must keep them in place.
    carry = CORE_FIELDS[local_big_jit._FIXED_CAPACITY_CARRY_START : local_big_jit._FIXED_CAPACITY_CARRY_STOP]
    assert carry[0] == "noise_scale_aa" and carry[-1] == "probs_sum_t" and len(carry) == 10


def test_producer_builds_every_result_from_the_core_and_returns_plain_tuples():
    source = inspect.getsource(local_big_jit.run_local_bucket_big_jit)
    assert source.count("*_LocalBigJitCore(") == 4
    assert source.count("result = (\n") == 4
    # No positional 22-tuple literal remains.
    assert "        result = (\n            Ft_y,\n" not in source
    assert "    result = (\n        Ft_y,\n" not in source


def test_engine_unpacks_the_core_once_and_the_extras_per_layout():
    source = inspect.getsource(local_em_engine)
    assert source.count("big_jit_core = _LocalBigJitCore._make(") == 1
    assert source.count(") = big_jit_core") == 1
    assert source.count(") = big_jit_extras") == 2
    assert source.count("summed, ctf_probs = big_jit_extras") == 1
    assert source.count("\n                ) = big_jit_result\n") == 0


def test_core_flattens_to_the_same_leaves_as_the_tuple():
    import jax

    values = tuple(range(22))
    core = local_big_jit._LocalBigJitCore(*values)
    assert tuple(core) == values
    assert jax.tree_util.tree_leaves((*core, "extra")) == [*values, "extra"]
