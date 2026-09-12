"""The sparse pass-2 execution-policy switches and device-memory budgets have their own owners."""

import inspect

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed, sparse_pass2_budget, sparse_pass2_policy

from recovar.em.dense_single_volume.helpers import sparse_pass2_window


def test_owners_hold_the_definitions_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for name in ("_resolve_bpref_execution_bucket_policy", "_relion_wavg_direct_modes", "_compact_pair_execution_enabled_for_pass", "_tail_bucket_coalesce_params_for_pass"):
        assert inspect.getmodule(getattr(sparse_pass2_policy, name)) is sparse_pass2_policy and f"\ndef {name}(" not in pass2_src
    for name in ("_device_memory_limit_bytes", "_max_hypotheses_per_microbatch_for_pass", "_projection_budget_pixels_for_pass", "_optional_positive_int_env"):
        assert inspect.getmodule(getattr(sparse_pass2_budget, name)) is sparse_pass2_budget and f"\ndef {name}(" not in pass2_src
    assert sparse_pass2_bucketed._compact_pair_execution_enabled_for_pass is sparse_pass2_policy._compact_pair_execution_enabled_for_pass
    assert sparse_pass2_window._device_memory_limit_bytes is sparse_pass2_budget._device_memory_limit_bytes
    for mod in (sparse_pass2_policy, sparse_pass2_budget):
        assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(mod)
    assert "sparse_pass2_policy import" not in inspect.getsource(sparse_pass2_budget)
