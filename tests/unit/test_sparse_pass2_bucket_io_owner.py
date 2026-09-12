"""The sparse pass-2 bucket planning, per-bucket input preparation and dumps have their own owners."""

import inspect

from recovar.em.dense_single_volume.helpers import (
    sparse_pass2_bucket_io,
    sparse_pass2_bucket_plan,
    sparse_pass2_bucketed,
    sparse_pass2_dump,
)


def test_owners_hold_the_definitions_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for mod, names in (
        (sparse_pass2_bucket_plan, ("_hybrid_k_class_compact_pair_execution_buckets", "_validate_k_class_execution_bucket_partition", "_compact_k_class_pair_plan_stats", "_compact_pair_hybrid_threshold_reports")),
        (sparse_pass2_bucket_io, ("_prepare_bucket_io", "_half_translation_phase_table_for_indices", "_relion_translation_angles_f32", "_divide_by_safe_ctf")),
        (sparse_pass2_dump, ("Pass2DumpComplete", "_pass2_dump_requested_for_bucket", "_prioritize_stopped_pass2_dump_buckets", "_log_sparse_kclass_group_timing")),
    ):
        for name in names:
            assert inspect.getmodule(getattr(mod, name)) is mod and f"\ndef {name}(" not in pass2_src and f"\nclass {name}(" not in pass2_src
    assert sparse_pass2_bucketed._prepare_bucket_io is sparse_pass2_bucket_io._prepare_bucket_io
    assert sparse_pass2_bucketed.Pass2DumpComplete is sparse_pass2_dump.Pass2DumpComplete
    for mod in (sparse_pass2_bucket_plan, sparse_pass2_bucket_io, sparse_pass2_dump):
        assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(mod)
