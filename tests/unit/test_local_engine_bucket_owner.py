"""The exact local engine's bucket stages, BPref capture and physical grid have their own owners."""

import inspect

from recovar.em.dense_single_volume import (
    local_bpref_capture,
    local_bucket_stages,
    local_em_engine,
    local_physical_grid,
)


def test_owners_hold_the_definitions_and_the_engine_routes_to_them():
    engine_src = inspect.getsource(local_em_engine)
    for mod, names in (
        (local_bucket_stages, ("_project_local_bucket", "_postprocess_local_bucket", "_accumulate_packed_noise_chunk", "_plan_local_fine_job_capacities", "_noise_norm_capacity")),
        (local_bpref_capture, ("_exact_local_bpref_capture_static_kwargs", "_bpref_capture_priors", "_filter_buckets_to_debug_targets")),
        (local_physical_grid, ("_accumulate_relion_vdam_physical_particle_grid", "_accumulate_relion_physical_particle_grid", "_source_faithful_bpref_particle_slices")),
    ):
        for name in names:
            assert inspect.getmodule(getattr(mod, name)) is mod and f"\ndef {name}(" not in engine_src
    assert local_em_engine._project_local_bucket is local_bucket_stages._project_local_bucket
    assert local_em_engine._accumulate_relion_vdam_physical_particle_grid is local_physical_grid._accumulate_relion_vdam_physical_particle_grid
    for mod in (local_bucket_stages, local_bpref_capture, local_physical_grid):
        assert "dense_single_volume.local_em_engine import" not in inspect.getsource(mod)
