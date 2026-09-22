"""Accounting for current source-faithful local BPref storage, not activation."""
import pytest

from recovar.em.local.local_batch_planning import local_source_bpref_staging_bytes


@pytest.mark.parametrize("current,compiled_plus_native", [
    (56, 55136428), (564, 52119414060), (800, 148361512876),
])
def test_local_budget_covers_compiled_wrapper_and_native_payload(current, compiled_plus_native):
    # Immutable diagnostic: local_native_compiled_memory_v1/RESULT.json plus
    # outer half slab and the two native copies; C64/F32, denominator omitted.
    pad = 2 * current + 3
    shape = (pad, pad, pad // 2 + 1)
    voxels = shape[0] * shape[1] * shape[2]
    global_texture_and_bpref = voxels * (8 + 12)
    budget = global_texture_and_bpref + local_source_bpref_staging_bytes(shape, shape)
    assert budget >= compiled_plus_native
    # Reserve is one F32 accumulator beyond the observed conversion workspace.
    assert budget - compiled_plus_native <= 4 * voxels


def test_local_budget_uses_distinct_projector_and_accumulator_shapes():
    # Projector 11^2*6, BPref 19^2*10; reserve each at its own size.
    assert local_source_bpref_staging_bytes((11, 11, 6), (19, 19, 10)) == (
        11**3 * 8 + 2 * (11 * 11 * 6) * 8 + 2 * (19 * 19 * 10) * 12
    )


@pytest.mark.parametrize("shape", [(0, 0, 1), (10, 10, 6), (11, 9, 6), (11, 11, 5), (11, 11), (11.5, 11.5, 6)])
def test_local_budget_rejects_invalid_geometry(shape):
    with pytest.raises(ValueError, match="half shape"):
        local_source_bpref_staging_bytes(shape, (11, 11, 6))
    with pytest.raises(ValueError, match="half shape"):
        local_source_bpref_staging_bytes((11, 11, 6), shape)


@pytest.mark.parametrize("name", [
    "RECOVAR_EXACT_LOCAL_PROJECTOR_CAPACITY", "RECOVAR_EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY",
    "RECOVAR_EXACT_LOCAL_BPREF_TRANSACTION", "RECOVAR_EXACT_LOCAL_BPREF_PARTICLE_CAPACITY",
    "RECOVAR_DISABLE_LOCAL_BIG_JIT", "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY",
    "RECOVAR_EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB",
])
def test_alternate_local_topologies_do_not_use_source_bpref_budget(monkeypatch, name):
    from recovar.em.local.local_batch_planning import local_source_bpref_planning_supported
    monkeypatch.setenv(name, "1")
    assert not local_source_bpref_planning_supported()
    monkeypatch.setenv(name, "0")
    assert local_source_bpref_planning_supported()
