"""The sparse pass-2 wavg terms, adjoint accumulation and projection blocks have their own owners."""

import inspect

from recovar.em.sparse_pass2 import (
    sparse_pass2_adjoint,
    sparse_pass2_bucketed,
    sparse_pass2_projection_blocks,
    sparse_pass2_wavg,
)


def test_owners_hold_the_definitions_and_the_pass2_module_routes_to_them():
    pass2_src = inspect.getsource(sparse_pass2_bucketed)
    for mod, names in (
        (sparse_pass2_wavg, ("RelionWavgRectangle", "_make_relion_wavg_rectangle", "_relion_wavg_atomic_triplet_terms", "_weighted_image_power_shells_and_per_image")),
        (sparse_pass2_adjoint, ("_accumulate_adjoint_block_chunked", "_accumulate_relion_x_half_per_particle_launches", "_split_compact_pair_buckets_by_projection_gather_budget")),
        (sparse_pass2_projection_blocks, ("_compute_sparse_pass2_projections_block", "_compute_sparse_pass2_windowed_projections_block")),
    ):
        for name in names:
            assert inspect.getmodule(getattr(mod, name)) is mod and f"\ndef {name}(" not in pass2_src and f"\nclass {name}(" not in pass2_src
    assert sparse_pass2_bucketed._accumulate_adjoint_block_chunked is sparse_pass2_adjoint._accumulate_adjoint_block_chunked
    assert sparse_pass2_bucketed._make_relion_wavg_rectangle is sparse_pass2_wavg._make_relion_wavg_rectangle
    for mod in (sparse_pass2_wavg, sparse_pass2_adjoint, sparse_pass2_projection_blocks):
        assert "helpers.sparse_pass2_bucketed import" not in inspect.getsource(mod)
