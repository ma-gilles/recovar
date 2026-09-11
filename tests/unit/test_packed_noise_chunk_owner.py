"""The exact local M-step accumulates each packed rotation chunk's noise terms through one owner."""

from __future__ import annotations

import inspect

import pytest

import recovar.em.dense_single_volume.local_em_engine as engine

pytestmark = pytest.mark.unit


def test_both_projection_sources_accumulate_through_the_owner():
    source = inspect.getsource(engine.run_local_em_exact)
    assert source.count("_accumulate_packed_noise_chunk(") == 2
    assert "chunk_noise_shells, chunk_a2_shells, chunk_xa_shells = _compute_noise_block(" not in source
    # The single-pass (non-chunked) accumulation keeps its own inline call.
    assert source.count("noise_scale_xa.at[bucket_group_ids].add(") == 1


def test_owner_returns_the_six_accumulators_in_order():
    source = inspect.getsource(engine._accumulate_packed_noise_chunk)
    assert source.rstrip().endswith(
        "return block_noise_shells, block_a2_shells, block_xa_shells, block_norm_residual, noise_scale_xa, noise_scale_aa"
    )
    params = inspect.signature(engine._accumulate_packed_noise_chunk).parameters
    assert list(params)[:3] == ["chunk_proj_for_noise", "chunk_start", "chunk_stop"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in params.items() if name != "chunk_proj_for_noise")
