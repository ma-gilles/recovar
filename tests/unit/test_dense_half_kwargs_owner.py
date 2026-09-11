"""The adaptive and single-pass dense half-scoring calls share one keyword set."""

from __future__ import annotations

import inspect

import pytest

import recovar.em.dense_single_volume.iteration_loop as iteration_loop

pytestmark = pytest.mark.unit


def test_dense_scoring_branches_share_one_keyword_set():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("dense_half_kwargs = dict(") == 1
    assert source.count("**dense_half_kwargs,") == 2
    assert "adaptive_result = _score_half_dense_in_bpref_scope(" not in source
    assert "single_pass_result = _score_half_dense_in_bpref_scope(" not in source
    # Only the adaptive branch carries the pass-1 grid and batch/size overrides.
    block = source[source.index("dense_half_kwargs = dict("):]
    block = block[: block.index("score_result = dense_result")]
    for name in ("firstiter_coarse_current_size=coarse_cs", "k_class_image_batch_size_override=", "significance_rotation_block_size_override="):
        assert block.count(name) == 1
    assert "if not use_adaptive and debug.save_intermediates_dir is not None:" in source
