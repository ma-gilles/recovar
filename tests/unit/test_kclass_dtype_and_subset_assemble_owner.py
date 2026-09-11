"""The pose/score dtype rule and the subset passes' result assembly have one owner each in ``k_class``."""

import inspect

import numpy as np

from recovar.em.dense_single_volume import k_class


def test_dtype_rules():
    assert k_class._pose_dtype_from_kwargs({}) is np.float32 and k_class._score_dtype_from_kwargs({}) is np.float32
    assert k_class._pose_dtype_from_kwargs({"use_float64_projections": True}) is np.float64
    assert k_class._score_dtype_from_kwargs({"use_float64_projections": True}) is np.float32
    assert k_class._score_dtype_from_kwargs({"use_float64_scoring": True}) is np.float64


def test_rules_and_assembly_are_not_repeated():
    src = inspect.getsource(k_class)
    assert src.count('kwargs.get("use_float64_projections", False)') == 1
    assert src.count("_pose_dtype_from_kwargs(") == 4 and src.count("_score_dtype_from_kwargs(") == 4
    for fn in (k_class._run_firstiter_global_winner_subset_pass2, k_class._run_sparse_firstiter_global_winner_subset_pass2):
        body = inspect.getsource(fn)
        assert "results.assemble(" in body and "_assemble_result(" not in body
    assert "firstiter_winner_take_all=True" in inspect.getsource(k_class._PerClassSubsetResults.assemble)
