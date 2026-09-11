"""``read_sparse_pass2_result`` is the single reader of the positional sparse pass-2 tuple."""

import inspect

import numpy as np

from recovar.em.dense_single_volume import k_class
from recovar.em.dense_single_volume.helpers import types


def test_reader_inverts_the_builder_for_every_optional_combination():
    for stats_only in (False, True):
        for noise in (False, True):
            for eulers in (False, True):
                built = types.sparse_pass2_result(
                    "y", "c", "h", "r", "t", "i",
                    relion_stats="stats",
                    score_log_z=types.OMITTED if stats_only else np.array([1.0]),
                    noise_stats="noise" if noise else types.OMITTED,
                    source_eulers="eul" if eulers else types.OMITTED,
                )
                read = types.read_sparse_pass2_result(
                    built, includes_score_log_z=not stats_only, accumulate_noise=noise, return_source_eulers=eulers
                )
                assert read[:7] == ("y", "c", "h", "r", "t", "i", "stats")
                assert (read.score_log_z is None) == stats_only
                assert read.noise_stats == ("noise" if noise else None)
                assert read.source_eulers == ("eul" if eulers else None)


def test_reader_is_the_only_positional_consumer_in_k_class():
    source = inspect.getsource(k_class)
    assert source.count("read_sparse_pass2_result(") == 2
    assert "output[:7]" not in source
    assert "output[next_index]" not in source
    assert "output[7]" not in source
    assert "output[-1]" not in source


def test_reader_lives_next_to_the_builder():
    source = inspect.getsource(types)
    assert source.index("def sparse_pass2_result(") < source.index("class SparsePass2Output(NamedTuple)") < source.index("def read_sparse_pass2_result(")
