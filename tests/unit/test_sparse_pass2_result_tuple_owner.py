"""The sparse pass-2 functions assemble their return tuple through one owner."""

import inspect

from recovar.em.helpers import oversampling
from recovar.em.helpers.types import OMITTED, sparse_pass2_result
from recovar.em.sparse_pass2 import sparse_pass2_bucketed


def test_optional_entries_follow_one_fixed_order():
    base = ("y", "ctf", "hard", "rots", "trans", "rot_ids")
    assert sparse_pass2_result(*base) == base
    assert sparse_pass2_result(*base, relion_stats="stats") == base + ("stats",)
    assert sparse_pass2_result(*base, noise_stats="noise") == base + ("noise",)
    assert sparse_pass2_result(*base, relion_stats="stats", noise_stats="noise") == base + ("stats", "noise")
    assert sparse_pass2_result(*base, relion_stats="stats", score_log_z="z", noise_stats="noise", source_eulers="eulers") == base + ("stats", "z", "noise", "eulers")
    # order is the declaration order, not the keyword order at the call site
    assert sparse_pass2_result(*base, source_eulers="eulers", noise_stats="noise", score_log_z="z", relion_stats="stats") == base + ("stats", "z", "noise", "eulers")
    assert sparse_pass2_result(*base, relion_stats=None, noise_stats=OMITTED) == base + (None,)  # None is a value, OMITTED is absence


def test_both_sparse_pass2_functions_use_the_owner():
    for module, name in (
        (oversampling, "_compute_pass2_stats_sparse_perimage_reference"),
        (sparse_pass2_bucketed, "compute_pass2_stats_sparse_bucketed"),
    ):
        source = inspect.getsource(getattr(module, name))
        assert source.count("return sparse_pass2_result(") == 1
        assert "result = result + (" not in source
        assert "relion_stats = OMITTED" in source
    bucketed = inspect.getsource(sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed)
    # the score-only log partition function rides with the statistics, as it always has
    assert "score_log_z=score_log_z if (return_stats and return_score_log_z) else OMITTED," in bucketed
    assert "source_eulers=best_eulers if return_source_eulers else OMITTED," in bucketed
