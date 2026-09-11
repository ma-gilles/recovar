"""The iteration loop snapshots its frozen scoring state through one late-binding closure."""

import inspect

from recovar.em.dense_single_volume import iteration_loop


def test_frozen_scoring_state_snapshot_has_one_owner():
    src = inspect.getsource(iteration_loop)
    assert src.count("_frozen_scoring_state_arrays(") == 1  # only the closure calls the imported snapshot builder
    assert src.count("def _frozen_scoring_state_now():") == 1
    assert src.count("= _frozen_scoring_state_now()") == 1 and src.count("_frozen_scoring_state_now(),") == 1
    d = src.index("def _frozen_scoring_state_now():")
    assert d < src.index("frozen_initial_scoring_state = _frozen_scoring_state_now()") < src.index("_assert_frozen_scoring_state_unchanged(", d)
