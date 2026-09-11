"""The state-swap snapshot and probe in the iteration loop take their loop values from one late-binding closure."""

import inspect

from recovar.em.dense_single_volume import iteration_loop


def test_state_swap_inputs_have_one_owner():
    src = inspect.getsource(iteration_loop)
    assert src.count("def _state_swap_inputs():") == 1 and src.count("**_state_swap_inputs()") == 2
    d = src.index("def _state_swap_inputs():")
    assert d < src.index("_snapshot_state_swap_inputs(**_state_swap_inputs())") < src.index("_apply_state_swap_probe(", d)
    # neither call lists the loop values inline any more
    assert "_snapshot_state_swap_inputs(\n" not in src and "recovar_snapshot=recovar_state_swap_snapshot,\n            state=state," not in src
