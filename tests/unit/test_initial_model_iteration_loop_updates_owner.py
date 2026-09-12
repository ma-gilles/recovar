"""The InitialModel E-step metadata updates and the subset schedule have their own owners."""

import inspect

from recovar.em.vdam import estep_meta_updates, iteration_loop, subset_schedule


def test_owners_hold_the_definitions_and_the_loop_routes_to_them():
    loop_src = inspect.getsource(iteration_loop)
    for name in ("update_noise_from_estep_meta", "update_probabilities_from_estep_meta", "_maybe_dump_noise_update_boundary"):
        assert inspect.getmodule(getattr(estep_meta_updates, name)) is estep_meta_updates and f"\ndef {name}(" not in loop_src
    for name in ("select_subset_for_iter", "restore_subset_order_for_continuation"):
        assert inspect.getmodule(getattr(subset_schedule, name)) is subset_schedule and f"\ndef {name}(" not in loop_src
    assert iteration_loop.update_noise_from_estep_meta is estep_meta_updates.update_noise_from_estep_meta
    assert iteration_loop.select_subset_for_iter is subset_schedule.select_subset_for_iter
    for mod in (estep_meta_updates, subset_schedule):
        assert "initial_model.iteration_loop import" not in inspect.getsource(mod)
