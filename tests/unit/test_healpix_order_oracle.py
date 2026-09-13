import pytest

from recovar.em.helpers.convergence import RefinementState, _apply_relion_healpix_order_oracle
from recovar.em.refinement.refinement_options import (
    AdaptiveOptions,
    RefinementOptions,
    RefinementSchedule,
    _validate_relion_healpix_orders,
    with_validated_sampling_schedule,
)


def test_validate_relion_healpix_orders_requires_complete_monotone_schedule():
    assert _validate_relion_healpix_orders(
        [3, 3, 4],
        max_iter=3,
        init_healpix_order=3,
        max_healpix_order=7,
    ) == (3, 3, 4)

    with pytest.raises(ValueError, match="at least max_iter"):
        _validate_relion_healpix_orders(
            [3, 3],
            max_iter=3,
            init_healpix_order=3,
            max_healpix_order=7,
        )
    with pytest.raises(ValueError, match="monotone nondecreasing"):
        _validate_relion_healpix_orders(
            [3, 4, 3],
            max_iter=3,
            init_healpix_order=3,
            max_healpix_order=7,
        )


def test_relion_healpix_order_oracle_holds_then_advances_sampling_state():
    state = RefinementState(
        healpix_order=3,
        adaptive_oversampling=1,
        translation_range=3.0,
        translation_step=1.0,
        max_healpix_order=7,
        auto_local_healpix_order=4,
        nr_iter_wo_resol_gain=5,
        nr_iter_wo_large_hidden_variable_changes=2,
    )

    held = _apply_relion_healpix_order_oracle(state, 3, iteration_number=8)
    assert held is state
    assert not held.do_local_search

    advanced = _apply_relion_healpix_order_oracle(held, 4, iteration_number=10)
    assert advanced.healpix_order == 4
    assert advanced.do_local_search
    assert advanced.nr_iter_wo_resol_gain == 0
    assert advanced.nr_iter_wo_large_hidden_variable_changes == 0

    with pytest.raises(ValueError, match="cannot coarsen the active state"):
        _apply_relion_healpix_order_oracle(advanced, 3, iteration_number=11)


@pytest.mark.parametrize("orders", [None, [3, 3, 4], (3, 3, 4)])
def test_sampling_validation_preserves_payloads_and_input_options(orders):
    current_sizes = [32, 40, 48]
    options = RefinementOptions(
        schedule=RefinementSchedule(max_iter=3, init_healpix_order=3),
        adaptive=AdaptiveOptions(relion_current_sizes=current_sizes, relion_healpix_orders=orders),
    )
    validated = with_validated_sampling_schedule(options)

    assert validated is not options
    assert validated.adaptive is not options.adaptive
    assert options.adaptive.relion_healpix_orders is orders
    assert validated.adaptive.relion_healpix_orders == (None if orders is None else (3, 3, 4))
    assert validated.adaptive.relion_current_sizes is current_sizes
    for group in ("schedule", "parity", "local_search", "k_class", "replay", "debug", "batching"):
        assert getattr(validated, group) is getattr(options, group)


@pytest.mark.parametrize(
    "sizes,orders,message",
    [
        ([], [4, 3], "relion_current_sizes must be non-empty"),
        ((), None, "relion_current_sizes must be non-empty"),
        (None, [3], "at least max_iter"),
        (None, [3, 4, 3], "monotone nondecreasing"),
        (None, [2, 3, 4], "cannot coarsen below init_healpix_order"),
        (None, [3, 4, 8], "exceeds max_healpix_order"),
    ],
)
def test_sampling_validation_runs_at_entry_and_preserves_error_precedence(sizes, orders, message):
    options = RefinementOptions(
        schedule=RefinementSchedule(max_iter=3, init_healpix_order=3),
        adaptive=AdaptiveOptions(relion_current_sizes=sizes, relion_healpix_orders=orders),
    )
    # Invalid schedules can still be constructed for later configuration/replay.
    with pytest.raises(ValueError, match=message):
        with_validated_sampling_schedule(options)
    assert options.adaptive.relion_current_sizes is sizes
    assert options.adaptive.relion_healpix_orders is orders


@pytest.mark.parametrize("use_defaults", [False, True])
def test_refinement_entry_passes_validated_options_to_loop(monkeypatch, use_defaults):
    from recovar.em.refinement import iteration_loop

    received = []
    sentinel = object()

    def run_loop(**kwargs):
        received.append(kwargs)
        return sentinel

    monkeypatch.setattr(iteration_loop, "_run_relion_iteration_loop", run_loop)
    options = None if use_defaults else RefinementOptions(
        schedule=RefinementSchedule(max_iter=3, init_healpix_order=3),
        adaptive=AdaptiveOptions(relion_healpix_orders=[3, 3, 4]),
    )
    inputs = [object() for _ in range(6)]
    assert iteration_loop.refine_single_volume(*inputs, options=options) is sentinel
    assert len(received) == 1
    validated = received[0]["options"]
    assert validated.adaptive.relion_healpix_orders == (None if use_defaults else (3, 3, 4))
    for name, value in zip(
        ("experiment_datasets", "init_volume", "init_noise_variance", "init_mean_variance", "rotations", "translations"),
        inputs,
    ):
        assert received[0][name] is value


def test_invalid_sampling_schedule_never_starts_refinement(monkeypatch):
    from recovar.em.refinement import iteration_loop

    def unexpected_loop(**kwargs):
        pytest.fail("invalid schedule reached refinement")

    monkeypatch.setattr(iteration_loop, "_run_relion_iteration_loop", unexpected_loop)
    options = RefinementOptions(adaptive=AdaptiveOptions(relion_current_sizes=[]))
    with pytest.raises(ValueError, match="relion_current_sizes must be non-empty"):
        iteration_loop.refine_single_volume(*([None] * 6), options=options)
