import pytest

from recovar.em.dense_single_volume.helpers.convergence import RefinementState
from recovar.em.dense_single_volume.iteration_loop import (
    _apply_relion_healpix_order_oracle,
    _validate_relion_healpix_orders,
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
