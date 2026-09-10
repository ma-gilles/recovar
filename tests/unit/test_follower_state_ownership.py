"""Follower corrections publish through the setup owner, without shadow state."""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume import relion_worker_scale as scale

pytestmark = pytest.mark.unit


@pytest.fixture
def correction_inputs():
    state = scale.make_relion_follower_scale_state(
        n_followers=2, group_counts=[1, 1], n_optics_groups=1, initial_group_scales=[0.8, 1.2]
    )
    setup = SimpleNamespace(
        follower_scale_state=state,
        follower_owners_per_half=[np.array([0, 1]), np.array([1, 0])],
    )
    halves = SimpleNamespace(
        group_ids=[np.array([0, 1]), np.array([1, 0])],
        scale_corrections=[None, None],
        image_corrections=[None, None],
    )
    noise = SimpleNamespace(
        wsum_scale_correction_xa=np.array([[2.0, 6.0], [4.0, 8.0]]),
        wsum_scale_correction_aa=np.ones((2, 2)),
    )
    norm = SimpleNamespace(
        norm_corrections_per_half=[np.array([2.0, 0.0]), np.array([4.0, 2.0])],
        avg_norm_correction_per_half=[4.0, 2.0],
    )
    return setup, dict(
        noise_stats_per_half=[noise, None],
        norm_scale_update=norm,
        relion_half_inputs=halves,
        logger=SimpleNamespace(info=lambda *args: None),
    )


@pytest.mark.parametrize("firstiter", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_updates_are_visible_through_owner(correction_inputs, firstiter, dtype):
    setup, kwargs = correction_inputs
    initial = setup.follower_scale_state
    if firstiter:
        kwargs["noise_stats_per_half"][0].wsum_scale_correction_xa = None
        kwargs["noise_stats_per_half"][0].wsum_scale_correction_aa = None
    for _ in range(2):
        diagnostic = scale._update_relion_follower_corrections(
            setup,
            **kwargs,
            relion_firstiter_cc_this_iter=firstiter,
            dtype=dtype,
        )
        assert isinstance(diagnostic, list) and len(diagnostic) == 2
        assert diagnostic[1] is None
        state = setup.follower_scale_state
        assert (state is initial) == firstiter
        np.testing.assert_array_equal(diagnostic[0], state.scales[0])
        halves = kwargs["relion_half_inputs"]
        for half, factor in enumerate(([2.0, 1.0], [0.5, 1.0])):
            expected = state.scales[setup.follower_owners_per_half[half], halves.group_ids[half]].astype(dtype)
            np.testing.assert_array_equal(halves.scale_corrections[half], expected)
            np.testing.assert_array_equal(halves.image_corrections[half], (expected * factor).astype(dtype))
            assert halves.scale_corrections[half].dtype == dtype
            assert halves.image_corrections[half].dtype == dtype


def test_missing_statistics_leave_owner_and_corrections_untouched(correction_inputs):
    setup, kwargs = correction_inputs
    original = setup.follower_scale_state
    kwargs["noise_stats_per_half"][0].wsum_scale_correction_xa = None
    with pytest.raises(RuntimeError, match="requires expanded XA/AA"):
        scale._update_relion_follower_corrections(
            setup,
            **kwargs,
            relion_firstiter_cc_this_iter=False,
            dtype=np.float32,
        )
    assert setup.follower_scale_state is original
    assert kwargs["relion_half_inputs"].scale_corrections == [None, None]
    assert kwargs["relion_half_inputs"].image_corrections == [None, None]
