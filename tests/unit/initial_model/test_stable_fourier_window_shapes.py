"""Host contract for low-cardinality exact VDAM Fourier-window shapes."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_current_size,
)
from recovar.em.dense_single_volume.shape_buckets import pad_axis

pytestmark = pytest.mark.unit

_IMAGE_SHAPE = (128, 128)
_N_HALF = 128 * 65
_GF46_CURRENT_SIZES_THROUGH_80 = (
    30,
    32,
    34,
    38,
    44,
    46,
    48,
    50,
    56,
    60,
    62,
    66,
    68,
    70,
    76,
    78,
    84,
    86,
    88,
    90,
    98,
    104,
    106,
    110,
    114,
    116,
    122,
    126,
    128,
)


def _plan(current_size, *, enabled=True, reconstruction_current_size=None):
    return make_stable_fourier_window_shape_plan(
        _IMAGE_SHAPE,
        current_size,
        _N_HALF,
        enabled=enabled,
        reconstruction_current_size=reconstruction_current_size,
    )


def _relion_lane_tree_sum(storage, logical_count):
    """Host float32 replay of the fine scorer's 256-lane issue order."""

    block_size = 256
    lanes = np.zeros(block_size, dtype=np.float32)
    for pixel in range(int(logical_count)):
        lane = pixel % block_size
        lanes[lane] = np.float32(lanes[lane] + np.float32(storage[pixel]))
    width = block_size // 2
    while width:
        lanes[:width] = np.float32(lanes[:width] + lanes[width : 2 * width])
        width //= 2
    return lanes[0]


def test_stable_window_size_uses_eight_pixel_classes_and_isolates_full_box():
    assert stable_fourier_window_current_size(30, 128) == 32
    assert stable_fourier_window_current_size(34, 128) == 40
    assert stable_fourier_window_current_size(56, 128) == 56
    assert stable_fourier_window_current_size(70, 128) == 72
    assert stable_fourier_window_current_size(84, 128) == 88
    assert stable_fourier_window_current_size(122, 128) == 126
    assert stable_fourier_window_current_size(126, 128) == 126
    assert stable_fourier_window_current_size(128, 128) == 128


@pytest.mark.parametrize(
    ("current_size", "image_size", "quantum"),
    ((0, 128, 8), (31, 128, 8), (130, 128, 8), (32, 127, 8), (32, 128, 3)),
)
def test_stable_window_size_rejects_invalid_shapes(current_size, image_size, quantum):
    with pytest.raises(ValueError):
        stable_fourier_window_current_size(current_size, image_size, quantum=quantum)


def test_shape_policy_is_default_off():
    plan = _plan(70, enabled=False)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 70
    assert plan.logical_loop_bounds == (
        plan.physical_score_pixels,
        plan.physical_reconstruction_pixels,
        plan.physical_projection_pixels,
        plan.physical_rectangle_pixels,
    )


def test_adjacent_cutoffs_share_one_physical_signature_but_keep_logical_bounds():
    plans = [_plan(current_size) for current_size in (68, 70, 72)]

    assert {plan.physical_current_size for plan in plans} == {72}
    assert len({plan.physical_signature for plan in plans}) == 1
    assert len({plan.logical_loop_bounds for plan in plans}) == 3


def test_score_and_reconstruction_cutoffs_are_bucketed_independently():
    plan = _plan(70, reconstruction_current_size=84)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 72
    assert plan.logical_reconstruction_current_size == 84
    assert plan.physical_reconstruction_current_size == 88
    assert plan.physical_projection_pixels >= plan.logical_projection_pixels


@pytest.mark.parametrize(
    ("current_size", "physical_size", "logical_projection", "physical_projection"),
    (
        (56, 56, 1276, 1276),
        (70, 72, 1980, 2093),
        (84, 88, 2835, 3105),
        (128, 128, 8320, 8320),
    ),
)
def test_gf46_checkpoint_projection_capacity(
    current_size,
    physical_size,
    logical_projection,
    physical_projection,
):
    plan = _plan(current_size)

    assert plan.physical_current_size == physical_size
    assert plan.logical_projection_pixels == logical_projection
    assert plan.physical_projection_pixels == physical_projection


def test_gf46_shape_trajectory_collapses_from_29_signatures_to_14():
    plans = [_plan(current_size) for current_size in _GF46_CURRENT_SIZES_THROUGH_80]

    assert len({plan.logical_current_size for plan in plans}) == 29
    assert len({plan.physical_signature for plan in plans}) == 14
    assert sorted({plan.physical_current_size for plan in plans}) == [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        126,
        128,
    ]


def test_padding_appends_storage_without_changing_score_pixel_order():
    plan = _plan(70)
    logical_indices, _ = make_fourier_window_indices_np(_IMAGE_SHAPE, 70)
    padded_indices = pad_axis(
        logical_indices,
        0,
        plan.physical_score_pixels,
        value=0,
    )

    np.testing.assert_array_equal(
        padded_indices[: plan.logical_score_pixels],
        logical_indices,
    )
    assert np.all(padded_indices[plan.logical_score_pixels :] == 0)


def test_packed_capacity_reorders_interleaved_physical_support_behind_logical_prefix():
    plan = _plan(70)
    logical = plan.logical_spec.score_indices_np
    physical_sorted = plan.physical_spec.score_indices_np

    # Enlarging a radial window inserts new flat-grid indices throughout the
    # sorted support; slicing the physical spec would therefore change order.
    assert not np.array_equal(physical_sorted[: logical.size], logical)

    packed = plan.packed_indices_np("score")
    np.testing.assert_array_equal(packed[: logical.size], logical)
    np.testing.assert_array_equal(
        np.sort(packed[logical.size :]),
        np.setdiff1d(physical_sorted, logical, assume_unique=True),
    )


@pytest.mark.parametrize("name", ("score", "recon"))
def test_logical_projection_takes_stay_at_front_of_packed_capacity(name):
    plan = _plan(70, reconstruction_current_size=84)
    packed_projection = plan.packed_indices_np("projection")
    packed_support = plan.packed_indices_np(name)
    packed_take = plan.packed_projection_take_np(name)
    logical_count = getattr(plan, f"logical_{'reconstruction' if name == 'recon' else name}_pixels")

    np.testing.assert_array_equal(
        packed_projection[packed_take[:logical_count]],
        packed_support[:logical_count],
    )


def test_runtime_rectangle_stride_is_the_logical_not_physical_width():
    plan = _plan(70)
    first_pixel_on_second_logical_row = plan.logical_current_size // 2 + 1
    logical_half_width = plan.logical_current_size // 2 + 1
    physical_half_width = plan.physical_current_size // 2 + 1

    assert divmod(first_pixel_on_second_logical_row, logical_half_width) == (1, 0)
    assert divmod(first_pixel_on_second_logical_row, physical_half_width) == (0, 36)


def test_runtime_logical_bound_preserves_fine_lane_reduction_bitwise():
    plan = _plan(70)
    rng = np.random.default_rng(17)
    logical = rng.standard_normal(plan.logical_rectangle_pixels).astype(np.float32)
    padded = pad_axis(
        logical,
        0,
        plan.physical_rectangle_pixels,
        value=np.float32(1.25e5),
    )

    baseline = _relion_lane_tree_sum(logical, plan.logical_rectangle_pixels)
    stable_shape = _relion_lane_tree_sum(padded, plan.logical_rectangle_pixels)
    wrong_physical_bound = _relion_lane_tree_sum(padded, plan.physical_rectangle_pixels)

    assert baseline.view(np.uint32) == stable_shape.view(np.uint32)
    assert baseline.view(np.uint32) != wrong_physical_bound.view(np.uint32)


def test_runtime_logical_bound_preserves_bpref_issue_sequence():
    plan = _plan(84)
    logical_issues = np.arange(plan.logical_reconstruction_pixels, dtype=np.int32)
    padded_issues = pad_axis(
        logical_issues,
        0,
        plan.physical_reconstruction_pixels,
        value=-1,
    )

    np.testing.assert_array_equal(
        padded_issues[: plan.logical_reconstruction_pixels],
        logical_issues,
    )
    assert np.all(padded_issues[plan.logical_reconstruction_pixels :] == -1)
