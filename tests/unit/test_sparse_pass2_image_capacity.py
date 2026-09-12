"""Image-axis capacity quantization for fused K-class pass 2.

The capacity exists to make bucket shapes repeat so JAX stops tracing and lowering a
new program for nearly every bucket. It must never exceed the caller's byte budget:
that cap carries the gather, preparation and dense-M-step limits, so overshooting it
would trade a compile saving for an out-of-memory failure.
"""

import pytest

from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    IMAGE_CAPACITY_ENV,
    IMAGE_CAPACITY_FLOOR,
    image_capacity_enabled,
    quantized_image_capacity,
)


def test_capacity_is_off_unless_the_flag_is_set(monkeypatch):
    monkeypatch.delenv(IMAGE_CAPACITY_ENV, raising=False)
    assert image_capacity_enabled() is False
    monkeypatch.setenv(IMAGE_CAPACITY_ENV, "1")
    assert image_capacity_enabled() is True
    monkeypatch.setenv(IMAGE_CAPACITY_ENV, "0")
    assert image_capacity_enabled() is False


def test_capacity_flag_rejects_anything_but_zero_or_one(monkeypatch):
    monkeypatch.setenv(IMAGE_CAPACITY_ENV, "yes")
    with pytest.raises(ValueError, match=IMAGE_CAPACITY_ENV):
        image_capacity_enabled()


@pytest.mark.parametrize(
    ("n_images", "expected"),
    [(1, 16), (5, 16), (16, 16), (17, 32), (33, 64), (64, 64), (65, 128), (280, 512), (489, 512)],
)
def test_small_counts_round_up_to_a_power_of_two_at_or_above_the_floor(n_images, expected):
    assert quantized_image_capacity(n_images) == expected


def test_empty_bucket_stays_empty():
    assert quantized_image_capacity(0) == 0


@pytest.mark.parametrize("n_images", list(range(1, 130)) + [200, 280, 489, 1000])
def test_capacity_never_loses_images_and_never_exceeds_the_cap(n_images):
    for cap in (1, 2, 7, 16, 17, 64, 280, 512, 4096):
        capacity = quantized_image_capacity(n_images, max_images=cap)
        assert capacity >= n_images, (n_images, cap, capacity)
        # Either the capacity honours the byte budget, or it declined to pad at all.
        assert capacity <= cap or capacity == n_images, (n_images, cap, capacity)


def test_capacity_declines_to_pad_when_no_power_of_two_fits_the_budget():
    # 20 images with a 24-image budget: the next power of two is 32, which overshoots.
    assert quantized_image_capacity(20, max_images=24) == 20


def test_capacity_pads_when_the_power_of_two_fits_the_budget():
    assert quantized_image_capacity(20, max_images=32) == 32


def test_buckets_at_or_above_the_budget_keep_their_own_shape():
    assert quantized_image_capacity(280, max_images=280) == 280
    assert quantized_image_capacity(500, max_images=280) == 500


def test_the_quantized_axis_takes_only_a_handful_of_values():
    """The whole point: many distinct image counts collapse to few capacities."""
    counts = [1, 2, 3, 4, 5, 12, 16, 17, 19, 33, 47, 58, 73, 79, 121, 126, 133, 213, 280, 489]
    assert len(set(counts)) == 20
    assert set(quantized_image_capacity(c) for c in counts) == {16, 32, 64, 128, 256, 512}


def test_floor_is_respected_and_configurable():
    assert quantized_image_capacity(3, floor=IMAGE_CAPACITY_FLOOR) == IMAGE_CAPACITY_FLOOR
    assert quantized_image_capacity(3, floor=4) == 4
    assert quantized_image_capacity(9, floor=4) == 16


def test_growth_bound_defaults_to_two_and_reads_the_env(monkeypatch):
    from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
        DEFAULT_IMAGE_CAPACITY_MAX_GROWTH,
        IMAGE_CAPACITY_MAX_GROWTH_ENV,
        image_capacity_max_growth,
    )

    monkeypatch.delenv(IMAGE_CAPACITY_MAX_GROWTH_ENV, raising=False)
    assert image_capacity_max_growth() == DEFAULT_IMAGE_CAPACITY_MAX_GROWTH == 2.0
    monkeypatch.setenv(IMAGE_CAPACITY_MAX_GROWTH_ENV, "1.5")
    assert image_capacity_max_growth() == 1.5
    monkeypatch.setenv(IMAGE_CAPACITY_MAX_GROWTH_ENV, "0.5")
    with pytest.raises(ValueError, match="at least 1.0"):
        image_capacity_max_growth()


@pytest.mark.parametrize("n_images", list(range(16, 600)))
def test_power_of_two_rounding_never_exceeds_the_default_growth_bound(n_images):
    """Above the floor, rounding up to a power of two is at most a doubling."""
    capacity = quantized_image_capacity(n_images, max_growth=2.0)
    assert capacity <= 2 * n_images
    assert capacity >= n_images


def test_growth_bound_of_one_disables_padding_above_the_floor():
    assert quantized_image_capacity(300, max_growth=1.0) == 300
    # At or below the floor the absolute row count stays tiny, so the floor still applies.
    assert quantized_image_capacity(3, max_growth=1.0) == IMAGE_CAPACITY_FLOOR


@pytest.mark.parametrize("n_images", [17, 33, 65, 130, 260, 400])
def test_a_tight_growth_bound_refuses_rather_than_half_pads(n_images):
    """When the bound forbids the next power of two, keep the exact count."""
    capacity = quantized_image_capacity(n_images, max_growth=1.01)
    assert capacity == n_images


def test_growth_bound_and_byte_budget_both_apply():
    # Growth allows 512, but the byte budget does not.
    assert quantized_image_capacity(300, max_images=320, max_growth=2.0) == 300
    # Both allow it.
    assert quantized_image_capacity(300, max_images=512, max_growth=2.0) == 512
