"""The image-axis capacity budget must be bounded by the compact-pair diff2 gather.

On the 100k/256 K=4 fixture, padding the image axis at all produced
``RESOURCE_EXHAUSTED: 16.30 GiB`` at the compact-pair diff2 gather -- the
(batch, pair_width, pixels) complex tensor built twice before the kernel -- because the
capacity budget was derived only from the smaller per-rotation-row projection bytes.
"""

import numpy as np

from recovar.em.scoring.sparse_bucket_arrays import quantized_image_capacity
from recovar.em.sparse_pass2.sparse_pass2_adjoint import (
    _compact_pair_diff2_gather_bytes_per_image,
)


def test_footprint_is_two_complex_rows_per_pair_per_image():
    # 12288 pairs x 8320 half-spectrum pixels (128^2) x complex64, twice.
    assert _compact_pair_diff2_gather_bytes_per_image(12288, 8320, np.complex64) == 2 * 12288 * 8320 * 8
    assert _compact_pair_diff2_gather_bytes_per_image(1, 1, np.complex128) == 2 * 16
    assert _compact_pair_diff2_gather_bytes_per_image(0, 0, np.complex64) == 1


def test_real_size_bucket_refuses_to_pad_under_a_two_percent_budget():
    """The failing configuration: 256^2 half-spectrum pixels, wide pair axis, 2% of 80 GB."""
    n_score_pixels = 256 * (256 // 2 + 1)
    pair_width = 8192
    per_image = _compact_pair_diff2_gather_bytes_per_image(pair_width, n_score_pixels, np.complex64)
    budget_bytes = int(0.02 * 80 * 1024**3)
    images_that_fit = budget_bytes // per_image
    # a 20-image bucket cannot be padded to 32 within that budget
    assert quantized_image_capacity(20, max_images=images_that_fit) == 20
    # and the footprint of what it would have allocated really is in the 16 GiB class
    assert 32 * per_image > 16 * 1024**3


def test_small_fixture_bucket_still_pads():
    """128^2 pixels and a narrow pair axis leave room to pad under the same 2% budget.

    A 512-pair bucket at 128^2 already costs 68 MB per image, so only 25 images fit in
    1.6 GiB and padding 20 -> 32 is (correctly) refused; that is why only a handful of
    5k/128 buckets ever padded. A 64-pair bucket costs 8.5 MB per image and pads freely.
    """
    n_score_pixels = 128 * (128 // 2 + 1)
    budget = int(0.02 * 80 * 1024**3)
    wide = _compact_pair_diff2_gather_bytes_per_image(512, n_score_pixels, np.complex64)
    assert quantized_image_capacity(20, max_images=budget // wide) == 20
    narrow = _compact_pair_diff2_gather_bytes_per_image(64, n_score_pixels, np.complex64)
    assert budget // narrow >= 32
    assert quantized_image_capacity(20, max_images=budget // narrow) == 32
