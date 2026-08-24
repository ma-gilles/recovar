"""Exact reduction tests for RELION CUDA-style fine Gaussian scores."""

import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_relion_noise_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _RELION_CUDA_FINE_REF3D_BLOCK_SIZE,
    _RELION_CUDA_POWERCLASS_BLOCK_SIZE,
    _relion_cuda_fine_diff2_min,
    _relion_cuda_fine_diff2_sum,
    _relion_cuda_fine_diff2_to_scores,
    _relion_cuda_fine_global_diff2_min,
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_cuda_fine_log_evidence_offset,
    _relion_cuda_fine_pixel_weights,
    _relion_cuda_fine_tree_sum,
    _relion_cuda_powerclass_highres_norm_units,
    _relion_cuda_powerclass_highres_xi2_half,
    _score_pass2_bucket_relion_gpu_diff2,
    _score_pass2_bucket_relion_gpu_diff2_from_raw,
    _score_pass2_bucket_relion_gpu_diff2_raw,
    _score_pass2_bucket_relion_gpu_diff2_single_cached,
    _score_pass2_bucket_relion_gpu_diff2_single_cached_raw,
    _score_pass2_pairs_relion_gpu_diff2,
    _score_pass2_pairs_relion_gpu_diff2_raw,
)
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.local_em_engine import (
    _relion_exact_fine_full_to_compact_lookup,
)
from recovar.em.dense_single_volume.local_big_jit import (
    _validate_relion_exact_fine_diff2_preconditions,
)

pytestmark = pytest.mark.unit


def _numpy_cuda_fine_tree(values):
    values = np.asarray(values, dtype=np.float32)
    lanes = np.zeros(values.shape[:-1] + (_RELION_CUDA_FINE_REF3D_BLOCK_SIZE,), dtype=np.float32)
    for pixel in range(values.shape[-1]):
        lane = pixel % _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
        lanes[..., lane] = np.float32(lanes[..., lane] + values[..., pixel])
    width = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE // 2
    while width:
        lanes[..., :width] = np.float32(lanes[..., :width] + lanes[..., width : 2 * width])
        width //= 2
    return lanes[..., 0]


def _numpy_cuda_fine_diff2(reference, shifted, weight):
    reference = np.asarray(reference)
    shifted = np.asarray(shifted)
    weight = np.asarray(weight, dtype=np.float32)
    diff_real = np.float32(reference.real - shifted.real)
    diff_imag = np.float32(reference.imag - shifted.imag)
    square_sum = np.float32(
        np.float32(diff_real * diff_real) + np.float32(diff_imag * diff_imag)
    )
    terms = np.float32(np.float32(square_sum * np.float32(0.5)) * weight)
    return _numpy_cuda_fine_tree(terms)


def _numpy_cuda_powerclass_highres_half(centered_image, current_size):
    centered_image = np.asarray(centered_image, dtype=np.complex64)
    batch, height, half_width = centered_image.shape
    relion_image = np.roll(centered_image, -(height // 2), axis=1)
    relion_image = np.complex64(relion_image / np.float32(height * height))
    lanes = np.zeros(
        (batch, (height * half_width + _RELION_CUDA_POWERCLASS_BLOCK_SIZE - 1)
         // _RELION_CUDA_POWERCLASS_BLOCK_SIZE, _RELION_CUDA_POWERCLASS_BLOCK_SIZE),
        dtype=np.float32,
    )
    for voxel in range(height * half_width):
        x = voxel % half_width
        row = voxel // half_width
        y = row if row < half_width else row - height
        shell = int(np.rint(np.sqrt(np.float32(x * x + y * y))))
        if shell <= 0 or shell >= half_width or (x == 0 and y < 0):
            continue
        if shell >= current_size // 2 + 1:
            value = relion_image.reshape(batch, -1)[:, voxel]
            power = np.float32(
                np.float32(value.real * value.real)
                + np.float32(value.imag * value.imag)
            )
            lanes[:, voxel // _RELION_CUDA_POWERCLASS_BLOCK_SIZE, voxel % _RELION_CUDA_POWERCLASS_BLOCK_SIZE] = power
    width = _RELION_CUDA_POWERCLASS_BLOCK_SIZE // 2
    while width:
        lanes[..., :width] = np.float32(lanes[..., :width] + lanes[..., width : 2 * width])
        width //= 2
    total = np.zeros((batch,), dtype=np.float32)
    for block_sum in lanes[..., 0].T:
        total = np.float32(total + block_sum)
    return np.float32(total * np.float32(0.5))


def test_relion_cuda_fine_tree_matches_256_lane_pass_and_tree_bitwise():
    # Alternating scales make sequential per-lane accumulation distinguishable
    # from a flat reduction while retaining deterministic float32 operands.
    index = np.arange(773, dtype=np.float32)
    values = np.stack(
        [
            np.where((index.astype(np.int32) & 1) == 0, index * 1.0e4, -index * 1.0e-3),
            np.sin(index).astype(np.float32) * np.float32(1.0e3),
        ]
    ).astype(np.float32)

    expected = _numpy_cuda_fine_tree(values)
    actual = np.asarray(_relion_cuda_fine_tree_sum(jnp.asarray(values)))

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_relion_cuda_fine_diff2_casts_complex128_operands_to_xfloat():
    rng = np.random.default_rng(365)
    reference64 = (
        rng.normal(size=(2, 259)) + 1j * rng.normal(size=(2, 259))
    ).astype(np.complex64)
    shifted64 = (
        rng.normal(size=(3, 259)) + 1j * rng.normal(size=(3, 259))
    ).astype(np.complex64)
    weight = rng.uniform(size=(1, 259)).astype(np.float32)

    expected = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference64[:, None, :]),
            jnp.asarray(shifted64[None, :, :]),
            jnp.asarray(weight),
        )
    )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference64.astype(np.complex128)[:, None, :]),
            jnp.asarray(shifted64.astype(np.complex128)[None, :, :]),
            jnp.asarray(weight, dtype=jnp.float64),
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_relion_cuda_fine_pixel_weight_casts_operands_before_multiply():
    corr = np.asarray([75351.31086994553], dtype=np.float64)
    half_weight = np.asarray([53814.33132654639], dtype=np.float64)
    expected = np.float32(corr) * np.float32(half_weight)
    wrong_float64_first = np.float32(corr * half_weight)

    actual = np.asarray(
        _relion_cuda_fine_pixel_weights(jnp.asarray(corr), jnp.asarray(half_weight))
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert not np.array_equal(actual, wrong_float64_first)


def test_relion_cuda_powerclass_highres_matches_128_lane_block_trees_bitwise():
    rng = np.random.default_rng(2297)
    height = 32
    centered = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    current_size = 14
    expected = _numpy_cuda_powerclass_highres_half(centered, current_size)
    actual = np.asarray(
        _relion_cuda_powerclass_highres_xi2_half(
            jnp.asarray(centered.reshape(2, -1)),
            image_shape=(height, height),
            current_size=current_size,
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_relion_cuda_powerclass_norm_units_preserve_divide_before_square():
    rng = np.random.default_rng(4021)
    height = 32
    centered = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    current_size = 14
    expected_half = _numpy_cuda_powerclass_highres_half(centered, current_size)
    expected = np.float32(
        np.float32(expected_half * np.float32(2.0))
        * np.float32((height * height) ** 2)
    )

    actual = np.asarray(
        _relion_cuda_powerclass_highres_norm_units(
            jnp.asarray(centered.reshape(2, -1)),
            image_shape=(height, height),
            current_size=current_size,
        )
    )
    processed = jnp.asarray(centered.reshape(2, -1))
    shells = jnp.asarray(make_relion_noise_shell_indices_half((height, height)))
    high_shell = (shells >= 0) & (shells < height // 2 + 1) & (shells > current_size // 2)
    generic_square_first = np.asarray(
        jnp.sum(jnp.where(high_shell[None, :], jnp.abs(processed) ** 2, 0.0), axis=-1).astype(jnp.float32)
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert not np.array_equal(actual, generic_square_first)


def test_relion_cuda_fine_raw_routes_add_powerclass_tail_once_per_hypothesis():
    shifted = jnp.zeros((1, 2, 5), dtype=jnp.complex64)
    projection = jnp.zeros((1, 3, 5), dtype=jnp.complex64)
    corr = jnp.ones((1, 5), dtype=jnp.float32)
    half_weight = jnp.ones((5,), dtype=jnp.float32)
    highres = jnp.asarray([0.02080046385526657], dtype=jnp.float32)
    dense = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2_raw(
            shifted, corr, projection, half_weight, None, highres
        )
    )
    cached = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
            shifted[0], corr[0], projection[0], half_weight, None, highres[0]
        )
    )
    local_rotation_row = jnp.asarray([[0, 2, 1]], dtype=jnp.int32)
    translation_idx = jnp.asarray([[1, 0, 1]], dtype=jnp.int32)
    pair_mask = jnp.ones((1, 3), dtype=bool)
    pairs = np.asarray(
        _score_pass2_pairs_relion_gpu_diff2_raw(
            shifted,
            corr,
            projection,
            half_weight,
            local_rotation_row,
            translation_idx,
            pair_mask,
            None,
            highres,
        )
    )

    expected = np.float32(np.asarray(highres)[0])
    np.testing.assert_array_equal(dense, np.full((1, 3, 2), expected, dtype=np.float32))
    np.testing.assert_array_equal(cached, dense[0])
    np.testing.assert_array_equal(pairs, np.full((1, 3), expected, dtype=np.float32))


def test_relion_cuda_fine_retained_raw_conversion_matches_recompute_bitwise():
    rng = np.random.default_rng(12131)
    shifted = jnp.asarray(
        (rng.normal(size=(2, 4, 17)) + 1j * rng.normal(size=(2, 4, 17))).astype(np.complex64)
    )
    projection = jnp.asarray(
        (rng.normal(size=(2, 5, 17)) + 1j * rng.normal(size=(2, 5, 17))).astype(np.complex64)
    )
    corr = jnp.asarray(rng.uniform(0.2, 2.0, size=(2, 17)), dtype=jnp.float32)
    half_weight = jnp.asarray(rng.choice([1.0, 2.0], size=17), dtype=jnp.float32)
    rotation_prior = jnp.asarray(rng.normal(size=(2, 5)), dtype=jnp.float32)
    translation_prior = jnp.asarray(rng.normal(size=(2, 4)), dtype=jnp.float32)
    candidate_mask = jnp.asarray(rng.random(size=(2, 5, 4)) > 0.2)

    raw = _score_pass2_bucket_relion_gpu_diff2_raw(
        shifted,
        corr,
        projection,
        half_weight,
    )
    common_min = _relion_cuda_fine_diff2_min(raw, candidate_mask)
    retained_scores = _score_pass2_bucket_relion_gpu_diff2_from_raw(
        raw,
        rotation_prior,
        translation_prior,
        candidate_mask,
        common_min,
    )
    recomputed_scores = _score_pass2_bucket_relion_gpu_diff2(
        shifted,
        corr,
        projection,
        half_weight,
        rotation_prior,
        translation_prior,
        candidate_mask,
        min_diff2=common_min,
    )

    np.testing.assert_array_equal(np.asarray(retained_scores), np.asarray(recomputed_scores))


def test_relion_cuda_fine_conversion_uses_common_min_and_source_operation_order():
    # Captured case-20 particle 469 values: RELION's candidates differ by one
    # ULP in positive diff2. The large common min must be inserted at the same
    # point as cuda_kernel_exponentiate_weights_fine.
    diff2 = np.asarray([[1214.7265625, 1214.7264404296875]], dtype=np.float32)
    rotation_prior = np.asarray([[-7.326465606689453, -7.326465606689453]], dtype=np.float64)
    translation_prior = np.asarray([[-7.363075256347656, -7.363075256347656]], dtype=np.float64)
    common_min = np.asarray([1214.0980224609375], dtype=np.float32)
    mask = np.ones_like(diff2, dtype=bool)

    actual = np.asarray(
        _relion_cuda_fine_diff2_to_scores(
            jnp.asarray(diff2),
            jnp.asarray(rotation_prior),
            jnp.asarray(translation_prior),
            jnp.asarray(mask),
            min_diff2=jnp.asarray(common_min),
        )
    )
    expected = np.float32(np.float32(np.float32(rotation_prior) + np.float32(translation_prior)) + common_min[:, None])
    expected = np.float32(expected - diff2)
    naive = np.float32(np.float32(-diff2 + np.float32(rotation_prior)) + np.float32(translation_prior))

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual[0, 1] > actual[0, 0]
    assert not np.array_equal(actual, naive)


def test_relion_cuda_fine_conversion_rejects_diff2_below_external_minimum():
    diff2 = jnp.asarray([[100.0, 100.00001, 100.00002]], dtype=jnp.float32)
    mask = jnp.ones_like(diff2, dtype=bool)
    scores = np.asarray(
        _relion_cuda_fine_diff2_to_scores(
            diff2,
            jnp.zeros_like(diff2),
            jnp.zeros_like(diff2),
            mask,
            min_diff2=jnp.asarray([100.00001], dtype=jnp.float32),
        )
    )

    assert np.isneginf(scores[0, 0])
    external_min = np.float32(100.00001)
    np.testing.assert_array_equal(
        scores[0, 1:],
        np.float32(external_min - np.asarray(diff2, dtype=np.float32)[0, 1:]),
    )


def test_relion_cuda_fine_common_min_spans_chunks_and_classes_and_sets_evidence_offset():
    raw_partitions = (
        np.asarray([[[1000.25, 999.75]], [[1200.0, 1199.5]]], dtype=np.float32),
        np.asarray([[[999.5]], [[1201.0]]], dtype=np.float32),
        np.asarray([[[1002.0, 1003.0]], [[1198.75, 1199.0]]], dtype=np.float32),
    )
    masks = tuple(np.ones(raw.shape, dtype=bool) for raw in raw_partitions)
    masks[1][1, 0, 0] = False
    common_min = np.asarray(
        _relion_cuda_fine_global_diff2_min(
            tuple(jnp.asarray(raw) for raw in raw_partitions),
            tuple(jnp.asarray(mask) for mask in masks),
        )
    )
    np.testing.assert_array_equal(common_min, np.asarray([999.5, 1198.75], dtype=np.float32))

    converted = []
    for raw, mask in zip(raw_partitions, masks, strict=True):
        converted.append(
            np.asarray(
                _relion_cuda_fine_diff2_to_scores(
                    jnp.asarray(raw),
                    jnp.zeros_like(jnp.asarray(raw)),
                    jnp.zeros_like(jnp.asarray(raw)),
                    jnp.asarray(mask),
                    min_diff2=jnp.asarray(common_min),
                )
            )
        )
    merged_scores = np.concatenate([score.reshape(2, -1) for score in converted], axis=1)
    merged_mask = np.concatenate([mask.reshape(2, -1) for mask in masks], axis=1)
    expected = np.float32(common_min[:, None] - np.concatenate(
        [raw.reshape(2, -1) for raw in raw_partitions], axis=1
    ))
    expected = np.where(merged_mask, expected, -np.inf)
    np.testing.assert_array_equal(merged_scores, expected)

    finite_scores = np.where(merged_mask, merged_scores, -np.inf).astype(np.float64)
    max_score = np.max(finite_scores, axis=1)
    centered_log_z = max_score + np.log(np.sum(np.exp(finite_scores - max_score[:, None]), axis=1))
    absolute_log_evidence = centered_log_z + np.asarray(
        _relion_cuda_fine_log_evidence_offset(jnp.asarray(common_min)), dtype=np.float64
    )
    np.testing.assert_allclose(
        absolute_log_evidence,
        centered_log_z - common_min.astype(np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_relion_cuda_fine_common_min_ignores_invalid_partitions_and_nonfinite_padding():
    raw_partitions = (
        jnp.asarray(
            [
                [[np.nan, np.inf]],
                [[np.nan, 50.0]],
                [[np.nan, np.inf]],
            ],
            dtype=jnp.float32,
        ),
        jnp.asarray(
            [
                [[9.0, 7.0]],
                [[11.0, 13.0]],
                [[np.nan, np.inf]],
            ],
            dtype=jnp.float32,
        ),
    )
    masks = (
        jnp.asarray(
            [
                [[False, False]],
                [[True, False]],
                [[False, False]],
            ]
        ),
        jnp.asarray(
            [
                [[True, True]],
                [[True, True]],
                [[True, True]],
            ]
        ),
    )

    # Image zero has a completely invalid first class: it must not inject the
    # local all-invalid sentinel zero ahead of the valid class minimum seven.
    # Image one ignores both the masked finite 50 and a candidate NaN. Image
    # two is globally invalid, for which zero is only an inert output sentinel.
    common_min = np.asarray(_relion_cuda_fine_global_diff2_min(raw_partitions, masks))
    np.testing.assert_array_equal(common_min, np.asarray([7.0, 11.0, 0.0], dtype=np.float32))

    local_min = np.asarray(_relion_cuda_fine_diff2_min(raw_partitions[0], masks[0]))
    np.testing.assert_array_equal(local_min, np.zeros(3, dtype=np.float32))

    converted = [
        np.asarray(
            _relion_cuda_fine_diff2_to_scores(
                raw,
                jnp.zeros_like(raw),
                jnp.zeros_like(raw),
                mask,
                min_diff2=jnp.asarray(common_min),
            )
        )
        for raw, mask in zip(raw_partitions, masks, strict=True)
    ]
    assert np.all(np.isneginf(converted[0]))
    np.testing.assert_array_equal(converted[1][0, 0], np.asarray([-2.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(converted[1][1, 0], np.asarray([0.0, -2.0], dtype=np.float32))
    assert np.all(np.isneginf(converted[1][2]))


def test_relion_cuda_fine_host_staged_common_min_serializes_raw_device_uploads(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    original_partition_min = bucketed_mod._relion_cuda_fine_partition_diff2_min_or_inf
    raw_device_refs = []
    max_prior_raw_uploads_alive = 0

    def tracked_partition_min(raw_device, mask_device):
        nonlocal max_prior_raw_uploads_alive
        gc.collect()
        max_prior_raw_uploads_alive = max(
            max_prior_raw_uploads_alive,
            sum(ref() is not None for ref in raw_device_refs),
        )
        result = original_partition_min(raw_device, mask_device)
        jax.block_until_ready(result)
        raw_device_refs.append(weakref.ref(raw_device))
        return result

    monkeypatch.setattr(
        bucketed_mod,
        "_relion_cuda_fine_partition_diff2_min_or_inf",
        tracked_partition_min,
    )
    host_raw = tuple(
        np.full((2, 3, 4), 20.0 - index, dtype=np.float32)
        for index in range(4)
    )
    host_masks = tuple(np.ones(raw.shape, dtype=bool) for raw in host_raw)
    minimum = np.asarray(
        bucketed_mod._relion_cuda_fine_global_diff2_min(host_raw, host_masks)
    )

    np.testing.assert_array_equal(minimum, np.asarray([17.0, 17.0], dtype=np.float32))
    assert len(raw_device_refs) == 4
    assert max_prior_raw_uploads_alive <= 1


def test_relion_cuda_fine_raw_scorer_hlo_avoids_hypothesis_pixel_temporary():
    batch, translations, rotations, pixels = 2, 7, 5, 517
    args = (
        jnp.zeros((batch, translations, pixels), dtype=jnp.complex64),
        jnp.ones((batch, pixels), dtype=jnp.float32),
        jnp.zeros((batch, rotations, pixels), dtype=jnp.complex64),
        jnp.ones((pixels,), dtype=jnp.float32),
    )
    lowered = _score_pass2_bucket_relion_gpu_diff2_raw.lower(*args)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert "stablehlo.dot_general" not in stablehlo
    assert stablehlo.count("stablehlo.while") == 1

    memory = lowered.compile().memory_analysis()
    full_hypothesis_pixel_bytes = batch * rotations * translations * pixels * 4
    assert memory.temp_size_in_bytes < full_hypothesis_pixel_bytes


def test_relion_cuda_fine_diff2_matches_reference_without_pixel_tensor():
    rng = np.random.default_rng(18)
    reference = (
        rng.normal(size=(2, 3, 1, 517)) + 1j * rng.normal(size=(2, 3, 1, 517))
    ).astype(np.complex64)
    shifted = (
        rng.normal(size=(2, 1, 4, 517)) + 1j * rng.normal(size=(2, 1, 4, 517))
    ).astype(np.complex64)
    weight = rng.uniform(0.0, 3.0, size=(2, 1, 1, 517)).astype(np.float32)

    expected = np.empty((2, 3, 4), dtype=np.float32)
    for batch in range(2):
        for rotation in range(3):
            for translation in range(4):
                expected[batch, rotation, translation] = _numpy_cuda_fine_diff2(
                    reference[batch, rotation, 0],
                    shifted[batch, 0, translation],
                    weight[batch, 0, 0],
                )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference), jnp.asarray(shifted), jnp.asarray(weight)
        )
    )

    # XLA may contract the pointwise square/add/multiply into FMAs whereas
    # NumPy evaluates each operation separately. The lane pass and reduction
    # are exact (tested above); pointwise contraction is bounded to one ULP.
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
    assert actual.shape == (2, 3, 4)
    assert actual.dtype == np.float32


def test_relion_cuda_fine_diff2_preserves_full_grid_zero_gap_lane_topology():
    full_size = 130
    compact_indices = np.asarray([0, 1, 129], dtype=np.int32)
    lookup = np.full(full_size, -1, dtype=np.int32)
    lookup[compact_indices] = np.arange(compact_indices.size, dtype=np.int32)
    # The two small terms combine before the large term only when their full
    # pixel lanes (1 and 129) are retained. Compacting them into lanes 1 and 2
    # loses both to float32 rounding against the large lane-0 term.
    terms = np.asarray([1.0e8, 4.0, 4.0], dtype=np.float32)
    reference = np.sqrt(np.float32(2.0) * terms).astype(np.complex64)
    shifted = np.zeros(compact_indices.size, dtype=np.complex64)
    weight = np.ones(compact_indices.size, dtype=np.float32)
    full_reference = np.zeros(full_size, dtype=np.complex64)
    full_shifted = np.zeros(full_size, dtype=np.complex64)
    full_weight = np.zeros(full_size, dtype=np.float32)
    full_reference[compact_indices] = reference
    full_shifted[compact_indices] = shifted
    full_weight[compact_indices] = weight

    expected = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(full_reference),
            jnp.asarray(full_shifted),
            jnp.asarray(full_weight),
        )
    )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(lookup),
        )
    )
    wrong_compact_consecutive = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual != wrong_compact_consecutive


def test_case20_current_grid_lookup_has_relion_56_by_29_topology():
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )

    compact_indices, count = make_fourier_window_indices_np((256, 256), 56)
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        (256, 256), 56, compact_indices
    )

    assert count == 1275
    assert lookup.shape == (56 * 29,)
    assert np.count_nonzero(lookup >= 0) == count
    np.testing.assert_array_equal(np.sort(lookup[lookup >= 0]), np.arange(count, dtype=np.int32))


def test_full_size_lookup_is_a_bijection_of_the_complete_half_spectrum():
    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window_spec = make_fourier_window_spec(image_shape, image_shape[0], n_half)

    assert window_spec.use_window is False
    lookup = _relion_exact_fine_full_to_compact_lookup(
        image_shape,
        image_shape[0],
        n_half,
        window_spec,
    )

    assert lookup.shape == (n_half,)
    np.testing.assert_array_equal(np.sort(lookup), np.arange(n_half, dtype=np.int32))

    # At full size ``use_window`` is false, but the identity lookup above is a
    # valid exact-fine representation. The big-JIT gate must therefore depend
    # only on the exact operand/preprocessing contract.
    _validate_relion_exact_fine_diff2_preconditions(
        relion_exact_fine_diff2=True,
        relion_exact_bpref_operands=True,
        use_relion_cuda_preprocess=True,
    )


def test_exact_fine_big_jit_still_rejects_missing_exact_preprocessing():
    with pytest.raises(ValueError, match="exact operands and CUDA preprocessing"):
        _validate_relion_exact_fine_diff2_preconditions(
            relion_exact_fine_diff2=True,
            relion_exact_bpref_operands=True,
            use_relion_cuda_preprocess=False,
        )


def test_dense_cached_and_compact_gaussian_routes_use_same_exact_scores():
    rng = np.random.default_rng(469)
    batch, rotations, translations, pixels = 2, 4, 3, 513
    shifted = (
        rng.normal(size=(batch, translations, pixels))
        + 1j * rng.normal(size=(batch, translations, pixels))
    ).astype(np.complex64)
    projection = (
        rng.normal(size=(batch, rotations, pixels))
        + 1j * rng.normal(size=(batch, rotations, pixels))
    ).astype(np.complex64)
    corr = rng.uniform(0.01, 2.0, size=(batch, pixels)).astype(np.float32)
    half_weight = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=pixels)
    # Global x64 is enabled in RECOVAR; scorers must nevertheless follow
    # RELION GPU XFLOAT for both orientation and translation log-priors.
    rotation_prior = rng.normal(scale=0.1, size=(batch, rotations)).astype(np.float64)
    translation_prior = rng.normal(scale=0.1, size=(batch, translations)).astype(np.float64)
    candidate_mask = np.ones((batch, rotations, translations), dtype=bool)
    candidate_mask[0, 2, 1] = False

    dense_raw = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2_raw(
            jnp.asarray(shifted),
            jnp.asarray(corr),
            jnp.asarray(projection),
            jnp.asarray(half_weight),
        )
    )

    dense = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2(
            jnp.asarray(shifted),
            jnp.asarray(corr),
            jnp.asarray(projection),
            jnp.asarray(half_weight),
            jnp.asarray(rotation_prior),
            jnp.asarray(translation_prior),
            jnp.asarray(candidate_mask),
        )
    )
    cached = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2_single_cached(
            jnp.asarray(shifted[0]),
            jnp.asarray(corr[0]),
            jnp.asarray(projection[0]),
            jnp.asarray(half_weight),
            jnp.asarray(rotation_prior[0]),
            jnp.asarray(translation_prior[0]),
            jnp.asarray(candidate_mask[0]),
        )
    )
    cached_raw = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
            jnp.asarray(shifted[0]),
            jnp.asarray(corr[0]),
            jnp.asarray(projection[0]),
            jnp.asarray(half_weight),
        )
    )
    np.testing.assert_array_equal(cached_raw, dense_raw[0])
    np.testing.assert_array_equal(cached, dense[0])

    pair_count = max(int(np.count_nonzero(row)) for row in candidate_mask)
    local_rotation_row = np.zeros((batch, pair_count), dtype=np.int32)
    translation_idx = np.zeros((batch, pair_count), dtype=np.int32)
    pair_mask = np.zeros((batch, pair_count), dtype=bool)
    for batch_index in range(batch):
        rows_valid, translations_valid = np.nonzero(candidate_mask[batch_index])
        count = rows_valid.size
        local_rotation_row[batch_index, :count] = rows_valid
        translation_idx[batch_index, :count] = translations_valid
        pair_mask[batch_index, :count] = True
    pair_rotation_prior = np.take_along_axis(rotation_prior, local_rotation_row, axis=1)
    compact = np.asarray(
        _score_pass2_pairs_relion_gpu_diff2(
            jnp.asarray(shifted),
            jnp.asarray(corr),
            jnp.asarray(projection),
            jnp.asarray(half_weight),
            jnp.asarray(pair_rotation_prior),
            jnp.asarray(translation_prior),
            jnp.asarray(local_rotation_row),
            jnp.asarray(translation_idx),
            jnp.asarray(pair_mask),
        )
    )
    compact_raw = np.asarray(
        _score_pass2_pairs_relion_gpu_diff2_raw(
            jnp.asarray(shifted),
            jnp.asarray(corr),
            jnp.asarray(projection),
            jnp.asarray(half_weight),
            jnp.asarray(local_rotation_row),
            jnp.asarray(translation_idx),
            jnp.asarray(pair_mask),
        )
    )
    rows = np.arange(batch)[:, None]
    expected_compact_raw = dense_raw[rows, local_rotation_row, translation_idx]
    np.testing.assert_array_equal(compact_raw, expected_compact_raw)
    expected_compact = dense[rows, local_rotation_row, translation_idx]
    expected_compact = np.where(pair_mask, expected_compact, -np.inf)
    np.testing.assert_array_equal(compact, expected_compact)
    assert dense.dtype == cached.dtype == compact.dtype == np.float32
    assert np.isneginf(dense[0, 2, 1])
    assert np.all(np.isneginf(compact[~pair_mask]))


def test_relion_cuda_fine_diff2_handles_zero_pixels_and_nonfinite_scores():
    zero = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.zeros((2, 1, 0), dtype=jnp.complex64),
            jnp.zeros((1, 3, 0), dtype=jnp.complex64),
            jnp.zeros((1, 1, 0), dtype=jnp.float32),
        )
    )
    np.testing.assert_array_equal(zero, np.zeros((2, 3), dtype=np.float32))

    # Sentinel gaps gather row zero for bounds safety; masking must use where,
    # because NaN * False would still contaminate RELION's zero-corr slots.
    gap_only = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray([np.nan + 1j * np.nan], dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.ones(1, dtype=jnp.float32),
            jnp.asarray([-1, -1, -1], dtype=jnp.int32),
        )
    )
    np.testing.assert_array_equal(gap_only, np.asarray(0.0, dtype=np.float32))

    shifted = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    projection = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    scores = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2(
            shifted,
            jnp.asarray([[np.nan, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            projection,
            jnp.ones(4, dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1, 1), dtype=bool),
        )
    )
    assert np.isneginf(scores[0, 0, 0])
