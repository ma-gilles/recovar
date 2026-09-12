from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_big_jit, local_em_engine
from recovar.em.dense_single_volume import local_bucket_stages

pytestmark = pytest.mark.unit


def _make_noise_inputs(pixel_batch_size: int):
    rng = np.random.default_rng(7413 + pixel_batch_size)
    dense_batch_size = 42
    dense_rotations = 3
    pixel_rotations = 2
    translations = 3
    pixels = 4
    shell_count = 4

    scalar_probs = rng.uniform(
        0.0,
        0.4,
        size=(dense_batch_size, dense_rotations, translations),
    ).astype(np.float32)
    valid = np.arange(dense_batch_size) < pixel_batch_size
    scalar_probs[~valid] = 0.0
    pixel_probs = scalar_probs[:pixel_batch_size, :pixel_rotations].copy()
    pixel_probs[-1, -1] = 0.0
    projection = (
        rng.normal(size=(pixel_batch_size, pixel_rotations, pixels))
        + 1j * rng.normal(size=(pixel_batch_size, pixel_rotations, pixels))
    ).astype(np.complex64)
    projection[-1, -1] = 0.0
    ctf_probs = rng.uniform(
        0.2,
        1.4,
        size=(pixel_batch_size, pixel_rotations, pixels),
    ).astype(np.float32)
    ctf_probs[-1, -1] = 0.0
    shifted = (
        rng.normal(size=(dense_batch_size, translations, pixels))
        + 1j * rng.normal(size=(dense_batch_size, translations, pixels))
    ).astype(np.complex64)
    processed = (rng.normal(size=(dense_batch_size, pixels)) + 1j * rng.normal(size=(dense_batch_size, pixels))).astype(
        np.complex64
    )
    image_only_corr = rng.uniform(0.8, 1.2, dense_batch_size).astype(np.float32)
    translation_sqdist = rng.uniform(
        0.0,
        5.0,
        size=(dense_batch_size, translations),
    ).astype(np.float32)
    batch_scale = rng.uniform(0.7, 1.3, dense_batch_size).astype(np.float32)

    return {
        "noise_wsum": jnp.asarray(np.linspace(0.1, 0.4, shell_count), dtype=jnp.float64),
        "noise_img_power": jnp.asarray(np.linspace(0.2, 0.5, shell_count), dtype=jnp.float32),
        "noise_a2": jnp.asarray(np.linspace(0.3, 0.6, shell_count), dtype=jnp.float32),
        "noise_xa": jnp.asarray(np.linspace(0.4, 0.7, shell_count), dtype=jnp.float32),
        "noise_scale_xa": jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        "noise_scale_aa": jnp.asarray([0.75, 1.0], dtype=jnp.float32),
        "noise_norm_correction": jnp.zeros(64, dtype=jnp.float64),
        "noise_sigma2_offset": jnp.asarray(0.125, dtype=jnp.float32),
        "noise_sumw": jnp.asarray(0.25, dtype=jnp.float32),
        "scalar_reconstruction_probs": jnp.asarray(scalar_probs),
        "pixel_reconstruction_probs": jnp.asarray(pixel_probs),
        "pixel_proj_for_noise": jnp.asarray(projection),
        "pixel_ctf_probs": jnp.asarray(ctf_probs),
        "shifted_noise_split": jnp.asarray(shifted),
        "processed_score_half": jnp.asarray(processed),
        "image_only_corr": jnp.asarray(image_only_corr),
        "translation_sqdist_ang": jnp.asarray(translation_sqdist),
        "valid_image_mask": jnp.asarray(valid),
        "shell_indices_half": jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        "shell_indices_noise": jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        "noise_variance_for_noise": jnp.asarray([0.5, 0.75, 1.0, 1.25], dtype=jnp.float32),
        "scale_correction_pixel_mask": jnp.asarray([True, True, False, True]),
        "group_ids": jnp.asarray(np.arange(dense_batch_size) % 2, dtype=jnp.int32),
        "ctf_rfloat_half": jnp.ones((dense_batch_size, pixels), dtype=jnp.float64),
        "batch_scale": jnp.asarray(batch_scale),
        "relion_score_translation_angles": jnp.zeros((translations, 2), dtype=jnp.float32),
        "relion_wavg_rectangle_indices": jnp.arange(pixels, dtype=jnp.int32),
        "relion_wavg_exact_positions": jnp.arange(pixels, dtype=jnp.int32),
        "relion_wavg_rectangle_shell_indices": jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        "recon_window_indices": jnp.arange(pixels, dtype=jnp.int32),
        "bucket_image_indices": jnp.arange(pixel_batch_size - 1, -1, -1, dtype=jnp.int32),
        "runtime_logical_current_size": jnp.asarray(2, dtype=jnp.int32),
        "image_shape": (2, 2),
        "shell_count": shell_count,
    }


def _legacy_inline_noise(inputs, *, return_noise_split, accumulate_scale_correction):
    """The pre-extraction EM operations, with separate scalar/pixel lanes."""

    support_mass, _translation_posterior, noise_sumw_offset = local_big_jit.compute_local_noise_scalar_terms(
        inputs["scalar_reconstruction_probs"],
        inputs["translation_sqdist_ang"],
        inputs["valid_image_mask"],
    )
    processed_noise_power_half = inputs["processed_score_half"] * inputs["image_only_corr"][:, None]
    batch_img_power_shells, batch_img_power_per_image = local_big_jit._noise_image_power_shells_and_per_image(
        processed_noise_power_half,
        support_mass,
        inputs["shell_indices_half"],
        inputs["valid_image_mask"],
        inputs["runtime_logical_current_size"] // 2,
        shell_count=inputs["shell_count"],
        image_shape=inputs["image_shape"],
        current_size=None,
        runtime_current_size=None,
        include_unweighted_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=True,
    )
    noise_sumw = inputs["noise_sumw"] + jnp.sum(support_mass)

    pixel_batch_size = inputs["pixel_reconstruction_probs"].shape[0]
    pixel_valid = inputs["valid_image_mask"][:pixel_batch_size]
    shifted_for_noise = jnp.where(
        support_mass[:pixel_batch_size, None, None] != 0.0,
        inputs["shifted_noise_split"][:pixel_batch_size],
        0.0,
    )
    summed_masked_noise = local_big_jit.compute_local_weighted_sums(
        inputs["pixel_reconstruction_probs"],
        shifted_for_noise,
    )
    projection = inputs["pixel_proj_for_noise"]
    projection_flat = projection.reshape(-1, projection.shape[-1])
    projection_abs2 = jnp.abs(projection) ** 2
    block_noise, block_a2, block_xa = local_big_jit._compute_noise_block(
        projection_flat,
        (jnp.abs(projection_flat) ** 2),
        summed_masked_noise.reshape(-1, summed_masked_noise.shape[-1]),
        inputs["pixel_ctf_probs"].reshape(-1, inputs["pixel_ctf_probs"].shape[-1]),
        inputs["noise_variance_for_noise"],
        inputs["shell_indices_noise"],
        inputs["shell_count"],
        return_noise_split,
    )
    noise_wsum = inputs["noise_wsum"] + block_noise
    noise_img_power = inputs["noise_img_power"] + batch_img_power_shells
    noise_a2 = inputs["noise_a2"]
    noise_xa = inputs["noise_xa"]
    if return_noise_split:
        noise_a2 = noise_a2 + block_a2
        noise_xa = noise_xa + block_xa

    noise_scale_xa = inputs["noise_scale_xa"]
    noise_scale_aa = inputs["noise_scale_aa"]
    if accumulate_scale_correction:
        scale_xa, scale_aa = local_big_jit._compute_scale_correction_terms_per_image(
            projection,
            projection_abs2,
            summed_masked_noise,
            inputs["pixel_ctf_probs"],
            inputs["noise_variance_for_noise"],
            inputs["batch_scale"][:pixel_batch_size],
            inputs["scale_correction_pixel_mask"],
        )
        scale_xa = jnp.where(pixel_valid, scale_xa, 0.0)
        scale_aa = jnp.where(pixel_valid, scale_aa, 0.0)
        groups = inputs["group_ids"][:pixel_batch_size]
        noise_scale_xa = noise_scale_xa.at[groups].add(scale_xa)
        noise_scale_aa = noise_scale_aa.at[groups].add(scale_aa)

    bucket_norm = batch_img_power_per_image[:pixel_batch_size]
    bucket_norm = bucket_norm + local_big_jit._compute_norm_residual_per_image(
        projection,
        projection_abs2,
        summed_masked_noise,
        inputs["pixel_ctf_probs"],
        inputs["noise_variance_for_noise"],
    )
    bucket_norm = jnp.where(pixel_valid, bucket_norm, 0.0).astype(jnp.float64)
    return (
        noise_wsum,
        noise_img_power,
        noise_a2,
        noise_xa,
        noise_scale_xa,
        noise_scale_aa,
        bucket_norm,
        inputs["noise_sigma2_offset"] + noise_sumw_offset,
        noise_sumw,
        jnp.zeros((1, 3), dtype=jnp.float64),
    )


def _call_shared(
    inputs,
    *,
    return_noise_split,
    accumulate_scale_correction,
    use_relion_wavg_cutoff=False,
    return_debug_wavg_cutoff_triplet=False,
    native_residual_statistics=False,
):
    return local_big_jit.compute_local_exact_noise(
        *(
            inputs[name]
            for name in (
                "noise_wsum",
                "noise_img_power",
                "noise_a2",
                "noise_xa",
                "noise_scale_xa",
                "noise_scale_aa",
                "noise_sigma2_offset",
                "noise_sumw",
                "scalar_reconstruction_probs",
                "pixel_reconstruction_probs",
                "pixel_proj_for_noise",
                "pixel_ctf_probs",
                "shifted_noise_split",
                "processed_score_half",
                "image_only_corr",
                "translation_sqdist_ang",
                "valid_image_mask",
                "shell_indices_half",
                "shell_indices_noise",
                "noise_variance_for_noise",
                "scale_correction_pixel_mask",
                "group_ids",
                "ctf_rfloat_half",
                "batch_scale",
                "relion_score_translation_angles",
                "relion_wavg_rectangle_indices",
                "relion_wavg_exact_positions",
                "relion_wavg_rectangle_shell_indices",
                "recon_window_indices",
            )
        ),
        inputs["runtime_logical_current_size"] // 2,
        inputs["runtime_logical_current_size"] // 2,
        None,
        None,
        None,
        image_shape=inputs["image_shape"],
        shell_count=inputs["shell_count"],
        norm_current_size=None,
        include_unweighted_norm_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=True,
        accumulate_scale_correction=accumulate_scale_correction,
        return_noise_split=return_noise_split,
        use_relion_wavg_cutoff=use_relion_wavg_cutoff,
        relion_wavg_sequential_cuda=True,
        native_residual_statistics=native_residual_statistics,
        return_debug_wavg_cutoff_triplet=(
            return_debug_wavg_cutoff_triplet
        ),
    )


@pytest.mark.parametrize("pixel_batch_size", (42, 32))
@pytest.mark.parametrize("return_noise_split", (False, True))
@pytest.mark.parametrize("accumulate_scale_correction", (False, True))
def test_shared_exact_noise_matches_pre_extraction_oracle_exactly(
    pixel_batch_size,
    return_noise_split,
    accumulate_scale_correction,
):
    inputs = _make_noise_inputs(pixel_batch_size)
    expected = _legacy_inline_noise(
        inputs,
        return_noise_split=return_noise_split,
        accumulate_scale_correction=accumulate_scale_correction,
    )
    actual = _call_shared(
        inputs,
        return_noise_split=return_noise_split,
        accumulate_scale_correction=accumulate_scale_correction,
    )

    assert len(actual) == len(expected) == 10
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.shape == expected_value.shape
        assert actual_value.dtype == expected_value.dtype
        np.testing.assert_array_equal(
            np.asarray(actual_value),
            np.asarray(expected_value),
        )


def test_shared_exact_noise_preserves_wavg_cutoff_wiring(monkeypatch):
    inputs = _make_noise_inputs(32)
    direct = jnp.asarray(
        [
            [11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0],
            [31.0, 32.0, 33.0, 34.0],
        ],
        dtype=jnp.float64,
    )
    debug = jnp.asarray([[41.0, 42.0, 43.0]], dtype=jnp.float64)

    def fake_direct_wavg(*args, **kwargs):
        assert args[0].shape[0] == 32
        assert args[6].shape == inputs["pixel_proj_for_noise"].shape
        assert args[9].shape == inputs["pixel_reconstruction_probs"].shape
        assert args[10].shape == (32,)
        assert int(np.asarray(kwargs["cutoff_shell"])) == 1
        assert kwargs["logical_recon_pixel_count"] is None
        assert kwargs["logical_rectangle_pixel_count"] is None
        assert kwargs["return_per_image_cutoff"] is True
        return direct, debug

    monkeypatch.setattr(
        local_big_jit,
        "_relion_wavg_direct_triplet_shells",
        fake_direct_wavg,
    )
    with_wavg = _call_shared(
        inputs,
        return_noise_split=True,
        accumulate_scale_correction=False,
        use_relion_wavg_cutoff=True,
        return_debug_wavg_cutoff_triplet=True,
    )

    cutoff = 1
    expected_cutoff_values = (
        np.asarray(inputs["noise_wsum"])[cutoff] + np.asarray(direct)[2, cutoff],
        np.asarray(inputs["noise_img_power"])[cutoff],
        np.asarray(inputs["noise_a2"])[cutoff] + np.asarray(direct)[1, cutoff],
        np.asarray(inputs["noise_xa"])[cutoff] + np.asarray(direct)[0, cutoff],
    )
    for output_index, expected_cutoff in enumerate(expected_cutoff_values):
        np.testing.assert_array_equal(
            np.asarray(with_wavg[output_index])[cutoff],
            expected_cutoff,
        )
    np.testing.assert_array_equal(np.asarray(with_wavg[-1]), np.asarray(debug))


def _call_deferred_wrapper(inputs, *, source_faithful_spectrum_norm=True):
    return local_big_jit.run_deferred_local_exact_noise_jit(
        *(
            inputs[name]
            for name in (
                "noise_wsum",
                "noise_img_power",
                "noise_a2",
                "noise_xa",
                "noise_scale_xa",
                "noise_scale_aa",
                "noise_norm_correction",
                "noise_sigma2_offset",
                "noise_sumw",
                "scalar_reconstruction_probs",
                "pixel_reconstruction_probs",
                "pixel_proj_for_noise",
                "pixel_ctf_probs",
                "shifted_noise_split",
                "processed_score_half",
                "image_only_corr",
                "translation_sqdist_ang",
                "valid_image_mask",
                "shell_indices_half",
                "shell_indices_noise",
                "noise_variance_for_noise",
                "scale_correction_pixel_mask",
                "group_ids",
                "ctf_rfloat_half",
                "batch_scale",
                "relion_score_translation_angles",
                "relion_wavg_rectangle_indices",
                "relion_wavg_exact_positions",
                "relion_wavg_rectangle_shell_indices",
                "recon_window_indices",
                "bucket_image_indices",
                "runtime_logical_current_size",
            )
        ),
        image_shape=inputs["image_shape"],
        shell_count=inputs["shell_count"],
        norm_current_size=None,
        stable_fourier_window_shapes=False,
        include_unweighted_norm_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        accumulate_scale_correction=True,
        return_noise_split=True,
        use_relion_wavg_cutoff=False,
        relion_wavg_sequential_cuda=True,
    )


def _legacy_deferred_outputs(inputs):
    local = _legacy_inline_noise(
        inputs,
        return_noise_split=True,
        accumulate_scale_correction=True,
    )
    global_norm = inputs["noise_norm_correction"].at[
        inputs["bucket_image_indices"]
    ].add(local[6])
    return (*local[:6], global_norm, *local[7:9])


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_norm_capacity_reuses_noise_executable_and_preserves_logical_carries(dtype):
    function = local_big_jit.run_deferred_local_exact_noise_jit
    carry_names = (
        "noise_wsum", "noise_img_power", "noise_a2", "noise_xa",
        "noise_scale_xa", "noise_scale_aa", "noise_norm_correction",
        "noise_sigma2_offset", "noise_sumw",
    )
    function.clear_cache()
    seen_capacities = set()
    seen_shapes = set()
    try:
        for n_images in (200, 208, 272, 1000, 1024, 1025, 2000):
            capacity = local_bucket_stages._noise_norm_capacity(n_images, enabled=True)
            assert 0 <= capacity - n_images < 1024
            inputs = _make_noise_inputs(32)
            # Nonzero existing carries and repeated, non-prefix image indices.
            inputs["noise_norm_correction"] = jnp.arange(n_images, dtype=dtype) / 8
            inputs["bucket_image_indices"] = jnp.asarray(
                (np.arange(32) // 2 * 17 + n_images - 1) % n_images,
                dtype=jnp.int32,
            )
            padded = dict(inputs)
            padded["noise_norm_correction"] = jnp.pad(
                inputs["noise_norm_correction"], (0, capacity - n_images)
            )
            for _ in range(2):
                expected = _call_deferred_wrapper(inputs, source_faithful_spectrum_norm=dtype == jnp.float64)
                seen_shapes.add(n_images)
                before = function._cache_size()
                actual = _call_deferred_wrapper(padded, source_faithful_spectrum_norm=dtype == jnp.float64)
                assert function._cache_size() == before + (capacity not in seen_shapes)
                seen_shapes.add(capacity)
                seen_capacities.add(capacity)
                for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
                    assert got.dtype == want.dtype
                    if index == 6:
                        assert got.shape == (capacity,) and want.shape == (n_images,)
                        np.testing.assert_array_equal(np.asarray(got[:n_images]), np.asarray(want))
                        np.testing.assert_array_equal(np.asarray(got[n_images:]), np.zeros(capacity - n_images))
                    else:
                        assert got.shape == want.shape
                        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
                inputs.update(zip(carry_names, expected, strict=True))
                padded.update(zip(carry_names, actual, strict=True))
        assert seen_capacities == {1024, 2048}
        assert function._cache_size() == len(seen_shapes)
    finally:
        function.clear_cache()


def test_deferred_exact_noise_wrapper_has_one_boundary_and_b42_b32_cache_keys():
    function = local_big_jit.run_deferred_local_exact_noise_jit
    assert not hasattr(local_big_jit.compute_local_exact_noise, "lower")
    assert "compute_local_exact_noise(" in inspect.getsource(local_big_jit.run_local_bucket_big_jit)

    dense_inputs = _make_noise_inputs(42)
    tail_inputs = _make_noise_inputs(32)
    function.clear_cache()
    try:
        dense_first = _call_deferred_wrapper(dense_inputs)
        for actual, expected in zip(
            dense_first,
            _legacy_deferred_outputs(dense_inputs),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        dense_cache_size = function._cache_size()
        assert dense_cache_size == 1
        dense_second = _call_deferred_wrapper(dense_inputs)
        assert function._cache_size() == dense_cache_size
        for first, second in zip(dense_first, dense_second, strict=True):
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

        tail_output = _call_deferred_wrapper(tail_inputs)
        for actual, expected in zip(
            tail_output,
            _legacy_deferred_outputs(tail_inputs),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert function._cache_size() == dense_cache_size + 1

        def trace_with_dynamic_current_size(runtime_current_size):
            traced_inputs = dict(dense_inputs)
            traced_inputs["runtime_logical_current_size"] = runtime_current_size
            return _call_deferred_wrapper(traced_inputs)

        wrapper_jaxpr = str(
            jax.make_jaxpr(trace_with_dynamic_current_size)(
                dense_inputs["runtime_logical_current_size"]
            )
        )
        assert wrapper_jaxpr.count("name=run_deferred_local_exact_noise_jit") == 1
        assert "name=compute_local_exact_noise" not in wrapper_jaxpr
    finally:
        function.clear_cache()


@pytest.mark.parametrize("return_split", (False, True))
@pytest.mark.parametrize("compute_scale", (False, True))
@pytest.mark.parametrize("mask_missing", (False, True))
def test_native_noise_composition_preserves_all_carries(
    monkeypatch, return_split, compute_scale, mask_missing,
):
    """Isolate JAX composition from the separately qualified native reduction."""
    from recovar import cuda_noise_residual as native

    inputs = _make_noise_inputs(32)
    inputs["noise_variance_for_noise"] = inputs["noise_variance_for_noise"].astype(jnp.float64)
    # The old scale must be cast to projection precision before division.
    inputs["batch_scale"] = inputs["batch_scale"].astype(jnp.float64) + 1e-9
    if mask_missing:
        inputs["scale_correction_pixel_mask"] = None
    calls = []

    def reference(*args, **kwargs):
        calls.append(kwargs["compute_scale"])
        native._output_shapes(*args)
        return native.reference_statistics(*args, **kwargs)

    monkeypatch.setattr(native, "residual_statistics", reference)
    # This checks composition, not differing nested-XLA fusion boundaries.
    # Compiled native arithmetic is qualified separately on saved GPU operands.
    with jax.disable_jit():
        expected = _call_shared(inputs, return_noise_split=return_split,
                                accumulate_scale_correction=compute_scale)
        assert calls == []  # Default branch must not request the optional CUDA helper.
        actual = _call_shared(inputs, return_noise_split=return_split,
                              accumulate_scale_correction=compute_scale,
                              native_residual_statistics=True)
    assert calls == [compute_scale]
    for a, b in zip(actual, expected, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
