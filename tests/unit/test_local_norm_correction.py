"""Focused tests for exact-local RELION norm-correction image power."""

import jax
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.half_spectrum import make_relion_noise_shell_indices_half
from recovar.em.dense_single_volume.helpers.sparse_pass2_wavg import (
    _relion_cuda_translate_wavg_norm_images,
    _replace_untranslated_low_shell_norm_power,
    _translated_wavg_low_shell_power_pixels,
    _weighted_image_power_shells_and_per_image,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_scoring import (
    _relion_cuda_powerclass_highres_norm_units,
    _relion_cuda_powerclass_spectrum_highres_norm_units,
)
from recovar.em.dense_single_volume.local_big_jit import (
    _noise_image_power_shells_and_per_image,
    _norm_correction_image_power_mass,
    _norm_correction_image_power_per_image,
)
from recovar.em.dense_single_volume.local_em_engine import _noise_wsum_initial_dtype


@pytest.mark.parametrize(
    ("relion_exact_fine_diff2", "use_window", "expected"),
    (
        (True, True, np.float64),
        (True, False, np.float32),
        (False, True, np.float32),
        (False, False, np.float32),
    ),
)
def test_noise_wsum_initial_dtype_matches_direct_wavg_output(
    relion_exact_fine_diff2,
    use_window,
    expected,
):
    actual = _noise_wsum_initial_dtype(
        relion_exact_fine_diff2=relion_exact_fine_diff2,
        use_window=use_window,
    )

    assert np.dtype(actual) == np.dtype(expected)


def test_noise_wsum_float64_zero_is_bitwise_equivalent_to_first_bucket_promotion():
    direct_wavg_shells = jnp.asarray(
        [0.0, np.nextafter(1.0, 2.0), -3.25, 2**40 + 0.25],
        dtype=jnp.float64,
    )
    add_bucket = jax.jit(lambda carry, block: carry + block)

    legacy = add_bucket(jnp.zeros(4, dtype=jnp.float32), direct_wavg_shells)
    canonical = add_bucket(
        jnp.zeros(
            4,
            dtype=_noise_wsum_initial_dtype(
                relion_exact_fine_diff2=True,
                use_window=True,
            ),
        ),
        direct_wavg_shells,
    )

    assert np.asarray(legacy).dtype == np.float64
    assert np.asarray(canonical).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(canonical), np.asarray(legacy))


def test_norm_correction_mass_drops_invalid_shells_and_keeps_valid_outer_shell():
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    # -1 represents the redundant negative x=0 half-spectrum axis; 2 is the
    # out-of-circle drop-bin sentinel equal to shell_count. Shell 1 remains a
    # valid outer shell beyond the current model cutoff and receives full mass.
    shell_indices = jnp.asarray([-1, 0, 1, 2], dtype=jnp.int32)
    valid_image_mask = jnp.asarray([True, False])

    actual = _norm_correction_image_power_mass(
        support_mass,
        shell_indices,
        valid_image_mask,
        projection_max_r=0,
        shell_count=2,
    )

    expected = np.asarray(
        [
            [0.0, 0.25, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_norm_correction_mass_drops_invalid_shells_without_model_window():
    support_mass = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
    shell_indices = jnp.asarray([-1, 0, 1, 2], dtype=jnp.int32)

    actual = _norm_correction_image_power_mass(
        support_mass,
        shell_indices,
        jnp.asarray([True, True]),
        projection_max_r="auto",
        shell_count=2,
    )

    expected = np.asarray(
        [
            [0.0, 0.25, 0.25, 0.0],
            [0.0, 0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_norm_correction_mass_can_omit_shared_unweighted_outer_shell():
    actual = _norm_correction_image_power_mass(
        jnp.asarray([0.25], dtype=jnp.float32),
        jnp.asarray([-1, 0, 1, 2], dtype=jnp.int32),
        jnp.asarray([True]),
        projection_max_r=0,
        shell_count=2,
        include_unweighted_high_shell=False,
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray([[0.0, 0.25, 0.0, 0.0]], dtype=np.float32),
    )


def test_norm_correction_power_uses_relion_powerclass_once_or_not_at_all():
    rng = np.random.default_rng(7042)
    height = 32
    current_size = 15
    processed = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    processed = jnp.asarray(processed.reshape(2, -1))
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    valid_images = jnp.asarray([True, True])
    shell_indices = jnp.asarray(make_relion_noise_shell_indices_half((height, height)))
    shell_count = height // 2 + 1
    cutoff = current_size // 2

    with_shared_high = _norm_correction_image_power_per_image(
        processed,
        support_mass,
        shell_indices,
        valid_images,
        cutoff,
        shell_count=shell_count,
        image_shape=(height, height),
        current_size=current_size,
        include_unweighted_high_shell=True,
    )
    without_shared_high = _norm_correction_image_power_per_image(
        processed,
        support_mass,
        shell_indices,
        valid_images,
        cutoff,
        shell_count=shell_count,
        image_shape=(height, height),
        current_size=current_size,
        include_unweighted_high_shell=False,
    )

    pixel_power = jnp.abs(processed) ** 2
    valid_low = (shell_indices >= 0) & (shell_indices < shell_count) & (shell_indices <= cutoff)
    expected_low = jnp.sum(
        jnp.where(valid_low[None, :], pixel_power * support_mass[:, None], 0.0),
        axis=-1,
    ).astype(jnp.float32)
    expected_high = _relion_cuda_powerclass_highres_norm_units(
        processed,
        image_shape=(height, height),
        current_size=current_size,
    )

    np.testing.assert_allclose(np.asarray(without_shared_high), np.asarray(expected_low), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(with_shared_high),
        np.asarray(expected_low + expected_high),
        rtol=2e-7,
        atol=2e-2,
    )


def test_local_noise_spectrum_uses_unweighted_high_shell_particle_power():
    processed_prefix = np.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    processed = jnp.asarray(np.pad(processed_prefix, ((0, 0), (0, 9))))
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 2] + [3] * 9, dtype=jnp.int32)

    shells, _ = _noise_image_power_shells_and_per_image(
        processed,
        support_mass,
        shell_indices,
        jnp.asarray([True, True]),
        projection_max_r=1,
        shell_count=3,
        image_shape=(4, 4),
        current_size=2,
    )

    power = np.abs(processed_prefix) ** 2
    expected = np.asarray(
        [power[0, 0] * 0.25, power[0, 1] * 0.25, power[0, 2] + power[1, 2]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(shells), expected)


def test_local_noise_spectrum_omits_shared_high_shell_for_non_owner_class():
    processed = jnp.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=jnp.complex64)
    shells, _ = _noise_image_power_shells_and_per_image(
        processed,
        jnp.asarray([0.25], dtype=jnp.float32),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jnp.asarray([True]),
        projection_max_r=1,
        shell_count=3,
        image_shape=(4, 4),
        current_size=2,
        include_unweighted_high_shell=False,
    )

    np.testing.assert_array_equal(np.asarray(shells), np.asarray([0.25, 1.0, 0.0], dtype=np.float32))


def test_local_norm_correction_can_use_relion_powerclass_spectrum_tail():
    rng = np.random.default_rng(21084)
    height = 32
    current_size = 15
    processed = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    processed = jnp.asarray(processed.reshape(2, -1))
    shell_indices = jnp.asarray(make_relion_noise_shell_indices_half((height, height)))
    zero_support = jnp.zeros(2, dtype=jnp.float32)
    valid_images = jnp.ones(2, dtype=bool)

    historical = _norm_correction_image_power_per_image(
        processed,
        zero_support,
        shell_indices,
        valid_images,
        current_size // 2,
        shell_count=height // 2 + 1,
        image_shape=(height, height),
        current_size=current_size,
    )
    source_faithful = _norm_correction_image_power_per_image(
        processed,
        zero_support,
        shell_indices,
        valid_images,
        current_size // 2,
        shell_count=height // 2 + 1,
        image_shape=(height, height),
        current_size=current_size,
        source_faithful_spectrum_norm=True,
    )

    expected_historical = _relion_cuda_powerclass_highres_norm_units(
        processed,
        image_shape=(height, height),
        current_size=current_size,
    )
    expected_source = _relion_cuda_powerclass_spectrum_highres_norm_units(
        processed,
        image_shape=(height, height),
        current_size=current_size,
    )
    np.testing.assert_array_equal(np.asarray(historical), np.asarray(expected_historical))
    # The source-faithful powerClass spectrum is binned with GPU scatter-adds.
    # Independent invocations can differ by a few float32 ULPs even though the
    # final accumulated value is float64.
    np.testing.assert_allclose(
        np.asarray(source_faithful),
        np.asarray(expected_source),
        rtol=2 * np.finfo(np.float32).eps,
        atol=0.0,
    )
    assert np.asarray(source_faithful).dtype == np.float64
    assert np.any(np.asarray(source_faithful) != np.asarray(historical, dtype=np.float64))


def test_powerclass_spectrum_norm_sums_shell_bins_in_host_precision():
    height = 8
    current_size = 4
    half_width = height // 2 + 1
    centered = np.arange(height * half_width, dtype=np.float32).reshape(height, half_width)
    centered = ((centered % 4) + 1j * (centered % 3)).astype(np.complex64)
    processed = centered.reshape(1, -1) * np.float32(height * height)

    actual = _relion_cuda_powerclass_spectrum_highres_norm_units(
        jnp.asarray(processed),
        image_shape=(height, height),
        current_size=current_size,
    )

    relion_image = np.roll(centered, -(height // 2), axis=0)
    expected = np.float64(0.0)
    for y in range(height):
        signed_y = y if y < half_width else y - height
        for x in range(half_width):
            shell = int(np.rint(np.sqrt(np.float32(x * x + signed_y * signed_y))))
            if (
                shell >= current_size // 2 + 1
                and shell < half_width
                and not (x == 0 and signed_y < 0)
            ):
                value = relion_image[y, x]
                expected += np.float64(
                    np.float32(value.real * value.real + value.imag * value.imag)
                )
    expected *= np.float64((height * height) ** 2)

    assert np.asarray(actual).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(actual), np.asarray([expected]))


def test_powerclass_spectrum_norm_keeps_double_kernel_inputs():
    height = 8
    current_size = 4
    half_width = height // 2 + 1
    centered = np.arange(height * half_width, dtype=np.float64).reshape(height, half_width)
    centered = (centered + 1j * (centered / 3.0 + 2.0**-35)).astype(np.complex128)
    processed = centered.reshape(1, -1) * np.float64(height * height)

    actual = np.asarray(
        _relion_cuda_powerclass_spectrum_highres_norm_units(
            jnp.asarray(processed),
            image_shape=(height, height),
            current_size=current_size,
        )
    )

    relion_image = np.roll(centered, -(height // 2), axis=0)
    expected = np.float64(0.0)
    for y in range(height):
        signed_y = y if y < half_width else y - height
        for x in range(half_width):
            shell = int(np.rint(np.sqrt(np.float64(x * x + signed_y * signed_y))))
            if (
                shell >= current_size // 2 + 1
                and shell < half_width
                and not (x == 0 and signed_y < 0)
            ):
                value = relion_image[y, x]
                expected += np.float64(value.real * value.real + value.imag * value.imag)
    expected *= np.float64((height * height) ** 2)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, np.asarray([expected]), rtol=3e-16, atol=1e-8)
    assert not np.array_equal(actual, actual.astype(np.float32).astype(np.float64))


def test_powerclass_spectrum_norm_runtime_current_size_reuses_trace_and_matches_static():
    height = 8
    half_width = height // 2 + 1
    centered = np.arange(height * half_width, dtype=np.float32).reshape(height, half_width)
    centered = ((centered % 4) + 1j * (centered % 3)).astype(np.complex64)
    processed = jnp.asarray(centered.reshape(1, -1) * np.float32(height * height))

    function = _relion_cuda_powerclass_spectrum_highres_norm_units
    function.clear_cache()
    try:
        dynamic_by_size = {}
        for current_size in (4, 6):
            dynamic_by_size[current_size] = function(
                processed,
                image_shape=(height, height),
                current_size=None,
                runtime_current_size=jnp.asarray(current_size, dtype=jnp.int32),
            )
            if current_size == 4:
                dynamic_cache_size = function._cache_size()
                assert dynamic_cache_size == 1
            else:
                assert function._cache_size() == dynamic_cache_size

        assert np.any(np.asarray(dynamic_by_size[4]) != np.asarray(dynamic_by_size[6]))
        for current_size in (4, 6):
            static = function(
                processed,
                image_shape=(height, height),
                current_size=current_size,
            )
            np.testing.assert_array_equal(
                np.asarray(dynamic_by_size[current_size]),
                np.asarray(static),
            )
    finally:
        function.clear_cache()


def test_translated_wavg_low_shell_power_preserves_per_pixel_boundary():
    shifted = np.asarray(
        [
            [
                [3 + 4j, 5 + 12j, 8 + 15j, 7 + 24j],
                [6 + 8j, 9 + 12j, 20 + 21j, 10 + 24j],
            ]
        ],
        dtype=np.complex64,
    )
    posterior = np.asarray([[0.25, 0.75]], dtype=np.float32)
    shells = np.asarray([0, 1, 2, -1], dtype=np.int32)

    actual = _translated_wavg_low_shell_power_pixels(
        jnp.asarray(shifted),
        jnp.asarray(posterior),
        jnp.asarray(shells),
        jnp.asarray(1, dtype=jnp.int32),
    )

    power = shifted.real * shifted.real
    power = np.asarray(power + shifted.imag * shifted.imag, dtype=np.float32)
    expected = np.sum(posterior[:, :, None] * power, axis=1, dtype=np.float32)
    expected[:, 2:] = 0.0
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_relion_wavg_norm_translation_uses_raw_windowed_image(monkeypatch):
    from recovar import cuda_backproject

    processed = jnp.asarray(
        [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]],
        dtype=jnp.complex64,
    )
    indices = jnp.asarray([0, 2, 3], dtype=jnp.int32)
    angles = jnp.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=jnp.float32)

    def fake_translate(images, received_angles, received_indices, image_shape):
        np.testing.assert_array_equal(np.asarray(images), np.asarray(processed[:, indices]))
        np.testing.assert_array_equal(np.asarray(received_angles), np.asarray(angles))
        np.testing.assert_array_equal(np.asarray(received_indices), np.asarray(indices))
        assert image_shape == (4, 4)
        return jnp.arange(6, dtype=jnp.float32).astype(jnp.complex64).reshape(2, 3)

    monkeypatch.setattr(cuda_backproject, "relion_translate_score_f32", fake_translate)
    actual = _relion_cuda_translate_wavg_norm_images(
        processed,
        angles,
        indices,
        (4, 4),
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.arange(6, dtype=np.float32).astype(np.complex64).reshape(1, 2, 3),
    )


def test_translated_wavg_norm_replaces_only_untranslated_low_shell_power(monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM", "1")
    processed = np.asarray([[3 + 4j, 5 + 12j, 8 + 15j, 7 + 24j]], dtype=np.complex64)
    shifted = np.asarray(
        [
            [
                [3 + 4j, 5 + 12j, 8 + 15j, 7 + 24j],
                [6 + 8j, 9 + 12j, 20 + 21j, 10 + 24j],
            ]
        ],
        dtype=np.complex64,
    )
    posterior = np.asarray([[0.25, 0.75]], dtype=np.float32)
    shells = np.asarray([0, 1, 2, -1], dtype=np.int32)
    high_and_residual = np.float64(123.5)
    _, baseline = _weighted_image_power_shells_and_per_image(
        jnp.asarray(processed),
        jnp.asarray(shells),
        jnp.ones(1, dtype=jnp.float32),
        shell_count=3,
        norm_unweighted_shell_cutoff=1,
        norm_unweighted_high_shell=jnp.asarray([high_and_residual], dtype=jnp.float64),
    )

    actual = _replace_untranslated_low_shell_norm_power(
        baseline,
        jnp.asarray(processed),
        jnp.asarray(shifted),
        jnp.asarray(posterior),
        jnp.asarray(shells),
        jnp.arange(4, dtype=jnp.int32),
        shell_cutoff=1,
    )

    shifted_power = shifted.real * shifted.real
    shifted_power = np.asarray(
        shifted_power + shifted.imag * shifted.imag,
        dtype=np.float32,
    )
    translated_low_pixels = np.sum(
        posterior[:, :, None] * shifted_power,
        axis=1,
        dtype=np.float32,
    )[:, :2]
    expected = high_and_residual + np.sum(translated_low_pixels, dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray([expected]))
