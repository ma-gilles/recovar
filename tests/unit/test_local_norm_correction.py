"""Focused tests for exact-local RELION norm-correction image power."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.half_spectrum import make_relion_noise_shell_indices_half
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_powerclass_highres_norm_units,
    _relion_cuda_powerclass_spectrum_highres_norm_units,
    _relion_cuda_translate_wavg_norm_images,
    _replace_untranslated_low_shell_norm_power,
    _translated_wavg_low_shell_power_pixels,
    _weighted_image_power_shells_and_per_image,
)
from recovar.em.dense_single_volume.local_big_jit import (
    _norm_correction_image_power_mass,
    _norm_correction_image_power_per_image,
)


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
