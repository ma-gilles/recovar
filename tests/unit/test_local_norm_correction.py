"""Focused tests for exact-local RELION norm-correction image power."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.half_spectrum import make_relion_noise_shell_indices_half
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_powerclass_highres_norm_units,
    _relion_cuda_powerclass_spectrum_highres_norm_units,
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
