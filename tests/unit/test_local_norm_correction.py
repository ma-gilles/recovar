"""Focused tests for exact-local RELION norm-correction image power."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.local_big_jit import _norm_correction_image_power_mass


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
