import numpy as np


def test_relion_project_half_uses_projector_matrix_directly():
    """RELION accelerator euler matrices are projector matrices, not inverses."""
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half

    n = 8
    volume = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
    source_value = 7.0 + 3.0j
    # Output pixel (x=1, y=2) under this RELION projector matrix maps to
    # (xp=-2, yp=1, zp=0), then RELION flips negative xp to the Hermitian
    # partner (xp=2, yp=-1, zp=0) and conjugates the interpolated value.
    volume[n // 2, n // 2 - 1, 2] = source_value
    relion_projector_matrix = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    projected = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(relion_projector_matrix),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )

    np.testing.assert_allclose(projected[2, 1], np.conj(source_value), rtol=0.0, atol=1e-12)


def test_centered_row_projector_transposes_scorer_rotations():
    """Centered-row RECOVAR scoring uses the transpose at the raw Projector handoff."""
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half
    from recovar.em.dense_single_volume.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows,
    )

    n = 8
    rng = np.random.default_rng(7)
    volume = (
        rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(size=(n, n, n // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    row_order = np.fft.fftshift(np.arange(n))

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (n, n),
            r_max=n // 2,
            padding_factor=1,
        )
    )
    expected_raw = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )
    direct_raw = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )

    expected = expected_raw[row_order, :].reshape(1, -1)
    direct_centered = direct_raw[row_order, :].reshape(1, -1)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(got - direct_centered)) > 1e-3


def test_centered_row_projector_scatters_cropped_ppref_into_full_box():
    """Cropped RELION ``PPref`` output must land in full-box RECOVAR row order."""
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half
    from recovar.em.dense_single_volume.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows,
    )

    full_n = 8
    current_size = 4
    rng = np.random.default_rng(11)
    volume = (
        rng.normal(size=(current_size, current_size, current_size // 2 + 1))
        + 1j * rng.normal(size=(current_size, current_size, current_size // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (full_n, full_n),
            r_max=current_size // 2,
            padding_factor=1,
        )
    ).reshape(full_n, full_n // 2 + 1)
    expected_crop = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            current_size,
            r_max=current_size // 2,
            padding_factor=1,
        )
    )

    crop_rows = np.arange(current_size, dtype=np.int32)
    crop_ky = np.where(crop_rows <= current_size // 2, crop_rows, crop_rows - current_size)
    full_rows = crop_ky + full_n // 2
    for crop_row, full_row in enumerate(full_rows):
        np.testing.assert_allclose(
            got[full_row, : current_size // 2 + 1],
            expected_crop[crop_row],
            rtol=1e-12,
            atol=1e-12,
        )


def test_centered_row_projector_can_use_explicit_coarse_output_size():
    """RELION pass-1 projects PPref into the current-size Fimg box."""
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half
    from recovar.em.dense_single_volume.helpers.projection import (
        project_relion_projector_half_spectrum_centered_rows,
    )

    full_n = 8
    projector_n = 8
    coarse_n = 4
    rng = np.random.default_rng(13)
    volume = (
        rng.normal(size=(projector_n, projector_n, projector_n // 2 + 1))
        + 1j * rng.normal(size=(projector_n, projector_n, projector_n // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (full_n, full_n),
            r_max=projector_n // 2,
            padding_factor=1,
            projector_output_size=coarse_n,
        )
    ).reshape(full_n, full_n // 2 + 1)
    expected_crop = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            coarse_n,
            r_max=projector_n // 2,
            padding_factor=1,
        )
    )

    crop_rows = np.arange(coarse_n, dtype=np.int32)
    crop_ky = np.where(crop_rows <= coarse_n // 2, crop_rows, crop_rows - coarse_n)
    full_rows = crop_ky + full_n // 2
    for crop_row, full_row in enumerate(full_rows):
        np.testing.assert_allclose(
            got[full_row, : coarse_n // 2 + 1],
            expected_crop[crop_row],
            rtol=1e-12,
            atol=1e-12,
        )


def test_relion_acc_double_floorf_quirk_matches_default_away_from_integer_boundary():
    """Away from an integer coordinate, the GPU floorf-narrowing quirk is a no-op."""
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half

    n = 16
    rng = np.random.default_rng(23)
    volume = (
        rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(size=(n, n, n // 2 + 1))
    ).astype(np.complex128)
    # A generic (non-axis-aligned) rotation, far from any integer-coordinate edge case.
    rotation = np.asarray(
        [
            [0.36, -0.48, 0.8],
            [0.8, 0.6, 0.0],
            [-0.48, 0.64, 0.6],
        ],
        dtype=np.float64,
    )

    default = np.asarray(
        relion_project_half(jnp.asarray(volume), jnp.asarray(rotation), n, r_max=n // 2, padding_factor=1)
    )
    quirked = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
            relion_acc_double_floorf_quirk=True,
        )
    )
    np.testing.assert_allclose(quirked, default, rtol=1e-6, atol=1e-6)


def test_relion_acc_double_floorf_quirk_flips_bucket_at_integer_boundary():
    """RELION's GPU no_tex3D/BP.cuh floor with float32 ``floorf`` on the double
    coordinate, unconditionally, even under ``ACC_DOUBLE_PRECISION``
    (relion/src/acc/cuda/cuda_kernels/cuda_device_utils.cuh, BP.cuh). A
    coordinate within a float32 ULP of an integer therefore floors to a
    *different* integer than a native double floor would give, selecting a
    different pair of interpolation neighbors and a fractional weight
    computed against that (now wrong) integer -- not merely a smaller
    rounding error.
    """
    import jax.numpy as jnp

    from recovar.core.relion_project import relion_project_half

    n = 32
    # xp = 10 - 4e-7: float64 floors to 9. Cast to float32, this value rounds
    # up to exactly 10.0 (float32 ULP at this magnitude is ~9.5e-7, so 4e-7 is
    # within half a ULP) and floors to 10 -- a different integer bucket.
    offset = 4e-7
    xp0 = 10.0 - offset
    assert np.floor(xp0) == 9.0
    assert np.floor(np.float64(np.float32(xp0))) == 10.0

    volume = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
    y0r = n // 2  # yp = 0 -> y0 = 0 -> y0r = 0 - starting_y = n//2
    z0r = n // 2
    # A sharp spike at x-index 10 so the two floor choices (bracket [9,10] vs
    # [10,11]) give visibly different interpolated values.
    volume[z0r, y0r, 9] = 0.0
    volume[z0r, y0r, 10] = 1.0
    volume[z0r, y0r, 11] = 0.0

    # R_relion is a synthetic (non-orthonormal) matrix, chosen only to make
    # xp land exactly on xp0 at pixel (x=1, y=0) with yp = zp = 0.
    rotation = np.asarray(
        [
            [xp0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    default = np.asarray(
        relion_project_half(jnp.asarray(volume), jnp.asarray(rotation), n, r_max=n // 2, padding_factor=1)
    )
    quirked = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
            relion_acc_double_floorf_quirk=True,
        )
    )

    fx_off = xp0 - 9
    fx_on = xp0 - 10
    expected_default = 0.0 + fx_off * (1.0 - 0.0)
    expected_quirked = 1.0 + fx_on * (0.0 - 1.0)

    np.testing.assert_allclose(default[0, 1], expected_default, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(quirked[0, 1], expected_quirked, rtol=0.0, atol=1e-12)
    # The two floor choices select different neighbor pairs; the resulting
    # values differ by ~8e-7 here -- far above float64 rounding noise (~1e-15)
    # and proof the flag actually changes which bucket is used.
    assert abs(quirked[0, 1] - default[0, 1]) > 1e-8
