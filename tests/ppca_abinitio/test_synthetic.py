"""Tests for the synthetic data harness.

Pins the per-family invariants and the train/val split contract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticDataset,
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)
N_FULL = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
N_HALF_VOL = VOLUME_SHAPE[0] * VOLUME_SHAPE[1] * (VOLUME_SHAPE[2] // 2 + 1)


def _make_grid():
    return build_fixed_grid(healpix_order=2, max_shift=1)


# ---------------------------------------------------------------------------
# Family B (matched-grid heterogeneous) — primary positive control
# ---------------------------------------------------------------------------


def test_family_B_shapes_and_dtypes():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=3,
        n_images_train=10,
        n_images_val=4,
        sigma_real=0.5,
        seed=0,
    )

    assert isinstance(ds, SyntheticDataset)
    assert ds.family is SyntheticFamily.MATCHED_GRID_HET
    assert ds.n_img == 14
    assert ds.q == 3

    assert ds.mu_half_true.shape == (N_HALF_VOL,)
    assert ds.mu_half_true.dtype == jnp.complex128
    assert ds.U_half_true.shape == (3, N_HALF_VOL)
    assert ds.U_half_true.dtype == jnp.complex128
    assert ds.s_true.shape == (3,)
    assert ds.s_true.dtype == jnp.float64

    assert ds.batch_full.shape == (14, N_FULL)
    assert ds.batch_full.dtype == jnp.complex128
    assert ds.noise_variance_full.shape == (N_FULL,)
    assert ds.noise_variance_full.dtype == jnp.float64

    assert ds.r_true_idx.shape == (14,)
    assert ds.t_true_idx.shape == (14,)
    assert ds.alpha_true.shape == (14, 3)


def test_family_B_s_true_is_positive_and_descending():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=4,
        n_images_train=4,
        n_images_val=2,
        seed=0,
    )
    s = np.asarray(ds.s_true)
    assert np.all(s > 0)
    assert np.all(np.diff(s) <= 0), f"s_true not descending: {s}"


def test_family_B_alphas_match_s_variance():
    """alpha_i ~ N(0, diag(s)). With many samples, the empirical
    variance should match s within sampling noise."""
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=3,
        n_images_train=512,
        n_images_val=128,
        seed=42,
    )
    emp_var = np.var(ds.alpha_true, axis=0)
    s = np.asarray(ds.s_true)
    np.testing.assert_allclose(emp_var, s, rtol=0.25)


def test_family_B_noise_variance_recipe():
    """noise_variance must equal sigma_real² · N_full per spec Section 4.3."""
    grid = _make_grid()
    sigma_real = 0.7
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=sigma_real,
        seed=0,
    )
    expected = sigma_real**2 * N_FULL
    assert float(ds.noise_variance_full[0]) == pytest.approx(expected, rel=1e-12)
    assert bool(jnp.all(ds.noise_variance_full == ds.noise_variance_full[0]))


def test_family_B_pose_indices_in_range():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=32,
        n_images_val=16,
        seed=7,
    )
    assert int(ds.r_true_idx.min()) >= 0
    assert int(ds.r_true_idx.max()) < ds.n_rot
    assert int(ds.t_true_idx.min()) >= 0
    assert int(ds.t_true_idx.max()) < ds.n_trans


def test_family_B_batch_is_finite():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=4,
        seed=11,
    )
    assert bool(jnp.all(jnp.isfinite(jnp.real(ds.batch_full))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(ds.batch_full))))


def test_family_B_batch_is_hermitian_symmetric():
    """The image batch is generated from real-space sources (clean
    projection + real noise), so each row should be the FT of a
    real volume — i.e. when round-tripped via idft2 → real, the
    imaginary energy fraction is below 1e-10."""
    import recovar.core.fourier_transform_utils as ftu

    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        seed=0,
    )
    for i in range(ds.n_img):
        ft = jnp.asarray(ds.batch_full[i]).reshape(IMAGE_SHAPE)
        real_image = ftu.get_idft2(ft)
        imag_frac = float(jnp.sum(jnp.abs(real_image.imag) ** 2) / jnp.sum(jnp.abs(real_image) ** 2))
        assert imag_frac < 1e-12, f"image {i} has imag fraction {imag_frac}"


# ---------------------------------------------------------------------------
# Family A (null) — negative control
# ---------------------------------------------------------------------------


def test_family_A_s_is_zero():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.NULL,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=3,
        n_images_train=8,
        n_images_val=4,
        seed=0,
    )
    np.testing.assert_array_equal(np.asarray(ds.s_true), np.zeros(3))


def test_family_A_alphas_are_zero():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.NULL,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=3,
        n_images_train=8,
        n_images_val=4,
        seed=0,
    )
    np.testing.assert_array_equal(ds.alpha_true, np.zeros((12, 3)))


def test_family_A_U_is_still_present():
    """Family A keeps the same U structure as family B (only s=0
    suppresses the contribution); this lets the same kernel call be
    used as a true negative-control without changing q."""
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.NULL,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=3,
        n_images_train=4,
        n_images_val=2,
        seed=0,
    )
    assert ds.U_half_true.shape == (3, N_HALF_VOL)
    # U should not be zero
    assert float(jnp.sum(jnp.abs(ds.U_half_true) ** 2)) > 0


# ---------------------------------------------------------------------------
# Train/val split + reproducibility
# ---------------------------------------------------------------------------


def test_train_val_split_sizes_and_disjoint():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=20,
        n_images_val=8,
        seed=0,
    )
    assert ds.train_idx.shape == (20,)
    assert ds.val_idx.shape == (8,)
    overlap = np.intersect1d(ds.train_idx, ds.val_idx)
    assert overlap.size == 0
    union = np.union1d(ds.train_idx, ds.val_idx)
    assert union.size == 28


def test_seed_reproducibility():
    grid = _make_grid()
    kwargs = dict(
        family=SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=4,
        seed=123,
    )
    ds1 = make_synthetic_fixed_grid_dataset(**kwargs)
    ds2 = make_synthetic_fixed_grid_dataset(**kwargs)

    np.testing.assert_array_equal(np.asarray(ds1.batch_full), np.asarray(ds2.batch_full))
    np.testing.assert_array_equal(ds1.r_true_idx, ds2.r_true_idx)
    np.testing.assert_array_equal(ds1.alpha_true, ds2.alpha_true)


def test_only_family_E_is_unimplemented():
    grid = _make_grid()
    with pytest.raises(NotImplementedError):
        make_synthetic_fixed_grid_dataset(
            SyntheticFamily.CTF_ZERO_HET,
            volume_shape=VOLUME_SHAPE,
            image_shape=IMAGE_SHAPE,
            grid=grid,
            q=2,
            n_images_train=2,
            n_images_val=1,
        )


# ---------------------------------------------------------------------------
# Family C — misspecified pose (off-grid jitter)
# ---------------------------------------------------------------------------


def test_family_C_runs_and_records_grid_indices():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MISSPECIFIED_POSE,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=4,
        sigma_real=0.3,
        seed=0,
    )
    assert ds.family is SyntheticFamily.MISSPECIFIED_POSE
    assert ds.batch_full.shape == (12, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    # The recorded r_true_idx / t_true_idx are the **nearest** grid pose,
    # not the exact off-grid pose.
    assert ds.r_true_idx.shape == (12,)
    assert int(ds.r_true_idx.max()) < ds.n_rot
    assert ds.t_true_idx.shape == (12,)


def test_family_C_batch_differs_from_family_B():
    """Same seed, same q, family B vs family C should produce
    different batches because C jitters poses."""
    grid = _make_grid()
    kwargs = dict(
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.0,  # zero noise so the difference is purely from jitter
        seed=0,
    )
    ds_b = make_synthetic_fixed_grid_dataset(SyntheticFamily.MATCHED_GRID_HET, **kwargs)
    ds_c = make_synthetic_fixed_grid_dataset(SyntheticFamily.MISSPECIFIED_POSE, **kwargs)
    # The batches must differ
    assert not np.allclose(np.asarray(ds_b.batch_full), np.asarray(ds_c.batch_full))


def test_family_C_pose_jitter_range():
    """Custom pose_jitter_deg_range. Verifies the parameter is plumbed."""
    grid = _make_grid()
    # Tiny jitter — batches should be CLOSE to family B
    ds_tiny = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MISSPECIFIED_POSE,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.0,
        pose_jitter_deg_range=(0.01, 0.02),
        pose_jitter_trans_range=(0.001, 0.002),
        seed=0,
    )
    ds_big = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MISSPECIFIED_POSE,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.0,
        pose_jitter_deg_range=(5.0, 10.0),
        pose_jitter_trans_range=(1.0, 2.0),
        seed=0,
    )
    # Bigger jitter → bigger deviation from the matched-grid version
    ds_b = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.0,
        seed=0,
    )
    diff_tiny = float(jnp.sum(jnp.abs(ds_tiny.batch_full - ds_b.batch_full) ** 2))
    diff_big = float(jnp.sum(jnp.abs(ds_big.batch_full - ds_b.batch_full) ** 2))
    assert diff_big > diff_tiny


# ---------------------------------------------------------------------------
# Family D — per-particle contrast
# ---------------------------------------------------------------------------


def test_family_D_records_contrast_factors():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.PER_PARTICLE_CONTRAST,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=16,
        n_images_val=4,
        sigma_real=0.3,
        seed=0,
    )
    assert ds.family is SyntheticFamily.PER_PARTICLE_CONTRAST
    assert ds.contrast_true is not None
    assert ds.contrast_true.shape == (20,)
    # Default range is [0.8, 1.2]
    assert float(ds.contrast_true.min()) >= 0.8 - 1e-9
    assert float(ds.contrast_true.max()) <= 1.2 + 1e-9


def test_family_D_contrast_actually_scales_clean_image():
    """With sigma=0 and contrast in [2, 2] (so c=2), the image should
    equal twice the family-B image (modulo float roundoff)."""
    grid = _make_grid()
    kwargs = dict(
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.0,
        seed=0,
    )
    ds_b = make_synthetic_fixed_grid_dataset(SyntheticFamily.MATCHED_GRID_HET, **kwargs)
    ds_d = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.PER_PARTICLE_CONTRAST,
        contrast_range=(2.0, 2.0),
        **kwargs,
    )
    # Every image in D should equal 2 * the matching image in B
    np.testing.assert_allclose(
        np.asarray(ds_d.batch_full),
        2.0 * np.asarray(ds_b.batch_full),
        rtol=1e-10,
        atol=1e-12,
    )


def test_family_AB_contrast_field_is_none():
    grid = _make_grid()
    ds_a = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.NULL,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=2,
        n_images_val=1,
        seed=0,
    )
    ds_b = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=2,
        n_images_val=1,
        seed=0,
    )
    assert ds_a.contrast_true is None
    assert ds_b.contrast_true is None
