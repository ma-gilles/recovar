"""Tests for the synthetic data harness.

Pins the per-family invariants and the train/val split contract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticDataset,
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
    subset_synthetic_dataset,
)

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)
N_FULL = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
N_HALF_VOL = VOLUME_SHAPE[0] * VOLUME_SHAPE[1] * (VOLUME_SHAPE[2] // 2 + 1)


def _make_grid():
    return build_fixed_grid(healpix_order=2, max_shift=1)


def _make_external_volumes(K=5):
    z = np.linspace(-1.0, 1.0, VOLUME_SHAPE[0])
    y = np.linspace(-1.0, 1.0, VOLUME_SHAPE[1])
    x = np.linspace(-1.0, 1.0, VOLUME_SHAPE[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    base = np.exp(-(X**2 + Y**2 + Z**2) / (2 * 0.45**2))
    pc1 = X * base
    pc2 = (Y - 0.3 * Z) * base
    coeffs = np.array(
        [
            [-1.0, -0.4],
            [-0.4, 0.2],
            [0.0, 0.0],
            [0.7, -0.1],
            [1.2, 0.5],
        ],
        dtype=np.float64,
    )
    coeffs = coeffs[:K]
    vols = np.stack([base + a * pc1 + b * pc2 for a, b in coeffs], axis=0)
    return vols.astype(np.float64)


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


def test_external_volumes_real_sets_mu_U_and_empirical_spectrum():
    grid = _make_grid()
    ext = _make_external_volumes(K=5)
    q = 2
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=q,
        n_images_train=16,
        n_images_val=4,
        sigma_real=0.2,
        seed=0,
        external_volumes_real=ext,
    )

    mu_real = ext.mean(axis=0)
    centered_flat = (ext - mu_real[None]).reshape(ext.shape[0], -1)
    _u_img, s_svd, vh = np.linalg.svd(centered_flat, full_matrices=False)
    mu_half_expected = ftu.get_dft3_real(jnp.asarray(mu_real)).reshape(-1)
    U_half_expected = jnp.stack(
        [
            ftu.get_dft3_real(jnp.asarray(vh[k].reshape(VOLUME_SHAPE))).reshape(-1)
            for k in range(q)
        ]
    )
    s_expected = (s_svd[:q] ** 2 / (ext.shape[0] - 1)).astype(np.float64)

    np.testing.assert_allclose(np.asarray(ds.mu_half_true), np.asarray(mu_half_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ds.s_true), s_expected, rtol=1e-10, atol=1e-10)
    assert projector_frobenius_error(ds.U_half_true, U_half_expected, VOLUME_SHAPE) < 1e-10


def test_external_volumes_real_null_family_still_zeroes_s():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.NULL,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=4,
        seed=0,
        external_volumes_real=_make_external_volumes(K=4),
    )
    np.testing.assert_array_equal(np.asarray(ds.s_true), np.zeros(2))


def test_external_volumes_real_validates_shape_and_available_rank():
    grid = _make_grid()
    with pytest.raises(ValueError, match="must be 4D"):
        make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=VOLUME_SHAPE,
            image_shape=IMAGE_SHAPE,
            grid=grid,
            q=2,
            n_images_train=4,
            n_images_val=2,
            seed=0,
            external_volumes_real=np.zeros(VOLUME_SHAPE, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="spatial shape"):
        make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=VOLUME_SHAPE,
            image_shape=IMAGE_SHAPE,
            grid=grid,
            q=2,
            n_images_train=4,
            n_images_val=2,
            seed=0,
            external_volumes_real=np.zeros((3, 6, 6, 6), dtype=np.float64),
        )

    with pytest.raises(ValueError, match="exceeds available PCs"):
        make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=VOLUME_SHAPE,
            image_shape=IMAGE_SHAPE,
            grid=grid,
            q=3,
            n_images_train=4,
            n_images_val=2,
            seed=0,
            external_volumes_real=_make_external_volumes(K=2),
        )


def test_external_volumes_discrete_mode_records_labels_and_coords():
    grid = _make_grid()
    ext = _make_external_volumes(K=5)
    q = 2
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=q,
        n_images_train=12,
        n_images_val=4,
        sigma_real=0.0,
        seed=0,
        external_volumes_real=ext,
        external_sampling_mode="discrete_volumes",
    )

    centered_flat = (ext - ext.mean(axis=0, keepdims=True)).reshape(ext.shape[0], -1)
    _u_img, _s_svd, vh = np.linalg.svd(centered_flat, full_matrices=False)
    coords_expected = centered_flat @ vh[:q].T

    assert ds.state_label_true is not None
    assert ds.state_coords_true is not None
    assert ds.state_label_true.shape == (16,)
    assert ds.state_coords_true.shape == (5, q)
    np.testing.assert_allclose(ds.state_coords_true, coords_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ds.alpha_true, ds.state_coords_true[ds.state_label_true], rtol=1e-10, atol=1e-10)
    assert int(ds.state_label_true.min()) >= 0
    assert int(ds.state_label_true.max()) < ext.shape[0]


def test_external_sampling_mode_validates_inputs():
    grid = _make_grid()
    with pytest.raises(ValueError, match="external_sampling_mode"):
        make_synthetic_fixed_grid_dataset(
            SyntheticFamily.MATCHED_GRID_HET,
            volume_shape=VOLUME_SHAPE,
            image_shape=IMAGE_SHAPE,
            grid=grid,
            q=2,
            n_images_train=4,
            n_images_val=2,
            seed=0,
            external_sampling_mode="bad_mode",
        )


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


def test_subset_synthetic_dataset_slices_per_image_fields_and_resets_split():
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
    sub = subset_synthetic_dataset(ds, ds.train_idx[:5])

    assert sub.n_img == 5
    np.testing.assert_array_equal(sub.train_idx, np.arange(5, dtype=np.int32))
    assert sub.val_idx.shape == (0,)
    np.testing.assert_allclose(np.asarray(sub.batch_full), np.asarray(ds.batch_full)[ds.train_idx[:5]])
    np.testing.assert_array_equal(sub.r_true_idx, ds.r_true_idx[ds.train_idx[:5]])
    np.testing.assert_array_equal(sub.t_true_idx, ds.t_true_idx[ds.train_idx[:5]])
    np.testing.assert_allclose(sub.alpha_true, ds.alpha_true[ds.train_idx[:5]])
    np.testing.assert_allclose(np.asarray(sub.mu_half_true), np.asarray(ds.mu_half_true))
    np.testing.assert_allclose(np.asarray(sub.U_half_true), np.asarray(ds.U_half_true))


def test_subset_synthetic_dataset_validates_indices():
    grid = _make_grid()
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=8,
        n_images_val=4,
        seed=0,
    )

    with pytest.raises(ValueError, match="non-empty"):
        subset_synthetic_dataset(ds, [])
    with pytest.raises(IndexError, match="out of bounds"):
        subset_synthetic_dataset(ds, [ds.n_img])


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
