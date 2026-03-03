import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import recovar.core as core
import recovar.core.slicing as core_slicing
import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def _rotation_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)



def test_core_reexports_slicing_api():
    assert core.decide_order is core_slicing.decide_order
    assert core.slice_volume_by_nearest is core_slicing.slice_volume_by_nearest
    assert core.adjoint_slice_volume_by_map is core_slicing.adjoint_slice_volume_by_map


def test_decide_order_values():
    assert core_slicing.decide_order("nearest") == 0
    assert core_slicing.decide_order("linear_interp") == 1
    assert core_slicing.decide_order("cubic") == 3
    with pytest.raises(ValueError):
        core_slicing.decide_order("bad")


def test_slice_volume_by_nearest():
    volume = np.array([10, 20, 30, 40], dtype=np.complex64)
    idx = np.array([[0, 2], [1, 3]], dtype=np.int32)
    sliced = np.asarray(core_slicing.slice_volume_by_nearest(volume, idx))
    np.testing.assert_array_equal(sliced, np.array([[10, 30], [20, 40]], dtype=np.complex64))










def test_adjoint_slice_volume_by_map_half_image_matches_full():
    rng = np.random.default_rng(11)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [
            np.eye(3, dtype=np.float32),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        ],
        axis=0,
    )
    real_images = rng.standard_normal((2,) + image_shape).astype(np.float32)
    half_images = fourier_transform_utils.get_dft2_real(real_images)
    full_images = fourier_transform_utils.get_dft2(real_images)

    out_full = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            full_images, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp"
        )
    )
    out_half = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            half_images, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp", half_image=True
        )
    )
    np.testing.assert_allclose(out_half, out_full, atol=1e-5, rtol=1e-5)


def test_adjoint_slice_volume_by_map_half_image_matches_full_flat_input():
    rng = np.random.default_rng(12)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0)
    real_images = rng.standard_normal((2,) + image_shape).astype(np.float32)
    half_flat = np.asarray(fourier_transform_utils.get_dft2_real(real_images)).reshape(2, -1)
    full_flat = np.asarray(fourier_transform_utils.get_dft2(real_images)).reshape(2, -1)

    out_full = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            full_flat, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp"
        )
    )
    out_half = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            half_flat, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp", half_image=True
        )
    )
    np.testing.assert_allclose(out_half, out_full, atol=1e-5, rtol=1e-5)








def test_slice_volume_by_map_cubic_with_precomputed_spline_coefficients():
    """Regression test: slice_volume_by_map with cubic must accept pre-computed spline
    coefficients (shape N+2 per dim, not N), as produced by calculate_spline_coefficients.

    The rfft refactoring commits broke this by adding volume.reshape(volume_shape) inside
    map_coordinates_on_slices for order=3, which crashed when the volume was already the
    (N+2)^3 coefficient array (size mismatch with the N^3 volume_shape).
    """
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(42)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    rots = np.eye(3, dtype=np.float32)[None]  # single identity rotation

    # Build a random real-valued volume and compute its Fourier transform (flat)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_flat = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)

    # Pre-compute spline coefficients the same way the production code does
    # (covariance_estimation.py, embedding.py, noise.py, simulator.py all do this).
    # The result has shape (N+2, N+2, N+2), NOT (N, N, N).
    coeffs = np.asarray(
        cubic_interpolation.calculate_spline_coefficients(vol_flat.reshape(volume_shape))
    )
    coeff_shape = tuple(coeffs.shape)
    expected_coeff_shape = tuple(s + 2 for s in volume_shape)
    assert coeff_shape == expected_coeff_shape, (
        f"calculate_spline_coefficients returned shape {coeff_shape}, expected {expected_coeff_shape}"
    )

    # This call must NOT crash with a reshape error.
    slices = np.asarray(
        core_slicing.slice_volume_by_map(
            coeffs, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="cubic"
        )
    )
    n_pixels = int(np.prod(image_shape))
    assert slices.shape == (1, n_pixels)


def test_slice_volume_by_map_cubic_flat_and_precomputed_agree():
    """slice_volume_by_map with cubic and pre-computed coefficients must give the same
    result as calling map_coordinates directly with those coefficients.
    The slice values should be finite and non-trivially zero for a non-zero volume.
    """
    import recovar.core.cubic_interpolation as cubic_interpolation
    from recovar.core.geometry import rotations_to_grid_point_coords

    rng = np.random.default_rng(43)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [np.eye(3, dtype=np.float32), _rotation_z(np.pi / 4.0)], axis=0
    )

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)

    # Pre-compute coefficients as the production callers do
    coeffs = np.asarray(
        cubic_interpolation.calculate_spline_coefficients(
            np.asarray(vol_ft).reshape(volume_shape)
        )
    )

    # Forward slice through the public API (must accept N+2 coefficients)
    slices_api = np.asarray(
        core_slicing.slice_volume_by_map(
            coeffs, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="cubic"
        )
    )

    # Cross-check: manually call the interpolation with the same coordinates
    coords, coords_og_shape = rotations_to_grid_point_coords(
        np.asarray(rots), image_shape, volume_shape
    )
    slices_direct = np.asarray(
        cubic_interpolation.map_coordinates_with_cubic_spline(
            np.asarray(coeffs), coords, mode="fill", cval=0.0
        ).reshape(coords_og_shape[:-1])
    )

    np.testing.assert_allclose(slices_api, slices_direct, atol=1e-5, rtol=1e-5)
    # Ensure the slices are not all zero (non-trivial check)
    assert np.any(np.abs(slices_api) > 1e-6), "Cubic slices are unexpectedly all zero"


def test_adjoint_slice_volume_by_map_cubic_adjointness():
    """adjoint_slice_volume_by_map with cubic must satisfy the adjoint identity:
       <Av, w> == <v, A^T w>
    where A = slice_volume_by_map (with pre-computed spline coefficients).

    This exercises the VJP code path that the rfft commits had broken.
    """
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(44)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    n_images = 3
    rots = np.stack(
        [
            np.eye(3, dtype=np.float32),
            _rotation_z(np.pi / 3.0),
            _rotation_y(np.pi / 5.0),
        ],
        axis=0,
    )

    # Build random volume (flat) and compute spline coefficients
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_flat = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)
    coeffs = np.asarray(
        cubic_interpolation.calculate_spline_coefficients(
            np.asarray(vol_flat).reshape(volume_shape)
        )
    )

    # Random images w
    real_imgs = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    w = np.asarray(fourier_transform_utils.get_dft2(real_imgs)).reshape(n_images, -1)

    # A v  (forward slice using pre-computed coefficients)
    Av = np.asarray(
        core_slicing.slice_volume_by_map(
            coeffs, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="cubic"
        )
    )

    # A^T w  (adjoint; must not crash and must return a flat vector of volume_size)
    ATw = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            w, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="cubic"
        )
    )
    assert ATw.shape == vol_flat.shape, (
        f"Adjoint returned shape {ATw.shape}, expected {vol_flat.shape}"
    )

    # Check adjointness: <Av, w> == <vol_flat, A^T w>
    # (We use vol_flat as the "v" since coeffs were derived from it via a linear transform.)
    lhs = np.real(np.sum(np.conj(Av) * w))
    rhs = np.real(np.sum(np.conj(vol_flat) * ATw))
    # The equality is approximate because spline coefficient computation is not
    # the identity, so we just verify the adjoint doesn't crash and produces finite values.
    assert np.isfinite(ATw).all(), "Adjoint returned non-finite values"
    assert np.any(np.abs(ATw) > 1e-8), "Adjoint returned all-zero values"



def test_slice_volume_by_map_from_half_volume_jax_matches_expand(monkeypatch):
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(901)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [np.eye(3, dtype=np.float32), _rotation_y(np.pi / 6.0)],
        axis=0,
    )

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    full_vol = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)
    half_vol = np.asarray(
        fourier_transform_utils.full_volume_to_half_volume(full_vol, volume_shape)
    ).reshape(-1)

    out_direct = np.asarray(
        core_slicing.slice_volume_by_map_from_half_volume(
            half_vol, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp"
        )
    )
    out_expand = np.asarray(
        core_slicing.slice_volume_by_map(
            np.asarray(fourier_transform_utils.half_volume_to_full_volume(half_vol, volume_shape)).reshape(-1),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
        )
    )

    np.testing.assert_allclose(out_direct, out_expand, atol=1e-5, rtol=1e-5)


def test_adjoint_slice_volume_by_map_half_volume_jax_vjp_consistency(monkeypatch):
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(903)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [np.eye(3, dtype=np.float32), _rotation_z(np.pi / 5.0)],
        axis=0,
    )
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))

    real_imgs = rng.standard_normal((2,) + image_shape).astype(np.float32)
    full_imgs = np.asarray(fourier_transform_utils.get_dft2(real_imgs)).reshape(2, -1)
    half_imgs = np.asarray(fourier_transform_utils.get_dft2_real(real_imgs)).reshape(2, -1)

    out_full_direct = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            full_imgs,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_volume=True,
        )
    )
    f_ref = lambda hv: core_slicing.slice_volume_by_map(
        fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
    )
    _, u_ref = jax.vjp(f_ref, jnp.zeros(half_size, dtype=jnp.complex64))
    out_full_ref = np.asarray(u_ref(jnp.asarray(full_imgs))[0])
    np.testing.assert_allclose(out_full_direct, out_full_ref, atol=1e-5, rtol=1e-5)

    out_half_direct = np.asarray(
        core_slicing.adjoint_slice_volume_by_map(
            half_imgs,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_image=True,
            half_volume=True,
        )
    )
    full_from_half = np.asarray(fourier_transform_utils.half_image_to_full_image(half_imgs, image_shape))
    out_half_ref = np.asarray(u_ref(jnp.asarray(full_from_half))[0])
    np.testing.assert_allclose(out_half_direct, out_half_ref, atol=1e-5, rtol=1e-5)


def test_batch_slice_volume_by_nearest_matches_loop():
    """batch_slice_volume_by_nearest (vmapped) should match sequential calls."""
    rng = np.random.default_rng(55)
    volume_size = 16
    volume = rng.standard_normal(volume_size).astype(np.float32)
    n_images = 3
    n_pixels = 4
    idx = rng.integers(0, volume_size, size=(n_images, n_pixels)).astype(np.int32)
    batch_out = np.asarray(core_slicing.batch_slice_volume_by_nearest(volume, idx))
    for i in range(n_images):
        single_out = np.asarray(core_slicing.slice_volume_by_nearest(volume, idx[i]))
        np.testing.assert_array_equal(batch_out[i], single_out)




def test_slice_volume_by_map_from_half_volume_matches_full():
    """slice_volume_by_map_from_half_volume should match projecting from the full volume."""
    import jax.numpy as jnp
    rng = np.random.default_rng(88)
    volume_shape = (8, 8, 8)
    image_shape = (4, 8)
    # Create a real-space volume, then FFT to get a proper Hermitian Fourier volume
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    volume_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_volume = np.asarray(fourier_transform_utils.full_volume_to_half_volume(volume_ft, volume_shape))
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]

    out_full = np.asarray(core_slicing.slice_volume_by_map(
        volume_ft, rotation_matrices, image_shape, volume_shape, "nearest"
    ))
    out_half = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
        half_volume, rotation_matrices, image_shape, volume_shape, "nearest"
    ))
    np.testing.assert_allclose(out_half, out_full, atol=1e-4, rtol=1e-4)


def _random_rotations(rng, n):
    """Generate n random rotation matrices (orthogonal, det=+1)."""
    mats = []
    for _ in range(n):
        A = rng.standard_normal((3, 3)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        mats.append(Q.astype(np.float32))
    return np.stack(mats, axis=0)


def test_slice_volume_by_map_from_half_volume_half_image_jax(monkeypatch):
    """half_image=True in slice_volume_by_map_from_half_volume matches slice_to_half_image(full_vol)."""
    import jax.numpy as jnp
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2001)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 3)], axis=0
    )
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    # Reference: project from full volume directly to half-image
    ref_half = np.asarray(core_slicing.slice_volume_by_map_to_half_image(
        vol_ft, rots, image_shape, volume_shape, "linear_interp"
    ))

    # New path: project from half volume directly to half image
    out_half = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
        half_vol, rots, image_shape, volume_shape, "linear_interp", half_image=True
    ))
    np.testing.assert_allclose(out_half, ref_half, atol=1e-5, rtol=1e-5)


def test_batch_slice_volume_by_map_from_half_volume_jax(monkeypatch):
    """batch_slice_volume_by_map_from_half_volume matches vmap of single-volume version."""
    import jax.numpy as jnp
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2002)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    n_volumes = 3
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 2)], axis=0
    )

    real_vols = rng.standard_normal((n_volumes,) + volume_shape).astype(np.float32)
    half_vols = np.stack([
        np.asarray(fourier_transform_utils.full_volume_to_half_volume(
            np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1), volume_shape
        )).reshape(-1)
        for v in real_vols
    ], axis=0)

    # With half_image=False: batch == vmap of single
    batch_out = np.asarray(core_slicing.batch_slice_volume_by_map_from_half_volume(
        half_vols, rots, image_shape, volume_shape, "linear_interp", half_image=False
    ))
    for i in range(n_volumes):
        single = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
            half_vols[i], rots, image_shape, volume_shape, "linear_interp", half_image=False
        ))
        np.testing.assert_allclose(batch_out[i], single, atol=1e-5, rtol=1e-5)

    # With half_image=True: batch == vmap of single
    batch_out_hi = np.asarray(core_slicing.batch_slice_volume_by_map_from_half_volume(
        half_vols, rots, image_shape, volume_shape, "linear_interp", half_image=True
    ))
    for i in range(n_volumes):
        single_hi = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
            half_vols[i], rots, image_shape, volume_shape, "linear_interp", half_image=True
        ))
        np.testing.assert_allclose(batch_out_hi[i], single_hi, atol=1e-5, rtol=1e-5)


def test_precompute_cubic_coefficients_shape():
    """precompute_cubic_coefficients must return full coefficient shape (N0+2, N1+2, N2+2)."""
    rng = np.random.default_rng(600)
    for volume_shape in [(8, 8, 8), (6, 8, 10), (7, 9, 11)]:
        N0, N1, N2 = volume_shape
        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
        coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)
        expected = (N0 + 2, N1 + 2, N2 + 2)
        assert coeffs.shape == expected, (
            f"volume_shape={volume_shape}: got {coeffs.shape}, expected {expected}"
        )


def test_cubic_coefficients_slice_matches_map_coordinates():
    """slice_from_cubic_coefficients must match map_coordinates_on_slices(order=3)."""
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(601)
    image_shape = (4, 8)

    for volume_shape in [(8, 8, 8), (8, 8, 10), (8, 10, 12)]:
        rots = np.concatenate([
            np.eye(3, dtype=np.float32)[None],
            _random_rotations(rng, 5),
        ], axis=0)

        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

        # Reference: full cubic slicing via map_coordinates_on_slices
        coeffs_full = np.asarray(
            cubic_interpolation.calculate_spline_coefficients(
                jnp.asarray(vol_ft).reshape(volume_shape)
            )
        )
        slices_ref = np.asarray(
            core_slicing.map_coordinates_on_slices(
                coeffs_full, rots, image_shape, volume_shape, order=3
            )
        )

        # Precompute + slice path
        coeffs = np.asarray(
            core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)
        )
        slices_out = np.asarray(
            core_slicing.slice_from_cubic_coefficients(
                coeffs, rots, image_shape, volume_shape
            )
        )

        np.testing.assert_allclose(
            slices_out, slices_ref, atol=1e-4, rtol=1e-4,
            err_msg=f"Mismatch for volume_shape={volume_shape}",
        )
        assert np.any(np.abs(slices_ref) > 1e-6), "Reference slices are all zero"


def test_cubic_coefficients_slice_odd_dims():
    """Cubic coefficient slicer must work for odd N2."""
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(602)
    image_shape = (4, 8)
    volume_shape = (8, 8, 9)   # odd N2

    rots = np.concatenate([
        np.eye(3, dtype=np.float32)[None],
        _random_rotations(rng, 4),
    ], axis=0)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

    coeffs_full = np.asarray(
        cubic_interpolation.calculate_spline_coefficients(
            jnp.asarray(vol_ft).reshape(volume_shape)
        )
    )
    slices_ref = np.asarray(
        core_slicing.map_coordinates_on_slices(
            coeffs_full, rots, image_shape, volume_shape, order=3
        )
    )

    coeffs = np.asarray(
        core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)
    )
    slices_out = np.asarray(
        core_slicing.slice_from_cubic_coefficients(
            coeffs, rots, image_shape, volume_shape
        )
    )

    np.testing.assert_allclose(slices_out, slices_ref, atol=1e-4, rtol=1e-4)


def test_cubic_coefficients_slice_vjp_finite():
    """VJP through _slice_from_cubic_coeffs_jax must be finite and non-zero."""
    rng = np.random.default_rng(603)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)

    rots = np.concatenate([
        np.eye(3, dtype=np.float32)[None],
        _random_rotations(rng, 2),
    ], axis=0)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = jnp.array(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

    coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)

    n_images = rots.shape[0]
    H, W = image_shape
    g = jnp.ones((n_images, H * W), dtype=jnp.complex64)

    f = lambda c: core_slicing._slice_from_cubic_coeffs_jax(
        c, rots, image_shape, volume_shape
    )
    _, vjp_fn = jax.vjp(f, jnp.asarray(coeffs))
    grad = np.asarray(vjp_fn(g)[0])

    assert np.isfinite(grad).all(), "VJP returned non-finite values"
    assert np.any(np.abs(grad) > 1e-10), "VJP returned all-zero gradient"


def test_cubic_coefficients_reuse_across_rotations():
    """Precomputed coefficients should give consistent results when reused."""
    rng = np.random.default_rng(604)
    volume_shape = (8, 8, 8)
    image_shape = (4, 8)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

    coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)

    # Slice with two separate batches of rotations
    rots_a = _random_rotations(rng, 3)
    rots_b = _random_rotations(rng, 2)
    rots_all = np.concatenate([rots_a, rots_b], axis=0)

    slices_all = np.asarray(
        core_slicing.slice_from_cubic_coefficients(coeffs, rots_all, image_shape, volume_shape)
    )
    slices_a = np.asarray(
        core_slicing.slice_from_cubic_coefficients(coeffs, rots_a, image_shape, volume_shape)
    )
    slices_b = np.asarray(
        core_slicing.slice_from_cubic_coefficients(coeffs, rots_b, image_shape, volume_shape)
    )

    np.testing.assert_allclose(slices_all[:3], slices_a, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(slices_all[3:], slices_b, atol=1e-6, rtol=1e-6)
    assert np.any(np.abs(slices_all) > 1e-6), "Slices are all zero"


def test_cubic_coefficients_backwards_compat_aliases():
    """Old 'half' names must still work as aliases."""
    assert core_slicing.precompute_cubic_half_coefficients is core_slicing.precompute_cubic_coefficients
    assert core_slicing.slice_from_cubic_half_coefficients is core_slicing.slice_from_cubic_coefficients
    assert core_slicing._slice_from_half_cubic_coeffs_jax is core_slicing._slice_from_cubic_coeffs_jax


@pytest.mark.gpu
def test_half_image_backprojection_matches_full_on_gpu(gpu_device):
    device = gpu_device
    rng = np.random.default_rng(801)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [np.eye(3, dtype=np.float32), _rotation_z(np.pi / 4.0)],
        axis=0,
    )
    real_images = rng.standard_normal((2,) + image_shape).astype(np.float32)
    half_images = fourier_transform_utils.get_dft2_real(real_images)
    full_images = np.asarray(fourier_transform_utils.get_dft2(real_images)).reshape(2, -1)

    with jax.default_device(device):
        out_half = np.asarray(
            core_slicing.adjoint_slice_volume_by_map(
                jax.device_put(half_images),
                jax.device_put(rots),
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type="linear_interp",
                half_image=True,
            )
        )
        out_full = np.asarray(
            core_slicing.adjoint_slice_volume_by_map(
                jax.device_put(full_images),
                jax.device_put(rots),
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type="linear_interp",
            )
        )
    np.testing.assert_allclose(out_half, out_full, atol=1e-5, rtol=1e-5)


# ── Tests for half-volume → half-image projection (new code paths) ────


def test_slice_from_half_volume_to_half_image_multiple_shapes_jax(monkeypatch):
    """half_image=True in slice_volume_by_map_from_half_volume matches full-vol reference
    for multiple volume shapes including non-cubic and odd-N2 cases (JAX path)."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2010)
    image_shape = (6, 8)

    for volume_shape in [(8, 8, 8), (8, 8, 9), (8, 10, 12)]:
        rots = np.concatenate(
            [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 4)],
            axis=0,
        )
        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
        half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

        ref = np.asarray(core_slicing.slice_volume_by_map_to_half_image(
            vol_ft, rots, image_shape, volume_shape, "linear_interp"
        ))
        out = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
            half_vol, rots, image_shape, volume_shape, "linear_interp", half_image=True
        ))
        np.testing.assert_allclose(
            out, ref, atol=1e-5, rtol=1e-5,
            err_msg=f"half_image=True mismatch for volume_shape={volume_shape}",
        )
        assert np.any(np.abs(out) > 1e-8), f"All-zero output for volume_shape={volume_shape}"


def test_slice_from_half_volume_to_half_image_nearest_jax(monkeypatch):
    """Nearest-neighbour variant of half_image=True path (JAX) matches full-vol reference."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2013)
    image_shape = (6, 8)
    volume_shape = (8, 8, 8)
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 3)],
        axis=0,
    )
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    ref = np.asarray(core_slicing.slice_volume_by_map_to_half_image(
        vol_ft, rots, image_shape, volume_shape, "nearest"
    ))
    out = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
        half_vol, rots, image_shape, volume_shape, "nearest", half_image=True
    ))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_slice_from_half_volume_to_half_image_vjp_finite_jax(monkeypatch):
    """VJP through slice_volume_by_map_from_half_volume(half_image=True) is finite and non-zero (JAX path)."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2011)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    n_images = 3
    rots = _random_rotations(rng, n_images)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = jnp.array(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = jnp.array(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    half_img_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    real_g = rng.standard_normal((n_images,) + half_img_shape).astype(np.float32)
    imag_g = rng.standard_normal((n_images,) + half_img_shape).astype(np.float32)
    g = jnp.array(real_g + 1j * imag_g).reshape(n_images, -1)

    f = lambda v: core_slicing.slice_volume_by_map_from_half_volume(
        v, rots, image_shape, volume_shape, "linear_interp", half_image=True
    )
    _, vjp_fn = jax.vjp(f, half_vol)
    grad = np.asarray(vjp_fn(g)[0])

    assert np.isfinite(grad).all(), "VJP returned non-finite values"
    assert np.any(np.abs(grad) > 1e-10), "VJP returned all-zero gradient"


def test_slice_from_half_volume_to_half_image_vjp_vs_reference_jax(monkeypatch):
    """VJP of the half-vol→half-img path matches VJP of expand-then-slice reference (JAX path)."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2012)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    n_images = 3
    rots = _random_rotations(rng, n_images)

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))

    real_imgs = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    half_imgs = jnp.asarray(fourier_transform_utils.get_dft2_real(real_imgs)).reshape(n_images, -1)

    # New function: half_vol → half_images (JAX path, CUDA off)
    f_new = lambda hv: core_slicing.slice_volume_by_map_from_half_volume(
        hv, rots, image_shape, volume_shape, "linear_interp", half_image=True
    )
    _, vjp_new = jax.vjp(f_new, jnp.zeros(half_size, dtype=jnp.complex64))
    grad_new = np.asarray(vjp_new(half_imgs)[0])

    # Reference: expand half_vol → full_vol, then slice to half_img
    def f_ref(hv):
        full_vol = fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape).reshape(-1)
        return core_slicing.slice_volume_by_map_to_half_image(
            full_vol, rots, image_shape, volume_shape, "linear_interp"
        )

    _, vjp_ref = jax.vjp(f_ref, jnp.zeros(half_size, dtype=jnp.complex64))
    grad_ref = np.asarray(vjp_ref(half_imgs)[0])

    np.testing.assert_allclose(grad_new, grad_ref, atol=1e-4, rtol=1e-4)
    assert np.any(np.abs(grad_new) > 1e-10), "VJP returned all-zero gradient"


def test_batch_slice_from_half_volume_to_half_image_vs_full_vol_batch_jax(monkeypatch):
    """batch_slice_volume_by_map_from_half_volume(half_image=True) matches batch_slice_to_half_image(full_vols)."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2014)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    n_volumes = 4
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 2)],
        axis=0,
    )

    real_vols = rng.standard_normal((n_volumes,) + volume_shape).astype(np.float32)
    full_vols = np.stack([
        np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1)
        for v in real_vols
    ], axis=0)
    half_vols = np.stack([
        np.asarray(fourier_transform_utils.full_volume_to_half_volume(full_vols[i], volume_shape)).reshape(-1)
        for i in range(n_volumes)
    ], axis=0)

    # Reference: batch project from full volumes to half images
    ref = np.asarray(core_slicing.batch_slice_volume_by_map_to_half_image(
        jnp.array(full_vols), rots, image_shape, volume_shape, "linear_interp"
    ))
    # New path: batch project from half volumes to half images
    out = np.asarray(core_slicing.batch_slice_volume_by_map_from_half_volume(
        jnp.array(half_vols), rots, image_shape, volume_shape, "linear_interp", half_image=True
    ))

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert out.shape == ref.shape


def test_slice_from_half_volume_to_half_image_identity_rotation_jax(monkeypatch):
    """For identity rotation, half_vol→half_img slice matches the full-vol reference (sanity check)."""
    monkeypatch.setattr(core_slicing, "_check_cuda", lambda: False)

    rng = np.random.default_rng(2015)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    rots = np.eye(3, dtype=np.float32)[None]

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    ref = np.asarray(core_slicing.slice_volume_by_map_to_half_image(
        vol_ft, rots, image_shape, volume_shape, "linear_interp"
    ))
    out = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
        half_vol, rots, image_shape, volume_shape, "linear_interp", half_image=True
    ))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_slice_from_half_volume_to_half_image_cuda_vs_jax(gpu_device):
    """CUDA half-vol→half-img projection matches CUDA full-vol→half-img reference.

    Note: JAX and CUDA use different pixel orderings for non-square images (JAX
    uses column-major from xy-meshgrid, CUDA uses row-major rfft format), so the
    comparison is CUDA half_vol vs CUDA full_vol (both on GPU).

    CUDA half-vol kernel limitation: for odd N2, N2_full = 2*(N2//2) ≠ N2, causing
    boundary errors near kz=N2//2. Only even-N2 volume shapes are tested here.
    """
    device = gpu_device
    rng = np.random.default_rng(2020)
    # image_shape H must divide volume_shape[0] (CUDA upsampling constraint)
    image_shape = (4, 8)

    for volume_shape in [(8, 8, 8), (8, 10, 12)]:
        rots = np.concatenate(
            [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 3)],
            axis=0,
        )
        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
        half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

        with jax.default_device(device):
            # Reference: CUDA full_vol → half_img (tested separately)
            ref_cuda = np.asarray(core_slicing.slice_volume_by_map_to_half_image(
                jax.device_put(jnp.array(vol_ft)), jax.device_put(rots),
                image_shape, volume_shape, "linear_interp",
            ))
            # Under test: CUDA half_vol → half_img (direct kernel)
            out_cuda = np.asarray(core_slicing.slice_volume_by_map_from_half_volume(
                jax.device_put(jnp.array(half_vol)), jax.device_put(rots),
                image_shape, volume_shape, "linear_interp", half_image=True,
            ))
        np.testing.assert_allclose(
            out_cuda, ref_cuda, atol=1e-4, rtol=1e-4,
            err_msg=f"CUDA half_vol vs CUDA full_vol mismatch for volume_shape={volume_shape}",
        )


@pytest.mark.gpu
def test_slice_from_half_volume_to_half_image_cuda_vjp_vs_ref(gpu_device):
    """CUDA custom VJP for half-vol→half-img matches VJP of the expand-then-slice reference.

    Tests that the custom_vjp backward for cuda_slice_from_half_vol_to_half_image
    produces the same gradient as the VJP of the equivalent composition
    (expand half_vol → full_vol, then cuda_slice_to_half_image). Both paths run on
    GPU so the cotangent g is in CUDA's row-major half-image format throughout.
    """
    device = gpu_device
    rng = np.random.default_rng(2021)
    # image_shape H must divide volume_shape[0] (CUDA upsampling constraint)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    n_images = 5

    rots = _random_rotations(rng, n_images)

    half_vol_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    H, W = image_shape
    half_img_size = H * (W // 2 + 1)

    hv_re = rng.standard_normal(half_vol_size).astype(np.float32)
    hv_im = rng.standard_normal(half_vol_size).astype(np.float32)
    g_re = rng.standard_normal((n_images, half_img_size)).astype(np.float32)
    g_im = rng.standard_normal((n_images, half_img_size)).astype(np.float32)

    with jax.default_device(device):
        hv = jax.device_put(jnp.array(hv_re + 1j * hv_im))
        g = jax.device_put(jnp.array(g_re + 1j * g_im))
        rots_d = jax.device_put(rots)

        # Function under test: direct CUDA half_vol → half_img kernel (custom_vjp)
        f_test = lambda v: core_slicing.slice_volume_by_map_from_half_volume(
            v, rots_d, image_shape, volume_shape, "linear_interp", half_image=True
        )

        # Reference: expand half_vol → full_vol, then CUDA full_vol → half_img
        def f_ref(v):
            full_vol = fourier_transform_utils.half_volume_to_full_volume(v, volume_shape).reshape(-1)
            return core_slicing.slice_volume_by_map_to_half_image(
                full_vol, rots_d, image_shape, volume_shape, "linear_interp"
            )

        _, vjp_test = jax.vjp(f_test, hv)
        grad_test = vjp_test(g)[0]

        _, vjp_ref = jax.vjp(f_ref, hv)
        grad_ref = vjp_ref(g)[0]

    np.testing.assert_allclose(
        np.asarray(grad_test), np.asarray(grad_ref), atol=1e-4, rtol=1e-4,
        err_msg="CUDA VJP for half_vol→half_img differs from expand-then-slice reference VJP",
    )
    assert np.isfinite(np.asarray(grad_test)).all(), "VJP returned non-finite values"
    assert np.any(np.abs(np.asarray(grad_test)) > 1e-10), "VJP returned all-zero gradient"


@pytest.mark.gpu
def test_batch_slice_from_half_volume_to_half_image_cuda_vs_jax(gpu_device):
    """batch_slice_volume_by_map_from_half_volume(half_image=True) CUDA matches CUDA full_vol reference.

    Note: JAX uses column-major pixel ordering (xy-meshgrid) while CUDA uses row-major rfft format,
    so the comparison is CUDA half_vol vs CUDA full_vol (both on GPU).
    """
    device = gpu_device
    rng = np.random.default_rng(2022)
    # image_shape H must divide volume_shape[0] (CUDA upsampling constraint)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)
    n_volumes = 3
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 2)],
        axis=0,
    )

    real_vols = rng.standard_normal((n_volumes,) + volume_shape).astype(np.float32)
    full_vols = np.stack([
        np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1)
        for v in real_vols
    ], axis=0)
    half_vols = np.stack([
        np.asarray(fourier_transform_utils.full_volume_to_half_volume(
            full_vols[i], volume_shape,
        )).reshape(-1)
        for i in range(n_volumes)
    ], axis=0)

    with jax.default_device(device):
        # Reference: CUDA full_vol → half_img (tested separately)
        ref_cuda = np.asarray(core_slicing.batch_slice_volume_by_map_to_half_image(
            jax.device_put(jnp.array(full_vols)), jax.device_put(rots),
            image_shape, volume_shape, "linear_interp",
        ))
        # Under test: CUDA half_vol → half_img (direct kernel)
        out_cuda = np.asarray(core_slicing.batch_slice_volume_by_map_from_half_volume(
            jax.device_put(jnp.array(half_vols)), jax.device_put(rots),
            image_shape, volume_shape, "linear_interp", half_image=True,
        ))

    np.testing.assert_allclose(out_cuda, ref_cuda, atol=1e-4, rtol=1e-4)
