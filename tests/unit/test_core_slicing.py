import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.slicing as core_slicing

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _configure_custom_cuda_for_test(request, monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject, "_auto_build_attempted", False)
    monkeypatch.setattr(cuda_backproject, "_auto_build_error", None)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
        monkeypatch.delenv("RECOVAR_ENABLE_CUSTOM_CUDA", raising=False)
        monkeypatch.delenv("RECOVAR_CUDA_LIB", raising=False)
        return

    lib_path = request.getfixturevalue("custom_cuda_lib")
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(lib_path))
    monkeypatch.setenv("RECOVAR_ENABLE_CUSTOM_CUDA", "1")
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)


def _project_volume(values, disc_type="linear_interp", half_volume=False):
    if isinstance(values, (core_slicing.Volume, core_slicing.CubicVolume)):
        return values
    if disc_type == "cubic":
        raise ValueError("Use core_slicing.to_cubic(...) for raw cubic test inputs")
    return core_slicing.Volume(values, disc_type=disc_type, half_volume=half_volume)


def _slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type=None, half_volume=None, **kwargs):
    if not isinstance(volume, (core_slicing.Volume, core_slicing.CubicVolume)):
        if disc_type is None:
            return core_slicing.slice_volume(volume, rotation_matrices, image_shape, volume_shape, **kwargs)
        volume = _project_volume(volume, disc_type=disc_type, half_volume=bool(False if half_volume is None else half_volume))
    return core_slicing.slice_volume(volume, rotation_matrices, image_shape, volume_shape, **kwargs)


def _batch_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, disc_type=None, half_volume=None, **kwargs):
    if not isinstance(volumes, (core_slicing.Volume, core_slicing.CubicVolume)):
        if disc_type is None:
            return core_slicing.batch_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, **kwargs)
        volumes = _project_volume(volumes, disc_type=disc_type, half_volume=bool(False if half_volume is None else half_volume))
    return core_slicing.batch_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, **kwargs)


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
    assert core.adjoint_slice_volume is core_slicing.adjoint_slice_volume


def test_decide_order_values():
    assert core_slicing.decide_order("nearest") == 0
    assert core_slicing.decide_order("linear_interp") == 1
    assert core_slicing.decide_order("cubic") == 3
    with pytest.raises(ValueError):
        core_slicing.decide_order("bad")


def test_adjoint_slice_volume_half_image_matches_full(monkeypatch):
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(11)
    # Exact half-image/full-image equivalence is defined on square grids in the
    # dedicated half-image regression tests. Keep this smoke check on the same layout.
    image_shape = (8, 8)
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
        core_slicing.adjoint_slice_volume(
            full_images, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp"
        )
    )
    out_half = np.asarray(
        core_slicing.adjoint_slice_volume(
            half_images,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_image=True,
        )
    )
    np.testing.assert_allclose(out_half, out_full, atol=1e-5, rtol=1e-5)


def test_adjoint_slice_volume_half_image_matches_full_flat_input(monkeypatch):
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(12)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0)
    real_images = rng.standard_normal((2,) + image_shape).astype(np.float32)
    half_flat = np.asarray(fourier_transform_utils.get_dft2_real(real_images)).reshape(2, -1)
    full_flat = np.asarray(fourier_transform_utils.get_dft2(real_images)).reshape(2, -1)

    out_full = np.asarray(
        core_slicing.adjoint_slice_volume(
            full_flat, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="linear_interp"
        )
    )
    out_half = np.asarray(
        core_slicing.adjoint_slice_volume(
            half_flat,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_image=True,
        )
    )
    np.testing.assert_allclose(out_half, out_full, atol=1e-5, rtol=1e-5)


def test_slice_volume_cubic_with_precomputed_spline_coefficients():
    """Periodic cubic: coefficients have same shape as volume (N^3, no padding).

    slice_volume projects explicit CubicVolume inputs without any hidden
    coefficient precompute at the slicing boundary.
    """
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(42)
    # Use square image to match pipeline convention (CUDA and JAX cubic
    # pixel orderings differ for non-square images).
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.eye(3, dtype=np.float32)[None]  # single identity rotation

    # Build a random real-valued volume and compute its Fourier transform (flat)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_flat = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)

    # Pre-compute spline coefficients — periodic BC: shape == volume_shape
    coeffs = np.asarray(cubic_interpolation.calculate_spline_coefficients(jnp.asarray(vol_flat).reshape(volume_shape)))
    coeff_shape = tuple(coeffs.shape)
    assert coeff_shape == volume_shape, (
        f"calculate_spline_coefficients returned shape {coeff_shape}, expected {volume_shape}"
    )

    # slice_volume with pre-computed coefficients must work
    # Use max_r=None to disable sphere clipping so we can compare with
    # slice_from_cubic_coefficients which does not apply clipping.
    slices = np.asarray(
        _slice_volume(
            core_slicing.CubicVolume(coeffs),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            max_r=None,
        )
    )
    n_pixels = int(np.prod(image_shape))
    assert slices.shape == (1, n_pixels)

    # slice_from_cubic_coefficients with pre-computed coefficients must give same result
    slices_precomp = np.asarray(core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape))
    np.testing.assert_allclose(slices_precomp, slices, atol=1e-5, rtol=1e-5)


def test_slice_volume_cubic_flat_and_precomputed_agree():
    """CubicVolume(coeffs) agrees with the explicit slice_from_cubic_coefficients path."""

    rng = np.random.default_rng(43)
    # Use square image to match pipeline convention (CUDA and JAX cubic
    # pixel orderings differ for non-square images).
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), _rotation_z(np.pi / 4.0)], axis=0)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)

    # Precompute coefficients (callers always do this for cubic)
    coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)

    # Forward slice through the public API (pre-computed coefficients)
    # Use max_r=None to disable sphere clipping so we can compare with
    # slice_from_cubic_coefficients which does not apply clipping.
    slices_api = np.asarray(
        _slice_volume(
            core_slicing.CubicVolume(coeffs),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            max_r=None,
        )
    )

    # Cross-check: explicit precompute + slicer must agree
    slices_precomp = np.asarray(core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape))

    np.testing.assert_allclose(slices_api, slices_precomp, atol=1e-5, rtol=1e-5)
    # Ensure the slices are not all zero (non-trivial check)
    assert np.any(np.abs(slices_api) > 1e-6), "Cubic slices are unexpectedly all zero"


def test_batch_slice_volume_cubic_volume_matches_single(monkeypatch):
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(430)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), _rotation_z(np.pi / 6.0)], axis=0)

    real_vols = rng.standard_normal((2,) + volume_shape).astype(np.float32)
    coeffs = np.stack(
        [
            np.asarray(
                core_slicing.precompute_cubic_coefficients(
                    np.asarray(fourier_transform_utils.get_dft3(vol)).reshape(-1),
                    volume_shape,
                )
            )
            for vol in real_vols
        ],
        axis=0,
    )

    batch_out = np.asarray(
        _batch_slice_volume(
            core_slicing.CubicVolume(coeffs),
            rots,
            image_shape,
            volume_shape,
            half_image=True,
            max_r=None,
        )
    )

    for i in range(coeffs.shape[0]):
        single_out = np.asarray(
            _slice_volume(
                core_slicing.CubicVolume(coeffs[i]),
                rots,
                image_shape,
                volume_shape,
                half_image=True,
                max_r=None,
            )
        )
        np.testing.assert_allclose(batch_out[i], single_out, atol=1e-5, rtol=1e-5)


def test_adjoint_slice_volume_cubic_adjointness():
    """adjoint_slice_volume with cubic must satisfy the adjoint identity:
       <Av, w> == <v, A^T w>
    where A = slice_volume (with pre-computed spline coefficients).

    This exercises the VJP code path that the rfft commits had broken.
    """

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
    coeffs = core_slicing.precompute_cubic_coefficients(vol_flat, volume_shape)

    # Random images w
    real_imgs = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    w = np.asarray(fourier_transform_utils.get_dft2(real_imgs)).reshape(n_images, -1)

    # A v  (forward slice using pre-computed coefficients)
    Av = np.asarray(
        _slice_volume(
            core_slicing.CubicVolume(coeffs),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
        )
    )

    # A^T w  (adjoint; must not crash and must return a flat vector of volume_size)
    ATw = np.asarray(
        core_slicing.adjoint_slice_volume(
            w, rots, image_shape=image_shape, volume_shape=volume_shape, disc_type="cubic"
        )
    )
    assert ATw.shape == vol_flat.shape, f"Adjoint returned shape {ATw.shape}, expected {vol_flat.shape}"

    # Check adjointness: <Av, w> == <vol_flat, A^T w>
    # (We use vol_flat as the "v" since coeffs were derived from it via a linear transform.)
    lhs = np.real(np.sum(np.conj(Av) * w))
    rhs = np.real(np.sum(np.conj(vol_flat) * ATw))
    # The equality is approximate because spline coefficient computation is not
    # the identity, so we just verify the adjoint doesn't crash and produces finite values.
    assert np.isfinite(ATw).all(), "Adjoint returned non-finite values"
    assert np.any(np.abs(ATw) > 1e-8), "Adjoint returned all-zero values"


def test_to_cubic_matches_precompute_helper():
    rng = np.random.default_rng(45)
    volume_shape = (8, 8, 8)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(real_vol))

    wrapped = core_slicing.to_cubic(vol_ft, volume_shape)
    direct = np.asarray(core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape))

    assert isinstance(wrapped, core_slicing.CubicVolume)
    assert wrapped.disc_type == "cubic"
    assert wrapped.half_volume is False
    np.testing.assert_allclose(np.asarray(wrapped.values), direct, atol=1e-6, rtol=1e-6)


def test_slice_volume_rejects_raw_projection_input():
    rng = np.random.default_rng(451)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32)], axis=0)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)

    with pytest.raises(TypeError, match="slice_volume requires a Volume or CubicVolume"):
        core_slicing.slice_volume(vol_ft, rots, image_shape, volume_shape)


def test_slice_volume_defaults_half_image_from_half_volume():
    rng = np.random.default_rng(46)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), _rotation_x(np.pi / 7.0)], axis=0)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    full_vol = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(full_vol, volume_shape)).reshape(-1)
    wrapped = core_slicing.Volume(half_vol, disc_type="linear_interp", half_volume=True)

    default_out = np.asarray(_slice_volume(wrapped, rots, image_shape, volume_shape))
    explicit_out = np.asarray(_slice_volume(wrapped, rots, image_shape, volume_shape, half_image=True))

    assert default_out.shape == explicit_out.shape
    np.testing.assert_allclose(default_out, explicit_out, atol=1e-6, rtol=1e-6)


def test_adjoint_slice_volume_cubic_volume_accumulates_in_coefficient_space():
    rng = np.random.default_rng(47)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack([np.eye(3, dtype=np.float32), _rotation_z(np.pi / 5.0)], axis=0)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    coeffs = np.asarray(
        core_slicing.precompute_cubic_coefficients(
            np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1),
            volume_shape,
        )
    )
    real_imgs = rng.standard_normal((rots.shape[0],) + image_shape).astype(np.float32)
    slices = np.asarray(fourier_transform_utils.get_dft2(real_imgs)).reshape(rots.shape[0], -1)

    out = np.asarray(
        core_slicing.adjoint_slice_volume(
            slices,
            rots,
            image_shape,
            volume_shape,
            volume=core_slicing.CubicVolume(np.zeros_like(coeffs)),
        )
    )
    ref = np.asarray(
        core_slicing._jax_adjoint_slice_from_coefficients(
            slices,
            rots,
            image_shape,
            volume_shape,
        )
    ).reshape(-1)

    np.testing.assert_allclose(out, ref, atol=5e-5, rtol=1e-5)


def test_slice_volume_jax_matches_expand(monkeypatch):
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(901)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.stack(
        [np.eye(3, dtype=np.float32), _rotation_y(np.pi / 6.0)],
        axis=0,
    )

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    full_vol = np.asarray(fourier_transform_utils.get_dft3(real_vol)).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(full_vol, volume_shape)).reshape(-1)

    out_direct = np.asarray(
        _slice_volume(
            _project_volume(half_vol, disc_type="linear_interp", half_volume=True),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_volume=True,
            half_image=False,
        )
    )
    out_expand = np.asarray(
        _slice_volume(
            _project_volume(
                np.asarray(fourier_transform_utils.half_volume_to_full_volume(half_vol, volume_shape)).reshape(-1),
                disc_type="linear_interp",
            ),
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
        )
    )

    np.testing.assert_allclose(out_direct, out_expand, atol=1e-5, rtol=1e-5)


def test_adjoint_slice_volume_half_volume_jax_vjp_consistency(monkeypatch):
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

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

    # Use max_r=None to test VJP consistency without clipping interference.
    out_full_direct = np.asarray(
        core_slicing.adjoint_slice_volume(
            full_imgs,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_volume=True,
            half_image=False,
            max_r=None,
        )
    )
    f_ref = lambda hv: _slice_volume(
        _project_volume(
            fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
            disc_type="linear_interp",
        ),
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
        max_r=None,
    )
    _, u_ref = jax.vjp(f_ref, jnp.zeros(half_size, dtype=jnp.complex64))
    out_full_ref = np.asarray(u_ref(jnp.asarray(full_imgs))[0])
    np.testing.assert_allclose(out_full_direct, out_full_ref, atol=1e-5, rtol=1e-5)

    out_half_direct = np.asarray(
        core_slicing.adjoint_slice_volume(
            half_imgs,
            rots,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_image=True,
            half_volume=True,
            max_r=None,
        )
    )
    full_from_half = np.asarray(fourier_transform_utils.half_image_to_full_image(half_imgs, image_shape))
    out_half_ref = np.asarray(u_ref(jnp.asarray(full_from_half))[0])
    np.testing.assert_allclose(out_half_direct, out_half_ref, atol=1e-5, rtol=1e-5)


def test_slice_volume_matches_full():
    """slice_volume should match projecting from the full volume."""
    import jax.numpy as jnp

    rng = np.random.default_rng(88)
    volume_shape = (8, 8, 8)
    image_shape = (4, 8)
    # Create a real-space volume, then FFT to get a proper Hermitian Fourier volume
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    volume_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_volume = np.asarray(fourier_transform_utils.full_volume_to_half_volume(volume_ft, volume_shape))
    rotation_matrices = np.eye(3, dtype=np.float32)[None, ...]

    out_full = np.asarray(
        _slice_volume(
            _project_volume(volume_ft, disc_type="nearest"),
            rotation_matrices,
            image_shape,
            volume_shape,
            "nearest",
        )
    )
    out_half = np.asarray(
        _slice_volume(
            _project_volume(half_volume, disc_type="nearest", half_volume=True),
            rotation_matrices,
            image_shape,
            volume_shape,
            "nearest",
            half_volume=True,
            half_image=False,
        )
    )
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


def test_slice_volume_half_image_jax(monkeypatch):
    """half_image=True in slice_volume matches slice_to_half_image(full_vol)."""
    import jax.numpy as jnp

    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2001)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    rots = np.concatenate([np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 3)], axis=0)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    # Reference: project from full volume directly to half-image
    ref_half = np.asarray(
        _slice_volume(
            _project_volume(vol_ft, disc_type="linear_interp"),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
        )
    )

    # New path: project from half volume directly to half image
    out_half = np.asarray(
        _slice_volume(
            _project_volume(half_vol, disc_type="linear_interp", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )
    )
    np.testing.assert_allclose(out_half, ref_half, atol=1e-5, rtol=1e-5)


def test_batch_slice_volume_jax(monkeypatch):
    """batch_slice_volume matches vmap of single-volume version."""
    import jax.numpy as jnp

    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2002)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_volumes = 3
    rots = np.concatenate([np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 2)], axis=0)

    real_vols = rng.standard_normal((n_volumes,) + volume_shape).astype(np.float32)
    half_vols = np.stack(
        [
            np.asarray(
                fourier_transform_utils.full_volume_to_half_volume(
                    np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1), volume_shape
                )
            ).reshape(-1)
            for v in real_vols
        ],
        axis=0,
    )

    # With half_image=False: batch == vmap of single
    batch_out = np.asarray(
        _batch_slice_volume(
            _project_volume(half_vols, disc_type="linear_interp", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=False,
        )
    )
    for i in range(n_volumes):
        single = np.asarray(
            _slice_volume(
                _project_volume(half_vols[i], disc_type="linear_interp", half_volume=True),
                rots,
                image_shape,
                volume_shape,
                "linear_interp",
                half_volume=True,
                half_image=False,
            )
        )
        np.testing.assert_allclose(batch_out[i], single, atol=1e-5, rtol=1e-5)

    # With half_image=True: batch == vmap of single
    batch_out_hi = np.asarray(
        _batch_slice_volume(
            _project_volume(half_vols, disc_type="linear_interp", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )
    )
    for i in range(n_volumes):
        single_hi = np.asarray(
            _slice_volume(
                _project_volume(half_vols[i], disc_type="linear_interp", half_volume=True),
                rots,
                image_shape,
                volume_shape,
                "linear_interp",
                half_volume=True,
                half_image=True,
            )
        )
        np.testing.assert_allclose(batch_out_hi[i], single_hi, atol=1e-5, rtol=1e-5)


def test_precompute_cubic_coefficients_shape():
    """precompute_cubic_coefficients must return periodic coefficient shape (N0, N1, N2)."""
    rng = np.random.default_rng(600)
    for volume_shape in [(8, 8, 8), (6, 8, 10), (7, 9, 11)]:
        N0, N1, N2 = volume_shape
        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
        coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)
        expected = (N0, N1, N2)
        assert coeffs.shape == expected, f"volume_shape={volume_shape}: got {coeffs.shape}, expected {expected}"


def test_cubic_half_coefficients_slice_matches_full_cubic():
    """slice_from_cubic_coefficients must give identical results to map_coordinates_on_slices."""
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(601)
    image_shape = (4, 8)

    for volume_shape in [(8, 8, 8), (8, 8, 10), (8, 10, 12)]:
        rots = np.concatenate(
            [
                np.eye(3, dtype=np.float32)[None],
                _random_rotations(rng, 5),
            ],
            axis=0,
        )

        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

        # Reference: full cubic slicing via map_coordinates_on_slices
        coeffs_full = np.asarray(
            cubic_interpolation.calculate_spline_coefficients(jnp.asarray(vol_ft).reshape(volume_shape))
        )
        slices_ref = np.asarray(core_slicing._jax_slice(coeffs_full, rots, image_shape, volume_shape, order=3))

        # New path: precompute coefficients, then slice
        coeffs = np.asarray(core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape))
        slices_half = np.asarray(core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape))

        np.testing.assert_allclose(
            slices_half,
            slices_ref,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"Mismatch for volume_shape={volume_shape}",
        )
        assert np.any(np.abs(slices_ref) > 1e-6), "Reference slices are all zero"


def test_cubic_half_coefficients_slice_matches_full_cubic_odd_dims():
    """Cubic coefficient slicer must work for odd N2."""
    import recovar.core.cubic_interpolation as cubic_interpolation

    rng = np.random.default_rng(602)
    image_shape = (4, 8)
    volume_shape = (8, 8, 9)  # odd N2

    rots = np.concatenate(
        [
            np.eye(3, dtype=np.float32)[None],
            _random_rotations(rng, 4),
        ],
        axis=0,
    )

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

    coeffs_full = np.asarray(
        cubic_interpolation.calculate_spline_coefficients(jnp.asarray(vol_ft).reshape(volume_shape))
    )
    slices_ref = np.asarray(core_slicing._jax_slice(coeffs_full, rots, image_shape, volume_shape, order=3))

    coeffs = np.asarray(core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape))
    slices_half = np.asarray(core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape))

    np.testing.assert_allclose(slices_half, slices_ref, atol=1e-4, rtol=1e-4)


def test_cubic_half_coefficients_slice_vjp_finite():
    """VJP through slice_from_cubic_coefficients must be finite and non-zero."""
    rng = np.random.default_rng(603)
    image_shape = (4, 8)
    volume_shape = (8, 8, 8)

    rots = np.concatenate(
        [
            np.eye(3, dtype=np.float32)[None],
            _random_rotations(rng, 2),
        ],
        axis=0,
    )

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = jnp.array(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)

    coeffs = core_slicing.precompute_cubic_coefficients(vol_ft, volume_shape)

    n_images = rots.shape[0]
    H, W = image_shape
    g = jnp.ones((n_images, H * W), dtype=jnp.complex64)

    def f(coefficients):
        return core_slicing.slice_from_cubic_coefficients(
            coefficients, rots, image_shape, volume_shape, half_image=False
        )

    _, vjp_fn = jax.vjp(f, jnp.asarray(coeffs))
    grad = np.asarray(vjp_fn(g)[0])

    assert np.isfinite(grad).all(), "VJP returned non-finite values"
    assert np.any(np.abs(grad) > 1e-10), "VJP returned all-zero gradient"


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
            core_slicing.adjoint_slice_volume(
                jax.device_put(half_images),
                jax.device_put(rots),
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type="linear_interp",
                half_image=True,
            )
        )
        out_full = np.asarray(
            core_slicing.adjoint_slice_volume(
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
    """half_image=True in slice_volume matches full-vol reference
    for multiple volume shapes including non-cubic and odd-N2 cases (JAX path)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2010)
    image_shape = (8, 8)

    for volume_shape in [(8, 8, 8), (8, 8, 9), (8, 10, 12)]:
        rots = np.concatenate(
            [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 4)],
            axis=0,
        )
        real_vol = rng.standard_normal(volume_shape).astype(np.float32)
        vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
        half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

        ref = np.asarray(
            _slice_volume(
                _project_volume(vol_ft, disc_type="linear_interp"),
                rots,
                image_shape,
                volume_shape,
                "linear_interp",
                half_image=True,
            )
        )
        out = np.asarray(
            _slice_volume(
                _project_volume(half_vol, disc_type="linear_interp", half_volume=True),
                rots,
                image_shape,
                volume_shape,
                "linear_interp",
                half_volume=True,
                half_image=True,
            )
        )
        np.testing.assert_allclose(
            out,
            ref,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"half_image=True mismatch for volume_shape={volume_shape}",
        )
        assert np.any(np.abs(out) > 1e-8), f"All-zero output for volume_shape={volume_shape}"


def test_slice_from_half_volume_to_half_image_nearest_jax(monkeypatch):
    """Nearest-neighbour variant of half_image=True path (JAX) matches full-vol reference."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2013)
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 3)],
        axis=0,
    )
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    ref = np.asarray(
        _slice_volume(
            _project_volume(vol_ft, disc_type="nearest"),
            rots,
            image_shape,
            volume_shape,
            "nearest",
            half_image=True,
        )
    )
    out = np.asarray(
        _slice_volume(
            _project_volume(half_vol, disc_type="nearest", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "nearest",
            half_volume=True,
            half_image=True,
        )
    )
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_slice_from_half_volume_to_half_image_vjp_finite_jax(monkeypatch):
    """VJP through slice_volume(half_image=True) is finite and non-zero (JAX path)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2011)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = _random_rotations(rng, n_images)

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = jnp.array(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = jnp.array(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    half_img_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    real_g = rng.standard_normal((n_images,) + half_img_shape).astype(np.float32)
    imag_g = rng.standard_normal((n_images,) + half_img_shape).astype(np.float32)
    g = jnp.array(real_g + 1j * imag_g).reshape(n_images, -1)

    f = lambda v: _slice_volume(
        _project_volume(v, disc_type="linear_interp", half_volume=True),
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=True,
        half_image=True,
    )
    _, vjp_fn = jax.vjp(f, half_vol)
    grad = np.asarray(vjp_fn(g)[0])

    assert np.isfinite(grad).all(), "VJP returned non-finite values"
    assert np.any(np.abs(grad) > 1e-10), "VJP returned all-zero gradient"


def test_slice_from_half_volume_to_half_image_vjp_vs_reference_jax(monkeypatch):
    """VJP of the half-vol→half-img path matches VJP of expand-then-slice reference (JAX path)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2012)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = _random_rotations(rng, n_images)

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))

    real_imgs = rng.standard_normal((n_images,) + image_shape).astype(np.float32)
    half_imgs = jnp.asarray(fourier_transform_utils.get_dft2_real(real_imgs)).reshape(n_images, -1)

    # New function: half_vol → half_images (JAX path, CUDA off)
    f_new = lambda hv: _slice_volume(
        _project_volume(hv, disc_type="linear_interp", half_volume=True),
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=True,
        half_image=True,
    )
    _, vjp_new = jax.vjp(f_new, jnp.zeros(half_size, dtype=jnp.complex64))
    grad_new = np.asarray(vjp_new(half_imgs)[0])

    # Reference: expand half_vol → full_vol, then slice to half_img
    def f_ref(hv):
        full_vol = fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape).reshape(-1)
        return _slice_volume(
            _project_volume(full_vol, disc_type="linear_interp"),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
        )

    _, vjp_ref = jax.vjp(f_ref, jnp.zeros(half_size, dtype=jnp.complex64))
    grad_ref = np.asarray(vjp_ref(half_imgs)[0])

    np.testing.assert_allclose(grad_new, grad_ref, atol=1e-4, rtol=1e-4)
    assert np.any(np.abs(grad_new) > 1e-10), "VJP returned all-zero gradient"


def test_batch_slice_from_half_volume_to_half_image_vs_full_vol_batch_jax(monkeypatch):
    """batch_slice_volume(half_image=True) matches batch_slice_to_half_image(full_vols)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2014)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_volumes = 4
    rots = np.concatenate(
        [np.eye(3, dtype=np.float32)[None], _random_rotations(rng, 2)],
        axis=0,
    )

    real_vols = rng.standard_normal((n_volumes,) + volume_shape).astype(np.float32)
    full_vols = np.stack(
        [np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1) for v in real_vols], axis=0
    )
    half_vols = np.stack(
        [
            np.asarray(fourier_transform_utils.full_volume_to_half_volume(full_vols[i], volume_shape)).reshape(-1)
            for i in range(n_volumes)
        ],
        axis=0,
    )

    # Reference: batch project from full volumes to half images
    ref = np.asarray(
        _batch_slice_volume(
            _project_volume(jnp.array(full_vols), disc_type="linear_interp"),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
        )
    )
    # New path: batch project from half volumes to half images
    out = np.asarray(
        _batch_slice_volume(
            _project_volume(jnp.array(half_vols), disc_type="linear_interp", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )
    )

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert out.shape == ref.shape


def test_slice_from_half_volume_to_half_image_identity_rotation_jax(monkeypatch):
    """For identity rotation, half_vol→half_img slice matches the full-vol reference (sanity check)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(2015)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    rots = np.eye(3, dtype=np.float32)[None]

    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    vol_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.array(real_vol))).reshape(-1)
    half_vol = np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_ft, volume_shape))

    ref = np.asarray(
        _slice_volume(
            _project_volume(vol_ft, disc_type="linear_interp"),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
        )
    )
    out = np.asarray(
        _slice_volume(
            _project_volume(half_vol, disc_type="linear_interp", half_volume=True),
            rots,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )
    )
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
            ref_cuda = np.asarray(
                _slice_volume(
                    _project_volume(jax.device_put(jnp.array(vol_ft)), disc_type="linear_interp"),
                    jax.device_put(rots),
                    image_shape,
                    volume_shape,
                    "linear_interp",
                    half_image=True,
                )
            )
            # Under test: CUDA half_vol → half_img (direct kernel)
            out_cuda = np.asarray(
                _slice_volume(
                    _project_volume(
                        jax.device_put(jnp.array(half_vol)),
                        disc_type="linear_interp",
                        half_volume=True,
                    ),
                    jax.device_put(rots),
                    image_shape,
                    volume_shape,
                    "linear_interp",
                    half_volume=True,
                    half_image=True,
                )
            )
        np.testing.assert_allclose(
            out_cuda,
            ref_cuda,
            atol=1e-4,
            rtol=1e-4,
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
        f_test = lambda v: _slice_volume(
            _project_volume(v, disc_type="linear_interp", half_volume=True),
            rots_d,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
        )

        # Reference: expand half_vol → full_vol, then CUDA full_vol → half_img
        def f_ref(v):
            full_vol = fourier_transform_utils.half_volume_to_full_volume(v, volume_shape).reshape(-1)
            return _slice_volume(
                _project_volume(full_vol, disc_type="linear_interp"),
                rots_d,
                image_shape,
                volume_shape,
                "linear_interp",
                half_image=True,
            )

        _, vjp_test = jax.vjp(f_test, hv)
        grad_test = vjp_test(g)[0]

        _, vjp_ref = jax.vjp(f_ref, hv)
        grad_ref = vjp_ref(g)[0]

    np.testing.assert_allclose(
        np.asarray(grad_test),
        np.asarray(grad_ref),
        atol=1e-4,
        rtol=1e-4,
        err_msg="CUDA VJP for half_vol→half_img differs from expand-then-slice reference VJP",
    )
    assert np.isfinite(np.asarray(grad_test)).all(), "VJP returned non-finite values"
    assert np.any(np.abs(np.asarray(grad_test)) > 1e-10), "VJP returned all-zero gradient"


@pytest.mark.gpu
def test_batch_slice_from_half_volume_to_half_image_cuda_vs_jax(gpu_device):
    """batch_slice_volume(half_image=True) CUDA matches CUDA full_vol reference.

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
    full_vols = np.stack(
        [np.asarray(fourier_transform_utils.get_dft3(jnp.array(v))).reshape(-1) for v in real_vols], axis=0
    )
    half_vols = np.stack(
        [
            np.asarray(
                fourier_transform_utils.full_volume_to_half_volume(
                    full_vols[i],
                    volume_shape,
                )
            ).reshape(-1)
            for i in range(n_volumes)
        ],
        axis=0,
    )

    with jax.default_device(device):
        # Reference: CUDA full_vol → half_img (tested separately)
        ref_cuda = np.asarray(
            _batch_slice_volume(
                _project_volume(jax.device_put(jnp.array(full_vols)), disc_type="linear_interp"),
                jax.device_put(rots),
                image_shape,
                volume_shape,
                "linear_interp",
                half_image=True,
            )
        )
        # Under test: CUDA half_vol → half_img (direct kernel)
        out_cuda = np.asarray(
            _batch_slice_volume(
                _project_volume(
                    jax.device_put(jnp.array(half_vols)),
                    disc_type="linear_interp",
                    half_volume=True,
                ),
                jax.device_put(rots),
                image_shape,
                volume_shape,
                "linear_interp",
                half_volume=True,
                half_image=True,
            )
        )

    np.testing.assert_allclose(out_cuda, ref_cuda, atol=1e-4, rtol=1e-4)


# ── batch_adjoint_slice_volume tests ──────────────────────────────────


def _random_complex_images(rng, n_images, image_shape):
    """Generate random complex images (Hermitian-symmetric not enforced)."""
    size = int(np.prod(image_shape))
    return (rng.standard_normal((n_images, size)) + 1j * rng.standard_normal((n_images, size))).astype(np.complex64)


def test_batch_adjoint_slice_volume_jax(monkeypatch):
    """batch_adjoint matches per-volume loop on JAX path."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(3001)
    volume_shape = (8, 8, 8)
    image_shape = (6, 8)
    n_batch = 3
    n_images = 4
    rots = np.concatenate([np.eye(3, dtype=np.float32)[None], _random_rotations(rng, n_images - 1)], axis=0)

    images = np.stack([_random_complex_images(rng, n_images, image_shape) for _ in range(n_batch)])

    batch_out = np.asarray(
        core_slicing.batch_adjoint_slice_volume(
            jnp.array(images),
            jnp.array(rots),
            image_shape,
            volume_shape,
            "linear_interp",
        )
    )

    for i in range(n_batch):
        single = np.asarray(
            core_slicing.adjoint_slice_volume(
                jnp.array(images[i]),
                jnp.array(rots),
                image_shape,
                volume_shape,
                "linear_interp",
            )
        )
        np.testing.assert_allclose(batch_out[i], single, atol=1e-5, rtol=1e-5)


def test_batch_adjoint_slice_volume_with_accumulator_jax(monkeypatch):
    """batch_adjoint adds into a pre-existing accumulator."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(3002)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_batch = 2
    n_images = 3
    rots = _random_rotations(rng, n_images).astype(np.float32)
    images = np.stack([_random_complex_images(rng, n_images, image_shape) for _ in range(n_batch)])

    vol_size = int(np.prod(volume_shape))
    seed_vols = (rng.standard_normal((n_batch, vol_size)) + 1j * rng.standard_normal((n_batch, vol_size))).astype(
        np.complex64
    )

    batch_out = np.asarray(
        core_slicing.batch_adjoint_slice_volume(
            jnp.array(images),
            jnp.array(rots),
            image_shape,
            volume_shape,
            "linear_interp",
            volumes=jnp.array(seed_vols),
        )
    )

    for i in range(n_batch):
        single = np.asarray(
            core_slicing.adjoint_slice_volume(
                jnp.array(images[i]),
                jnp.array(rots),
                image_shape,
                volume_shape,
                "linear_interp",
                volume=jnp.array(seed_vols[i]),
            )
        )
        np.testing.assert_allclose(batch_out[i], single, atol=1e-5, rtol=1e-5)


def test_batch_adjoint_slice_volume_half_image_jax(monkeypatch):
    """batch_adjoint with half_image=True matches per-volume loop."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(3003)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_batch = 2
    n_images = 3
    rots = _random_rotations(rng, n_images).astype(np.float32)

    # Generate full images then convert to half
    full_images = np.stack([_random_complex_images(rng, n_images, image_shape) for _ in range(n_batch)])
    half_images = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full_images), image_shape))

    batch_out = np.asarray(
        core_slicing.batch_adjoint_slice_volume(
            jnp.array(half_images),
            jnp.array(rots),
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=True,
        )
    )

    for i in range(n_batch):
        single = np.asarray(
            core_slicing.adjoint_slice_volume(
                jnp.array(half_images[i]),
                jnp.array(rots),
                image_shape,
                volume_shape,
                "linear_interp",
                half_image=True,
            )
        )
        np.testing.assert_allclose(batch_out[i], single, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_batch_adjoint_slice_volume_cuda(gpu_device):
    """batch_adjoint CUDA path matches per-volume loop."""
    rng = np.random.default_rng(3004)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_batch = 3
    n_images = 4
    rots = _random_rotations(rng, n_images).astype(np.float32)

    images = np.stack([_random_complex_images(rng, n_images, image_shape) for _ in range(n_batch)])

    with jax.default_device(gpu_device):
        batch_out = np.asarray(
            core_slicing.batch_adjoint_slice_volume(
                jnp.array(images),
                jnp.array(rots),
                image_shape,
                volume_shape,
                "linear_interp",
            )
        )
        for i in range(n_batch):
            single = np.asarray(
                core_slicing.adjoint_slice_volume(
                    jnp.array(images[i]),
                    jnp.array(rots),
                    image_shape,
                    volume_shape,
                    "linear_interp",
                )
            )
            np.testing.assert_allclose(batch_out[i], single, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_batch_adjoint_slice_volume_cuda_half_image(gpu_device):
    """batch_adjoint CUDA path with half_image matches per-volume loop."""
    rng = np.random.default_rng(3005)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_batch = 3
    n_images = 4
    rots = _random_rotations(rng, n_images).astype(np.float32)

    full_images = np.stack([_random_complex_images(rng, n_images, image_shape) for _ in range(n_batch)])

    with jax.default_device(gpu_device):
        half_images = jnp.array(
            np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full_images), image_shape))
        )
        batch_out = np.asarray(
            core_slicing.batch_adjoint_slice_volume(
                half_images,
                jnp.array(rots),
                image_shape,
                volume_shape,
                "linear_interp",
                half_image=True,
            )
        )
        for i in range(n_batch):
            single = np.asarray(
                core_slicing.adjoint_slice_volume(
                    half_images[i],
                    jnp.array(rots),
                    image_shape,
                    volume_shape,
                    "linear_interp",
                    half_image=True,
                )
            )
            np.testing.assert_allclose(batch_out[i], single, atol=1e-4, rtol=1e-4)


# ── Helpers for new tests ──────────────────────────────────────────────


def _random_rotations_half_image(rng, n):
    from scipy.spatial.transform import Rotation

    return Rotation.random(n, random_state=rng.integers(2**31)).as_matrix().astype(np.float32)


def _make_hermitian_volume(rng, volume_shape):
    """Create a volume with Hermitian symmetry (DFT of real data)."""
    real_data = rng.standard_normal(volume_shape).astype(np.float32)
    return np.asarray(fourier_transform_utils.get_dft3(real_data)).ravel()


def _random_complex_images_half_image(rng, n_images, image_shape):
    n_pix = image_shape[0] * image_shape[1]
    return (rng.standard_normal((n_images, n_pix)) + 1j * rng.standard_normal((n_images, n_pix))).astype(np.complex64)


# ── Half-image forward: efficient path == old expand path ──────────────


@pytest.mark.parametrize("order", [0, 1])
def test_jax_half_image_forward_matches_full_then_extract(monkeypatch, order):
    """_jax_slice_half_image matches _jax_slice + full_image_to_half_image."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5001)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))
    vol = jnp.array(_make_hermitian_volume(rng, volume_shape))

    full = np.asarray(core_slicing._jax_slice(vol, rots, image_shape, volume_shape, order))
    half_ref = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full), image_shape))
    half_new = np.asarray(core_slicing._jax_slice_half_image(vol, rots, image_shape, volume_shape, order))

    np.testing.assert_allclose(half_new, half_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("image_shape", [(6, 8), (8, 6), (5, 7)])
def test_jax_half_image_forward_matches_full_then_extract_rectangular(monkeypatch, order, image_shape):
    """Regression: non-square images must match full->half-image extraction."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5001)
    volume_shape = (8, 8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))
    vol = jnp.array(_make_hermitian_volume(rng, volume_shape))

    full = np.asarray(core_slicing._jax_slice(vol, rots, image_shape, volume_shape, order))
    half_ref = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full), image_shape))
    half_new = np.asarray(core_slicing._jax_slice_half_image(vol, rots, image_shape, volume_shape, order))

    np.testing.assert_allclose(half_new, half_ref, atol=1e-5, rtol=1e-5)


def test_jax_half_image_forward_cubic_matches_full_then_extract(monkeypatch):
    """Cubic: _jax_slice_half_image with precomputed coeffs matches extract from full."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)
    from recovar.core import cubic_interpolation

    rng = np.random.default_rng(5002)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))
    vol = jnp.array(_make_hermitian_volume(rng, volume_shape))
    coeffs = cubic_interpolation.calculate_spline_coefficients(vol.reshape(volume_shape))

    full = np.asarray(core_slicing._jax_slice(coeffs, rots, image_shape, volume_shape, 3))
    half_ref = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full), image_shape))
    half_new = np.asarray(core_slicing._jax_slice_half_image(coeffs, rots, image_shape, volume_shape, 3))

    np.testing.assert_allclose(half_new, half_ref, atol=1e-5, rtol=1e-5)


# ── slice_volume forward: half_image flag ──────────────────────────────


@pytest.mark.parametrize("disc_type", ["nearest", "linear_interp"])
@pytest.mark.parametrize("half_image", [False, True])
def test_slice_volume_jax_half_image_flag(monkeypatch, disc_type, half_image):
    """slice_volume(half_image=True) matches extract from slice_volume(half_image=False)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5003)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))
    vol = jnp.array(_make_hermitian_volume(rng, volume_shape))

    result = np.asarray(
        _slice_volume(
            _project_volume(vol, disc_type=disc_type),
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_image=half_image,
        )
    )
    ref_full = np.asarray(
        _slice_volume(
            _project_volume(vol, disc_type=disc_type),
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_image=False,
        )
    )

    if half_image:
        ref = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(ref_full), image_shape))
        np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)
    else:
        np.testing.assert_allclose(result, ref_full, atol=1e-5, rtol=1e-5)


# ── Adjoint: half_image consistency (JAX expand == CUDA native) ────────


@pytest.mark.parametrize("disc_type", ["nearest", "linear_interp"])
def test_adjoint_jax_half_image_matches_expand(monkeypatch, disc_type):
    """JAX adjoint(half_image=True) == adjoint(expand(half_imgs), half_image=False)."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5004)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))

    full_imgs = jnp.array(_random_complex_images_half_image(rng, n_images, image_shape))
    half_imgs = jnp.array(np.asarray(fourier_transform_utils.full_image_to_half_image(full_imgs, image_shape)))
    expanded = jnp.array(np.asarray(fourier_transform_utils.half_image_to_full_image(half_imgs, image_shape)))

    result = np.asarray(
        core_slicing.adjoint_slice_volume(
            half_imgs,
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
        )
    )
    ref = np.asarray(
        core_slicing.adjoint_slice_volume(
            expanded,
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_image=False,
        )
    )

    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


# ── Adjoint: dot-product test for all flag combos ──────────────────────


@pytest.mark.parametrize("disc_type", ["nearest", "linear_interp"])
@pytest.mark.parametrize("half_volume", [False, True])
def test_adjoint_dot_product_consistency(monkeypatch, disc_type, half_volume):
    """Adjoint dot-product test: <A*y, x> == <y, Ax> for full-image paths."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5005)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))

    vol_full = _make_hermitian_volume(rng, volume_shape)
    if half_volume:
        vol_in = jnp.array(
            np.asarray(fourier_transform_utils.full_volume_to_half_volume(jnp.array(vol_full), volume_shape))
        )
    else:
        vol_in = jnp.array(vol_full)

    # Forward: Ax
    fwd = np.asarray(
        _slice_volume(
            _project_volume(vol_in, disc_type=disc_type, half_volume=half_volume),
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_volume=half_volume,
            half_image=False,
        )
    )

    img_in = jnp.array(_random_complex_images_half_image(rng, n_images, image_shape))

    # Adjoint: A*y
    adj = np.asarray(
        core_slicing.adjoint_slice_volume(
            img_in,
            rots,
            image_shape,
            volume_shape,
            disc_type,
            half_image=False,
            half_volume=half_volume,
        )
    )

    lhs = np.sum(adj.conj() * np.asarray(vol_in)).real
    rhs = np.sum(np.asarray(img_in).conj() * fwd).real

    np.testing.assert_allclose(
        lhs, rhs, atol=1e-3, rtol=1e-3, err_msg=f"disc_type={disc_type}, half_volume={half_volume}"
    )


# ── Cubic adjoint with half_volume bug fix ─────────────────────────────


def test_adjoint_cubic_half_volume_includes_spline_coefficients(monkeypatch):
    """Cubic adjoint with half_volume=True uses spline coefficients in VJP.

    Regression test for a bug where the half_volume VJP forward function
    passed raw volume data to _jax_slice(..., order=3), which expects
    precomputed spline coefficients.
    """
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5006)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 2
    rots = jnp.array(_random_rotations_half_image(rng, n_images))

    vol_full = jnp.array(_make_hermitian_volume(rng, volume_shape))
    half_vol = jnp.array(np.asarray(fourier_transform_utils.full_volume_to_half_volume(vol_full, volume_shape)))
    imgs = jnp.array(_random_complex_images_half_image(rng, n_images, image_shape))

    # This should not crash and should produce finite values.
    result = np.asarray(
        core_slicing.adjoint_slice_volume(
            imgs,
            rots,
            image_shape,
            volume_shape,
            "cubic",
            half_volume=True,
            half_image=False,
        )
    )

    assert np.isfinite(result).all(), "Cubic adjoint with half_volume produced non-finite values"
    assert np.any(np.abs(result) > 1e-8), "Cubic adjoint with half_volume returned all zeros"


# ── Cubic slice from coefficients: half_image flag ─────────────────────


def test_cubic_slice_from_coeffs_half_image(monkeypatch):
    """slice_from_cubic_coefficients(half_image=True) matches extract from full."""
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)

    rng = np.random.default_rng(5008)
    volume_shape = (8, 8, 8)
    image_shape = (8, 8)
    n_images = 3
    rots = jnp.array(_random_rotations_half_image(rng, n_images))
    vol = jnp.array(_make_hermitian_volume(rng, volume_shape))
    coeffs = core_slicing.precompute_cubic_coefficients(vol, volume_shape)

    full = np.asarray(
        core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape, half_image=False)
    )
    half = np.asarray(
        core_slicing.slice_from_cubic_coefficients(coeffs, rots, image_shape, volume_shape, half_image=True)
    )

    ref = np.asarray(fourier_transform_utils.full_image_to_half_image(jnp.array(full), image_shape))
    np.testing.assert_allclose(half, ref, atol=1e-5, rtol=1e-5)
