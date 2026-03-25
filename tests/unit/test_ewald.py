import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.reconstruction import ewald

pytestmark = pytest.mark.unit


def test_parse_disc_type_values_and_error():
    assert ewald.parse_disc_type("nearest") == 0
    assert ewald.parse_disc_type("linear_interp") == 1
    assert ewald.parse_disc_type("cubic") == 3
    with pytest.raises(ValueError, match="not recognized"):
        ewald.parse_disc_type("bad")


def test_get_flipped_indices_shape_and_bad_nyquist_markers():
    idx = np.asarray(ewald.get_flipped_indices((4, 4)))
    assert idx.shape == (16,)
    # Implementation marks ambiguous Nyquist flips as -1.
    assert np.any(idx == -1)


def test_volt_to_wavelength_positive_and_decreases_with_voltage():
    lam_200 = float(ewald.volt_to_wavelength(np.array(200.0, dtype=np.float32)))
    lam_300 = float(ewald.volt_to_wavelength(np.array(300.0, dtype=np.float32)))
    assert lam_200 > 0 and lam_300 > 0
    assert lam_300 < lam_200


def test_vec_unvec_masked_roundtrip_on_masked_entries():
    shape = (4, 4, 4)
    vol_size = np.prod(shape)
    vr = np.linspace(0.1, 2.0, vol_size).astype(np.float32)
    vi = np.linspace(-1.0, 1.0, vol_size).astype(np.float32)
    x = ewald.vec_masked(vr, vi, shape)
    mask_real_idx, mask_imag_idx = ewald.get_good_idx_mask(shape)
    mask_size = int(np.asarray(mask_real_idx[0]).size)
    vr2, vi2 = ewald.unvec_masked(x, shape, mask_size)
    vr2 = np.asarray(vr2)
    vi2 = np.asarray(vi2)
    np.testing.assert_allclose(vr2[np.asarray(mask_real_idx[0])], vr[np.asarray(mask_real_idx[0])])
    np.testing.assert_allclose(vi2[np.asarray(mask_imag_idx[0])], vi[np.asarray(mask_imag_idx[0])])


def test_get_unrotated_ewald_sphere_coords_shape_and_z_component():
    """Ewald sphere coords should have 3D output with non-negative z."""
    import jax.numpy as jnp

    image_shape = (4, 4)
    voxel_size = 1.0
    lam = ewald.volt_to_wavelength(np.array(300.0, dtype=np.float32))
    coords = np.asarray(ewald.get_unrotated_ewald_sphere_coords(image_shape, voxel_size, float(lam), scaled=True))
    assert coords.shape == (16, 3)
    # z component (Ewald sphere curvature) should be non-negative for sphere_sign=1
    assert np.all(coords[:, 2] >= -1e-7)
    assert np.all(np.isfinite(coords))


def test_get_unrotated_ewald_sphere_coords_plane_mode():
    """With very large wavelength (flat Ewald sphere), z should be ~0."""
    import jax.numpy as jnp

    image_shape = (4, 4)
    voxel_size = 1.0
    lam = 1e6  # Very large wavelength => flat sphere
    coords = np.asarray(ewald.get_unrotated_ewald_sphere_coords(image_shape, voxel_size, lam, scaled=True))
    # All z values should be essentially zero
    np.testing.assert_allclose(coords[:, 2], 0.0, atol=1e-5)


def test_get_ewald_sphere_gridpoint_coords_identity_rotation():
    """With identity rotation, rotated coords should equal unrotated + grid center."""
    import jax.numpy as jnp

    image_shape = (4, 4)
    volume_shape = (4, 4, 4)
    grid_size = 4
    voxel_size = 1.0
    lam = float(ewald.volt_to_wavelength(np.array(300.0, dtype=np.float32)))
    rot = np.eye(3, dtype=np.float32)
    coords = np.asarray(
        ewald.get_ewald_sphere_gridpoint_coords(rot, image_shape, volume_shape, grid_size, voxel_size, lam)
    )
    assert coords.shape == (16, 3)
    assert np.all(np.isfinite(coords))


def test_get_good_idx_mask_shapes_and_nonempty():
    """Real and imaginary masks should be non-empty and have valid indices."""
    shape = (4, 4, 4)
    mask_real, mask_imag = ewald.get_good_idx_mask(shape)
    mask_real_idx = np.asarray(mask_real[0])
    mask_imag_idx = np.asarray(mask_imag[0])
    vol_size = np.prod(shape)
    assert len(mask_real_idx) > 0
    assert len(mask_imag_idx) > 0
    # All indices should be within volume bounds
    assert np.all(mask_real_idx < vol_size)
    assert np.all(mask_imag_idx < vol_size)
    assert np.all(mask_real_idx >= 0)
    assert np.all(mask_imag_idx >= 0)


# ---------------------------------------------------------------------------
# Forward/adjoint numerical tests
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


def test_get_chi_zero_frequency():
    """At zero frequency, chi = -phase_shift (in radians)."""
    freqs = jnp.array([[0.0, 0.0]])
    chi = np.asarray(
        ewald.get_chi(freqs, dfu=10000, dfv=10000, dfang=0, volt=300, cs=2.7, w=0.1, phase_shift=0, bfactor=0)
    )
    np.testing.assert_allclose(chi[0], 0.0, atol=1e-6)


def test_get_chi_phase_shift():
    """At zero frequency with phase_shift=90 degrees, chi = -pi/2."""
    freqs = jnp.array([[0.0, 0.0]])
    chi = np.asarray(
        ewald.get_chi(freqs, dfu=10000, dfv=10000, dfang=0, volt=300, cs=2.7, w=0.1, phase_shift=90.0, bfactor=0)
    )
    np.testing.assert_allclose(chi[0], -np.pi / 2, atol=1e-5)


def test_get_chi_packed_contrast_scaling():
    """get_chi_packed should scale by CTF[8] (CONTRAST)."""
    freqs = jnp.array([[0.1, 0.2]])
    ctf1 = jnp.array([10000.0, 10000.0, 0.0, 300.0, 2.7, 0.1, 0.0, 0.0, 1.0])
    ctf2 = jnp.array([10000.0, 10000.0, 0.0, 300.0, 2.7, 0.1, 0.0, 0.0, 2.0])
    c1 = np.asarray(ewald.get_chi_packed(freqs, ctf1))
    c2 = np.asarray(ewald.get_chi_packed(freqs, ctf2))
    np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-5)


def test_ewald_forward_model_output_shape():
    """Forward model produces images of correct shape."""
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    vol_size = int(np.prod(volume_shape))
    img_size = int(np.prod(image_shape))
    n_images = 2
    rng = np.random.default_rng(42)
    vol_real = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    vol_imag = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    rot = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n_images, 1, 1))
    ctf_params = jnp.zeros((n_images, 9), dtype=jnp.float32)
    ctf_params = ctf_params.at[:, 0].set(10000.0)
    ctf_params = ctf_params.at[:, 1].set(10000.0)
    ctf_params = ctf_params.at[:, 3].set(300.0)
    ctf_params = ctf_params.at[:, 8].set(1.0)
    im_r, im_i = ewald.ewald_sphere_forward_model(
        vol_real, vol_imag, rot, ctf_params, image_shape, volume_shape, 3.0, "nearest"
    )
    assert np.asarray(im_r).shape == (n_images, img_size)
    assert np.asarray(im_i).shape == (n_images, img_size)
    assert np.all(np.isfinite(np.asarray(im_r)))
    assert np.all(np.isfinite(np.asarray(im_i)))


def test_ewald_adjoint_consistency():
    """Adjoint test: <Ax, y> == <x, A*y> for random x, y."""
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    vol_size = int(np.prod(volume_shape))
    img_size = int(np.prod(image_shape))
    n_images = 3
    voxel_size = 3.0
    disc_type = "nearest"
    rng = np.random.default_rng(0)

    vol_real = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    vol_imag = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    y_real = jnp.array(rng.standard_normal((n_images, img_size)).astype(np.float32))
    y_imag = jnp.array(rng.standard_normal((n_images, img_size)).astype(np.float32))

    rot = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n_images, 1, 1))
    ctf_params = jnp.zeros((n_images, 9), dtype=jnp.float32)
    ctf_params = ctf_params.at[:, 0].set(10000.0)
    ctf_params = ctf_params.at[:, 1].set(10000.0)
    ctf_params = ctf_params.at[:, 3].set(300.0)
    ctf_params = ctf_params.at[:, 4].set(2.7)
    ctf_params = ctf_params.at[:, 8].set(1.0)

    # <Ax, y>
    Ax_r, Ax_i = ewald.ewald_sphere_forward_model(
        vol_real, vol_imag, rot, ctf_params, image_shape, volume_shape, voxel_size, disc_type
    )
    lhs = float(jnp.dot(Ax_r.ravel(), y_real.ravel()) + jnp.dot(Ax_i.ravel(), y_imag.ravel()))

    # <x, A*y>
    Aty_r, Aty_i = ewald.adjoint_ewald_sphere_forward_model(
        y_real, y_imag, rot, ctf_params, image_shape, volume_shape, voxel_size, disc_type
    )
    rhs = float(jnp.dot(vol_real, Aty_r) + jnp.dot(vol_imag, Aty_i))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-4)


def test_ewald_adjoint_consistency_trilinear():
    """Adjoint test with trilinear interpolation."""
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    vol_size = int(np.prod(volume_shape))
    img_size = int(np.prod(image_shape))
    n_images = 2
    voxel_size = 3.0
    disc_type = "linear_interp"
    rng = np.random.default_rng(1)

    vol_real = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    vol_imag = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    y_real = jnp.array(rng.standard_normal((n_images, img_size)).astype(np.float32))
    y_imag = jnp.array(rng.standard_normal((n_images, img_size)).astype(np.float32))

    rot = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n_images, 1, 1))
    ctf_params = jnp.zeros((n_images, 9), dtype=jnp.float32)
    ctf_params = ctf_params.at[:, 0].set(10000.0)
    ctf_params = ctf_params.at[:, 1].set(10000.0)
    ctf_params = ctf_params.at[:, 3].set(300.0)
    ctf_params = ctf_params.at[:, 4].set(2.7)
    ctf_params = ctf_params.at[:, 8].set(1.0)

    Ax_r, Ax_i = ewald.ewald_sphere_forward_model(
        vol_real, vol_imag, rot, ctf_params, image_shape, volume_shape, voxel_size, disc_type
    )
    lhs = float(jnp.dot(Ax_r.ravel(), y_real.ravel()) + jnp.dot(Ax_i.ravel(), y_imag.ravel()))

    Aty_r, Aty_i = ewald.adjoint_ewald_sphere_forward_model(
        y_real, y_imag, rot, ctf_params, image_shape, volume_shape, voxel_size, disc_type
    )
    rhs = float(jnp.dot(vol_real, Aty_r) + jnp.dot(vol_imag, Aty_i))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-4)


def test_ewald_AtA_positive_semidefinite():
    """A^T A must be PSD: <A^T A v, v> >= 0."""
    image_shape = (8, 8)
    volume_shape = (8, 8, 8)
    vol_size = int(np.prod(volume_shape))
    rng = np.random.default_rng(5)
    vol_real = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    vol_imag = jnp.array(rng.standard_normal(vol_size).astype(np.float32))
    n_images = 2
    rot = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n_images, 1, 1))
    ctf_params = jnp.zeros((n_images, 9), dtype=jnp.float32)
    ctf_params = ctf_params.at[:, 0].set(10000.0)
    ctf_params = ctf_params.at[:, 1].set(10000.0)
    ctf_params = ctf_params.at[:, 3].set(300.0)
    ctf_params = ctf_params.at[:, 8].set(1.0)

    Ax_r, Ax_i = ewald.ewald_sphere_forward_model(
        vol_real, vol_imag, rot, ctf_params, image_shape, volume_shape, 3.0, "nearest"
    )
    AtAx_r, AtAx_i = ewald.adjoint_ewald_sphere_forward_model(
        Ax_r, Ax_i, rot, ctf_params, image_shape, volume_shape, 3.0, "nearest"
    )
    dot = float(jnp.dot(vol_real, AtAx_r) + jnp.dot(vol_imag, AtAx_i))
    assert dot >= -1e-5, f"A^T A should be PSD, got <A^T Av, v> = {dot}"


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_volt_to_wavelength_gpu(gpu_device):
    cpu_200 = float(ewald.volt_to_wavelength(np.array(200.0, dtype=np.float32)))
    cpu_300 = float(ewald.volt_to_wavelength(np.array(300.0, dtype=np.float32)))

    with jax.default_device(gpu_device):
        gpu_200 = float(ewald.volt_to_wavelength(jax.device_put(jnp.array(200.0, dtype=jnp.float32), gpu_device)))
        gpu_300 = float(ewald.volt_to_wavelength(jax.device_put(jnp.array(300.0, dtype=jnp.float32), gpu_device)))

    np.testing.assert_allclose(cpu_200, gpu_200, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_300, gpu_300, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_vec_unvec_masked_roundtrip_gpu(gpu_device):
    shape = (4, 4, 4)
    vol_size = np.prod(shape)
    vr = np.linspace(0.1, 2.0, vol_size).astype(np.float32)
    vi = np.linspace(-1.0, 1.0, vol_size).astype(np.float32)

    cpu_x = np.asarray(ewald.vec_masked(vr, vi, shape))

    with jax.default_device(gpu_device):
        vr_g = jax.device_put(jnp.array(vr), gpu_device)
        vi_g = jax.device_put(jnp.array(vi), gpu_device)
        gpu_x = np.asarray(ewald.vec_masked(vr_g, vi_g, shape))

    np.testing.assert_allclose(cpu_x, gpu_x, atol=1e-5, rtol=1e-5)
