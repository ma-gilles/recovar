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
    coords = np.asarray(ewald.get_ewald_sphere_gridpoint_coords(rot, image_shape, volume_shape, grid_size, voxel_size, lam))
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
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


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
