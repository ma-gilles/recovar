import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core.geometry as core_geometry

pytestmark = pytest.mark.unit


def test_core_reexports_geometry_api():
    assert core.round_to_int is core_geometry.round_to_int
    assert core.get_gridpoint_coords is core_geometry.get_gridpoint_coords
    assert core.translate_images is core_geometry.translate_images


def test_get_stencil_and_invalid_dim():
    s2 = np.asarray(core_geometry.get_stencil(2))
    s3 = np.asarray(core_geometry.get_stencil(3))
    assert s2.shape == (4, 2)
    assert s3.shape == (8, 3)
    with pytest.raises(ValueError):
        core_geometry.get_stencil(4)


def test_batch_find_frequencies_within_grid_dist_shapes():
    coords = np.array([[0, 0], [1, -1]], dtype=np.int32)
    out = np.asarray(core_geometry.batch_find_frequencies_within_grid_dist(coords, 1))
    assert out.shape == (2, 9, 2)

    coords3 = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.int32)
    out3 = np.asarray(core_geometry.batch_batch_find_frequencies_within_grid_dist(coords3, 1))
    assert out3.shape == (1, 2, 27, 3)


def test_find_frequencies_within_grid_dist_rejects_bad_dim():
    with pytest.raises(ValueError):
        core_geometry.find_frequencies_within_grid_dist(np.array([1, 2, 3, 4]), 1)


def test_translate_single_image_zero_translation_is_identity():
    image = np.array([1 + 0j, 2 + 1j], dtype=np.complex64)
    lattice = np.array([[0.0, 0.0], [0.1, -0.2]], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)
    out = np.asarray(core_geometry.translate_single_image(image, translation, lattice))
    np.testing.assert_allclose(out, image)


def test_translate_images_zero_translation_is_identity():
    image_shape = (2, 2)
    image = np.array([[1 + 0j, 2 + 1j, 3 - 1j, 4 + 0j]], dtype=np.complex64)
    translations = np.array([[0.0, 0.0]], dtype=np.float32)
    out = np.asarray(core_geometry.translate_images(image, translations, image_shape))
    np.testing.assert_allclose(out, image)


def test_batch_get_gridpoint_coords_and_rotations_helper():
    rots = np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0)
    image_shape = (2, 2)
    volume_shape = (4, 4, 4)

    coords = np.asarray(core_geometry.batch_get_gridpoint_coords(rots, image_shape, volume_shape))
    assert coords.shape == (2, 4, 3)

    flat, og_shape = core_geometry.rotations_to_grid_point_coords(rots, image_shape, volume_shape)
    assert tuple(og_shape) == (2, 4, 3)
    assert np.asarray(flat).shape == (3, 8)


def test_batch_trans_translate_images_shape():
    images = np.array([[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]], dtype=np.complex64)
    translations = np.array([[[0.0, 0.0], [0.1, -0.1]]], dtype=np.float32)
    out = np.asarray(core_geometry.batch_trans_translate_images(images, translations, (2, 2)))
    assert out.shape == (1, 2, 4)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------
import jax


@pytest.mark.gpu
def test_round_to_int_on_gpu(gpu_device):
    arr = np.array([1.2, -1.8, 0.0, 2.5], dtype=np.float32)
    cpu_out = np.asarray(core_geometry.round_to_int(arr))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_geometry.round_to_int(jax.device_put(arr)))
    np.testing.assert_array_equal(gpu_out, cpu_out)


@pytest.mark.gpu
def test_find_frequencies_within_grid_dist_on_gpu(gpu_device):
    coords_2d = np.array([1, -1], dtype=np.int32)
    coords_3d = np.array([1, 2, 3], dtype=np.int32)
    cpu_2d = np.asarray(core_geometry.find_frequencies_within_grid_dist(coords_2d, 1))
    cpu_3d = np.asarray(core_geometry.find_frequencies_within_grid_dist(coords_3d, 1))
    with jax.default_device(gpu_device):
        gpu_2d = np.asarray(core_geometry.find_frequencies_within_grid_dist(jax.device_put(coords_2d), 1))
        gpu_3d = np.asarray(core_geometry.find_frequencies_within_grid_dist(jax.device_put(coords_3d), 1))
    np.testing.assert_array_equal(gpu_2d, cpu_2d)
    np.testing.assert_array_equal(gpu_3d, cpu_3d)


@pytest.mark.gpu
def test_batch_find_frequencies_within_grid_dist_on_gpu(gpu_device):
    coords = np.array([[0, 0], [1, -1], [2, 3]], dtype=np.int32)
    cpu_out = np.asarray(core_geometry.batch_find_frequencies_within_grid_dist(coords, 1))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_geometry.batch_find_frequencies_within_grid_dist(jax.device_put(coords), 1))
    np.testing.assert_array_equal(gpu_out, cpu_out)


@pytest.mark.gpu
def test_get_gridpoint_coords_on_gpu(gpu_device):
    rot = np.eye(3, dtype=np.float32)
    image_shape = (2, 2)
    volume_shape = (4, 4, 4)
    cpu_out = np.asarray(core_geometry.get_gridpoint_coords(rot, image_shape, volume_shape))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_geometry.get_gridpoint_coords(jax.device_put(rot), image_shape, volume_shape))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_batch_get_gridpoint_coords_on_gpu(gpu_device):
    rots = np.stack([np.eye(3, dtype=np.float32)] * 3, axis=0)
    image_shape = (2, 2)
    volume_shape = (4, 4, 4)
    cpu_out = np.asarray(core_geometry.batch_get_gridpoint_coords(rots, image_shape, volume_shape))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_geometry.batch_get_gridpoint_coords(jax.device_put(rots), image_shape, volume_shape)
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_translate_single_image_on_gpu(gpu_device):
    image = np.array([1 + 0j, 2 + 1j, 3 - 1j], dtype=np.complex64)
    lattice = np.array([[0.0, 0.0], [0.1, -0.2], [0.05, 0.15]], dtype=np.float32)
    translation = np.array([0.5, -0.3], dtype=np.float32)
    cpu_out = np.asarray(core_geometry.translate_single_image(image, translation, lattice))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_geometry.translate_single_image(
                jax.device_put(image), jax.device_put(translation), jax.device_put(lattice),
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_translate_images_zero_translation_on_gpu(gpu_device):
    image_shape = (2, 2)
    image = np.array([[1 + 0j, 2 + 1j, 3 - 1j, 4 + 0j]], dtype=np.complex64)
    translations = np.array([[0.0, 0.0]], dtype=np.float32)
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_geometry.translate_images(jax.device_put(image), jax.device_put(translations), image_shape)
        )
    np.testing.assert_allclose(gpu_out, image, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_batch_trans_translate_images_on_gpu(gpu_device):
    images = np.array([[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]], dtype=np.complex64)
    translations = np.array([[[0.0, 0.0], [0.1, -0.1]]], dtype=np.float32)
    cpu_out = np.asarray(core_geometry.batch_trans_translate_images(images, translations, (2, 2)))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(
            core_geometry.batch_trans_translate_images(
                jax.device_put(images), jax.device_put(translations), (2, 2),
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_rotations_to_grid_point_coords_on_gpu(gpu_device):
    rots = np.stack([np.eye(3, dtype=np.float32)] * 2, axis=0)
    image_shape = (2, 2)
    volume_shape = (4, 4, 4)
    cpu_flat, cpu_shape = core_geometry.rotations_to_grid_point_coords(rots, image_shape, volume_shape)
    with jax.default_device(gpu_device):
        gpu_flat, gpu_shape = core_geometry.rotations_to_grid_point_coords(
            jax.device_put(rots), image_shape, volume_shape
        )
    np.testing.assert_allclose(np.asarray(gpu_flat), np.asarray(cpu_flat), atol=1e-5, rtol=1e-5)
    assert tuple(gpu_shape) == tuple(cpu_shape)
