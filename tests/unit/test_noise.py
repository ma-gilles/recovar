import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.noise as noise

pytestmark = pytest.mark.unit


def test_radial_noise_model_get_and_average():
    radial = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = noise.RadialNoiseModel(radial, image_shape=(8, 8))
    out = np.asarray(model.get())
    assert out.shape == (1, 64)
    np.testing.assert_array_equal(np.asarray(model.get_average_radial_noise()), radial)


def test_variable_radial_noise_model_get_and_average():
    radials = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float32,
    )
    dose_idx = np.array([0, 1, 1, 0], dtype=np.int32)
    model = noise.VariableRadialNoiseModel(radials, dose_idx, image_shape=(8, 8))

    out = np.asarray(model.get(np.array([0, 2, 3], dtype=np.int32)))
    assert out.shape == (3, 64)

    avg = np.asarray(model.get_average_radial_noise())
    expected = 0.5 * radials[0] + 0.5 * radials[1]
    np.testing.assert_allclose(avg, expected)


def test_make_radial_noise_scalar_and_vector():
    scalar = np.array([2.5], dtype=np.float32)
    out_scalar = np.asarray(noise.make_radial_noise(scalar, (6, 6)))
    assert out_scalar.shape == (6, 6)
    assert np.allclose(out_scalar, 2.5)

    vec = np.array([1.0, 2.0], dtype=np.float32)
    out_vec = np.asarray(noise.make_radial_noise(vec, (6, 6)))
    assert out_vec.shape == (36,)
    assert np.isfinite(out_vec).all()


def test_get_masked_noise_variance_from_noise_variance_shape():
    image_shape = (4, 4)
    image_masks = np.ones((2, *image_shape), dtype=np.float32)
    unmasked_noise = np.arange(np.prod(image_shape), dtype=np.float32)
    out = np.asarray(noise.get_masked_noise_variance_from_noise_variance(image_masks, unmasked_noise, image_shape))
    assert out.shape == (2, 4, 4)
    assert np.isfinite(out).all()


def test_to_batched_pixel_noise_normalizes_common_shapes():
    image_shape = (4, 4)

    n2 = np.arange(16, dtype=np.float32).reshape(4, 4)
    out2 = np.asarray(noise.to_batched_pixel_noise(n2, image_shape, batch_size=3))
    assert out2.shape == (3, 16)

    n3 = np.arange(2 * 16, dtype=np.float32).reshape(2, 4, 4)
    out3 = np.asarray(noise.to_batched_pixel_noise(n3, image_shape))
    assert out3.shape == (2, 16)

    n1 = np.arange(16, dtype=np.float32)
    out1 = np.asarray(noise.to_batched_pixel_noise(n1, image_shape))
    assert out1.shape == (1, 16)
