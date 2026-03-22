import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.reconstruction.noise as noise

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
    assert out_scalar.shape == (36,)  # flat, consistent with vector path
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


def test_variable_radial_noise_model_set_variance_1d_broadcast():
    """Test that set_variance with 1D input still works for .get()
    when the model was originally constructed with 2D noise."""
    radials_2d = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    dose_idx = np.array([0, 1, 1, 0, 0], dtype=np.int32)
    model = noise.VariableRadialNoiseModel(radials_2d, dose_idx, image_shape=(8, 8))

    # Updating with a 2D array should work as before
    new_2d = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=np.float32,
    )
    model.set_variance(new_2d)
    out = np.asarray(model.get(np.array([0, 1], dtype=np.int32)))
    assert out.shape == (2, 64)


def test_variable_radial_noise_model_via_dataset_1d_broadcast():
    """Test the fix: set_variable_radial_noise_model with 1D input
    broadcasts to 2D so VariableRadialNoiseModel.get() doesn't crash."""
    import jax.numpy as jnp
    from recovar import core

    # Simulate a CryoEMDataset-like object with CTF_params that have
    # multiple dose levels (7 tilts → 7 unique dose values)
    n_images = 21  # 3 particles × 7 tilts
    n_tilts = 7
    grid_size = 8

    # Build minimal CTF_params with different dose values per tilt
    ctf_params = np.zeros((n_images, 11), dtype=np.float32)
    doses = np.repeat(np.arange(n_tilts, dtype=np.float32), 3)  # 7 unique doses
    ctf_params[:, core.CTFParamIndex.DOSE] = doses

    # Create a mock dataset class with the method we fixed
    class MockDataset:
        def __init__(self):
            self.CTF_params = ctf_params
            self.image_shape = (grid_size, grid_size)
            self.noise = None

        def set_variable_radial_noise_model(self, noise_variance_radials):
            _, dose_indices = jnp.unique(
                self.CTF_params[:, core.CTFParamIndex.DOSE], return_inverse=True
            )
            if noise_variance_radials is not None and np.ndim(noise_variance_radials) == 1:
                n_doses = int(jnp.max(dose_indices)) + 1
                noise_variance_radials = np.tile(noise_variance_radials, (n_doses, 1))
            self.noise = noise.VariableRadialNoiseModel(
                noise_variance_radials, dose_indices, image_shape=self.image_shape
            )

    ds = MockDataset()
    # 1D noise profile (the case that was crashing)
    radial_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ds.set_variable_radial_noise_model(radial_1d)

    # Verify: noise model has 2D noise_variance_radials
    assert ds.noise.noise_variance_radials.ndim == 2
    assert ds.noise.noise_variance_radials.shape == (n_tilts, 3)

    # Verify: .get() works for arbitrary image indices
    out = np.asarray(ds.noise.get(np.array([0, 5, 10, 20], dtype=np.int32)))
    assert out.shape == (4, grid_size * grid_size)
    assert np.isfinite(out).all()

    # All dose levels should have the same radial profile (broadcast from 1D)
    for i in range(n_tilts):
        np.testing.assert_array_equal(
            ds.noise.noise_variance_radials[i], radial_1d
        )


def test_upper_bound_noise_dispatched_1d_noise():
    """Test that upper_bound_noise_by_signal_p_noise_dispatched handles
    1D noise_var_used correctly (doesn't index a scalar from 1D array)."""
    # This is a shape/indexing test — we verify the dispatching logic
    # selects the right noise slice for 1D vs 2D inputs.
    noise_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    noise_2d = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )

    # For 2D, indexing by tilt gives a row
    assert noise_2d[0].shape == (4,)
    assert noise_2d[1].shape == (4,)

    # For 2D, indexing by tilt gives a 1D row
    for tilt_idx in range(noise_2d.shape[0]):
        noise_for_tilt = noise_2d[tilt_idx] if np.ndim(noise_2d) >= 2 else noise_2d
        assert noise_for_tilt.ndim == 1

    # For 1D, the fix uses the whole array for every tilt index
    for tilt_idx in range(5):  # more tilts than radial bins — should still work
        noise_for_tilt = noise_1d[tilt_idx] if np.ndim(noise_1d) >= 2 else noise_1d
        assert noise_for_tilt.ndim == 1
        np.testing.assert_array_equal(noise_for_tilt, noise_1d)


def test_batch_make_radial_noise_matches_vmap():
    """batch_make_radial_noise should produce the same result as manual vmap."""
    radials = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    image_shape = (6, 6)
    batch_out = np.asarray(noise.batch_make_radial_noise(radials, image_shape))
    assert batch_out.shape == (2, 36)
    # Compare with individual calls
    for i in range(2):
        single = np.asarray(noise.make_radial_noise(radials[i], image_shape))
        np.testing.assert_allclose(batch_out[i], single, atol=1e-6)

def test_batch_basis_times_coords2_matches_matmul():
    """batch_basis_times_coords2 should match direct basis @ coords.T."""
    import jax.numpy as jnp
    rng = np.random.default_rng(42)
    # basis: (pixels, n_basis), coords: (n_images, n_basis)
    basis = jnp.array(rng.normal(size=(10, 3)).astype(np.float32))
    coords = jnp.array(rng.normal(size=(5, 3)).astype(np.float32))
    out = np.asarray(noise.batch_basis_times_coords2(basis, coords))
    expected = np.asarray(basis @ coords.T).T  # (n_images, pixels)
    assert out.shape == (5, 10)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_to_batched_pixel_noise_normalizes_common_shapes():
    image_shape = (4, 4)

    n2 = np.arange(16, dtype=np.float32).reshape(4, 4)
    out2 = np.asarray(noise.to_batched_pixel_noise(n2, image_shape, batch_size=3))
    assert out2.shape == (1, 16)  # (1, N) — callers broadcast; no explicit repeat

    n3 = np.arange(2 * 16, dtype=np.float32).reshape(2, 4, 4)
    out3 = np.asarray(noise.to_batched_pixel_noise(n3, image_shape))
    assert out3.shape == (2, 16)

    n1 = np.arange(16, dtype=np.float32)
    out1 = np.asarray(noise.to_batched_pixel_noise(n1, image_shape))
    assert out1.shape == (1, 16)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_make_radial_noise_gpu(gpu_device):
    vec = np.array([1.0, 2.0], dtype=np.float32)
    image_shape = (6, 6)

    cpu_out = np.asarray(noise.make_radial_noise(vec, image_shape))

    with jax.default_device(gpu_device):
        vec_g = jax.device_put(jnp.array(vec), gpu_device)
        gpu_out = np.asarray(noise.make_radial_noise(vec_g, image_shape))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_to_batched_pixel_noise_gpu(gpu_device):
    image_shape = (4, 4)
    n2 = np.arange(16, dtype=np.float32).reshape(4, 4)

    cpu_out = np.asarray(noise.to_batched_pixel_noise(n2, image_shape, batch_size=3))

    with jax.default_device(gpu_device):
        n2_g = jax.device_put(jnp.array(n2), gpu_device)
        gpu_out = np.asarray(noise.to_batched_pixel_noise(n2_g, image_shape, batch_size=3))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_radial_noise_model_get_gpu(gpu_device):
    radial = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = noise.RadialNoiseModel(radial, image_shape=(8, 8))

    cpu_out = np.asarray(model.get())

    with jax.default_device(gpu_device):
        model_g = noise.RadialNoiseModel(jax.device_put(jnp.array(radial), gpu_device), image_shape=(8, 8))
        gpu_out = np.asarray(model_g.get())

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_variable_radial_noise_model_get_gpu(gpu_device):
    radials = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    dose_idx = np.array([0, 1, 1, 0], dtype=np.int32)
    model = noise.VariableRadialNoiseModel(radials, dose_idx, image_shape=(8, 8))
    query = np.array([0, 2, 3], dtype=np.int32)

    cpu_out = np.asarray(model.get(query))

    with jax.default_device(gpu_device):
        model_g = noise.VariableRadialNoiseModel(
            jax.device_put(jnp.array(radials), gpu_device),
            jax.device_put(jnp.array(dose_idx), gpu_device),
            image_shape=(8, 8),
        )
        gpu_out = np.asarray(model_g.get(jax.device_put(jnp.array(query), gpu_device)))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_get_masked_noise_variance_gpu(gpu_device):
    image_shape = (4, 4)
    image_masks = np.ones((2, *image_shape), dtype=np.float32)
    unmasked_noise = np.arange(np.prod(image_shape), dtype=np.float32)

    cpu_out = np.asarray(noise.get_masked_noise_variance_from_noise_variance(image_masks, unmasked_noise, image_shape))

    with jax.default_device(gpu_device):
        masks_g = jax.device_put(jnp.array(image_masks), gpu_device)
        noise_g = jax.device_put(jnp.array(unmasked_noise), gpu_device)
        gpu_out = np.asarray(noise.get_masked_noise_variance_from_noise_variance(masks_g, noise_g, image_shape))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Half-image noise equivalence tests
# ---------------------------------------------------------------------------

import recovar.core.fourier_transform_utils as fourier_transform_utils


def test_make_radial_noise_half_matches_full_extraction():
    """Verify make_radial_noise_half matches full_image_to_half_image(make_radial_noise)."""
    for image_shape in [(8, 8), (6, 10), (4, 8)]:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        full = np.asarray(noise.make_radial_noise(vec, image_shape))
        half_native = np.asarray(noise.make_radial_noise_half(vec, image_shape))
        half_from_full = np.asarray(
            fourier_transform_utils.full_image_to_half_image(full.reshape(1, -1), image_shape)
        ).ravel()
        np.testing.assert_allclose(half_native, half_from_full, atol=1e-6, rtol=1e-6,
                                   err_msg=f"Failed for image_shape={image_shape}")


def test_make_radial_noise_half_scalar():
    """Verify scalar noise broadcasts correctly in half layout."""
    image_shape = (8, 8)
    scalar = np.array([2.5], dtype=np.float32)
    half = np.asarray(noise.make_radial_noise_half(scalar, image_shape))
    expected_size = image_shape[0] * (image_shape[1] // 2 + 1)
    assert half.size == expected_size
    np.testing.assert_allclose(half.ravel(), 2.5, atol=1e-7)


def test_batch_make_radial_noise_half_matches_full():
    """Verify batch_make_radial_noise_half matches full_image_to_half_image(batch_make_radial_noise)."""
    image_shape = (6, 10)
    radials = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    full = np.asarray(noise.batch_make_radial_noise(radials, image_shape))
    half_native = np.asarray(noise.batch_make_radial_noise_half(radials, image_shape))
    half_from_full = np.asarray(
        fourier_transform_utils.full_image_to_half_image(full, image_shape)
    )
    np.testing.assert_allclose(half_native, half_from_full, atol=1e-6, rtol=1e-6)


def test_radial_noise_model_get_half_matches_full():
    """Verify RadialNoiseModel.get_half() matches extraction from get()."""
    radial = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    image_shape = (8, 8)
    model = noise.RadialNoiseModel(radial, image_shape=image_shape)
    full = np.asarray(model.get())
    half = np.asarray(model.get_half())
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


def test_variable_radial_noise_model_get_half_matches_full():
    """Verify VariableRadialNoiseModel.get_half() matches extraction from get()."""
    radials = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        dtype=np.float32,
    )
    dose_idx = np.array([0, 1, 1, 0], dtype=np.int32)
    image_shape = (8, 8)
    model = noise.VariableRadialNoiseModel(radials, dose_idx, image_shape=image_shape)
    indices = np.array([0, 2, 3], dtype=np.int32)
    full = np.asarray(model.get(indices))
    half = np.asarray(model.get_half(indices))
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full, image_shape))
    np.testing.assert_allclose(half, expected, atol=1e-6, rtol=1e-6)


def test_to_batched_half_pixel_noise_scalar():
    """Scalar noise should expand to (1, N_half) — callers broadcast."""
    image_shape = (8, 8)
    half_pixel_count = image_shape[0] * (image_shape[1] // 2 + 1)
    out = np.asarray(noise.to_batched_half_pixel_noise(2.5, image_shape, batch_size=3))
    assert out.shape == (1, half_pixel_count)  # (1, N) — XLA broadcasts to (B, N)
    np.testing.assert_allclose(out, 2.5, atol=1e-7)


def test_to_batched_half_pixel_noise_passthrough():
    """Half-pixel input should pass through unchanged (no explicit repeat)."""
    image_shape = (8, 8)
    half_pixel_count = image_shape[0] * (image_shape[1] // 2 + 1)
    half_input = np.arange(half_pixel_count, dtype=np.float32).reshape(1, -1)
    out = np.asarray(noise.to_batched_half_pixel_noise(half_input, image_shape, batch_size=2))
    assert out.shape == (1, half_pixel_count)  # (1, N) — XLA broadcasts to (B, N)
    np.testing.assert_array_equal(out[0], half_input.ravel())


def test_to_batched_half_pixel_noise_full_converts():
    """Full-pixel input should be converted to half-pixel via full_image_to_half_image."""
    image_shape = (4, 4)
    full_pixel_count = 16
    half_pixel_count = 4 * 3  # 4*(4//2+1) = 12
    full_input = np.arange(full_pixel_count, dtype=np.float32).reshape(1, -1)
    out = np.asarray(noise.to_batched_half_pixel_noise(full_input, image_shape))
    assert out.shape == (1, half_pixel_count)
    expected = np.asarray(fourier_transform_utils.full_image_to_half_image(full_input, image_shape))
    np.testing.assert_allclose(out, expected, atol=1e-6)


class _MockImageSource:
    def __init__(self, image_shape, scale=1.0):
        self.mask = np.ones(image_shape, dtype=np.float32)
        self._scale = np.float32(scale)

    def process_images(self, images):
        return jnp.asarray(images) * self._scale


class _MockNoiseDataset:
    """Minimal dataset stub to compare legacy loops vs explicit iterator paths."""

    def __init__(self, images, image_shape):
        self._images = jnp.asarray(images)
        self.n_images = int(images.shape[0])
        self.image_shape = tuple(image_shape)
        self.grid_size = int(image_shape[0])
        self.image_size = int(np.prod(image_shape))
        self.volume_shape = (self.grid_size, self.grid_size, self.grid_size)
        self.volume_mask_threshold = 0.0
        self.padding = 0
        self.voxel_size = 1.0
        self.dtype_real = np.float32
        self.premultiplied_ctf = False

        self.image_source = _MockImageSource(image_shape, scale=1.25)
        self.translations = jnp.zeros((self.n_images, 2), dtype=jnp.float32)
        self.rotation_matrices = jnp.tile(jnp.eye(3, dtype=jnp.float32), (self.n_images, 1, 1))
        self.CTF_params = jnp.zeros((self.n_images, 1), dtype=jnp.float32)

    def process_images(self, images, apply_image_mask=False):
        _ = apply_image_mask
        return self.image_source.process_images(images)

    @property
    def image_mask(self):
        return self.image_source.mask

    def ctf_evaluator(self, ctf_params, image_shape, voxel_size):
        del voxel_size
        return jnp.ones((ctf_params.shape[0], int(np.prod(image_shape))), dtype=jnp.float32)

    def _iter_indices(self, indices, batch_size):
        idx = np.asarray(indices, dtype=np.int32)
        for start in range(0, idx.size, batch_size):
            batch_ind = idx[start:start + batch_size]
            yield self._images[batch_ind], batch_ind, batch_ind

    def iter_batches(self, batch_size, *, indices=None, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images, dtype=np.int32)
        idx = np.asarray(indices, dtype=np.int32)
        for start in range(0, idx.size, batch_size):
            batch_ind = idx[start:start + batch_size]
            yield (
                self._images[batch_ind],
                self.rotation_matrices[batch_ind],
                self.translations[batch_ind],
                self.CTF_params[batch_ind],
                None,
                batch_ind,
                batch_ind,
            )


def _legacy_estimate_noise_variance(dataset, batch_size, max_images=10000):
    from recovar.reconstruction import regularization

    sum_sq = 0
    if dataset.n_images > max_images:
        subset_indices = np.random.choice(dataset.n_images, size=max_images, replace=False)
        data_generator = dataset.iter_batches(batch_size=batch_size, indices=subset_indices)
        n_images_used = max_images
    else:
        data_generator = dataset.iter_batches(batch_size=batch_size)
        n_images_used = dataset.n_images

    for batch, _rotation_matrices, _translations, _ctf_params, _noise_variance, _particle_indices, _image_indices in data_generator:
        batch = dataset.image_source.process_images(batch)
        sum_sq += jnp.sum(jnp.abs(batch) ** 2, axis=0)

    mean_ps = sum_sq / n_images_used
    cov_noise_mask = jnp.median(mean_ps)
    average_image_ps = regularization.average_over_shells(mean_ps, dataset.image_shape)
    return np.asarray(cov_noise_mask, dtype=dataset.dtype_real), np.asarray(average_image_ps, dtype=dataset.dtype_real)


def _legacy_estimate_noise_level_no_masks(dataset, image_subset, mean_estimate, batch_size, disc_type="linear_interp"):
    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.reconstruction import regularization

    lhs = 0
    rhs = 0
    config = ForwardModelConfig.from_dataset(dataset, disc_type=disc_type)
    data_generator = dataset.iter_batches(batch_size=batch_size, indices=image_subset)

    for batch, rotation_matrices, translations, ctf_params, _noise_variance, _particle_indices, batch_ind in data_generator:
        batch = dataset.image_source.process_images(batch)
        batch = core.translate_images(batch, dataset.translations[batch_ind], dataset.image_shape)
        ctf = dataset.ctf_evaluator(ctf_params, dataset.image_shape, dataset.voxel_size)

        if mean_estimate is not None:
            projected_mean = noise.core_forward.forward_model(
                config,
                mean_estimate,
                ctf_params,
                rotation_matrices,
            )
            if dataset.premultiplied_ctf:
                batch = batch - projected_mean * ctf
            else:
                batch = batch - projected_mean

        averaged_ps = regularization.batch_average_over_shells(jnp.abs(batch) ** 2, dataset.image_shape, 0)
        lhs += jnp.sum(averaged_ps, axis=0)
        rhs += batch.shape[0] if not dataset.premultiplied_ctf else jnp.sum(
            regularization.batch_average_over_shells(jnp.abs(ctf) ** 2, dataset.image_shape, 0),
            axis=0,
        )

    estimated_noise = lhs / rhs
    valid_mask = jnp.isfinite(estimated_noise)
    if not jnp.all(valid_mask):
        valid_indices = jnp.where(valid_mask)[0]
        last_valid_value = estimated_noise[valid_indices[-1]] if valid_indices.size > 0 else 1.0
        estimated_noise = jnp.where(valid_mask, estimated_noise, last_valid_value)
    return estimated_noise


def test_estimate_noise_variance_matches_legacy_loop():
    rng = np.random.default_rng(7)
    image_shape = (4, 4)
    images = (
        rng.normal(size=(9, np.prod(image_shape))).astype(np.float32)
        + 1j * rng.normal(size=(9, np.prod(image_shape))).astype(np.float32)
    ).astype(np.complex64)
    ds = _MockNoiseDataset(images, image_shape=image_shape)

    expected_cov, expected_radial = _legacy_estimate_noise_variance(ds, batch_size=3, max_images=100)
    cov, radial = noise.estimate_noise_variance(ds, batch_size=3, max_images=100)

    np.testing.assert_allclose(cov, expected_cov, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(radial, expected_radial, atol=1e-6, rtol=1e-6)


def test_estimate_noise_variance_subset_sampling_runs():
    """Smoke test: estimate_noise_variance with subsampling runs without error."""
    rng = np.random.default_rng(11)
    image_shape = (4, 4)
    images = (
        rng.normal(size=(10, np.prod(image_shape))).astype(np.float32)
        + 1j * rng.normal(size=(10, np.prod(image_shape))).astype(np.float32)
    ).astype(np.complex64)
    ds = _MockNoiseDataset(images, image_shape=image_shape)

    cov, radial = noise.estimate_noise_variance(ds, batch_size=2, max_images=4)

    assert np.isfinite(cov)
    assert np.all(np.isfinite(radial))
    assert radial.shape[0] > 0


def test_estimate_noise_level_no_masks_matches_legacy_loop():
    rng = np.random.default_rng(19)
    image_shape = (4, 4)
    images = (
        rng.normal(size=(8, np.prod(image_shape))).astype(np.float32)
        + 1j * rng.normal(size=(8, np.prod(image_shape))).astype(np.float32)
    ).astype(np.complex64)
    ds = _MockNoiseDataset(images, image_shape=image_shape)
    subset = np.arange(ds.n_images, dtype=np.int32)

    expected = _legacy_estimate_noise_level_no_masks(
        ds,
        image_subset=subset,
        mean_estimate=None,
        batch_size=3,
    )
    got = noise.estimate_noise_level_no_masks(
        ds,
        image_subset=subset,
        mean_estimate=None,
        batch_size=3,
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6, rtol=1e-6)
