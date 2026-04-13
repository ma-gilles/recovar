from types import SimpleNamespace

import numpy as np
import pytest
import scipy.stats

pytest.importorskip("jax")

from recovar.heterogeneity import image_assignment as ia

pytestmark = pytest.mark.unit


def test_compute_residual_uses_forward_model_translation_and_noise_scaling(monkeypatch):
    images = np.array([[3.0, 5.0], [7.0, 11.0]], dtype=np.float32)
    projected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(ia.core, "translate_images", lambda x, *_args, **_kwargs: x)
    monkeypatch.setattr(ia.core_forward, "forward_model", lambda *_args, **_kwargs: projected)

    config = SimpleNamespace(image_shape=(1, 2))
    out = ia.compute_residual(
        images=images,
        ctf_params=None,
        rotation_matrices=None,
        translations=None,
        mean_estimate=ia.core.Volume(np.array([0.0], dtype=np.float32), disc_type="linear_interp"),
        config=config,
        noise_variance=np.array([4.0, 4.0], dtype=np.float32),
    )
    expected = np.linalg.norm((images - projected) / np.sqrt(4.0), axis=-1) ** 2
    assert np.allclose(np.asarray(out), expected)


class _MockDS:
    """Minimal mock dataset for iter_batches()."""

    dtype = np.complex64
    dtype_real = np.float32
    n_units = 4
    volume_shape = (1, 1, 1)
    image_shape = (1, 2)
    grid_size = 1
    voxel_size = 1.0
    padding = 0
    premultiplied_ctf = False
    volume_mask_threshold = 0.25
    CTF_params = np.zeros((4, 1), dtype=np.float32)
    rotation_matrices = np.zeros((4, 3, 3), dtype=np.float32)
    translations = np.zeros((4, 2), dtype=np.float32)
    ctf_evaluator = staticmethod(lambda *_args, **_kwargs: None)

    def iter_batches(self, batch_size, **kwargs):
        batch = np.zeros((2, 2), dtype=np.float32)
        particles_ind = np.array([1, 3], dtype=np.int32)
        batch_image_ind = np.array([1, 3], dtype=np.int32)
        yield (
            batch,
            self.rotation_matrices[batch_image_ind],
            self.translations[batch_image_ind],
            self.CTF_params[batch_image_ind],
            None,
            particles_ind,
            batch_image_ind,
        )


def test_compute_image_assignment_fills_residual_matrix(monkeypatch):
    def fake_compute_residual(images, _ctf_params, _rotation_matrices, _translations, mean_estimate, *_args, **_kwargs):
        values = mean_estimate.array if hasattr(mean_estimate, "array") else mean_estimate
        return np.full(images.shape[0], float(np.real(values[0])), dtype=np.float32)

    monkeypatch.setattr(ia, "compute_residual", fake_compute_residual)

    volumes = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    out = ia.compute_image_assignment(
        _MockDS(), volumes, noise_variance=np.array([1.0]), batch_size=2, disc_type="nearest"
    )

    assert out.shape == (3, 4)
    assert np.allclose(out[:, 1], [10.0, 20.0, 30.0])
    assert np.allclose(out[:, 3], [10.0, 20.0, 30.0])
    assert np.allclose(out[:, 0], 0.0)
    assert np.allclose(out[:, 2], 0.0)


def test_estimate_false_positive_rate_from_mocked_residual(monkeypatch):
    class _DS2(_MockDS):
        n_units = 2
        CTF_params = np.zeros((2, 1), dtype=np.float32)
        rotation_matrices = np.zeros((2, 3, 3), dtype=np.float32)
        translations = np.zeros((2, 2), dtype=np.float32)

        def iter_batches(self, batch_size, **kwargs):
            batch = np.zeros((2, 2), dtype=np.float32)
            particles_ind = np.array([0, 1], dtype=np.int32)
            batch_image_ind = np.array([0, 1], dtype=np.int32)
            yield (
                batch,
                self.rotation_matrices[batch_image_ind],
                self.translations[batch_image_ind],
                self.CTF_params[batch_image_ind],
                None,
                particles_ind,
                batch_image_ind,
            )

    monkeypatch.setattr(ia, "compute_residual", lambda *_args, **_kwargs: np.array([4.0, 9.0], dtype=np.float32))
    vols = np.array([[1.0], [2.0]], dtype=np.float32)
    gamma = ia.estimate_false_positive_rate(
        _DS2(), vols, noise_variance=np.array([1.0]), batch_size=2, disc_type="nearest"
    )

    expected_alphas = 0.5 * np.sqrt(np.array([4.0, 9.0]))
    expected_gamma = 1 - np.mean(scipy.stats.norm.cdf(expected_alphas))
    assert np.isclose(gamma, expected_gamma)
