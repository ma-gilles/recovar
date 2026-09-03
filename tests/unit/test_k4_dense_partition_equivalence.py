"""Deterministic partition-equivalence coverage for dense K-class EM."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)
N_IMAGES = 4


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    del voxel_size
    image_shape = IMAGE_SHAPE if image_shape is None else image_shape
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1 if half_image else image_shape[1])
    return jnp.ones((params.shape[0], n_pixels), dtype=jnp.float32)


def _process_images(batch, apply_image_mask=False):
    del apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape(images.shape[0], -1).astype(jnp.complex64)


def _process_images_half(batch, apply_image_mask=False):
    del apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape(images.shape[0], -1).astype(jnp.complex64)


class _DeterministicDataset:
    """Small dataset whose complex128 accumulator dtype exposes reduction error."""

    def __init__(self):
        self.image_shape = IMAGE_SHAPE
        self.image_size = int(np.prod(IMAGE_SHAPE))
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = int(np.prod(VOLUME_SHAPE))
        self.n_images = N_IMAGES
        self.n_units = N_IMAGES
        self.voxel_size = 1.0
        self.dtype = jnp.complex128
        self.CTF_params = np.zeros((N_IMAGES, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_process_images)
        self.process_images_half = staticmethod(_process_images_half)
        self._images = np.stack(
            [
                np.random.default_rng(seed).standard_normal(IMAGE_SHAPE).astype(np.float32)
                for seed in (1265, 9229, 4172, 6255)
            ],
        )

        class _ImageSource:
            process_images = staticmethod(_process_images)
            process_images_half = staticmethod(_process_images_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield (
                jnp.asarray(self._images[batch_indices]),
                None,
                None,
                jnp.asarray(self.CTF_params[batch_indices]),
                None,
                batch_indices,
                batch_indices,
            )

    def get_valid_frequency_indices(self, pixel_res):
        del pixel_res
        return np.ones(self.volume_size, dtype=bool)


def _hermitian_volume(seed):
    real_volume = np.random.default_rng(seed).standard_normal(VOLUME_SHAPE).astype(np.float32)
    fourier_volume = np.fft.fftshift(np.fft.fftn(real_volume)).ravel()
    return jnp.asarray(fourier_volume, dtype=jnp.complex64).astype(jnp.complex128)


def _rotations(n_rotations, seed):
    values = np.random.default_rng(seed).standard_normal((n_rotations, 3, 3))
    matrices, triangular = np.linalg.qr(values)
    matrices *= np.sign(np.diagonal(triangular, axis1=1, axis2=2))[:, None, :]
    matrices[np.linalg.det(matrices) < 0] *= -1
    return matrices.astype(np.float32)


def test_dense_k4_image_and_rotation_partition_equivalence() -> None:
    """Dense K=4 class-by-pose normalization is invariant to both partitions."""

    dataset = _DeterministicDataset()
    rotations = _rotations(5, seed=1729)
    translations = np.asarray(
        [[0.0, 0.0], [0.75, -0.5], [-0.5, 1.0]],
        dtype=np.float32,
    )
    means = jnp.stack(
        [
            _hermitian_volume(seed) * scale
            for scale, seed in zip((0.70, 0.85, 1.00, 1.15), (1001, 1002, 1003, 1004))
        ],
    )
    mean_variance = jnp.full(dataset.volume_size, 10.0, dtype=jnp.float64)
    noise_variance = jnp.full(dataset.image_size, 3.0, dtype=jnp.float64)
    common = {
        "class_log_priors": np.log(np.asarray([0.37, 0.29, 0.21, 0.13], dtype=np.float64)),
        "class_rotation_log_prior": np.asarray(
            [
                [0.0, -0.2, -0.4, -0.6, -0.8],
                [-0.7, 0.0, -0.2, -0.4, -0.6],
                [-0.5, -0.7, 0.0, -0.2, -0.4],
                [-0.3, -0.5, -0.7, 0.0, -0.2],
            ],
            dtype=np.float64,
        ),
        "translation_log_prior": np.asarray(
            [
                [0.0, -3.0, -4.0],
                [-4.0, 0.0, -3.0],
                [-3.0, -4.0, 0.0],
                [-0.5, 0.0, -1.0],
            ],
            dtype=np.float64,
        ),
        "current_size": 6,
        "sparse_pass2": False,
        "score_with_masked_images": True,
        "half_spectrum_scoring": True,
        "return_best_pose_details": True,
        "use_float64_scoring": True,
        "use_float64_projections": True,
    }

    unpartitioned = run_dense_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=N_IMAGES,
        rotation_block_size=rotations.shape[0],
        **common,
    )
    partitioned = run_dense_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=3,
        rotation_block_size=2,
        **common,
    )

    exact_fields = (
        "per_class_hard_assignments",
        "class_assignments",
        "pose_assignments",
        "class_responsibilities",
        "class_posterior_sums",
        "class_mstep_posterior_sums",
        "per_class_best_pose_rotations",
        "per_class_best_pose_translations",
        "per_class_best_pose_rotation_ids",
        "best_pose_rotations",
        "best_pose_translations",
        "best_pose_rotation_ids",
    )
    for field in exact_fields:
        np.testing.assert_array_equal(getattr(unpartitioned, field), getattr(partitioned, field))

    # Keep the fixture non-degenerate: three classes win and every translation
    # is selected, while the fifth rotation forces a padded tail block.
    assert np.unique(np.asarray(unpartitioned.class_assignments)).size == 3
    np.testing.assert_array_equal(
        np.unique(np.asarray(unpartitioned.pose_assignments) % translations.shape[0]),
        np.arange(translations.shape[0]),
    )
    assert rotations.shape[0] % 2 == 1

    reduction_tolerance = 1024 * np.finfo(np.float64).eps
    for field in ("Ft_y", "Ft_ctf", "new_means"):
        reference = np.asarray(getattr(unpartitioned, field))
        actual = np.asarray(getattr(partitioned, field))
        assert reference.dtype == np.complex128
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=reduction_tolerance,
            atol=reduction_tolerance,
        )

    for reference_stats, actual_stats in zip(
        unpartitioned.per_class_stats,
        partitioned.per_class_stats,
    ):
        np.testing.assert_allclose(
            actual_stats.log_evidence_per_image,
            reference_stats.log_evidence_per_image,
            rtol=reduction_tolerance,
            atol=reduction_tolerance,
        )
        np.testing.assert_allclose(
            actual_stats.best_log_score_per_image,
            reference_stats.best_log_score_per_image,
            rtol=reduction_tolerance,
            atol=reduction_tolerance,
        )
        np.testing.assert_array_equal(
            actual_stats.max_posterior_per_image,
            reference_stats.max_posterior_per_image,
        )
        np.testing.assert_array_equal(
            actual_stats.rotation_posterior_sums,
            reference_stats.rotation_posterior_sums,
        )
